# -*- coding: utf-8 -*-
"""桥梁警戒（Jetson 实时管线用）。

从 bridge_warning_seg.py 抽出的纯计算部分：预存桥梁库、目标桥选择、
桥面线检测（分割掩码 → 行带覆盖率 → 时序平滑）、警戒线像素行计算、
信息面板锚点。本模块不做任何 OpenCV 绘制——每帧只产出像素坐标与数据
dict，经 8765 导航数据端口发给 Godot 渲染。
"""

from __future__ import annotations

import configparser
import json
import math
import re
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np


# ===========================================================================
# 预存判断参数
# ===========================================================================
VESSEL_HEIGHT_M = 4.5      # 预存：船只高度 m

# 预存桥梁库：仅作 Redis 不可用时的回退。权威值来自平台 bridge_cache_data
# （BridgeAlertClient.bridge_library），下列净空已按 2026-10-08 平台表更正
# 江湾/海印/海心三座；广州大桥、猎德大桥尚无平台值，仍为 2026-09 核定数。
BRIDGE_LIBRARY = (
    {"name": "人民桥",   "lat": 23.10831081071863,
     "lon": 113.24509650751531, "clearance_m": 8.7},
    {"name": "解放大桥", "lat": 23.114188711469257,
     "lon": 113.25867522055131, "clearance_m": 8.7},
    {"name": "海珠桥",   "lat": 23.115293982708252,
     "lon": 113.26170900938332, "clearance_m": 8.6},
    {"name": "江湾大桥", "lat": 23.115745095725487,
     "lon": 113.27405533136901, "clearance_m": 10.8},
    {"name": "海印大桥", "lat": 23.112156053953058,
     "lon": 113.28181373234662, "clearance_m": 8.7},
    {"name": "广州大桥", "lat": 23.11001014017725,
     "lon": 113.31039131833792, "clearance_m": 7.0},
    {"name": "海心桥",   "lat": 23.11083639675565,
     "lon": 113.3135978491622, "clearance_m": 10.0},
    {"name": "猎德大桥", "lat": 23.112338165122107,
     "lon": 113.3285256890865, "clearance_m": 10.0},
)
BRIDGE_PANEL_RANGE_M = 300.0   # 船头±90° 内且直线距离≤该值 → 显示信息面板
BRIDGE_WARN_RANGE_M = 200.0    # 直线距离≤该值 → 显示净空警戒线

# WGS84 赤道半径，球面距离换算基准
EARTH_RADIUS_M = 6378137.0


def latlon_to_enu(lat: float, lon: float,
                  ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """经纬度 → 以 (ref_lat, ref_lon) 为原点的 ENU 水平偏移 (east, north) 米。"""
    east = (math.radians(lon - ref_lon)
            * EARTH_RADIUS_M * math.cos(math.radians(ref_lat)))
    north = math.radians(lat - ref_lat) * EARTH_RADIUS_M
    return east, north


def enu_to_body(east: float, north: float, heading_deg: float,
                yaw_offset_deg: float = 0.0) -> tuple[float, float]:
    """ENU (东,北) → 机体坐标系 (forward, left)，航向含安装偏航修正。"""
    h = math.radians(heading_deg + yaw_offset_deg)
    forward = east * math.sin(h) + north * math.cos(h)
    left = north * math.sin(h) - east * math.cos(h)
    return forward, left


def pick_target_bridge(lat: float, lon: float, heading_deg: float, *,
                       panel_range_m: float, library=BRIDGE_LIBRARY):
    """从桥梁库选目标桥：船头 ±90° 扇区内、直线距离 ≤ panel_range_m
    的最近一座；无候选返回 None。library 可传平台桥梁表。"""
    best = None
    for bridge in library:
        east, north = latlon_to_enu(bridge["lat"], bridge["lon"], lat, lon)
        dist = math.hypot(east, north)
        if dist > panel_range_m:
            continue
        bearing = math.degrees(math.atan2(east, north))
        off_bow = (bearing - heading_deg + 180.0) % 360.0 - 180.0
        if abs(off_bow) > 90.0:
            continue
        if best is None or dist < best["dist"]:
            best = {**bridge, "dist": dist}
    return best


# ===========================================================================
# 平台 Redis 桥梁预警（优先数据源；读不到退回预存桥梁库）
# ===========================================================================
BRIDGE_ALERT_PREFIX = "bridge_alert"
# 平台桥梁表缓存（id/name/经纬度/clearanceHeight），比硬编码库权威
BRIDGE_CACHE_KEY = "bridge_cache_data"
# fastjson 把 long 写成 1L/2L，需去掉后缀才是合法 JSON
_FASTJSON_LONG = re.compile(r"(?<=\d)L(?=\s*[,\}\]])")


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_field(data: dict, *names):
    for name in names:
        if name in data and data[name] not in (None, ""):
            return data[name]
    return None


class BridgeAlertClient:
    """读平台写入的 bridge_alert:{船id}:{桥id}:{预警级别}。

    每 refresh_interval 秒 SCAN 一次本船全部记录，返回距桥最近的一座：
      {bridge_id, alert_level, distance, ship_height, clearance_height}
    同一座桥多级并存时取级别最高的一条（3危险 > 2警告 > 1提示）。
    Redis 不可用 / 无记录 / 字段缺失时 nearest() 返回 None，调用方回退
    预存桥梁库，绝不影响主链路。
    量纲：`clearanceHeight` 与预存库的 `clearance_m` 是同一个量（桥高），
    面板"净空高度"仍 = 桥高 − 船高，不要当成已扣过船高的可用净空。
    """

    def __init__(self, ship_id, *, host: str = "192.168.3.120", port: int = 6379,
                 password: str | None = None, db: int = 1,
                 refresh_interval: float = 1.0, timeout: float = 1.0):
        # 前导 * ：兼容平台把键序列化成 "bridge_alert:..."（带引号）或加
        # cache: 之类前缀的写法
        self.pattern = f"*{BRIDGE_ALERT_PREFIX}:{ship_id}:*"
        self.host, self.port, self.db = host, int(port), int(db)
        self.refresh_interval = float(refresh_interval)
        self._next_refresh = 0.0
        self._best: dict | None = None
        self._library: list | None = None
        self._lib_next = 0.0
        self._warned = False
        self._client = None
        try:
            import redis
            client = redis.Redis(
                host=host, port=int(port), db=int(db), password=password,
                socket_timeout=float(timeout),
                socket_connect_timeout=float(timeout),
                decode_responses=True,
            )
            client.ping()
            self._client = client
            print(f"[BridgeAlert] ✓ Redis {host}:{port}/db{db}，"
                  f"扫描模式 {self.pattern}")
        except Exception as e:
            print(f"[BridgeAlert] ✗ Redis 不可用（{type(e).__name__}: {e}）"
                  "→ 全程使用预存桥梁库")
            self._client = None

    def nearest(self) -> dict | None:
        if self._client is None:
            return None
        now = time.monotonic()
        if now >= self._next_refresh:
            self._next_refresh = now + self.refresh_interval
            try:
                self._best = self._scan_nearest()
                self._warned = False
            except Exception as e:
                if not self._warned:
                    print(f"[BridgeAlert] 读取失败（{e}）→ 沿用上一次/预存值")
                    self._warned = True
        return self._best

    def bridge_library(self, refresh_interval: float = 300.0):
        """读平台桥梁表 bridge_cache_data → 与预存桥梁库同构的 list（多一个 id）。

        净空/坐标以平台为准（硬编码库按 2026-09 抄的，已有几座不符）；
        读不到返回 None，调用方继续用硬编码库。默认 5 分钟刷一次。
        """
        now = time.monotonic()
        if self._client is None or now < self._lib_next:
            return self._library
        self._lib_next = now + float(refresh_interval)
        try:
            data = self._json(self._client.get(BRIDGE_CACHE_KEY))
        except Exception as e:
            print(f"[BridgeAlert] 桥梁表读取失败（{e}）→ 沿用上一份")
            return self._library
        if not isinstance(data, list):
            return self._library
        lib = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            lat = _as_float(item.get("latitude"))
            lon = _as_float(item.get("longitude"))
            clearance = _as_float(item.get("clearanceHeight"))
            if not name or lat is None or lon is None or clearance is None:
                continue
            lib.append({"name": str(name), "lat": lat, "lon": lon,
                        "clearance_m": clearance, "id": item.get("id")})
        if not lib:
            return self._library
        if self._library is None:
            print(f"[BridgeAlert] ✓ 平台桥梁表 {len(lib)} 座"
                  f"（{BRIDGE_CACHE_KEY}）："
                  + "、".join(f"{b['name']} {b['clearance_m']:.1f}m"
                              for b in lib))
        self._library = lib
        return lib

    def probe(self, show_keys: int = 20) -> None:
        """现场排查：连接 → 命中键 → 解析结果，逐步打印，不抛异常。"""
        if self._client is None:
            print(f"[BridgeAlert] probe ✗ 未连上 {self.host}:{self.port}/db{self.db}")
            return
        try:
            print(f"[BridgeAlert] probe ping={self._client.ping()} "
                  f"{self.host}:{self.port}/db{self.db} "
                  f"dbsize={self._client.dbsize()} 模式={self.pattern}")
        except Exception as e:
            print(f"[BridgeAlert] probe ✗ ping/dbsize 失败：{type(e).__name__}: {e}")
            return
        lib = self.bridge_library()
        if lib:
            print(f"[BridgeAlert] probe {BRIDGE_CACHE_KEY} → {len(lib)} 座："
                  + "、".join(f"{b['id']}:{b['name']} {b['clearance_m']:.1f}m"
                              for b in lib[:12]))
        else:
            print(f"[BridgeAlert] probe {BRIDGE_CACHE_KEY} 读不到"
                  "→ 使用硬编码预存桥梁库")
        keys = []
        try:
            for key in self._client.scan_iter(match=self.pattern, count=1000):
                keys.append(str(key))
                if len(keys) >= show_keys:
                    break
        except Exception as e:
            print(f"[BridgeAlert] probe ✗ SCAN 失败：{type(e).__name__}: {e}")
            return
        print(f"[BridgeAlert] probe 命中 {len(keys)} 个键"
              f"（最多显示 {show_keys} 个）")
        for key in keys:
            try:
                print(f"  {key}  type={self._client.type(key)} "
                      f"ttl={self._client.ttl(key)}")
            except Exception as e:
                print(f"  {key}  读取失败 {e}")
        if keys:
            try:
                k0 = keys[0]
                kind = self._client.type(k0)
                raw = (self._client.hgetall(k0) if kind == "hash"
                       else self._client.lrange(k0, 0, 1) if kind == "list"
                       else self._client.get(k0))
                print(f"[BridgeAlert] probe 首个键原始值 type={kind}: "
                      f"{str(raw)[:300]}")
            except Exception as e:
                print(f"[BridgeAlert] probe 首个键原始值读取失败：{e}")
            print(f"[BridgeAlert] probe nearest = {self._scan_nearest()}")
            return
        # 没命中：有限遍历，优先展示疑似桥梁预警键，避免整库扫到像卡死
        related, scanned = [], 0
        try:
            for key in self._client.scan_iter(count=1000):
                scanned += 1
                low = str(key).lower()
                if "bridge" in low or "alert" in low:
                    related.append(str(key))
                if scanned >= 5000:
                    break
        except Exception as e:
            print(f"[BridgeAlert] probe ✗ 列键失败：{e}")
            return
        print(f"[BridgeAlert] probe 模式 {self.pattern} 没命中；已扫 {scanned} "
              f"个键，其中含 bridge/alert 字样的 {len(related)} 个"
              "（据此改 --ship-id 或 BRIDGE_ALERT_PREFIX）：")
        for key in related[:show_keys]:
            print(f"  {key}")
        if not related:
            print("  （没找到疑似键，可能预警数据不在这个 db，或平台还没写）")

    @staticmethod
    def _json(text):
        if text is None:
            return None
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            pass
        # 平台是 fastjson：long 序列化成 1L/2L，不是合法 JSON，去掉后缀重试
        try:
            return json.loads(_FASTJSON_LONG.sub("", text))
        except (TypeError, ValueError):
            return None

    def _read(self, key) -> list[dict]:
        """一个键 → 若干条记录。string/hash 各一条；list 每个元素一条
        （平台习惯 RPUSH JSON，见 db1 里键名为船id 的 list）。"""
        kind = self._client.type(key)
        if kind == "hash":
            raws = [self._client.hgetall(key)]
        elif kind == "string":
            raws = [self._json(self._client.get(key))]
        elif kind == "list":
            raws = [self._json(v)
                    for v in self._client.lrange(key, 0, -1)]
        else:
            raws = []

        records = []
        for raw in raws:
            # list 元素可能是单个对象，也可能整个是 JSON 数组
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                if not isinstance(item, dict):
                    continue
                records.append({
                    "bridge_id": _pick_field(item, "bridgeId", "bridge_id",
                                             "桥梁ID", "桥梁id"),
                    "alert_level": _pick_field(item, "alertLevel",
                                               "alert_level", "预警级别"),
                    "distance": _as_float(_pick_field(
                        item, "distance", "bridgeDistance",
                        "distanceToBridge", "距离桥梁")),
                    "ship_height": _as_float(_pick_field(
                        item, "shipHeight", "ship_height", "船舶高度")),
                    "clearance_height": _as_float(_pick_field(
                        item, "clearanceHeight", "clearance_height",
                        "currentClearance", "当前净空高度")),
                })
        return records

    def _scan_nearest(self) -> dict | None:
        per_bridge: dict[str, dict] = {}
        for key in self._client.scan_iter(match=self.pattern, count=1000):
            parts = str(key).split(":")
            tail_bridge = parts[-2].strip("\"' ") if len(parts) >= 2 else ""
            tail_level = _as_float(parts[-1].strip("\"' ")) if parts else None
            for rec in self._read(key):
                # key 尾段（桥id / 级别）作为 value 缺字段时的兜底
                if rec["bridge_id"] in (None, ""):
                    rec["bridge_id"] = tail_bridge or "?"
                if rec["distance"] is None:
                    continue
                level = _as_float(rec["alert_level"])
                if level is None:
                    level = tail_level
                rec["alert_level"] = int(level) if level is not None else 0
                bridge_id = str(rec["bridge_id"])
                cur = per_bridge.get(bridge_id)
                if cur is None or rec["alert_level"] >= cur["alert_level"]:
                    per_bridge[bridge_id] = rec
        if not per_bridge:
            return None
        return min(per_bridge.values(), key=lambda r: r["distance"])


# ===========================================================================
# RTK Excel 回放（离线验证用：IMG.MOV + Excel 代替实时 GNSS）
# ===========================================================================
def read_rtk_excel(path: str) -> dict:
    """读取 RTK Excel 首个含表头的 sheet，返回 dict(lat, lon, elev, time, heading)。

    zipfile + XML 解析，不依赖 openpyxl。表头支持中文（时间/纬度/经度/
    高程或海拔/航向或首向）与英文（Latitude / Longitude / Elevation /
    Time / Heading / Course，大小写不敏感，子串匹配）。无高程列补 0；
    无航向列时 heading 为 None（由 compute_heading_series 差分回退）。
    """
    spreadsheet_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    column_keys = (
        (("纬度", "latitude"), "lat"),
        (("经度", "longitude"), "lon"),
        (("高程", "海拔", "elevation"), "elev"),
        (("时间", "时刻", "time"), "time"),
        (("航向", "首向", "艏向", "heading", "course"), "heading"),
    )
    with zipfile.ZipFile(str(path)) as zf:
        names = sorted(
            n for n in zf.namelist()
            if n.startswith("xl/worksheets/") and n.endswith(".xml")
        )
        if not names:
            raise RuntimeError(f"Excel 中没有工作表: {path}")

        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            tag = f"{{{spreadsheet_ns}}}t"
            for si in root.iter(f"{{{spreadsheet_ns}}}si"):
                shared.append("".join(t.text or "" for t in si.iter(tag)))

        target: list[list[str]] = []
        index_map: dict[str, int] = {}
        for name in names:
            root = ET.fromstring(zf.read(name))
            rows: list[list[str]] = []
            for row in root.iter(f"{{{spreadsheet_ns}}}row"):
                values: list[str] = []
                for cell in row.findall(f"{{{spreadsheet_ns}}}c"):
                    val = cell.find(f"{{{spreadsheet_ns}}}v")
                    if val is None:
                        values.append("")
                    elif cell.get("t") == "s":
                        values.append(shared[int(val.text)])
                    else:
                        values.append(val.text)
                rows.append(values)
            if not rows:
                continue
            header = [h.strip().lower() for h in rows[0]]
            index_map = {}
            for aliases, key in column_keys:
                for idx, header_name in enumerate(header):
                    if any(alias in header_name for alias in aliases):
                        index_map[key] = idx
                        break
            if {"lat", "lon", "time"} <= set(index_map):
                target = rows[1:]
                break

        if not target:
            raise RuntimeError(
                "RTK Excel 表头缺少必需列：时间/纬度/经度"
                "（或 Latitude/Longitude/Time），实际表头: "
                f"{header}")

    time_col = np.asarray(
        [float(row[index_map["time"]]) for row in target], dtype=np.float64)
    lat_col = np.asarray(
        [float(row[index_map["lat"]]) for row in target], dtype=np.float64)
    lon_col = np.asarray(
        [float(row[index_map["lon"]]) for row in target], dtype=np.float64)
    if "elev" in index_map:
        elev_col = np.asarray(
            [float(row[index_map["elev"]]) for row in target], dtype=np.float64)
    else:
        elev_col = np.zeros_like(time_col)
    heading_col = None
    if "heading" in index_map:
        heading_col = np.asarray(
            [float(row[index_map["heading"]]) for row in target],
            dtype=np.float64) % 360.0
    return {
        "lat": lat_col,
        "lon": lon_col,
        "elev": elev_col,
        "time": time_col,
        "heading": heading_col,
    }


def compute_heading_series(rtk: dict, smooth_window: int = 11) -> np.ndarray:
    """由 RTK 相邻点航向（course over ground）得到每个点的航迹向序列（度）。

    中心差分求航向后做滑动平均平滑。RTK 为直线航迹时结果恒定。
    """
    lat, lon = rtk["lat"], rtk["lon"]
    east = (np.radians(lon - lon[0]) * EARTH_RADIUS_M
            * np.cos(np.radians(lat)))
    north = np.radians(lat - lat[0]) * EARTH_RADIUS_M
    heading = np.zeros_like(lat)
    if len(lat) < 3:
        return heading
    de = east[2:] - east[:-2]
    dn = north[2:] - north[:-2]
    heading[1:-1] = np.degrees(np.arctan2(de, dn)) % 360.0
    heading[0] = heading[1]
    heading[-1] = heading[-2]
    k = max(1, int(smooth_window) | 1)
    if len(heading) >= k:
        pad = k // 2
        kernel = np.ones(k) / k
        padded = np.pad(heading, (pad, pad), mode="edge")
        heading = np.convolve(padded, kernel, mode="valid")
    return heading


class RtkReplay:
    """离线回放器：按视频时间从 RTK Excel 插值出船位与艏向。

    与实时模式的分工：给 bridge_warning_engine.update() 提供
    (lat, lon, heading_deg)，替代 RealtimeNavigationState.latest_vessel_fix()。
    视频时间 → RTK 时间映射与 bridge_warning_seg.py 相同：
    t_rtk = (t_video − t0) × scale；超出 Excel 时长后 np.interp 钳到末点
    （桥位）。elev_mean 供相机高度兜底（无高程列时为 0）。
    """

    def __init__(self, path: str):
        rtk = read_rtk_excel(path)
        self.time = rtk["time"]
        self.lat = rtk["lat"]
        self.lon = rtk["lon"]
        self.t_end = float(rtk["time"][-1])
        self.elev_mean = float(np.mean(rtk["elev"]))
        if rtk["heading"] is not None:
            self.heading = np.degrees(np.unwrap(np.radians(rtk["heading"])))
            self.heading_src = "Excel 航向列"
        else:
            self.heading = compute_heading_series(rtk)
            self.heading_src = "相邻 RTK 点差分"

    def fix_at(self, t_video: float, t0: float, scale: float = 1.0):
        """视频时间 s → (lat, lon, heading_deg)，越界钳到端点。"""
        t_rtk = (float(t_video) - float(t0)) * float(scale)
        lat = float(np.interp(t_rtk, self.time, self.lat))
        lon = float(np.interp(t_rtk, self.time, self.lon))
        hdg = float(np.interp(t_rtk, self.time, self.heading)) % 360.0
        return lat, lon, hdg


# ===========================================================================
# 针孔投影器：标定内外参 + 畸变
# ===========================================================================
class PinholeProjector:
    """读 calibration.ini 内参/外参/畸变，把机体坐标 (forward, left, up)
    三维点投影到图像像素。机体坐标系即标定文件的 LiDAR forward 系
    （X前 Y左 Z上，原点在相机/RTK 天线处）。"""

    def __init__(self, ini_path: str, width: int, height: int):
        cfg = configparser.ConfigParser()
        if not cfg.read(str(ini_path), encoding="utf-8"):
            raise FileNotFoundError(f"读不到标定文件: {ini_path}")
        cal_w = cfg.getfloat("calibration", "width")
        cal_h = cfg.getfloat("calibration", "height")
        if cal_w <= 0 or cal_h <= 0:
            raise ValueError(f"标定文件分辨率非法: {cal_w}x{cal_h}")
        sx = width / cal_w
        sy = height / cal_h
        self.fx = cfg.getfloat("calibration", "fx") * sx
        self.fy = cfg.getfloat("calibration", "fy") * sy
        self.cx = cfg.getfloat("calibration", "cx") * sx
        self.cy = cfg.getfloat("calibration", "cy") * sy
        K = np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.array(
            [cfg.getfloat("distortion", key)
             for key in ("k1", "k2", "p1", "p2", "k3")],
            dtype=np.float64)
        extrinsic = np.asarray(
            [[float(v) for v in cfg.get("calibration",
                                        f"extrinsic_row{i}").split(",")]
             for i in range(4)], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(extrinsic[:3, :3])
        self._K, self._dist, self._rvec = K, dist, rvec
        self._t = extrinsic[:3, 3]

    def project_body(self, pts) -> np.ndarray:
        """pts: Nx3 机体坐标 (forward, left, up) 单位米 → Nx2 像素 (u, v)。"""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        pixels, _ = cv2.projectPoints(pts, self._rvec, self._t,
                                      self._K, self._dist)
        return pixels.reshape(-1, 2)


# ===========================================================================
# 桥面线提取：横向连续性高的行带 = 桥面，其底行 = 桥底顶线
# ===========================================================================
def bridge_deck_bottom_from_masks(
    bridge: np.ndarray,
    *,
    cover_thresh: float = 0.55,
    x_band: tuple = (0.10, 0.90),
    min_band_rows: int = 2,
    max_row_ratio: float = 0.97,
):
    """桥面掩码下边缘行（桥底顶线 / 桥洞上边界）的掩码行。

    桥掩码 = 桥面 + 桥墩，桥墩识别不准会跳动，不能直接取整块掩码下边缘。
    桥面是横跨画面的水平带：按行统计覆盖率（行内桥梁像素 / 列带宽），
    桥面行覆盖率很高，桥墩行覆盖率很低。取覆盖率 ≥ cover_thresh 的
    **最靠下**的连续行带，其底行即桥面下边缘。
    掩码缺失、无达标行带、或底行贴到帧底时返回 None。
    """
    h, w = bridge.shape
    x0 = int(w * x_band[0])
    x1 = max(x0 + 1, int(w * x_band[1]))
    cover = bridge[:, x0:x1].mean(axis=1)
    rows = np.flatnonzero(cover >= cover_thresh)
    if rows.size < min_band_rows:
        return None

    runs = []
    start = prev = int(rows[0])
    for r in rows[1:]:
        r = int(r)
        if r == prev + 1:
            prev = r
            continue
        runs.append((start, prev))
        start = prev = r
    runs.append((start, prev))

    top, bottom = runs[-1]
    if bottom - top + 1 < min_band_rows:
        return None
    if bottom >= h * max_row_ratio:
        return None
    return float(bottom)


def bridge_panel_anchor(bridge: np.ndarray, *, scale_x: float,
                        scale_y: float, frac: float = 0.20,
                        half_win: int = 12):
    """面板锚点：掩码跨度分位数（2%/98%）的 frac 处，返回该列窗口
    逐列顶/底行的中位数（视频坐标）。"""
    cols = np.flatnonzero(bridge.any(axis=0))
    if cols.size < 16:
        return None
    left = float(np.percentile(cols, 2.0))
    right = float(np.percentile(cols, 98.0))
    xc = left + frac * (right - left)
    x0 = max(0, int(xc) - half_win)
    x1 = min(bridge.shape[1], int(xc) + half_win + 1)
    tops, bots = [], []
    for col in bridge[:, x0:x1].T:
        rows = np.flatnonzero(col)
        if rows.size:
            tops.append(rows[0])
            bots.append(rows[-1])
    if len(tops) < max(4, (x1 - x0) // 2):
        return None
    return (xc * scale_x,
            float(np.median(tops)) * scale_y,
            float(np.median(bots)) * scale_y)


class RobustRowTracker:
    """掩码边缘行跟踪：EMA + 跳变抑制（抗分割抖动）+ 丢失容忍。

    桥墩识别不准会让桥面线单帧跳动：
      1. 跳变抑制：偏离当前值超过 max_jump 像素需连续多帧一致才接受；
      2. 丢失容忍：连续 missing_allow 帧无有效检测才置空。
    """

    def __init__(self, alpha: float = 0.6, max_jump: float = 40.0,
                 jump_confirm: int = 5, missing_allow: int = 8):
        self.alpha = float(alpha)
        self.max_jump = float(max_jump)
        self.jump_confirm = int(jump_confirm)
        self.missing_allow = int(missing_allow)
        self.row: float | None = None
        self.missing = 0
        self._pending: float | None = None
        self._pending_count = 0

    def reset(self) -> None:
        self.row = None
        self.missing = 0
        self._pending = None
        self._pending_count = 0

    def update(self, row: float | None) -> float | None:
        if row is None:
            self.missing += 1
            if self.missing > self.missing_allow:
                self.row = None
            return self.row
        self.missing = 0

        if self.row is None:
            self.row = float(row)
            self._pending = None
            self._pending_count = 0
            return self.row

        if abs(row - self.row) > self.max_jump:
            if (self._pending is not None
                    and abs(row - self._pending) <= self.max_jump):
                self._pending_count += 1
            else:
                self._pending = float(row)
                self._pending_count = 1
            if self._pending_count < self.jump_confirm:
                return self.row
            self.row = self._pending
            self._pending = None
            self._pending_count = 0
            return self.row

        self._pending = None
        self._pending_count = 0
        self.row = self.alpha * float(row) + (1.0 - self.alpha) * self.row
        return self.row


# ===========================================================================
# 实时引擎：每帧吃 桥梁掩码 + 船位，产出待渲染的像素坐标 dict
# ===========================================================================
class BridgeWarningEngine:
    """每帧只算坐标与数据，绘制交给 Godot。

    update() 输出的 dict 经 8765 导航数据包 bridge_warning 字段下发：
      bridge_name / dist_m / bridge_h / ship_h / clearance_m / pass_ok / too_close
      bridge_id / alert_level  仅平台 Redis 有数据时附带
      warn: {y, x0, x1}        净空警戒线线段（视频像素坐标）
      panel: {x, top, bottom}  信息面板锚点（视频像素坐标）
    坐标均为完整视频分辨率空间；Godot 按 nav 包 width/height 自行缩放。
    距离/桥高/船高优先取平台 Redis（BridgeAlertClient），取不到退回预存
    桥梁库；目标桥与像素几何始终由桥梁库经纬度 + 分割桥面线决定。
    净空高度 = bridge_h − ship_h，与数值来源无关——两条路径同式，面板含义
    才唯一（bridge_h 是桥高，平台给实时桥高、预存库给核定桥高）。
    """

    def __init__(self, *, width: int, height: int, calib_path: str,
                 ship_height: float = VESSEL_HEIGHT_M,
                 camera_height: float | None = None,
                 yaw_offset_deg: float = 0.0,
                 panel_range_m: float = BRIDGE_PANEL_RANGE_M,
                 warn_range_m: float = BRIDGE_WARN_RANGE_M,
                 deck_cover_thresh: float = 0.55,
                 deck_row_offset: float = 5.0,
                 deck_alpha: float = 0.6,
                 deck_max_jump: float = 40.0,
                 min_draw_dist: float = 10.0,
                 coord_scale: float = 1.0,
                 diag_interval: float = 1.0,
                 alert_client: "BridgeAlertClient | None" = None):
        self.width = int(width)
        self.height = int(height)
        self.ship_height = float(ship_height)
        self.camera_h = (float(camera_height) if camera_height is not None
                         else float(ship_height))
        self.yaw_offset_deg = float(yaw_offset_deg)
        self.panel_range_m = float(panel_range_m)
        self.warn_range_m = float(warn_range_m)
        self.deck_cover_thresh = float(deck_cover_thresh)
        self.min_draw_dist = float(min_draw_dist)
        self.diag_interval = float(diag_interval)
        self._diag_last = 0.0
        self.alert_client = alert_client
        self._alert_logged = None
        # 坐标空间缩放：离线脚本的像素偏移（桥面线上移 5px、警戒线中心
        # w/2−50、半长 25）按整幅分辨率标定；本引擎在发送尺寸空间出坐标，
        # 这些偏移必须乘 coord_scale（send_w/width）才能逐像素对齐离线结果
        self.deck_row_offset = float(deck_row_offset) * float(coord_scale)
        self.warn_center_offset = 50.0 * float(coord_scale)
        self.warn_half_len = 25.0 * float(coord_scale)
        self.up_warn = self.ship_height - self.camera_h
        self.projector = PinholeProjector(calib_path, self.width, self.height)

        self.deck_tracker = RobustRowTracker(
            alpha=deck_alpha, max_jump=deck_max_jump)
        # 面板锚点比桥面线更慢更稳；只在推理帧喂一次，避免同一噪声样本
        # 被 EMA 连吃多次
        self.panel_x_tracker = RobustRowTracker(alpha=0.15, max_jump=60.0)
        self.panel_top_tracker = RobustRowTracker(alpha=0.15, max_jump=30.0)
        self.panel_bot_tracker = RobustRowTracker(alpha=0.15, max_jump=30.0)
        self.current_bridge: str | None = None

    def _reset_trackers(self) -> None:
        self.deck_tracker.reset()
        self.panel_x_tracker.reset()
        self.panel_top_tracker.reset()
        self.panel_bot_tracker.reset()

    def update(self, bridge_mask: np.ndarray, lat: float, lon: float,
               heading_deg: float):
        """bridge_mask：分割输出的桥梁掩码（任意分辨率，按自身 shape 缩放到
        视频坐标）；lat/lon/heading_deg：当前船位与艏向（度，真方位）。
        返回 bridge_warning dict；无目标桥 / 无船位时返回 None。"""
        result = self._compute(bridge_mask, lat, lon, heading_deg)
        self._diag(bridge_mask, lat, lon, heading_deg, result)
        return result

    def _diag(self, bridge_mask, lat, lon, heading_deg, result) -> None:
        if self.diag_interval <= 0.0:
            return
        now = time.monotonic()
        if now - self._diag_last < self.diag_interval:
            return
        self._diag_last = now
        if bridge_mask is None or not bridge_mask.any():
            mstat = "掩码为空"
        else:
            rr = np.flatnonzero(bridge_mask.any(axis=1))
            cc = np.flatnonzero(bridge_mask.any(axis=0))
            mstat = (f"掩码{bridge_mask.shape[1]}x{bridge_mask.shape[0]} "
                     f"行{rr[0]}~{rr[-1]} 列{cc[0]}~{cc[-1]}")
        

    def _compute(self, bridge_mask: np.ndarray, lat: float, lon: float,
                 heading_deg: float):
        if bridge_mask is None:
            return None
        mh, mw = bridge_mask.shape
        scale_x = self.width / float(mw)
        scale_y = self.height / float(mh)

        deck_mask = bridge_deck_bottom_from_masks(
            bridge_mask, cover_thresh=self.deck_cover_thresh)
        deck_raw = (deck_mask * scale_y - self.deck_row_offset
                    if deck_mask is not None else None)
        panel_raw = bridge_panel_anchor(
            bridge_mask, scale_x=scale_x, scale_y=scale_y)
        panel_x = self.panel_x_tracker.update(
            panel_raw[0] if panel_raw is not None else None)
        panel_top = self.panel_top_tracker.update(
            panel_raw[1] if panel_raw is not None else None)
        panel_bot = self.panel_bot_tracker.update(
            panel_raw[2] if panel_raw is not None else None)
        deck_obs = self.deck_tracker.update(deck_raw)

        # 桥梁候选集：平台桥梁表优先（净空/坐标以平台为准），读不到用硬编码库
        library = ((self.alert_client.bridge_library()
                    if self.alert_client is not None else None)
                   or BRIDGE_LIBRARY)
        target = pick_target_bridge(
            lat, lon, heading_deg,
            panel_range_m=self.panel_range_m, library=library)
        if target is None:
            if self.current_bridge is not None:
                self._reset_trackers()
                self.current_bridge = None
            return None
        if target["name"] != self.current_bridge:
            # 切换目标桥：复位全部时序平滑，避免旧桥锚点混入新桥
            self._reset_trackers()
            self.current_bridge = target["name"]
            deck_obs = panel_x = panel_top = panel_bot = None
            

        forward, left = enu_to_body(
            *latlon_to_enu(target["lat"], target["lon"], lat, lon),
            heading_deg, self.yaw_offset_deg)
        dist = float(target["dist"])
        bridge_h = float(target["clearance_m"])
        ship_h = self.ship_height
        up_warn = self.up_warn
        alert_extra: dict = {}

        # 平台 Redis 优先：距离/桥高/船高用平台值，取不到才退回预存桥梁库
        rec = self.alert_client.nearest() if self.alert_client is not None else None
        if rec is not None:
            dist = float(rec["distance"])
            if rec["clearance_height"] is not None:
                bridge_h = float(rec["clearance_height"])
            if rec["ship_height"] is not None:
                ship_h = float(rec["ship_height"])
                up_warn = ship_h - self.camera_h
            alert_extra = {"bridge_id": rec["bridge_id"],
                           "alert_level": rec["alert_level"]}
            if self._alert_logged != rec["bridge_id"]:
                self._alert_logged = rec["bridge_id"]
                print(f"[BridgeAlert] 平台 桥id={rec['bridge_id']} "
                      f"级别={rec['alert_level']} 距桥={dist:.1f}m "
                      f"船高={ship_h:.1f}m 桥高={bridge_h:.1f}m | "
                      f"预存库 {target['name']} 距桥={float(target['dist']):.1f}m "
                      f"桥高={float(target['clearance_m']):.1f}m")

        # 净空高度 = 桥高 − 船高，与数值来自平台还是预存库无关（两条路径同式，
        # 否则同一标签一个 9.0 一个 8.7，读的人无法判断含义）
        pass_ok = ship_h <= bridge_h
        clearance_m = bridge_h - ship_h

        result = {
            "bridge_name": target["name"],
            "bridge_id": target.get("id"),
            "dist_m": round(dist, 1),
            "bridge_h": bridge_h,
            "ship_h": ship_h,
            "clearance_m": round(clearance_m, 1),
            "pass_ok": bool(pass_ok),
            "too_close": bool(forward < self.min_draw_dist),
            **alert_extra,
        }

        # 警戒线：仅直线距离 ≤ warn-range 且桥面线有效时计算
        if (dist <= self.warn_range_m and deck_obs is not None
                and forward >= self.min_draw_dist):
            pixels = self.projector.project_body(
                [[forward, left, bridge_h - self.camera_h],
                 [forward, left, up_warn]])
            warn_y = float(deck_obs) + float(pixels[1, 1] - pixels[0, 1])
            if 0.0 <= warn_y < self.height:
                xc = self.width / 2.0 - self.warn_center_offset
                result["warn"] = {
                    "y": round(warn_y, 1),
                    "x0": round(xc - self.warn_half_len, 1),
                    "x1": round(xc + self.warn_half_len, 1),
                }

        if (panel_x is not None and panel_top is not None
                and forward >= self.min_draw_dist):
            result["panel"] = {
                "x": round(float(panel_x), 1),
                "top": round(float(panel_top), 1),
                "bottom": round(
                    float(panel_bot if panel_bot is not None
                          else panel_top), 1),
            }
        return result
