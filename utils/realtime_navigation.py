from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


EARTH_RADIUS_M = 6378137.0

# 导航日志统一写入 realtime_navigation.log（不再打印到终端）；
# mqtt_navigation.py 通过 nav_log 复用同一个 logger 与文件。
_NAV_LOGGER = logging.getLogger("realtime_navigation")
if not _NAV_LOGGER.handlers:
    _handler = logging.FileHandler("realtime_navigation.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _NAV_LOGGER.addHandler(_handler)
    _NAV_LOGGER.setLevel(logging.INFO)
_NAV_LOGGER.propagate = False


def nav_log(message: str, *, level: int = logging.INFO) -> None:
    """写入导航日志文件；替代原先的终端 print。"""
    _NAV_LOGGER.log(level, message)


@dataclass(frozen=True)
class GnssSample:
    timestamp: float
    latitude_rad: float
    longitude_rad: float
    heading_rad: float | None = None


@dataclass(frozen=True)
class NavigationSnapshot:
    route_active: bool
    moving: bool
    speed_mps: float
    time_to_60m_s: float
    progress_m: float
    route_length_m: float


def geodetic_distance_m(first: GnssSample, second: GnssSample) -> float:
    d_lat = second.latitude_rad - first.latitude_rad
    d_lon = second.longitude_rad - first.longitude_rad
    hav = (
        math.sin(d_lat * 0.5) ** 2
        + math.cos(first.latitude_rad)
        * math.cos(second.latitude_rad)
        * math.sin(d_lon * 0.5) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, hav)))


def estimate_window_speed(samples: Sequence[GnssSample]) -> float:
    """Estimate ground speed from cumulative distance over the whole window."""
    if len(samples) < 2:
        return 0.0
    elapsed_s = float(samples[-1].timestamp - samples[0].timestamp)
    if elapsed_s <= 1e-6:
        return 0.0
    distance_m = sum(
        geodetic_distance_m(first, second)
        for first, second in zip(samples[:-1], samples[1:])
        if second.timestamp > first.timestamp
    )
    speed_mps = distance_m / elapsed_s
    return speed_mps if math.isfinite(speed_mps) else 0.0


def _mission_points(mission: Mapping) -> list[Mapping]:
    if not isinstance(mission, Mapping):
        raise ValueError("Mission must be a JSON object")
    if not all(key in mission for key in ("count", "tolerance", "points")):
        raise ValueError("Mission must contain count, tolerance and points")
    points = mission["points"]
    if not isinstance(points, list) or not points:
        raise ValueError("Mission points must be a non-empty array")
    try:
        repeat_value = mission["count"]
        repeat = int(repeat_value)
        tolerance = float(mission["tolerance"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Mission count/tolerance is invalid") from exc
    if isinstance(repeat_value, bool) or float(repeat_value) != repeat:
        raise ValueError("Mission count must be an integer")
    if repeat < 0 or not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Mission count/tolerance cannot be negative")
    expanded_count = len(points) * (2 if repeat == 0 else repeat)
    if expanded_count > 10000:
        raise ValueError("Expanded mission exceeds 10000 points")
    if repeat == 0:
        return [*points, *reversed(points)]
    return points * repeat


@dataclass(frozen=True)
class EnuMissionRoute:
    east_m: np.ndarray
    north_m: np.ndarray
    arc_m: np.ndarray
    mission_east_m: np.ndarray
    mission_north_m: np.ndarray
    alignment_east_m: float
    alignment_north_m: float

    @classmethod
    def from_mission(
        cls,
        mission: Mapping,
        origin_latitude_rad: float,
        origin_longitude_rad: float,
    ) -> "EnuMissionRoute":
        expanded = _mission_points(mission)
        latitude = []
        longitude = []
        for index, point in enumerate(expanded):
            if not isinstance(point, Mapping):
                raise ValueError(f"Mission point {index} must be an object")
            try:
                lat = float(point["latitude"])
                lon = float(point["longitude"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Mission point {index} needs finite latitude/longitude"
                ) from exc
            if (
                not math.isfinite(lat)
                or not math.isfinite(lon)
                or abs(lat) > math.pi * 0.5
                or abs(lon) > math.pi
            ):
                raise ValueError(f"Mission point {index} has invalid radian coordinates")
            latitude.append(lat)
            longitude.append(lon)

        latitude_np = np.asarray(latitude, dtype=np.float64)
        longitude_np = np.asarray(longitude, dtype=np.float64)
        raw_mission_east = (
            EARTH_RADIUS_M
            * (longitude_np - float(origin_longitude_rad))
            * math.cos(float(origin_latitude_rad))
        )
        raw_mission_north = EARTH_RADIUS_M * (
            latitude_np - float(origin_latitude_rad)
        )

        # Translate the complete avoidance trajectory by the first-point/GNSS
        # offset. This preserves every route segment while making the received
        # first point coincide with the vessel position at activation time.
        alignment_east = float(raw_mission_east[0])
        alignment_north = float(raw_mission_north[0])
        mission_east = raw_mission_east - alignment_east
        mission_north = raw_mission_north - alignment_north

        # Keep the explicit vessel anchor for route semantics. The aligned first
        # mission point is a duplicate and is removed by the filter below.
        east = np.concatenate(([0.0], mission_east))
        north = np.concatenate(([0.0], mission_north))
        segment = np.hypot(np.diff(east), np.diff(north))
        keep = np.concatenate(([True], segment > 0.002))
        east = east[keep]
        north = north[keep]
        if len(east) < 2:
            raise ValueError("Mission route has no measurable length")
        arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(east), np.diff(north)))))
        if not math.isfinite(float(arc[-1])) or arc[-1] <= 0.01:
            raise ValueError("Mission route has no measurable length")

        arrays = (east, north, arc, mission_east, mission_north)
        for array in arrays:
            array.setflags(write=False)
        return cls(
            *arrays,
            alignment_east_m=alignment_east,
            alignment_north_m=alignment_north,
        )

    @property
    def length_m(self) -> float:
        return float(self.arc_m[-1])

    def _position(self, arc_m: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(arc_m, dtype=np.float64)
        clipped = np.clip(query, 0.0, self.length_m)
        east = np.interp(clipped, self.arc_m, self.east_m)
        north = np.interp(clipped, self.arc_m, self.north_m)
        beyond = np.maximum(0.0, query - self.length_m)
        if np.any(beyond > 0.0):
            final_heading = self.heading_at(self.length_m)
            east = east + beyond * math.sin(final_heading)
            north = north + beyond * math.cos(final_heading)
        return east, north

    def heading_at(self, progress_m: float, point_span: int = 5) -> float:
        progress = float(np.clip(progress_m, 0.0, self.length_m))
        span = max(2, int(point_span))
        # 定位当前进度落在哪一段折线（按累计弧长），从该折点向前取 span 个折点
        index = int(np.searchsorted(self.arc_m, progress, side="right")) - 1
        index = max(0, min(index, len(self.east_m) - 2))
        first = index
        last = min(len(self.east_m) - 1, index + span)
        if last - first < 1:
            first = max(0, len(self.east_m) - 1 - span)
            last = len(self.east_m) - 1
        de = float(self.east_m[last] - self.east_m[first])
        dn = float(self.north_m[last] - self.north_m[first])
        return math.atan2(de, dn)

    def nearest_arc(self, east_m: float, north_m: float) -> float:
        """返回航线上离给定 ENU 点（米）最近的弧长（米）。

        逐段做点到折线段的正交投影，取最近的投影点对应的累计弧长；用于用
        当前 GNSS 船位在避障路径上找最近点，矫正运行点（抗速度积分漂移）。
        """
        ex = self.east_m
        ny = self.north_m
        if len(ex) < 2:
            return 0.0
        p0e, p0n = ex[:-1], ny[:-1]
        p1e, p1n = ex[1:], ny[1:]
        de = p1e - p0e
        dn = p1n - p0n
        seg_len2 = de * de + dn * dn
        denom = np.where(seg_len2 > 1e-12, seg_len2, 1.0)
        t = ((east_m - p0e) * de + (north_m - p0n) * dn) / denom
        t = np.clip(t, 0.0, 1.0)
        proj_e = p0e + t * de
        proj_n = p0n + t * dn
        d2 = (proj_e - east_m) ** 2 + (proj_n - north_m) ** 2
        idx = int(np.argmin(d2))
        arc_start = float(self.arc_m[idx])
        arc_span = float(self.arc_m[idx + 1] - self.arc_m[idx])
        return arc_start + float(t[idx]) * arc_span

    def body_window(
        self,
        progress_m: float,
        lookahead_m: float,
        sample_spacing_m: float,
        heading_offset_deg: float = 0.0,
        yaw_rad: float | None = None,
    ) -> np.ndarray:
        progress = float(np.clip(progress_m, 0.0, self.length_m))
        count = max(2, int(round(float(lookahead_m) / sample_spacing_m)) + 1)
        local_arc = np.linspace(0.0, float(lookahead_m), count)
        east, north = self._position(progress + local_arc)
        anchor_east, anchor_north = float(east[0]), float(north[0])
        if yaw_rad is None:
            yaw_rad = self.heading_at(progress)
        yaw = float(yaw_rad) + math.radians(float(heading_offset_deg))
        delta_east = east - anchor_east
        delta_north = north - anchor_north
        # 第 0 列：沿航线方向（前向）；第 1 列：横向，取负使左舷为正，
        # 与标定 forward 约定（X前 Y左 Z上）一致，避免左右镜像。
        body = np.column_stack(
            (
                delta_east * math.sin(yaw) + delta_north * math.cos(yaw),
                -(delta_east * math.cos(yaw) - delta_north * math.sin(yaw)),
            )
        )
        body[0] = [0.0, 0.0]
        return body.astype(np.float32)


class RealtimeNavigationState:
    """Thread-safe GNSS buffer and time-driven mission navigation state."""

    # 航线日志的最小间隔（秒）：预测消息高频到达时每 3 秒最多打印一批
    _LOG_INTERVAL_S = 3.0

    def __init__(
        self,
        *,
        lookahead_m: float = 30.0,
        sample_spacing_m: float = 0.25,
        speed_window_points: int = 15,
        speed_update_hz: float = 10.0,
        moving_threshold_mps: float = 1.0,
        timestamp_tolerance_s: float = 1.0,
        heading_offset_deg: float = 0.0,
        gnss_cache_window_s: float = 2.0,
        route_heading_points: int = 5,
        gnss_heading_points: int = 5,
        heading_ema_alpha: float = 0.25,
    ):
        if lookahead_m <= 0.0 or sample_spacing_m <= 0.0:
            raise ValueError("Navigation distances must be greater than zero")
        if speed_window_points < 2 or speed_update_hz <= 0.0:
            raise ValueError("GNSS speed window and update rate are invalid")
        if moving_threshold_mps < 0.0 or timestamp_tolerance_s <= 0.0:
            raise ValueError("Navigation thresholds are invalid")
        if gnss_cache_window_s <= 0.0:
            raise ValueError("GNSS cache window must be greater than zero")
        if route_heading_points < 2 or gnss_heading_points < 2:
            raise ValueError("Heading point spans must be at least 2")
        if not 0.0 < heading_ema_alpha <= 1.0:
            raise ValueError("Heading EMA alpha must be in (0, 1]")

        self.lookahead_m = float(lookahead_m)
        self.sample_spacing_m = float(sample_spacing_m)
        self.speed_window_points = int(speed_window_points)
        self.speed_update_interval_s = 1.0 / float(speed_update_hz)
        self.moving_threshold_mps = float(moving_threshold_mps)
        self.timestamp_tolerance_s = float(timestamp_tolerance_s)
        self.heading_offset_deg = float(heading_offset_deg)
        self.gnss_cache_window_s = float(gnss_cache_window_s)
        self.route_heading_points = int(route_heading_points)
        self.gnss_heading_points = int(gnss_heading_points)
        self.heading_ema_alpha = float(heading_ema_alpha)

        self._lock = threading.RLock()
        # 本地 GNSS 缓冲区：按时间裁剪，只保留最近 gnss_cache_window_s 秒的样本
        self._samples: deque[GnssSample] = deque()
        self._route: EnuMissionRoute | None = None
        self._route_origin_lat_rad = 0.0
        self._route_origin_lon_rad = 0.0
        self._trigger_timestamp = 0.0
        self._progress_m = 0.0
        self._speed_mps = 0.0
        self._moving = False
        # 平滑后的导航带朝向（真方位角，弧度）及其是否已初始化
        self._band_yaw_rad = 0.0
        self._band_yaw_valid = False
        self._last_speed_update_monotonic: float | None = None
        self._last_progress_monotonic: float | None = None
        # 最近一次 prediction 报文里的 dcpaResults 目标（避障指定的单一目标），
        # 连同到达时刻；TTL 到期即视为无目标
        self._dcpa_alert: Mapping | None = None
        self._dcpa_alert_monotonic: float | None = None
        # 上一次放行航线日志的单调时刻（None 表示尚未打印过）
        self._last_route_log_monotonic: float | None = None
        self._stop_event = threading.Event()
        self._update_thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._update_thread is not None and self._update_thread.is_alive():
                return
            self._stop_event.clear()
            self._update_thread = threading.Thread(
                target=self._update_loop,
                name="Navigation10Hz",
                daemon=True,
            )
            self._update_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._update_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._update_thread = None

    def _update_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            self.update()
            remaining = self.speed_update_interval_s - (time.monotonic() - started)
            self._stop_event.wait(max(0.0, remaining))

    def add_gnss(
        self,
        timestamp: float,
        latitude_rad: float,
        longitude_rad: float,
        heading_rad: float | None = None,
        *,
        received_at: float | None = None,
    ) -> bool:
        received = time.time() if received_at is None else float(received_at)
        try:
            sample = GnssSample(
                float(timestamp),
                float(latitude_rad),
                float(longitude_rad),
                None if heading_rad is None else float(heading_rad),
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not math.isfinite(sample.timestamp)
            or not math.isfinite(sample.latitude_rad)
            or not math.isfinite(sample.longitude_rad)
            or sample.timestamp <= 0.0
            or abs(sample.latitude_rad) > math.pi * 0.5
            or abs(sample.longitude_rad) > math.pi
            or sample.timestamp > received + self.timestamp_tolerance_s
        ):
            return False
        with self._lock:
            if self._samples and sample.timestamp <= self._samples[-1].timestamp:
                return False
            self._samples.append(sample)
            # 按时间裁剪：只保留最近 gnss_cache_window_s 秒的样本
            cutoff = sample.timestamp - self.gnss_cache_window_s
            while self._samples[0].timestamp < cutoff:
                self._samples.popleft()
        return True

    def latest_gnss_timestamp(self) -> float | None:
        """GNSS 流最新样本的时间戳；尚未收到任何样本时返回 None。"""
        with self._lock:
            if not self._samples:
                return None
            return self._samples[-1].timestamp

    def latest_vessel_fix(self) -> tuple[float, float, float] | None:
        """当前船位（纬度/经度，度）与艏向（度，真方位，0=北）。

        尚无 GNSS 样本或艏向不可解（静止且报文无 heading）时返回 None。
        供桥梁警戒等旁路模块取实时船位。
        """
        with self._lock:
            if not self._samples:
                return None
            heading = self._vessel_heading_locked()
            if heading is None:
                return None
            latest = self._samples[-1]
            return (
                math.degrees(latest.latitude_rad),
                math.degrees(latest.longitude_rad),
                math.degrees(heading) % 360.0,
            )

    def set_dcpa_alert(self, alert: Mapping | None) -> None:
        """记录避障回传的单一目标信息；传 None 表示当前无目标。"""
        with self._lock:
            self._dcpa_alert = alert
            self._dcpa_alert_monotonic = time.monotonic() if alert else None

    def dcpa_alert(self, max_age_s: float = 5.0) -> Mapping | None:
        """最近一次 dcpaResults 目标；超过 max_age_s 未更新则视为无目标。"""
        with self._lock:
            if self._dcpa_alert is None or self._dcpa_alert_monotonic is None:
                return None
            if time.monotonic() - self._dcpa_alert_monotonic > max_age_s:
                return None
            return self._dcpa_alert

    def enu_minimap(self) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """返回俯视小图所需的当前航线 ENU 轨迹与船位。

        Returns ``(east_m, north_m, cur_east, cur_north)``：east/north 为航线
        折点坐标（米，首点即起点 [0,0]），cur_east/cur_north 为按当前进度
        插值得到的船位（米）。未激活避障航线时返回 None。
        """
        with self._lock:
            route = self._route
            if route is None:
                return None
            east = route.east_m
            north = route.north_m
            cur_e, cur_n = route._position(np.asarray([self._progress_m]))
            return east, north, float(cur_e[0]), float(cur_n[0])

    def activate_mission(
        self,
        mission: Mapping,
        trigger_timestamp: float,
        *,
        now_epoch: float | None = None,
        now_monotonic: float | None = None,
    ) -> EnuMissionRoute:
        trigger = float(trigger_timestamp)
        epoch = time.time() if now_epoch is None else float(now_epoch)
        monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not math.isfinite(trigger) or trigger <= 0.0:
            raise ValueError("Avoidance command timestamp is invalid")

        with self._lock:
            preceding = [sample for sample in self._samples if sample.timestamp <= trigger]
            if len(preceding) < self.speed_window_points:
                raise ValueError(
                    f"Need {self.speed_window_points} GNSS points before the command timestamp"
                )
            aligned = preceding[-1]
            skew = trigger - aligned.timestamp
            if skew < 0.0 or skew > self.timestamp_tolerance_s:
                raise ValueError(
                    f"GNSS/command timestamp skew {skew:.3f}s exceeds "
                    f"{self.timestamp_tolerance_s:.3f}s"
                )
            speed_window = preceding[-self.speed_window_points :]
            initial_speed = estimate_window_speed(speed_window)
            route = EnuMissionRoute.from_mission(
                mission,
                aligned.latitude_rad,
                aligned.longitude_rad,
            )

            elapsed_since_command = max(0.0, epoch - trigger)
            initial_progress = (
                initial_speed * elapsed_since_command
                if initial_speed >= self.moving_threshold_mps
                else 0.0
            )
            self._route = route
            self._route_origin_lat_rad = aligned.latitude_rad
            self._route_origin_lon_rad = aligned.longitude_rad
            self._band_yaw_valid = False  # 新航线激活，重新锚定带子朝向
            self._trigger_timestamp = trigger
            self._progress_m = min(route.length_m, initial_progress)
            self._speed_mps = initial_speed
            self._moving = initial_speed >= self.moving_threshold_mps
            self._last_speed_update_monotonic = monotonic
            self._last_progress_monotonic = monotonic

        self._log_route(route, aligned, skew, now_monotonic=monotonic)
        return route

    def _log_route(
        self,
        route: EnuMissionRoute,
        origin: GnssSample,
        skew_s: float,
        *,
        now_monotonic: float | None = None,
    ) -> None:
        # 每条预测消息都会重建航线，逐点 ENU 打印会刷屏：按 3 秒节流放行一批
        monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if (
            self._last_route_log_monotonic is not None
            and monotonic - self._last_route_log_monotonic < self._LOG_INTERVAL_S
        ):
            return
        self._last_route_log_monotonic = monotonic
        segment = np.hypot(np.diff(route.east_m), np.diff(route.north_m))
        nav_log(
            "[RealtimeNav] ENU route ready: "
            f"points={len(route.mission_east_m)}, length={route.length_m:.2f}m, "
            f"max_segment={float(np.max(segment)):.2f}m, timestamp_skew={skew_s:.3f}s"
        )
        nav_log(
            "[RealtimeNav] ENU origin: "
            f"lat={origin.latitude_rad:.10f}rad lon={origin.longitude_rad:.10f}rad"
        )
        nav_log(
            "[RealtimeNav] route first-point alignment: "
            f"east={route.alignment_east_m:.3f}m "
            f"north={route.alignment_north_m:.3f}m; applied to all route points"
        )

    def _latest_speed_locked(self, now_epoch: float) -> float:
        if len(self._samples) < self.speed_window_points:
            return 0.0
        latest = self._samples[-1]
        if abs(float(now_epoch) - latest.timestamp) > self.timestamp_tolerance_s:
            return 0.0
        return estimate_window_speed(list(self._samples)[-self.speed_window_points :])

    def _vessel_heading_locked(self) -> float | None:
        """取船当前"艏向"（真方位角，弧度）。

        优先使用 GNSS 报文自带的 heading 字段（罗经艏向，即船头真实朝向）；
        报文未携带 heading 时才回退到最近 gnss_heading_points 个 GNSS 位置差分
        解算的航迹向（COG）。船静止（位移过小）时航迹向不可靠，返回 None。
        """
        if not self._samples:
            return None
        latest = self._samples[-1]

        # 优先：GNSS 报文自带的艏向（罗经值），避免位置差分受延迟/漂移影响
        if latest.heading_rad is not None and math.isfinite(latest.heading_rad):
            heading = float(latest.heading_rad)
            return math.atan2(math.sin(heading), math.cos(heading))

        # 回退：位置差分航迹向（COG），静止时位移过小视为无效
        tail = list(self._samples)[-self.gnss_heading_points :]
        if len(tail) < 2:
            return None
        first = tail[0]
        last = tail[-1]
        d_lat = last.latitude_rad - first.latitude_rad
        d_lon = last.longitude_rad - first.longitude_rad
        d_east = d_lon * math.cos(first.latitude_rad)
        d_north = d_lat
        if math.hypot(d_east, d_north) * EARTH_RADIUS_M < 0.3:
            return None
        return math.atan2(d_east, d_north)

    def _route_enu_of_sample(self, sample: GnssSample) -> tuple[float, float]:
        """把某条 GNSS 样本投影为相对航线原点的 ENU 米坐标。"""
        d_east = (
            (sample.longitude_rad - self._route_origin_lon_rad)
            * math.cos(self._route_origin_lat_rad)
            * EARTH_RADIUS_M
        )
        d_north = (
            (sample.latitude_rad - self._route_origin_lat_rad) * EARTH_RADIUS_M
        )
        return float(d_east), float(d_north)

    def _correct_progress_locked(self) -> None:
        """用当前 GNSS 船位在避障路径上找最近点，矫正运行点。

        速度推进属于开环积分，长期会因速度估计误差/避障消息延迟而漂移；
        这里用最新 GNSS 位置在路线上做最近点投影，把运行点拉回船实际位置。
        """
        route = self._route
        if route is None or not self._samples:
            return
        east, north = self._route_enu_of_sample(self._samples[-1])
        nearest = route.nearest_arc(east, north)
        if not math.isfinite(nearest):
            return
        self._progress_m = float(np.clip(nearest, 0.0, route.length_m))

    def _update_band_yaw_locked(self) -> None:
        """融合前方航线航向与后方船航向，得到平滑的导航带朝向。

        前向用航线上 route_heading_points 个折点的方向（目标方向），后向用 GNSS
        位置差分得到的船航向（锚定起始方向），二者通过一阶低通 EMA 平滑，避免
        航线方向突变时导航带突甩。
        """
        route = self._route
        if route is None:
            self._band_yaw_rad = 0.0
            self._band_yaw_valid = False
            return
        route_heading = route.heading_at(self._progress_m, self.route_heading_points)
        if not self._band_yaw_valid:
            vessel_heading = self._vessel_heading_locked()
            self._band_yaw_rad = (
                vessel_heading if vessel_heading is not None else route_heading
            )
            self._band_yaw_valid = True
            return
        diff = math.atan2(
            math.sin(route_heading - self._band_yaw_rad),
            math.cos(route_heading - self._band_yaw_rad),
        )
        self._band_yaw_rad += self.heading_ema_alpha * diff

    def _advance_locked(self, now_monotonic: float) -> None:
        if self._last_progress_monotonic is None:
            self._last_progress_monotonic = now_monotonic
            return
        elapsed = max(0.0, now_monotonic - self._last_progress_monotonic)
        self._last_progress_monotonic = now_monotonic
        if self._route is None or not self._moving:
            return
        self._progress_m = min(
            self._route.length_m,
            self._progress_m + self._speed_mps * elapsed,
        )
        if self._progress_m >= self._route.length_m - 1e-6:
            nav_log("[RealtimeNav] Avoidance route completed; returning to straight band")
            self._route = None
            self._progress_m = 0.0
            self._moving = False

    def update(
        self,
        *,
        now_epoch: float | None = None,
        now_monotonic: float | None = None,
    ) -> None:
        epoch = time.time() if now_epoch is None else float(now_epoch)
        monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        with self._lock:
            self._advance_locked(monotonic)
            update_due = (
                self._last_speed_update_monotonic is None
                or monotonic - self._last_speed_update_monotonic
                >= self.speed_update_interval_s - 1e-9
            )
            if update_due:
                self._speed_mps = self._latest_speed_locked(epoch)
                self._moving = self._speed_mps >= self.moving_threshold_mps
                self._correct_progress_locked()
                self._update_band_yaw_locked()
                self._last_speed_update_monotonic = monotonic

    def body_path(
        self,
        *,
        now_epoch: float | None = None,
        now_monotonic: float | None = None,
    ) -> np.ndarray:
        epoch = time.time() if now_epoch is None else float(now_epoch)
        monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        self.update(now_epoch=epoch, now_monotonic=monotonic)
        with self._lock:
            if self._route is not None and self._moving:
                return self._route.body_window(
                    self._progress_m,
                    self.lookahead_m,
                    self.sample_spacing_m,
                    self.heading_offset_deg,
                    self._band_yaw_rad,
                )

        distance = np.linspace(
            0.0,
            self.lookahead_m,
            max(2, int(round(self.lookahead_m / self.sample_spacing_m)) + 1),
            dtype=np.float32,
        )
        return np.column_stack((distance, np.zeros_like(distance))).astype(np.float32)

    def snapshot(self) -> NavigationSnapshot:
        with self._lock:
            route_length = self._route.length_m if self._route is not None else 0.0
            time_to_60 = (
                self.lookahead_m / self._speed_mps
                if self._speed_mps >= self.moving_threshold_mps
                else math.inf
            )
            return NavigationSnapshot(
                route_active=self._route is not None,
                moving=self._moving,
                speed_mps=self._speed_mps,
                time_to_60m_s=time_to_60,
                progress_m=self._progress_m,
                route_length_m=route_length,
            )
