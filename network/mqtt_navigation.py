from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable

from utils.realtime_navigation import RealtimeNavigationState

try:
    import paho.mqtt.client as mqtt
except ImportError:  # Kept optional so the pure navigation logic remains testable.
    mqtt = None


DEFAULT_PREDICTION_TOPIC = "v1/11/prediction/result"
DEFAULT_GNSS_TOPIC = "v1/11/sensor/gnss/gnss_01"


@dataclass(frozen=True)
class MqttNavigationConfig:
    host: str = "127.0.0.1"
    port: int = 1883
    prediction_topic: str = DEFAULT_PREDICTION_TOPIC
    gnss_topic: str = DEFAULT_GNSS_TOPIC
    client_prefix: str = "nav_visualizer"
    username: str | None = None
    password: str | None = None
    keepalive_s: int = 30


def _mqtt_client(client_id: str):
    if mqtt is None:
        raise RuntimeError(
            "paho-mqtt is required for realtime navigation; "
            "install paho-mqtt>=2.0"
        )
    callback_api = getattr(mqtt, "CallbackAPIVersion", None)
    if callback_api is not None:
        return mqtt.Client(
            callback_api.VERSION1,
            client_id=client_id,
            protocol=mqtt.MQTTv5,
        )
    return mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5)


def _reason_code_value(reason_code) -> int:
    return int(getattr(reason_code, "value", reason_code))


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _extract_dcpa_alert(data: Mapping) -> dict | None:
    """取 dcpaResults 首条的 DCPA/TCPA/距离与提示文案。

    collisionAlert=0 时仍可能带有效目标，因此只按条目自身的字段是否可用判定。
    """
    results = data.get("dcpaResults")
    if not isinstance(results, list) or not results:
        return None

    entry = results[0]
    if not isinstance(entry, Mapping):
        return None

    dcpa = _optional_float(entry.get("dcpa"))
    tcpa = _optional_float(entry.get("tcpa"))
    distance = _optional_float(entry.get("distance"))
    if dcpa is None or tcpa is None or distance is None:
        return None

    alert_level = entry.get("alertLevel")
    alert_info = entry.get("alertInfo")
    alert = {
        "dcpa": dcpa,
        "tcpa": tcpa,
        "distance": distance,
        "alert_level": str(alert_level) if alert_level else "",
        "alert_info": str(alert_info) if alert_info else "",
    }

    # 避障回传的目标像素框：四个值齐全且构成合法矩形才可用
    pixels = [
        _optional_float(entry.get(key))
        for key in ("pixel_x1", "pixel_y1", "pixel_x2", "pixel_y2")
    ]
    if all(value is not None for value in pixels):
        x1, y1, x2, y2 = pixels
        if x2 > x1 and y2 > y1:
            alert["pixel_bbox"] = (x1, y1, x2, y2)

    return alert


def _parse_trajectory_points(
    raw_points: Sequence,
) -> tuple[list[dict], list[float | None]]:
    """Parse ``{latitude, longitude, timestamp}`` points into radians.

    The latest navigation payload sends coordinates in degrees (e.g.
    22.7459629) while the internal mission format uses radians, so degrees
    are detected and converted automatically. Returns the mission points and
    the per-point timestamps (None where a point carries none).
    """
    parsed: list[tuple[float, float]] = []
    timestamps: list[float | None] = []
    for point in raw_points:
        if not isinstance(point, Mapping):
            continue
        latitude = point.get("latitude", point.get("lat"))
        longitude = point.get("longitude", point.get("lon", point.get("lng")))
        if latitude is None or longitude is None:
            continue
        try:
            lat = float(latitude)
            lon = float(longitude)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue
        timestamps.append(_optional_float(point.get("timestamp")))
        parsed.append((lat, lon))

    if not parsed:
        raise ValueError("Trajectory list has no valid latitude/longitude points")

    uses_degrees = any(
        abs(lat) > math.pi / 2.0 or abs(lon) > math.pi for lat, lon in parsed
    )
    scale = math.pi / 180.0 if uses_degrees else 1.0
    points = [
        {"latitude": lat * scale, "longitude": lon * scale} for lat, lon in parsed
    ]
    return points, timestamps


def _normalise_prediction(
    data: Mapping,
) -> tuple[dict, list[float | None] | None, str]:
    """Convert prediction payloads into the internal mission shape.

    Supports the latest flat navigation payload (``avoidanceTrajectory`` with
    degree coordinates) and the legacy avoidance-result shapes (radian
    coordinates). Returns ``(mission, per_point_timestamps, source)``; the
    timestamps are None for legacy payloads that carry none.
    """
    # 导航带只使用避障路径 avoidanceTrajectory，不再回退到 positionList（本船历史航迹）
    points = data.get("avoidanceTrajectory")
    if isinstance(points, list) and points:
        mission_points, timestamps = _parse_trajectory_points(points)
        return (
            {"count": 1, "tolerance": 5.0, "points": mission_points},
            timestamps,
            "avoidanceTrajectory",
        )

    candidate = data
    points = None
    for key in ("result", "prediction"):
        nested = candidate.get(key)
        if isinstance(nested, Mapping):
            candidate = nested
            break
        if isinstance(nested, list):
            points = nested
            break

    if points is None:
        points = candidate.get("points")
    if not isinstance(points, list):
        for key in ("path", "trajectory", "route", "waypoints"):
            value = candidate.get(key)
            if isinstance(value, list):
                points = value
                break
    if not isinstance(points, list) or not points:
        raise ValueError("Prediction result needs a non-empty path array")

    normalised_points = []
    for index, point in enumerate(points):
        if isinstance(point, Mapping):
            latitude = point.get("latitude", point.get("lat"))
            longitude = point.get("longitude", point.get("lon", point.get("lng")))
            tolerance = point.get("tolerance")
        elif (
            isinstance(point, Sequence)
            and not isinstance(point, (str, bytes, bytearray))
            and len(point) >= 2
        ):
            latitude, longitude = point[0], point[1]
            tolerance = None
        else:
            raise ValueError(f"Prediction point {index} has an unsupported shape")
        item = {"latitude": latitude, "longitude": longitude}
        if tolerance is not None:
            item["tolerance"] = tolerance
        normalised_points.append(item)

    return (
        {
            "count": candidate.get("count", 1),
            "tolerance": candidate.get("tolerance", 5.0),
            "points": normalised_points,
        },
        None,
        "legacy",
    )


class MqttNavigationClient:
    # 预测诊断打印的最小间隔（秒）：高频消息下每 5 秒最多打印一批，防止刷屏
    _PREDICTION_LOG_INTERVAL_S = 5.0

    def __init__(
        self,
        state: RealtimeNavigationState,
        config: MqttNavigationConfig,
        *,
        clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mqtt is None:
            raise RuntimeError(
                "paho-mqtt is required for realtime navigation; "
                "install paho-mqtt>=2.0"
            )
        if not config.host.strip():
            raise ValueError("MQTT host cannot be empty")
        if not 1 <= config.port <= 65535:
            raise ValueError("MQTT port must be between 1 and 65535")
        if not config.prediction_topic.strip() or not config.gnss_topic.strip():
            raise ValueError("MQTT topics cannot be empty")
        self.state = state
        self.config = config
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._client = None
        self._gnss_received_count = 0
        self._gnss_rejected_count = 0
        self._connect_fail_count = 0
        # 上一次放行预测诊断打印的单调时刻（None 表示尚未打印过）
        self._last_prediction_log_mono: float | None = None
        # 上一次放行 ignored 错误打印的单调时刻（独立于正常日志的节流窗口）
        self._last_ignored_log_mono: float | None = None
        # 解析校验统计（每 30 秒汇总打印一次，用于确认 MQTT 数据解析是否成功）
        self._stats = {
            "prediction_received": 0,
            "prediction_parsed": 0,
            "prediction_failed": 0,
            "prediction_activated": 0,
            "gnss_ok": 0,
            "gnss_failed": 0,
            "decode_failed": 0,
        }
        self._last_report_mono: float | None = None

    def start(self) -> None:
        suffix = uuid.uuid4().hex[:8]
        client = _mqtt_client(f"{self.config.client_prefix}_{suffix}")
        if self.config.username:
            client.username_pw_set(self.config.username, self.config.password)
        client.reconnect_delay_set(min_delay=1, max_delay=10)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        # connect_async 连接失败时不会抛异常，只会静默重试；必须挂回调才可见
        client.on_connect_fail = self._on_connect_fail
        self._client = client
        client.connect_async(
            self.config.host, port=self.config.port, keepalive=self.config.keepalive_s
        )
        client.loop_start()
        print(
            f"[MQTT Nav] starting broker={self.config.host}:{self.config.port}; "
            f"prediction={self.config.prediction_topic}; GNSS={self.config.gnss_topic}"
        )

    def _on_connect_fail(self, _client) -> None:
        # 前 3 次 + 之后每 30 次打印一次，避免长时间运行刷屏
        self._connect_fail_count += 1
        if self._connect_fail_count <= 3 or self._connect_fail_count % 30 == 0:
            print(
                f"[MQTT Nav] CANNOT CONNECT to broker {self.config.host}:{self.config.port} "
                f"(attempt #{self._connect_fail_count}); 检查 broker 是否运行/"
                f"--mqtt-host 是否正确，后台将继续重试",
                flush=True,
            )

    def stop(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            client.disconnect()
        except Exception as exc:
            print(f"[MQTT Nav] disconnect warning: {exc}")
        finally:
            client.loop_stop()
        self._client = None

    def _on_connect(self, client, _userdata, _flags, reason_code, _properties=None) -> None:
        if _reason_code_value(reason_code) != 0:
            print(f"[MQTT Nav] connection rejected: {reason_code}")
            return
        client.subscribe(self.config.prediction_topic, qos=1)
        client.subscribe(self.config.gnss_topic, qos=0)
        print(
            f"[MQTT Nav] subscribed {self.config.prediction_topic} and "
            f"{self.config.gnss_topic}"
        )

    @staticmethod
    def _on_disconnect(_client, _userdata, reason_code, _properties=None) -> None:
        if _reason_code_value(reason_code) != 0:
            print(f"[MQTT Nav] disconnected unexpectedly: {reason_code}")

    @staticmethod
    def _decode_message(message) -> tuple[dict, float | None]:
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid MQTT JSON on {message.topic}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"MQTT message on {message.topic} is not a JSON object")
        data = payload.get("data")
        if isinstance(data, dict):
            # Legacy envelope: {"ts": ..., "data": {...}}.
            return data, _optional_float(payload.get("ts"))
        # Latest navigation payload is the message body itself (flat object).
        return payload, None

    def _on_message(self, _client, _userdata, message) -> None:
        try:
            try:
                data, envelope_ts = self._decode_message(message)
            except ValueError as exc:
                self._stats["decode_failed"] += 1
                raise exc
            if message.topic == self.config.prediction_topic:
                self._stats["prediction_received"] += 1
                self._handle_prediction(data, envelope_ts)
            elif message.topic == self.config.gnss_topic:
                self._handle_gnss(data, envelope_ts)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            # 按来源分类失败计数，供周期性校验报告汇总
            if message.topic == self.config.prediction_topic:
                self._stats["prediction_failed"] += 1
            elif message.topic == self.config.gnss_topic:
                self._stats["gnss_failed"] += 1
            # 高频失败同样刷屏：ignored 按独立 5 秒窗口节流，
            # 被静默的错误仍计入统计，30 秒的 PARSE CHECK 汇总可见
            now_mono = self._monotonic_clock()
            if (
                self._last_ignored_log_mono is None
                or now_mono - self._last_ignored_log_mono
                >= self._PREDICTION_LOG_INTERVAL_S
            ):
                self._last_ignored_log_mono = now_mono
                print(f"[MQTT Nav] ignored {message.topic}: {exc}", flush=True)
        finally:
            self._maybe_report_stats()

    def _maybe_report_stats(self, interval_s: float = 30.0) -> None:
        """每 interval_s 秒打印一次解析校验汇总（收包/解析/失败/激活）。"""
        now_mono = self._monotonic_clock()
        if self._last_report_mono is None:
            self._last_report_mono = now_mono
            return
        if now_mono - self._last_report_mono < interval_s:
            return
        self._last_report_mono = now_mono
        s = self._stats
        pred_total = s["prediction_parsed"] + s["prediction_failed"]
        pred_rate = (
            f"{s['prediction_parsed'] / pred_total * 100:.0f}%"
            if pred_total > 0
            else "n/a"
        )
        print(
            f"[MQTT Nav] PARSE CHECK: prediction {s['prediction_parsed']}/{pred_total} "
            f"parsed OK ({pred_rate}), activated={s['prediction_activated']}; "
            f"GNSS ok={s['gnss_ok']} rejected={self._gnss_rejected_count} "
            f"failed={s['gnss_failed']}; "
            f"decode errors={s['decode_failed']}",
            flush=True,
        )

    def _prediction_log_ready(self) -> bool:
        """预测消息高频到达，诊断打印按 5 秒节流放行一批，避免刷屏。"""
        now_mono = self._monotonic_clock()
        if (
            self._last_prediction_log_mono is not None
            and now_mono - self._last_prediction_log_mono
            < self._PREDICTION_LOG_INTERVAL_S
        ):
            return False
        self._last_prediction_log_mono = now_mono
        return True

    def _handle_prediction(self, data: Mapping, envelope_ts: float | None) -> None:
        # 诊断：打印收到的原始关键字段，确认消息到达与格式识别；
        # 按 5 秒节流放行一批打印，防止高频消息刷屏
        log = self._prediction_log_ready()
        if log:
            position_list = data.get("positionList")
            avoidance = data.get("avoidanceTrajectory")
            print(
                f"[MQTT Nav] received prediction: shipId={data.get('shipId')}, "
                f"positionList={len(position_list) if isinstance(position_list, list) else 'missing'} pts, "
                f"avoidanceTrajectory={len(avoidance) if isinstance(avoidance, list) else 'missing'} pts, "
                f"trajectoryLength_m={data.get('trajectoryLength_m')}, "
                f"collisionAlert={data.get('collisionAlert')}, "
                f"dcpaResults={len(data.get('dcpaResults') or [])}, "
                f"obstacleAlerts={len(data.get('obstacleAlerts') or [])}",
                flush=True,
            )
            # 打印完整 avoidanceTrajectory 数据（所有点的经纬度），供确认避障路径内容
            if isinstance(avoidance, list):
                print(
                    f"[MQTT Nav] avoidanceTrajectory data ({len(avoidance)} pts):",
                    flush=True,
                )
                for idx, pt in enumerate(avoidance):
                    if isinstance(pt, Mapping):
                        lat = pt.get("latitude", pt.get("lat"))
                        lon = pt.get("longitude", pt.get("lon", pt.get("lng")))
                        print(f"  [{idx:02d}] lat={lat}, lon={lon}", flush=True)
                    else:
                        print(f"  [{idx:02d}] {pt!r}", flush=True)

        mission, _, source = _normalise_prediction(data)
        # 触发时刻 = 收到避障路径那一刻 GNSS 流的最新信封时间戳（envelope_ts）：
        # 与缓冲区样本同源同时刻，往前取 15 个点算初速必然命中。
        # 屏蔽 result 里携带的所有时间戳（inputPosition.timestamp、positionList/
        # avoidanceTrajectory 逐点 timestamp），一律不作为时间基准。
        # 缓冲区为空（开机未收到 GNSS）时回退本机时钟。
        now = self._clock()
        trigger = now
        trigger_name = "local-clock"
        latest_gnss_ts = self.state.latest_gnss_timestamp()
        if latest_gnss_ts is not None:
            trigger = latest_gnss_ts
            trigger_name = "gnss-latest"

        timestamp = trigger
        # 走到这里 = 坐标解析通过，计为解析成功
        self._stats["prediction_parsed"] += 1
        # DCPA 目标与航线激活解耦：激活失败（如开机 GNSS 点不足）时该信息仍有效
        self.state.set_dcpa_alert(_extract_dcpa_alert(data))

        # 诊断：打印解析结果（坐标单位、触发时间戳、首点坐标）；同样按 5 秒节流
        first_point = mission["points"][0]
        if log:
            print(
                f"[MQTT Nav] parsed OK: source={source}, "
                f"points={len(mission['points'])}, "
                f"trigger_ts={timestamp:.3f} ({trigger_name}), "
                f"first=({first_point['latitude']:.8f}, "
                f"{first_point['longitude']:.8f}) rad",
                flush=True,
            )

        try:
            route = self.state.activate_mission(
                mission,
                float(timestamp),
                now_epoch=now,
                now_monotonic=self._monotonic_clock(),
            )
        except ValueError as exc:
            # 已确认的策略：激活失败（如启动初期 15 个 GNSS 点不足）即放弃本条，
            # 等待下一条预测消息（后端持续发布，其 input_ts 更新后自然满足）
            if log:
                print(
                    f"[MQTT Nav] parsed but NOT activated (source={source}): {exc}",
                    flush=True,
                )
            raise
        self._stats["prediction_activated"] += 1
        if log:
            print(
                f"[MQTT Nav] activated navigation band ({source}): "
                f"{len(route.mission_east_m)} points, {route.length_m:.2f}m",
                flush=True,
            )

    def _handle_gnss(self, data: Mapping, envelope_ts: float | None) -> None:
        timestamp = envelope_ts
        if timestamp is None:
            raise ValueError("GNSS sample has no timestamp")
        timestamp = float(timestamp)
        # 调试：定位 latest_ts 与 inputPosition.timestamp 偏差来源
        # （信封 ts=发布时刻 vs data.sample_ts=采样时刻），验证后删除
        if self._gnss_received_count % 100 == 0:
            sample_ts = data.get("sample_ts")
            delta = ""
            if sample_ts is not None and envelope_ts is not None:
                delta = f" delta={float(sample_ts) - float(envelope_ts):.3f}s"
            print(
                f"[MQTT Nav][ts-dbg] envelope_ts={envelope_ts} "
                f"sample_ts={sample_ts}{delta}",
                flush=True,
            )
        # GNSS 主题（协议 v3.3 §5.1）约定 lat/lon/heading 均为弧度制；
        # 发布端已改为直接发送弧度，此处直接接收，不再做角度→弧度转换。
        latitude = float(data["lat"])
        longitude = float(data["lon"])
        heading_value = data.get("heading")
        heading = None if heading_value is None else float(heading_value)
        accepted = self.state.add_gnss(
            timestamp,
            latitude,
            longitude,
            heading,
            received_at=self._clock(),
        )
        if not accepted:
            # 20Hz 拒绝会刷屏：前 3 条 + 之后每 100 条打印一次
            self._gnss_rejected_count += 1
            if self._gnss_rejected_count <= 3 or self._gnss_rejected_count % 100 == 0:
                print(
                    f"[MQTT Nav] rejected GNSS sample ts={timestamp:.3f} "
                    f"lat={data.get('lat')} lon={data.get('lon')} "
                    f"(#{self._gnss_rejected_count})",
                    flush=True,
                )
            return
        self._gnss_received_count += 1
        self._stats["gnss_ok"] += 1
        # 20Hz 下每 100 个样本约 5 秒，打印一条心跳即可
        if self._gnss_received_count % 100 == 0:
            print(
                f"[MQTT Nav] received GNSS={self._gnss_received_count} "
                f"latest_ts={timestamp:.3f}",
                flush=True,
            )
