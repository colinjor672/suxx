import socket
import struct
import threading
import time
import math
import json as _json

class FusionReceiver:
    HEADER_FMT = ">IHHIQII"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)
    MAGIC = 0x4655534E

    def __init__(
        self,
        host,
        port,
        reconnect_interval=10.0,
        connect_timeout=5.0,
        recv_timeout=2.0,
        data_ttl=0.5,
        max_payload_size=10 * 1024 * 1024,
    ):
        self.host = str(host)
        self.port = int(port)

        # 连接失败或连接断开后，等待10秒重新连接
        self.reconnect_interval = float(reconnect_interval)

        # 建立TCP连接超时时间
        self.connect_timeout = float(connect_timeout)

        # 已连接后，单次recv等待超时时间
        self.recv_timeout = float(recv_timeout)

        # 融合数据默认有效期
        self.data_ttl = float(data_ttl)

        # 防止异常包声明超大payload
        self.max_payload_size = int(max_payload_size)

        self._lock = threading.Lock()
        self._ships: list = []
        self._last_update_time = 0.0
        self._last_packet_timestamp = 0.0

        self._running = False
        self._connected = False

        self._sock = None
        self._thread = None

        # 用Event代替固定time.sleep，
        # stop时可以立即唤醒重连等待
        self._stop_event = threading.Event()

    def start(self):
        if self._running:
            print("[Fusion] 接收线程已经启动")
            return

        self._running = True
        self._connected = False
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._recv_loop,
            daemon=True,
            name="FusionRecv",
        )
        self._thread.start()

        print(
            f"[Fusion] 接收线程已启动，"
            f"断线后每 {self.reconnect_interval:.0f} 秒重连"
        )

    def stop(self):
        self._running = False
        self._connected = False
        self._stop_event.set()

        sock = self._sock
        self._sock = None

        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except (OSError, AttributeError):
                pass

            try:
                sock.close()
            except OSError:
                pass

        if (
            self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=3.0)

        self._clear_ships()

        print("[Fusion] 已停止")

    def get_ships(self, ttl=None):
        if ttl is None:
            ttl = self.data_ttl

        now = time.time()

        with self._lock:
            if (
                self._last_update_time <= 0.0
                or now - self._last_update_time > float(ttl)
            ):
                self._ships = []
                return []

            # 外层复制列表，避免调用者修改内部列表
            return list(self._ships)

    def is_connected(self):
        return self._connected

    def get_last_timestamp(self):
        with self._lock:
            return self._last_packet_timestamp

    def _clear_ships(self):
        with self._lock:
            self._ships = []
            self._last_update_time = 0.0
            self._last_packet_timestamp = 0.0

    def _close_socket(self):
        sock = self._sock
        self._sock = None

        if sock is None:
            return

        try:
            sock.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass

        try:
            sock.close()
        except OSError:
            pass

    def _wait_before_reconnect(self):

        if not self._running:
            return

        print(
            f"[Fusion] {self.reconnect_interval:.0f} 秒后重新尝试连接..."
        )

        self._stop_event.wait(
            timeout=self.reconnect_interval
        )

    def _recv_loop(self):
        while self._running:
            disconnected_reason = None

            try:
                print(
                    f"[Fusion] 正在连接 "
                    f"{self.host}:{self.port} ..."
                )

                sock = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                )

                # 减少小数据包发送延迟
                sock.setsockopt(
                    socket.IPPROTO_TCP,
                    socket.TCP_NODELAY,
                    1,
                )

                # 可选TCP保活
                sock.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_KEEPALIVE,
                    1,
                )

                sock.settimeout(self.connect_timeout)
                self._sock = sock

                sock.connect((self.host, self.port))

                if not self._running:
                    break

                self._connected = True

                print(
                    f"[Fusion] ✅ 已连接 "
                    f"{self.host}:{self.port}"
                )

                sock.settimeout(self.recv_timeout)

                buf = bytearray()

                while self._running:
                    try:
                        data = sock.recv(65536)

                        # recv返回空字节表示对端正常关闭
                        if not data:
                            disconnected_reason = "对端已关闭连接"
                            break

                        buf.extend(data)
                        buf = self._parse_buffer(buf)

                    except socket.timeout:
                        # 暂时没有数据不代表断线
                        continue

                    except ConnectionResetError:
                        disconnected_reason = "连接被对端重置"
                        break

                    except ConnectionAbortedError:
                        disconnected_reason = "连接被中止"
                        break

                    except OSError as e:
                        if self._running:
                            disconnected_reason = f"接收异常：{e}"
                        break

            except socket.timeout:
                if self._running:
                    disconnected_reason = (
                        f"连接超时，超过 "
                        f"{self.connect_timeout:.1f} 秒"
                    )

            except ConnectionRefusedError as e:
                if self._running:
                    disconnected_reason = f"对端拒绝连接：{e}"

            except OSError as e:
                if self._running:
                    disconnected_reason = f"连接失败：{e}"

            except Exception as e:
                if self._running:
                    disconnected_reason = f"未知异常：{e}"

            finally:
                self._connected = False
                self._close_socket()
                self._clear_ships()

            if not self._running:
                break

            if disconnected_reason:
                print(f"[Fusion] ⚠️ {disconnected_reason}")

            self._wait_before_reconnect()

    def _parse_buffer(self, buf):
        if not isinstance(buf, bytearray):
            buf = bytearray(buf)

        magic_bytes = struct.pack(">I", self.MAGIC)

        while len(buf) >= self.HEADER_SIZE:
            # 包头不以MAGIC开头，查找下一个MAGIC
            if buf[:4] != magic_bytes:
                idx = buf.find(magic_bytes, 1)

                if idx >= 0:
                    del buf[:idx]
                else:
                    # MAGIC是4字节，保留最后3字节，
                    # 防止MAGIC刚好跨两个recv数据块
                    if len(buf) > 3:
                        del buf[:-3]
                    break

                continue

            try:
                (
                    magic,
                    version,
                    msg_type,
                    payload_len,
                    ts_us,
                    seq,
                    reserved,
                ) = struct.unpack_from(
                    self.HEADER_FMT,
                    buf,
                    0,
                )

            except struct.error:
                break

            # payload_len是无符号整数，无需判断小于0
            if payload_len > self.max_payload_size:
                print(
                    f"[Fusion] ⚠️ 非法payload长度："
                    f"{payload_len} 字节，跳过当前MAGIC"
                )

                # 移除当前MAGIC，继续寻找后续正常包
                del buf[:4]
                continue

            total_size = self.HEADER_SIZE + payload_len

            # 当前缓存还不是完整包
            if len(buf) < total_size:
                break

            payload = bytes(
                buf[self.HEADER_SIZE:total_size]
            )

            # 从缓存移除当前完整包
            del buf[:total_size]

            try:
                packet = _json.loads(
                    payload.decode("utf-8")
                )

                packet_timestamp = float(packet["timestamp"])
                if not math.isfinite(packet_timestamp):
                    raise ValueError("invalid obstacle packet timestamp")

                obstacles = packet.get("obstacles", [])

                if not isinstance(obstacles, list):
                    print(
                        "[Fusion] ⚠️ obstacles不是列表，"
                        "本包已忽略"
                    )
                    continue

                ships = self._convert(obstacles, packet_timestamp)

                with self._lock:
                    self._ships = ships
                    self._last_update_time = time.time()
                    self._last_packet_timestamp = packet_timestamp

            except UnicodeDecodeError as e:
                print(f"[Fusion] UTF-8解析失败：{e}")

            except _json.JSONDecodeError as e:
                print(f"[Fusion] JSON解析失败：{e}")

            except Exception as e:
                print(f"[Fusion] 数据处理失败：{e}")

        return buf

    def _convert(self, obstacles, packet_timestamp=0.0):
        ships = []

        for obs in obstacles:
            if not isinstance(obs, dict):
                continue

            try:
                length = float(obs["length"])
                width = float(obs["width"])
                height = float(obs["height"])
                center_x = float(obs["center_x"])
                center_y = float(obs["center_y"])
                center_z = float(obs["center_z"])
                yaw = float(obs["yaw"])
                score = float(obs["score"])
                obstacle_timestamp = float(obs["timestamp"])

                x1 = int(obs.get("pixel_x1", -1))
                y1 = int(obs.get("pixel_y1", -1))
                x2 = int(obs.get("pixel_x2", -1))
                y2 = int(obs.get("pixel_y2", -1))

                if (
                    x1 < 0
                    or y1 < 0
                    or x2 <= x1
                    or y2 <= y1
                ):
                    continue

                north_vel = float(
                    obs.get("north_vel", 0.0)
                )
                east_vel = float(
                    obs.get("east_vel", 0.0)
                )

                speed = math.hypot(
                    north_vel,
                    east_vel,
                )

                bearing = (
                    math.degrees(
                        math.atan2(
                            east_vel,
                            north_vel,
                        )
                    )
                    % 360.0
                )

                distance = float(
                    obs.get("distance", 0.0)
                )

                bridge_pier_distance = float(
                    obs.get(
                        "bridge_pier_distance",
                        obs.get(
                            "pier_distance",
                            obs.get(
                                "bridge_distance",
                                -1.0,
                            ),
                        ),
                    )
                )

                class_id = int(
                    obs.get("class_id", 0)
                )

                confidence = float(
                    obs.get("confidence", 0.0)
                )

                numeric_values = (
                    length,
                    width,
                    height,
                    center_x,
                    center_y,
                    center_z,
                    yaw,
                    distance,
                    confidence,
                    score,
                    obstacle_timestamp,
                    north_vel,
                    east_vel,
                )
                if not all(math.isfinite(value) for value in numeric_values):
                    continue

            except (
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                # 当前目标字段异常，只跳过这一条
                continue

            # 距离无效时，不应被误判为最高危险等级
            if distance > 0:
                if distance < 30:
                    threat_level = 2
                elif distance < 80:
                    threat_level = 1
                else:
                    threat_level = 0
            else:
                threat_level = 0

            ships.append({
                "ship_id": len(ships),
                "label": f"Target-{class_id}",
                "length": length,
                "width": width,
                "height": height,
                "center_x": center_x,
                "center_y": center_y,
                "center_z": center_z,
                "yaw": yaw,
                "class_id": class_id,
                "confidence": confidence,
                "score": score,
                "timestamp": obstacle_timestamp,
                "packet_timestamp": float(packet_timestamp),
                "pixel_x1": x1,
                "pixel_y1": y1,
                "pixel_x2": x2,
                "pixel_y2": y2,
                "bbox": [x1, y1, x2, y2],
                "center": [
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                ],
                "conf": confidence,
                "speed": round(speed, 2),
                "bearing": round(bearing, 1),
                "distance": distance,
                "north_vel": north_vel,
                "east_vel": east_vel,
                "threat_level": threat_level,
                "hasSpeedBearing": True,
                "bridge_pier_distance": round(
                    bridge_pier_distance,
                    1,
                ),
            })

        return ships
