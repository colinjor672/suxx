import socket
import threading
import struct
import json
import time
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import json as _j
import queue
from concurrent.futures import ThreadPoolExecutor


#  工具函数

def mask_to_polygons(mask, min_area=500):
    """二值 mask → 多边形列表 [[[x,y], ...], ...]"""
    if mask is None:
        return []
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        eps = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        poly = approx.reshape(-1, 2).tolist()
        if len(poly) >= 3:
            polygons.append(poly)
    return polygons


def prepare_frame_data(width, height, curve, frame_id, timestamp,
                       water_mask=None, bridge_mask=None, ships_data=None,
                       coord_w=None, coord_h=None):
    """
    组装发送给 Unity 的导航数据。

    width / height:
        当前 mask、curve、polygon 的发送坐标空间。

    coord_w / coord_h:
        YOLO bbox 的坐标空间。
        如果 YOLO bbox 已经缩放到 width×height，则 coord_w=width, coord_h=height。
    """

    width = int(width)
    height = int(height)

    if coord_w is None:
        coord_w = width
    if coord_h is None:
        coord_h = height

    coord_w = int(coord_w)
    coord_h = int(coord_h)

    # ── 曲线 ──
    if isinstance(curve, np.ndarray):
        curve_points = curve.tolist()
    elif curve is None:
        curve_points = []
    else:
        curve_points = curve

    # ── 水面掩码 / 多边形 ──
    if isinstance(water_mask, np.ndarray) and water_mask.ndim == 2:
        water_polygons = mask_to_polygons(water_mask)
    elif isinstance(water_mask, list):
        water_polygons = water_mask
    else:
        water_polygons = []

    # ── 桥梁掩码 / 多边形 ──
    if isinstance(bridge_mask, np.ndarray) and bridge_mask.ndim == 2:
        bridge_polygons = mask_to_polygons(bridge_mask)
    elif isinstance(bridge_mask, list):
        bridge_polygons = bridge_mask
    else:
        bridge_polygons = []

    # ── 船只数据 ──
    ships = []

    if ships_data:
        for s in ships_data:
            if not s:
                continue

            bbox = s.get("bbox", [0, 0, 0, 0])
            center = s.get("center", None)

            if center is None or len(center) < 2:
                center = [
                    (float(bbox[0]) + float(bbox[2])) * 0.5,
                    (float(bbox[1]) + float(bbox[3])) * 0.5,
                ]

            ship_msg = {
                "ship_id": int(s.get("ship_id", 0)),
                "label": str(s.get("label", "ship")),

                # bbox 保持 float，避免小目标被 int 截断
                "bbox": [float(v) for v in bbox],
                "center": [float(v) for v in center],

                "conf": float(s.get("conf", s.get("confidence", 0.0))),
                "speed": float(s.get("speed", 0.0)),
                "bearing": float(s.get("bearing", 0.0)),
                "distance": float(s.get("distance", 0.0)),
                "north_vel": float(s.get("north_vel", 0.0)),
                "east_vel": float(s.get("east_vel", 0.0)),
                "yaw": float(s.get("yaw", 0.0)),
                "threat_level": int(s.get("threat_level", 0)),
                "has_fusion_data": bool(s.get("has_fusion_data", False)),

                "hasSpeedBearing": bool(
                    s.get("hasSpeedBearing", False)
                    or abs(float(s.get("speed", 0.0))) > 0.001
                    or abs(float(s.get("bearing", 0.0))) > 0.001
                ),
            }

            if "bridge_pier_distance" in s:
                ship_msg["bridge_pier_distance"] = float(
                    s.get("bridge_pier_distance", -1.0)
                )

            # 避障指定的单一目标：只有这一个框在 Godot 端显示信息面板
            if s.get("is_dcpa_target"):
                ship_msg["is_dcpa_target"] = True
                ship_msg["dcpa"] = float(s.get("dcpa", 0.0))
                ship_msg["tcpa"] = float(s.get("tcpa", 0.0))
                ship_msg["dcpa_alert_level"] = str(s.get("dcpa_alert_level", ""))
                ship_msg["dcpa_alert_info"] = str(s.get("dcpa_alert_info", ""))

            ships.append(ship_msg)

    msg = {
        "type": "nav_data",
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),

        # mask / curve / polygon 坐标空间
        "width": width,
        "height": height,

        # YOLO bbox 坐标空间
        "coord_w": coord_w,
        "coord_h": coord_h,
        "curve": curve_points,
        "curves": curve_points,
        "water_polygons": water_polygons,
        "bridge_polygons": bridge_polygons,
        "ships": ships,
    }

    return msg


def prepare_video_frame(frame, frame_id, width, height, jpeg_quality=85):
    """将一帧编码为 JPEG 并包装成 dict"""
    ret, jpeg = cv2.imencode('.jpg', frame,
                             [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ret:
        return None
    return {
        'type': 'video_frame',
        'frame_id': frame_id,
        'width': width,
        'height': height,
        'timestamp': time.time(),
        'jpeg_data': jpeg.tobytes()
    }

#  双端口 TCP 服务器

class NavigationDataServer:
    """
    端口 8765 ── 导航数据（曲线 / 水面 / 桥梁 / 船只信息）
    端口 8766 ── 同帧视频流（JPEG）
    """
    # 发送超时（秒）：超过此时间未发完则丢弃该帧
    SEND_TIMEOUT = 0.05

    def __init__(self, host="0.0.0.0", port=8765, video_port=8766,
                 target_fps=25, jpeg_quality=85):
        self.host = host
        self.port = port                 # 导航数据端口
        self.video_port = video_port     # 视频帧端口
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality

        # 导航数据 TCP
        self.nav_socket: Optional[socket.socket] = None
        self.nav_clients: List[socket.socket] = []
        self._nav_lock = threading.Lock()

        # 导航数据异步发送槽位：只保留最新一帧，旧帧自动替换
        self._nav_slot_lock = threading.Lock()
        self._nav_send_event = threading.Event()
        self._nav_slot = None
        self._nav_replaced = 0
        self._nav_worker_thread = None

        #视频帧 TCP
        self.video_socket: Optional[socket.socket] = None
        self.video_clients: List[socket.socket] = []
        self._video_lock = threading.Lock()

        # 视频帧异步发送槽位：只保留最新一帧
        self._video_slot_lock = threading.Lock()
        self._video_send_event = threading.Event()
        self._video_slot = None
        self._video_replaced = 0
        self._video_worker_thread = None

        self._encode_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Encode")
        # ── 统计 ──
        self._nav_send_drops = 0
        self._video_send_drops = 0

        self.is_running = False

    # 生命周期

    def start(self):
        self.is_running = True

        # 导航数据服务器
        self.nav_socket = self._create_server(self.port)
        threading.Thread(target=self._accept_loop,
                         args=(self.nav_socket, self.nav_clients,
                               self._nav_lock, "导航数据"),
                         daemon=True).start()

        # 视频帧服务器
        self.video_socket = self._create_server(self.video_port)
        threading.Thread(target=self._accept_loop,
                        args=(self.video_socket, self.video_clients,
                            self._video_lock, "视频帧"),
                        daemon=True).start()

        # 启动异步发送线程
        self._nav_worker_thread = threading.Thread(
            target=self._send_worker,
            args=("nav",),
            daemon=True,
            name="NavSendWorker"
        )
        self._nav_worker_thread.start()

        self._video_worker_thread = threading.Thread(
            target=self._send_worker,
            args=("video",),
            daemon=True,
            name="VideoSendWorker"
        )
        self._video_worker_thread.start()

        print(f"[Server] ✓ 导航数据端口     : {self.host}:{self.port}")
        print(f"[Server] ✓ 视频帧端口       : {self.host}:{self.video_port}")

    def stop(self):
        self.is_running = False

        for lock, clients in [(self._nav_lock, self.nav_clients),
                              (self._video_lock, self.video_clients)]:
            with lock:
                for c in clients:
                    try:
                        c.close()
                    except:
                        pass
                clients.clear()

        for sock in [self.nav_socket, self.video_socket]:
            if sock:
                try:
                    sock.close()
                except:
                    pass

        if self._nav_send_drops > 0 or self._video_send_drops > 0:
            print(f"[Server] 发送统计: 导航丢弃={self._nav_send_drops}, "
                  f"视频丢弃={self._video_send_drops}")
        print("[Server] 已停止")

    #公共发送接口

    def get_ships(self) -> list:
        """获取船只信息（已移除 ar_tcp_server 依赖，始终返回空列表）"""
        return []

    def send_nav_data(self, frame_id: int, width: int, height: int,
                  curve_points=None, water_polygons=None,
                  bridge_polygons=None, ships=None,
                  coord_w=None, coord_h=None):
       
        if ships is None:
            ships = self.get_ships()

        if coord_w is None:
            coord_w = width
        if coord_h is None:
            coord_h = height

        #曲线 JsonUtility 兼容格式
        curves_json = []
        for pt in (curve_points or []):
            if isinstance(pt, dict):
                curves_json.append({
                    "x": float(pt.get("x", 0)),
                    "y": float(pt.get("y", 0)),
                })
            else:
                curves_json.append({
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                })

        # 多边形 JsonUtility 兼容格式
        def convert_polygons(polygons):
            result = []
            for poly in (polygons or []):
                # 兼容已经是 {"points": [...]} 的格式
                if isinstance(poly, dict) and "points" in poly:
                    pts_src = poly["points"]
                else:
                    pts_src = poly

                pts = []
                for p in pts_src:
                    if isinstance(p, dict):
                        pts.append({
                            "x": float(p.get("x", 0)),
                            "y": float(p.get("y", 0)),
                        })
                    else:
                        pts.append({
                            "x": float(p[0]),
                            "y": float(p[1]),
                        })

                if len(pts) >= 3:
                    result.append({"points": pts})
            return result

        water_json = convert_polygons(water_polygons)
        bridge_json = convert_polygons(bridge_polygons)

        # 船只数据：只做 JSON 安全转换，不在这里缩放
        ships_json = []
        for s in (ships or []):
            bbox = s.get("bbox", [0, 0, 0, 0])
            center = s.get("center", [
                (float(bbox[0]) + float(bbox[2])) * 0.5,
                (float(bbox[1]) + float(bbox[3])) * 0.5,
            ])

            ship_msg = {
                "ship_id": int(s.get("ship_id", 0)),
                "label": str(s.get("label", "ship")),
                "bbox": [float(v) for v in bbox],
                "center": [float(v) for v in center],
                "conf": float(s.get("conf", s.get("confidence", 0.0))),
                "speed": float(s.get("speed", 0.0)),
                "bearing": float(s.get("bearing", 0.0)),
                "distance": float(s.get("distance", 0.0)),
                "north_vel": float(s.get("north_vel", 0.0)),
                "east_vel": float(s.get("east_vel", 0.0)),
                "yaw": float(s.get("yaw", 0.0)),
                "threat_level": int(s.get("threat_level", 0)),
                "has_fusion_data": bool(s.get("has_fusion_data", False)),
            }

            # 可选字段，godot如果有对应字段就能解析，没有也不影响
            if "hasSpeedBearing" in s:
                ship_msg["hasSpeedBearing"] = bool(s.get("hasSpeedBearing"))
            if "bridge_pier_distance" in s:
                ship_msg["bridge_pier_distance"] = float(s.get("bridge_pier_distance", -1.0))
            ships_json.append(ship_msg)

        json_obj = {
            "type": "nav_data",
            "frame_id": int(frame_id),
            "timestamp": time.time(),
            "width": int(width),
            "height": int(height),
            "coord_w": int(coord_w),
            "coord_h": int(coord_h),
            "curves": curves_json,
            "water_polygons": water_json,
            "bridge_polygons": bridge_json,
            "ships": ships_json,
        }

        self.send_prepared_nav(json_obj, frame_id)

    def send_video_frame(self, frame, frame_id: int, width: int = None, 
                         height: int = None, jpeg_quality: int = None):
        """异步 JPEG 编码"""
        if frame is None:
            return
        if width is None or height is None:
            height, width = frame.shape[:2]
        quality = jpeg_quality if jpeg_quality is not None else self.jpeg_quality

        #  提交到线程池异步编码
        self._encode_pool.submit(
            self._encode_and_send_video,
            frame.copy(),  # 必须 copy，避免原帧被修改
            frame_id, width, height, quality
        )

    def _encode_and_send_video(self, frame, frame_id, width, height, quality):
        """在线程池中执行编码 + 发送"""
        ret, jpeg = cv2.imencode('.jpg', frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ret:
            return
        
        jpeg_bytes = jpeg.tobytes()
        payload = struct.pack('<iiiI', frame_id, width, height,
                              len(jpeg_bytes)) + jpeg_bytes
        message = struct.pack('<I', len(payload)) + payload
        self._enqueue_video_message(message)

    def send_prepared_nav(self, nav_dict: dict, frame_id: int = 0):
        if not isinstance(nav_dict, dict):
            return

        # 没有 Godot 导航客户端时，不做 JSON 序列化
        if not self._has_clients(self.nav_clients, self._nav_lock):
            return

        fid = int(nav_dict.get("frame_id", frame_id))
        w = int(nav_dict.get("width", 1920))
        h = int(nav_dict.get("height", 1080))

        if "coord_w" not in nav_dict:
            nav_dict["coord_w"] = w
        if "coord_h" not in nav_dict:
            nav_dict["coord_h"] = h

        nav_dict["type"] = nav_dict.get("type", "nav_data")
        nav_dict["frame_id"] = fid
        nav_dict["width"] = w
        nav_dict["height"] = h

        # 同步字段
        if "video_time" not in nav_dict:
            nav_dict["video_time"] = float(nav_dict.get("timestamp", 0.0))
        if "mask_video_time" not in nav_dict:
            nav_dict["mask_video_time"] = float(nav_dict.get("video_time", 0.0))
        if "sync_mode" not in nav_dict:
            nav_dict["sync_mode"] = "file_time"
        if "source_type" not in nav_dict:
            nav_dict["source_type"] = "file"

        # 关键：不再把 [[x,y]] 转成 {"x":x,"y":y}
        # 你的 Godot 端 NavigationBand 和 MaskLayer 已经兼容 Array 点格式。
        json_bytes = json.dumps(
            nav_dict,
            ensure_ascii=False,
            separators=(",", ":")
        ).encode("utf-8")

        payload = struct.pack(
            "<iiiI",
            fid,
            w,
            h,
            len(json_bytes)
        ) + json_bytes

        message = struct.pack("<I", len(payload)) + payload

        self._enqueue_nav_message(message)

    @property
    def nav_client_count(self):
        with self._nav_lock:
            return len(self.nav_clients)

    @property
    def video_client_count(self):
        with self._video_lock:
            return len(self.video_clients)

    # 内部方法
    def _has_clients(self, client_list: list, lock: threading.Lock) -> bool:
        with lock:
            return len(client_list) > 0


    def _enqueue_nav_message(self, message: bytes):
        if not self._has_clients(self.nav_clients, self._nav_lock):
            return

        with self._nav_slot_lock:
            if self._nav_slot is not None:
                self._nav_replaced += 1
            self._nav_slot = message

        self._nav_send_event.set()


    def _enqueue_video_message(self, message: bytes):
        if not self._has_clients(self.video_clients, self._video_lock):
            return

        with self._video_slot_lock:
            if self._video_slot is not None:
                self._video_replaced += 1
            self._video_slot = message

        self._video_send_event.set()


    def _send_worker(self, kind: str):
        if kind == "nav":
            event = self._nav_send_event
            slot_lock = self._nav_slot_lock
            clients = self.nav_clients
            clients_lock = self._nav_lock
        else:
            event = self._video_send_event
            slot_lock = self._video_slot_lock
            clients = self.video_clients
            clients_lock = self._video_lock

        while self.is_running:
            event.wait(timeout=0.05)
            event.clear()

            if not self.is_running:
                break

            with slot_lock:
                if kind == "nav":
                    message = self._nav_slot
                    self._nav_slot = None
                else:
                    message = self._video_slot
                    self._video_slot = None

            if message is None:
                continue

            drops = self._broadcast(
                message,
                clients,
                clients_lock
            )

            if kind == "nav":
                self._nav_send_drops += drops
            else:
                self._video_send_drops += drops
    def _create_server(self, port) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, port))
        s.listen(5)
        s.settimeout(1.0)
        return s

    def _accept_loop(self, server_socket, client_list, lock, name):
        while self.is_running:
            try:
                client, addr = server_socket.accept()
                # 关键优化：TCP_NODELAY + 大发送缓冲区 
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512 * 1024)
                # 不设全局 timeout，由 _safe_send 单独控制
                with lock:
                    client_list.append(client)
                print(f"[Server] Unity [{name}] 客户端已连接: {addr}")
            except socket.timeout:
                continue
            except OSError:
                break

    
    def send_jpeg_video_frame(self, jpeg_bytes: bytes, frame_id: int,
                            width: int, height: int):
        if not jpeg_bytes:
            return

        # 没有视频客户端时，不组包
        if not self._has_clients(self.video_clients, self._video_lock):
            return

        data_len = len(jpeg_bytes)

        payload = (
            struct.pack("<iiiI", int(frame_id), int(width), int(height), data_len)
            + jpeg_bytes
        )

        message = struct.pack("<I", len(payload)) + payload

        self._enqueue_video_message(message)

    def send_raw_video_frame(self, raw_bytes: bytes, frame_id: int,
                            width: int, height: int):
        if not raw_bytes:
            return

        if not self._has_clients(self.video_clients, self._video_lock):
            return

        data_len = len(raw_bytes)

        payload = (
            struct.pack("<iiiI", int(frame_id), int(width), int(height), data_len)
            + raw_bytes
        )

        message = struct.pack("<I", len(payload)) + payload

        self._enqueue_video_message(message)

    def _safe_send(self, client: socket.socket, data: bytes):
        try:
            client.settimeout(self.SEND_TIMEOUT)
            client.sendall(data)
            return True

        except socket.timeout:
            return False

        except BlockingIOError:
            return False

        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            OSError,
        ):
            return None


    def _broadcast(
        self,
        data: bytes,
        client_list: list,
        lock: threading.Lock,
    ) -> int:
        # 先复制客户端列表，不要拿着锁 sendall
        with lock:
            clients = list(client_list)

        if not clients:
            return 0

        dead = []
        drops = 0

        for c in clients:
            result = self._safe_send(c, data)

            if result is None:
                dead.append(c)
            elif result is False:
                drops += 1

        if dead:
            with lock:
                for d in dead:
                    if d in client_list:
                        client_list.remove(d)

                    try:
                        d.close()
                    except Exception:
                        pass

                print(
                    f"[Server] 客户端断开，已移除 "
                    f"(剩余 {len(client_list)})"
                )

        return drops
