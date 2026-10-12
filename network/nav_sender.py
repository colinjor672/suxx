import time
import threading

import numpy as np

from data_server import prepare_frame_data
from utils.geometry import extract_polygons_json, scale_ships_for_send
from utils.thread_utils import set_thread_name
class AsyncNavDataSender:
    def __init__(self, nav_server, mask_send_scale: float = 0.5,
             use_polygon_encoding: bool = True,
             polygon_epsilon: float = 0.005):
        self.nav_server      = nav_server
        self.mask_send_scale = float(max(0.25, min(1.0, mask_send_scale)))
        self.use_polygon_encoding = use_polygon_encoding
        self.polygon_epsilon      = polygon_epsilon
        self._poly_bytes_last     = 0
        self._lock   = threading.Lock()
        self._slot   = None
        self._event  = threading.Event()
        self.running = False
        self.dropped = 0
        self.sent    = 0

        self._send_method = self._find_send_method()

        self._thread = threading.Thread(
            target=self._run,
            name="AsyncNavSender",
            daemon=True
        )
        self.running = True
        self._thread.start()

    def _find_send_method(self):
        candidates = [
            'send_prepared_nav',
            'send_nav_data',
            'send_frame_data',
            'broadcast_nav',
            'send',
        ]
        for name in candidates:
            m = getattr(self.nav_server, name, None)
            if m is not None and callable(m):
                print(f"[AsyncNavSender] ✓ 使用发送方法: nav_server.{name}()")
                return m
        available = [
            attr for attr in dir(self.nav_server)
            if not attr.startswith('_') and callable(getattr(self.nav_server, attr))
        ]
        print(f"[AsyncNavSender] ✗ 未找到任何发送方法！可用方法: {available}")
        return None

    def submit(self, width, height, curve, frame_id, timestamp,
        water_mask, bridge_mask, ships_data,
        ship_src_w=None, ship_src_h=None,
        video_time: float = 0.0,
        sync_mode: str = "file_time",
        source_fps: float = 30.0,
        mask_frame_id: int = -1,
        mask_video_time: float = 0.0,
        bridge_warning: dict = None):
        if not self.running:
            return
        ship_src_w = int(ship_src_w) if ship_src_w is not None else int(width)
        ship_src_h = int(ship_src_h) if ship_src_h is not None else int(height)

        item = (
            width, height,
            curve.copy() if (curve is not None and len(curve) > 0) else curve,
            frame_id, timestamp,
            water_mask,
            bridge_mask,
            list(ships_data) if ships_data else [],
            ship_src_w,
            ship_src_h,
            float(video_time),
            str(sync_mode),
            float(source_fps),
            int(mask_frame_id),
            float(mask_video_time),
            bridge_warning,
        )
        with self._lock:
            if self._slot is not None:
                self.dropped += 1
            self._slot = item
        self._event.set()

    def shutdown(self, timeout: float = 2.0):
        self.running = False
        self._event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            print("[AsyncNavSender] ⚠️ 后台线程未在超时内退出")
        else:
            print(f"[AsyncNavSender] ✓ 已停止 | 发送={self.sent} 丢弃={self.dropped}")
        if self.use_polygon_encoding and self._poly_bytes_last > 0:
            ratio = 460800 // max(1, self._poly_bytes_last)
            print(f"  多边形末帧大小: {self._poly_bytes_last}B  压缩率≈{ratio}x")
    
    def _run(self):
        set_thread_name("NavSender")
        while self.running:
            triggered = self._event.wait(timeout=0.05)
            self._event.clear()
            if not self.running:
                break

            with self._lock:
                item       = self._slot
                self._slot = None
            if item is None:
                continue

            (
                width, height,
                curve, frame_id, timestamp,
                water_mask, bridge_mask, ships_data,
                ship_src_w, ship_src_h,
                video_time, sync_mode, source_fps,
                mask_frame_id, mask_video_time,
                bridge_warning,
            ) = item

            # ★ mask 已在 AsyncSeg 中 resize 到 send 尺寸，直接使用，无需 cuda_resize
            # ★ 下游 extract_polygons_json 不修改原数组，无需 .copy()
            wm = water_mask
            bm = bridge_mask
            send_w, send_h = width, height
            if ships_data:
                sx_ship = send_w / float(ship_src_w)
                sy_ship = send_h / float(ship_src_h)

                ships_data = scale_ships_for_send(
                    ships_data,
                    sx_ship,
                    sy_ship,
                    dst_w=send_w,
                    dst_h=send_h,
                )

                
            try:
                if self._send_method is None:
                    self.dropped += 1
                    continue

                if self.use_polygon_encoding:
                    w_polys, b_polys = extract_polygons_json(
                        wm, bm,
                        epsilon_ratio = self.polygon_epsilon,
                        min_area_px   = max(100, int(send_w * send_h * 0.001)),
                    )

                    self._poly_bytes_last = (
                        sum(len(p) for p in w_polys) +
                        sum(len(p) for p in b_polys)
                    ) * 12

                    empty = np.zeros((1, 1), dtype=np.uint8)
                    msg = prepare_frame_data(
                        width       = send_w,
                        height      = send_h,
                        curve       = curve,
                        frame_id    = frame_id,
                        timestamp   = timestamp,
                        water_mask  = empty,
                        bridge_mask = empty,
                        ships_data  = ships_data,
                    )
                    msg["source_type"] = "file"
                    msg["sync_mode"] = sync_mode

                    # Godot 使用该字段把整包数据对齐到对应的视频帧。
                    msg["video_time"] = float(mask_video_time)

                    msg["send_video_time"] = float(video_time)
                    msg["source_fps"] = float(source_fps)

                    msg["mask_frame_id"] = int(mask_frame_id)
                    msg["mask_video_time"] = float(mask_video_time)

                    msg["send_frame_id"] = int(frame_id)
                    msg["sent_wall_time"] = time.time()
                    msg['water_polygons']  = w_polys
                    msg['bridge_polygons'] = b_polys
                    msg['has_polygons']    = True

                else:
                    msg = prepare_frame_data(
                        width       = send_w,
                        height      = send_h,
                        curve       = curve,
                        frame_id    = frame_id,
                        timestamp   = timestamp,
                        water_mask  = wm,
                        bridge_mask = bm,
                        ships_data  = ships_data,
                    )

                # 桥梁警戒（警戒线/信息面板像素坐标）：无目标桥时省略该字段
                if bridge_warning:
                    msg["bridge_warning"] = bridge_warning

                try:
                    self._send_method(msg, frame_id)
                except TypeError:
                    self._send_method(msg)
                self.sent += 1

            except OSError as e:
                print(f"[AsyncNavSender] 网络异常(丢帧): {e}")
                self.dropped += 1
            except Exception as e:
                import traceback
                print(f"[AsyncNavSender] ✗ 未知异常 (帧{frame_id}): {e}")
                traceback.print_exc()
                self.dropped += 1
