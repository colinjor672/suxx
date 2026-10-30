import threading
import time
from typing import Optional

import numpy as np
import torch

from utils.thread_utils import set_thread_name


class AsyncYOLODetector:
    def __init__(
        self,
        model,
        imgsz=768,
        conf=0.5,
        iou=0.45,
        device="cuda:0",
        class_names=None,
        detect_every=2,
        allowed_class_ids=None,
        on_result=None,
        track_iou=0.3,
        track_max_age=5,
        track_ema=0.45,
    ):
        self.model = model
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.class_names = class_names or ["ship"]
        self.detect_every = detect_every
        self.allowed_class_ids = allowed_class_ids

        # 轻量跟踪：用 IoU 关联前后帧，EMA 平滑框，丢检时短时保持
        self.track_iou = float(track_iou)
        self.track_max_age = int(track_max_age)
        self.track_ema = float(track_ema)
        self._tracks: list = []
        self._next_track_id: int = 0
        # 推理线程产出新结果时立刻回调 (ships, frame_id, timestamp)，
        # 用于绕开主循环节拍直接下发检测框；回调异常不得影响推理线程。
        self.on_result = on_result

        self._input_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id: int = -1
        self._latest_frame_timestamp: float = 0.0

        self._result_lock = threading.Lock()
        self._latest_result: list = []
        self._result_frame_id: int = -1

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._new_frame_event = threading.Event()

        self.infer_fps = 0.0
        self.frames_submitted = 0
        self.frames_processed = 0
        self.frames_skipped = 0

        self.start()

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._infer_loop,
            daemon=True,
            name="YOLO-Async",
        )
        self._thread.start()

        print(
            f"[AsyncYOLO] ✓ 后台推理线程已启动 "
            f"(conf={self.conf}, iou={self.iou}, detect_every={self.detect_every}, "
            f"track_iou={self.track_iou}, max_age={self.track_max_age}, "
            f"ema={self.track_ema})"
        )

    def shutdown(self):
        self.stop()

    def stop(self):
        self._running = False
        self._new_frame_event.set()

        if self._thread:
            self._thread.join(timeout=3.0)

        print(
            f"[AsyncYOLO] 已停止 | "
            f"处理={self.frames_processed} 跳过={self.frames_skipped}"
        )

    def submit(self, frame: np.ndarray, frame_id: int, timestamp: float = None):
        if frame_id % self.detect_every != 0:
            return

        frame_timestamp = time.time() if timestamp is None else float(timestamp)

        with self._input_lock:
            old_id = self._latest_frame_id
            # GStreamer 每帧返回新 buffer，无需 copy
            self._latest_frame = frame
            self._latest_frame_id = frame_id
            self._latest_frame_timestamp = frame_timestamp
            self.frames_submitted += 1

            if old_id != -1 and old_id != frame_id:
                self.frames_skipped += 1

        self._new_frame_event.set()

    def get_result(self) -> list:
        with self._result_lock:
            return self._latest_result

    def _notify_result(self, ships, frame_id, frame_timestamp):
        cb = self.on_result
        if cb is None:
            return
        try:
            cb(ships, frame_id, frame_timestamp)
        except Exception as e:
            print(f"[AsyncYOLO] on_result 回调异常: {e}")

    def _infer_loop(self):
        set_thread_name("AsyncYOLO")
        last_time = time.time()
        frame_count = 0

        while self._running:
            self._new_frame_event.wait(timeout=0.1)
            self._new_frame_event.clear()

            if not self._running:
                break

            with self._input_lock:
                frame = self._latest_frame
                frame_id = self._latest_frame_id
                frame_timestamp = self._latest_frame_timestamp
                self._latest_frame = None

            if frame is None:
                continue

            try:
                # torch.inference_mode(): 禁用 autograd 跟踪，减少 CPU 开销
                # 关闭 ultralytics 的 checks / logger / 每次调用的环境检查
                with torch.inference_mode():
                    results = self.model(
                        frame,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        device=self.device,
                        verbose=False,
                        classes=self.allowed_class_ids,
                    )

                ships = self._parse_to_ships(results, frame_id, frame_timestamp)

            except Exception as e:
                print(f"[AsyncYOLO] 推理异常: {e}")
                continue

            ships = remove_duplicate_boxes(ships, iou_thresh=0.4)
            ships = self._update_tracks(ships, frame_id, frame_timestamp)

            with self._result_lock:
                self._latest_result = ships
                self._result_frame_id = frame_id

            self._notify_result(ships, frame_id, frame_timestamp)

            self.frames_processed += 1
            frame_count += 1

            now = time.time()
            elapsed = now - last_time

            if elapsed >= 1.0:
                self.infer_fps = frame_count / elapsed
                frame_count = 0
                last_time = now

    def _parse_to_ships(
        self,
        results,
        frame_id: int,
        frame_timestamp: float,
    ) -> list:
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes

        # ★ 合并 3 次 .cpu() 为 1 次（减少 GPU→CPU 同步）
        # xyxy: [N,4], cls: [N], conf: [N] → 拼成 [N,6] 一次性下载
        combined = torch.cat(
            [
                boxes.xyxy,
                boxes.cls.unsqueeze(1),
                boxes.conf.unsqueeze(1),
            ],
            dim=1,
        ).detach().cpu().numpy()

        ships = []

        for row in combined:
            x1, y1, x2, y2, cls_id, conf = row
            cls_id = int(cls_id)

            if self.allowed_class_ids is not None and cls_id not in self.allowed_class_ids:
                continue

            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

            if cls_id < len(self.class_names):
                label = self.class_names[cls_id]
            else:
                label = result.names.get(cls_id, f"class_{cls_id}")

            ships.append(
                {
                    "ship_id": len(ships),
                    "class_id": cls_id,
                    "label": label,
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
                    "conf": float(conf),
                    "speed": 0.0,
                    "bearing": 0.0,
                    "distance": 0.0,
                    "threat_level": 0,
                    "hasSpeedBearing": False,
                    "source_frame_id": frame_id,
                    "timestamp": float(frame_timestamp),
                }
            )

        return ships

    @staticmethod
    def _bbox_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = ix2 - ix1, iy2 - iy1
        if iw <= 0.0 or ih <= 0.0:
            return 0.0
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    def _update_tracks(self, detections, frame_id: int,
                       frame_timestamp: float) -> list:
        """IoU 关联前后帧 → EMA 平滑框 → 丢检时保持 max_age 次推理。

        ship_id 使用稳定的 track_id，不再按列表下标重排。
        输出字典每次重建，避免下游（merge_ship_data / DCPA 吸附）改坏跟踪状态。
        """
        pairs = []
        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(detections):
                if det["class_id"] != track["class_id"]:
                    continue
                iou = self._bbox_iou(track["bbox"], det["bbox"])
                if iou >= self.track_iou:
                    pairs.append((iou, ti, di))
        # 全局从高 IoU 开始贪心匹配，保证一对一
        pairs.sort(reverse=True)

        matched_tracks = set()
        matched_dets = set()
        alpha = self.track_ema

        for _iou, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)

            track = self._tracks[ti]
            det = detections[di]
            track["bbox"] = [
                alpha * cur + (1.0 - alpha) * prev
                for prev, cur in zip(track["bbox"], det["bbox"])
            ]
            track["conf"] = det["conf"]
            track["class_id"] = det["class_id"]
            track["label"] = det["label"]
            track["last_frame_id"] = frame_id
            track["timestamp"] = frame_timestamp

        # 未匹配的检测 → 新建 track
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            self._tracks.append({
                "track_id": self._next_track_id,
                "bbox": list(det["bbox"]),
                "conf": det["conf"],
                "class_id": det["class_id"],
                "label": det["label"],
                "last_frame_id": frame_id,
                "timestamp": frame_timestamp,
                "missing": 0,
            })
            self._next_track_id += 1

        # 未在本帧命中的 track 老化；连续 max_age 次推理未命中才删除
        for track in self._tracks:
            if track["last_frame_id"] != frame_id:
                track["missing"] += 1
        self._tracks = [
            t for t in self._tracks if t["missing"] <= self.track_max_age
        ]

        # 输出（包含仍在保持期内的 track）
        ships = []
        for track in self._tracks:
            x1, y1, x2, y2 = track["bbox"]
            ships.append({
                "ship_id": track["track_id"],
                "class_id": track["class_id"],
                "label": track["label"],
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
                "conf": float(track["conf"]),
                "speed": 0.0,
                "bearing": 0.0,
                "distance": 0.0,
                "threat_level": 0,
                "hasSpeedBearing": False,
                "source_frame_id": track["last_frame_id"],
                "timestamp": float(track["timestamp"]),
            })
        return ships

    def __repr__(self):
        return (
            f"AsyncYOLODetector(infer_fps={self.infer_fps:.1f}, "
            f"processed={self.frames_processed}, "
            f"skipped={self.frames_skipped})"
        )


def remove_duplicate_boxes(ships, iou_thresh=0.4):
    if len(ships) <= 1:
        return ships

    ships_sorted = sorted(ships, key=lambda s: s["conf"], reverse=True)
    keep = []

    for s in ships_sorted:
        is_dup = any(calc_iou(s["bbox"], k["bbox"]) > iou_thresh for k in keep)
        if not is_dup:
            keep.append(s)

    for i, s in enumerate(keep):
        s["ship_id"] = i

    return keep


def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = a1 + a2 - inter

    return inter / union if union > 0 else 0
