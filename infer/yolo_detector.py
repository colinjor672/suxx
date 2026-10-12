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
    ):
        self.model = model
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.class_names = class_names or ["ship"]
        self.detect_every = detect_every
        self.allowed_class_ids = allowed_class_ids
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
            f"(conf={self.conf}, iou={self.iou}, detect_every={self.detect_every})"
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
