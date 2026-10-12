"""把 yolo_sender 实际发出的检测框按帧记入日志文件。

只记录 class_ids 指定的类别（默认仅船只，不含桥墩）；
box_id 按框中心 x 从右到左递增（最右为 1），与报文里自带的 src_id 无关；
过滤后为空的帧不写行。
"""

import os


class YoloBoxLogger:
    HEADER = (
        "# frame_id  ts_unix          box_id  class  conf  src_id"
        "      x1       y1       x2       y2\n"
    )

    def __init__(self, path: str, class_ids=(0,)):
        self._path = str(path)
        self._class_ids = frozenset(int(c) for c in class_ids)
        self._fh = None
        self._disabled = False

    def log(self, frame_idx: int, timestamp: float, detections: list) -> None:
        if self._disabled or not detections:
            return

        try:
            detections = self._selected(detections)
            if not detections:
                return

            if self._fh is None:
                is_new = not os.path.exists(self._path)
                self._fh = open(self._path, "a", encoding="utf-8")
                if is_new:
                    self._fh.write(self.HEADER)

            for box_id, det in enumerate(self._right_to_left(detections), start=1):
                x1, y1, x2, y2 = det["bbox"]
                self._fh.write(
                    f"{int(frame_idx):8d}  {float(timestamp):14.3f}  {box_id:6d}"
                    f"  {int(det['class_id']):5d}  {float(det['score']):4.2f}"
                    f"  {int(det['id']):6d}"
                    f"  {x1:8.1f} {y1:8.1f} {x2:8.1f} {y2:8.1f}\n"
                )
            self._fh.flush()

        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"[YoloBoxLogger] 写入失败，已停用: {e}")
            self.close()
            self._disabled = True

    def _selected(self, detections: list) -> list:
        return [
            det for det in detections
            if int(det.get("class_id", -1)) in self._class_ids
        ]

    @staticmethod
    def _right_to_left(detections: list) -> list:
        return sorted(detections, key=lambda det: -(det["bbox"][0] + det["bbox"][2]))

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
