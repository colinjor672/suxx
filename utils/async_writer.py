import queue
from threading import Thread

import numpy as np


class AsyncVideoWriter:
    def __init__(self, video_writer, maxsize: int = 6):
        self.writer = video_writer
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.running = True
        self.dropped = 0
        self.processed = 0

        self.thread = Thread(
            target=self._run,
            daemon=True,
            name="AsyncWriter",
        )
        self.thread.start()

    def submit(self, frame: np.ndarray):
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while self.running:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            try:
                self.writer.write(item)
                self.processed += 1
            except Exception as e:
                print(f"⚠️ VideoWriter 错误: {e}")

    def shutdown(self):
        self.running = False

        try:
            self._q.put_nowait(None)
        except Exception:
            pass

        self.thread.join(timeout=5)

        print(
            f"  📊 写盘统计: "
            f"处理 {self.processed} 帧, 丢弃 {self.dropped} 帧"
        )