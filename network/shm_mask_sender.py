import numpy as np

from network.shm_sync import ShmSyncPy


class ShmMaskSender:
    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self._data_size = self.width * self.height * 2
        self._packed = np.empty((self.height, self.width, 2), dtype=np.uint8)
        self._shm = None
        self._running = False
        self._frame_count = 0
        self._error_count = 0

    def start(self) -> bool:
        try:
            self._shm = ShmSyncPy("shm_mask", self._data_size, is_writer=True)
            self._running = True
            print(
                f"[ShmMaskSender] mmap+semaphore 已启动 → "
                f"{self.width}×{self.height} RG8",
                flush=True,
            )
            return True
        except Exception as e:
            print(f"[ShmMaskSender] 启动失败: {e}", flush=True)
            self._shm = None
            return False

    def send(self, water_mask: np.ndarray, bridge_mask: np.ndarray, frame_id: int):
        if not self._running:
            return

        try:
            expected_shape = (self.height, self.width)
            if water_mask.shape != expected_shape or bridge_mask.shape != expected_shape:
                raise ValueError(
                    f"掩码尺寸不一致: water={water_mask.shape}, "
                    f"bridge={bridge_mask.shape}, expected={expected_shape}"
                )

            np.not_equal(water_mask, 0, out=self._packed[:, :, 0])
            np.not_equal(bridge_mask, 0, out=self._packed[:, :, 1])
            self._packed *= 255

            self._shm.set_frame_id(int(frame_id) & 0xFFFFFFFF)
            self._shm.write_raw(memoryview(self._packed))
            self._shm.signal_new_frame()

            self._frame_count += 1
            if self._frame_count == 1 or self._frame_count % 100 == 0:
                print(
                    f"[ShmMaskSender] 已写 {self._frame_count} 张掩码 "
                    f"frame_id={frame_id}",
                    flush=True,
                )
        except Exception as e:
            self._error_count += 1
            if self._error_count <= 3:
                print(f"[ShmMaskSender] send() 异常: {e}", flush=True)

    def stop(self):
        self._running = False
        if self._shm:
            self._shm.close()
            self._shm = None
        print(
            f"[ShmMaskSender] 已停止 (共 {self._frame_count} 张, "
            f"错误 {self._error_count})",
            flush=True,
        )
