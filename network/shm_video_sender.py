#!/usr/bin/env python3
"""
共享内存视频发送器（Python → Godot）
使用 ShmSyncPy 的 mmap + semaphore，替换文件 I/O

单路 RTSP 拉流，Python 端分叉：
  GstTeeCapture → RGBA 640×360 帧
    ├──→ AI 推理（BGR）
    └──→ ShmVideoSender.send() → mmap → Godot

协议：
  - 共享内存: /shm_input
  - 信号量:   /shm_input_sem  (Python post, Godot wait)
  - 数据格式: [4字节 frame_id:u32 LE][RGBA 640×360×4]
"""

import threading
import numpy as np

from network.shm_sync import ShmSyncPy

INPUT_W = 640
INPUT_H = 360
INPUT_SIZE = INPUT_W * INPUT_H * 4  # RGBA


class ShmVideoSender:
    """
    接收 RGBA 帧 → mmap 直接写入 → semaphore 通知 Godot

    用法:
        sender = ShmVideoSender()
        sender.start()
        while True:
            ret, bgr_frame, rgba_frame = cap.read()
            sender.send(rgba_frame)
        sender.stop()
    """

    def __init__(self):
        self._shm: ShmSyncPy = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = False
        self._frame_count = 0
        self._error_count = 0

    def start(self) -> bool:
        try:
            self._shm = ShmSyncPy("shm_input", INPUT_SIZE, is_writer=True)
            self._running = True
            print(f"[ShmVideoSender] mmap+semaphore 已启动 → {INPUT_W}×{INPUT_H} RGBA", flush=True)
            return True
        except Exception as e:
            print(f"[ShmVideoSender] 启动失败: {e}", flush=True)
            return False

    def send(self, rgba_frame: np.ndarray):
        """接收 RGBA 640×360 帧（nvvidconv GPU 已转换）→ mmap + semaphore"""
        if not self._running or rgba_frame is None:
            if self._frame_count == 0:
                print(f"[ShmVideoSender] send() 被跳过 running={self._running} "
                      f"frame_none={rgba_frame is None}", flush=True)
            return

        if self._frame_count == 0:
            print(f"[ShmVideoSender] send() 首帧 shape={rgba_frame.shape} "
                  f"dtype={rgba_frame.dtype} contiguous={rgba_frame.flags['C_CONTIGUOUS']}",
                  flush=True)

        try:
            with self._lock:
                self._frame_id += 1
                fid = self._frame_id

            # 确保数组连续
            if not rgba_frame.flags["C_CONTIGUOUS"]:
                rgba_frame = np.ascontiguousarray(rgba_frame)

            # 写入 mmap（memoryview 零拷贝：numpy → mmap 直写）
            self._shm.set_frame_id(fid)
            self._shm.write_raw(memoryview(rgba_frame))

            # 通知 Godot 新帧就绪
            self._shm.signal_new_frame()

            self._frame_count += 1
            if self._frame_count % 100 == 0:
                print(f"[ShmVideoSender] 已写 {self._frame_count} 帧 (mmap+semaphore)", flush=True)

        except Exception as e:
            self._error_count += 1
            if self._error_count <= 3:
                print(f"[ShmVideoSender] send() 异常: {e}", flush=True)
                import traceback
                traceback.print_exc()

    def stop(self):
        self._running = False
        if self._shm:
            self._shm.close()
            self._shm = None
        print(f"[ShmVideoSender] 已停止 (共 {self._frame_count} 帧, 错误 {self._error_count})",
              flush=True)