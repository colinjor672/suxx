#!/usr/bin/env python3
"""
RTSP 推流器（Godot 合成帧 → EasyDarwin）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
从 /dev/shm 双缓冲读取 Godot 合成帧 (1280×720 RGBA)
→ x264enc 软件编码 → RTSP 推流到 EasyDarwin

双缓冲读取协议：
  /dev/shm/godot_frame_0.raw   — 槽0 原始 RGBA (1280×720×4 = 3686400 字节)
  /dev/shm/godot_frame_1.raw   — 槽1
  /dev/shm/godot_frame_ctrl    — 8 字节: [write_index:u32 LE][frame_id:u32 LE]
  只读最新帧，编码跟不上时丢弃旧帧，不补读。
"""

import os
import struct
import threading
import time
import sys

FRAME_W = 1280
FRAME_H = 720
FRAME_SIZE = FRAME_W * FRAME_H * 4  # RGBA

SHM_FRAME_0   = "/dev/shm/godot_frame_0.raw"
SHM_FRAME_1   = "/dev/shm/godot_frame_1.raw"
SHM_FRAME_CTRL = "/dev/shm/godot_frame_ctrl"

from network.gst_utils import IS_JETSON, HAS_GSTREAMER


class RtspPublisher:
    """
    双缓冲读取 Godot 合成帧 → x264enc → RTSP 推流到 EasyDarwin

    用法:
        pub = RtspPublisher(rtsp_url="rtsp://127.0.0.1:15544/stream")
        pub.start()
        # 后台线程自动轮询 /dev/shm，编码并推流
        pub.stop()
    """

    def __init__(self, rtsp_url: str, fps: int = 20, bitrate: int = 4_000_000):
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.bitrate = bitrate
        self._pipeline = None
        self._appsrc = None
        self._running = False
        self._last_read_index = -1
        self._frame_count = 0
        self._drop_count = 0

        os.makedirs("/dev/shm", exist_ok=True)

    def start(self) -> bool:
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst, GLib
            Gst.init(None)
        except ImportError:
            print("[RtspPublisher] pygobject 不可用", flush=True)
            return False

        # x264enc CPU 软编码 → RTSP 推流到 EasyDarwin
        pipeline_str = (
            f"appsrc name=src format=time is-live=true do-timestamp=true ! "
            f"video/x-raw,format=RGBA,width={FRAME_W},height={FRAME_H},"
            f"framerate={self.fps}/1 ! "
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc bitrate={self.bitrate} speed-preset=ultrafast "
            f"tune=zerolatency key-int-max=30 ! "
            f"video/x-h264,stream-format=byte-stream,alignment=au ! "
            f"h264parse ! "
            f"rtspclientsink location={self.rtsp_url} "
            f"name=sink"
        )

        print(f"[RtspPublisher] x264enc → RTSP: {self.rtsp_url}", flush=True)

        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
            self._appsrc = self._pipeline.get_by_name("src")

            self._pipeline.set_state(Gst.State.PLAYING)
            self._running = True

            # 轮询线程：从 /dev/shm 读帧 → 推入 GStreamer
            self._poll_thread = threading.Thread(
                target=self._poll_loop, daemon=True, name="RtspPublisher"
            )
            self._poll_thread.start()

            print(f"[RtspPublisher] 已启动 {FRAME_W}×{FRAME_H} @ {self.fps}fps", flush=True)
            return True

        except Exception as e:
            print(f"[RtspPublisher] 启动失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

    def _poll_loop(self):
        """轮询 /dev/shm 双缓冲，读取最新 Godot 合成帧并推流"""
        while self._running:
            try:
                # 读控制结构
                if not os.path.exists(SHM_FRAME_CTRL):
                    time.sleep(0.005)
                    continue

                with open(SHM_FRAME_CTRL, "rb") as f:
                    ctrl = f.read(8)
                if len(ctrl) < 8:
                    time.sleep(0.005)
                    continue

                write_index = struct.unpack("<I", ctrl[:4])[0]
                frame_id = struct.unpack("<I", ctrl[4:8])[0]

                if write_index == self._last_read_index:
                    time.sleep(0.003)
                    continue

                # 有新帧 → 跳过旧帧，只读最新
                if write_index > self._last_read_index + 1 and self._last_read_index >= 0:
                    self._drop_count += (write_index - self._last_read_index - 1)

                self._last_read_index = write_index

                slot_idx = write_index % 2
                slot_path = SHM_FRAME_0 if slot_idx == 0 else SHM_FRAME_1

                if not os.path.exists(slot_path):
                    continue

                with open(slot_path, "rb") as f:
                    rgba_data = f.read(FRAME_SIZE)

                if len(rgba_data) != FRAME_SIZE:
                    continue

                # 推入 GStreamer 编码管线
                self._push_frame(rgba_data)
                self._frame_count += 1

                if self._frame_count % 100 == 0:
                    print(f"[RtspPublisher] 已推送 {self._frame_count} 帧 "
                          f"(丢弃 {self._drop_count})", flush=True)

            except Exception:
                time.sleep(0.01)

    def _push_frame(self, rgba_data: bytes):
        """推入 GStreamer appsrc"""
        if self._appsrc is None:
            return
        try:
            from gi.repository import Gst
            buf = Gst.Buffer.new_allocate(None, len(rgba_data), None)
            buf.fill(0, rgba_data)
            self._appsrc.emit("push-buffer", buf)
        except Exception:
            pass

    def stop(self):
        self._running = False
        if self._pipeline:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
        print(f"[RtspPublisher] 已停止 (推送 {self._frame_count} 帧, "
              f"丢弃 {self._drop_count} 帧)", flush=True)