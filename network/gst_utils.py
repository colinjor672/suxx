"""
GStreamer + CUDA 硬件加速工具模块（修正版）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Jetson Orin Nano 硬件能力：
  ✓ nvv4l2decoder  — NVDEC 硬件 H.264 解码（独立解码单元，零 CPU）
  ✓ nvvidconv       — 硬件色彩转换/缩放（VIC 单元，零 CPU）
  ✗ NVENC           — Orin Nano 物理无此硬件编码单元
  ✓ CUDA resize     — 通用 GPU 算力缩放（占用 CUDA 核心，不占 CPU）
  ✓ nvJPEG          — GPU JPEG 编解码（占用 CUDA 核心，不占 CPU）

CPU 软方案：
  x264enc           — CPU H.264 编码（替代不存在的 NVENC）
"""

import os
import sys
import struct
import threading
import time
from collections import deque

import cv2
import numpy as np

from utils.thread_utils import set_thread_name

# ═══════════════════════════════════════════════════════════════
# 平台检测
# ═══════════════════════════════════════════════════════════════

def _is_jetson() -> bool:
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            return "jetson" in model or "orin" in model
    except FileNotFoundError:
        pass
    if os.path.exists("/usr/lib/aarch64-linux-gnu/tegra"):
        return True
    return False

def _is_orin_nano() -> bool:
    """Orin Nano 无 NVENC 硬件编码单元"""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            return "orin" in model and "nano" in model
    except FileNotFoundError:
        pass
    return False

IS_JETSON    = _is_jetson()
IS_ORIN_NANO = _is_orin_nano()
HAS_NVENC    = IS_JETSON and not IS_ORIN_NANO   # 只有 AGX Orin / Xavier 有 NVENC

# ═══════════════════════════════════════════════════════════════
# CUDA / GPU 能力检测
# ═══════════════════════════════════════════════════════════════

HAS_CUDA_RESIZE = False
try:
    _gpu_mat = cv2.cuda_GpuMat()
    _test = np.zeros((64, 64, 3), dtype=np.uint8)
    _gpu_mat.upload(_test)
    _resized = cv2.cuda.resize(_gpu_mat, (32, 32))
    _resized.download()
    HAS_CUDA_RESIZE = True
except Exception:
    pass

HAS_NVJPEG = False
_nvjpeg_encoder = None
try:
    from pynvjpeg import NvJpeg
    _nvjpeg_encoder = NvJpeg()
    _test = np.zeros((64, 64, 3), dtype=np.uint8)
    _nvjpeg_encoder.encode(_test, quality=85)
    HAS_NVJPEG = True
except Exception:
    pass

HAS_GSTREAMER = False
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    HAS_GSTREAMER = True
except Exception:
    # 回退：尝试 cv2.VideoCapture + GStreamer 后端
    try:
        _test_pipe = (
            "videotestsrc num-buffers=1 ! "
            "videoconvert ! video/x-raw,format=BGR,width=64,height=64 ! appsink"
        )
        _cap = cv2.VideoCapture(_test_pipe, cv2.CAP_GSTREAMER)
        if _cap.isOpened():
            _cap.read()
            _cap.release()
            HAS_GSTREAMER = True
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# GStreamer 视频捕获（替代 cv2.VideoCapture，直接使用 gi.repository.Gst）
# ═══════════════════════════════════════════════════════════════

class GstVideoCapture:
    """
    用 gi.repository.Gst 直接创建 GStreamer 管线抓帧。
    接口兼容 cv2.VideoCapture (isOpened / read / get / release)。
    解决 OpenCV pip 版无 GStreamer 后端的问题。
    """

    def __init__(self, pipeline_str: str, width: int = 1280, height: int = 720,
                 fps: float = 25.0):
        self._width = width
        self._height = height
        self._fps = fps
        self._opened = False
        self._pipeline = None
        self._loop = None
        self._frame_queue = deque(maxlen=2)
        self._frame_ready = threading.Condition()
        self._error_msg = ""
        self._cb_count = 0

        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst, GLib
            Gst.init(None)

            self._pipeline = Gst.parse_launch(pipeline_str)
            appsink = self._pipeline.get_by_name("sink")
            if appsink is None:
                from gi.repository import Gst
                it = self._pipeline.iterate_elements()
                while True:
                    result, elem = it.next()
                    if result != Gst.IteratorResult.OK:
                        break
                    if elem.get_factory().get_name() == "appsink":
                        appsink = elem
                        break

            if appsink is None:
                self._error_msg = "未找到 appsink"
                return

            def on_new_sample(sink):
                try:
                    self._cb_count += 1
                    sample = sink.emit("pull-sample")
                    if not sample:
                        if self._cb_count <= 1:
                            print("[GstCapture] pull-sample 返回 null", flush=True)
                        return Gst.FlowReturn.OK
                    buf = sample.get_buffer()
                    result, map_info = buf.map(Gst.MapFlags.READ)
                    if not result:
                        if self._cb_count <= 1:
                            print("[GstCapture] buffer map 失败", flush=True)
                        return Gst.FlowReturn.OK
                    data = np.frombuffer(map_info.data, dtype=np.uint8).copy()
                    buf.unmap(map_info)
                    expected = self._height * self._width * 3
                    if len(data) != expected:
                        if self._cb_count <= 1:
                            print(f"[GstCapture] 帧大小异常: {len(data)} != {expected}",
                                  flush=True)
                        return Gst.FlowReturn.OK
                    frame = data.reshape((self._height, self._width, 3))
                    with self._frame_ready:
                        self._frame_queue.append(frame)
                        self._frame_ready.notify()
                    if self._cb_count == 1:
                        print(f"[GstCapture] 首帧到达 shape={frame.shape}", flush=True)
                    return Gst.FlowReturn.OK
                except Exception as e:
                    if self._cb_count <= 1:
                        print(f"[GstCapture] 回调异常: {e}", flush=True)
                    return Gst.FlowReturn.OK

            appsink.connect("new-sample", on_new_sample)

            # 监听总线
            bus = self._pipeline.get_bus()
            bus.add_signal_watch()
            def on_bus_message(bus, msg):
                t = msg.type
                if t == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    self._error_msg = str(err.message)
                    print(f"[GstCapture] 管线错误: {err.message}", flush=True)
                    print(f"  Debug: {dbg}", flush=True)
                elif t == Gst.MessageType.WARNING:
                    err, dbg = msg.parse_warning()
                    print(f"[GstCapture] 管线警告: {err.message}", flush=True)
                elif t == Gst.MessageType.STATE_CHANGED:
                    if msg.src == self._pipeline:
                        old, new, pending = msg.parse_state_changed()
                        print(f"[GstCapture] 状态: {old.value_nick} → {new.value_nick}",
                              flush=True)
            bus.connect("message", on_bus_message)

            self._pipeline.set_state(Gst.State.PLAYING)
            self._loop = GLib.MainLoop()
            self._gst_thread = threading.Thread(target=self._loop.run, daemon=True)
            self._gst_thread.start()
            self._opened = True

        except Exception as e:
            self._error_msg = str(e)
            print(f"[GstCapture] 初始化异常: {e}", flush=True)
            self._opened = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return -1  # RTSP 无总帧数
        return 0

    def read(self):
        with self._frame_ready:
            if not self._frame_ready.wait_for(
                lambda: bool(self._frame_queue) or not self._opened,
                timeout=3.0,
            ):
                return (False, None)
            frame = None
            while self._frame_queue:
                frame = self._frame_queue.popleft()
            return (frame is not None, frame)

    def release(self):
        self._opened = False
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._pipeline:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
        if self._loop:
            try:
                self._loop.quit()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# GStreamer Tee 双路捕获：nvvidconv GPU 分叉，零 CPU 转换
# ═══════════════════════════════════════════════════════════════

# RTSP 显式硬解链，键为 ffprobe 报出的 codec_name。
# H.264/H.265 必须显式挂 nvv4l2decoder：交给 decodebin 时它常选 avdec_h264，
# 出的是系统内存帧，而 nvvidconv 只接 NVMM，结果是 caps 协商不完、管线停在
# PAUSED 且一帧不出（appsink 侧表现为 read 超时返回 False）。
_RTSP_EXPLICIT_DECODERS = {
    "h264": "rtph264depay ! h264parse ! nvv4l2decoder ",
    "hevc": "rtph265depay ! h265parse ! nvv4l2decoder ",
    "h265": "rtph265depay ! h265parse ! nvv4l2decoder ",
}


def _probe_rtsp_codec(source: str, timeout: float = 5.0):
    """探测 RTSP 流的视频编码名，探不到返回 None。

    ffprobe 只做 DESCRIBE/SETUP 就退出，不会占用后续管线；探测失败时调用方
    退回 decodebin（即平台 MJPEG 源那条已验证的路径）。
    """
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-rtsp_transport", "tcp", "-v", "error",
             "-select_streams", "v:0", "-show_entries", "stream=codec_name",
             "-of", "csv=p=0", source],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception as e:
        print(f"[GstTee] 编码探测失败（ffprobe 不可用？）: {e}", flush=True)
        return None

    text = (out.stdout or "").strip()
    if not text:
        return None
    return text.splitlines()[0].strip().lower() or None


class GstTeeCapture:
    """
    GStreamer 双路捕获：nvvidconv (VIC) 分两路出 RGBA，零 CPU 色彩转换

      rtspsrc → nvv4l2decoder → tee
        ├─ nvvidconv → RGBA 640×360  → appsink(ai_sink)    → AI 推理（BGR view）
        └─ nvvidconv → RGBA 1280×720 → appsink(godot_sink) → shm_input → Godot

    AI 分支保持 640×360，推理输入与耗时不变；Godot 分支原生 720p，
    避免 640×360 在 1280×720 画布上放大再缩回导致的糊化。
    回调中每路只做 1 次拷贝（GStreamer buffer 生命周期要求），
    BGR 通过 rgba[:, :, :3] view 获取，零额外拷贝。
    """
    AI_W, AI_H       = 640, 360
    GODOT_W, GODOT_H = 1280, 720
    GST_CLOCK_TIME_NONE = 18446744073709551615   # 2^64 - 1，GStreamer 表示无效时间戳

    def __init__(self, source: str, fps: float = 30.0):
        self._fps = fps
        self._opened = False
        self._pipeline = None
        self._loop = None
        self._error_msg = ""

        # 两个分支各自只保留最新一帧；read() 等两路都有数据后一起取走。
        # 主循环 24fps、源 25~30fps 本来就会丢帧，逐帧配对没有意义。
        self._ai_frame = None      # (bgr_view, pts_ns, pts_latency_s)
        self._godot_frame = None   # (rgba, pts_ns)
        self._frame_ready = threading.Condition()
        self._ai_count = 0
        self._godot_count = 0
        self._err_count = 0

        # 本地文件由 GStreamer 时钟和主循环 FramePacer 双重保证播放节拍。
        import os
        is_rtsp = source.startswith("rtsp://") or source.startswith("rtspt://")
        is_file = os.path.isfile(source)
        self._is_file = is_file
        self._is_rtsp = is_rtsp

        # read() 返回的帧对应的"编码端出图 → 解码出帧"耗时（秒），
        # 由 PTS 与管道 running-time 的差估算；无法估算时为 None。
        self.last_pts_latency_s = None

        if not is_rtsp and not is_file:
            self._error_msg = f"不支持的源: {source}"
            return

        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst, GLib
            Gst.init(None)

            if is_rtsp:
                # 按实际编码分叉：H.264/H.265 走显式 nvv4l2decoder，其余（平台 MJPEG
                # 源经 decodebin 出 NVMM 硬解，日志有 NVMMLITE_NVVIDEODEC）走 decodebin。
                # 平台那条换成 videoscale ! videoconvert 的纯 CPU 链会立刻
                # "Delayed linking failed" 起不来（已实测，别再试）。
                # protocols=4 在本机 GStreamer 上就是 TCP（HTTP 是 0x10），location
                # 必须加引号（URL 里有 @ 和 :）。
                # jitterbuffer 退回保守值：drop-on-latency 会丢掉还没同步好的 IDR，
                # 自写推流端的 RTCP SR 时间锚点稍有偏差就整段零帧；延迟由下游两路
                # leaky queue 兜住，不靠 jitterbuffer 丢帧。
                codec   = _probe_rtsp_codec(source)
                decoder = _RTSP_EXPLICIT_DECODERS.get(codec or "")
                print(f"[GstTee] RTSP 编码探测: {codec or '未知'} → "
                      f"{'显式 nvv4l2decoder' if decoder else 'decodebin'}")

                pipeline_str = (
                    f'rtspsrc location="{source}" protocols=4 latency=200 ! '
                    f'{decoder or "decodebin "}! tee name=t '
                    "t. ! queue leaky=downstream max-size-buffers=2 "
                    "max-size-bytes=0 max-size-time=0 ! "
                    f"nvvidconv ! video/x-raw,format=RGBA,"
                    f"width={self.AI_W},height={self.AI_H} ! "
                    "appsink name=ai_sink emit-signals=true max-buffers=2 "
                    "drop=true sync=false "
                    "t. ! queue leaky=downstream max-size-buffers=2 "
                    "max-size-bytes=0 max-size-time=0 ! "
                    f"nvvidconv ! video/x-raw,format=RGBA,"
                    f"width={self.GODOT_W},height={self.GODOT_H} ! "
                    "appsink name=godot_sink emit-signals=true max-buffers=2 "
                    "drop=true sync=false"
                )
            else:
                # 本地文件由 appsink 按媒体 PTS 同步到 GStreamer 时钟。
                abs_path = os.path.abspath(source)
                uri = "file://" + abs_path
                pipeline_str = (
                    f"uridecodebin uri={uri} ! tee name=t "
                    "t. ! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! "
                    f"nvvidconv ! video/x-raw,format=RGBA,"
                    f"width={self.AI_W},height={self.AI_H} ! "
                    "appsink name=ai_sink emit-signals=true max-buffers=2 "
                    "drop=false sync=true "
                    "t. ! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! "
                    f"nvvidconv ! video/x-raw,format=RGBA,"
                    f"width={self.GODOT_W},height={self.GODOT_H} ! "
                    "appsink name=godot_sink emit-signals=true max-buffers=2 "
                    "drop=false sync=true"
                )
                print(f"[GstTee] 本地视频文件 (PTS 限帧): {abs_path}")

            self._pipeline = Gst.parse_launch(pipeline_str)

            ai_sink = self._pipeline.get_by_name("ai_sink")
            godot_sink = self._pipeline.get_by_name("godot_sink")
            if ai_sink is None or godot_sink is None:
                self._error_msg = "未找到 appsink"
                return

            def make_on_sample(kind, w, h):
                expected = w * h * 4  # RGBA

                def on_sample(s):
                    try:
                        if kind == "ai":
                            self._ai_count += 1
                            count = self._ai_count
                        else:
                            self._godot_count += 1
                            count = self._godot_count

                        sample = s.emit("pull-sample")
                        if not sample:
                            return Gst.FlowReturn.OK
                        buf = sample.get_buffer()
                        pts_ns = buf.pts  # GStreamer PTS 时间戳（纳秒）
                        pts_latency_s = self._estimate_pts_latency(pts_ns)
                        result, map_info = buf.map(Gst.MapFlags.READ)
                        if not result:
                            return Gst.FlowReturn.OK
                        # 每路唯一一次拷贝（GStreamer buffer 在 unmap 后失效）
                        rgba = np.frombuffer(
                            map_info.data, dtype=np.uint8, count=expected
                        ).reshape((h, w, 4)).copy()
                        buf.unmap(map_info)

                        if kind == "ai":
                            # RGBA 前 3 通道是 RGB，反转为 BGR（view，零拷贝，非连续）
                            bgr_view = rgba[:, :, :3][:, :, ::-1]
                            with self._frame_ready:
                                self._ai_frame = (bgr_view, pts_ns, pts_latency_s)
                                self._frame_ready.notify_all()
                            if count == 1:
                                print(f"[GstTee] AI 分支首帧 RGBA {rgba.shape} "
                                      f"(BGR view 零拷贝)", flush=True)
                        else:
                            with self._frame_ready:
                                self._godot_frame = (rgba, pts_ns)
                                self._frame_ready.notify_all()
                            if count == 1:
                                print(f"[GstTee] Godot 分支首帧 RGBA {rgba.shape}",
                                      flush=True)
                        return Gst.FlowReturn.OK
                    except Exception as e:
                        self._err_count += 1
                        if self._err_count <= 2:
                            print(f"[GstTee] 回调异常({kind}): {e}", flush=True)
                        return Gst.FlowReturn.OK
                return on_sample

            ai_sink.connect("new-sample", make_on_sample("ai", self.AI_W, self.AI_H))
            godot_sink.connect("new-sample",
                               make_on_sample("godot", self.GODOT_W, self.GODOT_H))

            # 监听总线
            bus = self._pipeline.get_bus()
            bus.add_signal_watch()
            def on_bus_message(bus, msg):
                t = msg.type
                if t == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    self._error_msg = str(err.message)
                    print(f"[GstTee] 管线错误: {err.message}", flush=True)
                elif t == Gst.MessageType.WARNING:
                    err, dbg = msg.parse_warning()
                    print(f"[GstTee] 管线警告: {err.message}", flush=True)
                elif t == Gst.MessageType.STATE_CHANGED:
                    if msg.src == self._pipeline:
                        old, new, pending = msg.parse_state_changed()
                        print(f"[GstTee] 状态: {old.value_nick} → {new.value_nick}", flush=True)
            bus.connect("message", on_bus_message)

            self._pipeline.set_state(Gst.State.PLAYING)
            self._loop = GLib.MainLoop()
            self._gst_thread = threading.Thread(target=self._gst_main, daemon=True)
            self._gst_thread.start()

            self._opened = True

        except Exception as e:
            self._error_msg = str(e)
            print(f"[GstTee] 初始化异常: {e}", flush=True)
            self._opened = False

    def _gst_main(self):
        set_thread_name("GstMain")
        self._loop.run()

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.AI_W
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.AI_H
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return -1
        return 0

    def read(self):
        """返回 (True, (bgr_640x360, rgba_1280x720)) 或 (False, (None, None))

        bgr_frame 来自 AI 分支，是其自身 rgba 的 view（零拷贝）；
        rgba_frame 来自 Godot 分支，原生 1280×720。
        两路各自只保留最新帧，等两路都就绪后一起取走。
        本地视频文件由 appsink 媒体时钟限帧，RTSP 摄像头不限帧。
        """
        with self._frame_ready:
            if not self._frame_ready.wait_for(
                lambda: (self._ai_frame is not None
                         and self._godot_frame is not None) or not self._opened,
                timeout=3.0,
            ):
                return (False, (None, None))
            if self._ai_frame is None or self._godot_frame is None:
                return (False, (None, None))

            bgr, _ai_pts, pts_latency_s = self._ai_frame
            rgba = self._godot_frame[0]
            self._ai_frame = None
            self._godot_frame = None
            self.last_pts_latency_s = pts_latency_s
            return (True, (bgr, rgba))

    def _estimate_pts_latency(self, pts_ns):
        """估算 rtspsrc 传输 + jitterbuffer + 解码的耗时（秒）。

        rtspsrc 借 RTCP SR 把 RTP 时间戳映射到管道时钟，因此
        running_time - pts 就是该帧从"编码端出图"到"解码完成出 appsink"的时长。
        本地文件的 PTS 从 0 起按媒体时钟推进，没有这个含义，返回 None。
        """
        if not self._is_rtsp or pts_ns == self.GST_CLOCK_TIME_NONE:
            return None

        clock = self._pipeline.get_clock()
        if clock is None:
            return None

        running_ns = clock.get_time() - self._pipeline.get_base_time()
        latency_s = (running_ns - pts_ns) / 1e9

        # RTCP 尚未同步或时钟异常时差值会明显失真，超出合理区间的一律丢弃
        if 0.0 <= latency_s < 5.0:
            return latency_s
        return None

    def release(self):
        self._opened = False
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._pipeline:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
        if self._loop:
            try:
                self._loop.quit()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# 硬件解码：nvv4l2decoder + nvvidconv（零 CPU 色彩转换）
# ═══════════════════════════════════════════════════════════════

def create_hw_decode_capture(source: str, fps: float = 30.0):
    """
    硬件解码的视频捕获器（统一 RGBA，tee 双路）。
    nvv4l2decoder → nvvidconv (GPU VIC 零 CPU)
      ├── RGBA 640×360  → appsink → BGR view → AI 推理
      └── RGBA 1280×720 → appsink → 共享内存 → Godot

    source: RTSP URL 或本地视频文件路径
    """
    import os
    is_rtsp = source.startswith("rtsp://") or source.startswith("rtspt://")
    is_file = os.path.isfile(source)

    source_fps = float(fps)
    if is_file:
        probe = cv2.VideoCapture(source)
        try:
            detected_fps = float(probe.get(cv2.CAP_PROP_FPS))
            if np.isfinite(detected_fps) and detected_fps > 0:
                source_fps = detected_fps
        finally:
            probe.release()

    if IS_JETSON and HAS_GSTREAMER and (is_rtsp or is_file):
        if is_rtsp:
            print(f"[GstUtils] ✓ RTSP 硬解: nvv4l2decoder → nvvidconv (零 CPU)")
        else:
            print(f"[GstUtils] ✓ 本地视频硬解: uridecodebin → nvvidconv (零 CPU)")
        cap = GstTeeCapture(source, fps=source_fps)
        if cap.isOpened():
            if is_rtsp:
                import time as _time
                _time.sleep(2.0)
            return cap
        else:
            print(f"[GstUtils] ⚠ GstTeeCapture 失败: {cap._error_msg}")
            cap.release()

    print(f"[GstUtils] CPU 软解 (cv2.VideoCapture)")
    return cv2.VideoCapture(source)


# ═══════════════════════════════════════════════════════════════
# CPU 编码：x264enc（Orin Nano 无 NVENC，用 CPU 软编）
# ═══════════════════════════════════════════════════════════════

def create_x264enc_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
    bitrate: int = 8_000_000,
):
    """
    x264enc CPU 软编码 VideoWriter（GStreamer 后端）。
    比 OpenCV mp4v 快，且支持更多参数调节。
    """
    if not HAS_GSTREAMER:
        return None

    pipeline = (
        f"appsrc ! "
        f"video/x-raw,format=BGR,width={width},height={height},"
        f"framerate={fps:.0f}/1 ! "
        f"videoconvert ! video/x-raw,format=I420 ! "
        f"x264enc bitrate={bitrate} speed-preset=ultrafast "
        f"tune=zerolatency ! "
        f"h264parse ! qtmux ! "
        f"filesink location={output_path}"
    )

    writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height))
    if writer.isOpened():
        print(f"[GstUtils] ✓ x264enc CPU 编码 ({width}×{height} @ {fps:.0f}fps, "
              f"{bitrate//1_000_000}Mbps, ultrafast)")
        return writer
    else:
        writer.release()
        return None


# ═══════════════════════════════════════════════════════════════
# H.264 解码桥接 → 共享内存（供 Godot 读取）
# ═══════════════════════════════════════════════════════════════

# 共享内存路径
SHM_VIDEO_PATH  = "/dev/shm/godot_video.raw"   # RGBA 原始帧
SHM_META_PATH   = "/dev/shm/godot_video.meta"  # 帧元数据 (frame_id, w, h, ts)

class H264DecoderBridge:
    """
    GStreamer H.264 解码桥接。
    
    用法（在 Godot 侧启动）:
      bridge = H264DecoderBridge(width=1280, height=720)
      bridge.start()
      # 从 Python 侧接收 H.264 数据后调用:
      bridge.feed(h264_data)
      # Godot 读取 /dev/shm/godot_video.raw + .meta
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width  = width
        self.height = height
        self.fps    = fps
        self._pipeline = None
        self._appsink  = None
        self._appsrc   = None
        self._running  = False
        self._lock     = threading.Lock()
        self._latest_frame = None
        self._latest_meta  = (0, 0.0)  # (frame_id, timestamp)

        # 确保共享内存目录存在
        if not os.path.exists("/dev/shm"):
            os.makedirs("/dev/shm", exist_ok=True)

    def _build_pipeline(self) -> str:
        """构建 GStreamer 解码管线：appsrc → h264parse → nvv4l2decoder → nvvidconv → RGBA → appsink"""
        if IS_JETSON and HAS_GSTREAMER:
            # Jetson: nvv4l2decoder 硬解 + nvvidconv 零 CPU 色彩转换
            return (
                f"appsrc name=src format=time is-live=true do-timestamp=true ! "
                f"video/x-h264,width={self.width},height={self.height},"
                f"framerate={self.fps}/1,stream-format=byte-stream,alignment=au ! "
                f"h264parse ! nvv4l2decoder ! "
                f"nvvidconv ! video/x-raw,format=RGBA ! "
                f"appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
            )
        else:
            # PC: avdec_h264 软解
            return (
                f"appsrc name=src format=time is-live=true do-timestamp=true ! "
                f"video/x-h264,width={self.width},height={self.height},"
                f"framerate={self.fps}/1,stream-format=byte-stream,alignment=au ! "
                f"h264parse ! avdec_h264 ! "
                f"videoconvert ! video/x-raw,format=RGBA ! "
                f"appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
            )

    def start(self) -> bool:
        """启动 GStreamer 管线"""
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst, GLib
            Gst.init(None)
        except ImportError:
            print("[H264Bridge] ⚠ pygobject/GStreamer 不可用，使用子进程模式")
            return self._start_subprocess()

        pipeline_str = self._build_pipeline()
        print(f"[H264Bridge] 管线: {pipeline_str[:120]}...")

        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
            self._appsrc  = self._pipeline.get_by_name("src")
            self._appsink = self._pipeline.get_by_name("sink")

            if self._appsink:
                self._appsink.set_property("emit-signals", True)
                self._appsink.connect("new-sample", self._on_new_sample)

            self._pipeline.set_state(Gst.State.PLAYING)
            self._running = True

            # 启动 GLib 主循环在后台线程
            self._loop = GLib.MainLoop()
            self._thread = threading.Thread(target=self._loop.run, daemon=True)
            self._thread.start()

            print(f"[H264Bridge] ✓ 启动成功 "
                  f"({'nvv4l2decoder GPU' if IS_JETSON else 'avdec_h264 CPU'})")
            return True
        except Exception as e:
            print(f"[H264Bridge] ✗ 启动失败: {e}")
            return False

    def _start_subprocess(self) -> bool:
        """回退：使用子进程 GStreamer"""
        print("[H264Bridge] 子进程模式暂未实现，请安装 pygobject")
        return False

    def _on_new_sample(self, sink) -> int:
        """GStreamer 回调：收到解码帧 → 写入共享内存"""
        try:
            from gi.repository import Gst
            sample = sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR

            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)

            w = structure.get_int("width").value if structure.has_field("width") else self.width
            h = structure.get_int("height").value if structure.has_field("height") else self.height

            result, map_info = buf.map(Gst.MapFlags.READ)
            if not result:
                return Gst.FlowReturn.ERROR

            frame_data = bytes(map_info.data)
            buf.unmap(map_info)

            # 写入共享内存
            self._write_shm(frame_data, w, h)

            return Gst.FlowReturn.OK
        except Exception as e:
            print(f"[H264Bridge] 帧回调异常: {e}")
            return Gst.FlowReturn.ERROR

    def feed(self, h264_data: bytes, frame_id: int = 0, timestamp: float = 0.0):
        """喂入 H.264 数据"""
        if self._appsrc is None:
            return

        try:
            from gi.repository import Gst
            buf = Gst.Buffer.new_allocate(None, len(h264_data), None)
            buf.fill(0, h264_data)
            self._appsrc.emit("push-buffer", buf)
            self._latest_meta = (frame_id, timestamp)
        except Exception as e:
            print(f"[H264Bridge] feed 异常: {e}")

    def _write_shm(self, rgba_data: bytes, w: int, h: int):
        """写入共享内存：原始 RGBA 帧 + 元数据"""
        expected_size = w * h * 4
        if len(rgba_data) != expected_size:
            print(f"[H264Bridge] 帧大小异常: {len(rgba_data)} != {expected_size}")
            return

        try:
            with open(SHM_VIDEO_PATH, "wb") as f:
                f.write(rgba_data)

            frame_id, ts = self._latest_meta
            meta = struct.pack("!I I I d", frame_id, w, h, ts)
            with open(SHM_META_PATH, "wb") as f:
                f.write(meta)
        except Exception as e:
            print(f"[H264Bridge] 写共享内存失败: {e}")

    def stop(self):
        """停止管线"""
        self._running = False
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
        if hasattr(self, '_loop') and self._loop:
            try:
                self._loop.quit()
            except Exception:
                pass
        print("[H264Bridge] 已停止")


# ═══════════════════════════════════════════════════════════════
# 共享内存读取（供 Godot 调用）
# ═══════════════════════════════════════════════════════════════

def read_video_shm() -> tuple:
    """
    读取共享内存中的视频帧。
    返回 (rgba_bytes, frame_id, width, height, timestamp) 或 None
    """
    try:
        with open(SHM_META_PATH, "rb") as f:
            meta = f.read(20)
            if len(meta) < 20:
                return None
            frame_id, w, h, ts = struct.unpack("!I I I d", meta)
    except FileNotFoundError:
        return None
    except Exception:
        return None

    expected_size = w * h * 4
    try:
        with open(SHM_VIDEO_PATH, "rb") as f:
            data = f.read(expected_size)
            if len(data) != expected_size:
                return None
    except Exception:
        return None

    return (data, frame_id, w, h, ts)


# ═══════════════════════════════════════════════════════════════
# CUDA 加速 resize
# ═══════════════════════════════════════════════════════════════

def cuda_resize(frame: np.ndarray, dsize: tuple,
                interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """GPU 缩放（回退 CPU）"""
    if not HAS_CUDA_RESIZE:
        return cv2.resize(frame, dsize, interpolation=interpolation)

    try:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(frame)
        resized = cv2.cuda.resize(gpu, dsize, interpolation=interpolation)
        return resized.download()
    except cv2.error:
        return cv2.resize(frame, dsize, interpolation=interpolation)


# ═══════════════════════════════════════════════════════════════
# nvJPEG 硬件编码
# ═══════════════════════════════════════════════════════════════

def nvjpeg_encode(frame: np.ndarray, quality: int = 85) -> bytes:
    """GPU JPEG 编码（回退返回空）"""
    if not HAS_NVJPEG or _nvjpeg_encoder is None:
        return b''

    try:
        return _nvjpeg_encoder.encode(frame, quality)
    except Exception:
        return b''


# ═══════════════════════════════════════════════════════════════
# 信息打印
# ═══════════════════════════════════════════════════════════════

def print_capabilities():
    """打印硬件加速能力"""
    print(f"[GstUtils] ── 硬件加速能力 ──")
    print(f"  平台          : {'Orin Nano' if IS_ORIN_NANO else 'Jetson' if IS_JETSON else 'PC / 其他'}")
    print(f"  GStreamer     : {'✓' if HAS_GSTREAMER else '✗'}")
    print(f"  nvv4l2decoder : {'✓ 硬件解码 (零 CPU)' if IS_JETSON and HAS_GSTREAMER else '✗'}")
    print(f"  nvvidconv     : {'✓ 硬件色彩转换 (零 CPU)' if IS_JETSON and HAS_GSTREAMER else '✗'}")
    print(f"  NVENC         : {'✓ 硬件编码' if HAS_NVENC else '✗ (Orin Nano 无此单元)'}")
    print(f"  CUDA resize   : {'✓' if HAS_CUDA_RESIZE else '✗'}")
    print(f"  nvJPEG        : {'✓' if HAS_NVJPEG else '✗'}")
    print(f"  CPU 编码方案   : x264enc (speed-preset=ultrafast)")


# ═══════════════════════════════════════════════════════════════
# H.264 编码器（x264enc CPU）—— 供 Python 发送端使用
# ═══════════════════════════════════════════════════════════════

class H264Encoder:
    """
    x264enc CPU H.264 编码器。
    将 BGR numpy 帧编码为 H.264 byte-stream，通过 TCP 发送给 Godot 侧的 H264DecoderBridge。
    
    用法:
        enc = H264Encoder(width=1280, height=720, fps=30)
        for frame in frames:
            h264_data = enc.encode(frame)   # bytes
            tcp_socket.send(h264_data)
        enc.close()
    """

    def __init__(self, width: int, height: int, fps: int = 30,
                 bitrate: int = 4_000_000):
        self.width   = width
        self.height  = height
        self.fps     = fps
        self.bitrate = bitrate
        self._pipeline = None
        self._appsrc   = None
        self._appsink  = None
        self._started  = False

    def start(self) -> bool:
        """启动 GStreamer 编码管线"""
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
            Gst.init(None)
        except ImportError:
            print("[H264Encoder] ⚠ pygobject 不可用")
            return False

        pipeline_str = (
            f"appsrc name=src format=time is-live=true ! "
            f"video/x-raw,format=BGR,width={self.width},height={self.height},"
            f"framerate={self.fps}/1 ! "
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc bitrate={self.bitrate} speed-preset=ultrafast "
            f"tune=zerolatency key-int-max=30 ! "
            f"video/x-h264,stream-format=byte-stream,alignment=au ! "
            f"appsink name=sink emit-signals=true max-buffers=1 sync=false"
        )

        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
            self._appsrc   = self._pipeline.get_by_name("src")
            self._appsink  = self._pipeline.get_by_name("sink")
            self._pipeline.set_state(Gst.State.PLAYING)
            self._started  = True
            print(f"[H264Encoder] ✓ x264enc ({self.width}×{self.height} @ {self.fps}fps, "
                  f"{self.bitrate//1_000_000}Mbps, ultrafast)")
            return True
        except Exception as e:
            print(f"[H264Encoder] ✗ 启动失败: {e}")
            return False

    def encode(self, frame_bgr: np.ndarray) -> bytes:
        """编码一帧 BGR → H.264 byte-stream，返回 NAL 单元"""
        if not self._started:
            return b''

        try:
            from gi.repository import Gst

            h, w = frame_bgr.shape[:2]
            if w != self.width or h != self.height:
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

            # 推入 appsrc
            data = frame_bgr.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            self._appsrc.emit("push-buffer", buf)

            # 从 appsink 拉取 H.264
            sample = self._appsink.emit("pull-sample")
            if sample is None:
                return b''

            gst_buf = sample.get_buffer()
            result, map_info = gst_buf.map(Gst.MapFlags.READ)
            if not result:
                return b''

            h264_data = bytes(map_info.data)
            gst_buf.unmap(map_info)
            return h264_data
        except Exception as e:
            print(f"[H264Encoder] 编码异常: {e}")
            return b''

    def close(self):
        """关闭编码器"""
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
        self._started = False
        print("[H264Encoder] 已关闭")


# ═══════════════════════════════════════════════════════════════
# nvJPEG GPU 硬件 JPEG 编码器（GStreamer nvjpegenc）
# ═══════════════════════════════════════════════════════════════

class NvJpegEncoder:
    """
    GStreamer nvjpegenc GPU 硬件 JPEG 编码器。
    将 RGBA numpy 帧编码为 JPEG bytes。

    管线:
      appsrc → nvvidconv (RGBA→I420, GPU VIC) → nvjpegenc (GPU) → appsink → JPEG bytes

    用法:
        enc = NvJpegEncoder(width=1280, height=720, fps=20, quality=80)
        enc.start()
        jpeg_bytes = enc.encode(rgba_frame)  # numpy RGBA → JPEG bytes
        enc.stop()
    """

    def __init__(self, width: int, height: int, fps: int = 20,
                 quality: int = 80):
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self._pipeline = None
        self._appsrc = None
        self._loop = None
        self._started = False
        self._jpeg_queue = deque(maxlen=2)
        self._jpeg_ready = threading.Condition()
        self._frame_count = 0
        self._error_msg = "" 

    def start(self) -> bool:
        """启动 GStreamer nvjpegenc 编码管线"""
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst, GLib
            Gst.init(None)
        except ImportError:
            self._error_msg = "pygobject 不可用"
            print(f"[NvJpeg] {self._error_msg}")
            return False

        pipeline_str = (
            f"appsrc name=src format=time is-live=true block=true ! "
            f"video/x-raw,format=RGBA,width={self.width},height={self.height},"
            f"framerate={self.fps}/1 ! "
            f"nvvidconv ! video/x-raw,format=I420 ! "
            f"nvjpegenc quality={self.quality} ! "
            f"appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )

        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
            self._appsrc = self._pipeline.get_by_name("src")

            appsink = self._pipeline.get_by_name("sink")
            if appsink is None:
                self._error_msg = "未找到 appsink"
                print(f"[NvJpeg] {self._error_msg}")
                return False

            def on_jpeg_sample(sink):
                try:
                    sample = sink.emit("pull-sample")
                    if not sample:
                        return Gst.FlowReturn.OK
                    buf = sample.get_buffer()
                    result, map_info = buf.map(Gst.MapFlags.READ)
                    if not result:
                        return Gst.FlowReturn.OK
                    jpeg_data = bytes(map_info.data)
                    buf.unmap(map_info)
                    with self._jpeg_ready:
                        self._jpeg_queue.append(jpeg_data)
                        self._jpeg_ready.notify()
                    return Gst.FlowReturn.OK
                except Exception:
                    return Gst.FlowReturn.OK

            appsink.connect("new-sample", on_jpeg_sample)

            # 监听总线
            bus = self._pipeline.get_bus()
            bus.add_signal_watch()
            def on_bus_message(bus, msg):
                t = msg.type
                if t == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    self._error_msg = str(err.message)
                    print(f"[NvJpeg] 管线错误: {err.message}", flush=True)
                elif t == Gst.MessageType.WARNING:
                    err, dbg = msg.parse_warning()
                    if "nvjpegenc" not in str(err.message).lower():
                        print(f"[NvJpeg] 管线警告: {err.message}", flush=True)
            bus.connect("message", on_bus_message)

            self._pipeline.set_state(Gst.State.PLAYING)
            self._loop = GLib.MainLoop()
            self._gst_thread = threading.Thread(target=self._loop.run, daemon=True)
            self._gst_thread.start()
            self._started = True

            print(f"[NvJpeg] ✓ GPU JPEG 编码 {self.width}×{self.height} "
                  f"@ {self.fps}fps, quality={self.quality}", flush=True)
            return True

        except Exception as e:
            self._error_msg = str(e)
            print(f"[NvJpeg] 启动失败: {e}", flush=True)
            return False

    def encode(self, rgba_frame: np.ndarray) -> bytes:
        """编码一帧 RGBA → JPEG bytes（GPU 硬件编码）"""
        return self.encode_bytes(rgba_frame.tobytes())

    def encode_bytes(self, rgba_bytes: bytes) -> bytes:
        """编码 RGBA bytes → JPEG bytes（GPU 硬件编码，零额外拷贝）
        
        rgba_bytes: 921600 bytes (640×360×4 RGBA)
        返回: JPEG bytes
        """
        if not self._started:
            return b''

        try:
            from gi.repository import Gst

            buf = Gst.Buffer.new_wrapped(rgba_bytes)
            self._appsrc.emit("push-buffer", buf)

            # Block until the callback publishes the encoded frame.
            with self._jpeg_ready:
                if not self._jpeg_ready.wait_for(
                    lambda: bool(self._jpeg_queue) or not self._started,
                    timeout=0.5,
                ):
                    return b''
                if not self._jpeg_queue:
                    return b''
                jpeg = self._jpeg_queue.popleft()
                self._frame_count += 1
                if self._frame_count % 200 == 0:
                    print(f"[NvJpeg] 已编码 {self._frame_count} 帧", flush=True)
                return jpeg
        except Exception as e:
            if self._frame_count <= 3:
                print(f"[NvJpeg] 编码异常: {e}", flush=True)
            return b''

    def stop(self):
        """关闭编码器"""
        self._started = False
        with self._jpeg_ready:
            self._jpeg_ready.notify_all()
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
        if self._loop is not None:
            try:
                self._loop.quit()
            except Exception:
                pass
        print(f"[NvJpeg] 已停止 (共 {self._frame_count} 帧)", flush=True)
