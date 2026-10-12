import os
import sys
import time
import threading

import cv2
import numpy as np
from network.gst_utils import (
    print_capabilities,
    create_hw_decode_capture,
    create_x264enc_writer,
    H264Encoder,
    cuda_resize,
    nvjpeg_encode,
    HAS_CUDA_RESIZE,
    HAS_NVJPEG,
    IS_JETSON,
)

_encoder = None
_encoder_name = "raw"

if HAS_NVJPEG:
    try:
        from pynvjpeg import NvJpeg
        _encoder = NvJpeg()
        _encoder_name = "nvjpeg"
        print("[VideoStream] ✓ 编码器: nvJPEG (GPU)")
    except Exception:
        pass
if _encoder is None:
    try:
        from turbojpeg import TurboJPEG

        if sys.platform.startswith("win"):
                _encoder = TurboJPEG(lib_path=r"C:/libjpeg-turbo64/bin/turbojpeg.dll")
        else:
            _encoder = TurboJPEG()
        _encoder_name = "turbojpeg"

    except (ImportError, RuntimeError, OSError) as e:
        print(f"⚠️ TurboJPEG 不可用，改用 cv2.imencode: {e}")
        _encoder = None
        _encoder_name = "cv2"


print(f"[VideoStream] 编码器: {_encoder_name}")


def encode_jpeg(bgr_frame: np.ndarray, quality: int = 70) -> bytes:
    if _encoder_name == "turbojpeg":
        result = nvjpeg_encode(bgr_frame, quality)
        if result:
            return result
        # nvJPEG 失败，回退到 TurboJPEG
    if _encoder_name == "turbojpeg" and _encoder is not None:
        try:
            return _encoder.encode(bgr_frame, quality=quality)
        except Exception:
            pass
    # cv2 fallback（最慢）
    ret, buf = cv2.imencode('.jpg', bgr_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ret else b''



class AsyncTCPVideoStreamer:
    def __init__(
        self,
        nav_server_ref,
        stream_scale: float = 0.33,
        send_interval: int = 1,
        jpeg_quality: int = 75,
        min_send_interval_ms: float = 18.0,
    ):
        self.nav_server = nav_server_ref
        self.stream_scale = float(np.clip(stream_scale, 0.1, 1.0))
        self.send_interval = max(1, int(send_interval))
        self.jpeg_quality = jpeg_quality

        self._lock = threading.Lock()
        self._slot = None
        self._event = threading.Event()

        self.running = True
        self.dropped = 0
        self.sent = 0

        self._send_ms_ema = 5.0
        self._min_interval = min_send_interval_ms / 1000.0
        self._last_send_time = 0.0

        self._send_thread = threading.Thread(
            target=self._send_loop,
            daemon=True,
            name="VideoSend",
        )
        self._send_thread.start()

        print(
            f"  视频流: encoder={_encoder_name}, "
            f"scale={self.stream_scale:.2f}, "
            f"quality={self.jpeg_quality}, "
            f"interval={self.send_interval}"
        )

    def submit(self, frame: np.ndarray, frame_idx: int, orig_w: int, orig_h: int):
        if self._send_ms_ema > 50.0:
            effective = max(self.send_interval, 3)
        elif self._send_ms_ema > 25.0:
            effective = max(self.send_interval, 2)
        else:
            effective = self.send_interval

        if frame_idx % effective != 0:
            return

        with self._lock:
            if self._slot is not None:
                self.dropped += 1

            self._slot = (frame, frame_idx, orig_w, orig_h)

        self._event.set()

    def _send_loop(self):
        while self.running:
            fired = self._event.wait(timeout=0.05)
            self._event.clear()

            if not fired:
                continue

            with self._lock:
                item, self._slot = self._slot, None

            if item is None:
                continue

            now = time.perf_counter()
            elapsed = now - self._last_send_time

            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            frame, frame_idx, orig_w, orig_h = item

            t0 = time.perf_counter()

            try:
                if self.stream_scale < 0.99:
                    nw = max(1, int(orig_w * self.stream_scale))
                    nh = max(1, int(orig_h * self.stream_scale))

                    send_frame = cuda_resize(
                        frame,
                        (nw, nh),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    send_frame, nw, nh = frame, orig_w, orig_h

                jpeg_bytes = encode_jpeg(send_frame, self.jpeg_quality)

                if jpeg_bytes:
                    self.nav_server.send_jpeg_video_frame(
                        jpeg_bytes,
                        frame_idx,
                        nw,
                        nh,
                    )
                    self.sent += 1
                    self._last_send_time = time.perf_counter()

            except Exception as e:
                if self.sent % 200 == 1:
                    print(f"⚠️ VideoSend: {e}")

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._send_ms_ema = 0.15 * dt_ms + 0.85 * self._send_ms_ema

    def shutdown(self):
        self.running = False
        self._event.set()

        if hasattr(self, "_send_thread"):
            self._send_thread.join(timeout=2)

        print(
            f"  视频流统计: encoder={_encoder_name} "
            f"sent={self.sent} drop={self.dropped} "
            f"latency≈{self._send_ms_ema:.1f}ms"
        )