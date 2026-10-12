#!/usr/bin/env python3
"""
H.264 解码桥接进程（独立运行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
从 stdin 读取 H.264 byte-stream → nvv4l2decoder GPU 硬解
→ nvvidconv 零 CPU 色彩转换 → RGBA 写入共享内存
→ Godot 从 /dev/shm/godot_video.raw 读取

用法:
  python3 h264_shm_bridge.py --width 1280 --height 720 --fps 30

协议（stdin 二进制）:
  [4 字节 frame_id 大端] [4 字节 data_len 大端] [data_len 字节 H.264 NAL]
"""

import sys
import os
import struct
import argparse
import signal

# 确保 gst_utils 可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gst_utils import (
    IS_JETSON, HAS_GSTREAMER,
    SHM_VIDEO_PATH, SHM_META_PATH,
)

running = True

def signal_handler(sig, frame):
    global running
    running = False
    print("\n[H264Bridge] 收到退出信号")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    parser = argparse.ArgumentParser(description="H.264 解码桥接")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps",    type=int, default=30)
    args = parser.parse_args()

    width, height, fps = args.width, args.height, args.fps

    print(f"[H264Bridge] 启动 {width}×{height} @ {fps}fps", flush=True)

    # 确保 /dev/shm 存在
    os.makedirs("/dev/shm", exist_ok=True)

    if IS_JETSON and HAS_GSTREAMER:
        _run_gstreamer(width, height, fps)
    else:
        _run_avdec(width, height, fps)


def _run_gstreamer(width: int, height: int, fps: int):
    """nvv4l2decoder GPU 硬解 + nvvidconv 零 CPU 色彩转换"""
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst, GLib
        Gst.init(None)
    except ImportError:
        print("[H264Bridge] ✗ pygobject 不可用，请安装: pip install pygobject", flush=True)
        sys.exit(1)

    pipeline_str = (
        f"appsrc name=src format=time is-live=true do-timestamp=true ! "
        f"video/x-h264,width={width},height={height},"
        f"framerate={fps}/1,stream-format=byte-stream,alignment=au ! "
        f"h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw,format=RGBA ! "
        f"appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
    )

    print(f"[H264Bridge] GStreamer: nvv4l2decoder → nvvidconv (零 CPU)", flush=True)

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        appsrc   = pipeline.get_by_name("src")
        appsink  = pipeline.get_by_name("sink")

        latest_frame_id = 0
        latest_ts = 0.0

        def on_new_sample(sink):
            nonlocal latest_frame_id, latest_ts
            try:
                sample = sink.emit("pull-sample")
                if not sample:
                    return Gst.FlowReturn.ERROR

                buf = sample.get_buffer()
                result, map_info = buf.map(Gst.MapFlags.READ)
                if not result:
                    return Gst.FlowReturn.ERROR

                rgba_data = bytes(map_info.data)
                buf.unmap(map_info)

                # 写入共享内存
                try:
                    with open(SHM_VIDEO_PATH, "wb") as f:
                        f.write(rgba_data)
                    meta = struct.pack("!I I I d", latest_frame_id, width, height, latest_ts)
                    with open(SHM_META_PATH, "wb") as f:
                        f.write(meta)
                except Exception:
                    pass

                return Gst.FlowReturn.OK
            except Exception as e:
                print(f"[H264Bridge] 帧回调异常: {e}", flush=True)
                return Gst.FlowReturn.ERROR

        appsink.connect("new-sample", on_new_sample)
        pipeline.set_state(Gst.State.PLAYING)

        # GLib 主循环在后台线程
        loop = GLib.MainLoop()
        import threading
        thread = threading.Thread(target=loop.run, daemon=True)
        thread.start()

        print("[H264Bridge] ✓ 管线就绪，等待 stdin H.264 数据...", flush=True)

        # 主循环：从 stdin 读取 H.264 帧
        while running:
            # 读取帧头: [frame_id:4] [data_len:4]
            header = sys.stdin.buffer.read(8)
            if len(header) < 8:
                if len(header) == 0:
                    break  # EOF
                continue

            frame_id = struct.unpack("!I", header[:4])[0]
            data_len = struct.unpack("!I", header[4:8])[0]

            if data_len <= 0 or data_len > 10_000_000:
                continue

            h264_data = sys.stdin.buffer.read(data_len)
            if len(h264_data) < data_len:
                break

            latest_frame_id = frame_id
            latest_ts = time.time()

            # 推入 GStreamer 管线
            try:
                gst_buf = Gst.Buffer.new_allocate(None, len(h264_data), None)
                gst_buf.fill(0, h264_data)
                appsrc.emit("push-buffer", gst_buf)
            except Exception as e:
                print(f"[H264Bridge] push 异常: {e}", flush=True)

        pipeline.set_state(Gst.State.NULL)
        loop.quit()
        print("[H264Bridge] 已停止", flush=True)

    except Exception as e:
        print(f"[H264Bridge] ✗ GStreamer 异常: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _run_avdec(width: int, height: int, fps: int):
    """PC 回退：avdec_h264 CPU 软解"""
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst, GLib
        Gst.init(None)
    except ImportError:
        print("[H264Bridge] ✗ pygobject 不可用", flush=True)
        sys.exit(1)

    pipeline_str = (
        f"appsrc name=src format=time is-live=true do-timestamp=true ! "
        f"video/x-h264,width={width},height={height},"
        f"framerate={fps}/1,stream-format=byte-stream,alignment=au ! "
        f"h264parse ! avdec_h264 ! "
        f"videoconvert ! video/x-raw,format=RGBA ! "
        f"appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
    )

    print(f"[H264Bridge] GStreamer: avdec_h264 (CPU)", flush=True)

    # 与 _run_gstreamer 相同的逻辑，只是管线不同
    import time as _time
    latest_frame_id = 0
    latest_ts = 0.0

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        appsrc   = pipeline.get_by_name("src")
        appsink  = pipeline.get_by_name("sink")

        def on_new_sample(sink):
            nonlocal latest_frame_id, latest_ts
            try:
                sample = sink.emit("pull-sample")
                if not sample:
                    return Gst.FlowReturn.ERROR
                buf = sample.get_buffer()
                result, map_info = buf.map(Gst.MapFlags.READ)
                if not result:
                    return Gst.FlowReturn.ERROR
                rgba_data = bytes(map_info.data)
                buf.unmap(map_info)
                try:
                    with open(SHM_VIDEO_PATH, "wb") as f:
                        f.write(rgba_data)
                    meta = struct.pack("!I I I d", latest_frame_id, width, height, latest_ts)
                    with open(SHM_META_PATH, "wb") as f:
                        f.write(meta)
                except Exception:
                    pass
                return Gst.FlowReturn.OK
            except Exception:
                return Gst.FlowReturn.ERROR

        appsink.connect("new-sample", on_new_sample)
        pipeline.set_state(Gst.State.PLAYING)

        loop = GLib.MainLoop()
        import threading
        thread = threading.Thread(target=loop.run, daemon=True)
        thread.start()

        print("[H264Bridge] ✓ 管线就绪 (CPU)，等待 stdin H.264 数据...", flush=True)

        while running:
            header = sys.stdin.buffer.read(8)
            if len(header) < 8:
                break
            frame_id = struct.unpack("!I", header[:4])[0]
            data_len = struct.unpack("!I", header[4:8])[0]
            if data_len <= 0 or data_len > 10_000_000:
                continue
            h264_data = sys.stdin.buffer.read(data_len)
            if len(h264_data) < data_len:
                break
            latest_frame_id = frame_id
            latest_ts = _time.time()
            try:
                gst_buf = Gst.Buffer.new_allocate(None, len(h264_data), None)
                gst_buf.fill(0, h264_data)
                appsrc.emit("push-buffer", gst_buf)
            except Exception:
                pass

        pipeline.set_state(Gst.State.NULL)
        loop.quit()
        print("[H264Bridge] 已停止", flush=True)

    except Exception as e:
        print(f"[H264Bridge] ✗ avdec 异常: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    import time
    main()