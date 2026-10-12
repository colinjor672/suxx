import ctypes
import os
import signal
import subprocess
import time
import threading

from network.gst_utils import NvJpegEncoder
from network.shm_sync import ShmSyncPy

_godot_proc    = None
_godot_log     = None
_ffmpeg_proc   = None
_nvjpeg_enc    = None
_reader_thread = None
_reader_stop   = False
_frame_shm     = None  # ShmSyncPy Reader

FRAME_W      = 640
FRAME_H      = 360
FRAME_SIZE   = FRAME_W * FRAME_H * 4  # RGBA
FRAME_FPS    = 20

RTSP_PUSH_URL = "rtsp://127.0.0.1:15544/stream"

# 旧 /dev/shm 文件路径（清理用）
_OLD_SHM_PATHS = [
    "/dev/shm/godot_frame_ctrl",
    "/dev/shm/godot_frame_0.raw",
    "/dev/shm/godot_frame_1.raw",
    "/dev/shm/godot_input_ctrl",
    "/dev/shm/godot_input_0.raw",
    "/dev/shm/godot_input_1.raw",
]


def _start_ffmpeg_mux(rtsp_url: str, use_h264: bool = False) -> bool:
    """FFmpeg 推流模式：读 JPEG 字节流 → RTSP 或 RTMP

    use_h264=False: 纯封装 -c:v copy（零 CPU 编码，仅 RTSP 可用）
    use_h264=True : 转码 libx264 ultrafast
    rtmp:// 地址自动按 -f flv 封装并强制 libx264（FLV 容器不支持 MJPEG）
    """
    global _ffmpeg_proc

    is_rtmp = rtsp_url.startswith("rtmp://")
    if is_rtmp and not use_h264:
        print("[FFmpeg] RTMP 分发不支持 MJPEG，自动切换 libx264 转码")
        use_h264 = True

    # 输入端显式声明尺寸/帧率：pipe 探测不到 mjpeg 分辨率，
    # 否则 SDP 里带宽高信息缺失，订阅端报 "codec parameters ... unspec"
    input_args = [
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-video_size", f"{FRAME_W}x{FRAME_H}",
        "-framerate", str(FRAME_FPS),
        "-i", "pipe:0",
    ]
    if use_h264:
        output_args = [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-g", str(FRAME_FPS),
            "-pix_fmt", "yuv420p",
        ]
    else:
        output_args = ["-c:v", "copy"]

    if is_rtmp:
        mux_args = ["-f", "flv"]
    else:
        mux_args = ["-f", "rtsp", "-rtsp_transport", "tcp"]

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        *input_args,
        *output_args,
        *mux_args,
        rtsp_url,
    ]

    print(f"[FFmpeg] {'libx264 转码' if use_h264 else '纯封装 -c:v copy'} → "
          f"{'RTMP/FLV' if is_rtmp else 'RTSP'} {rtsp_url}")
    try:
        _ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except FileNotFoundError:
        print("[FFmpeg] 找不到 ffmpeg，跳过")
        return False
    except Exception as e:
        print(f"[FFmpeg] 启动失败: {e}")
        return False


def _frame_reader_loop():
    """semaphore 等待 Godot 合成帧 → nvjpegenc GPU 编码 → FFmpeg 推流"""
    global _reader_stop, _ffmpeg_proc, _nvjpeg_enc, _frame_shm

    last_frame_id = -1
    drop_count = 0
    push_count = 0
    enc_fail = 0
    timeout_ms = 200  # 200ms 超时

    while not _reader_stop:
        try:
            if not _frame_shm or not _frame_shm.is_open:
                time.sleep(0.1)
                continue

            # semaphore 阻塞等待新帧（替代文件轮询，CPU 零开销）
            if not _frame_shm.wait_for_new_frame(timeout_ms):
                continue

            frame_id = _frame_shm.get_frame_id()

            if frame_id == last_frame_id:
                continue

            if frame_id > last_frame_id + 1 and last_frame_id >= 0:
                drop_count += (frame_id - last_frame_id - 1)

            last_frame_id = frame_id

            # 读取 RGBA 数据（mmap 零拷贝）
            rgba_data = _frame_shm.get_data()
            if len(rgba_data) != FRAME_SIZE:
                continue

            # ★ nvjpegenc GPU 硬件编码：RGBA bytes → JPEG bytes
            if _nvjpeg_enc:
                jpeg_data = _nvjpeg_enc.encode_bytes(rgba_data)
                if not jpeg_data:
                    enc_fail += 1
                    if enc_fail <= 3:
                        print("[Reader] nvjpegenc 编码返回空", flush=True)
                    continue

                if _ffmpeg_proc and _ffmpeg_proc.poll() is None:
                    try:
                        _ffmpeg_proc.stdin.write(jpeg_data)
                        _ffmpeg_proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
            else:
                # 回退：直接写 RGBA 到 FFmpeg
                if _ffmpeg_proc and _ffmpeg_proc.poll() is None:
                    try:
                        _ffmpeg_proc.stdin.write(rgba_data)
                        _ffmpeg_proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass

            push_count += 1
            if push_count % 200 == 0:
                print(f"[Reader] 已推送 {push_count} 帧 "
                      f"(丢弃 {drop_count}, 编码失败 {enc_fail})")

        except Exception:
            time.sleep(0.01)


# Godot 进程名匹配串（pgrep -f 使用，需与导出的可执行文件名一致）
_GODOT_MATCH = "Visualization1"


def _signal_proc(pid: int, sig: int) -> None:
    """向 Godot 进程发信号。

    Godot 以 start_new_session=True 启动，是进程组组长，
    用 killpg 连同其所有子进程一起终止，避免只杀到包装进程而漏掉真正的窗口进程。
    防止误伤自身：跳过自己所属的进程组。
    """
    try:
        pgid = os.getpgid(pid)
        own_pgid = os.getpgid(0)
        if pgid > 0 and pgid != own_pgid and pgid != os.getpid():
            os.killpg(pgid, sig)
            return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _set_pdeathsig():
    """父进程（python 主程序）死亡时，内核自动给 Godot 发送 SIGTERM。

    无论主程序是 Ctrl+C 正常退出、崩溃还是被 kill -9 强杀，
    Godot 都会被内核自动终止，杜绝残留实例。仅 Linux 有效。
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


def _cleanup_stale_godot():
    """清理所有残留的 Godot 实例。

    多次启动或异常退出会留下孤儿实例，每个实例都会独立渲染并占用
    约一个 CPU 核，且相互竞争同一套共享内存信号量。
    启动前与退出时调用，保证同一时刻最多只有一个 Godot 在运行。
    """
    def _pgrep_pids():
        result = subprocess.run(
            ["pgrep", "-f", _GODOT_MATCH],
            capture_output=True,
        )
        if result.returncode != 0:
            return []
        return [
            int(pid)
            for pid in result.stdout.decode().strip().split()
            if pid.isdigit() and int(pid) != os.getpid()
        ]

    pids = _pgrep_pids()
    if not pids:
        return

    print(f"[Godot] 清理残留实例 {len(pids)} 个: {pids}")
    for pid in pids:
        _signal_proc(pid, signal.SIGTERM)

    # 最多等待 2s，仍未退出的强制 SIGKILL
    deadline = time.time() + 2.0
    while time.time() < deadline:
        pids = _pgrep_pids()
        if not pids:
            return
        time.sleep(0.1)

    for pid in pids:
        _signal_proc(pid, signal.SIGKILL)
    print(f"[Godot] 已强制结束残留实例: {pids}")


def start_godot(godot_exe_path: str, rtsp_push_url: str = None,
                wait_seconds: float = 4.0, use_h264: bool = False):
    global _godot_proc, _godot_log, _ffmpeg_proc, _nvjpeg_enc
    global _reader_thread, _reader_stop, _frame_shm

    if not godot_exe_path or not godot_exe_path.strip():
        print("[Godot] 未指定路径，跳过启动")
        return None

    if not os.path.isfile(godot_exe_path):
        raise FileNotFoundError(f"找不到 Godot 程序: {godot_exe_path}")

    # 启动前清理历史残留实例，防止多次启动累积（每个实例约一个 CPU 核）
    _cleanup_stale_godot()

    work_dir = os.path.dirname(os.path.abspath(godot_exe_path))
    push_url = rtsp_push_url or RTSP_PUSH_URL

    # 清理旧 /dev/shm 文件
    for p in _OLD_SHM_PATHS:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    # 清理旧 POSIX 共享内存（如果存在）
    for name in ["shm_frame", "shm_input"]:
        try:
            from ctypes import CDLL
            lib = CDLL("librt.so.1")
            lib.shm_unlink(name.encode())
        except Exception:
            pass

    # 1. 启动 nvjpegenc GPU 硬件编码器
    print("[NvJpeg] 初始化 GPU 硬件 JPEG 编码器...")
    _nvjpeg_enc = NvJpegEncoder(width=FRAME_W, height=FRAME_H, fps=FRAME_FPS, quality=80)
    if not _nvjpeg_enc.start():
        print("[NvJpeg] GPU 编码器启动失败，将回退到 FFmpeg MJPEG CPU 编码")
        _nvjpeg_enc = None

    # 2. 启动 FFmpeg 推流（默认纯封装 copy，可切 H.264 转码）
    _start_ffmpeg_mux(push_url, use_h264=use_h264)

    # 3. 打开 ShmSyncPy Reader（Godot 端是 Writer）
    #    等待 Godot 启动后再打开，因为 Godot 创建共享内存
    print("[ShmSync] 等待 Godot 创建共享内存...")

    # 4. 启动 Godot
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    env["__GL_SYNC_TO_VBLANK"] = "0"

    cmd = [godot_exe_path]

    print(f"[Godot] 启动: {godot_exe_path}")
    _godot_log = open("/tmp/godot_stream.log", "w")
    _godot_proc = subprocess.Popen(
        cmd,
        cwd=work_dir,
        stdout=_godot_log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        preexec_fn=_set_pdeathsig,  # 主程序死亡时内核自动终止 Godot
    )

    time.sleep(wait_seconds)

    if _godot_proc.poll() is not None:
        _godot_log.flush()
        print(f"[Godot] 进程退出，退出码={_godot_proc.returncode}")
        return None

    # 5. 等待 Godot 创建共享内存后，打开 Reader
    for attempt in range(30):
        try:
            _frame_shm = ShmSyncPy("shm_frame", FRAME_SIZE, is_writer=False)
            print("[ShmSync] Reader 已连接 shm_frame (mmap+semaphore)")
            break
        except OSError:
            if attempt == 0:
                print("[ShmSync] 等待 Godot 创建 shm_frame...")
            time.sleep(0.5)
    else:
        print("[ShmSync] ⚠ 无法连接 shm_frame，回退到文件轮询模式")
        _frame_shm = None

    # 6. 启动帧读取线程
    _reader_stop = False
    _reader_thread = threading.Thread(target=_frame_reader_loop, daemon=True)
    _reader_thread.start()

    print(f"[Godot] PID={_godot_proc.pid}  窗口已在显示器上显示")
    print(f"[Godot] RTSP 推流: {push_url}")
    print("[Godot] 日志: /tmp/godot_stream.log")
    return _godot_proc


def stop_godot() -> None:
    global _godot_proc, _godot_log, _ffmpeg_proc, _nvjpeg_enc
    global _reader_thread, _reader_stop, _frame_shm

    _reader_stop = True

    # 关闭共享内存
    if _frame_shm:
        _frame_shm.close()
        _frame_shm = None

    # 停止 nvjpegenc 编码器
    if _nvjpeg_enc:
        _nvjpeg_enc.stop()
        _nvjpeg_enc = None

    # 停止 FFmpeg
    if _ffmpeg_proc and _ffmpeg_proc.poll() is None:
        print("[FFmpeg] 停止推流")
        try:
            _ffmpeg_proc.stdin.close()
        except Exception:
            pass
        _ffmpeg_proc.terminate()
        try:
            _ffmpeg_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ffmpeg_proc.kill()
        _ffmpeg_proc = None

    if _godot_proc and _godot_proc.poll() is None:
        print(f"[Godot] 停止 PID={_godot_proc.pid}")
        _godot_proc.terminate()
        try:
            _godot_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _godot_proc.kill()
        _godot_proc = None

    if _godot_log:
        _godot_log.close()
        _godot_log = None

    # 兜底：清理所有残留 Godot 实例（不依赖自跟踪的 _godot_proc），
    # 防止异常退出留下的孤儿实例持续烧 CPU。
    _cleanup_stale_godot()

    print("[Godot] 已停止")