"""RTSP 视频发布服务: 把本地视频文件当成一台 RTSP 摄像机发布出去。

用途: 数据集回放时现场摄像机不可用, 但下游程序(scripts/raw_sensor_viewer)只会从
RTSP 拉流。本模块在本地起一个轻量 RTSP 服务, 把 video.mp4 按真实帧率循环推流,
URL / 端口 / 账号密码都可以和现场摄像机保持一致, 下游程序不需要改动。

特点:
  * 两种发布格式: --codec h264(默认, RTP/H264, RFC 6184)与 --codec jpeg(RTP/JPEG,
    RFC 2435)。下游只要 H.264/x264 时用前者; JPEG 通路每帧自包含, 后连立刻能解,
    留作没有 ffmpeg 时的退路;
  * h264 需要一个 ffmpeg 可执行文件做编码(纯 OpenCV 编不出 H.264: 它的 libopenh264
    在 Windows 预编译包里跑不起来), 编码是"把帧喂进子进程 stdin, 从 stdout 收 Annex-B",
    不产生中间文件; 只依赖 ffmpeg 这一个外部程序, 不需要 mediamtx;
  * 支持 RTSP over TCP(interleaved)与 RTSP over UDP 两种传输, 客户端自动协商;
  * 支持 Digest / Basic 认证(用户名密码与摄像机一致, 例如 admin/gzsia2011);
  * 路径可自定义, 例如 /Streaming/Channels/101(海康摄像机主码流路径);
  * 没有客户端连接时不解码不编码, 不占 CPU; 播放时钟从第一个客户端 PLAY 开始。

单独运行(不跑回放, 只发视频):
    python scripts\replay\rtsp_video_server.py --video data\testdata20260911_180053\video.mp4
    python scripts\replay\rtsp_video_server.py --video video.mp4 --port 5540

验证发出去的是 x264(不是 mjpeg):
    ffprobe -rtsp_transport tcp -v error -show_entries stream=codec_name ^
            -of csv=p=0 rtsp://admin:gzsia2011@127.0.0.1:554/Streaming/Channels/101
    期望输出 h264; 若输出 mjpeg 说明客户端连的还是 --codec jpeg 那套 SDP。
    也可以直接看 SDP: ffprobe 前面加 -v debug, 找 a=rtpmap:96 H264/90000。

关于地址: 现场摄像机是 192.168.1.164, 本机是 192.168.1.8, 所以视频流发布在本机上,
路径 / 账号 / 密码 / 端口都保持和摄像机一致, 只有主机地址不同:
    Viewer 里填 rtsp://admin:gzsia2011@127.0.0.1:554/Streaming/Channels/101
如果某个下游程序把 192.168.1.164 写死了, 且现场摄像机当前不在线, 可以用管理员权限
把该地址临时挂到本机网卡(摄像机一上线就要删掉, 否则地址冲突):
    netsh interface ipv4 add address "WLAN" 192.168.1.164 255.255.255.255
    netsh interface ipv4 delete address "WLAN" 192.168.1.164
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - 运行环境缺依赖时给出明确提示
    cv2 = None


# ============================================================
#  常量
# ============================================================
DEFAULT_FPS = 25.0            # 视频文件读不出帧率时的兜底值
DEFAULT_QUALITY = 80          # JPEG 质量 (1~100)
RTP_PAYLOAD_TYPE_JPEG = 26    # RFC 2435 RTP/JPEG
RTP_PAYLOAD_TYPE_H264 = 96    # RFC 6184 RTP/H264, 动态 payload type
RTP_CLOCK = 90000             # RTP 时钟频率(JPEG 与 H264 都是 90kHz)
DEFAULT_MAX_PAYLOAD = 1400    # 单个 RTP 包的负载上限(字节)
DEFAULT_REALM = "IP Camera"
DEFAULT_PATH = "/Streaming/Channels/101"
DEFAULT_CRF = 23              # x264 质量: 越小越清晰, 18~28 是常用区间
DEFAULT_PRESET = "veryfast"   # x264 速度档: ultrafast ... medium
FU_A_NAL_TYPE = 28            # RFC 6184 分片包用的 NAL 类型


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _http_date() -> str:
    return time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())


# ============================================================
#  视频帧源: 按帧号取图并编码为 JPEG 或 H.264
# ============================================================
class VideoFileSource:
    """把一个本地视频文件包装成"可按帧号取帧"的源(支持循环)。"""

    def __init__(self, path: str, fps: Optional[float] = None,
                 quality: int = DEFAULT_QUALITY, loop: bool = True,
                 verbose: bool = True, codec: str = "h264",
                 crf: int = DEFAULT_CRF, gop: Optional[int] = None,
                 bitrate: Optional[int] = None, preset: str = DEFAULT_PRESET,
                 ffmpeg: Optional[str] = None) -> None:
        if cv2 is None:
            raise RuntimeError("缺少 opencv-python (cv2), 无法读取视频文件")
        if codec not in ("h264", "jpeg"):
            raise ValueError(f"不支持的编码: {codec} (只认 h264 / jpeg)")
        self.path = str(path)
        self.loop = bool(loop)
        self.quality = int(quality)
        self.codec = codec
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self.path}")

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        src_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = float(fps) if fps else (src_fps if src_fps > 1.0 else DEFAULT_FPS)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._last_index = -1
        self._lock = threading.Lock()

        # gop = 两个 IDR 之间的帧数: 决定后连客户端最多要补发多少帧, 以及丢帧后
        # 最快多久能恢复干净画面。默认 2 秒。
        self.gop = int(gop) if gop else max(2, int(round(self.fps * 2)))
        if codec == "h264":
            if self.width % 2 or self.height % 2:
                raise ValueError(f"x264 要求偶数分辨率, 当前 {self.width}x{self.height}")
            self._enc = _H264Encoder(
                self.width, self.height, self.fps, crf=crf, gop=self.gop,
                bitrate=bitrate, preset=preset, verbose=verbose,
                ffmpeg_path=_find_ffmpeg(ffmpeg))
        else:
            self.gop = 1          # JPEG 每帧自包含, 没有 GOP 概念
            self._enc = _JpegEncoder(self.quality)

        if verbose:
            dur = self.frame_count / self.fps if self.fps else 0.0
            extra = ("JPEG 质量=%d" % self.quality if self.codec == "jpeg"
                     else "x264 crf=%d gop=%d preset=%s" % (crf, self.gop, preset))
            print(f"[RTSP] 视频源: {Path(self.path).name} "
                  f"{self.width}x{self.height} @ {self.fps:g}fps, "
                  f"{self.frame_count} 帧 ({dur:.1f}s), {self.codec.upper()}, {extra}")

    _GRAB_SKIP_MAX = 32       # 往前跨这么多帧以内都用 grab 跳, 不 seek

    def frame_at(self, index: int) -> Optional["np.ndarray"]:
        """取第 index 帧(循环时自动取模), 失败返回 None。"""
        n = self.frame_count
        if n <= 0:
            idx = 0
        elif self.loop:
            idx = int(index) % n
        elif index >= n:
            return None
        else:
            idx = int(index)

        with self._lock:
            if idx != self._last_index + 1:
                # seek 会冲刷解码器、从上一个关键帧整段重解(1080p 实测一次 ~0.1s)。
                # 泵一旦被它拖慢, 下一轮的跨度就更大、又要 seek, 帧率直接塌成个位数,
                # 所以小跨度一律用 grab() 顺序跳过去(只解码不取图, 便宜一个数量级)。
                gap = idx - (self._last_index + 1) if self._last_index >= 0 else -1
                if 0 < gap <= self._GRAB_SKIP_MAX:
                    for _ in range(gap):
                        if not self._cap.grab():
                            self._last_index = -1
                            break
                        self._last_index += 1
                    if self._last_index + 1 != idx:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        self._last_index = idx - 1
                else:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    self._last_index = idx - 1
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # 个别文件 seek 后首帧读取失败, 再试一次
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self._last_index = idx - 1
                ok, frame = self._cap.read()
            self._last_index = idx if (ok and frame is not None) else -1
        return frame if (ok and frame is not None) else None

    # ---------- 编码通路(交给 _JpegEncoder / _H264Encoder) ----------
    def submit(self, frame: "np.ndarray", index: int) -> None:
        self._enc.submit(frame, index)

    def poll(self, timeout: float = 0.0) -> Optional["_AU"]:
        return self._enc.poll(timeout)

    def restart_encoder(self) -> None:
        self._enc.restart()

    def parameter_sets(self) -> Tuple[Optional[bytes], Optional[bytes]]:
        return self._enc.parameter_sets()

    def close(self) -> None:
        with self._lock:
            try:
                self._cap.release()
            except Exception:
                pass
        self._enc.close()


# ============================================================
#  RTP 打包 (RFC 2435 RTP/JPEG)
# ============================================================
def _rtp_header(seq: int, timestamp: int, ssrc: int, marker: bool,
                payload_type: int = RTP_PAYLOAD_TYPE_JPEG) -> bytes:
    """12 字节 RTP 头 (V=2, 无扩展, 无 CSRC)。"""
    return struct.pack(
        "!BBHII",
        0x80,                                        # V=2, P=0, X=0, CC=0
        (0x80 if marker else 0x00) | (payload_type & 0x7F),
        seq & 0xFFFF,
        timestamp & 0xFFFFFFFF,
        ssrc & 0xFFFFFFFF,
    )


_SOF_MARKERS = (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB)


def parse_jpeg_frame(jpeg: bytes) -> Tuple[bytes, int, bytes]:
    """按 RFC 2435 的需要拆解一帧 JPEG。

    返回 (量化表块, 类型, 熵编码数据):
      * 量化表块: 两条 8bit 量化表(亮度, 色度)各 64 字节, 共 128 字节;
      * 类型: 1=YCbCr 4:2:0, 0=YCbCr 4:2:2(与 SOF 里的采样因子对应);
      * 熵编码数据: SOS 段之后到 EOI 之前的扫描数据。
    接收端(ffmpeg)会自己生成 SOI/DQT/DHT/SOF/SOS 头, 只需要这两样东西。
    """
    if len(jpeg) < 4 or jpeg[0] != 0xFF or jpeg[1] != 0xD8:
        raise ValueError("不是 JPEG 数据(缺少 SOI)")
    qtables: Dict[int, bytes] = {}
    y_sampling = (2, 2)
    pos = 2
    total = len(jpeg)
    while pos + 3 < total:
        if jpeg[pos] != 0xFF:
            pos += 1
            continue
        marker = jpeg[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        seg_len = int.from_bytes(jpeg[pos + 2:pos + 4], "big")
        payload = jpeg[pos + 4:pos + 2 + seg_len]
        if marker == 0xDB:                      # DQT
            o = 0
            while o < len(payload):
                pq_tq = payload[o]
                o += 1
                size = 128 if (pq_tq >> 4) else 64
                qtables[pq_tq & 0x0F] = payload[o:o + size]
                o += size
        elif marker in _SOF_MARKERS and len(payload) >= 8:   # SOFn
            hv = payload[7]                     # 第一个分量(Y)的采样因子
            y_sampling = (hv >> 4, hv & 0x0F)
        elif marker == 0xDA:                    # SOS: 后面就是扫描数据
            scan = jpeg[pos + 2 + seg_len:]
            if scan.endswith(b"\xff\xd9"):
                scan = scan[:-2]
            qblock = b"".join(qtables.get(i, b"") for i in (0, 1))
            if len(qblock) != 128:
                raise ValueError("量化表解析失败(需要亮度/色度两条表)")
            jtype = 1 if y_sampling == (2, 2) else 0
            return qblock, jtype, scan
        pos += 2 + seg_len
    raise ValueError("JPEG 里没有找到 SOS 段")


def packetize_jpeg(jpeg: bytes, timestamp: int, seq: int, ssrc: int,
                   width: int, height: int,
                   max_payload: int = DEFAULT_MAX_PAYLOAD) -> List[bytes]:
    """把一帧 JPEG 切成 RTP/JPEG(RFC 2435)包。

    8 字节 RFC 2435 头: 类型特定(1, 必须为 0) + 分片偏移(3, 单位: 字节) +
                        类型(1) + Q(1) + 宽/8(1) + 高/8(1)
    Q=255 表示量化表随包下发, 因此第一包只装 4+128 字节的量化表, 偏移为 0;
    之后每个包的偏移 = 已发出的扫描数据字节数(接收端就是这么核对的)。
    """
    qblock, jtype, scan = parse_jpeg_frame(jpeg)
    w8 = max(1, min(255, (int(width) + 7) // 8)) & 0xFF
    h8 = max(1, min(255, (int(height) + 7) // 8)) & 0xFF
    step = max(64, (int(max_payload) // 8) * 8)

    def wrap(offset: int, payload: bytes, marker: bool) -> bytes:
        header = (bytes((0,)) + int(offset).to_bytes(3, "big") +
                  bytes((jtype, 255, w8, h8)))
        return _rtp_header(seq, timestamp, ssrc, marker) + header + payload

    packets: List[bytes] = []
    # 第一包: 偏移 0, 装 4+128 字节的量化表块 + 一段扫描数据。
    # (接收端把偏移 0 视为帧头包, 所以量化表必须和扫描数据同包发出)
    table_block = struct.pack("!BBH", 0, 0, len(qblock)) + qblock
    first_data = max(0, step - len(table_block))
    head = scan[:first_data]
    pos = len(head)
    packets.append(wrap(0, table_block + head, pos >= len(scan)))
    seq = (seq + 1) & 0xFFFF
    offset = pos
    while pos < len(scan):
        chunk = scan[pos:pos + step]
        pos += len(chunk)
        packets.append(wrap(offset, chunk, pos >= len(scan)))
        seq = (seq + 1) & 0xFFFF
        offset += len(chunk)
    return packets


# ============================================================
#  H.264: 访问单元 / Annex-B 解析 / ffmpeg 编码器 / RTP 打包
# ============================================================
class _AU:
    """一个访问单元 = 源视频里的一帧, nals 是去掉起始码后的 NAL 列表。"""

    __slots__ = ("nals", "idr", "index", "ts")

    def __init__(self, nals: List[bytes], idr: bool, index: int, ts: int = 0) -> None:
        self.nals = nals
        self.idr = idr          # 能否作为解码起点(IDR 帧 / JPEG 帧)
        self.index = index      # 源视频帧号, -1 表示对不上号
        self.ts = ts            # RTP 时间戳(90kHz), 由泵填


def _nal_type(nal: bytes) -> int:
    return nal[0] & 0x1F if nal else 0


def _first_mb_in_slice(nal: bytes) -> int:
    """取 slice_header 的第一个语法元素 first_mb_in_slice(Exp-Golomb ue(v))。

    一帧被切成多条 slice 时, 只有第一条的值为 0, 后面的都非零 —— 靠它区分
    "同一帧的下一片" 和 "下一帧的第一片"。读不出来就返回 0, 当作新帧开头。
    """
    nbits = len(nal) * 8
    bit = 8                       # 跳过 1 字节的 NAL 头
    zeros = 0
    while bit < nbits and not ((nal[bit >> 3] >> (7 - (bit & 7))) & 1):
        zeros += 1
        bit += 1
    if bit >= nbits or zeros > 24:
        return 0
    bit += 1                      # ue(v) 中间那个分隔用的 1
    val = 0
    for _ in range(zeros):
        if bit >= nbits:
            return 0
        val = (val << 1) | ((nal[bit >> 3] >> (7 - (bit & 7))) & 1)
        bit += 1
    return (1 << zeros) - 1 + val


def _quiet_close(stream) -> None:
    """关掉本线程自己持有的那个管道(只允许持有它的线程调用)。"""
    try:
        if stream is not None:
            stream.close()
    except Exception:
        pass


def _find_ffmpeg(explicit: Optional[str] = None) -> str:
    """找 ffmpeg 可执行文件: --ffmpeg 指定 > PATH > 常见安装位置。"""
    if explicit:
        found = shutil.which(explicit)
        if found:
            return found
        if os.path.isfile(explicit):
            return explicit
        raise RuntimeError("找不到 ffmpeg: --ffmpeg 给的路径不存在 (%s)" % explicit)
    for name in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(name)
        if found:
            return found
    for cand in (r"C:\ffmpeg\bin\ffmpeg.exe",
                 os.path.expanduser(r"~\scoop\shims\ffmpeg.exe"),
                 "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"):
        if os.path.isfile(cand):
            return cand
    raise RuntimeError("找不到 ffmpeg。装一个并加进 PATH(或用 --ffmpeg 指定路径), "
                       "例如: winget install Gyan.FFmpeg; 或改用 --codec jpeg")


class _AnnexBParser:
    """把 Annex-B 字节流增量切成 NAL(去掉起始码)。

    当前 NAL 的长度要靠"下一个起始码"才能定下来, 所以每个 AU 最后那个 NAL 要等下一
    批数据到了才吐得出来 —— 整条通路因此固定晚一帧。
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    @staticmethod
    def _find_sc(buf: bytearray, start: int) -> Optional[Tuple[int, int]]:
        """返回 (起始码在 buf 里的起点, 长度 3 或 4)。"""
        i = buf.find(b"\x00\x00\x01", max(0, start))
        if i < 0:
            return None
        if i > start and buf[i - 1] == 0:
            return (i - 1, 4)
        return (i, 3)

    def feed(self, data: bytes) -> List[bytes]:
        self._buf.extend(data)
        nals: List[bytes] = []
        while True:
            head = self._find_sc(self._buf, 0)
            if head is None:
                # 一个起始码都没找到: 这些字节不属于任何 NAL, 但结尾可能是被切开的
                # 起始码(00 / 00 00), 留着和下一批数据拼
                keep = 2 if self._buf[-2:] == b"\x00\x00" else \
                    (1 if self._buf[-1:] == b"\x00" else 0)
                del self._buf[:len(self._buf) - keep]
                break
            if head[0] > 0:
                del self._buf[:head[0]]  # 起始码之前残留的字节不属于任何 NAL
            body = head[1]
            nxt = self._find_sc(self._buf, body)
            if nxt is None:
                break                    # 这个 NAL 还没写完, 等下一块数据
            # 注意: 不能 rstrip 掉结尾的 0x00 —— H.264 的 RBSP 末尾本来就是"结束位 1 +
            # 补零到字节边界", 这些零是 NAL 的一部分, 砍掉等于把切片截短, 解码器只能
            # 隐藏出错宏块, 整帧变成一片死灰。起始码自带的前导零由 _find_sc 处理。
            nal = bytes(self._buf[body:nxt[0]])
            del self._buf[:nxt[0]]
            if nal:
                nals.append(nal)
        return nals


def packetize_h264(nals: List[bytes], timestamp: int, seq: int, ssrc: int,
                   max_payload: int = DEFAULT_MAX_PAYLOAD) -> List[bytes]:
    """按 RFC 6184 把一帧的 NAL 列表切成 RTP/H264 包。

    * NAL 装得进 MTU: 单 NAL 包, RTP 负载就是这条 NAL 本身;
    * 装不进: FU-A 分片(FU indicator 保留原 NAL 头的 F/NRI 位、类型写 28,
      FU header 用 S/E 标出这一帧的第一片/最后一片);
    * 每个 AU 的最后一个包 marker=1, 客户端据此判断"一帧齐了可以解码"。

    不做 STAP-A 聚合: 每包都带完整 NAL 头更 straightforward, 且收端两种都支持。
    """
    # 留 2 字节给 FU indicator/header, 保证任何一个包的业务负载都不超过 max_payload
    step = max(144, int(max_payload) - 2)
    packets: List[bytes] = []
    for ni, nal in enumerate(nals):
        if not nal:
            continue
        last_nal = ni == len(nals) - 1
        if len(nal) <= step:
            packets.append(_rtp_header(seq, timestamp, ssrc, last_nal,
                                       RTP_PAYLOAD_TYPE_H264) + nal)
            seq = (seq + 1) & 0xFFFF
            continue
        nal_type = _nal_type(nal)
        body = nal[1:]
        chunks = [body[o:o + step] for o in range(0, len(body), step)] or [b""]
        fu_ind = bytes([(nal[0] & 0xE0) | FU_A_NAL_TYPE])
        for fi, chunk in enumerate(chunks):
            fu_hdr = bytes([((0x80 if fi == 0 else 0) |
                             (0x40 if fi == len(chunks) - 1 else 0) | nal_type)])
            packets.append(_rtp_header(seq, timestamp, ssrc,
                                       last_nal and fi == len(chunks) - 1,
                                       RTP_PAYLOAD_TYPE_H264) + fu_ind + fu_hdr + chunk)
            seq = (seq + 1) & 0xFFFF
    return packets


class _JpegEncoder:
    """帧 -> JPEG。每帧自包含, 所以 idr 恒为 True(任何时候都能独立解码)。"""

    def __init__(self, quality: int) -> None:
        self._params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(quality)))]
        self._aus: deque = deque(maxlen=8)

    def submit(self, frame: "np.ndarray", index: int) -> None:
        ok, buf = cv2.imencode(".jpg", frame, self._params)
        if ok:
            self._aus.append(_AU([buf.tobytes()], True, int(index)))

    def poll(self, timeout: float = 0.0) -> Optional[_AU]:
        try:
            return self._aus.popleft()
        except IndexError:
            return None

    def restart(self) -> None:
        self._aus.clear()

    def parameter_sets(self) -> Tuple[Optional[bytes], Optional[bytes]]:
        return (None, None)

    def close(self) -> None:
        self._aus.clear()


class _H264Encoder:
    """常驻 ffmpeg 子进程: BGR24 裸帧进 stdin, H.264 Annex-B 出 stdout。

    写线程喂 stdin 的队列一满就丢最旧的**输入**帧(而不是让推流线程等编码器), 这样
    慢机器上表现为帧率下降, 输出流本身仍然连续、不会花屏。

    profile-level-id / sprop-parameter-sets 要在客户端 DESCRIBE 时就能给出, 所以构造
    时先用两帧空白帧把 SPS/PPS 骗出来, 然后把编码器重启成干净状态。
    """

    def __init__(self, width: int, height: int, fps: float, crf: int = DEFAULT_CRF,
                 gop: int = 50, bitrate: Optional[int] = None,
                 preset: str = DEFAULT_PRESET, ffmpeg_path: str = "ffmpeg",
                 verbose: bool = True) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps else DEFAULT_FPS
        self.crf = int(crf)
        self.gop = max(2, int(gop))
        self.bitrate = int(bitrate) if bitrate else None
        self.preset = preset
        self.ffmpeg_path = ffmpeg_path
        self.verbose = verbose
        self.dropped_inputs = 0
        self.error_lines: deque = deque(maxlen=30)

        self._inbox: deque = deque(maxlen=4)          # 待写入 stdin 的帧
        self._fed: deque = deque(maxlen=64)           # 已写入 stdin、还没配上 AU 的帧号
        self._aus: deque = deque(maxlen=max(8, self.gop * 2))
        self._in_event = threading.Event()
        self._out_event = threading.Event()
        self._sps: Optional[bytes] = None
        self._pps: Optional[bytes] = None
        self._lock = threading.Lock()
        self._gen = 0                                 # 换进程的次数(线程靠它重新绑定)
        self._dead = False                            # ffmpeg 真的没了, 不再喂帧
        self._running = True
        self._proc: Optional[subprocess.Popen] = None
        self._threads: List[threading.Thread] = []

        self._spawn()
        self._start_threads()                          # 线程只建一次, restart 不重建
        self._prime()

    # ---------- 进程 ----------
    def _cmd(self) -> List[str]:
        cmd = [self.ffmpeg_path, "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "%dx%d" % (self.width, self.height),
               "-r", "%g" % self.fps, "-i", "pipe:",
               "-an", "-c:v", "libx264", "-preset", self.preset,
               "-tune", "zerolatency", "-profile:v", "baseline", "-bf", "0",
               "-pix_fmt", "yuv420p", "-g", str(self.gop),
               # 单线程 => 一帧只有一条 slice。zerolatency 默认开 sliced-threads,
               # 会把一帧切成十几条, Jetson 的 NVDEC 逐 slice 提交解码, 能慢一个量级。
               "-threads", "1",
               "-x264-params",
               "keyint=%d:min-keyint=%d:scenecut=0:open-gop=0:aud=1"
               % (self.gop, self.gop)]
        if self.bitrate:
            cmd += ["-b:v", str(self.bitrate), "-maxrate", str(self.bitrate),
                    "-bufsize", str(self.bitrate * 2)]
        else:
            cmd += ["-crf", str(self.crf)]
        cmd += ["-flush_packets", "1", "-f", "h264", "pipe:1"]
        return cmd

    def _spawn(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._proc = subprocess.Popen(self._cmd(), stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          bufsize=0, creationflags=flags)
        except (OSError, ValueError) as exc:
            raise RuntimeError("ffmpeg 启动失败: %s (命令: %s)"
                               % (exc, " ".join(self._cmd()))) from exc

    def _start_threads(self) -> None:
        for target, name in ((self._write_loop, "x264-writer"),
                             (self._read_loop, "x264-reader"),
                             (self._drain_stderr, "x264-stderr")):
            th = threading.Thread(target=target, name=name, daemon=True)
            th.start()
            self._threads.append(th)

    def _terminate(self, proc: Optional[subprocess.Popen]) -> None:
        """收掉换代前的旧进程: 只 kill, 不碰它的任何管道。

        读/写线程此刻正阻塞在旧进程的 stdout/stdin 上, 从本线程 close() 一个正在被
        读的管道句柄, Windows 上会让那个线程永远醒不过来(新进程就没人读了)。进程一死,
        OS 自己关掉它那端, 阻塞的 read 返回 EOF、write 报 OSError, 线程据此重新绑定。
        """
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=1.0)
        except Exception:
            pass

    def _kill_proc(self) -> None:
        proc, self._proc = self._proc, None
        self._gen += 1
        self._terminate(proc)

    # ---------- 三个后台线程(全程只有这一组, 靠 _gen 感知"换了新进程"后重新绑定) ----------
    def _write_loop(self) -> None:
        gen, proc = -1, None
        while self._running:
            if gen != self._gen:
                new = self._proc
                if new is None:
                    time.sleep(0.01)     # 还没换好进程: 保持旧绑定, 下一轮再试
                    continue
                if proc is not None:
                    _quiet_close(proc.stdin)
                gen, proc = self._gen, new
            self._in_event.wait(0.25)
            self._in_event.clear()
            while self._running and self._gen == gen:
                try:
                    index, frame = self._inbox.popleft()
                except IndexError:
                    break
                self._fed.append(index)
                try:
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                except (BrokenPipeError, OSError, ValueError):
                    if self._gen != gen or not self._running:
                        break                  # 是 restart/close 干的, 重新绑定就行
                    self._fail("往 ffmpeg 写帧失败")
                    return
                if len(self._fed) > self._fed.maxlen:
                    self._fed.clear()          # 输出比输入慢得离谱, 帧号记账已不可靠

    def _read_loop(self) -> None:
        gen = -1
        stdout = None
        parser = _AnnexBParser()
        cur: List[bytes] = []
        has_vcl = False           # cur 里已经装进了本帧的 slice
        idr = False
        while self._running:
            if gen != self._gen:
                proc = self._proc
                if proc is None:
                    time.sleep(0.01)     # 换代还没完成: 下一轮再试
                    continue
                gen = self._gen
                _quiet_close(stdout)            # 本线程自己的句柄, 换进程了就该关
                stdout = proc.stdout
                parser = _AnnexBParser()
                cur = []
                has_vcl = False
                idr = False
            try:
                data = stdout.read(65536)
            except (OSError, ValueError):
                data = b""
            if not data:
                if self._gen != gen:
                    continue                   # 上面重新绑定
                proc = self._proc
                if proc is not None and proc.poll() is not None:
                    self._fail("ffmpeg 进程退出(code=%s)" % proc.returncode)
                    return
                time.sleep(0.005)
                continue
            if self._gen != gen:
                continue                       # 这批字节属于已退休的进程: 帧号对不上, 丢掉
            for nal in parser.feed(data):
                nal_type = _nal_type(nal)
                if nal_type == 7:
                    self._sps = nal
                elif nal_type == 8:
                    self._pps = nal
                vcl = nal_type in (1, 5)
                # 一帧 = 一个 AU。-tune zerolatency 会让 x264 开 sliced-threads, 把一帧
                # 切成几十条 slice, 它们必须整批打包给客户端: 只发第一条的话解出来就是
                # "一小条有画面、一大片死灰"。新 AU 的苗头 = 又来参数集/SEI, 或 slice
                # 的起始宏块回到了 0。
                if cur and has_vcl and (not vcl or _first_mb_in_slice(nal) == 0):
                    au = self._close_au(cur, idr)
                    cur = []
                    has_vcl = False
                    if au is not None:
                        # maxlen 挤掉旧 AU 时, 泵会靠帧号断档发现并重新同步
                        self._aus.append(au)
                        self._out_event.set()
                cur.append(nal)
                if vcl:
                    has_vcl = True
                    idr = nal_type == 5

    def _drain_stderr(self) -> None:
        gen = -1
        stderr = None
        while self._running:
            if gen != self._gen:
                proc = self._proc
                if proc is None:
                    time.sleep(0.01)     # 换代还没完成: 下一轮再试
                    continue
                gen = self._gen
                _quiet_close(stderr)
                stderr = proc.stderr
            try:
                line = stderr.readline()
            except (OSError, ValueError):
                line = b""
            if not line:
                if self._gen != gen:
                    continue
                proc_now = self._proc
                if proc_now is None or proc_now.poll() is not None:
                    return          # stderr EOF 且进程没了: 再等也等不到东西
                time.sleep(0.01)
                continue
            text = line.decode("utf-8", "replace").strip()
            if text:
                self.error_lines.append(text)

    def _fail(self, what: str) -> None:
        """ffmpeg 真的没了(不是被 restart 掉的): 打印一次原因并停止喂帧。"""
        self._dead = True
        if self.verbose:
            print("[RTSP][x264] %s: %s" % (what, self.describe_failure()), flush=True)

    def _close_au(self, nals: List[bytes], idr: bool) -> Optional[_AU]:
        try:
            index = self._fed.popleft()
        except IndexError:
            index = -1
        if index < 0:
            return None
        if idr and not any(_nal_type(n) == 7 for n in nals):
            # IDR 必须带参数集, 否则后连客户端拿到它也解不开
            nals = ([self._sps] if self._sps else []) + \
                   ([self._pps] if self._pps else []) + nals
        return _AU(nals, idr, index)

    # ---------- 对外 ----------
    def submit(self, frame: "np.ndarray", index: int) -> None:
        if self._dead:
            return
        if len(self._inbox) == self._inbox.maxlen:
            self._inbox.popleft()       # 编码器跟不上: 丢输入帧比拖慢推流时钟好
            self.dropped_inputs += 1
        self._inbox.append((int(index), frame))
        self._in_event.set()

    def poll(self, timeout: float = 0.0) -> Optional[_AU]:
        if not self._aus:
            if timeout <= 0:
                return None
            self._out_event.wait(timeout)
        try:
            au = self._aus.popleft()
        except IndexError:
            return None
        if not self._aus:
            self._out_event.clear()
        return au

    def restart(self) -> None:
        """丢掉在途数据、重起一个编码器: 下一帧就是干净的 IDR(循环回绕/换播放位置时)。

        线程不重建 —— 它们发现 _gen 变了会自己绑到新进程上; 重建的话就会有两个线程
        同时读同一个 stdout, 把码流撕成互相穿插的碎片。

        顺序必须是"先起新进程 -> 换代 -> 再收旧进程": _gen 变化时 self._proc 一定要
        已经指向活着的进程, 否则线程会把 None 当成新一代绑上, 之后再也不重绑,
        表现是编码器彻底不产出 AU、客户端一个 RTP 包都收不到。
        """
        with self._lock:
            if not self._running:
                return
            old = self._proc
            self._inbox.clear()
            self._fed.clear()
            self._aus.clear()
            self._out_event.clear()
            self._dead = False
            self._spawn()                 # self._proc = 新进程
            self._gen += 1                # 换代: 线程据此绑到新进程
            self._in_event.set()          # 叫醒写线程, 让它尽快察觉换代
            self._terminate(old)          # 旧进程的管道此时才关, 线程会读到 EOF 后重绑

    def parameter_sets(self) -> Tuple[Optional[bytes], Optional[bytes]]:
        return (self._sps, self._pps)

    def describe_failure(self) -> str:
        tail = " | ".join(list(self.error_lines)[-3:])
        return tail or "(ffmpeg 没有报错输出)"

    def _prime(self) -> None:
        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.submit(blank, -1)
        self.submit(blank, -1)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._sps and self._pps:
                break
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                raise RuntimeError("ffmpeg 编码 H.264 失败: %s" % self.describe_failure())
            time.sleep(0.02)
        self.restart()
        if not (self._sps and self._pps):
            raise RuntimeError("拿不到 SPS/PPS, 编码器可能没正常工作: %s"
                               % self.describe_failure())

    def close(self) -> None:
        self._running = False
        self._in_event.set()
        self._out_event.set()
        self._kill_proc()
        for th in self._threads:
            if th.is_alive() and th is not threading.current_thread():
                th.join(timeout=1.0)
        self._threads = []


def _rtcp_sender_report(ssrc: int, rtp_timestamp: int, packet_count: int,
                        octet_count: int) -> bytes:
    """28 字节 RTCP SR, 给客户端提供 RTP 与 NTP 的时间锚点。"""
    now = time.time()
    ntp = int(now + 2208988800.0) & 0xFFFFFFFF
    ntp_frac = int((now - int(now)) * 4294967296.0) & 0xFFFFFFFF
    return struct.pack(
        "!BBHIIIIII",
        0x80, 200, 6,                 # V=2, PT=SR(200), length=6
        ssrc & 0xFFFFFFFF,
        ntp, ntp_frac,
        rtp_timestamp & 0xFFFFFFFF,
        packet_count & 0xFFFFFFFF,
        octet_count & 0xFFFFFFFF,
    )


# ============================================================
#  推流泵: 一个解码线程, 按真实时间把帧发给所有会话
# ============================================================
class _Pump:
    def __init__(self, source: VideoFileSource, verbose: bool = True,
                 video_offset: float = 0.0) -> None:
        self._source = source
        self._verbose = verbose
        self._video_offset = float(video_offset)   # 视频比回放起点提前/滞后多少秒
        self._sessions: List["_Session"] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._t0: Optional[float] = None          # 播放时钟起点
        self._origin_fixed = False                # True: 时钟起点由外部指定, 不随客户端进出重置
        self._frame_index = -1
        self._au_index = None                     # 最后一个交出去的 AU 的帧号
        self._resync = False                      # True: 中间断了, 要丢到下一个 IDR
        self.dropped_encodes = 0
        self._kick_requested = False          # 有会话掉队, 请求重起编码器(立刻出 IDR)
        self._last_kick = 0.0                 # 上一次重起编码器的时刻(节流)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="rtsp-pump", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            sessions = list(self._sessions)
        for s in sessions:
            s.close()

    def _request_kick(self) -> None:
        """某个会话掉队时打个标记, 由泵线程去重起编码器(任何线程都能安全调用)。"""
        self._kick_requested = True

    def _restart_encoder(self) -> None:
        """重起编码器。失败(多是瞬时资源问题)就隔 1 秒重试: 泵线程一旦被异常打死,
        流就永久冻住、客户端只能等超时, 比丢几秒画面严重得多。"""
        while self._running:
            try:
                self._source.restart_encoder()
                return
            except Exception as exc:
                if self._verbose:
                    print("[RTSP][警告] 重起编码器失败, 1 秒后重试: %s" % exc)
                time.sleep(1.0)

    def add_session(self, session: "_Session") -> None:
        session.on_lag = self._request_kick
        with self._lock:
            self._sessions.append(session)
            if self._t0 is None:
                self._t0 = time.time()      # 第一个客户端开始播放时起算
                if self._verbose:
                    print("[RTSP] 开始推流(播放时钟已启动)")
        # 中途接入不补发历史帧: 补发会把最多一个 GOP(几秒、几 MB)瞬间灌给客户端,
        # 时间戳全是"过去"的, GStreamer 一律按迟到丢弃, 显示端冻在第一帧; 突发还
        # 会顶满 TCP 让队列溢出。和真摄像头一致: 客户端等下一个 IDR 自然上线。


    def set_origin(self, t0: float) -> None:
        """把播放时钟的起点固定成 t0(墙钟秒)。

        回放工具用它在"数据开始回放的那一刻"就把视频时间轴对齐, 这样 Viewer 是
        先连还是后连都不影响对齐(后连的会直接从对应的画面位置切入)。
        """
        with self._lock:
            self._t0 = float(t0)
            self._origin_fixed = True

    def remove_session(self, session: "_Session") -> None:
        with self._lock:
            if session in self._sessions:
                self._sessions.remove(session)
            if not self._sessions and not self._origin_fixed:
                self._t0 = None             # 没有客户端了, 下次连接重新计时
                self._frame_index = -1
                if self._verbose:
                    print("[RTSP] 客户端断开, 暂停推流")

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _run(self) -> None:
        fps = self._source.fps or DEFAULT_FPS
        while self._running:
            with self._lock:
                sessions = list(self._sessions)
                t0 = self._t0
            if not sessions or t0 is None:
                time.sleep(0.03)
                continue

            elapsed = time.time() - t0 + self._video_offset   # 视频时间轴
            index = int(elapsed * fps)
            if self._kick_requested:
                self._kick_requested = False
                now = time.time()
                if now - self._last_kick >= 1.0:
                    self._last_kick = now
                    # 消费端跟不上时, 原来要等下一个自然 IDR(gop=60 最长 2 秒,
                    # 客户端会整段黑屏甚至超时断流); 改为立刻重起编码器, 约 0.1 秒
                    # 就能拿到新 IDR。所有会话一起重同步, 各丢几帧, 换来不饿死。
                    self._restart_encoder()
                    self._au_index = None
                    if self._verbose:
                        print("[RTSP] 客户端掉队, 重起编码器补 IDR(消费速度跟不上 30fps)")
            if index != self._frame_index:
                if index < self._frame_index or self._wrapped(index):
                    # 往回 seek / 循环到下一圈: 编码器重起, 保证这一帧是干净 IDR
                    self._restart_encoder()
                    self._au_index = None
                frame = self._source.frame_at(index)
                if frame is None:
                    self.dropped_encodes += 1
                else:
                    self._frame_index = index
                    self._source.submit(frame, index)
            # H.264 的 AU 要等下一帧的起始码才算写完, 所以这里通常拿到的是上一帧
            au = self._source.poll(timeout=0.005)
            if au is not None:
                self._emit(au, fps, sessions)

            # 对齐到下一帧时刻(编码慢时自然跳到最新帧, 不会越积越多)
            target = (self._frame_index + 1) / fps
            delay = target - (time.time() - t0 + self._video_offset)
            time.sleep(delay if delay > 0.0005 else 0.0005)

    def _wrapped(self, index: int) -> bool:
        """播放位置是否跨进了循环的下一圈(帧号本身单调增长, 取模发生在源里)。"""
        n = self._source.frame_count
        if not n or not getattr(self._source, "loop", False) or self._frame_index < 0:
            return False
        return index // n != self._frame_index // n

    def _emit(self, au: "_AU", fps: float, sessions: List["_Session"]) -> None:
        if self._au_index is not None and au.index > self._au_index + 1:
            self._resync = True            # 帧号断档 => 交出去的流有洞
        self._au_index = au.index
        if self._resync:
            if not au.idr:
                return                     # 等一个能独立解码的起点
            self._resync = False
            for s in sessions:
                s.flush()                  # 把洞之前的积压清掉, 从这条 IDR 重新起
        ts = int((au.index / fps) * RTP_CLOCK) & 0xFFFFFFFF
        au.ts = ts
        for s in sessions:
            s.push(ts, au)


# ============================================================
#  RTSP 会话 (一条客户端连接里的一个 video 流)
# ============================================================
class _Session:
    def __init__(self, conn: socket.socket, session_id: str, ssrc: int,
                 transport: str, channel: Tuple[int, int],
                 udp_rtp: Optional[socket.socket], udp_rtcp: Optional[socket.socket],
                 peer_rtp: Optional[Tuple[str, int]], peer_rtcp: Optional[Tuple[str, int]],
                 width: int, height: int, lock: threading.Lock,
                 max_payload: int = DEFAULT_MAX_PAYLOAD, verbose: bool = True,
                 codec: str = "h264", queue_len: int = 16) -> None:
        self.conn = conn
        self.session_id = session_id
        self.ssrc = ssrc
        self.transport = transport              # "tcp" | "udp"
        self.channel = channel                  # TCP interleaved 通道号 (rtp, rtcp)
        self.udp_rtp = udp_rtp
        self.udp_rtcp = udp_rtcp
        self.peer_rtp = peer_rtp
        self.peer_rtcp = peer_rtcp
        self.width = width
        self.height = height
        self.max_payload = max_payload
        self.verbose = verbose
        self.codec = codec

        self._lock = lock                      # 与连接共用的一把写锁
        # 队列要装得下一个完整 GOP, 否则新客户端刚连上就溢出、又要重新同步等 IDR
        self._queue: deque = deque(maxlen=max(4, int(queue_len)))
        self._broken = False                   # True: 队列挤掉了帧, 流有洞, 等 IDR
        self.on_lag = None                     # 队列满时的回调(泵线程会据此重起编码器)
        self._event = threading.Event()
        self._running = True
        self._seq = random.randint(0, 0xFFFF)
        self._sent_packets = 0
        self._sent_octets = 0
        self.encode_errors = 0
        self.last_ts = 0
        self._rtcp_t = 0.0
        self._thread = threading.Thread(target=self._send_loop, name="rtsp-sender",
                                        daemon=True)

    def start(self) -> None:
        self._thread.start()

    def push(self, timestamp: int, au: "_AU") -> None:
        if len(self._queue) == self._queue.maxlen:
            self._broken = True        # 马上要挤掉最旧的一帧: 发送端跟不上, 流断了
            if self.on_lag is not None:
                self.on_lag()          # 通知泵: 立刻重起编码器, 别让我饿着等自然 IDR
        self._queue.append((timestamp, au))
        self._event.set()

    def flush(self) -> None:
        """丢掉还没发出去的积压, 并从下一个 IDR 才重新发(上游出现空洞时用)。"""
        self._queue.clear()
        self._broken = True
        self._event.set()

    def close(self) -> None:
        self._running = False
        self._event.set()
        for s in (self.udp_rtp, self.udp_rtcp):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
        self.udp_rtp = self.udp_rtcp = None
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    # ---------- 发送线程 ----------
    def _send_loop(self) -> None:
        while self._running:
            self._event.wait(timeout=0.5)
            self._event.clear()
            while self._running and self._queue:
                try:
                    ts, au = self._queue.popleft()
                except IndexError:
                    break
                if self._broken:
                    if not au.idr:
                        continue         # 流有洞, 从下一个能独立解码的帧重新起
                    self._broken = False
                self.last_ts = ts
                try:
                    if self.codec == "h264":
                        packets = packetize_h264(au.nals, ts, self._seq, self.ssrc,
                                                 self.max_payload)
                    else:
                        packets = packetize_jpeg(au.nals[0], ts, self._seq, self.ssrc,
                                                 self.width, self.height,
                                                 self.max_payload)
                except Exception:
                    # 个别帧打包异常时跳过这一帧, 不让发送线程挂掉
                    self.encode_errors += 1
                    continue
                self._seq = (self._seq + len(packets)) & 0xFFFF
                if not self._send(packets):
                    self._running = False
                    try:
                        # 发送线程死了会话就成了"僵尸": RTSP 线程还阻塞在 recv 上,
                        # 泵会一直往这个队列里塞帧、反复触发重起编码器。shutdown 叫醒
                        # 对端线程, 让它走正常清理(finally 里 remove_session + close)。
                        self.conn.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    break
            # 每秒一个 RTCP SR, 让客户端拿到时间锚点
            now = time.time()
            if self._running and now - self._rtcp_t >= 1.0:
                self._rtcp_t = now
                self._send_rtcp()

    def _send(self, packets: List[bytes]) -> bool:
        try:
            if self.transport == "tcp":
                if self.conn.fileno() < 0:
                    return False
                with self._lock:
                    for pkt in packets:
                        self.conn.sendall(b"$" + bytes((self.channel[0],)) +
                                          struct.pack("!H", len(pkt)) + pkt)
            else:
                if self.udp_rtp is None or self.peer_rtp is None:
                    return False
                for pkt in packets:
                    self.udp_rtp.sendto(pkt, self.peer_rtp)
            self._sent_packets += len(packets)
            self._sent_octets += sum(len(p) for p in packets)
            return True
        except Exception as exc:
            if self.verbose:
                print("[RTSP] 给客户端发流失败, 停止本会话推流: %s" % exc)
            return False

    def _send_rtcp(self) -> None:
        try:
            sr = _rtcp_sender_report(self.ssrc, self.last_ts,
                                     self._sent_packets, self._sent_octets)
            if self.transport == "tcp":
                with self._lock:
                    self.conn.sendall(b"$" + bytes((self.channel[1],)) +
                                      struct.pack("!H", len(sr)) + sr)
            elif self.udp_rtcp is not None and self.peer_rtcp is not None:
                self.udp_rtcp.sendto(sr, self.peer_rtcp)
        except Exception:
            pass


# ============================================================
#  RTSP 服务
# ============================================================
class RtspVideoServer:
    """把本地视频文件按 RTSP 发布出去的最小实现(RTP/H264, 也可选 RTP/JPEG)。

    典型用法::

        srv = RtspVideoServer("video.mp4", port=554,
                              path="/Streaming/Channels/101",
                              users={"admin": "gzsia2011"})
        srv.start()
        print(srv.url)
        srv.wait_for_client(30)     # 等第一个客户端开始拉流
        ...
        srv.stop()
    """

    def __init__(self, video_path: str, host: str = "0.0.0.0", port: int = 554,
                 path: str = DEFAULT_PATH, users: Optional[Dict[str, str]] = None,
                 fps: Optional[float] = None, quality: int = DEFAULT_QUALITY,
                 loop: bool = True, realm: str = DEFAULT_REALM,
                 max_payload: int = DEFAULT_MAX_PAYLOAD, verbose: bool = True,
                 transport: str = "tcp", debug_requests: bool = False,
                 start_offset: float = 0.0, codec: str = "h264",
                 crf: int = DEFAULT_CRF, gop: Optional[int] = None,
                 bitrate: Optional[int] = None, preset: str = DEFAULT_PRESET,
                 ffmpeg: Optional[str] = None) -> None:
        self.video_path = str(video_path)
        self.host = host
        self.port = int(port)
        self.path = path if path.startswith("/") else "/" + path
        self.users = dict(users or {})
        self.realm = realm
        self.max_payload = int(max_payload)
        self.verbose = verbose
        self.debug_requests = bool(debug_requests)
        # "tcp" = 一律用 RTSP over TCP(interleaved), 丢包率低, 推荐;
        # "auto" = 客户端要 UDP 就给 UDP; "udp" = 一律 UDP
        self.transport = transport.lower()

        self._source = VideoFileSource(self.video_path, fps=fps, quality=quality,
                                       loop=loop, verbose=verbose, codec=codec,
                                       crf=crf, gop=gop, bitrate=bitrate,
                                       preset=preset, ffmpeg=ffmpeg)
        self._pump = _Pump(self._source, verbose=verbose, video_offset=start_offset)
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None
        self._clients: List[socket.socket] = []
        self._client_lock = threading.Lock()
        self._nonce = "%08x%08x" % (random.getrandbits(32), int(time.time()))
        self._first_client = threading.Event()

    # ---------- 生命周期 ----------
    def start(self) -> None:
        """绑定端口并开始监听(此时还不解码, 等客户端来)。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(8)
        sock.settimeout(0.5)
        self._sock = sock
        self._running = True
        self._pump.start()
        self._accept_thread = threading.Thread(target=self._accept_loop,
                                               name="rtsp-accept", daemon=True)
        self._accept_thread.start()
        if self.verbose:
            print(f"[RTSP] 视频服务已启动: {self.url}")

    def stop(self) -> None:
        self._running = False
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.5)
            self._accept_thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        with self._client_lock:
            clients = list(self._clients)
            self._clients.clear()
        for conn in clients:
            try:
                conn.close()
            except Exception:
                pass
        self._pump.stop()
        self._source.close()
        if self.verbose:
            print("[RTSP] 视频服务已停止")

    # ---------- 对外查询/等待 ----------
    @property
    def url(self) -> str:
        """下游程序可用的完整 RTSP 地址(带账号密码)。"""
        if self.host not in ("0.0.0.0", "::", ""):
            host = self.host
        else:
            host = _guess_local_ip()
        user = next(iter(self.users), None)
        cred = f"{user}:{self.users[user]}@" if user else ""
        return f"rtsp://{cred}{host}:{self.port}{self.path}"

    @property
    def loopback_url(self) -> str:
        """本机回环地址(同一台机器上的 Viewer 用它最稳)。"""
        user = next(iter(self.users), None)
        cred = f"{user}:{self.users[user]}@" if user else ""
        return f"rtsp://{cred}127.0.0.1:{self.port}{self.path}"

    @property
    def client_count(self) -> int:
        return self._pump.session_count

    def _debug_log(self, text: str) -> None:
        print(text, flush=True)

    def wait_for_client(self, timeout: Optional[float] = None) -> bool:
        """等第一个客户端开始拉流(PLAY)。返回是否在超时前等到。"""
        return self._first_client.wait(timeout)

    def set_origin(self, t0: float) -> None:
        """把视频时间轴锚定到 t0(墙钟秒, 一般传回放起点)。

        锚定后 Viewer 无论早连晚连, 拿到的都是与数据同一时刻的画面。
        """
        self._pump.set_origin(t0)

    # ---------- 监听循环 ----------
    def _accept_loop(self) -> None:
        while self._running and self._sock is not None:
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(30.0)
            with self._client_lock:
                self._clients.append(conn)
            if self.verbose:
                print(f"[RTSP] 客户端接入: {addr[0]}:{addr[1]}")
            threading.Thread(target=self._client_loop, args=(conn, addr),
                             name="rtsp-client", daemon=True).start()

    # ---------- 单个客户端 ----------
    def _client_loop(self, conn: socket.socket, addr) -> None:
        write_lock = threading.Lock()
        session: Optional[_Session] = None
        session_id = "%08x" % random.getrandbits(32)
        ssrc = random.getrandbits(32)
        buf = bytearray()

        def respond(code: int, reason: str, cseq: Optional[str],
                    extra: Optional[List[str]] = None, body: str = "") -> bool:
            headers = [f"RTSP/1.0 {code} {reason}", f"CSeq: {cseq or '0'}",
                       f"Date: {_http_date()}", "Server: pyfusion-rtsp/1.0"]
            headers.extend(extra or [])
            body_bytes = body.encode("utf-8") if body else b""
            if body_bytes:
                headers.append("Content-Length: %d" % len(body_bytes))
            raw = ("\r\n".join(headers) + "\r\n\r\n").encode("utf-8") + body_bytes
            try:
                with write_lock:
                    conn.sendall(raw)
                return True
            except Exception:
                return False

        try:
            while self._running:
                # ---- 取一条 RTSP 请求(跳过 TCP interleaved 进来的 RTCP) ----
                while True:
                    if buf[:1] == b"$":
                        if len(buf) < 4:
                            if not _recv_more(conn, buf):
                                return
                            continue
                        size = int.from_bytes(buf[2:4], "big")
                        if len(buf) < 4 + size:
                            if not _recv_more(conn, buf):
                                return
                            continue
                        del buf[:4 + size]
                        continue
                    end = buf.find(b"\r\n\r\n")
                    if end >= 0:
                        head = bytes(buf[:end + 4])
                        del buf[:end + 4]
                        break
                    if not _recv_more(conn, buf):
                        return

                lines = head.decode("utf-8", "replace").split("\r\n")
                parts = lines[0].split()
                if len(parts) < 2:
                    if self.debug_requests:
                        self._debug_log("    <- 无法解析的首行: "
                                        + repr(head[:80]))
                    continue
                if self.debug_requests and parts[0] not in (
                        "OPTIONS", "DESCRIBE", "SETUP", "PLAY", "PAUSE",
                        "TEARDOWN", "GET_PARAMETER", "SET_PARAMETER"):
                    self._debug_log("    <- 异常请求行: " + repr(head[:120]))
                method, uri = parts[0].upper(), parts[1]
                headers: Dict[str, str] = {}
                for line in lines[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        headers[k.strip().lower()] = v.strip()
                cseq = headers.get("cseq")

                if self.verbose and self.debug_requests:
                    self._debug_log(
                        f"    <- {method} {uri}"
                        + (f"  [{headers.get('transport')}]" if headers.get("transport") else ""))

                if not self._authorized(headers.get("authorization"), method, uri):
                    if not respond(401, "Unauthorized", cseq, extra=[
                            'WWW-Authenticate: Digest realm="%s", nonce="%s"'
                            % (self.realm, self._nonce)]):
                        return
                    continue

                if method == "OPTIONS":
                    if not respond(200, "OK", cseq, extra=[
                            "Public: OPTIONS, DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN, "
                            "GET_PARAMETER, SET_PARAMETER"]):
                        return
                    continue

                if method == "DESCRIBE":
                    if not respond(200, "OK", cseq, extra=[
                            "Content-Type: application/sdp",
                            "Content-Base: %s/" % self._base_uri(uri)],
                            body=self._build_sdp()):
                        return
                    continue

                if method == "SETUP":
                    session = self._setup(conn, uri, headers, respond, cseq,
                                          session_id, ssrc, write_lock)
                    if session is None:
                        return
                    continue

                if method == "PLAY":
                    if session is None:
                        if not respond(455, "Method Not Valid In This State", cseq):
                            return
                        continue
                    self._pump.add_session(session)
                    self._first_client.set()
                    extra = [f"Session: {session.session_id};timeout=60",
                             "Range: npt=0.000-",
                             "RTP-Info: url=%s/trackID=0;seq=%d;rtptime=%d"
                             % (self._base_uri(uri), session._seq, session.last_ts)]
                    if not respond(200, "OK", cseq, extra=extra):
                        return
                    if self.verbose:
                        print(f"[RTSP] 开始播放 -> {addr[0]} ({session.transport.upper()})")
                    continue

                if method == "PAUSE":
                    if session is not None:
                        self._pump.remove_session(session)
                    if not respond(200, "OK", cseq,
                                   extra=[f"Session: {session_id};timeout=60"]):
                        return
                    continue

                if method == "TEARDOWN":
                    if session is not None:
                        self._pump.remove_session(session)
                        session.close()
                    respond(200, "OK", cseq)
                    return

                if method in ("GET_PARAMETER", "SET_PARAMETER"):
                    if not respond(200, "OK", cseq,
                                   extra=[f"Session: {session_id};timeout=60"]):
                        return
                    continue

                if not respond(405, "Method Not Allowed", cseq, extra=[
                        "Public: OPTIONS, DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN, "
                        "GET_PARAMETER"]):
                    return
        except OSError:
            pass
        finally:
            if session is not None:
                self._pump.remove_session(session)
                session.close()
            with self._client_lock:
                if conn in self._clients:
                    self._clients.remove(conn)
            try:
                conn.close()
            except Exception:
                pass
            if self.verbose:
                print(f"[RTSP] 客户端断开: {addr[0]}")

    # ---------- SETUP ----------
    def _setup(self, conn, uri, headers, respond, cseq, session_id, ssrc,
               write_lock) -> Optional[_Session]:
        transport_hdr = headers.get("transport", "")
        lower = transport_hdr.lower()
        channel = (0, 1)
        if "interleaved=" in lower:
            try:
                ch = lower.split("interleaved=", 1)[1].split(";")[0].split("-")
                channel = (int(ch[0]), int(ch[1]) if len(ch) > 1 else int(ch[0]) + 1)
            except Exception:
                channel = (0, 1)
        if self.transport == "tcp":
            mode = "tcp"
        elif self.transport == "udp":
            mode = "udp"
        else:                                     # auto: 按客户端要求
            mode = "tcp" if ("tcp" in lower or "interleaved=" in lower) else "udp"

        if mode == "udp":
            try:
                ports = lower.split("client_port=", 1)[1].split(";")[0].split("-")
                peer_rtp_port = int(ports[0])
                peer_rtcp_port = int(ports[1]) if len(ports) > 1 else peer_rtp_port + 1
            except Exception:
                peer_rtp_port = peer_rtcp_port = 0
            if peer_rtp_port <= 0:
                mode = "tcp"                      # 信息不全就退回 TCP interleaved

        peer_ip = conn.getpeername()[0]
        udp_rtp = udp_rtcp = None
        peer_rtp = peer_rtcp = None
        extra: List[str] = []
        if mode == "udp":
            udp_rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_rtp.bind(("0.0.0.0", 0))
            udp_rtcp.bind(("0.0.0.0", 0))
            server_port = udp_rtp.getsockname()[1]
            server_port_rtcp = udp_rtcp.getsockname()[1]
            peer_rtp = (peer_ip, peer_rtp_port)
            peer_rtcp = (peer_ip, peer_rtcp_port)
            extra.append(
                "Transport: RTP/AVP;unicast;client_port=%d-%d;server_port=%d-%d;ssrc=%08X"
                % (peer_rtp_port, peer_rtcp_port, server_port, server_port_rtcp, ssrc))
        else:
            extra.append(
                "Transport: RTP/AVP/TCP;unicast;interleaved=%d-%d;ssrc=%08X"
                % (channel[0], channel[1], ssrc))
        extra.append(f"Session: {session_id};timeout=60")

        session = _Session(conn, session_id, ssrc, mode, channel, udp_rtp, udp_rtcp,
                           peer_rtp, peer_rtcp, self._source.width, self._source.height,
                           write_lock, self.max_payload, self.verbose,
                           codec=self._source.codec,
                           queue_len=max(16, self._source.gop * 2))
        session.start()
        if self.verbose and self.debug_requests:
            self._debug_log("    -> " + "; ".join(extra))
        if not respond(200, "OK", cseq, extra=extra):
            session.close()
            return None
        return session

    # ---------- SDP ----------
    @staticmethod
    def _base_uri(uri: str) -> str:
        return uri.split("?")[0]

    def _build_sdp(self) -> str:
        fps = self._source.fps or DEFAULT_FPS
        h264 = self._source.codec == "h264"
        # 会话级行按 RFC 4566 的顺序: v, o, s, i, c, t, 再 a=
        lines = [
            "v=0",
            "o=- %d 1 IN IP4 0.0.0.0" % int(time.time()),
            "s=pyfusion replay video",
            "i=Replayed video stream (%s)" % ("RTSP/RTP-H264" if h264
                                              else "RTSP/RTP-JPEG"),
            "c=IN IP4 0.0.0.0",
            "t=0 0",
            "a=control:*",
            "a=range:npt=0-",
        ]
        if h264:
            lines.append("m=video 0 RTP/AVP %d" % RTP_PAYLOAD_TYPE_H264)
            lines.append("a=rtpmap:%d H264/%d" % (RTP_PAYLOAD_TYPE_H264, RTP_CLOCK))
            lines.append("a=fmtp:%d %s" % (RTP_PAYLOAD_TYPE_H264, self._h264_fmtp()))
        else:
            lines.append("m=video 0 RTP/AVP %d" % RTP_PAYLOAD_TYPE_JPEG)
            lines.append("a=rtpmap:%d JPEG/%d" % (RTP_PAYLOAD_TYPE_JPEG, RTP_CLOCK))
            lines.append("a=fmtp:%d x-dimensions=%d,%d" % (RTP_PAYLOAD_TYPE_JPEG,
                                                            self._source.width,
                                                            self._source.height))
            lines.append("a=quality:5")
        lines.append("a=framerate:%.6f" % fps)
        lines.append("a=control:trackID=0")
        lines.append("")
        return "\r\n".join(lines)

    def _h264_fmtp(self) -> str:
        """a=fmtp: packetization-mode=1 (Annex-B 分片) + profile-level-id + 参数集。

        sprop-parameter-sets 是 base64(SPS),base64(PPS): 让客户端在收到第一个 IDR 之前
        就能建好解码器(海康、GStreamer、ffmpeg 都靠这行提前知道分辨率和 profile)。
        """
        sps, pps = self._source.parameter_sets()
        fmtp = "packetization-mode=1"
        if sps:
            fmtp += ";profile-level-id=%s" % sps[1:4].hex()
        if sps and pps:
            fmtp += ";sprop-parameter-sets=%s,%s" % (
                base64.b64encode(sps).decode("ascii"),
                base64.b64encode(pps).decode("ascii"))
        return fmtp

    # ---------- 认证 ----------
    def _authorized(self, header: Optional[str], method: str, uri: str) -> bool:
        if not self.users:
            return True
        if not header:
            return False
        scheme, _, rest = header.partition(" ")
        scheme = scheme.lower()
        if scheme == "basic":
            try:
                user, _, pwd = base64.b64decode(rest.strip()).decode("utf-8").partition(":")
            except Exception:
                return False
            return self.users.get(user) == pwd
        if scheme == "digest":
            params: Dict[str, str] = {}
            for item in rest.split(","):
                k, _, v = item.partition("=")
                params[k.strip().lower()] = v.strip().strip('"')
            user = params.get("username", "")
            password = self.users.get(user)
            if password is None:
                return False
            ha1 = _md5(f"{user}:{params.get('realm', self.realm)}:{password}")
            ha2 = _md5(f"{method}:{params.get('uri', uri)}")
            nonce = params.get("nonce", "")
            qop = params.get("qop")
            if qop:
                expect = _md5(f"{ha1}:{nonce}:{params.get('nc', '')}:"
                              f"{params.get('cnonce', '')}:{qop}:{ha2}")
            else:
                expect = _md5(f"{ha1}:{nonce}:{ha2}")
            return expect == params.get("response", "")
        return False


# ============================================================
#  辅助函数
# ============================================================
def _recv_more(conn: socket.socket, buf: bytearray, size: int = 4096) -> bool:
    """读一段数据追加到 buf; 返回 False 表示连接已断。"""
    try:
        data = conn.recv(size)
    except socket.timeout:
        return True                      # 超时不算断开, 继续等
    except OSError:
        return False
    if not data:
        return False
    buf.extend(data)
    return True


def _guess_local_ip() -> str:
    """猜一个本机对外 IP(用于打印可达的 RTSP 地址)。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
    finally:
        sock.close()


def parse_users(text: Optional[str]) -> Dict[str, str]:
    """把 "admin:gzsia2011" 这样的字符串解析成 {user: password}。"""
    users: Dict[str, str] = {}
    if not text:
        return users
    for chunk in text.split(","):
        if ":" in chunk:
            user, _, pwd = chunk.partition(":")
            if user.strip():
                users[user.strip()] = pwd
    return users


def run_selftest(url: str, seconds: float, verbose: bool = True) -> int:
    """用 OpenCV 拉一次自己的流, 验证整条链路能解码。"""
    if cv2 is None:
        print("[RTSP][自检] 缺少 cv2, 跳过")
        return 1
    if verbose:
        print(f"[RTSP][自检] 用 OpenCV 拉流验证: {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("[RTSP][自检] 失败: 无法打开流")
        return 1
    # 先丢弃启动缓冲, 再统计稳态解码帧率(否则会把开头的缓冲一起算进来)
    settle = min(2.0, max(0.5, seconds * 0.4))
    t_settle = time.time() + settle
    frame = None
    while time.time() < t_settle:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[RTSP][自检] 失败: 读帧中断")
            cap.release()
            return 1
    t_end = time.time() + max(1.0, seconds - settle)
    frames = 0
    t0 = time.time()
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[RTSP][自检] 失败: 读帧中断")
            cap.release()
            return 1
        frames += 1
    dt = time.time() - t0
    cap.release()
    shape = frame.shape if frame is not None else (0, 0, 0)
    print(f"[RTSP][自检] 成功: 稳态解码 {frames} 帧 / {dt:.1f}s = "
          f"{frames / max(dt, 1e-6):.2f} 帧/秒, 分辨率 {shape[1]}x{shape[0]}")
    return 0


# ============================================================
#  命令行入口
# ============================================================
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="把本地视频文件按 RTSP 发布(H.264 或 MJPEG), 供 raw_sensor_viewer 等程序拉流")
    parser.add_argument("video", nargs="?", default=None, help="视频文件路径")
    parser.add_argument("--video", dest="video_opt", default=None, help="视频文件路径")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0=所有网卡)")
    parser.add_argument("--port", type=int, default=554, help="监听端口 (默认 554)")
    parser.add_argument("--path", default=DEFAULT_PATH,
                        help=f"RTSP 路径 (默认 {DEFAULT_PATH})")
    parser.add_argument("--user", default="admin", help="用户名 (默认 admin)")
    parser.add_argument("--password", default="gzsia2011", help="密码 (默认 gzsia2011)")
    parser.add_argument("--users", default=None,
                        help='多账号 "user1:pass1,user2:pass2" (覆盖 --user/--password)')
    parser.add_argument("--no-auth", action="store_true", help="关闭认证")
    parser.add_argument("--fps", type=float, default=None, help="覆盖视频帧率")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY,
                        help="JPEG 质量 1~100 (仅 --codec jpeg)")
    parser.add_argument("--codec", default="h264", choices=["h264", "jpeg"],
                        help="发布格式: h264=x264(RFC 6184, 需要 ffmpeg, 默认); "
                             "jpeg=RTP/JPEG(RFC 2435, 下游不接受 mjpeg 时别用)")
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF,
                        help="x264 质量, 越小越清晰越吃码率, 常用 18~28 (默认 %d)" % DEFAULT_CRF)
    parser.add_argument("--gop", type=int, default=None,
                        help="两个 IDR 之间的帧数, 默认 2 秒; 越小后连恢复越快、码率越高")
    parser.add_argument("--bitrate", type=int, default=None,
                        help="x264 目标码率(bps), 给了就不用 --crf, 例如 4000000")
    parser.add_argument("--preset", default=DEFAULT_PRESET,
                        help="x264 速度档: ultrafast/superfast/veryfast/faster/fast…")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg 可执行文件路径(默认自动找)")
    parser.add_argument("--no-loop", action="store_true", help="放完即停(默认循环)")
    parser.add_argument("--start-offset", type=float, default=0.0,
                        help="视频相对推流起点提前(负值)/滞后(正值)多少秒, 用于微调对齐")
    parser.add_argument("--selftest", type=float, default=0.0,
                        help="启动后用 OpenCV 拉流 N 秒做自检")
    parser.add_argument("--debug", action="store_true", help="打印每个客户端 RTSP 请求(排查用)")
    parser.add_argument("--transport", default="tcp", choices=["tcp", "udp", "auto"],
                        help="传输方式: tcp=一律 RTSP over TCP(默认, 最稳); "
                             "udp=一律 UDP; auto=按客户端要求")
    parser.add_argument("--quiet", action="store_true", help="少打印")
    args = parser.parse_args(argv)

    video = args.video_opt or args.video
    if not video:
        parser.error("请给出视频文件路径, 例如: --video video.mp4")
    if not os.path.exists(video):
        parser.error(f"视频文件不存在: {video}")

    users = {} if args.no_auth else (parse_users(args.users) or {args.user: args.password})
    try:
        srv = RtspVideoServer(video, host=args.host, port=args.port, path=args.path,
                              users=users, fps=args.fps, quality=args.quality,
                              loop=not args.no_loop, verbose=not args.quiet,
                              transport=args.transport, debug_requests=args.debug,
                              start_offset=args.start_offset, codec=args.codec,
                              crf=args.crf, gop=args.gop, bitrate=args.bitrate,
                              preset=args.preset, ffmpeg=args.ffmpeg)
    except RuntimeError as exc:      # 主要是缺 ffmpeg / ffmpeg 参数不被接受
        print(f"[RTSP] 启动失败: {exc}")
        return 2
    srv.start()
    print("\n下游程序使用地址(可直接粘到 Viewer 的 RTSP 输入框):"
          f"\n  {srv.url}\n  本机回环: {srv.loopback_url}\n")

    if args.selftest > 0:
        rc = run_selftest(srv.loopback_url, args.selftest, verbose=not args.quiet)
        srv.stop()
        return rc

    print("按 Ctrl+C 停止推流 ...")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[RTSP] 收到中断")
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
