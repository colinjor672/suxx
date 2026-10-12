"""
Python POSIX 共享内存 + 信号量工具（与 GDExtension ShmSync 对应）

使用 ctypes 调用 libc 的 shm_open/mmap/sem_open/sem_wait/sem_post，
零外部依赖，与 Godot GDExtension ShmSync 完全兼容。

协议：
  - 共享内存: /shm_input / shm_frame
  - 信号量:   /shm_input_sem / shm_frame_sem
  - 数据格式: [4字节 frame_id:u32 LE][像素数据]
"""

import ctypes
import ctypes.util
import mmap
import os
import struct
from typing import Optional


# ── libc 函数 ──
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_librt = ctypes.CDLL(ctypes.util.find_library("rt"), use_errno=True)

# shm_open
_librt.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_librt.shm_open.restype = ctypes.c_int

# shm_unlink
_librt.shm_unlink.argtypes = [ctypes.c_char_p]
_librt.shm_unlink.restype = ctypes.c_int

# ftruncate
_libc.ftruncate.argtypes = [ctypes.c_int, ctypes.c_int64]
_libc.ftruncate.restype = ctypes.c_int

# sem_open
_libc.sem_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_libc.sem_open.restype = ctypes.c_void_p

# sem_wait
_libc.sem_wait.argtypes = [ctypes.c_void_p]
_libc.sem_wait.restype = ctypes.c_int

# sem_trywait
_libc.sem_trywait.argtypes = [ctypes.c_void_p]
_libc.sem_trywait.restype = ctypes.c_int

# sem_timedwait
_libc.sem_timedwait.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_libc.sem_timedwait.restype = ctypes.c_int

# sem_post
_libc.sem_post.argtypes = [ctypes.c_void_p]
_libc.sem_post.restype = ctypes.c_int

# sem_close
_libc.sem_close.argtypes = [ctypes.c_void_p]
_libc.sem_close.restype = ctypes.c_int

# sem_unlink
_libc.sem_unlink.argtypes = [ctypes.c_char_p]
_libc.sem_unlink.restype = ctypes.c_int

# close
_libc.close.argtypes = [ctypes.c_int]
_libc.close.restype = ctypes.c_int

# O_CREAT, O_RDWR, O_RDONLY, O_EXCL
O_RDONLY = 0
O_RDWR   = 2
O_CREAT  = 0o100
O_EXCL   = 0o200

SEM_FAILED = ctypes.c_void_p(0).value  # (void*)-1


class ShmSyncPy:
    """Python 端 POSIX 共享内存 + 信号量（与 GDExtension ShmSync 兼容）"""

    def __init__(self, name: str, size: int, is_writer: bool):
        self._name = name
        self._data_size = size
        self._total_size = size + 4  # 4 字节帧 ID 头
        self._is_writer = is_writer
        self._shm_fd = -1
        self._mmap_obj: Optional[mmap.mmap] = None
        self._sem = None

        self._open()

    def _open(self):
        """打开/创建共享内存和信号量"""
        name_bytes = self._name.encode("utf-8")

        # 1. shm_open
        oflag = O_RDWR
        if self._is_writer:
            oflag |= O_CREAT

        self._shm_fd = _librt.shm_open(name_bytes, oflag, 0o666)
        if self._shm_fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"shm_open('{self._name}') 失败: {os.strerror(err)}")

        # 2. Writer: ftruncate
        if self._is_writer:
            if _libc.ftruncate(self._shm_fd, self._total_size) < 0:
                err = ctypes.get_errno()
                _libc.close(self._shm_fd)
                raise OSError(err, f"ftruncate 失败: {os.strerror(err)}")

        # 3. mmap（ACCESS_WRITE = MAP_SHARED + PROT_READ|PROT_WRITE）
        self._mmap_obj = mmap.mmap(self._shm_fd, self._total_size,
                                   access=mmap.ACCESS_WRITE)

        # 4. sem_open
        sem_name = f"/{self._name}_sem"
        sem_bytes = sem_name.encode("utf-8")
        if self._is_writer:
            self._sem = _libc.sem_open(sem_bytes, O_CREAT, 0o666, 0)
        else:
            # ctypes 强制 4 参数，非创建时补 0o666, 0
            self._sem = _libc.sem_open(sem_bytes, 0, 0o666, 0)

        if self._sem == SEM_FAILED or self._sem is None:
            err = ctypes.get_errno()
            print(f"[ShmSyncPy] sem_open('{sem_name}') 失败: {os.strerror(err)}")
            self._sem = None
        else:
            print(f"[ShmSyncPy] {'Writer' if self._is_writer else 'Reader'} "
                  f"'{self._name}' 已打开, {self._data_size} 字节, semaphore OK")

    def close(self):
        """关闭共享内存和信号量"""
        if self._sem:
            _libc.sem_close(self._sem)
            self._sem = None

        if self._is_writer and self._name:
            sem_name = f"/{self._name}_sem"
            _libc.sem_unlink(sem_name.encode("utf-8"))

        if self._mmap_obj:
            self._mmap_obj.close()
            self._mmap_obj = None

        if self._shm_fd >= 0:
            _libc.close(self._shm_fd)
            self._shm_fd = -1

        if self._is_writer and self._name:
            _librt.shm_unlink(self._name.encode("utf-8"))

    @property
    def is_open(self) -> bool:
        return self._mmap_obj is not None

    def get_frame_id(self) -> int:
        """读取帧 ID（前 4 字节）"""
        if not self._mmap_obj:
            return 0
        return struct.unpack_from("<I", self._mmap_obj, 0)[0]

    def set_frame_id(self, fid: int):
        """写入帧 ID（前 4 字节）"""
        if not self._mmap_obj:
            return
        struct.pack_into("<I", self._mmap_obj, 0, fid)

    def get_data(self) -> bytes:
        """读取数据区（跳过 4 字节帧 ID 头）"""
        if not self._mmap_obj:
            return b""
        self._mmap_obj.seek(4)
        return self._mmap_obj.read(self._data_size)

    def store_data(self, data: bytes):
        """写入数据区"""
        if not self._mmap_obj:
            return
        self._mmap_obj.seek(4)
        size = min(len(data), self._data_size)
        self._mmap_obj.write(data[:size])

    def write_raw(self, data: memoryview):
        """直接写入原始内存（零拷贝，使用 memoryview）"""
        if not self._mmap_obj:
            return
        self._mmap_obj.seek(4)
        size = min(data.nbytes, self._data_size)
        self._mmap_obj.write(data[:size])

    def wait_for_new_frame(self, timeout_ms: int = 0) -> bool:
        """等待新帧（Reader 调用）"""
        if not self._sem:
            return True  # 无信号量，假装有新帧

        if timeout_ms < 0:
            return _libc.sem_trywait(self._sem) == 0

        if timeout_ms == 0:
            # 无限等待
            while _libc.sem_wait(self._sem) != 0:
                err = ctypes.get_errno()
                if err != 4:  # EINTR
                    return False
            return True

        # 带超时
        # sem_timedwait 需要绝对时间（自 1970 起的时刻），必须用当前时间 + 偏移，
        # 否则传入相对时间会立即超时返回，导致上层忙循环空转（高 CPU）
        import time
        now = time.time()
        sec = int(now) + timeout_ms // 1000
        ns = int((now - int(now)) * 1_000_000_000) + (timeout_ms % 1000) * 1_000_000
        if ns >= 1_000_000_000:
            sec += 1
            ns -= 1_000_000_000
        ts = struct.pack("@lL", sec, ns)
        while True:
            if _libc.sem_timedwait(self._sem, ts) == 0:
                return True
            err = ctypes.get_errno()
            if err != 4:  # EINTR 被打断则重试
                return False

    def signal_new_frame(self):
        """通知新帧就绪（Writer 调用）"""
        if self._sem:
            _libc.sem_post(self._sem)