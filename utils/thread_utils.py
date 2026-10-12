"""
线程工具：为 Python 线程设置 OS 级名称（Linux pthread_setname_np）

默认情况下 PyTorch 会把所有线程命名为 "pt_main_thread"，
top -H / htop 无法区分。设置后可直接看到各线程的职责。
"""

import ctypes
import threading


def set_thread_name(name: str) -> None:
    """设置当前线程的 OS 名称（最长 15 字符，超出截断）"""
    try:
        libc = ctypes.CDLL("libc.so.6")
        # pthread_t 在 Linux 上等于 TID（unsigned long）
        libc.pthread_setname_np.argtypes = [ctypes.c_ulong, ctypes.c_char_p]
        tid = ctypes.c_ulong(threading.get_ident())
        libc.pthread_setname_np(tid, name.encode("utf-8")[:15])
    except Exception:
        pass  # 非 Linux 或权限不足时静默失败，不影响功能
