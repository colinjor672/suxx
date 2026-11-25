"""运行时日志初始化。

main.py 启动时调用 setup_logging()：
- print/stdout/stderr 同步写入 log/main.log（控制台输出不变）
- 同一条刷屏日志 60 秒内只落盘一次，避免逐帧日志把磁盘写满
- 日志按天滚动，只保留 5 天；启动时和之后每天各清理一次过期文件
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import sys
import threading
import time
from logging.handlers import TimedRotatingFileHandler

# 日志统一放在项目根目录的 log/ 下，与启动时的 cwd 无关
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log"
)

RETENTION_DAYS = 5
CLEANUP_INTERVAL_S = 24 * 60 * 60
SPAM_INTERVAL_S = 60.0
_MAX_SIGNATURES = 2000

# 数字、浮点数、地址等易变部分抹掉后用于判断是否为同一条刷屏日志
_DYNAMIC_TOKEN = re.compile(r"0x[0-9a-fA-F]+|\d+(?:\.\d+)?")
_LOG_LOCK = threading.Lock()
_EMIT_STATE = threading.local()
_READY = False
_TEES = []


def _signature(line: str) -> str:
    return _DYNAMIC_TOKEN.sub("#", line.strip())[:200]


def _cleanup_once(log_dir: str) -> None:
    """删除 log_dir 下超过保留期的日志文件。"""
    cutoff = time.time() - RETENTION_DAYS * 86400
    try:
        names = os.listdir(log_dir)
    except OSError:
        return
    for name in names:
        path = os.path.join(log_dir, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def _cleanup_loop(log_dir: str) -> None:
    while True:
        time.sleep(CLEANUP_INTERVAL_S)
        _cleanup_once(log_dir)


class _LogTee:
    """把标准输出/错误同时写入 logger，并对重复刷屏做 60 秒节流。"""

    def __init__(self, stream, logger: logging.Logger):
        self._stream = stream
        self._logger = logger
        self._pending = ""
        self._seen = {}  # signature -> [上次落盘时间, 期间省略条数]
        self._lock = threading.Lock()

    def write(self, text):
        if not text:
            return 0
        try:
            self._stream.write(text)
        except Exception:
            pass

        lines = []
        with self._lock:
            self._pending += text
            if "\n" in self._pending:
                *lines, self._pending = self._pending.split("\n")
        for line in lines:
            self._emit(line)
        return len(text)

    def _emit(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return

        sig = _signature(line)
        now = time.monotonic()
        with self._lock:
            last = self._seen.get(sig)
            if last is not None and now - last[0] < SPAM_INTERVAL_S:
                last[1] += 1
                return
            skipped = last[1] if last is not None else 0
            self._seen[sig] = [now, 0]
            if len(self._seen) > _MAX_SIGNATURES:
                self._prune(now)

        if skipped:
            line += f"（此前 {int(SPAM_INTERVAL_S)} 秒内相同日志 {skipped} 条已省略）"

        if getattr(_EMIT_STATE, "active", False):
            return
        _EMIT_STATE.active = True
        try:
            self._logger.info(line)
        except Exception:
            pass
        finally:
            _EMIT_STATE.active = False

    def _prune(self, now: float) -> None:
        stale = [
            key for key, value in self._seen.items()
            if now - value[0] > SPAM_INTERVAL_S * 2
        ]
        for key in stale:
            self._seen.pop(key, None)

    def flush_pending(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, ""
        if pending.strip():
            self._emit(pending)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _flush_all() -> None:
    for tee in _TEES:
        tee.flush_pending()
        tee.flush()


def _make_handler(path: str) -> TimedRotatingFileHandler:
    handler = TimedRotatingFileHandler(
        path,
        when="midnight",
        backupCount=RETENTION_DAYS,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    return handler


def get_rotating_logger(name: str, filename: str) -> logging.Logger:
    """给独立模块日志（如导航日志）使用，同样写入 log/ 并按天滚动。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    os.makedirs(LOG_DIR, exist_ok=True)
    _cleanup_once(LOG_DIR)
    logger.addHandler(_make_handler(os.path.join(LOG_DIR, filename)))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def setup_logging(log_dir: str = LOG_DIR) -> logging.Logger:
    """初始化全局日志，重复调用只生效一次。"""
    global _READY

    with _LOG_LOCK:
        if _READY:
            return logging.getLogger("main")

        os.makedirs(log_dir, exist_ok=True)
        _cleanup_once(log_dir)

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(_make_handler(os.path.join(log_dir, "main.log")))

        _TEES.append(_LogTee(sys.stdout, root))
        _TEES.append(_LogTee(sys.stderr, root))
        sys.stdout, sys.stderr = _TEES[0], _TEES[1]
        atexit.register(_flush_all)

        threading.Thread(
            target=_cleanup_loop,
            args=(log_dir,),
            name="log-cleanup",
            daemon=True,
        ).start()
        _READY = True

    return logging.getLogger("main")
