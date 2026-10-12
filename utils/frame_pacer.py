from __future__ import annotations

import math
import time
from collections.abc import Callable


class FramePacer:
    """Keep consecutive frame releases at or below a fixed frame rate."""

    def __init__(
        self,
        fps: float,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("播放帧率必须大于 0")
        self.fps = float(fps)
        self.interval = 1.0 / self.fps
        self._clock = clock
        self._sleeper = sleeper
        self._last_release: float | None = None

    def wait(self) -> float:
        """Wait until the next frame slot and return the actual sleep duration."""
        now = self._clock()
        if self._last_release is None:
            self._last_release = now
            return 0.0

        delay = self._last_release + self.interval - now
        slept = max(0.0, delay)
        if slept > 0:
            self._sleeper(slept)
        self._last_release = self._clock()
        return slept
