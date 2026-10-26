from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import time
from typing import Any


@dataclass(slots=True)
class BufferedFrame:
    frame: Any
    timestamp_monotonic: float
    sequence: int


class FrameBuffer:
    """Thread-safe latest-frame buffer with monotonic freshness metadata."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: BufferedFrame | None = None
        self._sequence = 0

    def put(self, frame: Any) -> None:
        with self._lock:
            self._sequence += 1
            self._latest = BufferedFrame(
                frame=frame.copy(),
                timestamp_monotonic=time.monotonic(),
                sequence=self._sequence,
            )

    def latest_packet(
        self,
        *,
        max_age_seconds: float | None = None,
    ) -> BufferedFrame | None:
        with self._lock:
            item = self._latest
            if item is None:
                return None
            if (
                max_age_seconds is not None
                and time.monotonic() - item.timestamp_monotonic
                > max_age_seconds
            ):
                return None
            return BufferedFrame(
                frame=item.frame.copy(),
                timestamp_monotonic=item.timestamp_monotonic,
                sequence=item.sequence,
            )

    def latest(self, *, max_age_seconds: float | None = None) -> Any | None:
        item = self.latest_packet(max_age_seconds=max_age_seconds)
        return None if item is None else item.frame

    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    def age_seconds(self) -> float | None:
        with self._lock:
            if self._latest is None:
                return None
            return time.monotonic() - self._latest.timestamp_monotonic
