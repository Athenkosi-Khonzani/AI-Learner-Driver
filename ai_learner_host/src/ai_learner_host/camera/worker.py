from __future__ import annotations

import time
from threading import Event

import cv2
from PySide6.QtCore import QThread, Signal

from .frame_buffer import FrameBuffer


class CameraWorker(QThread):
    """Dedicated blocking capture thread for IP/USB camera streams."""

    frame_ready = Signal(object)
    status_changed = Signal(bool, str)
    metrics_changed = Signal(float, float)  # fps, approximate frame age/latency ms

    def __init__(self, frame_buffer: FrameBuffer, parent=None) -> None:
        super().__init__(parent)
        self.frame_buffer = frame_buffer
        self._stop = Event()
        self._source = ""
        self._reconnect_seconds = 3.0

    def configure(self, source: str, reconnect_seconds: float = 3.0) -> None:
        self._source = source.strip()
        self._reconnect_seconds = max(0.5, reconnect_seconds)

    def start_capture(self) -> None:
        self._stop.clear()
        if not self.isRunning():
            self.start()

    def stop_capture(self) -> None:
        self._stop.set()
        self.wait(2500)

    def run(self) -> None:
        if not self._source:
            self.status_changed.emit(False, "CAMERA URL NOT SET")
            return

        while not self._stop.is_set():
            capture = cv2.VideoCapture(self._normalize_source(self._source))
            if not capture.isOpened():
                self.status_changed.emit(False, "CAMERA OFFLINE")
                capture.release()
                self._stop.wait(self._reconnect_seconds)
                continue

            # Keep the decoder/backend buffer as shallow as the active OpenCV
            # backend allows. The dedicated worker continuously drains the
            # stream, and the host only retains the newest frame.
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            self.status_changed.emit(True, "CAMERA LIVE")
            window_start = time.monotonic()
            frame_count = 0

            while not self._stop.is_set():
                read_start = time.monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                self.frame_buffer.put(frame)
                self.frame_ready.emit(frame)
                frame_count += 1

                now = time.monotonic()
                if now - window_start >= 1.0:
                    fps = frame_count / max(now - window_start, 1e-6)
                    self.metrics_changed.emit(fps, (now - read_start) * 1000.0)
                    window_start = now
                    frame_count = 0

            capture.release()
            self.status_changed.emit(False, "CAMERA RECONNECTING")
            if not self._stop.is_set():
                self._stop.wait(self._reconnect_seconds)

        self.status_changed.emit(False, "CAMERA STOPPED")

    @staticmethod
    def _normalize_source(source: str):
        stripped = source.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
