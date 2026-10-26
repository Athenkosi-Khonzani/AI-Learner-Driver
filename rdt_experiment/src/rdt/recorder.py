"""Records the trial video and the rover trail from the overhead camera.

The recorder runs its own PerceptionService instance on the shared frame
buffer, so its pose filter never interferes with the mission controller's
captures. Output frames are written at a fixed rate in wall-clock time (the
latest processed frame is repeated when processing is slower than the video
rate), so the video plays back in real time.
"""

from __future__ import annotations

import asyncio
import copy
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ai_learner_host.camera.frame_buffer import FrameBuffer
from ai_learner_host.perception.service import PerceptionService

from rdt.metrics import TrailSample
from rdt.video import (
    ROUND_COLOURS,
    Canvas,
    draw_goal,
    draw_polyline,
    draw_pose,
    prompt_strip,
    put_text,
)


class TrajectoryRecorder:
    def __init__(
        self,
        frame_buffer: FrameBuffer,
        host_config,
        canvas: Canvas,
        video_path: Path,
        *,
        fps: float = 8.0,
        codec: str = "mp4v",
        prompt_text: str = "",
        label: str = "",
        goal: tuple[float, float, float, float | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        t0: float | None = None,
    ) -> None:
        self.frame_buffer = frame_buffer
        self.perception = PerceptionService(frame_buffer, copy.deepcopy(host_config))
        self.canvas = canvas
        self.video_path = Path(video_path)
        self.fps = float(fps)
        self.codec = codec
        self.prompt_text = prompt_text
        self.label = label
        self.goal = goal  # (x, y, tolerance, heading or None)
        self.clock = clock
        self.t0 = t0 if t0 is not None else clock()
        self.samples: list[TrailSample] = []
        self.sample_rounds: list[int] = []
        self.round_index = 0
        self.status_text = ""
        self._writer: cv2.VideoWriter | None = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._frames_written = 0
        self._record_start = 0.0
        self.last_birdseye: np.ndarray | None = None
        self.last_output: np.ndarray | None = None
        self.on_frame: Callable[[np.ndarray], None] | None = None
        self.errors: list[str] = []

    # ------------------------------------------------------------------ control
    def open(self, title_image: np.ndarray, title_seconds: float) -> None:
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (self.canvas.width, self.canvas.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.video_path} (codec {self.codec})")
        cv2.imwrite(str(self.video_path.with_name("title.png")), title_image)
        for _ in range(int(round(title_seconds * self.fps))):
            self._writer.write(title_image)

    async def start(self) -> None:
        self._running = True
        self._record_start = self.clock()
        self._frames_written = 0
        self._task = asyncio.create_task(self._loop(), name="rdt-recorder")

    def set_round(self, index: int) -> None:
        self.round_index = int(index)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        self._task = None

    def finish(self, final_image: np.ndarray | None, hold_seconds: float) -> None:
        if self._writer is None:
            return
        if final_image is not None:
            frame = self._fit(final_image)
            for _ in range(int(round(hold_seconds * self.fps))):
                self._writer.write(frame)
        self._writer.release()
        self._writer = None

    # ------------------------------------------------------------------ internals
    def _fit(self, img: np.ndarray) -> np.ndarray:
        if img.shape[1] != self.canvas.width or img.shape[0] != self.canvas.height:
            return cv2.resize(img, (self.canvas.width, self.canvas.height))
        return img

    async def _loop(self) -> None:
        period = 1.0 / self.fps
        last_seq = -1
        while self._running:
            tick = self.clock()
            packet = self.frame_buffer.latest_packet(max_age_seconds=3.0)
            if packet is not None and packet.sequence != last_seq:
                last_seq = packet.sequence
                try:
                    world = await asyncio.to_thread(self.perception._process_world, packet.frame)
                    self._add_sample(world.scene_json)
                    self.last_birdseye = world.annotated_frame
                except Exception as exc:  # keep recording; note the problem
                    if len(self.errors) < 20:
                        self.errors.append(f"{self.clock() - self.t0:.2f}s {exc}")
            if self.last_birdseye is not None:
                frame = self._compose(self.last_birdseye)
                self.last_output = frame
                if self.on_frame is not None:
                    try:
                        self.on_frame(frame)
                    except Exception:
                        pass
                due = int((self.clock() - self._record_start) * self.fps) + 1
                while self._frames_written < due and self._writer is not None:
                    self._writer.write(frame)
                    self._frames_written += 1
            await asyncio.sleep(max(0.0, period - (self.clock() - tick)))

    def _add_sample(self, scene: dict[str, Any]) -> None:
        rover = scene.get("rover") or {}
        pose = rover.get("pose")
        detected = bool(pose) and bool(rover.get("pose_valid")) and float(rover.get("age_s", 1.0)) < 0.05
        self.samples.append(
            TrailSample(
                t=round(self.clock() - self.t0, 3),
                x_mm=pose["x_mm"] if pose else None,
                y_mm=pose["y_mm"] if pose else None,
                heading_deg=pose["heading_deg"] if pose else None,
                confidence=float(rover.get("confidence", 0.0)),
                detected=detected,
            )
        )
        self.sample_rounds.append(self.round_index)

    def trail_by_round(self) -> dict[int, list[tuple[float, float]]]:
        out: dict[int, list[tuple[float, float]]] = {}
        for s, r in zip(self.samples, self.sample_rounds):
            if s.detected and s.x_mm is not None:
                out.setdefault(r, []).append((s.x_mm, s.y_mm))
        return out

    def _compose(self, birdseye: np.ndarray) -> np.ndarray:
        img = self.canvas.from_birdseye(birdseye)
        if self.goal is not None:
            gx, gy, tol, gh = self.goal
            draw_goal(img, self.canvas, gx, gy, tol, gh)
        for r, pts in sorted(self.trail_by_round().items()):
            draw_polyline(img, self.canvas, pts, ROUND_COLOURS[r % len(ROUND_COLOURS)], 3)
        last = next((s for s in reversed(self.samples) if s.detected and s.x_mm is not None), None)
        if last is not None:
            draw_pose(img, self.canvas, last.x_mm, last.y_mm, last.heading_deg,
                      ROUND_COLOURS[self.round_index % len(ROUND_COLOURS)])
        elapsed = self.clock() - self._record_start
        rnd = "first pass" if self.round_index == 0 else f"correction {self.round_index}"
        put_text(img, f"{self.label}   t = {elapsed:5.1f} s   {rnd}   {self.status_text}", (16, 84), 0.6)
        prompt_strip(img, self.prompt_text, "REC")
        cv2.circle(img, (img.shape[1] - 30, img.shape[0] - 19), 7, (0, 0, 255), -1)
        return img
