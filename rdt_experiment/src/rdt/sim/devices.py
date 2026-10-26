"""Simulated rover (WebSocket server speaking the real protocol) and camera feed."""

from __future__ import annotations

import asyncio
import collections
import json
import math
import threading
import time

import numpy as np

from ai_learner_host.camera.frame_buffer import FrameBuffer
from llm_test.simulator.geometry import polygons_intersect

from rdt.sim.synthetic import SimWorld, SyntheticCamera
from rdt.worldstate import rover_polygon


class SimRover:
    """Executes path messages in real time with realistic, seedable errors."""

    def __init__(
        self,
        world: SimWorld,
        *,
        rover_length_mm: float = 155.5,
        rover_width_mm: float = 121.3,
        speed_mm_s: float = 220.0,
        turn_deg_s: float = 120.0,
        distance_scale_sd: float = 0.02,
        drift_deg_per_m_sd: float = 1.5,
        turn_scale_sd: float = 0.02,
        tof_stop_mm: float = 40.0,
        time_scale: float = 1.0,
        seed: int = 1,
    ) -> None:
        self.world = world
        self.L, self.W = rover_length_mm, rover_width_mm
        self.speed = speed_mm_s
        self.turn_rate = turn_deg_s
        self.dist_sd = distance_scale_sd
        self.drift_sd = drift_deg_per_m_sd
        self.turn_sd = turn_scale_sd
        self.tof_stop = tof_stop_mm
        self.time_scale = time_scale
        self.rng = np.random.default_rng(seed)
        self.vbat = 8.1
        self._telemetry = False
        self.paths_received: list[dict] = []

    async def handle(self, websocket) -> None:
        telem_task = asyncio.create_task(self._telemetry_loop(websocket))
        try:
            async for raw in websocket:
                msg = json.loads(raw)
                if msg.get("t") == "telem":
                    self._telemetry = bool(msg.get("on"))
                elif msg.get("t") == "path":
                    self.paths_received.append(msg)
                    await self._execute(websocket, msg)
                elif msg.get("t") == "ping":
                    await websocket.send(json.dumps({"t": "pong"}))
        except Exception:
            pass
        finally:
            telem_task.cancel()

    async def _telemetry_loop(self, websocket) -> None:
        while True:
            await asyncio.sleep(0.5)
            if self._telemetry:
                self.vbat = max(7.0, self.vbat - 0.0005)
                try:
                    await websocket.send(json.dumps({"t": "telem", "vbat": round(self.vbat, 2), "soc": 80, "dist": 500}))
                except Exception:
                    return

    def _blocked_ahead(self, x, y, h, sign) -> bool:
        th = math.radians(h)
        ahead = rover_polygon(x + sign * self.tof_stop * math.sin(th), y + sign * self.tof_stop * math.cos(th), h, self.L, self.W)
        with self.world.lock:
            blocks = list(self.world.blocks)
        for b in blocks:
            poly = rover_polygon(b.x_mm, b.y_mm, b.orientation_deg, b.length_mm, b.breadth_mm)
            if polygons_intersect(ahead, poly):
                return True
        return False

    async def _execute(self, websocket, msg) -> None:
        path_id = int(msg["id"])
        dt = 0.04
        for i, cmd in enumerate(msg["cmds"]):
            op = cmd[0]
            if op == "STOP":
                await websocket.send(json.dumps({"t": "ack", "of": "cmd", "id": path_id, "i": i}))
                break
            value = float(cmd[1])
            x, y, h = self.world.pose()
            if op == "TURN":
                actual = value * (1.0 + self.rng.normal(0, self.turn_sd))
                steps = max(1, int(abs(actual) / (self.turn_rate * dt)))
                for k in range(steps):
                    self.world.set_pose(x, y, h + actual * (k + 1) / steps)
                    await asyncio.sleep(dt * self.time_scale)
            else:
                sign = 1.0 if op == "FWD" else -1.0
                actual = value * (1.0 + self.rng.normal(0, self.dist_sd))
                drift_total = self.rng.normal(0, self.drift_sd) * actual / 1000.0
                steps = max(1, int(actual / (self.speed * dt)))
                for k in range(steps):
                    cx, cy, ch = self.world.pose()
                    if sign > 0 and self._blocked_ahead(cx, cy, ch, sign):
                        await websocket.send(
                            json.dumps({"t": "nack", "id": path_id, "i": i, "reason": "obstacle_tof"})
                        )
                        return
                    hh = h + drift_total * (k + 1) / steps
                    th = math.radians(hh)
                    d = sign * actual / steps
                    self.world.set_pose(cx + d * math.sin(th), cy + d * math.cos(th), hh)
                    await asyncio.sleep(dt * self.time_scale)
            await websocket.send(json.dumps({"t": "ack", "of": "cmd", "id": path_id, "i": i}))
        await websocket.send(json.dumps({"t": "ack", "of": "path", "id": path_id}))


class SimCameraFeed:
    """Background thread that renders frames into a FrameBuffer, with stream lag."""

    def __init__(self, camera: SyntheticCamera, frame_buffer: FrameBuffer, *, fps: float = 12.0, lag_s: float = 0.4) -> None:
        self.camera = camera
        self.buffer = frame_buffer
        self.fps = fps
        self.lag = lag_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sim-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)
        self.connected = False

    def _run(self) -> None:
        queue: collections.deque = collections.deque()
        period = 1.0 / self.fps
        while not self._stop.is_set():
            now = time.monotonic()
            queue.append((now, self.camera.render()))
            while queue and now - queue[0][0] >= self.lag:
                self.buffer.put(queue.popleft()[1])
                self.connected = True
            self._stop.wait(period)
