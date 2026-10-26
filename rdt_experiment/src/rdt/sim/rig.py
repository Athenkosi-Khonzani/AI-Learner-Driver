"""Assemble the simulated rig: synthetic camera + simulated rover + scripted planner."""

from __future__ import annotations

import numpy as np
import websockets

from ai_learner_host.camera.frame_buffer import FrameBuffer

from rdt.runtime import RuntimeParts
from rdt.sim.devices import SimCameraFeed, SimRover
from rdt.sim.scripted_llm import ScriptedLLM
from rdt.sim.synthetic import SimWorld, SyntheticCamera, synthetic_host_config


class SimulatedRig:
    def __init__(self, *, seed: int = 7, time_scale: float = 1.0, camera_lag_s: float = 0.4,
                 placement_noise_mm: float = 6.0, use_real_llm: bool = False, settle_s: float = 1.0) -> None:
        self.world = SimWorld()
        self.host_config = synthetic_host_config()
        self.host_config.heading_reply_settle_seconds = settle_s
        self.frame_buffer = FrameBuffer()
        self.camera = SyntheticCamera(self.world, self.host_config)
        self.feed = SimCameraFeed(self.camera, self.frame_buffer, lag_s=camera_lag_s)
        self.rover = SimRover(self.world, time_scale=time_scale, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.noise = placement_noise_mm
        self.llm = None if use_real_llm else ScriptedLLM()
        self._server = None
        self.uri = ""

    def parts(self) -> RuntimeParts:
        return RuntimeParts(host_config=self.host_config, llm=self.llm, frame_buffer=self.frame_buffer,
                            skip_overrides=self.keep_overrides())

    def keep_overrides(self) -> list[str]:
        """Host override keys the simulated rig must not take from rdt_config (calibration-dependent)."""
        return [
            "heading_reply_settle_seconds",
            "parallax_camera_height_mm",
            "parallax_marker_height_mm",
            "parallax_nadir_x_mm",
            "parallax_nadir_y_mm",
        ]

    async def start(self) -> str:
        self.feed.start()
        self._server = await websockets.serve(self.rover.handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.uri = f"ws://127.0.0.1:{port}"
        return self.uri

    async def stop(self) -> None:
        self.feed.stop()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def place(self, task) -> None:
        """What the operator does by hand: put rover and blocks on their marks."""
        self.world.place_task(task, self.rng, self.noise)
