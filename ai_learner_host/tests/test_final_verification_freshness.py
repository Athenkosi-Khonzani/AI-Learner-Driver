from __future__ import annotations

import asyncio
import unittest

import numpy as np

from ai_learner_host.camera.frame_buffer import FrameBuffer
from ai_learner_host.config import AppConfig
from ai_learner_host.perception.service import PerceptionService


class SettledVerificationCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_frames_after_settle_baseline(self):
        config = AppConfig()
        config.heading_reply_settle_seconds = 0.03
        config.heading_reply_timeout_seconds = 1.0
        config.heading_reply_samples = 2

        buffer = FrameBuffer()
        service = PerceptionService(buffer, config)
        service.readiness_issues = lambda: []
        service._process_world = lambda frame: int(frame[0, 0, 0])

        # Pre-path / stale frame.
        buffer.put(np.full((2, 2, 3), 1, dtype=np.uint8))

        async def feed_frames():
            # This frame arrives during the settle period and must be rejected
            # by the post-delay sequence baseline.
            await asyncio.sleep(0.01)
            buffer.put(np.full((2, 2, 3), 2, dtype=np.uint8))

            # These are the two frames that arrive after settling.
            await asyncio.sleep(0.04)
            buffer.put(np.full((2, 2, 3), 3, dtype=np.uint8))
            await asyncio.sleep(0.02)
            buffer.put(np.full((2, 2, 3), 4, dtype=np.uint8))

        feeder = asyncio.create_task(feed_frames())
        result = await service.capture_settled_world_state()
        await feeder

        self.assertEqual(result, 4)


if __name__ == "__main__":
    unittest.main()
