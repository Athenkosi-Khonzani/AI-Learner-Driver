from __future__ import annotations

import asyncio
import unittest

import numpy as np

from ai_learner_host.camera.frame_buffer import FrameBuffer
from ai_learner_host.application.event_bus import EventBus
from ai_learner_host.application.mission_controller import MissionController
from ai_learner_host.application.state_store import AppStateStore
from ai_learner_host.config import AppConfig
from ai_learner_host.domain.events import HeadingRequestedEvent


class FrameBufferFreshnessTests(unittest.TestCase):
    def test_sequence_increments_and_metadata_is_preserved(self):
        buffer = FrameBuffer()

        buffer.put(np.zeros((2, 2, 3), dtype=np.uint8))
        first = buffer.latest_packet()
        self.assertIsNotNone(first)

        buffer.put(np.ones((2, 2, 3), dtype=np.uint8))
        second = buffer.latest_packet()
        self.assertIsNotNone(second)

        self.assertEqual(first.sequence, 1)
        self.assertEqual(second.sequence, 2)
        self.assertGreaterEqual(
            second.timestamp_monotonic,
            first.timestamp_monotonic,
        )


class FakePerception:
    def __init__(self):
        self.request_times = []

    async def get_fresh_heading(self, *, requested_at_monotonic=None):
        self.request_times.append(requested_at_monotonic)
        await asyncio.sleep(0)
        return 91.25


class FakeRover:
    def __init__(self):
        self.headings = []

    async def send_heading(self, heading):
        self.headings.append(heading)


class FakeRepository:
    def __init__(self):
        self.rows = []

    async def log(self, run_id, level, category, message):
        self.rows.append((run_id, level, category, message))


class HeadingHandshakeTests(unittest.IsolatedAsyncioTestCase):
    async def test_heading_request_uses_fresh_heading_api(self):
        perception = FakePerception()
        rover = FakeRover()
        repository = FakeRepository()

        controller = MissionController(
            config=AppConfig(),
            event_bus=EventBus(),
            state_store=AppStateStore(),
            planner=None,
            perception=perception,
            validator=None,
            verifier=None,
            rover=rover,
            execution=None,
            repository=repository,
        )

        await controller._on_heading_requested(
            HeadingRequestedEvent({})
        )

        self.assertEqual(rover.headings, [91.25])
        self.assertEqual(len(perception.request_times), 1)
        self.assertIsNotNone(perception.request_times[0])
        self.assertTrue(
            any("fresh reply" in row[3] for row in repository.rows)
        )


if __name__ == "__main__":
    unittest.main()
