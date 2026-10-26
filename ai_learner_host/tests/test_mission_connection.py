from __future__ import annotations

import unittest

from ai_learner_host.application.event_bus import EventBus
from ai_learner_host.application.mission_controller import MissionController
from ai_learner_host.application.state_store import AppStateStore
from ai_learner_host.config import AppConfig
from ai_learner_host.domain.events import RoverConnectedEvent


class FakeRover:
    def __init__(self):
        self.telemetry_calls = []

    async def set_telemetry(self, on: bool):
        self.telemetry_calls.append(on)


class MissionConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_connection_turns_telemetry_on(self):
        rover = FakeRover()
        state = AppStateStore()
        controller = MissionController(
            config=AppConfig(telemetry_on_start=True),
            event_bus=EventBus(),
            state_store=state,
            planner=None,
            perception=None,
            validator=None,
            verifier=None,
            rover=rover,
            execution=None,
            repository=None,
        )

        await controller._on_rover_connected(RoverConnectedEvent("ws://test"))

        self.assertTrue(state.state.rover_connected)
        self.assertEqual(rover.telemetry_calls, [True])
        self.assertTrue(any("telem" in line for line in state.state.run_log))


if __name__ == "__main__":
    unittest.main()
