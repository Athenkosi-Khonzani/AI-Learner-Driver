import asyncio
import unittest

import websockets

from tools.fake_rover import FakeRover
from ai_learner_host.application.event_bus import EventBus
from ai_learner_host.domain.events import (
    CommandAckEvent,
    HeadingRequestedEvent,
    PathAckEvent,
    RoverConnectedEvent,
    TelemetryEvent,
)
from ai_learner_host.domain.models import DriveCommand, ValidatedPlan
from ai_learner_host.rover.gateway import RoverGateway


class GatewayIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_bidirectional_protocol_flow(self):
        fake = FakeRover()
        bus = EventBus()
        gateway = RoverGateway(bus, reconnect_seconds=0.1)
        seen = {"connected": False, "telem": 0, "ack": 0, "path": False, "heading": 0}

        async def connected(_event):
            seen["connected"] = True

        async def telem(_event):
            seen["telem"] += 1

        async def ack(_event):
            seen["ack"] += 1

        async def path(_event):
            seen["path"] = True

        async def heading(_event):
            seen["heading"] += 1
            await gateway.send_heading(90.0)

        bus.subscribe(RoverConnectedEvent, connected)
        bus.subscribe(TelemetryEvent, telem)
        bus.subscribe(CommandAckEvent, ack)
        bus.subscribe(PathAckEvent, path)
        bus.subscribe(HeadingRequestedEvent, heading)

        async with websockets.serve(fake.handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            await gateway.start(f"ws://127.0.0.1:{port}")
            for _ in range(50):
                if seen["connected"]:
                    break
                await asyncio.sleep(0.02)
            await gateway.set_telemetry(True)
            await gateway.send_path(
                ValidatedPlan(
                    1,
                    [DriveCommand("TURN", 90), DriveCommand("FWD", 100), DriveCommand("STOP")],
                )
            )
            for _ in range(120):
                if seen["path"] and seen["telem"]:
                    break
                await asyncio.sleep(0.02)
            await gateway.stop()

        self.assertTrue(seen["connected"])
        self.assertGreaterEqual(seen["telem"], 1)
        self.assertEqual(seen["ack"], 3)
        self.assertTrue(seen["path"])
        self.assertEqual(seen["heading"], 3)


if __name__ == "__main__":
    unittest.main()
