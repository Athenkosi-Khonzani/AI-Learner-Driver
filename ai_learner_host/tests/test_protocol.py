import json
import unittest

from ai_learner_host.domain.events import (
    CommandAckEvent,
    HeadingRequestedEvent,
    NackEvent,
    PathAckEvent,
    TelemetryEvent,
)
from ai_learner_host.domain.models import DriveCommand, ValidatedPlan
from ai_learner_host.rover.protocol import ProtocolCodec


class ProtocolCodecTests(unittest.TestCase):
    def test_encode_path_uses_distance(self):
        plan = ValidatedPlan(
            7,
            [DriveCommand("FWD", 200), DriveCommand("TURN", -30), DriveCommand("STOP")],
        )
        data = json.loads(ProtocolCodec.encode_path(plan))
        self.assertEqual(data["t"], "path")
        self.assertEqual(data["id"], 7)
        self.assertEqual(data["cmds"], [["FWD", 200], ["TURN", -30], ["STOP"]])

    def test_decode_command_ack_uses_zero_based_wire_index(self):
        event = ProtocolCodec.decode('{"t":"ack","of":"cmd","id":7,"i":1}')
        self.assertIsInstance(event, CommandAckEvent)
        self.assertEqual(event.command_index, 1)

    def test_decode_production_ultrasonic_distance_telemetry(self):
        event = ProtocolCodec.decode('{"t":"telem","vbat":7.82,"soc":74,"dist":14}')
        self.assertIsInstance(event, TelemetryEvent)
        self.assertEqual(event.sample.vbat, 7.82)
        self.assertEqual(event.sample.soc, 74)
        self.assertEqual(event.sample.dist, 14)
        self.assertIsNone(event.sample.cur)

    def test_decode_legacy_current_telemetry_without_breaking(self):
        event = ProtocolCodec.decode('{"t":"telem","vbat":7.82,"soc":74,"cur":1.4}')
        self.assertIsInstance(event, TelemetryEvent)
        self.assertEqual(event.sample.cur, 1.4)
        self.assertIsNone(event.sample.dist)


    def test_decode_all_rover_message_types(self):
        path = ProtocolCodec.decode('{"t":"ack","of":"path","id":7}')
        self.assertIsInstance(path, PathAckEvent)

        nack = ProtocolCodec.decode('{"t":"nack","id":7,"i":2,"reason":"obstacle"}')
        self.assertIsInstance(nack, NackEvent)
        self.assertEqual(nack.command_index, 2)
        self.assertEqual(nack.reason, "obstacle")

        heading = ProtocolCodec.decode('{"t":"req_heading"}')
        self.assertIsInstance(heading, HeadingRequestedEvent)

    def test_decode_rover_invalid_message_is_protocol_error(self):
        from ai_learner_host.domain.events import ProtocolErrorEvent
        event = ProtocolCodec.decode('{"t":"invalid","message":"invalid command"}')
        self.assertIsInstance(event, ProtocolErrorEvent)
        self.assertIn("Rover rejected host message", event.message)

    def test_encode_all_host_message_types(self):
        self.assertEqual(json.loads(ProtocolCodec.encode_heading(92.5)), {"t": "heading", "deg": 92.5})
        self.assertEqual(json.loads(ProtocolCodec.encode_telemetry_switch(True)), {"t": "telem", "on": True})
        self.assertEqual(json.loads(ProtocolCodec.encode_speed(180)), {"t": "speed", "base": 180})
        self.assertEqual(
            json.loads(ProtocolCodec.encode_pid(1.2, 0.0, 0.4)),
            {"t": "pid", "kp": 1.2, "ki": 0.0, "kd": 0.4},
        )


if __name__ == "__main__":
    unittest.main()
