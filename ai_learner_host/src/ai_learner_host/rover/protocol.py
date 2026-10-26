from __future__ import annotations

import json
from typing import Any

from ai_learner_host.domain.events import (
    CommandAckEvent,
    HeadingRequestedEvent,
    NackEvent,
    PathAckEvent,
    ProtocolErrorEvent,
    TelemetryEvent,
)
from ai_learner_host.domain.models import DriveCommand, TelemetrySample, ValidatedPlan


class ProtocolCodec:
    """Encodes/decodes the compact JSON WebSocket protocol.

    Current rover telemetry contract: vbat + soc + dist, where `dist` is the
    ultrasonic distance measurement reported by the rover. A legacy `cur`
    field is accepted only so older logs/firmware revisions do not crash the host.
    There is no emergency-stop message in this implementation.
    """

    @staticmethod
    def encode_path(plan: ValidatedPlan) -> str:
        cmds: list[list[Any]] = []
        for command in plan.commands:
            if command.value is None:
                cmds.append([command.op])
            else:
                cmds.append([command.op, command.value])
        return json.dumps({"t": "path", "id": plan.path_id, "cmds": cmds}, separators=(",", ":"))

    @staticmethod
    def encode_heading(deg: float) -> str:
        return json.dumps({"t": "heading", "deg": round(float(deg), 2)}, separators=(",", ":"))

    @staticmethod
    def encode_telemetry_switch(on: bool) -> str:
        return json.dumps({"t": "telem", "on": bool(on)}, separators=(",", ":"))

    @staticmethod
    def encode_speed(base: int) -> str:
        return json.dumps({"t": "speed", "base": int(base)}, separators=(",", ":"))

    @staticmethod
    def encode_pid(kp: float, ki: float, kd: float) -> str:
        return json.dumps({"t": "pid", "kp": kp, "ki": ki, "kd": kd}, separators=(",", ":"))

    @staticmethod
    def decode(raw: str):
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ProtocolErrorEvent(f"Invalid JSON from rover: {exc.msg}", raw)

        if not isinstance(message, dict):
            return ProtocolErrorEvent("Rover message is not a JSON object", raw)

        msg_type = message.get("t")
        try:
            if msg_type == "ack":
                path_id = int(message["id"])
                if message.get("of") == "cmd":
                    # The ESP32 main.cpp uses zero-based command indexes on the wire.
                    return CommandAckEvent(path_id=path_id, command_index=int(message["i"]))
                if message.get("of") == "path":
                    return PathAckEvent(path_id=path_id)
                return ProtocolErrorEvent("ACK has unknown 'of' field", raw)

            if msg_type == "nack":
                index = message.get("i")
                return NackEvent(
                    path_id=int(message["id"]),
                    command_index=int(index) if index is not None else None,
                    reason=str(message.get("reason", "unknown")),
                )

            if msg_type == "telem":
                return TelemetryEvent(
                    TelemetrySample(
                        vbat=float(message["vbat"]) if message.get("vbat") is not None else None,
                        soc=float(message["soc"]) if message.get("soc") is not None else None,
                        dist=float(message["dist"]) if message.get("dist") is not None else None,
                        cur=float(message["cur"]) if message.get("cur") is not None else None,
                    )
                )

            if msg_type == "req_heading":
                return HeadingRequestedEvent(payload=message)

            if msg_type == "invalid":
                detail = str(message.get("message", "invalid command"))
                return ProtocolErrorEvent(f"Rover rejected host message: {detail}", raw)

            return ProtocolErrorEvent(f"Unknown rover message type: {msg_type!r}", raw)
        except (KeyError, TypeError, ValueError) as exc:
            return ProtocolErrorEvent(f"Malformed {msg_type!r} message: {exc}", raw)


SUPPORTED_OPS = {"FWD", "BACK", "TURN", "STOP"}


def command_from_pair(pair: list[Any]) -> DriveCommand:
    if not pair or not isinstance(pair[0], str):
        raise ValueError("Command must be [op] or [op, value]")
    op = pair[0].upper()
    if op not in SUPPORTED_OPS:
        raise ValueError(f"Unsupported command {op}")
    value = pair[1] if len(pair) > 1 else None
    return DriveCommand(op, value)
