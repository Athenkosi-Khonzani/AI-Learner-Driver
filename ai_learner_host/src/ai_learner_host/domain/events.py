from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import TelemetrySample


@dataclass(slots=True, frozen=True)
class RoverConnectedEvent:
    uri: str


@dataclass(slots=True, frozen=True)
class RoverDisconnectedEvent:
    reason: str = ""


@dataclass(slots=True, frozen=True)
class CommandAckEvent:
    path_id: int
    command_index: int


@dataclass(slots=True, frozen=True)
class PathAckEvent:
    path_id: int


@dataclass(slots=True, frozen=True)
class NackEvent:
    path_id: int
    command_index: int | None
    reason: str


@dataclass(slots=True, frozen=True)
class TelemetryEvent:
    sample: TelemetrySample


@dataclass(slots=True, frozen=True)
class HeadingRequestedEvent:
    payload: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ProtocolErrorEvent:
    message: str
    raw: str = ""
