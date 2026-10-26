from __future__ import annotations

from enum import Enum, auto


class MissionState(Enum):
    IDLE = auto()
    ACQUIRING_PROMPT = auto()
    PREPARING_PROMPT = auto()
    CAPTURING_WORLD = auto()
    PLANNING = auto()
    VALIDATING_PLAN = auto()
    AWAITING_APPROVAL = auto()
    SENDING_PATH = auto()
    EXECUTING = auto()
    REPLANNING = auto()
    VERIFYING = auto()
    COMPLETED = auto()
    NEEDS_ATTENTION = auto()
    CANCELLED = auto()
    FAILED = auto()


class RunStatus(Enum):
    RUNNING = "RUNNING"
    DONE = "DONE"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class InputMode(Enum):
    TEXT = "TEXT"
    VOICE = "VOICE"
