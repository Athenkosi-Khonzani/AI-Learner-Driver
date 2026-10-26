from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

from .enums import InputMode, MissionState, RunStatus


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(slots=True, frozen=True)
class DriveCommand:
    op: str
    value: float | int | None = None

    def display(self) -> str:
        if self.op in {"FWD", "BACK"} and self.value is not None:
            label = "FORWARD" if self.op == "FWD" else "BACKWARD"
            return f"{label} {int(self.value)} mm"
        if self.op == "TURN" and self.value is not None:
            sign = "+" if float(self.value) > 0 else ""
            return f"TURN {sign}{self.value:g}°"
        return self.op


@dataclass(slots=True)
class PlanProposal:
    commands: list[DriveCommand]
    raw_response: str = ""
    rationale: str = ""


@dataclass(slots=True)
class ValidatedPlan:
    path_id: int
    commands: list[DriveCommand]
    source: str = "planner"
    revision: int = 0


@dataclass(slots=True)
class RoverPose:
    x_mm: float | None = None
    y_mm: float | None = None
    heading_deg: float | None = None
    confidence: float = 0.0


@dataclass(slots=True)
class DetectedObject:
    label: str
    x_mm: float | None = None
    y_mm: float | None = None
    confidence: float = 0.0


@dataclass(slots=True)
class WorldStateSnapshot:
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=utc_now_iso)
    raw_frame: Any = None
    annotated_frame: Any = None
    rover_pose: RoverPose = field(default_factory=RoverPose)
    detected_objects: list[DetectedObject] = field(default_factory=list)
    obstacles: list[DetectedObject] = field(default_factory=list)
    arena_width_mm: float | None = None
    arena_height_mm: float | None = None
    scene_json: dict[str, Any] = field(default_factory=dict)
    scene_markdown: str = ""
    scene_text_map: str = ""


@dataclass(slots=True)
class TelemetrySample:
    vbat: float | None = None
    soc: float | None = None
    # Production telemetry field: ultrasonic distance reported by the rover.
    dist: float | None = None
    # Legacy compatibility only. Older host/firmware revisions used `cur`.
    cur: float | None = None
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class VerificationResult:
    success: bool
    correction: PlanProposal | None = None
    message: str = ""


@dataclass(slots=True)
class MissionRun:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: str = ""
    input_mode: InputMode = InputMode.TEXT
    status: RunStatus = RunStatus.RUNNING
    state: MissionState = MissionState.IDLE
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str | None = None
    current_path_id: int | None = None
    current_command_index: int = -1
    correction_round: int = 0
    revision: int = 0


@dataclass(slots=True)
class RecentRunView:
    run_id: str
    prompt: str
    status: str
    input_mode: str
    timestamp: str
