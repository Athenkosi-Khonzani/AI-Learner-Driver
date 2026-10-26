from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from ai_learner_host.domain.enums import MissionState
from ai_learner_host.domain.models import DriveCommand, RecentRunView, TelemetrySample


@dataclass(slots=True)
class AppState:
    rover_connected: bool = False
    rover_status_text: str = "ROVER OFFLINE"
    camera_connected: bool = False
    camera_status_text: str = "CAMERA OFFLINE"
    llm_ready: bool = False
    llm_status_text: str = "LLM NOT CHECKED"
    voice_ready: bool = False
    voice_busy: bool = False
    voice_status_text: str = "VOICE NOT CHECKED"
    mission_state: MissionState = MissionState.IDLE
    current_run_id: str | None = None
    current_prompt: str = ""
    current_plan: list[DriveCommand] = field(default_factory=list)
    current_command_index: int = -1
    correction_round: int = 0
    telemetry: TelemetrySample = field(default_factory=TelemetrySample)
    run_log: list[str] = field(default_factory=list)
    recent_runs: list[RecentRunView] = field(default_factory=list)
    processed_frame: Any = None
    world_summary: str = "No processed world-state yet."
    pending_approval: bool = False
    manual_busy: bool = False
    last_error: str = ""


class AppStateStore:
    def __init__(self) -> None:
        self._state = AppState()
        self._subscribers: list[Callable[[AppState], None]] = []

    @property
    def state(self) -> AppState:
        return self._state

    def subscribe(self, callback: Callable[[AppState], None], *, fire_immediately: bool = True) -> None:
        self._subscribers.append(callback)
        if fire_immediately:
            callback(self.snapshot())

    def snapshot(self) -> AppState:
        # Lists are copied to prevent UI code accidentally mutating backend state.
        return replace(
            self._state,
            current_plan=list(self._state.current_plan),
            run_log=list(self._state.run_log),
            recent_runs=list(self._state.recent_runs),
        )

    def update(self, **changes: Any) -> None:
        for key, value in changes.items():
            if not hasattr(self._state, key):
                raise AttributeError(f"Unknown state field: {key}")
            setattr(self._state, key, value)
        self._notify()

    def append_log(self, message: str, *, limit: int = 300) -> None:
        self._state.run_log.append(message)
        if len(self._state.run_log) > limit:
            self._state.run_log = self._state.run_log[-limit:]
        self._notify()

    def _notify(self) -> None:
        snap = self.snapshot()
        for callback in tuple(self._subscribers):
            callback(snap)
