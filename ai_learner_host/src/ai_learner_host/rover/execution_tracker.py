from __future__ import annotations

from dataclasses import dataclass

from ai_learner_host.domain.models import ValidatedPlan


@dataclass(slots=True)
class ExecutionState:
    path_id: int | None = None
    total_commands: int = 0
    last_acked_index: int = -1
    completed: bool = False
    failed_reason: str = ""


class ExecutionTracker:
    def __init__(self) -> None:
        self.state = ExecutionState()

    def start(self, plan: ValidatedPlan) -> None:
        self.state = ExecutionState(
            path_id=plan.path_id,
            total_commands=len(plan.commands),
            last_acked_index=-1,
        )

    def command_ack(self, path_id: int, index: int) -> None:
        if path_id != self.state.path_id:
            return
        self.state.last_acked_index = max(self.state.last_acked_index, index)

    def path_ack(self, path_id: int) -> None:
        if path_id == self.state.path_id:
            self.state.completed = True

    def fail(self, path_id: int, reason: str) -> None:
        if path_id == self.state.path_id:
            self.state.failed_reason = reason
