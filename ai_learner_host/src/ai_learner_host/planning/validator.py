from __future__ import annotations

from itertools import count

from ai_learner_host.config import AppConfig
from ai_learner_host.domain.models import DriveCommand, PlanProposal, ValidatedPlan


class PlanValidationError(ValueError):
    pass


class PlanValidator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._path_ids = count(1)

    def validate(self, proposal: PlanProposal, *, source: str = "planner", revision: int = 0) -> ValidatedPlan:
        commands = proposal.commands
        if not commands:
            raise PlanValidationError("Plan is empty")
        if len(commands) > self.config.max_primitives:
            raise PlanValidationError(
                f"Plan has {len(commands)} primitives; maximum is {self.config.max_primitives}"
            )
        if commands[-1].op != "STOP":
            raise PlanValidationError("Plan must end in STOP")
        if any(command.op == "STOP" for command in commands[:-1]):
            raise PlanValidationError("STOP may only appear at the end of the path")

        normalized: list[DriveCommand] = []
        for command in commands:
            op = command.op.upper()
            if op not in {"FWD", "BACK", "TURN", "STOP"}:
                raise PlanValidationError(f"Unsupported operation: {op}")
            if op == "STOP":
                if command.value is not None:
                    raise PlanValidationError("STOP takes no value")
                normalized.append(DriveCommand("STOP"))
                continue
            if not isinstance(command.value, (int, float)):
                raise PlanValidationError(f"{op} requires a numeric value")
            value = float(command.value)
            if op == "FWD" and not (0 < value <= self.config.max_forward_mm):
                raise PlanValidationError(f"FWD distance must be 1..{self.config.max_forward_mm} mm")
            if op == "BACK" and not (0 < value <= self.config.max_backward_mm):
                raise PlanValidationError(f"BACK distance must be 1..{self.config.max_backward_mm} mm")
            if op == "TURN" and not (0 < abs(value) <= self.config.max_turn_deg):
                raise PlanValidationError(f"TURN magnitude must be 1..{self.config.max_turn_deg} degrees")
            normalized_value: int | float = int(value) if value.is_integer() else value
            normalized.append(DriveCommand(op, normalized_value))

        return ValidatedPlan(
            path_id=next(self._path_ids),
            commands=normalized,
            source=source,
            revision=revision,
        )
