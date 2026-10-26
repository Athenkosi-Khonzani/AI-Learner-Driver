from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StructuredCommand(BaseModel):
    """Provider-neutral command schema used by all production LLM adapters."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["TURN", "FWD", "BACK", "STOP"]
    value: float | None = None


class StructuredPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmds: list[StructuredCommand] = Field(min_length=1)


class StructuredVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    message: str
    # Always present. Use an empty array when no corrective path is needed.
    cmds: list[StructuredCommand]
