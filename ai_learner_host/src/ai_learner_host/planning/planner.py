from __future__ import annotations

import asyncio
import json
from typing import Protocol

from ai_learner_host.domain.models import DriveCommand, PlanProposal, WorldStateSnapshot
from .llm_providers import StructuredLLMResult
from .prompt_builder import PromptBuilder


class LLMPort(Protocol):
    async def generate_plan(self, system_prompt: str, user_prompt: str) -> StructuredLLMResult: ...

    async def generate_verification(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredLLMResult: ...


class PlaceholderLLMAdapter:
    """Development-only deterministic adapter retained for unit/offline testing."""

    async def generate_plan(self, system_prompt: str, user_prompt: str) -> StructuredLLMResult:
        _ = system_prompt
        await asyncio.sleep(0.01)
        lower = user_prompt.lower()
        if "back" in lower or "reverse" in lower:
            cmds = [{"op": "BACK", "value": 250}, {"op": "STOP", "value": None}]
        elif "left" in lower and "turn" in lower:
            cmds = [{"op": "TURN", "value": -90}, {"op": "STOP", "value": None}]
        elif "right" in lower and "turn" in lower:
            cmds = [{"op": "TURN", "value": 90}, {"op": "STOP", "value": None}]
        else:
            cmds = [{"op": "FWD", "value": 300}, {"op": "STOP", "value": None}]
        data = {"cmds": cmds}
        return StructuredLLMResult(data, json.dumps(data), "placeholder", "placeholder-v1")

    async def generate_verification(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredLLMResult:
        _ = (system_prompt, user_prompt)
        data = {"success": True, "message": "placeholder verification", "cmds": []}
        return StructuredLLMResult(data, json.dumps(data), "placeholder", "placeholder-v1")


class PlannerService:
    def __init__(self, llm: LLMPort, prompt_builder: PromptBuilder) -> None:
        self.llm = llm
        self.prompt_builder = prompt_builder

    async def request_plan(
        self,
        user_prompt: str,
        world: WorldStateSnapshot,
    ) -> tuple[PlanProposal, str]:
        system, user = self.prompt_builder.build_planning_messages(user_prompt, world)
        result = await self.llm.generate_plan(system, user)
        proposal = self._parse(result.data)
        proposal.raw_response = result.raw_text
        enhanced = f"SYSTEM:\n{system}\n\n{user}"
        return proposal, enhanced

    @staticmethod
    def _parse(data: dict) -> PlanProposal:
        if not isinstance(data, dict) or not isinstance(data.get("cmds"), list):
            raise ValueError("Planner response must be an object containing a cmds array")
        commands: list[DriveCommand] = []
        for item in data["cmds"]:
            if isinstance(item, dict):
                op = str(item.get("op", "")).upper()
                value = item.get("value")
            elif isinstance(item, list) and item:
                # Accept legacy wire-shaped planner data during transition/testing.
                op = str(item[0]).upper()
                value = item[1] if len(item) > 1 else None
            else:
                raise ValueError("Each planner command must contain an operation")
            commands.append(DriveCommand(op, value))
        return PlanProposal(commands=commands)
