from __future__ import annotations

from ai_learner_host.domain.models import (
    DriveCommand,
    PlanProposal,
    ValidatedPlan,
    VerificationResult,
    WorldStateSnapshot,
)
from .planner import LLMPort
from .prompt_builder import PromptBuilder


class VerificationService:
    """LLM + perception post-path verifier using the same selected strong-tier model."""

    def __init__(self, llm: LLMPort, prompt_builder: PromptBuilder) -> None:
        self.llm = llm
        self.prompt_builder = prompt_builder

    async def verify(
        self,
        original_prompt: str,
        before: WorldStateSnapshot,
        after: WorldStateSnapshot,
        executed_plan: ValidatedPlan,
    ) -> VerificationResult:
        executed_plan_text = " -> ".join(command.display() for command in executed_plan.commands)
        system, user = self.prompt_builder.build_verification_messages(
            original_prompt,
            before,
            after,
            executed_plan_text,
        )
        result = await self.llm.generate_verification(system, user)
        data = result.data
        success = bool(data.get("success"))
        message = str(data.get("message") or ("goal verified" if success else "goal not yet achieved"))

        if success:
            return VerificationResult(success=True, message=message)

        correction_data = data.get("cmds")
        correction = None
        if isinstance(correction_data, list) and correction_data:
            commands: list[DriveCommand] = []
            for item in correction_data:
                if not isinstance(item, dict):
                    raise ValueError("Verification correction command must be an object")
                commands.append(
                    DriveCommand(
                        str(item.get("op", "")).upper(),
                        item.get("value"),
                    )
                )
            correction = PlanProposal(commands=commands, raw_response=result.raw_text)

        return VerificationResult(
            success=False,
            correction=correction,
            message=message,
        )
