from __future__ import annotations

import json

from ai_learner_host.config import AppConfig
from ai_learner_host.domain.models import WorldStateSnapshot


class PromptBuilder:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_planning_messages(
        self,
        user_prompt: str,
        world: WorldStateSnapshot,
    ) -> tuple[str, str]:
        system = f"""You are the high-level motion planner for the AI Learner Rover Driver.

Plan a short, safe, executable path from the supplied user instruction and structured world state.
Use only the supplied information. Do not use tools, web search, or outside knowledge.

ROVER PHYSICAL FOOTPRINT:
- Length / front-to-back: {self.config.rover_length_mm:.1f} mm.
- Width / left-to-right: {self.config.rover_width_mm:.1f} mm.
- Plan for the FULL rectangular rover body, not only the rover centre.
- Keep the full footprint inside the arena and clear of every obstacle.
- Where geometry allows, preserve approximately {self.config.planning_clearance_cm * 10:.0f} mm of extra clearance beyond the rover body.

COORDINATE FRAME:
- Arena origin is the bottom-left corner.
- +X points right; +Y points up.
- Heading 0° points +Y/up.
- Heading 90° points +X/right.
- Heading 180° points down.
- Heading 270° points left.

ROVER COMMAND SEMANTICS:
- FWD: move forward by the supplied distance in millimetres.
- BACK: move backward by the supplied distance in millimetres.
- TURN: in-place signed rotation in degrees; positive = right/clockwise, negative = left/anticlockwise.
- STOP: terminal path marker with no value.

HOST SAFETY / PROTOCOL LIMITS:
- Maximum {self.config.max_primitives} commands total, including STOP.
- FWD must be >0 and <= {self.config.max_forward_mm} mm.
- BACK must be >0 and <= {self.config.max_backward_mm} mm.
- TURN magnitude must be >0 and <= {self.config.max_turn_deg}°.
- STOP must appear exactly once and must be the final command.
- The host assigns the path id and transport metadata. Never invent them.
- Prefer a simple route, but safety and task completion take priority over shortest distance.

Return only the structured command-plan response required by the API schema. Do not add explanations."""

        user = (
            "USER INSTRUCTION:\n"
            + user_prompt.strip()
            + "\n\nWORLD STATE JSON:\n"
            + json.dumps(world.scene_json, indent=2, sort_keys=True)
        )
        return system, user

    # Backwards-compatible helper useful for logging/debugging.
    def build_planning_prompt(self, user_prompt: str, world: WorldStateSnapshot) -> str:
        system, user = self.build_planning_messages(user_prompt, world)
        return f"SYSTEM:\n{system}\n\n{user}"

    def build_verification_messages(
        self,
        original_prompt: str,
        before: WorldStateSnapshot,
        after: WorldStateSnapshot,
        executed_plan_text: str,
    ) -> tuple[str, str]:
        system = f"""You verify whether the AI Learner Rover completed a requested driving task.

Use only the before/after structured world states, the original instruction, and the executed plan.
The rover footprint is {self.config.rover_length_mm:.0f} mm long × {self.config.rover_width_mm:.0f} mm wide.

If the task is complete and safely satisfied, return success=true and an empty cmds array.
If the task is not complete, return success=false and, only when a safe small correction is clear, return that correction path using FWD, BACK, TURN and final STOP.
Do not fabricate perception information. If the after-state is insufficient to judge success, return success=false with an empty cmds array and explain that verification is uncertain.

Correction limits are the same as normal planning: at most {self.config.max_primitives} commands, FWD <= {self.config.max_forward_mm} mm, BACK <= {self.config.max_backward_mm} mm, |TURN| <= {self.config.max_turn_deg}°, STOP final.
Return only the structured verification response required by the API schema."""

        user = f"""ORIGINAL USER INSTRUCTION:
{original_prompt.strip()}

EXECUTED PLAN:
{executed_plan_text}

BEFORE WORLD STATE JSON:
{json.dumps(before.scene_json, indent=2, sort_keys=True)}

AFTER WORLD STATE JSON:
{json.dumps(after.scene_json, indent=2, sort_keys=True)}"""
        return system, user

    def build_verification_prompt(
        self,
        original_prompt: str,
        before: WorldStateSnapshot,
        after: WorldStateSnapshot,
        executed_plan_text: str,
    ) -> str:
        system, user = self.build_verification_messages(
            original_prompt,
            before,
            after,
            executed_plan_text,
        )
        return f"SYSTEM:\n{system}\n\n{user}"
