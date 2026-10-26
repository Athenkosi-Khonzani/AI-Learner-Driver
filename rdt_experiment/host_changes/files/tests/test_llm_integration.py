from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

from ai_learner_host.config import AppConfig
from ai_learner_host.domain.models import WorldStateSnapshot
from ai_learner_host.planning.llm_providers import (
    MODEL_OPTIONS,
    MultiProviderLLMAdapter,
    display_name,
)
from ai_learner_host.planning.planner import PlaceholderLLMAdapter, PlannerService
from ai_learner_host.planning.prompt_builder import PromptBuilder
from ai_learner_host.planning.verification import VerificationService


class PlannerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_placeholder_plan_still_exercises_service_without_network(self):
        config = AppConfig()
        service = PlannerService(PlaceholderLLMAdapter(), PromptBuilder(config))
        world = WorldStateSnapshot(
            scene_json={
                "arena": {"width_mm": 1000, "height_mm": 1000},
                "rover": {
                    "pose": {"x_mm": 100, "y_mm": 100, "heading_deg": 0},
                    "footprint_mm": {"length": 155.5, "width": 121.3},
                },
                "obstacles": [],
            }
        )
        proposal, enhanced = await service.request_plan("move forward", world)
        self.assertEqual(proposal.commands[-1].op, "STOP")
        self.assertIn("155.5 mm", enhanced)
        self.assertIn("121.3 mm", enhanced)
        self.assertNotIn("26 cm", enhanced)
        self.assertNotIn("20 cm", enhanced)

    async def test_verifier_can_return_correction(self):
        class FakeLLM(PlaceholderLLMAdapter):
            async def generate_verification(self, system_prompt, user_prompt):
                from ai_learner_host.planning.llm_providers import StructuredLLMResult

                data = {
                    "success": False,
                    "message": "move slightly forward",
                    "cmds": [
                        {"op": "FWD", "value": 50},
                        {"op": "STOP", "value": None},
                    ],
                }
                return StructuredLLMResult(data, str(data), "fake", "fake")

        config = AppConfig()
        prompt_builder = PromptBuilder(config)
        verifier = VerificationService(FakeLLM(), prompt_builder)
        from ai_learner_host.domain.models import DriveCommand, ValidatedPlan

        before = WorldStateSnapshot(scene_json={"arena": {}, "rover": {}, "obstacles": []})
        after = WorldStateSnapshot(scene_json={"arena": {}, "rover": {}, "obstacles": []})
        result = await verifier.verify(
            "move to target",
            before,
            after,
            ValidatedPlan(1, [DriveCommand("FWD", 100), DriveCommand("STOP")]),
        )
        self.assertFalse(result.success)
        self.assertIsNotNone(result.correction)
        self.assertEqual(result.correction.commands[-1].op, "STOP")


class ModelSelectionTests(unittest.TestCase):
    def test_three_strong_tier_models_are_available(self):
        self.assertEqual(
            [(provider, model) for _label, provider, model in MODEL_OPTIONS],
            [
                ("openai", "gpt-6-sol"),
                ("anthropic", "claude-sonnet-5"),
                ("gemini", "gemini-3.1-pro-preview"),
            ],
        )
        self.assertEqual(display_name("openai", "gpt-6-sol"), "GPT-6 Sol")

    def test_gpt6_sol_is_default(self):
        config = AppConfig()
        self.assertEqual(config.planner_provider, "openai")
        self.assertEqual(config.planner_model, "gpt-6-sol")

    def test_readiness_checks_selected_provider_key(self):
        config = AppConfig(planner_provider="anthropic", planner_model="claude-sonnet-5")
        with patch.dict(os.environ, {}, clear=True):
            ready, message = MultiProviderLLMAdapter(config).readiness()
        self.assertFalse(ready)
        self.assertIn("ANTHROPIC_API_KEY", message)


if __name__ == "__main__":
    unittest.main()
