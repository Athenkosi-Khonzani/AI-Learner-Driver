from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel

from ai_learner_host.config import AppConfig
from .schemas import StructuredPlan, StructuredVerification


T = TypeVar("T", bound=BaseModel)


@dataclass(slots=True)
class StructuredLLMResult:
    data: dict[str, Any]
    raw_text: str
    provider: str
    model: str


MODEL_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("GPT-6 Sol", "openai", "gpt-6-sol"),
    ("Claude Sonnet 5", "anthropic", "claude-sonnet-5"),
    ("Gemini 3.1 Pro", "gemini", "gemini-3.1-pro-preview"),
)

MODEL_TO_PROVIDER = {model: provider for _label, provider, model in MODEL_OPTIONS}
PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def display_name(provider: str, model: str) -> str:
    for label, candidate_provider, candidate_model in MODEL_OPTIONS:
        if candidate_provider == provider and candidate_model == model:
            return label
    return model


class MultiProviderLLMAdapter:
    """One production planning port with three selectable strong-tier providers.

    API SDK imports are lazy so the desktop application can still open Settings and
    show a useful missing-dependency/API-key error instead of failing at import time.
    The same adapter instance is shared by planning and post-path goal verification.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @property
    def provider(self) -> str:
        return self.config.planner_provider

    @property
    def model(self) -> str:
        return self.config.planner_model

    @property
    def required_api_key_env(self) -> str:
        return PROVIDER_KEY_ENV.get(self.provider, "")

    def readiness(self) -> tuple[bool, str]:
        if self.model not in MODEL_TO_PROVIDER:
            return False, f"Unsupported planner model: {self.model}"
        expected_provider = MODEL_TO_PROVIDER[self.model]
        if self.provider != expected_provider:
            return False, (
                f"Planner provider/model mismatch: {self.provider!r} / {self.model!r}"
            )
        env_name = self.required_api_key_env
        if not env_name or not os.getenv(env_name):
            return False, f"{env_name or 'API key'} is not set"
        return True, f"{display_name(self.provider, self.model)} ready"

    async def generate_plan(self, system_prompt: str, user_prompt: str) -> StructuredLLMResult:
        return await self._generate(StructuredPlan, system_prompt, user_prompt)

    async def generate_verification(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredLLMResult:
        return await self._generate(StructuredVerification, system_prompt, user_prompt)

    async def _generate(
        self,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredLLMResult:
        ready, message = self.readiness()
        if not ready:
            raise RuntimeError(message)

        return await asyncio.to_thread(
            self._generate_sync,
            schema,
            system_prompt,
            user_prompt,
        )

    def _generate_sync(
        self,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredLLMResult:
        if self.provider == "openai":
            return self._openai(schema, system_prompt, user_prompt)
        if self.provider == "anthropic":
            return self._anthropic(schema, system_prompt, user_prompt)
        if self.provider == "gemini":
            return self._gemini(schema, system_prompt, user_prompt)
        raise RuntimeError(f"Unsupported planner provider: {self.provider}")

    def _openai(
        self,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredLLMResult:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use GPT-6 Sol") from exc

        client = OpenAI()
        response = client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            reasoning={"effort": "medium"},
            max_output_tokens=int(self.config.planner_max_output_tokens),
            text_format=schema,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no parsed structured output")
        data = parsed.model_dump()
        raw = response.output_text or json.dumps(data, ensure_ascii=False)
        return StructuredLLMResult(data, raw, self.provider, self.model)

    def _anthropic(
        self,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredLLMResult:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install the 'anthropic' package to use Claude Sonnet 5") from exc

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.model,
            max_tokens=int(self.config.planner_max_output_tokens),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            thinking={"type": "adaptive"},
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": schema.model_json_schema(),
                },
                "effort": "medium",
            },
        )
        text_blocks = [
            getattr(block, "text", "")
            for block in getattr(message, "content", []) or []
            if getattr(block, "type", None) == "text"
        ]
        raw = "".join(text_blocks).strip()
        if not raw:
            raise RuntimeError(
                f"Claude returned no text output (stop_reason={getattr(message, 'stop_reason', None)!r})"
            )
        try:
            data = schema.model_validate_json(raw).model_dump()
        except Exception as exc:
            raise RuntimeError(f"Claude structured output could not be parsed: {exc}") from exc
        return StructuredLLMResult(data, raw, self.provider, self.model)

    def _gemini(
        self,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredLLMResult:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install the 'google-genai' package to use Gemini 3.1 Pro") from exc

        client = genai.Client()
        response = client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
                max_output_tokens=int(self.config.planner_max_output_tokens),
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
            ),
        )
        raw = (response.text or "").strip()
        parsed_attr = getattr(response, "parsed", None)
        if isinstance(parsed_attr, dict):
            data = schema.model_validate(parsed_attr).model_dump()
        elif hasattr(parsed_attr, "model_dump"):
            data = schema.model_validate(parsed_attr.model_dump()).model_dump()
        elif raw:
            data = schema.model_validate_json(raw).model_dump()
        else:
            raise RuntimeError("Gemini returned no structured output")
        if not raw:
            raw = json.dumps(data, ensure_ascii=False)
        return StructuredLLMResult(data, raw, self.provider, self.model)
