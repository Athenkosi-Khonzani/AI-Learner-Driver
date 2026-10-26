"""Builds the production host services without the host UI and instruments them.

The mission controller, perception, planner, validator, verifier, gateway and
repository are the host's own classes. Thin proxies observe what they do:

* PinnedPerception  - the next ``capture_world_state`` returns the frozen start
  state W0, so the planner sees exactly the state the simulator replays.
* ObservedRepository - forwards every write to the real repository and reports
  plans, snapshots, verdicts and the run's end to the trial runner.
* TimedPlanner / TimedVerifier - record latency, prompts and raw responses.
* ObservedGateway - time-stamps the moment each path is sent.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ai_learner_host.application.event_bus import EventBus
from ai_learner_host.application.mission_controller import MissionController
from ai_learner_host.application.state_store import AppStateStore
from ai_learner_host.camera.frame_buffer import FrameBuffer
from ai_learner_host.config import AppConfig, ConfigRepository
from ai_learner_host.domain.models import PlanProposal, DriveCommand, WorldStateSnapshot
from ai_learner_host.perception.service import PerceptionService
from ai_learner_host.persistence.sqlite_repository import RunRepository
from ai_learner_host.planning.llm_providers import MultiProviderLLMAdapter
from ai_learner_host.planning.planner import PlannerService
from ai_learner_host.planning.prompt_builder import PromptBuilder
from ai_learner_host.planning.validator import PlanValidator
from ai_learner_host.planning.verification import VerificationService
from ai_learner_host.rover.execution_tracker import ExecutionTracker
from ai_learner_host.rover.gateway import RoverGateway

from rdt._bootstrap import host_project_root
from rdt.config import ExperimentConfig

Listener = Callable[[str, dict[str, Any]], None]


class _Proxy:
    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class PinnedPerception(_Proxy):
    def __init__(self, inner: PerceptionService) -> None:
        super().__init__(inner)
        object.__setattr__(self, "_pinned", None)

    def pin(self, snapshot: WorldStateSnapshot | None) -> None:
        object.__setattr__(self, "_pinned", snapshot)

    async def capture_world_state(self) -> WorldStateSnapshot:
        pinned = self._pinned
        if pinned is not None:
            object.__setattr__(self, "_pinned", None)
            return pinned
        return await self._inner.capture_world_state()


class ObservedRepository(_Proxy):
    def __init__(self, inner: RunRepository, listener: Listener) -> None:
        super().__init__(inner)
        object.__setattr__(self, "_listener", listener)

    async def save_plan(self, run_id, plan):
        self._listener("plan", {"run_id": run_id, "plan": plan})
        await self._inner.save_plan(run_id, plan)

    async def save_snapshot(self, run_id, snapshot, role):
        self._listener("snapshot", {"run_id": run_id, "snapshot": snapshot, "role": role})
        await self._inner.save_snapshot(run_id, snapshot, role)

    async def save_verification(self, run_id, round_number, result):
        self._listener("verification_saved", {"run_id": run_id, "round": round_number, "result": result})
        await self._inner.save_verification(run_id, round_number, result)

    async def finish_run(self, run):
        await self._inner.finish_run(run)
        self._listener("finish", {"run_id": run.run_id, "run": run})

    async def log(self, run_id, level, component, message):
        self._listener("log", {"run_id": run_id, "level": level, "component": component, "message": message})
        await self._inner.log(run_id, level, component, message)


class TimedPlanner(_Proxy):
    def __init__(self, inner: PlannerService, listener: Listener) -> None:
        super().__init__(inner)
        object.__setattr__(self, "_listener", listener)
        object.__setattr__(self, "fixed_plan", None)

    def set_fixed_plan(self, commands: list[dict] | None) -> None:
        object.__setattr__(self, "fixed_plan", commands)

    async def request_plan(self, user_prompt: str, world: WorldStateSnapshot):
        self._listener("plan_request", {"prompt": user_prompt})
        if self.fixed_plan is not None:
            commands = [DriveCommand(str(c["op"]).upper(), c.get("value")) for c in self.fixed_plan]
            proposal = PlanProposal(commands=commands, raw_response="[fixed plan, no LLM]")
            self._listener("plan_response", {"proposal": proposal, "prompt_text": "[fixed plan]", "fixed": True})
            return proposal, "[fixed plan]"
        try:
            proposal, enhanced = await self._inner.request_plan(user_prompt, world)
        except Exception as exc:
            self._listener("plan_error", {"error": f"{type(exc).__name__}: {exc}"})
            raise
        self._listener("plan_response", {"proposal": proposal, "prompt_text": enhanced, "fixed": False})
        return proposal, enhanced


class TimedVerifier(_Proxy):
    def __init__(self, inner: VerificationService, listener: Listener) -> None:
        super().__init__(inner)
        object.__setattr__(self, "_listener", listener)

    async def verify(self, original_prompt, before, after, executed_plan):
        self._listener("verify_request", {"path_id": executed_plan.path_id})
        try:
            result = await self._inner.verify(original_prompt, before, after, executed_plan)
        except Exception as exc:
            self._listener("verify_error", {"error": f"{type(exc).__name__}: {exc}"})
            raise
        self._listener("verify_response", {"result": result, "path_id": executed_plan.path_id})
        return result


class ObservedGateway(_Proxy):
    """Time-stamps paths; can refuse further paths once a trial's round budget is used."""

    def __init__(self, inner: RoverGateway, listener: Listener) -> None:
        super().__init__(inner)
        object.__setattr__(self, "_listener", listener)
        object.__setattr__(self, "blocked", False)

    def set_blocked(self, value: bool) -> None:
        object.__setattr__(self, "blocked", bool(value))

    @property
    def connected(self) -> bool:
        return self._inner.connected

    async def send_path(self, plan):
        if self.blocked:
            self._listener("path_blocked", {"path_id": plan.path_id, "commands": plan.commands})
            return
        self._listener("path_sent", {"path_id": plan.path_id, "commands": plan.commands})
        await self._inner.send_path(plan)


def fingerprint_tree(root: Path, pattern: str = "*.py") -> str:
    h = hashlib.sha256()
    for path in sorted(root.rglob(pattern)):
        if "__pycache__" in path.parts:
            continue
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def system_prompt_sha256(config: AppConfig) -> tuple[str, str]:
    """SHA-256 of the planner's system prompt (it depends only on configuration)."""
    system, _ = PromptBuilder(config).build_planning_messages("x", WorldStateSnapshot(scene_json={}))
    return hashlib.sha256(system.encode("utf-8")).hexdigest(), system


@dataclass
class RuntimeParts:
    """Optional replacements, used by the simulated rig and the tests."""

    host_config: AppConfig | None = None
    llm: Any = None
    frame_buffer: FrameBuffer | None = None
    skip_overrides: list[str] = field(default_factory=list)


class ExperimentRuntime:
    """The production host stack, instrumented for the Rover Driving Test."""

    def __init__(self, exp_config: ExperimentConfig, session_dir: Path, parts: RuntimeParts | None = None) -> None:
        parts = parts or RuntimeParts()
        self.exp_config = exp_config
        self.session_dir = Path(session_dir)
        self.listeners: list[Listener] = []
        if parts.host_config is not None:
            self.config = parts.host_config
            self.settings_path = None
        else:
            self.settings_path = host_project_root() / "data" / "settings.json"
            self.config = ConfigRepository(self.settings_path).load()
        self.overrides_applied = exp_config.apply_to_host(self.config, skip=parts.skip_overrides)
        self.state_store = AppStateStore()
        self.event_bus = EventBus()
        self.frame_buffer = parts.frame_buffer or FrameBuffer()
        self.perception_inner = PerceptionService(self.frame_buffer, self.config)
        self.perception = PinnedPerception(self.perception_inner)
        self.prompt_builder = PromptBuilder(self.config)
        self.llm = parts.llm if parts.llm is not None else MultiProviderLLMAdapter(self.config)
        self.planner = TimedPlanner(PlannerService(self.llm, self.prompt_builder), self._emit)
        self.validator = PlanValidator(self.config)
        self.verifier = TimedVerifier(VerificationService(self.llm, self.prompt_builder), self._emit)
        db_dir = self.session_dir / "host_db"
        db_dir.mkdir(parents=True, exist_ok=True)
        self.repository = ObservedRepository(
            RunRepository(db_dir / "host_runs.sqlite3", db_dir / "snapshots"), self._emit
        )
        self.gateway = RoverGateway(
            self.event_bus,
            reconnect_seconds=self.config.reconnect_seconds,
            auto_reconnect=self.config.auto_reconnect,
        )
        self.rover = ObservedGateway(self.gateway, self._emit)
        self.execution = ExecutionTracker()
        self.controller = MissionController(
            config=self.config,
            event_bus=self.event_bus,
            state_store=self.state_store,
            planner=self.planner,
            perception=self.perception,
            validator=self.validator,
            verifier=self.verifier,
            rover=self.rover,
            execution=self.execution,
            repository=self.repository,
        )
        self.host_fingerprint = fingerprint_tree(Path(__import__("ai_learner_host").__file__).parent)
        self.prompt_sha256, self.system_prompt = system_prompt_sha256(self.config)

    # ------------------------------------------------------------------ events
    def add_listener(self, listener: Listener) -> None:
        self.listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        if listener in self.listeners:
            self.listeners.remove(listener)

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        for listener in list(self.listeners):
            listener(kind, payload)

    # ------------------------------------------------------------------ lifecycle
    async def start(self, rover_uri: str | None = None) -> None:
        await self.controller.initialize()
        await self.gateway.start(rover_uri or self.config.rover_uri)

    async def stop(self) -> None:
        await self.gateway.stop()

    def llm_readiness(self) -> tuple[bool, str]:
        readiness = getattr(self.llm, "readiness", None)
        return readiness() if readiness else (True, "planner ready")

    def planner_label(self) -> str:
        provider = getattr(self.llm, "provider", self.config.planner_provider)
        model = getattr(self.llm, "model", self.config.planner_model)
        return f"{provider}/{model}"

    def describe(self) -> dict[str, Any]:
        return {
            "host_fingerprint_sha256": self.host_fingerprint,
            "planner_system_prompt_sha256": self.prompt_sha256,
            "planner": self.planner_label(),
            "settings_path": str(self.settings_path) if self.settings_path else "synthetic",
            "host_overrides_changed": copy.deepcopy(self.overrides_applied),
            "rover_uri": self.config.rover_uri,
            "camera_url": self.config.camera_url,
            "arena_mm": [self.config.calibration.arena_width_mm, self.config.calibration.arena_height_mm],
        }
