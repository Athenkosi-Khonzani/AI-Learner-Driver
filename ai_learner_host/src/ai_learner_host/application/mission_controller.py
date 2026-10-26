from __future__ import annotations

import asyncio
from datetime import datetime
import time

from ai_learner_host.application.event_bus import EventBus
from ai_learner_host.application.state_store import AppStateStore
from ai_learner_host.config import AppConfig
from ai_learner_host.domain.enums import InputMode, MissionState, RunStatus
from ai_learner_host.domain.events import (
    CommandAckEvent,
    HeadingRequestedEvent,
    NackEvent,
    PathAckEvent,
    ProtocolErrorEvent,
    RoverConnectedEvent,
    RoverDisconnectedEvent,
    TelemetryEvent,
)
from ai_learner_host.domain.models import DriveCommand, MissionRun, PlanProposal, ValidatedPlan, utc_now_iso
from ai_learner_host.perception.service import PerceptionService
from ai_learner_host.persistence.sqlite_repository import RunRepository
from ai_learner_host.planning.planner import PlannerService
from ai_learner_host.planning.validator import PlanValidationError, PlanValidator
from ai_learner_host.planning.verification import VerificationService
from ai_learner_host.rover.execution_tracker import ExecutionTracker
from ai_learner_host.rover.gateway import RoverGateway


class MissionController:
    MAX_CORRECTION_ROUNDS = 2

    def __init__(
        self,
        *,
        config: AppConfig,
        event_bus: EventBus,
        state_store: AppStateStore,
        planner: PlannerService,
        perception: PerceptionService,
        validator: PlanValidator,
        verifier: VerificationService,
        rover: RoverGateway,
        execution: ExecutionTracker,
        repository: RunRepository,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.state_store = state_store
        self.planner = planner
        self.perception = perception
        self.validator = validator
        self.verifier = verifier
        self.rover = rover
        self.execution = execution
        self.repository = repository
        self.current_run: MissionRun | None = None
        self.current_plan: ValidatedPlan | None = None
        self.initial_world = None
        self.pending_plan: ValidatedPlan | None = None
        self._mission_lock = asyncio.Lock()
        self._manual_path_id: int | None = None
        # Every path sent in the current mission, with how much of it was executed.
        # The verifier is given all of it, because it compares the mission's
        # original before-state with the latest after-state.
        self._executed_rounds: list[dict] = []
        self._subscribe_events()

    def _subscribe_events(self) -> None:
        self.event_bus.subscribe(RoverConnectedEvent, self._on_rover_connected)
        self.event_bus.subscribe(RoverDisconnectedEvent, self._on_rover_disconnected)
        self.event_bus.subscribe(CommandAckEvent, self._on_command_ack)
        self.event_bus.subscribe(PathAckEvent, self._on_path_ack)
        self.event_bus.subscribe(NackEvent, self._on_nack)
        self.event_bus.subscribe(TelemetryEvent, self._on_telemetry)
        self.event_bus.subscribe(HeadingRequestedEvent, self._on_heading_requested)
        self.event_bus.subscribe(ProtocolErrorEvent, self._on_protocol_error)

    async def initialize(self) -> None:
        await self.refresh_recent_runs()

    async def submit_prompt(self, prompt: str, input_mode: InputMode = InputMode.TEXT) -> None:
        prompt = prompt.strip()
        if not prompt:
            return
        async with self._mission_lock:
            if self.current_run and self.current_run.status == RunStatus.RUNNING:
                raise RuntimeError("A mission is already active")
            run = MissionRun(prompt=prompt, input_mode=input_mode, state=MissionState.ACQUIRING_PROMPT)
            self.current_run = run
            self.current_plan = None
            self.pending_plan = None
            self._executed_rounds = []
            await self.repository.create_run(run)
            await self.refresh_recent_runs()
            self.state_store.update(
                current_run_id=run.run_id,
                current_prompt=prompt,
                current_plan=[],
                current_command_index=-1,
                correction_round=0,
                pending_approval=False,
                mission_state=MissionState.ACQUIRING_PROMPT,
                last_error="",
                run_log=[],
            )
            self._log_ui("prompt received")
            asyncio.create_task(self._plan_new_mission(run), name=f"mission-{run.run_id}")

    async def approve_pending_plan(self) -> None:
        if not self.current_run or not self.pending_plan:
            return
        plan = self.pending_plan
        self.pending_plan = None
        self.state_store.update(pending_approval=False)
        await self._send_plan(plan)

    async def reject_pending_plan(self) -> None:
        if not self.current_run:
            return
        await self._finish(RunStatus.STOPPED, MissionState.CANCELLED, "plan rejected by operator")

    async def manual_drive(self, op: str) -> None:
        if self.current_run and self.current_run.status == RunStatus.RUNNING:
            raise RuntimeError("Manual drive is disabled during an autonomous mission")
        if self._manual_path_id is not None:
            raise RuntimeError("A manual jog is already executing")
        if not self.rover.connected:
            raise RuntimeError("Rover is not connected")
        op = op.upper()
        if op == "FWD":
            proposal = PlanProposal([DriveCommand("FWD", self.config.manual_jog_mm), DriveCommand("STOP")])
        elif op == "BACK":
            proposal = PlanProposal([DriveCommand("BACK", self.config.manual_jog_mm), DriveCommand("STOP")])
        elif op == "LEFT":
            proposal = PlanProposal([DriveCommand("TURN", -self.config.manual_turn_deg), DriveCommand("STOP")])
        elif op == "RIGHT":
            proposal = PlanProposal([DriveCommand("TURN", self.config.manual_turn_deg), DriveCommand("STOP")])
        else:
            raise ValueError(op)
        plan = self.validator.validate(proposal, source="manual")
        self._manual_path_id = plan.path_id
        self.execution.start(plan)
        self.state_store.update(
            current_plan=plan.commands, current_command_index=-1, manual_busy=True
        )
        await self.rover.send_path(plan)
        self._log_ui(f"manual tx · {plan.commands[0].display()}")

    async def _plan_new_mission(self, run: MissionRun) -> None:
        try:
            self._set_state(MissionState.PREPARING_PROMPT)
            self._log_ui("prompt context prepared")

            self._set_state(MissionState.CAPTURING_WORLD)
            self._log_ui("capturing current world state")
            world = await self.perception.capture_world_state()
            self.initial_world = world
            await self.repository.save_snapshot(run.run_id, world, "before")
            self.state_store.update(processed_frame=world.annotated_frame, world_summary=world.scene_markdown)

            self._set_state(MissionState.PLANNING)
            self._log_ui("requesting planner")
            proposal, _enhanced = await self.planner.request_plan(run.prompt, world)

            self._set_state(MissionState.VALIDATING_PLAN)
            plan = self.validator.validate(proposal, source="planner", revision=run.revision)
            self.current_plan = plan
            await self.repository.save_plan(run.run_id, plan)
            self.state_store.update(current_plan=plan.commands, current_command_index=-1)
            self._log_ui(f"plan returned · {len(plan.commands)} primitives")

            if self.config.confirm_before_send:
                self.pending_plan = plan
                self._set_state(MissionState.AWAITING_APPROVAL)
                self.state_store.update(pending_approval=True)
                self._log_ui("waiting for operator confirmation")
                return
            await self._send_plan(plan)
        except Exception as exc:
            await self._fail(f"planning failed: {exc}")

    async def _send_plan(self, plan: ValidatedPlan) -> None:
        if not self.current_run:
            return
        if not self.rover.connected:
            await self._fail("rover is not connected")
            return
        self.current_plan = plan
        self.current_run.current_path_id = plan.path_id
        self._executed_rounds.append({"plan": plan, "completed": False, "last_acked": -1})
        self.execution.start(plan)
        self._set_state(MissionState.SENDING_PATH)
        await self.rover.send_path(plan)
        self._set_state(MissionState.EXECUTING)
        self._log_ui(f"tx path {plan.path_id} · {len(plan.commands)} primitives")

    async def _on_command_ack(self, event: CommandAckEvent) -> None:
        if self._manual_path_id == event.path_id:
            self.execution.command_ack(event.path_id, event.command_index)
            self.state_store.update(current_command_index=event.command_index)
            self._log_ui(f"manual ack command {event.command_index + 1}")
            return
        if not self.current_run or event.path_id != self.current_run.current_path_id:
            return
        self.execution.command_ack(event.path_id, event.command_index)
        for entry in self._executed_rounds:
            if entry["plan"].path_id == event.path_id:
                entry["last_acked"] = max(entry["last_acked"], event.command_index)
        self.current_run.current_command_index = event.command_index
        self.state_store.update(current_command_index=event.command_index)
        await self.repository.save_command_ack(self.current_run.run_id, event.path_id, event.command_index)
        self._log_ui(f"ack command {event.command_index + 1}")

    async def _on_path_ack(self, event: PathAckEvent) -> None:
        if self._manual_path_id == event.path_id:
            self.execution.path_ack(event.path_id)
            self._manual_path_id = None
            self.state_store.update(manual_busy=False)
            self._log_ui(f"manual path {event.path_id} complete")
            return
        if not self.current_run or event.path_id != self.current_run.current_path_id:
            return
        self.execution.path_ack(event.path_id)
        for entry in self._executed_rounds:
            if entry["plan"].path_id == event.path_id:
                entry["completed"] = True
        self._log_ui(f"ack path {event.path_id}")
        asyncio.create_task(self._verify_after_path(), name="mission-verification")

    async def _verify_after_path(self) -> None:
        if not self.current_run or not self.current_plan or self.initial_world is None:
            return
        try:
            self._set_state(MissionState.VERIFYING)
            settle_s = max(0.0, float(self.config.heading_reply_settle_seconds))
            self._log_ui(
                f"waiting {settle_s:.1f}s for camera before final verification"
            )
            after = await self.perception.capture_settled_world_state()
            self._log_ui("captured fresh final verification world state")
            await self.repository.save_snapshot(self.current_run.run_id, after, "after")
            self.state_store.update(processed_frame=after.annotated_frame, world_summary=after.scene_markdown)
            result = await self.verifier.verify(
                self.current_run.prompt,
                self.initial_world,
                after,
                self.cumulative_executed_plan(),
            )
            await self.repository.save_verification(
                self.current_run.run_id, self.current_run.correction_round, result
            )
            if result.success:
                await self._finish(RunStatus.DONE, MissionState.COMPLETED, result.message or "goal verified")
                return

            if result.correction is None:
                await self._finish(
                    RunStatus.FAILED,
                    MissionState.NEEDS_ATTENTION,
                    "goal not achieved and no correction was produced",
                )
                return

            if self.current_run.correction_round >= self.MAX_CORRECTION_ROUNDS:
                await self._finish(
                    RunStatus.FAILED,
                    MissionState.NEEDS_ATTENTION,
                    "maximum autonomous correction rounds reached",
                )
                return

            self.current_run.correction_round += 1
            self.current_run.revision += 1
            self.state_store.update(correction_round=self.current_run.correction_round)
            correction = self.validator.validate(
                result.correction,
                source="verification",
                revision=self.current_run.revision,
            )
            await self.repository.save_plan(self.current_run.run_id, correction)
            self.state_store.update(current_plan=correction.commands, current_command_index=-1)
            await self._send_plan(correction)
        except Exception as exc:
            await self._fail(f"verification failed: {exc}")

    async def _on_nack(self, event: NackEvent) -> None:
        if self._manual_path_id == event.path_id:
            self.execution.fail(event.path_id, event.reason)
            self._manual_path_id = None
            self.state_store.update(manual_busy=False)
            self._log_ui(f"manual nack · {event.reason}")
            return
        if not self.current_run or event.path_id != self.current_run.current_path_id:
            return
        self.execution.fail(event.path_id, event.reason)
        self._log_ui(f"nack · {event.reason}")
        asyncio.create_task(self._replan_after_nack(event), name="mission-replan")

    async def _replan_after_nack(self, event: NackEvent) -> None:
        if not self.current_run:
            return
        try:
            self._set_state(MissionState.REPLANNING)
            world = await self.perception.capture_world_state()
            await self.repository.save_snapshot(self.current_run.run_id, world, "replan")
            proposal, _ = await self.planner.request_plan(
                f"{self.current_run.prompt}\nThe previous path halted because: {event.reason}. Re-plan safely.",
                world,
            )
            self.current_run.revision += 1
            plan = self.validator.validate(proposal, source="replan", revision=self.current_run.revision)
            await self.repository.save_plan(self.current_run.run_id, plan)
            self.state_store.update(current_plan=plan.commands, current_command_index=-1)
            if self.config.confirm_before_send:
                self.pending_plan = plan
                self._set_state(MissionState.AWAITING_APPROVAL)
                self.state_store.update(pending_approval=True)
                return
            await self._send_plan(plan)
        except Exception as exc:
            await self._fail(f"re-plan failed: {exc}")

    async def _on_heading_requested(self, event: HeadingRequestedEvent) -> None:
        _ = event
        request_time = time.monotonic()
        self._log_ui("rover requested fresh camera heading")
        try:
            heading = await self.perception.get_fresh_heading(
                requested_at_monotonic=request_time
            )
            elapsed_ms = (time.monotonic() - request_time) * 1000.0
            await self.rover.send_heading(heading)
            self._log_ui(
                f"fresh heading reply · {heading:.1f}° · {elapsed_ms:.0f} ms"
            )
            await self.repository.log(
                self.current_run.run_id if self.current_run else None,
                "INFO",
                "heading",
                f"fresh reply {heading:.2f} deg after {elapsed_ms:.0f} ms",
            )
        except Exception as exc:
            # Never answer a heading request with the old cached frame. If the
            # stream cannot produce a stable fresh heading, send nothing and
            # let the rover's existing heading timeout/NACK path handle it.
            self._log_ui(f"fresh heading unavailable · {exc}")
            await self.repository.log(
                self.current_run.run_id if self.current_run else None,
                "WARNING",
                "heading",
                str(exc),
            )

    async def _on_telemetry(self, event: TelemetryEvent) -> None:
        self.state_store.update(telemetry=event.sample)
        await self.repository.save_telemetry(
            self.current_run.run_id if self.current_run else None, event.sample
        )

    async def _on_protocol_error(self, event: ProtocolErrorEvent) -> None:
        self._log_ui(f"protocol warning · {event.message}")
        await self.repository.log(
            self.current_run.run_id if self.current_run else None,
            "WARNING",
            "protocol",
            event.message,
        )

    async def _on_rover_connected(self, event: RoverConnectedEvent) -> None:
        self.state_store.update(rover_connected=True, rover_status_text="ROVER LINKED")
        self._log_ui(f"rover connected · {event.uri}")
        if self.config.telemetry_on_start:
            await self.rover.set_telemetry(True)
            self._log_ui("tx telem · on=true")

    async def _on_rover_disconnected(self, event: RoverDisconnectedEvent) -> None:
        self.state_store.update(rover_connected=False, rover_status_text="ROVER OFFLINE")
        if event.reason:
            self._log_ui(f"rover disconnected · {event.reason}")

    async def _fail(self, message: str) -> None:
        if self.current_run:
            await self._finish(RunStatus.FAILED, MissionState.FAILED, message)
        else:
            self.state_store.update(last_error=message)
            self._log_ui(message)

    async def _finish(self, status: RunStatus, state: MissionState, message: str) -> None:
        if not self.current_run:
            return
        self.current_run.status = status
        self.current_run.state = state
        self.current_run.ended_at = utc_now_iso()
        self.state_store.update(
            mission_state=state,
            pending_approval=False,
            last_error=message if status == RunStatus.FAILED else "",
        )
        self._log_ui(message)
        await self.repository.finish_run(self.current_run)
        await self.refresh_recent_runs()
        # Keep the completed run visible in the UI, but it is no longer active.

    def cumulative_executed_plan(self) -> ValidatedPlan:
        """Everything the rover executed in this mission, as one plan ending in STOP.

        Completed paths contribute all their motion commands; a path that was
        stopped part-way (NACK) contributes only the commands it acknowledged.
        """
        commands: list[DriveCommand] = []
        for entry in self._executed_rounds:
            plan_cmds = [c for c in entry["plan"].commands if c.op != "STOP"]
            if not entry["completed"]:
                plan_cmds = [c for i, c in enumerate(entry["plan"].commands)
                             if c.op != "STOP" and i <= entry["last_acked"]]
            commands.extend(plan_cmds)
        commands.append(DriveCommand("STOP"))
        last = self._executed_rounds[-1]["plan"] if self._executed_rounds else self.current_plan
        return ValidatedPlan(
            path_id=last.path_id if last else 0,
            commands=commands,
            source="cumulative",
            revision=last.revision if last else 0,
        )

    def _set_state(self, state: MissionState) -> None:
        if self.current_run:
            self.current_run.state = state
        self.state_store.update(mission_state=state)

    def _log_ui(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.state_store.append_log(f"{stamp}  {message}")

    async def refresh_recent_runs(self) -> None:
        runs = await self.repository.recent_runs(24)
        self.state_store.update(recent_runs=runs)
