"""One Rover Driving Test trial, from frozen start state to scored record.

Procedure (Section 3.8.2)
1. The operator places rover and blocks on the outlines and presses
   "Capture start": a settled capture becomes the frozen start state W0.
2. The placement check compares W0 with the task file.
3. "Run": W0 is pinned, so the planner receives exactly W0; recording starts.
4. The host's MissionController plans, validates, sends, waits for the camera
   to settle, verifies and (when enabled) corrects, as in normal operation.
5. When the mission ends the trial is scored from the saved record, the
   closing frame is drawn and everything is written to the trial folder.
6. The operator adds tape readings and observations; the trial is re-scored.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ai_learner_host.domain.enums import MissionState, RunStatus
from ai_learner_host.domain.events import (
    CommandAckEvent,
    NackEvent,
    PathAckEvent,
    ProtocolErrorEvent,
    RoverDisconnectedEvent,
    TelemetryEvent,
)
from ai_learner_host.domain.models import WorldStateSnapshot

import rdt
from rdt.config import ExperimentConfig
from rdt.recorder import TrajectoryRecorder
from rdt.runtime import ExperimentRuntime
from rdt.scoring import score_trial
from rdt.session import Session, Slot
from rdt.tasks import PlacementReport, RDTTask, sha256_file
from rdt.video import (
    GOAL_COLOUR,
    PLAN_COLOUR,
    ROUND_COLOURS,
    Canvas,
    draw_dashed,
    draw_goal,
    draw_polyline,
    draw_pose,
    info_panel,
    legend,
    prompt_strip,
    put_text,
    title_card,
)
from rdt.worldstate import MeasuredWorld, parse_scene


def _cmds(commands) -> list[list]:
    out = []
    for c in commands:
        out.append([c.op] if c.value is None else [c.op, c.value])
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class StartCapture:
    snapshot: WorldStateSnapshot
    world: MeasuredWorld
    placement: PlacementReport
    captured_at: float


class TrialCollector:
    """Listens to the instrumented host and builds the raw trial record."""

    def __init__(self, t0: float, clock: Callable[[], float], images_dir: Path) -> None:
        self.t0 = t0
        self.clock = clock
        self.images_dir = images_dir
        self.run_id: str | None = None
        self.rounds: list[dict[str, Any]] = []
        self.worlds: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.planner_calls: list[dict[str, Any]] = []
        self.verifier_calls: list[dict[str, Any]] = []
        self.finish: dict[str, Any] | None = None
        self.done = asyncio.Event()
        self.prompt_text: str | None = None
        self.vbat_samples: list[tuple[float, float]] = []
        self.on_round: Callable[[int], None] | None = None
        self.max_extra_rounds: int | None = None
        self.on_round_limit: Callable[[], None] | None = None
        self.round_limit_hit = False
        self._after_count = 0
        self._replan_count = 0
        self._last_after_key: str | None = None
        self._last_replan_key: str | None = None

    def t(self) -> float:
        return round(self.clock() - self.t0, 3)

    def _current_round(self) -> dict[str, Any] | None:
        sent = [r for r in self.rounds if r.get("t_path_sent") is not None]
        return sent[-1] if sent else (self.rounds[-1] if self.rounds else None)

    def _round_for_path(self, path_id: int) -> dict[str, Any] | None:
        return next((r for r in self.rounds if r["path_id"] == path_id), None)

    def event(self, kind: str, **data: Any) -> None:
        cur = self._current_round()
        self.events.append({"t": self.t(), "kind": kind, "round": cur["index"] if cur else 0, **data})

    def store_world(self, key: str, snapshot: WorldStateSnapshot, role: str) -> None:
        self.worlds[key] = {
            "t": self.t(),
            "role": role,
            "snapshot_id": snapshot.snapshot_id,
            "timestamp": snapshot.timestamp,
            "scene": snapshot.scene_json,
        }
        if snapshot.raw_frame is not None:
            cv2.imwrite(str(self.images_dir / f"{key}_raw.jpg"), snapshot.raw_frame)
        if snapshot.annotated_frame is not None:
            cv2.imwrite(str(self.images_dir / f"{key}_annotated.jpg"), snapshot.annotated_frame)

    # ------------------------------------------------------------- host proxy events
    def __call__(self, kind: str, p: dict[str, Any]) -> None:
        if p.get("run_id") and self.run_id and p["run_id"] != self.run_id:
            return
        if kind == "plan_request":
            self.planner_calls.append({"t_request": self.t(), "prompt": p["prompt"]})
        elif kind == "plan_response" and self.planner_calls:
            call = self.planner_calls[-1]
            call["t_response"] = self.t()
            call["latency_s"] = round(call["t_response"] - call["t_request"], 3)
            call["raw_response"] = p["proposal"].raw_response
            call["commands"] = _cmds(p["proposal"].commands)
            call["fixed_plan"] = p.get("fixed", False)
            if self.prompt_text is None:
                self.prompt_text = p.get("prompt_text")
        elif kind == "plan_error" and self.planner_calls:
            self.planner_calls[-1].update(t_response=self.t(), error=p["error"])
            self.event("planner_error", message=p["error"])
        elif kind == "plan":
            self.run_id = self.run_id or p["run_id"]
            plan = p["plan"]
            index = len(self.rounds)
            if index == 0:
                start_key = "W0"
            elif plan.source == "verification":
                start_key = self._last_after_key
            elif plan.source == "replan":
                start_key = self._last_replan_key
            else:
                start_key = self._last_after_key
            rnd: dict[str, Any] = {
                "index": index,
                "source": plan.source,
                "revision": plan.revision,
                "path_id": plan.path_id,
                "commands": _cmds(plan.commands),
                "t_plan_saved": self.t(),
                "start_world_key": start_key,
                "end_world_key": None,
                "t_path_sent": None,
                "t_path_ack": None,
                "command_acks": [],
                "nacks": [],
                "verifier": None,
            }
            if plan.source in ("planner", "replan") and self.planner_calls:
                rnd["planner_latency_s"] = self.planner_calls[-1].get("latency_s")
            elif plan.source == "verification" and self.verifier_calls:
                rnd["planner_latency_s"] = self.verifier_calls[-1].get("latency_s")
            self.rounds.append(rnd)
            if self.max_extra_rounds is not None and index > self.max_extra_rounds:
                rnd["blocked"] = True
                self.round_limit_hit = True
                self.event("round_limit", message=f"round {index} ({plan.source}) exceeds the budget of {self.max_extra_rounds} extra rounds; not sent")
                if self.on_round_limit:
                    self.on_round_limit()
        elif kind == "path_blocked":
            pass
        elif kind == "path_sent":
            rnd = self._round_for_path(p["path_id"])
            if rnd is not None:
                rnd["t_path_sent"] = self.t()
                if self.on_round:
                    self.on_round(rnd["index"])
        elif kind == "snapshot":
            self.run_id = self.run_id or p["run_id"]
            role = p["role"]
            snap = p["snapshot"]
            if role == "before":
                self.worlds.setdefault("W0", {})["host_snapshot_id"] = snap.snapshot_id
            elif role == "after":
                key = f"after_{self._after_count}"
                self._after_count += 1
                self.store_world(key, snap, role)
                self._last_after_key = key
                cur = self._current_round()
                if cur is not None and cur.get("end_world_key") is None:
                    cur["end_world_key"] = key
                    cur["t_end_capture"] = self.t()
            elif role == "replan":
                key = f"replan_{self._replan_count}"
                self._replan_count += 1
                self.store_world(key, snap, role)
                self._last_replan_key = key
                cur = self._current_round()
                if cur is not None and cur.get("end_world_key") is None:
                    cur["end_world_key"] = key
                    cur["t_end_capture"] = self.t()
        elif kind == "verify_request":
            self.verifier_calls.append({"t_request": self.t(), "path_id": p["path_id"]})
        elif kind == "verify_response" and self.verifier_calls:
            call = self.verifier_calls[-1]
            res = p["result"]
            call.update(
                t_response=self.t(),
                latency_s=round(self.t() - call["t_request"], 3),
                success=bool(res.success),
                message=res.message,
                correction=_cmds(res.correction.commands) if res.correction else None,
            )
            rnd = self._round_for_path(p["path_id"])
            if rnd is not None:
                rnd["verifier"] = {
                    "success": bool(res.success),
                    "message": res.message,
                    "latency_s": call["latency_s"],
                    "has_correction": res.correction is not None,
                    "correction": call["correction"],
                }
        elif kind == "verify_error" and self.verifier_calls:
            self.verifier_calls[-1].update(t_response=self.t(), error=p["error"])
            self.event("verifier_error", message=p["error"])
        elif kind == "finish":
            if self.run_id and p["run_id"] != self.run_id:
                return
            run = p["run"]
            self.finish = {
                "t": self.t(),
                "status": run.status.value,
                "state": run.state.name,
                "correction_rounds": run.correction_round,
            }
            self.done.set()
        elif kind == "log":
            self.event("host_log", level=p["level"], component=p["component"], message=p["message"])

    # ------------------------------------------------------------- event bus
    async def on_command_ack(self, e: CommandAckEvent) -> None:
        rnd = self._round_for_path(e.path_id)
        if rnd is not None:
            rnd["command_acks"].append({"i": e.command_index, "t": self.t()})

    async def on_path_ack(self, e: PathAckEvent) -> None:
        rnd = self._round_for_path(e.path_id)
        if rnd is not None:
            rnd["t_path_ack"] = self.t()
        self.event("path_ack", path_id=e.path_id)

    async def on_nack(self, e: NackEvent) -> None:
        rnd = self._round_for_path(e.path_id)
        if rnd is not None:
            rnd["nacks"].append({"i": e.command_index, "reason": e.reason, "t": self.t()})
        self.event("nack", path_id=e.path_id, command_index=e.command_index, reason=e.reason)

    async def on_telemetry(self, e: TelemetryEvent) -> None:
        if e.sample.vbat is not None:
            self.vbat_samples.append((self.t(), float(e.sample.vbat)))

    async def on_protocol_error(self, e: ProtocolErrorEvent) -> None:
        self.event("protocol_error", message=e.message, raw=e.raw[:300])

    async def on_disconnect(self, e: RoverDisconnectedEvent) -> None:
        self.event("rover_disconnected", message=e.reason)


class TrialRunner:
    def __init__(
        self,
        runtime: ExperimentRuntime,
        session: Session,
        exp_config: ExperimentConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.session = session
        self.cfg = exp_config
        self.clock = clock
        self.log = log or (lambda m: None)
        self._abort = asyncio.Event()
        self.active = False
        self.recorder: TrajectoryRecorder | None = None
        self.latest_vbat: float | None = None
        runtime.event_bus.subscribe(TelemetryEvent, self._telemetry)

    async def _telemetry(self, e: TelemetryEvent) -> None:
        if e.sample.vbat is not None:
            self.latest_vbat = float(e.sample.vbat)

    # ------------------------------------------------------------------ step 1-2
    async def capture_start(self, task: RDTTask) -> StartCapture:
        snapshot = await self.runtime.perception_inner.capture_settled_world_state()
        world = parse_scene(snapshot.scene_json)
        placement = task.placement_report(world, self.cfg.placement_limits)
        return StartCapture(snapshot, world, placement, self.clock())

    def preflight(self, battery_v: float | None) -> list[str]:
        problems = []
        if not self.runtime.rover.connected:
            problems.append("rover not connected")
        if self.runtime.frame_buffer.age_seconds() is None or self.runtime.frame_buffer.age_seconds() > 2.0:
            problems.append("camera stream is not live")
        ok, msg = self.runtime.llm_readiness()
        if not ok:
            problems.append(f"planner not ready: {msg}")
        if self.cfg.min_battery_v is not None:
            if battery_v is None:
                problems.append("battery voltage unknown (enter it manually)")
            elif battery_v < self.cfg.min_battery_v:
                problems.append(f"battery {battery_v:.2f} V below {self.cfg.min_battery_v:.2f} V")
        return problems

    def abort(self) -> None:
        self._abort.set()

    # ------------------------------------------------------------------ step 3-5
    async def run_trial(
        self,
        slot: Slot,
        task: RDTTask,
        start: StartCapture,
        *,
        verification_enabled: bool,
        battery_v: float | None = None,
        battery_source: str = "telemetry",
        override_reason: str = "",
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> Path:
        if self.active:
            raise RuntimeError("A trial is already running")
        if not start.placement.ok and not override_reason.strip():
            raise RuntimeError("Placement check failed; adjust the arena or give an override reason")
        if not start.world.has_pose:
            raise RuntimeError("The start state has no rover pose")
        self.active = True
        self._abort.clear()
        trial_dir = self.session.new_trial_dir(slot)
        t0 = self.clock()
        collector = TrialCollector(t0, self.clock, trial_dir)
        collector.store_world("W0", start.snapshot, "start")
        self.runtime.add_listener(collector)
        bus = self.runtime.event_bus
        handlers = [
            (CommandAckEvent, collector.on_command_ack),
            (PathAckEvent, collector.on_path_ack),
            (NackEvent, collector.on_nack),
            (TelemetryEvent, collector.on_telemetry),
            (ProtocolErrorEvent, collector.on_protocol_error),
            (RoverDisconnectedEvent, collector.on_disconnect),
        ]
        for etype, h in handlers:
            bus.subscribe(etype, h)

        verification_enabled = bool(verification_enabled) and task.fixed_plan is None
        max_rounds = self.cfg.max_correction_rounds if verification_enabled else 0
        controller = self.runtime.controller
        controller.MAX_CORRECTION_ROUNDS = max_rounds  # instance attribute shadows the class constant
        # NACK replans count against the same budget as verifier corrections.
        collector.max_extra_rounds = max_rounds
        self.runtime.rover.set_blocked(False)

        def _limit() -> None:
            self.runtime.rover.set_blocked(True)
            self._abort.set()

        collector.on_round_limit = _limit
        self.runtime.planner.set_fixed_plan(task.fixed_plan)

        rec_cfg = self.cfg.recording
        canvas = Canvas(start.world.arena_width_mm, start.world.arena_height_mm, int(rec_cfg.get("output_width_px", 1280)))
        goal = task.resolve_goal(start.world)
        label = f"{task.task_id} run {slot.rep}"
        planner_label = "fixed plan (no LLM)" if task.fixed_plan else self.runtime.planner_label()
        mode_text = (
            f"ON (up to {self.cfg.max_correction_rounds} corrections)" if verification_enabled else "OFF (first pass only)"
        )
        title = title_card(
            canvas,
            f"Rover Driving Test  |  {task.task_id}: {task.title}",
            [
                ("Instruction", task.instruction),
                ("Run", f"{slot.rep} (session slot {slot.slot} of {len(self.session.schedule)})"),
                ("Planner", planner_label),
                ("Verification", mode_text),
                ("Session", f"{self.session.name} ({self.session.mode})"),
                ("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ],
            footer="AI Learner Rover Driver | EEE4022S | trail: rover centre from the overhead camera",
            start_image=canvas.from_birdseye(start.snapshot.annotated_frame)
            if start.snapshot.annotated_frame is not None else None,
        )
        recorder = TrajectoryRecorder(
            self.runtime.frame_buffer,
            self.runtime.config,
            canvas,
            trial_dir / "video.mp4",
            fps=float(rec_cfg.get("fps", 8)),
            codec=str(rec_cfg.get("codec", "mp4v")),
            prompt_text=f"Prompt: {task.instruction}",
            label=label,
            goal=(goal.target_x_mm, goal.target_y_mm, goal.position_tolerance_mm,
                  goal.target_heading_deg if goal.require_heading else None),
            clock=self.clock,
            t0=t0,
        )
        recorder.on_frame = on_frame
        recorder.open(title, float(rec_cfg.get("title_seconds", 3.0)))
        self.recorder = recorder
        collector.on_round = recorder.set_round
        await recorder.start()

        meta = {
            "rdt_version": rdt.__version__,
            "session": self.session.name,
            "session_mode": self.session.mode,
            "slot": slot.slot,
            "task_id": task.task_id,
            "task_family": task.family,
            "task_difficulty": task.difficulty,
            "task_sha256": sha256_file(task.path) if task.path else None,
            "rep": slot.rep,
            "tape_check": slot.tape_check,
            "instruction": task.instruction,
            "fixed_plan": task.fixed_plan is not None,
            "verification_enabled": verification_enabled,
            "max_correction_rounds": max_rounds,
            "started_at": _now_iso(),
            "battery_v": battery_v,
            "battery_source": battery_source if battery_v is not None else None,
            "override_reason": override_reason.strip() or None,
            "experiment_config_sha256": self.cfg.sha256,
            **self.runtime.describe(),
        }
        status = "complete"
        try:
            t_submit = self.clock()
            self.runtime.perception.pin(start.snapshot)
            await controller.submit_prompt(task.instruction)
            done = asyncio.create_task(collector.done.wait())
            aborted = asyncio.create_task(self._abort.wait())
            finished, pending = await asyncio.wait(
                {done, aborted}, timeout=self.cfg.trial_timeout_seconds, return_when=asyncio.FIRST_COMPLETED
            )
            for task_ in pending:
                task_.cancel()
            if aborted in finished and not collector.done.is_set():
                status = "round_limit" if collector.round_limit_hit else "aborted"
            elif not collector.done.is_set():
                status = "timeout"
            if status != "complete" and controller.current_run and controller.current_run.status == RunStatus.RUNNING:
                await controller._finish(RunStatus.STOPPED, MissionState.CANCELLED, f"trial {status}")
            mission_time = (collector.finish or {}).get("t", collector.t()) - (t_submit - t0)
            if collector.finish is not None:
                collector.finish["message"] = self.runtime.state_store.state.last_error or ""
            # Every scored round must end on a settled capture. A NACKed, aborted or
            # timed-out round may only have an unsettled replan frame, or none.
            executed = [r for r in collector.rounds if r.get("t_path_sent") is not None]
            last = executed[-1] if executed else None
            if last is not None and not str(last.get("end_world_key") or "").startswith("after_"):
                try:
                    snap = await self.runtime.perception_inner.capture_settled_world_state()
                    collector.store_world("final_settled", snap, "final_settled")
                    last["unsettled_end_world_key"] = last.get("end_world_key")
                    last["end_world_key"] = "final_settled"
                    last["t_end_capture"] = collector.t()
                except Exception as exc:
                    collector.event("final_capture_failed", message=str(exc))
        finally:
            self.runtime.perception.pin(None)
            self.runtime.planner.set_fixed_plan(None)
            self.runtime.rover.set_blocked(False)
            await asyncio.sleep(0.5)
            await recorder.stop()
            self.runtime.remove_listener(collector)
            for etype, h in handlers:
                subs = bus._subscribers.get(etype, [])
                if h in subs:
                    subs.remove(h)

        trail = [
            {"t": s.t, "round": r, "x_mm": s.x_mm, "y_mm": s.y_mm, "heading_deg": s.heading_deg,
             "confidence": round(s.confidence, 3), "detected": s.detected}
            for s, r in zip(recorder.samples, recorder.sample_rounds)
        ]
        record = {
            "meta": meta,
            "status": status,
            "placement": start.placement.to_dict(),
            "run_id": collector.run_id,
            "finish": collector.finish,
            "rounds": collector.rounds,
            "worlds": collector.worlds,
            "planner_calls": collector.planner_calls,
            "verifier_calls": collector.verifier_calls,
            "events": collector.events,
            "telemetry_vbat": collector.vbat_samples,
            "timing": {
                "planner_latency_s": collector.planner_calls[0].get("latency_s") if collector.planner_calls else None,
                "mission_time_s": mission_time,
            },
            "trail": trail,
            "recorder_errors": recorder.errors,
            "annotations": {},
        }
        if collector.prompt_text:
            (trial_dir / "planner_prompt.txt").write_text(collector.prompt_text, encoding="utf-8")
        record["scores"] = score_trial(record, task, self.cfg)
        final_birdseye = self._final_birdseye(collector, recorder)
        final = render_final_frame(canvas, final_birdseye, record, task, label, recorder.trail_by_round())
        cv2.imwrite(str(trial_dir / "final.png"), final)
        recorder.finish(final, float(rec_cfg.get("final_hold_seconds", 4.0)))
        self._write(trial_dir, record)
        self.active = False
        self.recorder = None
        self.log(f"{task.task_id} run {slot.rep}: {status}; first pass "
                 f"{'success' if record['scores'].get('first_pass_success') else 'fail'}")
        return trial_dir

    @staticmethod
    def _final_birdseye(collector: TrialCollector, recorder: TrajectoryRecorder):
        keys = [r.get("end_world_key") for r in collector.rounds if r.get("end_world_key")]
        if keys:
            img = cv2.imread(str(collector.images_dir / f"{keys[-1]}_annotated.jpg"))
            if img is not None:
                return img
        return recorder.last_birdseye

    # ------------------------------------------------------------------ step 6
    def annotate(self, trial_dir: Path, task: RDTTask, annotations: dict[str, Any]) -> dict[str, Any]:
        return annotate_trial(trial_dir, task, self.cfg, annotations)

    @staticmethod
    def _write(trial_dir: Path, record: dict[str, Any]) -> None:
        write_trial(trial_dir, record)


def write_trial(trial_dir: Path, record: dict[str, Any]) -> None:
    (trial_dir / "trial.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    with (trial_dir / "trail.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_s", "round", "x_mm", "y_mm", "heading_deg", "confidence", "marker_detected"])
        for s in record.get("trail", []):
            writer.writerow([s["t"], s["round"], s["x_mm"], s["y_mm"], s["heading_deg"], s["confidence"], int(s["detected"])])
    with (trial_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        for e in record.get("events", []):
            fh.write(json.dumps(e, default=str) + "\n")


def annotate_trial(trial_dir: Path, task: RDTTask, cfg: ExperimentConfig, annotations: dict[str, Any]) -> dict[str, Any]:
    """Store operator annotations (tape, contact, void, notes) and re-score."""
    path = Path(trial_dir) / "trial.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    current = record.get("annotations") or {}
    current.update({k: v for k, v in annotations.items()})
    current["updated_at"] = _now_iso()
    record["annotations"] = current
    record["scores"] = score_trial(record, task, cfg)
    write_trial(Path(trial_dir), record)
    return record


# ---------------------------------------------------------------------- final frame

def render_final_frame(
    canvas: Canvas,
    birdseye: np.ndarray | None,
    record: dict[str, Any],
    task: RDTTask,
    label: str,
    trail_by_round: dict[int, list[tuple[float, float]]],
) -> np.ndarray:
    img = canvas.from_birdseye(birdseye) if birdseye is not None else canvas.blank()
    scores = record.get("scores", {})
    rounds = scores.get("rounds", [])
    goal = scores.get("goal", {})
    if goal:
        draw_goal(img, canvas, goal["x_mm"], goal["y_mm"], goal["position_tolerance_mm"], goal.get("heading_deg"))
    if rounds and rounds[0].get("predicted_path"):
        draw_dashed(img, canvas, rounds[0]["predicted_path"], PLAN_COLOUR, 2)
    for r, pts in sorted(trail_by_round.items()):
        draw_polyline(img, canvas, pts, ROUND_COLOURS[r % len(ROUND_COLOURS)], 3)
    w0 = record.get("worlds", {}).get("W0", {}).get("scene", {})
    pose = (w0.get("rover") or {}).get("pose")
    if pose:
        draw_pose(img, canvas, pose["x_mm"], pose["y_mm"], pose["heading_deg"], (200, 200, 200))

    def yes(v):
        return "SUCCESS" if v else "FAIL"

    def colour(v):
        return (90, 230, 90) if v else (80, 80, 255)

    fp = scores.get("first_pass_success")
    cl = scores.get("closed_loop_success")
    lines = [
        (f"{label}   status: {record.get('status')}", (255, 255, 255)),
        (f"First pass: {yes(fp)}   goal error {_fmt(scores.get('first_pass_goal_error_mm'))} mm"
         + (f", heading {_fmt(scores.get('first_pass_heading_error_deg'))} deg" if goal.get("require_heading") else ""),
         colour(fp)),
    ]
    if cl is not None:
        lines.append((f"Closed loop: {yes(cl)}   corrections used {scores.get('corrections_used', 0)}", colour(cl)))
    else:
        lines.append(("Closed loop: verification off", (180, 180, 180)))
    rs = scores.get("replay_success")
    lines += [
        (f"Simulator replay: {yes(rs)}", colour(rs)),
        (f"Execution error: e_pos {_fmt(scores.get('e_pos_mm'))} mm, e_theta {_fmt(scores.get('e_theta_deg'))} deg", (240, 240, 240)),
        (("Min. clearance to blocks " + (f"{_fmt(scores.get('min_block_clearance_mm'))} mm" if scores.get("min_block_clearance_mm") is not None else "n/a (no blocks)"))
         + f"   path efficiency {_fmt(scores.get('path_efficiency_first_pass'), 2)}", (240, 240, 240)),
    ]
    attr = (scores.get("attribution") or {}).get("first_pass")
    if attr:
        lines.append((f"First-pass failure class: {attr}", (0, 200, 255)))
    info_panel(img, lines)
    items = [("planned (ideal) path", PLAN_COLOUR, True), ("first pass (camera)", ROUND_COLOURS[0], False)]
    n_corr = n_replan = 0
    for r in rounds[1:]:
        k = r["index"]
        if r.get("source") == "replan":
            n_replan += 1
            text = f"replan {n_replan} (after stop)"
        else:
            n_corr += 1
            text = f"correction {n_corr}"
        items.append((text, ROUND_COLOURS[k % len(ROUND_COLOURS)], False))
    items.append(("goal and tolerance", GOAL_COLOUR, False))
    legend(img, items)
    prompt_strip(img, f"Prompt: {task.instruction}", "FINAL")
    return img


def _fmt(v, nd=0):
    if v is None:
        return "-"
    return f"{float(v):.{nd}f}"
