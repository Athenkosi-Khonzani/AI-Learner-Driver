"""Score one trial from its saved raw record.

``score_trial`` depends only on the contents of trial.json (plus the task file
and experiment config), so every trial can be re-scored offline and the
analysis always uses one version of the rules.

Definitions (Chapter 3)
-----------------------
first pass      round 0: the planner's original plan from the frozen start W0.
closed loop     the last executed round, with up to the configured number of
                verifier-triggered corrections (None when verification is off).
replay          the round-0 plan simulated from W0 with the Experiment 1
                simulator and success rule.
e_pos, e_theta  rover end pose after round 0 against the ideal kinematic end
                pose of the round-0 plan (Equations 3.1 and 3.2).
attribution     Table 3.8: protocol, planning, perception, execution, verification.
"""

from __future__ import annotations

import math
from typing import Any

from rdt.config import ExperimentConfig
from rdt.metrics import (
    TrailSample,
    block_displacements,
    collision_evidence,
    score_round,
    trail_metrics,
)
from rdt.replay import execution_error, predict_pose, reference_path, replay
from rdt.tasks import RDTTask
from rdt.worldstate import MeasuredWorld, parse_scene, scenario_from_world

SAFETY_NACK_WORDS = ("obstacle", "tof", "safety", "blocked", "too close")


def _world(record: dict, key: str | None) -> MeasuredWorld | None:
    if not key:
        return None
    item = record.get("worlds", {}).get(key)
    return parse_scene(item["scene"]) if item else None


def _trail(record: dict, round_index: int | None = None) -> list[TrailSample]:
    out = []
    for s in record.get("trail", []):
        if round_index is not None and int(s.get("round", 0)) != round_index:
            continue
        out.append(
            TrailSample(
                t=float(s["t"]),
                x_mm=s.get("x_mm"),
                y_mm=s.get("y_mm"),
                heading_deg=s.get("heading_deg"),
                confidence=float(s.get("confidence", 0.0)),
                detected=bool(s.get("detected", False)),
            )
        )
    return out


def _rnd(v, nd=1):
    return None if v is None else round(float(v), nd)


def score_trial(record: dict[str, Any], task: RDTTask, cfg: ExperimentConfig) -> dict[str, Any]:
    scoring = cfg.scoring
    L, W = cfg.rover_length_mm, cfg.rover_width_mm
    ann = record.get("annotations", {}) or {}
    contact_first = bool(ann.get("contact_first_pass", False))
    contact_later = bool(ann.get("contact_correction", False))
    disp_threshold = float(scoring.get("block_displacement_collision_mm", 25.0))
    min_conf = float(scoring.get("min_trail_confidence", 0.5))
    tape_threshold = float(scoring.get("camera_tape_disagreement_mm", 30.0))

    w0 = _world(record, "W0")
    if w0 is None:
        return {"error": "no start world state"}
    goal = task.resolve_goal(w0)
    instruction = record["meta"]["instruction"]

    rounds_out: list[dict[str, Any]] = []
    any_collision = False
    for rnd in record.get("rounds", []):
        if rnd.get("blocked") or rnd.get("t_path_sent") is None:
            continue  # planned but never executed (round budget used, or trial ended first)
        k = int(rnd["index"])
        start = _world(record, rnd.get("start_world_key")) or w0
        end = _world(record, rnd.get("end_world_key"))
        trail = trail_metrics(_trail(record, k), start, L, W, min_confidence=min_conf)
        disp = block_displacements(start, end) if end is not None else {}
        evidence = collision_evidence(
            trail, disp, operator_contact=contact_first if k == 0 else contact_later,
            displacement_threshold_mm=disp_threshold,
        )
        nacks = rnd.get("nacks", [])
        stop_reached = rnd.get("t_path_ack") is not None and not nacks
        score = score_round(
            end if end is not None else MeasuredWorld(w0.arena_width_mm, w0.arena_height_mm, None, None, None, L, W, 0.0, False, []),
            goal,
            stop_reached=stop_reached,
            collision_evidence=evidence,
            boundary_violation=trail.trail_leaves_arena,
        )
        any_collision |= score.collision
        sim = None
        predicted = None
        e_pos = e_theta = None
        if start.has_pose:
            sim = replay(start, rnd["commands"], goal, name=f"{task.task_id}_r{k}", instruction=instruction,
                         rover_length_mm=L, rover_width_mm=W)
            predicted, pred_path = predict_pose((start.rover_x_mm, start.rover_y_mm, start.rover_heading_deg), rnd["commands"])
            completed = rnd.get("t_path_ack") is not None and not rnd.get("nacks")
            if completed and end is not None and end.has_pose:
                e_pos, e_theta = execution_error((end.rover_x_mm, end.rover_y_mm, end.rover_heading_deg), predicted)
        verdict = rnd.get("verifier")
        rounds_out.append(
            {
                "index": k,
                "source": rnd.get("source"),
                "commands": rnd["commands"],
                "n_commands": len(rnd["commands"]),
                "score": score.to_dict(),
                "trail": trail.to_dict(),
                "block_displacement_mm": {c: _rnd(d) for c, d in disp.items()},
                "replay": None if sim is None else {
                    "success": sim.success,
                    "final_pose": [_rnd(sim.final_x_mm), _rnd(sim.final_y_mm), _rnd(sim.final_heading_deg)],
                    "position_error_mm": _rnd(sim.position_error_mm),
                    "failure_reason": sim.failure_reason,
                    "min_block_clearance_mm": _rnd(sim.min_block_clearance_mm),
                    "min_boundary_clearance_mm": _rnd(sim.min_boundary_clearance_mm),
                    "travel_mm": _rnd(sim.travel_distance_mm),
                    "rotation_deg": _rnd(sim.total_rotation_deg),
                    "path": sim.path,
                },
                "predicted_end_pose": None if predicted is None else [_rnd(v) for v in predicted],
                "predicted_path": None if predicted is None else [(round(x, 1), round(y, 1)) for x, y, _ in pred_path],
                "e_pos_mm": _rnd(e_pos),
                "e_theta_deg": _rnd(e_theta, 2),
                "planner_latency_s": _rnd(rnd.get("planner_latency_s"), 3),
                "drive_time_s": _rnd(
                    (rnd["t_path_ack"] - rnd["t_path_sent"]) if rnd.get("t_path_ack") and rnd.get("t_path_sent") else None, 2
                ),
                "nacks": nacks,
                "verifier": None if not verdict else {
                    "success": verdict.get("success"),
                    "message": verdict.get("message"),
                    "latency_s": _rnd(verdict.get("latency_s"), 3),
                    "has_correction": verdict.get("has_correction"),
                    "agrees_with_rule": None if verdict.get("success") is None else bool(verdict["success"]) == score.success,
                },
            }
        )

    verification_on = bool(record["meta"].get("verification_enabled"))
    first = rounds_out[0] if rounds_out else None
    last = rounds_out[-1] if rounds_out else None
    first_pass = bool(first and first["score"]["success"])
    closed_loop = None
    if verification_on:
        closed_loop = bool(last and last["score"]["success"] and not any_collision)
    corrections = sum(1 for r in rounds_out if r["source"] == "verification")

    # Reference path from the measured start to the resolved goal.
    ref_pts, ref_len = reference_path(
        scenario_from_world(w0, name=task.task_id, instruction=instruction, goal=goal,
                            rover_length_mm=L, rover_width_mm=W)
    ) if w0.has_pose else ([], None)
    first_completed = bool(first and first["drive_time_s"] is not None and not first["nacks"])
    driven0 = first["trail"]["path_length_mm"] if first else None
    driven_all = sum(r["trail"]["path_length_mm"] for r in rounds_out) if rounds_out else None

    # Camera against tape (taken at the end of the trial).
    tape = None
    if ann.get("tape_x_mm") is not None and ann.get("tape_y_mm") is not None and last:
        fx, fy = last["score"]["final_x_mm"], last["score"]["final_y_mm"]
        if fx is not None:
            diff = math.hypot(float(ann["tape_x_mm"]) - fx, float(ann["tape_y_mm"]) - fy)
            tape = {"camera_minus_tape_mm": round(diff, 1), "disagrees": diff > tape_threshold}

    attribution = _attribute(record, rounds_out, first_pass, closed_loop, verification_on, tape, w0, cfg)

    clearances = [r["trail"]["min_block_clearance_mm"] for r in rounds_out if r["trail"]["min_block_clearance_mm"] is not None]
    timing = record.get("timing", {})
    return {
        "goal": {
            "x_mm": _rnd(goal.target_x_mm), "y_mm": _rnd(goal.target_y_mm),
            "position_tolerance_mm": goal.position_tolerance_mm,
            "require_heading": goal.require_heading,
            "heading_deg": goal.target_heading_deg if goal.require_heading else None,
            "heading_tolerance_deg": goal.heading_tolerance_deg if goal.require_heading else None,
        },
        "first_pass_success": first_pass,
        "closed_loop_success": closed_loop,
        "corrections_used": corrections,
        "replay_success": bool(first and first["replay"] and first["replay"]["success"]),
        "first_pass_goal_error_mm": first["score"]["goal_position_error_mm"] if first else None,
        "first_pass_heading_error_deg": first["score"]["goal_heading_error_deg"] if first else None,
        "final_goal_error_mm": last["score"]["goal_position_error_mm"] if last else None,
        "e_pos_mm": first["e_pos_mm"] if first else None,
        "e_theta_deg": first["e_theta_deg"] if first else None,
        "collision_any_round": any_collision,
        "boundary_violation_any_round": any(r["score"]["boundary_violation"] for r in rounds_out),
        "min_block_clearance_mm": min(clearances) if clearances else None,
        "min_block_clearance_first_pass_mm": first["trail"]["min_block_clearance_mm"] if first else None,
        "replay_min_block_clearance_mm": first["replay"]["min_block_clearance_mm"] if first and first["replay"] else None,
        "reference_path_mm": _rnd(ref_len),
        "reference_path": ref_pts,
        "driven_path_first_pass_mm": _rnd(driven0),
        "driven_path_total_mm": _rnd(driven_all),
        "path_efficiency_first_pass": _rnd(ref_len / driven0, 3) if (ref_len and driven0 and first_completed) else None,
        "plan_commands_first_pass": first["n_commands"] if first else None,
        "plan_rotation_first_pass_deg": first["replay"]["rotation_deg"] if first and first["replay"] else None,
        "planning_latency_s": first["planner_latency_s"] if first else _rnd(timing.get("planner_latency_s"), 3),
        "drive_time_first_pass_s": first["drive_time_s"] if first else None,
        "mission_time_s": _rnd(timing.get("mission_time_s"), 2),
        "verifier_agreement": [r["verifier"]["agrees_with_rule"] for r in rounds_out if r["verifier"]],
        "camera_tape": tape,
        "attribution": attribution,
        "rounds": rounds_out,
    }


def _attribute(record, rounds, first_pass, closed_loop, verification_on, tape, w0, cfg) -> dict[str, Any]:
    """Assign Table 3.8 classes to the first-pass and closed-loop outcomes."""
    events = record.get("events", [])
    limits = cfg.placement_limits
    out: dict[str, Any] = {"first_pass": None, "closed_loop": None, "detail": "", "confirmed_by_tape": None}
    if first_pass:
        pass
    elif not rounds:
        out["first_pass"] = "planning"
        out["detail"] = record.get("finish", {}).get("message", "no executable plan")
    else:
        r0 = rounds[0]
        protocol_events = [
            e for e in events
            if e["kind"] in ("protocol_error", "rover_disconnected") and e.get("round", 0) == 0
        ]
        nacks = r0.get("nacks", [])
        safety_stop = any(any(w in str(n.get("reason", "")).lower() for w in SAFETY_NACK_WORDS) for n in nacks)
        if (nacks and not safety_stop) or protocol_events or (r0["drive_time_s"] is None and not nacks):
            out["first_pass"] = "protocol"
            out["detail"] = "; ".join([str(n.get("reason")) for n in nacks] + [e.get("message", "") for e in protocol_events]) or "path not acknowledged"
        elif r0["replay"] is not None and not r0["replay"]["success"]:
            out["first_pass"] = "planning"
            out["detail"] = f"replay failed ({r0['replay']['failure_reason'] or 'goal not reached in simulation'})"
            if safety_stop:
                out["detail"] += "; rover safety stop"
        elif (not w0.rover_pose_valid) or w0.rover_confidence < limits.min_pose_confidence or not record.get("placement", {}).get("ok", True):
            out["first_pass"] = "perception"
            out["detail"] = "start state below the placement or confidence limits"
        elif tape is not None and tape["disagrees"]:
            out["first_pass"] = "perception"
            out["detail"] = f"camera and tape disagree by {tape['camera_minus_tape_mm']} mm"
            out["confirmed_by_tape"] = True
        else:
            out["first_pass"] = "execution"
            out["detail"] = f"replay succeeded; rover ended {r0['e_pos_mm']} mm / {r0['e_theta_deg']} deg from the plan's end pose"
            if safety_stop:
                out["detail"] += "; rover safety stop"
            out["confirmed_by_tape"] = None if tape is None else True
    if verification_on and not first_pass and closed_loop is False:
        out["closed_loop"] = "verification"
    disagreements = [r["index"] for r in rounds if r["verifier"] and r["verifier"]["agrees_with_rule"] is False]
    out["verifier_disagreed_rounds"] = disagreements
    return out
