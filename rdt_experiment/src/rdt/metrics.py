"""Physical measures for one round of a trial.

Scoring uses the settled final capture (same rule as Experiment 1). The trail
recorded during driving supplies the safety and path measures.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from llm_test.scenario.scenario_schema import GoalSpec

from rdt.worldstate import (
    MeasuredWorld,
    angle_error_deg,
    block_polygon,
    boundary_clearance,
    polygon_distance,
    rover_polygon,
)


@dataclass(slots=True)
class TrailSample:
    t: float  # seconds since the trial's recording started
    x_mm: float | None
    y_mm: float | None
    heading_deg: float | None
    confidence: float
    detected: bool  # marker found in this frame


@dataclass(slots=True)
class TrailMetrics:
    samples: int
    valid_samples: int
    marker_lost_fraction: float
    path_length_mm: float
    rotation_deg: float
    min_block_clearance_mm: float | None
    min_boundary_clearance_mm: float | None
    trail_overlaps_block: bool
    trail_leaves_arena: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def trail_metrics(
    trail: Sequence[TrailSample],
    start_world: MeasuredWorld,
    rover_length_mm: float,
    rover_width_mm: float,
    *,
    min_confidence: float = 0.5,
    distance_deadband_mm: float = 10.0,
    rotation_deadband_deg: float = 3.0,
    boundary_tolerance_mm: float = 10.0,
) -> TrailMetrics:
    """Measures from the camera trail against the blocks measured at the start.

    Deadbands stop pose jitter from inflating path length and rotation: a step
    is only counted once the rover has moved more than the deadband since the
    last counted point.
    """
    valid = [
        s
        for s in trail
        if s.detected and s.x_mm is not None and s.heading_deg is not None and s.confidence >= min_confidence
    ]
    blocks = [block_polygon(b) for b in start_world.blocks]
    W, H = start_world.arena_width_mm, start_world.arena_height_mm
    path = 0.0
    rotation = 0.0
    min_block = math.inf if blocks else None
    min_boundary = math.inf if valid else None
    overlap = False
    leaves = False
    last_xy = last_h = None
    for s in valid:
        poly = rover_polygon(s.x_mm, s.y_mm, s.heading_deg, rover_length_mm, rover_width_mm)
        if blocks:
            d = min(polygon_distance(poly, b) for b in blocks)
            min_block = min(min_block, d)
            overlap |= d <= 0.0
        bc = boundary_clearance(poly, W, H)
        min_boundary = min(min_boundary, bc)
        leaves |= bc < -boundary_tolerance_mm
        if last_xy is None:
            last_xy, last_h = (s.x_mm, s.y_mm), s.heading_deg
            continue
        step = math.hypot(s.x_mm - last_xy[0], s.y_mm - last_xy[1])
        if step >= distance_deadband_mm:
            path += step
            last_xy = (s.x_mm, s.y_mm)
        turn = angle_error_deg(s.heading_deg, last_h)
        if turn >= rotation_deadband_deg:
            rotation += turn
            last_h = s.heading_deg
    n = len(trail)
    return TrailMetrics(
        samples=n,
        valid_samples=len(valid),
        marker_lost_fraction=(1.0 - len(valid) / n) if n else 1.0,
        path_length_mm=path,
        rotation_deg=rotation,
        min_block_clearance_mm=None if min_block is None or min_block == math.inf else float(min_block),
        min_boundary_clearance_mm=None if min_boundary is None else float(min_boundary),
        trail_overlaps_block=overlap,
        trail_leaves_arena=leaves,
    )


def block_displacements(start: MeasuredWorld, final: MeasuredWorld) -> dict[str, float | None]:
    """How far each start block moved by the final capture (None = not found after)."""
    out: dict[str, float | None] = {}
    for b in start.blocks:
        after = final.block_by_colour(b.colour)
        out[b.colour] = None if after is None else math.hypot(after.x_mm - b.x_mm, after.y_mm - b.y_mm)
    return out


@dataclass(slots=True)
class RoundScore:
    """Geometric score of one round (Experiment 1 rule on the measured final pose)."""

    success: bool
    final_x_mm: float | None
    final_y_mm: float | None
    final_heading_deg: float | None
    goal_position_error_mm: float | None
    goal_heading_error_deg: float | None
    position_ok: bool
    heading_ok: bool
    stop_reached: bool
    collision: bool
    collision_evidence: list[str] = field(default_factory=list)
    boundary_violation: bool = False
    pose_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_round(
    final_world: MeasuredWorld,
    goal: GoalSpec,
    *,
    stop_reached: bool,
    collision_evidence: list[str],
    boundary_violation: bool,
) -> RoundScore:
    if not final_world.has_pose:
        return RoundScore(
            success=False,
            final_x_mm=None,
            final_y_mm=None,
            final_heading_deg=None,
            goal_position_error_mm=None,
            goal_heading_error_deg=None,
            position_ok=False,
            heading_ok=False,
            stop_reached=stop_reached,
            collision=bool(collision_evidence),
            collision_evidence=list(collision_evidence),
            boundary_violation=boundary_violation,
            pose_available=False,
        )
    x, y, h = final_world.rover_x_mm, final_world.rover_y_mm, final_world.rover_heading_deg
    pos_err = math.hypot(x - goal.target_x_mm, y - goal.target_y_mm)
    head_err = angle_error_deg(h, goal.target_heading_deg) if goal.require_heading else None
    position_ok = pos_err <= goal.position_tolerance_mm
    heading_ok = (head_err <= goal.heading_tolerance_deg) if head_err is not None else True
    collision = bool(collision_evidence)
    success = position_ok and heading_ok and stop_reached and not collision and not boundary_violation
    return RoundScore(
        success=success,
        final_x_mm=x,
        final_y_mm=y,
        final_heading_deg=h,
        goal_position_error_mm=pos_err,
        goal_heading_error_deg=head_err,
        position_ok=position_ok,
        heading_ok=heading_ok,
        stop_reached=stop_reached,
        collision=collision,
        collision_evidence=list(collision_evidence),
        boundary_violation=boundary_violation,
    )


def collision_evidence(
    trail: TrailMetrics | None,
    displacements: dict[str, float | None],
    *,
    operator_contact: bool,
    displacement_threshold_mm: float,
    use_trail: bool = True,
) -> list[str]:
    evidence: list[str] = []
    if operator_contact:
        evidence.append("operator_observed_contact")
    moved = [c for c, d in displacements.items() if d is not None and d >= displacement_threshold_mm]
    if moved:
        evidence.append("block_displaced:" + ",".join(moved))
    if use_trail and trail is not None and trail.trail_overlaps_block:
        evidence.append("trail_overlaps_block")
    return evidence
