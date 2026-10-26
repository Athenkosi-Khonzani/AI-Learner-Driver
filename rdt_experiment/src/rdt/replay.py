"""Paired simulator replay: the exact plan, from the exact measured start state."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from llm_test.commands.command_schema import Command
from llm_test.dataset.feasibility import _point_safe, conservative_feasibility
from llm_test.scenario.scenario_schema import GoalSpec, Scenario
from llm_test.simulator.evaluator import evaluate
from llm_test.simulator.rover_simulator import RoverSimulator

from rdt.worldstate import (
    MeasuredWorld,
    angle_error_deg,
    block_polygon,
    boundary_clearance,
    polygon_distance,
    rover_polygon,
    scenario_from_world,
)


def to_sim_commands(commands: Iterable[Any]) -> list[Command]:
    """Accept host DriveCommand objects, dicts or [op, value] pairs."""
    out: list[Command] = []
    for item in commands:
        if isinstance(item, dict):
            op, value = item.get("op"), item.get("value")
        elif isinstance(item, (list, tuple)):
            op, value = item[0], (item[1] if len(item) > 1 else None)
        else:
            op, value = item.op, item.value
        op = str(op).upper()
        out.append(Command(op=op, value=None if op == "STOP" or value is None else float(value)))
    return out


@dataclass(slots=True)
class ReplayResult:
    success: bool
    final_x_mm: float
    final_y_mm: float
    final_heading_deg: float
    position_error_mm: float
    heading_error_deg: float | None
    collision: bool
    boundary_violation: bool
    stop_reached: bool
    failure_reason: str | None
    travel_distance_mm: float
    total_rotation_deg: float
    min_block_clearance_mm: float | None
    min_boundary_clearance_mm: float
    path: list[tuple[float, float, float]]  # decimated (x, y, heading)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def replay(
    world: MeasuredWorld,
    commands: Iterable[Any],
    goal: GoalSpec,
    *,
    name: str,
    instruction: str,
    rover_length_mm: float,
    rover_width_mm: float,
) -> ReplayResult:
    scenario = scenario_from_world(
        world,
        name=name,
        instruction=instruction,
        goal=goal,
        rover_length_mm=rover_length_mm,
        rover_width_mm=rover_width_mm,
    )
    return replay_scenario(scenario, commands)


def replay_scenario(scenario: Scenario, commands: Iterable[Any]) -> ReplayResult:
    sim_cmds = to_sim_commands(commands)
    result = RoverSimulator(scenario).simulate(sim_cmds)
    ev = evaluate(scenario, result)
    blocks = [block_polygon(b) for b in scenario.blocks]
    L, W = scenario.rover.length_mm, scenario.rover.breadth_mm
    min_block = math.inf if blocks else None
    min_boundary = math.inf
    states = result.states
    step = max(1, len(states) // 400)
    sampled = states[::step] + [states[-1]]
    for s in sampled:
        poly = rover_polygon(s.x_mm, s.y_mm, s.heading_deg, L, W)
        if blocks:
            min_block = min(min_block, min(polygon_distance(poly, b) for b in blocks))
        min_boundary = min(min_boundary, boundary_clearance(poly, scenario.arena_width_mm, scenario.arena_height_mm))
    final = result.final_state
    return ReplayResult(
        success=bool(ev.task_success),
        final_x_mm=final.x_mm,
        final_y_mm=final.y_mm,
        final_heading_deg=final.heading_deg % 360.0,
        position_error_mm=ev.position_error_mm,
        heading_error_deg=ev.heading_error_deg,
        collision=bool(result.collision),
        boundary_violation=bool(result.boundary_violation),
        stop_reached=bool(result.stop_reached),
        failure_reason=result.failure_reason,
        travel_distance_mm=result.travel_distance_mm,
        total_rotation_deg=result.total_rotation_deg,
        min_block_clearance_mm=None if min_block is None else float(min_block),
        min_boundary_clearance_mm=float(min_boundary),
        path=[(round(s.x_mm, 1), round(s.y_mm, 1), round(s.heading_deg % 360.0, 1)) for s in sampled],
    )


def predict_pose(
    start_xyh: tuple[float, float, float], commands: Iterable[Any], *, step_mm: float = 20.0
) -> tuple[tuple[float, float, float], list[tuple[float, float, float]]]:
    """Ideal kinematic end pose of the WHOLE plan, ignoring collisions and walls.

    Uses exactly the simulator's motion model (heading 0 = +Y, positive turn =
    clockwise, FWD along heading, BACK against it). The simulator itself halts at
    the first collision, which makes its end pose useless for measuring execution
    error; this prediction does not halt. Returns the end pose and a path.
    """
    x, y, h = (float(v) for v in start_xyh)
    path = [(x, y, h % 360.0)]
    for cmd in to_sim_commands(commands):
        if cmd.op == "STOP":
            break
        if cmd.op == "TURN":
            h = (h + float(cmd.value)) % 360.0
            path.append((x, y, h))
        elif cmd.op in ("FWD", "BACK"):
            d = float(cmd.value) * (1.0 if cmd.op == "FWD" else -1.0)
            n = max(1, int(abs(d) // step_mm))
            th = math.radians(h)
            for i in range(1, n + 1):
                path.append((x + math.sin(th) * d * i / n, y + math.cos(th) * d * i / n, h))
            x += math.sin(th) * d
            y += math.cos(th) * d
    return (x, y, h % 360.0), path


def execution_error(
    physical_xyh: tuple[float, float, float], predicted_xyh: tuple[float, float, float]
) -> tuple[float, float]:
    """Equations (3.1) and (3.2): e_pos and e_theta between rover and the plan's ideal end pose."""
    x, y, h = physical_xyh
    px, py, ph = predicted_xyh
    return math.hypot(x - px, y - py), angle_error_deg(h, ph)


# ---------------------------------------------------------------- reference path

def reference_path(scenario: Scenario, grid_mm: float = 25.0) -> tuple[list[tuple[float, float]], float | None]:
    """Conservative shortest path (A* then line-of-sight shortcutting).

    The circumscribed-circle model makes this an upper bound on the truly
    shortest collision-free path, so path efficiency against it can exceed 1
    when the rover legitimately passes closer to a block than the circle allows.
    """
    feas = conservative_feasibility(scenario, grid_mm=grid_mm)
    if not feas.feasible:
        return [], None
    radius = feas.clearance_radius_mm
    pts = list(feas.path)

    def visible(a, b) -> bool:
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(dist / 10.0))
        return all(
            _point_safe(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, scenario, radius)
            for i in range(n + 1)
        )

    smoothed = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not visible(pts[i], pts[j]):
            j -= 1
        smoothed.append(pts[j])
        i = j
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(smoothed, smoothed[1:]))
    return smoothed, length
