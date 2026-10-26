"""Convert between the host's measured world state and the simulator's scenario.

Conventions
-----------
* World frame: millimetres, origin bottom-left, +X right, +Y up.
* Rover heading: 0 deg = +Y, 90 deg = +X, positive = clockwise (host and simulator agree).
* Host blocks report ``orientation_deg`` as the angle of the LONG side measured
  anticlockwise from +X, modulo 180 (OpenCV ``minAreaRect`` style).
* Simulator blocks use the rover heading convention for their LONG axis
  (``length_mm``): 0 deg = long axis along +Y, 90 deg = long axis along +X.

  A long-side direction at host angle ``a`` is (cos a, sin a); in heading form it
  is (sin h, cos h). Therefore ``h = 90 - a`` (mod 180).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from llm_test.scenario.scenario_schema import BlockSpec, GoalSpec, RoverSpec, Scenario
from llm_test.simulator.geometry import RectPose, polygons_intersect, rectangle_polygon

Point = tuple[float, float]


def host_orientation_to_sim(host_deg: float) -> float:
    return (90.0 - float(host_deg)) % 180.0


def sim_orientation_to_host(sim_deg: float) -> float:
    return (90.0 - float(sim_deg)) % 180.0


def angle_error_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two headings in degrees."""
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def axis_error_deg(a: float, b: float) -> float:
    """Difference between two undirected axes (period 180 deg)."""
    d = abs((float(a) - float(b)) % 180.0)
    return min(d, 180.0 - d)


@dataclass(slots=True)
class MeasuredBlock:
    id: str
    colour: str
    x_mm: float
    y_mm: float
    length_mm: float  # long side
    breadth_mm: float  # short side
    orientation_deg: float  # simulator convention
    polygon: list[Point]

    def to_block_spec(self) -> BlockSpec:
        return BlockSpec(
            id=self.id,
            colour=self.colour,
            x_mm=self.x_mm,
            y_mm=self.y_mm,
            length_mm=self.length_mm,
            breadth_mm=self.breadth_mm,
            orientation_deg=self.orientation_deg,
        )


@dataclass(slots=True)
class MeasuredWorld:
    arena_width_mm: float
    arena_height_mm: float
    rover_x_mm: float | None
    rover_y_mm: float | None
    rover_heading_deg: float | None
    rover_length_mm: float
    rover_width_mm: float
    rover_confidence: float
    rover_pose_valid: bool
    blocks: list[MeasuredBlock]

    @property
    def has_pose(self) -> bool:
        return self.rover_x_mm is not None and self.rover_y_mm is not None and self.rover_heading_deg is not None

    def block_by_colour(self, colour: str) -> MeasuredBlock | None:
        matches = [b for b in self.blocks if b.colour == colour]
        if not matches:
            return None
        # Several blocks of one colour should not occur in the RDT; take the largest.
        return max(matches, key=lambda b: b.length_mm * b.breadth_mm)


def parse_scene(scene: dict[str, Any]) -> MeasuredWorld:
    """Parse the host's ``scene_json`` (schema ai_learner_world_state 1.0)."""
    arena = scene["arena"]
    rover = scene.get("rover") or {}
    pose = rover.get("pose") or None
    footprint = rover.get("footprint_mm") or {}
    blocks: list[MeasuredBlock] = []
    for item in scene.get("obstacles", []):
        size = item.get("size_mm", {})
        long_side = float(size.get("long_side", size.get("length", 0.0)))
        short_side = float(size.get("short_side", size.get("breadth", 0.0)))
        polygon = [(float(p["x"]), float(p["y"])) for p in item.get("polygon_mm", [])]
        blocks.append(
            MeasuredBlock(
                id=str(item.get("id", item.get("colour", "block"))),
                colour=str(item["colour"]),
                x_mm=float(item["centre_mm"]["x"]),
                y_mm=float(item["centre_mm"]["y"]),
                length_mm=long_side,
                breadth_mm=short_side,
                orientation_deg=host_orientation_to_sim(float(item.get("orientation_deg", 0.0))),
                polygon=polygon,
            )
        )
    return MeasuredWorld(
        arena_width_mm=float(arena["width_mm"]),
        arena_height_mm=float(arena["height_mm"]),
        rover_x_mm=float(pose["x_mm"]) if pose else None,
        rover_y_mm=float(pose["y_mm"]) if pose else None,
        rover_heading_deg=float(pose["heading_deg"]) if pose else None,
        rover_length_mm=float(footprint.get("length", 0.0)),
        rover_width_mm=float(footprint.get("width", footprint.get("breadth", 0.0))),
        rover_confidence=float(rover.get("confidence", 0.0)),
        rover_pose_valid=bool(rover.get("pose_valid", pose is not None)),
        blocks=blocks,
    )


def scenario_from_world(
    world: MeasuredWorld,
    *,
    name: str,
    instruction: str,
    goal: GoalSpec,
    rover_length_mm: float | None = None,
    rover_width_mm: float | None = None,
) -> Scenario:
    """Build a simulator scenario whose start state is exactly the measured world."""
    if not world.has_pose:
        raise ValueError("The measured world state has no rover pose; it cannot be replayed")
    return Scenario(
        name=name,
        arena_width_mm=world.arena_width_mm,
        arena_height_mm=world.arena_height_mm,
        rover=RoverSpec(
            x_mm=float(world.rover_x_mm),
            y_mm=float(world.rover_y_mm),
            length_mm=float(rover_length_mm or world.rover_length_mm),
            breadth_mm=float(rover_width_mm or world.rover_width_mm),
            heading_deg=float(world.rover_heading_deg) % 360.0,
        ),
        blocks=[b.to_block_spec() for b in world.blocks],
        user_instruction=instruction,
        hidden_goal=goal,
    )


# ---------------------------------------------------------------- polygons

def rover_polygon(x: float, y: float, heading: float, length: float, width: float) -> list[Point]:
    return rectangle_polygon(RectPose(x, y, length, width, heading))


def block_polygon(block: BlockSpec | MeasuredBlock) -> list[Point]:
    return rectangle_polygon(
        RectPose(block.x_mm, block.y_mm, block.length_mm, block.breadth_mm, block.orientation_deg)
    )


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def polygon_distance(poly_a: list[Point], poly_b: list[Point]) -> float:
    """Minimum distance between two convex polygons (0 when they touch or overlap)."""
    if polygons_intersect(poly_a, poly_b):
        return 0.0
    best = math.inf
    for poly, other in ((poly_a, poly_b), (poly_b, poly_a)):
        for p in poly:
            for i in range(len(other)):
                best = min(best, _point_segment_distance(p, other[i], other[(i + 1) % len(other)]))
    return best


def boundary_clearance(poly: Iterable[Point], width: float, height: float) -> float:
    """Signed clearance from a polygon to the arena edge (negative = outside)."""
    return min(min(x, y, width - x, height - y) for x, y in poly)
