"""Rover Driving Test task files.

A task file fixes the start pose, block layout, typed instruction and hidden
success criterion. At run time only the instruction and the *measured* world
state go to the planner; the nominal layout is a placement guide for the
operator, and the hidden goal is used for scoring.

Goal types
----------
``absolute``      fixed arena coordinate.
``block_offset``  offset from the MEASURED centre of a named block, so a block
                  placed a few millimetres off its mark does not count against
                  the planner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_test.dataset.feasibility import FeasibilityResult, conservative_feasibility
from llm_test.scenario.scenario_schema import BlockSpec, GoalSpec, RoverSpec, Scenario

from rdt.worldstate import MeasuredWorld, angle_error_deg, axis_error_deg

FAMILIES = (
    "basic_motion",
    "coordinate_navigation",
    "object_target",
    "single_obstacle",
    "multi_obstacle",
    "tight_boundary",
)


@dataclass(slots=True)
class TaskBlock:
    colour: str
    x_mm: float
    y_mm: float
    length_mm: float
    breadth_mm: float
    orientation_deg: float  # simulator convention: long axis heading, 0 = +Y

    def to_block_spec(self) -> BlockSpec:
        return BlockSpec(
            id=f"{self.colour}_block_01",
            colour=self.colour,
            x_mm=self.x_mm,
            y_mm=self.y_mm,
            length_mm=self.length_mm,
            breadth_mm=self.breadth_mm,
            orientation_deg=self.orientation_deg,
        )


@dataclass(slots=True)
class TaskGoal:
    type: str  # "absolute" | "block_offset"
    x_mm: float | None = None
    y_mm: float | None = None
    block_colour: str | None = None
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    position_tolerance_mm: float = 50.0
    require_heading: bool = False
    target_heading_deg: float = 0.0
    heading_tolerance_deg: float = 10.0


@dataclass(slots=True)
class RDTTask:
    task_id: str
    title: str
    family: str
    difficulty: int
    instruction: str
    arena_width_mm: float
    arena_height_mm: float
    start_x_mm: float
    start_y_mm: float
    start_heading_deg: float
    blocks: list[TaskBlock]
    goal: TaskGoal
    notes: str = ""
    fixed_plan: list[dict[str, Any]] | None = None  # subsystem (no-LLM) tasks only
    path: Path | None = field(default=None, compare=False)

    # ------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path | None = None) -> "RDTTask":
        family = data["family"]
        if family not in FAMILIES and family != "subsystem":
            raise ValueError(f"{data.get('task_id')}: unknown family {family!r}")
        start = data["start_pose"]
        return cls(
            task_id=data["task_id"],
            title=data["title"],
            family=family,
            difficulty=int(data.get("difficulty", 1)),
            instruction=data["instruction"],
            arena_width_mm=float(data["arena_mm"][0]),
            arena_height_mm=float(data["arena_mm"][1]),
            start_x_mm=float(start["x_mm"]),
            start_y_mm=float(start["y_mm"]),
            start_heading_deg=float(start["heading_deg"]),
            blocks=[TaskBlock(**b) for b in data.get("blocks", [])],
            goal=TaskGoal(**data["goal"]),
            notes=data.get("notes", ""),
            fixed_plan=data.get("fixed_plan"),
            path=path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RDTTask":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), path)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "task_id": self.task_id,
            "title": self.title,
            "family": self.family,
            "difficulty": self.difficulty,
            "instruction": self.instruction,
            "arena_mm": [self.arena_width_mm, self.arena_height_mm],
            "start_pose": {
                "x_mm": self.start_x_mm,
                "y_mm": self.start_y_mm,
                "heading_deg": self.start_heading_deg,
            },
            "blocks": [asdict(b) for b in self.blocks],
            "goal": asdict(self.goal),
            "notes": self.notes,
        }
        if self.fixed_plan is not None:
            out["fixed_plan"] = self.fixed_plan
        return out

    # ------------------------------------------------------------ goals
    def resolve_goal(self, world: MeasuredWorld | None = None) -> GoalSpec:
        """Hidden goal in arena coordinates, resolved against the measured start world."""
        g = self.goal
        if g.type == "absolute":
            x, y = float(g.x_mm), float(g.y_mm)
        elif g.type == "block_offset":
            ref_x = ref_y = None
            if world is not None:
                block = world.block_by_colour(str(g.block_colour))
                if block is not None:
                    ref_x, ref_y = block.x_mm, block.y_mm
            if ref_x is None:
                nominal = [b for b in self.blocks if b.colour == g.block_colour]
                if not nominal:
                    raise ValueError(f"{self.task_id}: goal block {g.block_colour!r} not in task")
                ref_x, ref_y = nominal[0].x_mm, nominal[0].y_mm
            x, y = ref_x + g.dx_mm, ref_y + g.dy_mm
        else:
            raise ValueError(f"{self.task_id}: unknown goal type {g.type!r}")
        return GoalSpec(
            goal_type="pose",
            target_x_mm=x,
            target_y_mm=y,
            position_tolerance_mm=float(g.position_tolerance_mm),
            require_heading=bool(g.require_heading),
            target_heading_deg=float(g.target_heading_deg),
            heading_tolerance_deg=float(g.heading_tolerance_deg),
        )

    # ------------------------------------------------------------ nominal scenario
    def nominal_scenario(self, rover_length_mm: float, rover_width_mm: float) -> Scenario:
        return Scenario(
            name=self.task_id,
            arena_width_mm=self.arena_width_mm,
            arena_height_mm=self.arena_height_mm,
            rover=RoverSpec(
                x_mm=self.start_x_mm,
                y_mm=self.start_y_mm,
                length_mm=rover_length_mm,
                breadth_mm=rover_width_mm,
                heading_deg=self.start_heading_deg,
            ),
            blocks=[b.to_block_spec() for b in self.blocks],
            user_instruction=self.instruction,
            hidden_goal=self.resolve_goal(None),
        )

    def feasibility(self, rover_length_mm: float, rover_width_mm: float) -> FeasibilityResult:
        return conservative_feasibility(self.nominal_scenario(rover_length_mm, rover_width_mm), grid_mm=25.0)

    # ------------------------------------------------------------ placement check
    def placement_report(self, world: MeasuredWorld, limits: "PlacementLimits") -> "PlacementReport":
        items: list[PlacementItem] = []
        if not world.has_pose:
            items.append(PlacementItem("rover", False, "rover marker not detected", None, None))
        else:
            dpos = ((world.rover_x_mm - self.start_x_mm) ** 2 + (world.rover_y_mm - self.start_y_mm) ** 2) ** 0.5
            dhead = angle_error_deg(world.rover_heading_deg, self.start_heading_deg)
            ok = dpos <= limits.rover_position_mm and dhead <= limits.rover_heading_deg
            if world.rover_confidence < limits.min_pose_confidence:
                ok = False
            items.append(
                PlacementItem(
                    "rover",
                    ok,
                    f"Δ {dpos:.0f} mm, {dhead:.1f}°, conf {world.rover_confidence:.2f}",
                    dpos,
                    dhead,
                )
            )
        expected_colours = {b.colour for b in self.blocks}
        for nominal in self.blocks:
            measured = world.block_by_colour(nominal.colour)
            if measured is None:
                items.append(PlacementItem(nominal.colour, False, "not detected", None, None))
                continue
            dpos = ((measured.x_mm - nominal.x_mm) ** 2 + (measured.y_mm - nominal.y_mm) ** 2) ** 0.5
            # Square-ish blocks have an ill-defined long axis; skip the angle check.
            square = abs(nominal.length_mm - nominal.breadth_mm) < 0.1 * nominal.length_mm
            drot = 0.0 if square else axis_error_deg(measured.orientation_deg, nominal.orientation_deg)
            ok = dpos <= limits.block_position_mm and drot <= limits.block_rotation_deg
            items.append(PlacementItem(nominal.colour, ok, f"Δ {dpos:.0f} mm, {drot:.1f}°", dpos, drot))
        extra = sorted({b.colour for b in world.blocks} - expected_colours)
        for colour in extra:
            items.append(PlacementItem(colour, False, "unexpected block in arena", None, None))
        return PlacementReport(items)


@dataclass(slots=True)
class PlacementLimits:
    rover_position_mm: float = 20.0
    rover_heading_deg: float = 5.0
    block_position_mm: float = 30.0
    block_rotation_deg: float = 10.0
    min_pose_confidence: float = 0.75


@dataclass(slots=True)
class PlacementItem:
    name: str
    ok: bool
    text: str
    position_error_mm: float | None
    angle_error_deg: float | None


@dataclass(slots=True)
class PlacementReport:
    items: list[PlacementItem]

    @property
    def ok(self) -> bool:
        return all(i.ok for i in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "items": [asdict(i) for i in self.items]}


# ---------------------------------------------------------------- task sets

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_task_set(task_dir: str | Path, manifest_name: str = "manifest.json") -> tuple[list[RDTTask], dict]:
    task_dir = Path(task_dir)
    manifest = json.loads((task_dir / manifest_name).read_text(encoding="utf-8"))
    tasks = [RDTTask.load(task_dir / entry["file"]) for entry in manifest["tasks"]]
    return tasks, manifest


def verify_manifest(task_dir: str | Path, manifest_name: str = "manifest.json") -> list[str]:
    """Return a list of problems; empty when every file matches its locked hash."""
    task_dir = Path(task_dir)
    manifest = json.loads((task_dir / manifest_name).read_text(encoding="utf-8"))
    problems: list[str] = []
    for entry in manifest["tasks"]:
        path = task_dir / entry["file"]
        if not path.exists():
            problems.append(f"{entry['file']}: missing")
        elif entry.get("sha256") and sha256_file(path) != entry["sha256"]:
            problems.append(f"{entry['file']}: hash differs from manifest (file edited after locking)")
    return problems
