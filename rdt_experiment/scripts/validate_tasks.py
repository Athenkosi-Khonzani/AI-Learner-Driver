"""Validate the Rover Driving Test task set, draw Figure 3.4 and optionally lock it.

    python scripts/validate_tasks.py                  # check + draw, no lock
    python scripts/validate_tasks.py --lock           # write SHA-256 hashes into the manifest
    python scripts/validate_tasks.py --set subsystem  # the fixed-plan T3-T5 tasks

Checks: geometry inside the arena, no overlapping blocks, start pose clear,
conservative A* feasibility for the final rover footprint, goal clear.
Locking refuses to run if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rdt  # noqa: E402,F401  (sets dependency paths)
from llm_test.simulator.geometry import polygon_inside_arena, polygons_intersect  # noqa: E402

from rdt.config import load_experiment_config  # noqa: E402
from rdt.layout_figure import task_grid  # noqa: E402
from rdt.tasks import RDTTask, sha256_file  # noqa: E402
from rdt.worldstate import block_polygon, rover_polygon  # noqa: E402


def check_task(task: RDTTask, L: float, W: float) -> tuple[list[str], dict]:
    problems: list[str] = []
    polys = []
    for b in task.blocks:
        poly = block_polygon(b.to_block_spec())
        if not polygon_inside_arena(poly, task.arena_width_mm, task.arena_height_mm):
            problems.append(f"{b.colour} block leaves the arena")
        for other_colour, other in polys:
            if polygons_intersect(poly, other):
                problems.append(f"{b.colour} block overlaps {other_colour} block")
        polys.append((b.colour, poly))
    start = rover_polygon(task.start_x_mm, task.start_y_mm, task.start_heading_deg, L, W)
    if not polygon_inside_arena(start, task.arena_width_mm, task.arena_height_mm):
        problems.append("start footprint leaves the arena")
    if any(polygons_intersect(start, p) for _, p in polys):
        problems.append("start footprint overlaps a block")
    info: dict = {}
    if task.fixed_plan is None:
        feas = task.feasibility(L, W)
        info = {
            "feasible": feas.feasible,
            "reason": feas.reason,
            "reference_path_mm": round(feas.path_length_mm, 1) if feas.path_length_mm else None,
            "clearance_radius_mm": round(feas.clearance_radius_mm, 1),
            "path": feas.path,
        }
        if not feas.feasible:
            problems.append(f"not conservatively feasible ({feas.reason})")
    return problems, info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["main", "subsystem"], default="main")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--figure", default=None, help="output figure path (PDF or PNG)")
    args = parser.parse_args()

    cfg = load_experiment_config()
    L, W = cfg.rover_length_mm, cfg.rover_width_mm
    task_dir = ROOT / "tasks" if args.set == "main" else ROOT / "tasks" / "subsystem"
    manifest_path = task_dir / "manifest.json"
    files = sorted(task_dir.glob("*.json"))
    files = [f for f in files if f.name != "manifest.json"]
    tasks = [RDTTask.load(f) for f in files]

    all_ok = True
    entries = []
    paths = {}
    print(f"Rover footprint {L} x {W} mm; arena {tasks[0].arena_width_mm:.0f} x {tasks[0].arena_height_mm:.0f} mm\n")
    for task, path in zip(tasks, files):
        problems, info = check_task(task, L, W)
        all_ok &= not problems
        status = "OK " if not problems else "FAIL"
        ref = info.get("reference_path_mm")
        print(f"[{status}] {task.task_id:<12} {task.family:<22} d{task.difficulty}  ref path {ref} mm")
        for p in problems:
            print(f"        - {p}")
        if info.get("path"):
            paths[task.task_id] = info["path"]
        entries.append(
            {
                "id": task.task_id,
                "file": path.name,
                "family": task.family,
                "difficulty": task.difficulty,
                "title": task.title,
                "reference_path_mm": ref,
                "sha256": sha256_file(path),
            }
        )

    figure = Path(args.figure) if args.figure else ROOT / "docs" / f"rdt_layouts_{args.set}.png"
    task_grid(tasks, figure, L, W, paths=paths)
    if figure.suffix.lower() == ".png":
        task_grid(tasks, figure.with_suffix(".pdf"), L, W, paths=paths)
    print(f"\nLayout figure: {figure}")

    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if old.get("locked") and not args.lock:
        print("Manifest is locked; leaving it unchanged.")
        return 0 if all_ok else 1
    if args.lock and not all_ok:
        print("Refusing to lock: fix the failures above first.")
        return 1
    manifest = {
        "set": args.set,
        "locked": bool(args.lock),
        "locked_on": date.today().isoformat() if args.lock else None,
        "rover_footprint_mm": [L, W],
        "rule": "Hidden goals and reference paths are never sent to the planner.",
        "tasks": entries if args.lock else [{**e, "sha256": None} for e in entries],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Manifest " + ("LOCKED with SHA-256 hashes." if args.lock else "written (unlocked)."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
