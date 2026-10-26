"""Top-down drawings of task layouts (report Figure 3.4) and trial outcomes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from rdt.tasks import RDTTask
from rdt.worldstate import block_polygon, rover_polygon

BLOCK_FACE = {"red": "#d9534f", "green": "#4caf50", "yellow": "#f2c94c", "blue": "#3d7dd8"}


def draw_task(ax, task: RDTTask, rover_length: float, rover_width: float, *, path=None, title=True) -> None:
    W, H = task.arena_width_mm, task.arena_height_mm
    ax.add_patch(Polygon([(0, 0), (W, 0), (W, H), (0, H)], closed=True, fill=False, lw=1.4, ec="#222"))
    for b in task.blocks:
        poly = block_polygon(b.to_block_spec())
        ax.add_patch(Polygon(poly, closed=True, fc=BLOCK_FACE.get(b.colour, "#999"), ec="#222", lw=0.8, alpha=0.9))
    if path:
        xs, ys = zip(*path)
        ax.plot(xs, ys, ls="--", lw=1.0, color="#555", label="Reference path (A*)")
    poly = rover_polygon(task.start_x_mm, task.start_y_mm, task.start_heading_deg, rover_length, rover_width)
    ax.add_patch(Polygon(poly, closed=True, fc="#ddd", ec="#222", lw=0.9))
    th = math.radians(task.start_heading_deg)
    ax.annotate(
        "",
        xy=(task.start_x_mm + 130 * math.sin(th), task.start_y_mm + 130 * math.cos(th)),
        xytext=(task.start_x_mm, task.start_y_mm),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#222"),
    )
    goal = task.resolve_goal(None)
    ax.add_patch(Circle((goal.target_x_mm, goal.target_y_mm), goal.position_tolerance_mm, fill=False, ec="#c00", lw=1.0))
    ax.plot([goal.target_x_mm], [goal.target_y_mm], marker="x", color="#c00", ms=6)
    if goal.require_heading:
        gh = math.radians(goal.target_heading_deg)
        ax.annotate(
            "",
            xy=(goal.target_x_mm + 120 * math.sin(gh), goal.target_y_mm + 120 * math.cos(gh)),
            xytext=(goal.target_x_mm, goal.target_y_mm),
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#c00"),
        )
    ax.set_xlim(-40, W + 40)
    ax.set_ylim(-40, H + 40)
    ax.set_aspect("equal")
    ax.set_xticks([0, 500, 1000, 1500])
    ax.set_yticks([0, 500, 1000])
    ax.tick_params(labelsize=6)
    if title:
        ax.set_title(f"{task.task_id}: {task.title}", fontsize=7.5)


def task_grid(
    tasks: Iterable[RDTTask],
    out: Path,
    rover_length: float,
    rover_width: float,
    paths: dict[str, list] | None = None,
    cols: int = 4,
) -> None:
    tasks = list(tasks)
    rows = math.ceil(len(tasks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.5 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, task in zip(axes, tasks):
        draw_task(ax, task, rover_length, rover_width, path=(paths or {}).get(task.task_id))
    for ax in axes[len(tasks):]:
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
