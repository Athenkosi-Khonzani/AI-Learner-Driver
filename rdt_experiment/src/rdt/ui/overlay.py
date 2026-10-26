"""Placement guide drawn over the live top-down view."""

from __future__ import annotations

import math

import cv2
import numpy as np

from rdt.tasks import PlacementReport, RDTTask
from rdt.video import Canvas, draw_dashed, draw_goal, put_text
from rdt.worldstate import block_polygon, rover_polygon

GHOST = {"red": (60, 60, 255), "green": (60, 220, 60), "yellow": (40, 230, 255), "blue": (255, 140, 40)}


def draw_placement_guide(img: np.ndarray, canvas: Canvas, task: RDTTask, rover_len: float, rover_wid: float,
                         report: PlacementReport | None = None) -> np.ndarray:
    out = img.copy()
    for b in task.blocks:
        poly = block_polygon(b.to_block_spec())
        draw_dashed(out, canvas, poly + [poly[0]], GHOST.get(b.colour, (255, 255, 255)), 2, 10)
        cx, cy = canvas.px(b.x_mm, b.y_mm)
        cv2.drawMarker(out, (cx, cy), GHOST.get(b.colour, (255, 255, 255)), cv2.MARKER_TILTED_CROSS, 14, 2)
    rp = rover_polygon(task.start_x_mm, task.start_y_mm, task.start_heading_deg, rover_len, rover_wid)
    draw_dashed(out, canvas, rp + [rp[0]], (255, 255, 255), 2, 8)
    th = math.radians(task.start_heading_deg)
    a = canvas.px(task.start_x_mm, task.start_y_mm)
    b = canvas.px(task.start_x_mm + 120 * math.sin(th), task.start_y_mm + 120 * math.cos(th))
    cv2.arrowedLine(out, a, b, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
    goal = task.resolve_goal(None)
    draw_goal(out, canvas, goal.target_x_mm, goal.target_y_mm, goal.position_tolerance_mm,
              goal.target_heading_deg if goal.require_heading else None)
    put_text(out, f"PLACEMENT GUIDE  {task.task_id}  (dashed = where things go)", (16, 92), 0.9, (255, 255, 255), 2)
    if report is not None:
        y = 132
        for item in report.items:
            colour = (90, 230, 90) if item.ok else (80, 80, 255)
            put_text(out, f"{'OK ' if item.ok else 'FIX'}  {item.name}: {item.text}", (16, y), 0.8, colour, 2)
            y += 34
    return out
