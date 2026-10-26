"""Drawing helpers for the trial video and its closing frame (OpenCV, BGR)."""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

# Round colours (BGR): round 0 = first pass, then corrections.
ROUND_COLOURS = [(255, 220, 0), (255, 0, 255), (0, 165, 255), (0, 255, 128)]
PLAN_COLOUR = (255, 255, 255)
GOAL_COLOUR = (60, 60, 255)
TEXT = (240, 240, 240)
PANEL = (25, 25, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(slots=True)
class Canvas:
    """Maps arena millimetres to output pixels for a correct-aspect top-down view."""

    arena_w: float
    arena_h: float
    width: int = 1280

    @property
    def height(self) -> int:
        h = int(round(self.width * self.arena_h / self.arena_w))
        return h + (h % 2)  # even height for video codecs

    def px(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        u = x_mm / self.arena_w * (self.width - 1)
        v = (self.arena_h - y_mm) / self.arena_h * (self.height - 1)
        return int(round(u)), int(round(v))

    def mm_to_px(self, mm: float) -> float:
        return mm / self.arena_w * (self.width - 1)

    def from_birdseye(self, birdseye: np.ndarray) -> np.ndarray:
        return cv2.resize(birdseye, (self.width, self.height), interpolation=cv2.INTER_AREA)

    def blank(self) -> np.ndarray:
        return np.full((self.height, self.width, 3), 30, np.uint8)


def put_text(img, text, org, scale=0.6, colour=TEXT, thickness=1) -> None:
    text = ascii_text(text.replace("Δ", "d"))
    x, y = org
    for dx, dy in ((1, 1), (2, 2)):
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)


def ascii_text(text: str) -> str:
    """OpenCV's Hershey fonts only draw ASCII."""
    return (
        text.replace("°", " deg").replace("×", "x").replace("–", "-").replace("—", "-")
        .replace("·", "|").replace("’", "'").replace("“", '"').replace("”", '"')
        .encode("ascii", "replace").decode("ascii")
    )


def title_card(
    canvas: Canvas,
    heading: str,
    lines: Sequence[tuple[str, str]],
    footer: str = "",
    start_image: np.ndarray | None = None,
) -> np.ndarray:
    img = canvas.blank()
    y = 80
    put_text(img, ascii_text(heading), (50, y), 1.05, (255, 255, 255), 2)
    y += 30
    cv2.line(img, (50, y), (canvas.width - 50, y), (90, 90, 90), 1)
    y += 45
    wrap = max(30, int((canvas.width - 300) / 13))
    for label, value in lines:
        put_text(img, ascii_text(label), (50, y), 0.62, (150, 200, 255), 1)
        for k, chunk in enumerate(textwrap.wrap(ascii_text(value), wrap) or [""]):
            put_text(img, chunk, (290, y + 30 * k), 0.68, TEXT, 1)
        y += 30 * max(1, len(textwrap.wrap(ascii_text(value), wrap))) + 16
    if start_image is not None:
        top = y + 4
        avail_h = canvas.height - top - 70
        if avail_h > 120:
            th = avail_h
            tw = int(th * canvas.width / canvas.height)
            thumb = cv2.resize(start_image, (tw, th), interpolation=cv2.INTER_AREA)
            x0 = (canvas.width - tw) // 2
            img[top : top + th, x0 : x0 + tw] = thumb
            cv2.rectangle(img, (x0 - 1, top - 1), (x0 + tw, top + th), (110, 110, 110), 1)
            put_text(img, "Start state given to the planner (frozen capture)", (x0, top + th + 24), 0.5, (170, 170, 170), 1)
    if footer:
        put_text(img, ascii_text(footer), (50, canvas.height - 18), 0.48, (150, 150, 150), 1)
    return img


def prompt_strip(img: np.ndarray, text: str, right_text: str = "") -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 38), (w, h), PANEL, -1)
    max_chars = int((w - 260) / 9.5)
    t = ascii_text(text)
    if len(t) > max_chars:
        t = t[: max_chars - 3] + "..."
    put_text(img, t, (12, h - 13), 0.55, TEXT, 1)
    if right_text:
        put_text(img, ascii_text(right_text), (w - 240, h - 13), 0.55, (150, 200, 255), 1)


def draw_polyline(img, canvas: Canvas, pts_mm: Iterable[tuple[float, float]], colour, thickness=3) -> None:
    pts = [canvas.px(x, y) for x, y in pts_mm]
    if len(pts) >= 2:
        cv2.polylines(img, [np.asarray(pts, np.int32)], False, colour, thickness, cv2.LINE_AA)


def draw_dashed(img, canvas: Canvas, pts_mm: Sequence[tuple[float, float]], colour, thickness=2, dash_px=14) -> None:
    pts = [np.asarray(canvas.px(x, y), float) for x, y in pts_mm]
    carry = 0.0
    draw = True
    for a, b in zip(pts, pts[1:]):
        seg = b - a
        length = float(np.hypot(*seg))
        pos = 0.0
        while pos < length:
            step = min(dash_px - carry, length - pos)
            p0 = a + seg * (pos / length) if length else a
            p1 = a + seg * ((pos + step) / length) if length else a
            if draw:
                cv2.line(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), colour, thickness, cv2.LINE_AA)
            pos += step
            carry += step
            if carry >= dash_px - 1e-9:
                carry = 0.0
                draw = not draw


def draw_pose(img, canvas: Canvas, x, y, h, colour, length_mm=110.0, radius=7) -> None:
    c = canvas.px(x, y)
    th = math.radians(h)
    e = canvas.px(x + length_mm * math.sin(th), y + length_mm * math.cos(th))
    cv2.circle(img, c, radius, colour, -1, cv2.LINE_AA)
    cv2.arrowedLine(img, c, e, colour, 2, cv2.LINE_AA, tipLength=0.3)


def draw_goal(img, canvas: Canvas, x, y, tol_mm, heading=None) -> None:
    c = canvas.px(x, y)
    cv2.circle(img, c, max(3, int(canvas.mm_to_px(tol_mm))), GOAL_COLOUR, 2, cv2.LINE_AA)
    cv2.drawMarker(img, c, GOAL_COLOUR, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    if heading is not None:
        th = math.radians(heading)
        e = canvas.px(x + 90 * math.sin(th), y + 90 * math.cos(th))
        cv2.arrowedLine(img, c, e, GOAL_COLOUR, 2, cv2.LINE_AA, tipLength=0.3)


def info_panel(img, lines: Sequence[tuple[str, tuple[int, int, int]]], origin=(16, 66), width=470) -> None:
    x0, y0 = origin
    h = 26 * len(lines) + 16
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + h), PANEL, -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
    for k, (text, colour) in enumerate(lines):
        put_text(img, ascii_text(text), (x0 + 12, y0 + 28 + 26 * k), 0.56, colour, 1)


def legend(img, items: Sequence[tuple[str, tuple[int, int, int], bool]], origin_right=16, top=66) -> None:
    w = img.shape[1]
    width = 270
    x0 = w - width - origin_right
    h = 24 * len(items) + 14
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, top), (x0 + width, top + h), PANEL, -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
    for k, (label, colour, dashed) in enumerate(items):
        y = top + 24 + 24 * k
        if dashed:
            for s in range(0, 34, 10):
                cv2.line(img, (x0 + 12 + s, y - 5), (x0 + 17 + s, y - 5), colour, 2)
        else:
            cv2.line(img, (x0 + 12, y - 5), (x0 + 46, y - 5), colour, 3)
        put_text(img, ascii_text(label), (x0 + 56, y), 0.5, TEXT, 1)
