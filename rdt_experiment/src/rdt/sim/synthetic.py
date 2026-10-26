"""Synthetic arena: a shared physical state and a camera renderer.

The renderer draws what an overhead camera would see (floor, coloured blocks,
the rover body and its ArUco marker) through a known homography, so the
host's real PerceptionService can be run on the images unchanged.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from ai_learner_host.config import (
    AppConfig,
    CameraIntrinsicsProfile,
    ColourCalibrationProfile,
)
from ai_learner_host.perception.service import CalibrationService

IMAGE_W, IMAGE_H = 1920, 1080
# Arena corners in the image, order TL, TR, BR, BL (a mild keystone, like a tilted phone).
IMAGE_CORNERS = [[330.0, 90.0], [1590.0, 90.0], [1690.0, 1010.0], [230.0, 1010.0]]

# Pure HSV colours inside each calibrated range (OpenCV H 0-179).
SIM_HSV = {"red": (177, 210, 190), "green": (82, 230, 175), "yellow": (34, 150, 240), "blue": (100, 240, 235)}
MARKER_SIZE_MM = 70.0


def _hsv_to_bgr(hsv: tuple[int, int, int]) -> tuple[int, int, int]:
    px = np.uint8([[list(hsv)]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def synthetic_host_config(arena_w: float = 1710.0, arena_h: float = 1254.0) -> AppConfig:
    """An AppConfig whose calibration matches the synthetic camera exactly."""
    cfg = AppConfig()
    cfg.intrinsics = CameraIntrinsicsProfile(
        camera_matrix=[[1500.0, 0.0, IMAGE_W / 2], [0.0, 1500.0, IMAGE_H / 2], [0.0, 0.0, 1.0]],
        distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0],
        image_width=IMAGE_W,
        image_height=IMAGE_H,
        rms_reprojection_error=0.0,
    )
    cfg.calibration = CalibrationService().compute(
        IMAGE_CORNERS, arena_w, arena_h, (IMAGE_H, IMAGE_W, 3)
    )
    colours = {}
    for name, (h, s, v) in SIM_HSV.items():
        lo, hi = h - 8, h + 8
        ranges = [[lo % 180, 179], [0, hi % 180]] if lo < 0 or hi > 179 else [[lo, hi]]
        colours[name] = {"hue_ranges": ranges, "sat_min": 90, "sat_max": 255, "val_min": 90, "val_max": 255}
    cfg.colour_calibration = ColourCalibrationProfile(colours=colours)
    cfg.rover_length_mm = 155.5
    cfg.rover_width_mm = 121.3
    cfg.rover_centre_left_mm = 0.0
    cfg.rover_centre_forward_mm = 63.0
    cfg.heading_reply_settle_seconds = 0.5
    cfg.heading_reply_samples = 2
    return cfg


@dataclass
class SimBlock:
    colour: str
    x_mm: float
    y_mm: float
    length_mm: float
    breadth_mm: float
    orientation_deg: float  # simulator convention


@dataclass
class SimWorld:
    """Ground truth of the simulated arena (thread-safe)."""

    arena_w: float = 1710.0
    arena_h: float = 1254.0
    x_mm: float = 400.0
    y_mm: float = 250.0
    heading_deg: float = 0.0
    blocks: list[SimBlock] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_pose(self, x: float, y: float, h: float) -> None:
        with self.lock:
            self.x_mm, self.y_mm, self.heading_deg = x, y, h % 360.0

    def pose(self) -> tuple[float, float, float]:
        with self.lock:
            return self.x_mm, self.y_mm, self.heading_deg

    def place_task(self, task, rng: np.random.Generator | None = None, noise_mm: float = 0.0) -> None:
        """Place rover and blocks as the operator would, with optional placement error."""

        def n(scale: float) -> float:
            return float(rng.normal(0.0, scale)) if rng is not None and scale > 0 else 0.0

        with self.lock:
            self.arena_w, self.arena_h = task.arena_width_mm, task.arena_height_mm
            self.x_mm = task.start_x_mm + n(noise_mm / 2)
            self.y_mm = task.start_y_mm + n(noise_mm / 2)
            self.heading_deg = (task.start_heading_deg + n(noise_mm / 10)) % 360.0
            self.blocks = [
                SimBlock(b.colour, b.x_mm + n(noise_mm), b.y_mm + n(noise_mm), b.length_mm, b.breadth_mm,
                         b.orientation_deg + n(noise_mm / 10))
                for b in task.blocks
            ]


class SyntheticCamera:
    """Renders SimWorld to a 1920x1080 BGR frame through the synthetic homography."""

    def __init__(self, world: SimWorld, config: AppConfig,
                 parallax: tuple[float, float, float, float] | None = None) -> None:
        """parallax = (camera height, marker height, nadir x, nadir y) in mm draws the
        marker where a floor homography would see a raised marker; None draws it on the floor."""
        self.world = world
        self.parallax = parallax
        h_img_to_world = np.asarray(config.calibration.homography, dtype=np.float64)
        self.h_world_to_img = np.linalg.inv(h_img_to_world)
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.rover_marker_dictionary))
        marker = cv2.aruco.generateImageMarker(dictionary, int(config.rover_marker_id), 200)
        self.marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        self.centre_forward = config.rover_centre_forward_mm
        self.centre_left = config.rover_centre_left_mm
        self.rover_len = config.rover_length_mm
        self.rover_wid = config.rover_width_mm
        self.colours = {k: _hsv_to_bgr(v) for k, v in SIM_HSV.items()}
        self.base = self._render_floor()

    def _to_img(self, pts_mm) -> np.ndarray:
        pts = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.h_world_to_img).reshape(-1, 2)

    def _render_floor(self) -> np.ndarray:
        frame = np.full((IMAGE_H, IMAGE_W, 3), 45, np.uint8)
        W, H = self.world.arena_w, self.world.arena_h
        quad = self._to_img([(0, 0), (W, 0), (W, H), (0, H)]).astype(np.int32)
        cv2.fillPoly(frame, [quad], (150, 150, 145))
        cv2.polylines(frame, [quad], True, (30, 30, 30), 3)
        return frame

    @staticmethod
    def _rect(cx, cy, length, breadth, heading):
        th = math.radians(heading)
        f = (math.sin(th), math.cos(th))
        r = (math.cos(th), -math.sin(th))
        hl, hb = length / 2, breadth / 2
        return [(cx + a * f[0] + b * r[0], cy + a * f[1] + b * r[1]) for a, b in ((hl, -hb), (hl, hb), (-hl, hb), (-hl, -hb))]

    def render(self) -> np.ndarray:
        frame = self.base.copy()
        with self.world.lock:
            blocks = list(self.world.blocks)
            x, y, h = self.world.x_mm, self.world.y_mm, self.world.heading_deg
        for b in blocks:
            poly = self._to_img(self._rect(b.x_mm, b.y_mm, b.length_mm, b.breadth_mm, b.orientation_deg))
            cv2.fillPoly(frame, [np.round(poly).astype(np.int32)], self.colours[b.colour])
        body = self._to_img(self._rect(x, y, self.rover_len, self.rover_wid, h))
        cv2.fillPoly(frame, [np.round(body).astype(np.int32)], (235, 235, 235))
        th = math.radians(h)
        fwd = np.array([math.sin(th), math.cos(th)])
        left = np.array([-math.cos(th), math.sin(th)])
        centre = np.array([x, y]) - self.centre_forward * fwd - self.centre_left * left
        s = MARKER_SIZE_MM / 2
        right = -left
        corners_mm = [centre + s * fwd - s * right, centre + s * fwd + s * right,
                      centre - s * fwd + s * right, centre - s * fwd - s * right]  # TL, TR, BR, BL
        if self.parallax is not None:
            H, hm, nx, ny = self.parallax
            n = np.array([nx, ny])
            corners_mm = [n + (c - n) * H / (H - hm) for c in corners_mm]
        dst = self._to_img(corners_mm).astype(np.float32)
        src = np.float32([[0, 0], [199, 0], [199, 199], [0, 199]])
        m = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(self.marker, m, (IMAGE_W, IMAGE_H), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((200, 200), 255, np.uint8), m, (IMAGE_W, IMAGE_H))
        frame[mask > 127] = warped[mask > 127]
        return frame
