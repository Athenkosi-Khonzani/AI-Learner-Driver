from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from threading import RLock
import time
from typing import Any

import cv2
import numpy as np

from ai_learner_host.camera.frame_buffer import FrameBuffer
from ai_learner_host.config import (
    AppConfig,
    CalibrationProfile,
    CameraIntrinsicsProfile,
    ColourCalibrationProfile,
)
from ai_learner_host.domain.models import DetectedObject, RoverPose, WorldStateSnapshot


COLOURS = ("red", "green", "yellow", "blue")
DISPLAY_COLOURS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
}


class CalibrationService:
    """Camera-intrinsic and arena-homography calibration helpers.

    Arena clicks are performed on the UNDISTORTED image. The world frame is the
    one used throughout the project: bottom-left origin, +X right, +Y up, mm.
    """

    def load_intrinsics_file(self, path: str | Path) -> CameraIntrinsicsProfile:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        camera_matrix = raw.get("camera_matrix", [])
        distortion = raw.get("distortion_coefficients", raw.get("dist_coeffs", []))

        # Some calibration files store distortion as [[...]]. Normalise it.
        if distortion and isinstance(distortion[0], list):
            distortion = distortion[0]

        width = int(raw.get("image_width", raw.get("camera_width", 0)))
        height = int(raw.get("image_height", raw.get("camera_height", 0)))
        rms = raw.get("rms_reprojection_error", raw.get("rms"))

        profile = CameraIntrinsicsProfile(
            camera_matrix=[[float(v) for v in row] for row in camera_matrix],
            distortion_coefficients=[float(v) for v in distortion],
            image_width=width,
            image_height=height,
            rms_reprojection_error=float(rms) if rms is not None else None,
        )
        if not profile.valid:
            raise ValueError(
                "Camera calibration JSON must contain camera_matrix, "
                "distortion_coefficients, image_width and image_height"
            )
        return profile

    def undistort(self, frame: np.ndarray, intrinsics: CameraIntrinsicsProfile) -> np.ndarray:
        if not intrinsics.valid:
            raise RuntimeError("Intrinsic camera calibration is not loaded")
        height, width = frame.shape[:2]
        if (width, height) != (intrinsics.image_width, intrinsics.image_height):
            raise RuntimeError(
                f"Camera frame is {width}×{height}, but intrinsic calibration expects "
                f"{intrinsics.image_width}×{intrinsics.image_height}"
            )
        camera_matrix = np.asarray(intrinsics.camera_matrix, dtype=np.float64)
        distortion = np.asarray(intrinsics.distortion_coefficients, dtype=np.float64)
        return cv2.undistort(frame, camera_matrix, distortion)

    def compute(
        self,
        image_points: list[list[float]],
        arena_width_mm: float,
        arena_height_mm: float,
        frame_shape: tuple[int, ...],
        *,
        birdseye_width: int = 1000,
        birdseye_height: int = 1000,
    ) -> CalibrationProfile:
        if len(image_points) != 4:
            raise ValueError("Exactly four image points are required")

        # Input order: TL, TR, BR, BL.
        src = np.asarray(image_points, dtype=np.float32)

        # Project world frame: bottom-left origin, +Y up.
        world = np.asarray(
            [
                [0.0, arena_height_mm],
                [arena_width_mm, arena_height_mm],
                [arena_width_mm, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )

        bird = np.asarray(
            [
                [0.0, 0.0],
                [birdseye_width - 1.0, 0.0],
                [birdseye_width - 1.0, birdseye_height - 1.0],
                [0.0, birdseye_height - 1.0],
            ],
            dtype=np.float32,
        )

        h_world = cv2.getPerspectiveTransform(src, world)
        h_bird = cv2.getPerspectiveTransform(src, bird)
        height, width = frame_shape[:2]

        return CalibrationProfile(
            image_points=[[float(x), float(y)] for x, y in src],
            arena_width_mm=float(arena_width_mm),
            arena_height_mm=float(arena_height_mm),
            homography=h_world.tolist(),
            birdseye_homography=h_bird.tolist(),
            birdseye_width=int(birdseye_width),
            birdseye_height=int(birdseye_height),
            camera_width=int(width),
            camera_height=int(height),
            uses_undistorted_frame=True,
        )

    def prepare_arena_frame(
        self,
        frame: np.ndarray,
        intrinsics: CameraIntrinsicsProfile,
    ) -> np.ndarray:
        return self.undistort(frame, intrinsics)

    def warp_birdseye(
        self,
        frame: np.ndarray,
        intrinsics: CameraIntrinsicsProfile,
        profile: CalibrationProfile,
    ) -> np.ndarray:
        if not profile.valid:
            raise RuntimeError("Arena homography calibration is not available")
        undistorted = self.undistort(frame, intrinsics)
        h = np.asarray(profile.birdseye_homography, dtype=np.float64)
        return cv2.warpPerspective(
            undistorted,
            h,
            (profile.birdseye_width, profile.birdseye_height),
        )

    def pixel_to_world(
        self, x: float, y: float, profile: CalibrationProfile
    ) -> tuple[float, float]:
        if not profile.valid:
            raise RuntimeError("Arena calibration is not available")
        h = np.asarray(profile.homography, dtype=np.float64)
        p = np.asarray([[[x, y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(p, h.astype(np.float32))[0][0]
        return float(result[0]), float(result[1])


class ColourCalibrationService:
    """Learns robust HSV thresholds from user-clicked colour patches."""

    HUE_MARGIN = 3.0
    MIN_HUE_HALF_WIDTH = 8.0
    SATURATION_MARGIN = 40.0
    VALUE_MARGIN = 40.0
    MIN_COLOUR_SATURATION = 40.0

    @staticmethod
    def patch_pixels(hsv: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
        h, w = hsv.shape[:2]
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)
        return hsv[y1:y2, x1:x2].reshape(-1, 3)

    @staticmethod
    def _circular_difference(hue: np.ndarray, centre: float) -> np.ndarray:
        return ((hue - centre + 90.0) % 180.0) - 90.0

    @staticmethod
    def _wrapped_ranges(centre: float, half_width: float) -> list[list[int]]:
        low = centre - half_width
        high = centre + half_width
        if low < 0:
            return [
                [0, int(min(179, math.ceil(high)))],
                [int(max(0, math.floor(180 + low))), 179],
            ]
        if high > 179:
            return [
                [int(max(0, math.floor(low))), 179],
                [0, int(min(179, math.ceil(high - 180)))],
            ]
        return [[int(max(0, math.floor(low))), int(min(179, math.ceil(high)))]]

    def _calculate_colour(self, patches: list[np.ndarray]) -> dict[str, Any]:
        pixels = np.concatenate(patches, axis=0).astype(np.float64)
        hue = pixels[:, 0]
        saturation = pixels[:, 1]
        value = pixels[:, 2]

        valid = saturation >= self.MIN_COLOUR_SATURATION
        hue_valid = hue[valid]
        sat_valid = saturation[valid]
        val_valid = value[valid]
        if len(hue_valid) == 0:
            raise ValueError("No sufficiently saturated colour pixels were sampled")

        angles = hue_valid * 2.0 * math.pi / 180.0
        centre_angle = math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        if centre_angle < 0:
            centre_angle += 2.0 * math.pi
        centre_hue = centre_angle * 180.0 / (2.0 * math.pi)

        differences = self._circular_difference(hue_valid, centre_hue)
        low_diff, high_diff = np.percentile(differences, [1, 99])
        measured_half = max(abs(float(low_diff)), abs(float(high_diff)))
        half_width = max(measured_half + self.HUE_MARGIN, self.MIN_HUE_HALF_WIDTH)

        return {
            "hue_ranges": self._wrapped_ranges(centre_hue, half_width),
            "sat_min": int(max(0, np.percentile(sat_valid, 1) - self.SATURATION_MARGIN)),
            "sat_max": 255,
            "val_min": int(max(0, np.percentile(val_valid, 1) - self.VALUE_MARGIN)),
            "val_max": 255,
            "samples": len(patches),
            "statistics": {
                "hue_circular_centre": float(centre_hue),
                "hue_half_width": float(half_width),
                "saturation_median": float(np.median(sat_valid)),
                "value_median": float(np.median(val_valid)),
                "pixel_count": int(len(pixels)),
            },
        }

    def compute(
        self,
        sample_patches: dict[str, list[np.ndarray]],
        *,
        patch_radius_px: int = 10,
    ) -> ColourCalibrationProfile:
        missing = [colour for colour in COLOURS if not sample_patches.get(colour)]
        if missing:
            raise ValueError(f"Missing colour samples: {', '.join(missing)}")
        colours = {
            colour: self._calculate_colour(sample_patches[colour]) for colour in COLOURS
        }
        return ColourCalibrationProfile(
            colour_space="OpenCV HSV",
            patch_radius_px=int(patch_radius_px),
            colours=colours,
        )

    @staticmethod
    def create_mask(hsv: np.ndarray, threshold: dict[str, Any]) -> np.ndarray:
        final = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hue_low, hue_high in threshold["hue_ranges"]:
            lower = np.asarray(
                [hue_low, threshold["sat_min"], threshold["val_min"]], dtype=np.uint8
            )
            upper = np.asarray(
                [hue_high, threshold.get("sat_max", 255), threshold.get("val_max", 255)],
                dtype=np.uint8,
            )
            final = cv2.bitwise_or(final, cv2.inRange(hsv, lower, upper))
        return final


@dataclass(slots=True)
class _PoseMeasurement:
    x_mm: float
    y_mm: float
    heading_deg: float
    marker_side_px: float
    timestamp: float


@dataclass(slots=True)
class _FilteredPose:
    x_mm: float
    y_mm: float
    heading_deg: float
    confidence: float
    valid: bool
    stale: bool
    age_s: float


class _RoverPoseFilter:
    POSITION_TAU_S = 0.12
    HEADING_TAU_S = 0.08
    MAX_TRANSLATIONAL_SPEED_MM_S = 2500.0
    MAX_ANGULAR_SPEED_DEG_S = 900.0
    POSITION_MARGIN_MM = 30.0
    HEADING_MARGIN_DEG = 8.0
    HOLD_SECONDS = 0.50
    CONFIDENCE_WINDOW = 20
    MIN_MARKER_SIDE_PX = 20.0
    FULL_MARKER_SIDE_PX = 60.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.initialised = False
        self.x_mm = self.y_mm = self.heading_deg = 0.0
        self.last_filter_time: float | None = None
        self.last_good_time: float | None = None
        self.recent_acceptance: deque[int] = deque(maxlen=self.CONFIDENCE_WINDOW)
        self.last_size_score = 0.0

    @staticmethod
    def _angle_delta(target: float, current: float) -> float:
        return float(((target - current + 180.0) % 360.0) - 180.0)

    @staticmethod
    def _alpha(dt: float, tau: float) -> float:
        return float(1.0 - math.exp(-max(0.0, dt) / tau))

    def _size_score(self, side_px: float) -> float:
        if side_px <= self.MIN_MARKER_SIDE_PX:
            return 0.0
        if side_px >= self.FULL_MARKER_SIDE_PX:
            return 1.0
        return float(
            (side_px - self.MIN_MARKER_SIDE_PX)
            / (self.FULL_MARKER_SIDE_PX - self.MIN_MARKER_SIDE_PX)
        )

    def _plausible(self, m: _PoseMeasurement) -> bool:
        if not self.initialised or self.last_filter_time is None:
            return True
        dt = max(1e-3, m.timestamp - self.last_filter_time)
        position_change = math.hypot(m.x_mm - self.x_mm, m.y_mm - self.y_mm)
        heading_change = abs(self._angle_delta(m.heading_deg, self.heading_deg))
        return (
            position_change <= self.MAX_TRANSLATIONAL_SPEED_MM_S * dt + self.POSITION_MARGIN_MM
            and heading_change <= self.MAX_ANGULAR_SPEED_DEG_S * dt + self.HEADING_MARGIN_DEG
        )

    def update(self, measurement: _PoseMeasurement | None, now: float) -> _FilteredPose | None:
        accepted = False
        if measurement is not None:
            size_score = self._size_score(measurement.marker_side_px)
            if size_score > 0.0 and self._plausible(measurement):
                accepted = True
                self.last_size_score = size_score
                if not self.initialised:
                    self.x_mm = measurement.x_mm
                    self.y_mm = measurement.y_mm
                    self.heading_deg = measurement.heading_deg
                    self.initialised = True
                else:
                    dt = max(1e-3, now - (self.last_filter_time or now - 1.0 / 30.0))
                    a_pos = self._alpha(dt, self.POSITION_TAU_S)
                    a_heading = self._alpha(dt, self.HEADING_TAU_S)
                    self.x_mm += a_pos * (measurement.x_mm - self.x_mm)
                    self.y_mm += a_pos * (measurement.y_mm - self.y_mm)
                    self.heading_deg = (
                        self.heading_deg
                        + a_heading * self._angle_delta(measurement.heading_deg, self.heading_deg)
                    ) % 360.0
                self.last_good_time = now

        self.recent_acceptance.append(1 if accepted else 0)
        self.last_filter_time = now
        if not self.initialised or self.last_good_time is None:
            return None

        age = max(0.0, now - self.last_good_time)
        valid = age <= self.HOLD_SECONDS
        hit_ratio = sum(self.recent_acceptance) / max(1, len(self.recent_acceptance))
        base = 0.65 * hit_ratio + 0.35 * self.last_size_score
        confidence = float(np.clip(base * math.exp(-3.0 * age / self.HOLD_SECONDS), 0.0, 1.0))
        if not valid:
            confidence = 0.0
        return _FilteredPose(
            x_mm=float(self.x_mm),
            y_mm=float(self.y_mm),
            heading_deg=float(self.heading_deg),
            confidence=confidence,
            valid=valid,
            stale=age > 0.05,
            age_s=float(age),
        )


class PerceptionService:
    """Production perception facade used by the mission controller.

    It combines intrinsic correction, arena homography, calibrated HSV block
    detection, ArUco rover pose, mechanical centre compensation, temporal
    filtering, annotated output, JSON, Markdown, and a text-map world state.
    """

    MIN_BLOCK_AREA_MM2 = 10_000.0
    MIN_BLOCK_RECTANGULARITY = 0.70
    MIN_BLOCK_SIDE_MM = 40.0
    BLOCK_KERNEL = 5
    TEXT_MAP_CELL_MM = 100.0

    def __init__(self, frame_buffer: FrameBuffer, config: AppConfig) -> None:
        self.frame_buffer = frame_buffer
        self.config = config
        self.calibration_service = CalibrationService()
        self.colour_service = ColourCalibrationService()
        self._pose_filter = _RoverPoseFilter()
        self._lock = RLock()
        self._aruco_cache: tuple[str, Any, Any] | None = None

    def readiness_issues(self) -> list[str]:
        issues: list[str] = []
        if not self.config.intrinsics.valid:
            issues.append("camera intrinsics")
        if not self.config.calibration.valid:
            issues.append("arena homography")
        if not self.config.colour_calibration.valid:
            issues.append("colour calibration")
        return issues

    async def capture_world_state(self) -> WorldStateSnapshot:
        frame = self.frame_buffer.latest(max_age_seconds=5.0)
        if frame is None:
            raise RuntimeError("No recent camera frame is available")
        issues = self.readiness_issues()
        if issues:
            raise RuntimeError("Perception setup incomplete: " + ", ".join(issues))
        return await asyncio.to_thread(self._process_world, frame)

    async def capture_settled_world_state(self) -> WorldStateSnapshot:
        """Capture the post-path world state only after the camera has caught up.

        The physical camera stream can be several seconds behind the rover.
        Verification must therefore not immediately reuse the latest cached
        frame when a PATH ACK arrives.  We intentionally reuse
        ``heading_reply_settle_seconds`` as the single physical-camera settle
        tuning value for both turn verification and final goal verification.

        After the settle delay, the current frame sequence becomes a new
        baseline.  We then require ``heading_reply_samples`` distinct frames
        newer than that baseline and process the newest one.  This prevents a
        frame that was already cached before/during the delay from becoming
        the LLM's final world state.
        """
        issues = self.readiness_issues()
        if issues:
            raise RuntimeError("Perception setup incomplete: " + ", ".join(issues))

        settle_s = max(0.0, float(self.config.heading_reply_settle_seconds))
        required = max(1, int(self.config.heading_reply_samples))
        total_timeout_s = max(
            settle_s + 0.5,
            float(self.config.heading_reply_timeout_seconds),
        )
        request_time = time.monotonic()

        if settle_s > 0.0:
            await asyncio.sleep(settle_s)

        # Everything already in the buffer at this instant is considered
        # potentially stale.  Only frames arriving after this point qualify.
        baseline_sequence = self.frame_buffer.sequence()
        last_sequence = baseline_sequence
        fresh_count = 0
        newest_frame = None
        deadline = request_time + total_timeout_s

        while time.monotonic() < deadline:
            packet = self.frame_buffer.latest_packet(max_age_seconds=1.5)
            if packet is None or packet.sequence <= last_sequence:
                await asyncio.sleep(0.015)
                continue

            last_sequence = packet.sequence
            fresh_count += 1
            newest_frame = packet.frame

            if fresh_count >= required:
                # Do not let the final pose remain blended with a pre-path
                # temporal-filter state.  The newest settled frame seeds a
                # fresh pose estimate for final goal verification.
                self.reset_pose_filter()
                return await asyncio.to_thread(self._process_world, newest_frame)

            await asyncio.sleep(0)

        age = self.frame_buffer.age_seconds()
        age_text = "none" if age is None else f"{age:.3f}s"
        raise RuntimeError(
            "Final verification camera state did not become fresh before timeout "
            f"(settled {settle_s:.1f}s, baseline frame #{baseline_sequence}, "
            f"latest #{last_sequence}, received {fresh_count}/{required} new frames, "
            f"buffer age {age_text})"
        )

    async def get_heading(self) -> float:
        """Return a heading from the current latest frame.

        This remains useful for non-handshake callers. Rover ``req_heading``
        responses use :meth:`get_fresh_heading` below so that a pre-turn cached
        frame cannot be sent back after the rover has already rotated.
        """
        frame = self.frame_buffer.latest(max_age_seconds=2.0)
        if frame is None:
            raise RuntimeError("Cannot estimate heading without a recent camera frame")
        if not self.config.intrinsics.valid or not self.config.calibration.valid:
            raise RuntimeError("Heading requires camera intrinsics and arena homography")
        pose = await asyncio.to_thread(self._process_rover_only, frame)
        if pose is None or not pose.valid or pose.confidence < 0.35:
            raise RuntimeError("Rover ArUco pose is not reliable enough for a heading reply")
        return pose.heading_deg

    async def get_fresh_heading(
        self,
        *,
        requested_at_monotonic: float | None = None,
    ) -> float:
        """Wait for a stable heading from frames captured *after* the request.

        The turn primitive can finish before the camera stream has visibly
        caught up. Returning ``frame_buffer.latest()`` immediately can
        therefore replay the rover's old pre-turn heading, causing a second
        unnecessary turn.

        Strategy:
        1. remember the heading-request time;
        2. wait through a conservative camera-settling interval;
        3. only then snapshot the current frame sequence;
        4. sample distinct frames newer than that post-delay baseline;
        5. use raw ArUco headings (not a lagging temporal filter);
        6. require N headings to agree within a bounded circular spread.

        If the camera never becomes fresh/stable, no fabricated/stale heading
        is returned; the caller lets the rover's heading timeout handle it.
        """
        if not self.config.intrinsics.valid or not self.config.calibration.valid:
            raise RuntimeError("Heading requires camera intrinsics and arena homography")

        request_time = (
            float(requested_at_monotonic)
            if requested_at_monotonic is not None
            else time.monotonic()
        )

        settle_s = max(0.0, float(self.config.heading_reply_settle_seconds))
        timeout_s = max(
            settle_s + 0.25,
            float(self.config.heading_reply_timeout_seconds),
        )
        required = max(1, int(self.config.heading_reply_samples))
        max_spread = max(0.5, float(self.config.heading_reply_max_spread_deg))

        # Deliberately wait before sampling. With the physical overhead camera
        # the stream can lag the rover by several seconds even while frames are
        # still arriving. A sequence number alone therefore cannot prove that
        # the image depicts the rover *after* its turn.
        #
        # v0.3.4 uses a conservative 10 s default settle period. Only AFTER
        # that delay do we establish a fresh sequence baseline, which forces
        # every heading sample to come from frames received after the full
        # settle interval.
        if settle_s > 0.0:
            await asyncio.sleep(settle_s)

        baseline_sequence = self.frame_buffer.sequence()
        last_sequence = baseline_sequence
        sampling_started = time.monotonic()
        deadline = request_time + timeout_s

        headings: deque[float] = deque(maxlen=required)

        while time.monotonic() < deadline:
            packet = self.frame_buffer.latest_packet(max_age_seconds=1.5)

            if packet is None or packet.sequence <= last_sequence:
                await asyncio.sleep(0.015)
                continue

            last_sequence = packet.sequence

            heading = await asyncio.to_thread(
                self._raw_heading_for_reply,
                packet.frame,
            )

            if heading is None:
                await asyncio.sleep(0)
                continue

            headings.append(float(heading) % 360.0)

            if len(headings) < required:
                await asyncio.sleep(0)
                continue

            mean_heading = self._circular_mean_deg(headings)
            spread = max(
                abs(self._angle_delta_deg(sample, mean_heading))
                for sample in headings
            )

            if spread <= max_spread:
                return mean_heading

            # Keep a rolling window so a transient/motion-blurred frame drops
            # out naturally once the camera has settled.
            await asyncio.sleep(0)

        age = self.frame_buffer.age_seconds()
        age_text = "none" if age is None else f"{age:.3f}s"
        raise RuntimeError(
            "Fresh camera heading did not stabilise before timeout "
            f"(baseline frame #{baseline_sequence}, latest #{last_sequence}, "
            f"buffer age {age_text}, required {required} samples within "
            f"{max_spread:.1f}°)"
        )

    def _raw_heading_for_reply(self, frame: np.ndarray) -> float | None:
        """Single-frame ArUco heading for the heading handshake.

        This intentionally bypasses the temporal pose filter. The filter is
        appropriate for continuous world-state display, but a post-turn
        verification reply must not be pulled toward the old pre-turn angle.
        """
        with self._lock:
            now = time.monotonic()
            undistorted = self._undistort(frame)
            raw = self._raw_rover_measurement(undistorted, now)

        if raw is None:
            return None

        if raw.marker_side_px <= self._pose_filter.MIN_MARKER_SIDE_PX:
            return None

        return float(raw.heading_deg) % 360.0

    @staticmethod
    def _angle_delta_deg(target: float, current: float) -> float:
        return float(((target - current + 180.0) % 360.0) - 180.0)

    @staticmethod
    def _circular_mean_deg(values) -> float:
        values = list(values)
        if not values:
            raise ValueError("Cannot average an empty heading sequence")

        sin_sum = sum(math.sin(math.radians(v)) for v in values)
        cos_sum = sum(math.cos(math.radians(v)) for v in values)

        if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
            # Pathological antipodal set. Returning the newest sample is safer
            # than pretending there is a meaningful circular average.
            return float(values[-1]) % 360.0

        return (
            math.degrees(math.atan2(sin_sum, cos_sum))
            + 360.0
        ) % 360.0

    def reset_pose_filter(self) -> None:
        with self._lock:
            self._pose_filter.reset()

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        return self.calibration_service.undistort(frame, self.config.intrinsics)

    def _birdseye(self, undistorted: np.ndarray) -> np.ndarray:
        profile = self.config.calibration
        h = np.asarray(profile.birdseye_homography, dtype=np.float64)
        return cv2.warpPerspective(
            undistorted,
            h,
            (profile.birdseye_width, profile.birdseye_height),
        )

    def _aruco_detector(self):
        dictionary_name = self.config.rover_marker_dictionary
        cache = self._aruco_cache
        if cache is not None and cache[0] == dictionary_name:
            return cache[1], cache[2]
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            raise RuntimeError(f"Unknown ArUco dictionary: {dictionary_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = (
            cv2.aruco.ArucoDetector(dictionary, parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self._aruco_cache = (dictionary_name, detector, (dictionary, parameters))
        return detector, (dictionary, parameters)

    def _detect_marker(self, undistorted: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        detector, legacy = self._aruco_detector()
        if detector is not None:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            dictionary, parameters = legacy
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, dictionary, parameters=parameters
            )
        if ids is None:
            return None
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if int(marker_id) == int(self.config.rover_marker_id):
                return marker_corners.reshape(4, 2).astype(np.float64)
        return None

    @staticmethod
    def _line_intersection(p1, p2, p3, p4) -> np.ndarray:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-9:
            return np.mean(np.asarray([p1, p2, p3, p4]), axis=0)
        d12 = x1 * y2 - y1 * x2
        d34 = x3 * y4 - y3 * x4
        return np.asarray(
            [
                (d12 * (x3 - x4) - (x1 - x2) * d34) / denominator,
                (d12 * (y3 - y4) - (y1 - y2) * d34) / denominator,
            ],
            dtype=np.float64,
        )

    def correct_marker_parallax(self, point_mm: np.ndarray) -> np.ndarray:
        """Map a marker point measured on the floor plane to its true floor position.

        See AppConfig.parallax_*: p_true = n + (p - n) * (H - h) / H.
        Returns the point unchanged when the correction is disabled.
        """
        H = float(self.config.parallax_camera_height_mm)
        h = float(self.config.parallax_marker_height_mm)
        if H <= 0.0 or h <= 0.0 or h >= H:
            return np.asarray(point_mm, dtype=np.float64)
        nadir = np.asarray(
            [self.config.parallax_nadir_x_mm, self.config.parallax_nadir_y_mm],
            dtype=np.float64,
        )
        return nadir + (np.asarray(point_mm, dtype=np.float64) - nadir) * (H - h) / H

    def _transform_world(self, points: np.ndarray) -> np.ndarray:
        h = np.asarray(self.config.calibration.homography, dtype=np.float64)
        src = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(src, h.astype(np.float32)).reshape(-1, 2)

    def _raw_rover_measurement(
        self, undistorted: np.ndarray, now: float
    ) -> _PoseMeasurement | None:
        corners = self._detect_marker(undistorted)
        if corners is None:
            return None
        world_corners = self._transform_world(corners).astype(np.float64)
        top = 0.5 * (world_corners[0] + world_corners[1])
        bottom = 0.5 * (world_corners[2] + world_corners[3])
        vector = top - bottom
        heading = (
            math.degrees(math.atan2(float(vector[0]), float(vector[1])))
            + self.config.rover_heading_offset_deg
        ) % 360.0

        centre_px = self._line_intersection(corners[0], corners[2], corners[1], corners[3])
        marker_world = self.correct_marker_parallax(self._transform_world(centre_px.reshape(1, 2))[0])

        theta = math.radians(heading)
        forward = np.asarray([math.sin(theta), math.cos(theta)])
        left = np.asarray([-math.cos(theta), math.sin(theta)])
        rover_world = (
            marker_world
            + self.config.rover_centre_forward_mm * forward
            + self.config.rover_centre_left_mm * left
        )

        side_lengths = [
            float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)
        ]
        return _PoseMeasurement(
            x_mm=float(rover_world[0]),
            y_mm=float(rover_world[1]),
            heading_deg=float(heading),
            marker_side_px=float(np.mean(side_lengths)),
            timestamp=now,
        )

    def _process_rover_only(self, frame: np.ndarray) -> _FilteredPose | None:
        with self._lock:
            now = time.monotonic()
            undistorted = self._undistort(frame)
            raw = self._raw_rover_measurement(undistorted, now)
            return self._pose_filter.update(raw, now)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.BLOCK_KERNEL, self.BLOCK_KERNEL)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _detect_blocks(self, birdseye: np.ndarray) -> list[dict[str, Any]]:
        profile = self.config.calibration
        hsv = cv2.cvtColor(birdseye, cv2.COLOR_BGR2HSV)
        mm_per_px_x = profile.arena_width_mm / max(1, profile.birdseye_width - 1)
        mm_per_px_y = profile.arena_height_mm / max(1, profile.birdseye_height - 1)

        def point_world(point: np.ndarray) -> np.ndarray:
            x_px, y_px = point
            return np.asarray(
                [
                    x_px * mm_per_px_x,
                    profile.arena_height_mm - y_px * mm_per_px_y,
                ],
                dtype=np.float64,
            )

        blocks: list[dict[str, Any]] = []
        for colour in COLOURS:
            threshold = self.config.colour_calibration.colours[colour]
            mask = self._clean_mask(self.colour_service.create_mask(hsv, threshold))
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if len(contour) < 3:
                    continue
                area_px = float(cv2.contourArea(contour))
                area_mm2 = area_px * mm_per_px_x * mm_per_px_y
                if area_mm2 < self.MIN_BLOCK_AREA_MM2:
                    continue
                rect = cv2.minAreaRect(contour)
                centre_px = np.asarray(rect[0], dtype=np.float64)
                box_px = cv2.boxPoints(rect).astype(np.float64)
                box_world = np.asarray([point_world(p) for p in box_px])
                edge01 = float(np.linalg.norm(box_world[1] - box_world[0]))
                edge12 = float(np.linalg.norm(box_world[2] - box_world[1]))
                long_side = max(edge01, edge12)
                short_side = min(edge01, edge12)
                if long_side < self.MIN_BLOCK_SIDE_MM or short_side < self.MIN_BLOCK_SIDE_MM:
                    continue
                rectangularity = area_mm2 / max(1e-6, long_side * short_side)
                if rectangularity < self.MIN_BLOCK_RECTANGULARITY:
                    continue
                centre_world = point_world(centre_px)
                if edge01 >= edge12:
                    vector = box_world[1] - box_world[0]
                else:
                    vector = box_world[2] - box_world[1]
                orientation = math.degrees(math.atan2(vector[1], vector[0])) % 180.0
                blocks.append(
                    {
                        "colour": colour,
                        "centre_px": centre_px,
                        "centre_mm": centre_world,
                        "box_px": box_px,
                        "box_world_mm": box_world,
                        "width_mm": long_side,
                        "height_mm": short_side,
                        "orientation_deg": orientation,
                    }
                )

        blocks.sort(key=lambda b: (COLOURS.index(b["colour"]), b["centre_mm"][0]))
        counts = {colour: 0 for colour in COLOURS}
        for block in blocks:
            colour = block["colour"]
            counts[colour] += 1
            block["id"] = f"{colour}_block_{counts[colour]:02d}"
        return blocks

    def _scene_json(self, pose: _FilteredPose | None, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        profile = self.config.calibration
        rover: dict[str, Any] = {
            "available": pose is not None,
            "pose_valid": bool(pose and pose.valid),
            "confidence": round(pose.confidence, 3) if pose else 0.0,
            "footprint_mm": {
                "length": round(float(self.config.rover_length_mm), 1),
                "width": round(float(self.config.rover_width_mm), 1),
            },
            "pose": None,
        }
        if pose is not None:
            rover.update(
                {
                    "stale": pose.stale,
                    "age_s": round(pose.age_s, 3),
                    "pose": {
                        "x_mm": round(pose.x_mm, 1),
                        "y_mm": round(pose.y_mm, 1),
                        "heading_deg": round(pose.heading_deg, 1),
                    },
                }
            )

        obstacles = []
        for block in blocks:
            obstacles.append(
                {
                    "id": block["id"],
                    "type": "coloured_block",
                    "colour": block["colour"],
                    "centre_mm": {
                        "x": round(float(block["centre_mm"][0]), 1),
                        "y": round(float(block["centre_mm"][1]), 1),
                    },
                    "size_mm": {
                        "long_side": round(float(block["width_mm"]), 1),
                        "short_side": round(float(block["height_mm"]), 1),
                    },
                    "orientation_deg": round(float(block["orientation_deg"]), 1),
                    "polygon_mm": [
                        {"x": round(float(p[0]), 1), "y": round(float(p[1]), 1)}
                        for p in block["box_world_mm"]
                    ],
                }
            )

        return {
            "schema": "ai_learner_world_state",
            "schema_version": "1.0",
            "coordinate_frame": {
                "units": "mm",
                "origin": "arena_bottom_left",
                "x_axis": "right",
                "y_axis": "up",
                "heading": {
                    "0_deg": "+Y/up",
                    "90_deg": "+X/right",
                    "180_deg": "-Y/down",
                    "270_deg": "-X/left",
                },
            },
            "arena": {
                "width_mm": round(profile.arena_width_mm, 1),
                "height_mm": round(profile.arena_height_mm, 1),
            },
            "rover": rover,
            "obstacles": obstacles,
            "perception": {
                "block_count": len(obstacles),
                "rover_marker_id": self.config.rover_marker_id,
                "rover_marker_dictionary": self.config.rover_marker_dictionary,
            },
        }

    @staticmethod
    def _markdown(scene: dict[str, Any]) -> str:
        arena = scene["arena"]
        lines = [
            "# AI Learner World State",
            "",
            f"Arena: {arena['width_mm']:.0f} mm × {arena['height_mm']:.0f} mm.",
            "Coordinate frame: origin at bottom-left, +X right, +Y up, heading 0° toward +Y.",
            "",
            "## Rover",
        ]
        rover = scene["rover"]
        if rover["pose"] is None:
            lines.append("Rover pose unavailable.")
        else:
            pose = rover["pose"]
            footprint = rover.get("footprint_mm", {})
            lines.extend(
                [
                    f"- Position: ({pose['x_mm']:.1f}, {pose['y_mm']:.1f}) mm",
                    f"- Heading: {pose['heading_deg']:.1f}°",
                    f"- Footprint: {footprint.get('length', 0):.0f} mm long × {footprint.get('width', 0):.0f} mm wide",
                    f"- Pose valid: {rover['pose_valid']}",
                    f"- Confidence: {rover['confidence']:.3f}",
                ]
            )
        lines.extend(["", "## Coloured blocks"])
        if not scene["obstacles"]:
            lines.append("No coloured blocks detected.")
        else:
            for block in scene["obstacles"]:
                c = block["centre_mm"]
                s = block["size_mm"]
                lines.append(
                    f"- {block['id']}: {block['colour']} block, centre=({c['x']:.1f}, "
                    f"{c['y']:.1f}) mm, size≈{s['long_side']:.1f}×{s['short_side']:.1f} mm, "
                    f"orientation={block['orientation_deg']:.1f}°."
                )
        return "\n".join(lines)

    def _text_map(self, scene: dict[str, Any], cell_mm: float | None = None) -> str:
        cell = float(cell_mm or self.TEXT_MAP_CELL_MM)
        width = float(scene["arena"]["width_mm"])
        height = float(scene["arena"]["height_mm"])
        cols = max(1, int(math.ceil(width / cell)))
        rows = max(1, int(math.ceil(height / cell)))
        grid = [["." for _ in range(cols)] for _ in range(rows)]

        def coords(x: float, y: float) -> tuple[int, int]:
            col = max(0, min(cols - 1, int(x // cell)))
            bottom_row = max(0, min(rows - 1, int(y // cell)))
            return rows - 1 - bottom_row, col

        def place(row: int, col: int, symbol: str) -> None:
            grid[row][col] = symbol if grid[row][col] in {".", symbol} else "*"

        symbols = {"red": "R", "green": "G", "yellow": "Y", "blue": "B"}
        for block in scene["obstacles"]:
            row, col = coords(block["centre_mm"]["x"], block["centre_mm"]["y"])
            place(row, col, symbols.get(block["colour"], "?"))

        rover = scene["rover"]
        if rover["pose"] is not None:
            pose = rover["pose"]
            heading_symbols = ["^", "/", ">", "\\", "v", "/", "<", "\\"]
            idx = int(((pose["heading_deg"] % 360.0) + 22.5) // 45.0) % 8
            row, col = coords(pose["x_mm"], pose["y_mm"])
            place(row, col, heading_symbols[idx])

        lines = [
            "AI LEARNER WORLD STATE - TEXT MAP",
            f"Arena: {width:.0f} mm x {height:.0f} mm | Cell: {cell:.0f} mm",
            "Origin: bottom-left | +X right | +Y up | 0 deg = up",
            "",
        ]
        header = "y\\x ".ljust(8) + "".join(
            f"{int(col * cell):>4}" for col in range(cols)
        )
        lines.append(header)
        for row in range(rows):
            y = int((rows - 1 - row) * cell)
            lines.append(f"{y:>4} ".ljust(8) + "".join(f"{v:>4}" for v in grid[row]))
        lines.extend(
            [
                "",
                "Legend: R red | G green | Y yellow | B blue | ^ > v < / \\ rover | * overlap",
            ]
        )
        return "\n".join(lines)

    def _annotate(
        self, birdseye: np.ndarray, pose: _FilteredPose | None, blocks: list[dict[str, Any]]
    ) -> np.ndarray:
        output = birdseye.copy()
        profile = self.config.calibration
        for block in blocks:
            colour = DISPLAY_COLOURS[block["colour"]]
            box = np.round(block["box_px"]).astype(np.int32)
            cv2.polylines(output, [box], True, colour, 3)
            cx, cy = np.round(block["centre_px"]).astype(int)
            cv2.circle(output, (cx, cy), 6, colour, -1)
            cv2.putText(
                output,
                block["id"],
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
                cv2.LINE_AA,
            )

        if pose is not None:
            x_px = pose.x_mm / profile.arena_width_mm * (profile.birdseye_width - 1)
            y_px = (
                (profile.arena_height_mm - pose.y_mm)
                / profile.arena_height_mm
                * (profile.birdseye_height - 1)
            )
            centre = np.asarray([x_px, y_px])
            cv2.circle(output, tuple(np.round(centre).astype(int)), 9, (255, 255, 0), -1)
            theta = math.radians(pose.heading_deg)
            direction = np.asarray([math.sin(theta), -math.cos(theta)])
            end = centre + direction * 100.0
            cv2.arrowedLine(
                output,
                tuple(np.round(centre).astype(int)),
                tuple(np.round(end).astype(int)),
                (0, 255, 255),
                4,
                tipLength=0.25,
            )

        cv2.rectangle(output, (0, 0), (profile.birdseye_width, 52), (20, 20, 20), -1)
        if pose is None:
            status = f"Rover unavailable · blocks {len(blocks)}"
            colour = (0, 220, 255)
        else:
            status = (
                f"Rover ({pose.x_mm:.0f}, {pose.y_mm:.0f}) mm · "
                f"{pose.heading_deg:.1f}° · conf {pose.confidence:.2f} · blocks {len(blocks)}"
            )
            colour = (0, 255, 0) if pose.valid and pose.confidence >= 0.75 else (0, 220, 255)
        cv2.putText(
            output, status, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2, cv2.LINE_AA
        )
        return output

    def _process_world(self, frame: np.ndarray) -> WorldStateSnapshot:
        with self._lock:
            now = time.monotonic()
            undistorted = self._undistort(frame)
            birdseye = self._birdseye(undistorted)
            blocks = self._detect_blocks(birdseye)
            raw_pose = self._raw_rover_measurement(undistorted, now)
            pose = self._pose_filter.update(raw_pose, now)
            scene = self._scene_json(pose, blocks)
            markdown = self._markdown(scene)
            text_map = self._text_map(scene)
            annotated = self._annotate(birdseye, pose, blocks)

            rover_pose = RoverPose(
                x_mm=pose.x_mm if pose else None,
                y_mm=pose.y_mm if pose else None,
                heading_deg=pose.heading_deg if pose else None,
                confidence=pose.confidence if pose else 0.0,
            )
            objects = [
                DetectedObject(
                    label=block["id"],
                    x_mm=float(block["centre_mm"][0]),
                    y_mm=float(block["centre_mm"][1]),
                    confidence=float(min(1.0, block.get("rectangularity", 1.0))),
                )
                for block in blocks
            ]
            return WorldStateSnapshot(
                raw_frame=frame,
                annotated_frame=annotated,
                rover_pose=rover_pose,
                detected_objects=objects,
                obstacles=list(objects),
                arena_width_mm=self.config.calibration.arena_width_mm,
                arena_height_mm=self.config.calibration.arena_height_mm,
                scene_json=scene,
                scene_markdown=markdown,
                scene_text_map=text_map,
            )
