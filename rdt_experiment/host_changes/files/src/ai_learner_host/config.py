from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CameraIntrinsicsProfile:
    camera_matrix: list[list[float]] = field(default_factory=list)
    distortion_coefficients: list[float] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    rms_reprojection_error: float | None = None

    @property
    def valid(self) -> bool:
        return (
            len(self.camera_matrix) == 3
            and all(len(row) == 3 for row in self.camera_matrix)
            and len(self.distortion_coefficients) >= 4
            and self.image_width > 0
            and self.image_height > 0
        )


@dataclass(slots=True)
class CalibrationProfile:
    """Arena homography profile.

    `homography` maps UNDISTORTED camera pixels into the project world frame:
      origin = arena bottom-left, +X right, +Y up, units = millimetres.

    `birdseye_homography` maps those same undistorted camera pixels to a
    top-down image used by the perception pipeline.
    """

    image_points: list[list[float]] = field(default_factory=list)
    arena_width_mm: float = 2500.0
    arena_height_mm: float = 2500.0
    homography: list[list[float]] = field(default_factory=list)
    birdseye_homography: list[list[float]] = field(default_factory=list)
    birdseye_width: int = 1000
    birdseye_height: int = 1000
    camera_width: int = 0
    camera_height: int = 0
    uses_undistorted_frame: bool = True

    @property
    def valid(self) -> bool:
        return (
            len(self.image_points) == 4
            and len(self.homography) == 3
            and all(len(row) == 3 for row in self.homography)
            and len(self.birdseye_homography) == 3
            and all(len(row) == 3 for row in self.birdseye_homography)
            and self.arena_width_mm > 0
            and self.arena_height_mm > 0
            and self.birdseye_width > 0
            and self.birdseye_height > 0
        )


@dataclass(slots=True)
class ColourCalibrationProfile:
    colour_space: str = "OpenCV HSV"
    patch_radius_px: int = 10
    colours: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        required = {"red", "green", "yellow", "blue"}
        if not required.issubset(self.colours):
            return False
        for colour in required:
            item = self.colours.get(colour, {})
            if not item.get("hue_ranges"):
                return False
            if "sat_min" not in item or "val_min" not in item:
                return False
        return True


@dataclass(slots=True)
class AppConfig:
    rover_uri: str = "ws://192.168.4.1:8266"
    camera_url: str = "http://192.168.4.2/stream"
    auto_reconnect: bool = True
    reconnect_seconds: float = 3.0
    microphone_name: str = "Default system microphone"
    push_to_talk_key: str = "Space"
    planner_provider: str = "openai"
    planner_model: str = "gpt-6-sol"
    planner_max_output_tokens: int = 8192
    confirm_before_send: bool = True
    max_primitives: int = 12
    max_forward_mm: int = 2000
    max_backward_mm: int = 1000
    max_turn_deg: int = 180
    planning_clearance_cm: int = 15
    manual_jog_mm: int = 100
    manual_turn_deg: int = 15
    telemetry_on_start: bool = True

    # Measured physical rover footprint used by planning and world-state JSON.
    rover_length_mm: float = 155.5
    rover_width_mm: float = 121.3

    # Voice transcription uses the dedicated OpenAI transcription endpoint.
    voice_transcription_model: str = "gpt-transcribe"

    # Perception configuration established during CV development.
    rover_marker_dictionary: str = "DICT_4X4_50"
    rover_marker_id: int = 29
    rover_centre_left_mm: float = 0.0
    rover_centre_forward_mm: float = 63.0
    rover_heading_offset_deg: float = 0.0

    # Marker height parallax. The arena homography maps the floor plane, but
    # the ArUco marker sits above the floor, so the camera sees it displaced
    # radially away from the point directly below the camera (the nadir) by
    # the factor H / (H - h). With both heights > 0 the measured marker centre
    # p is corrected to n + (p - n) * (H - h) / H before the marker-to-centre
    # offset is applied. Headings are unaffected (uniform scaling about a
    # point preserves directions). 0 disables the correction.
    parallax_camera_height_mm: float = 0.0
    parallax_marker_height_mm: float = 0.0
    parallax_nadir_x_mm: float = 0.0
    parallax_nadir_y_mm: float = 0.0

    # Camera-verified turn handshake.
    #
    # A heading request must never be answered from the cached frame that was
    # captured before the rover finished turning. The host therefore waits for
    # the stream to catch up, then requires several distinct fresh frames with
    # mutually-consistent raw ArUco headings before replying.
    heading_reply_settle_seconds: float = 10.0
    heading_reply_timeout_seconds: float = 18.0
    heading_reply_samples: int = 3
    heading_reply_max_spread_deg: float = 6.0

    intrinsics: CameraIntrinsicsProfile = field(default_factory=CameraIntrinsicsProfile)
    calibration: CalibrationProfile = field(default_factory=CalibrationProfile)
    colour_calibration: ColourCalibrationProfile = field(default_factory=ColourCalibrationProfile)


class ConfigRepository:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("data/settings.json")

    def load(self) -> AppConfig:
        if not self.path.exists():
            cfg = AppConfig()
            self.save(cfg)
            return cfg
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))

            intrinsics_raw = raw.pop("intrinsics", {})
            calibration_raw = raw.pop("calibration", {})
            colours_raw = raw.pop("colour_calibration", {})

            # Backwards compatibility with the V1 arena profile. Old profiles did
            # not store a bird's-eye matrix and are intentionally treated as
            # needing recalibration rather than silently changing coordinates.
            calibration = CalibrationProfile(**calibration_raw)
            intrinsics = CameraIntrinsicsProfile(**intrinsics_raw)
            colour_calibration = ColourCalibrationProfile(**colours_raw)

            # Migrate pre-LLM-integration settings to the production default.
            if raw.get("planner_model") in {None, "", "placeholder-v1"}:
                raw["planner_provider"] = "openai"
                raw["planner_model"] = "gpt-6-sol"
            if raw.get("planner_provider") == "Placeholder planner":
                raw["planner_provider"] = "openai"

            # Ignore stale/unknown keys rather than making an old settings file
            # prevent the application from starting after an upgrade.
            allowed = set(AppConfig.__dataclass_fields__) - {
                "intrinsics", "calibration", "colour_calibration"
            }
            raw = {key: value for key, value in raw.items() if key in allowed}

            config = AppConfig(
                **raw,
                intrinsics=intrinsics,
                calibration=calibration,
                colour_calibration=colour_calibration,
            )
            return config
        except (OSError, ValueError, TypeError):
            return AppConfig()

    def save(self, config: AppConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    def update(self, config: AppConfig, **changes: Any) -> AppConfig:
        for key, value in changes.items():
            if not hasattr(config, key):
                raise AttributeError(key)
            setattr(config, key, value)
        self.save(config)
        return config
