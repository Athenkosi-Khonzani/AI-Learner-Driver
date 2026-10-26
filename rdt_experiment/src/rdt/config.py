"""Experiment configuration (config/rdt_config.json)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdt._bootstrap import PACKAGE_ROOT

DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "rdt_config.json"


@dataclass(slots=True)
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path
    sha256: str
    host_overrides: dict[str, Any] = field(default_factory=dict)

    # convenience accessors -------------------------------------------------
    @property
    def rover_length_mm(self) -> float:
        return float(self.host_overrides.get("rover_length_mm", 155.5))

    @property
    def rover_width_mm(self) -> float:
        return float(self.host_overrides.get("rover_width_mm", 121.3))

    @property
    def verification_default(self) -> bool:
        return bool(self.raw.get("verification", {}).get("default_enabled", True))

    @property
    def max_correction_rounds(self) -> int:
        return int(self.raw.get("verification", {}).get("max_correction_rounds", 2))

    @property
    def repetitions(self) -> int:
        return int(self.raw.get("schedule", {}).get("repetitions", 3))

    @property
    def seed(self) -> int:
        return int(self.raw.get("schedule", {}).get("seed", 0))

    @property
    def randomise_order(self) -> bool:
        return bool(self.raw.get("schedule", {}).get("randomise_order", True))

    @property
    def tape_check_every(self) -> int:
        return int(self.raw.get("schedule", {}).get("tape_check_every", 3))

    @property
    def placement_limits(self):
        from rdt.tasks import PlacementLimits

        return PlacementLimits(**self.raw.get("placement_limits", {}))

    @property
    def min_battery_v(self) -> float | None:
        value = self.raw.get("battery", {}).get("min_voltage_v")
        return None if value is None else float(value)

    @property
    def recording(self) -> dict[str, Any]:
        return dict(self.raw.get("recording", {}))

    @property
    def scoring(self) -> dict[str, Any]:
        return dict(self.raw.get("scoring", {}))

    @property
    def trial_timeout_seconds(self) -> float:
        return float(self.raw.get("trial_timeout_seconds", 420))

    def apply_to_host(self, host_config, skip: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
        """Apply overrides in memory and return {key: (old, new)} for the record."""
        changed: dict[str, Any] = {}
        for key, value in self.host_overrides.items():
            if key in skip:
                continue
            if not hasattr(host_config, key):
                raise AttributeError(f"Host AppConfig has no field {key!r}")
            old = getattr(host_config, key)
            if old != value:
                changed[key] = [old, value]
            setattr(host_config, key, value)
        return changed


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = path.read_bytes()
    raw = json.loads(data.decode("utf-8"))
    return ExperimentConfig(
        raw=raw,
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        host_overrides=dict(raw.get("host_overrides", {})),
    )
