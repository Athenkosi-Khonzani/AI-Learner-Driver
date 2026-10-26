"""Sessions: a fixed, seeded schedule of trials and the folder that holds them."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rdt._bootstrap import PACKAGE_ROOT
from rdt.config import ExperimentConfig
from rdt.tasks import RDTTask, load_task_set, sha256_file, verify_manifest

RESULTS_ROOT = PACKAGE_ROOT / "results"
MODES = ("main", "dry_run", "subsystem")


@dataclass
class Slot:
    slot: int
    task_id: str
    rep: int
    tape_check: bool


class Session:
    def __init__(self, root: Path, data: dict[str, Any]) -> None:
        self.root = Path(root)
        self.data = data
        tasks, _ = load_task_set(self.task_dir)
        self.tasks: dict[str, RDTTask] = {t.task_id: t for t in tasks}

    # ------------------------------------------------------------------ create/open
    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def mode(self) -> str:
        return self.data["mode"]

    @property
    def task_dir(self) -> Path:
        return PACKAGE_ROOT / self.data["task_dir"]

    @property
    def trials_dir(self) -> Path:
        return self.root / "trials"

    @property
    def verification_default(self) -> bool:
        return bool(self.data.get("verification_default", True))

    @property
    def schedule(self) -> list[Slot]:
        return [Slot(**s) for s in self.data["schedule"]]

    @classmethod
    def create(
        cls,
        name: str,
        mode: str,
        cfg: ExperimentConfig,
        *,
        results_root: Path = RESULTS_ROOT,
        repetitions: int | None = None,
        task_ids: list[str] | None = None,
        verification_default: bool | None = None,
        allow_unlocked: bool = False,
    ) -> "Session":
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "session"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(results_root) / f"{stamp}_{mode}_{safe}"
        task_rel = "tasks" if mode != "subsystem" else "tasks/subsystem"
        task_dir = PACKAGE_ROOT / task_rel
        manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
        problems = verify_manifest(task_dir)
        if mode == "main":
            if not manifest.get("locked") and not allow_unlocked:
                raise RuntimeError("The main session needs a locked task manifest (scripts/validate_tasks.py --lock).")
            if problems:
                raise RuntimeError("Task files differ from the locked manifest: " + "; ".join(problems))
        tasks, _ = load_task_set(task_dir)
        if task_ids:
            tasks = [t for t in tasks if t.task_id in task_ids]
        reps = repetitions or (cfg.repetitions if mode == "main" else 1)
        pairs = [(t.task_id, r) for t in tasks for r in range(1, reps + 1)]
        if cfg.randomise_order:
            random.Random(cfg.seed).shuffle(pairs)
            # renumber repetitions in order of appearance so "run 1" always comes first
            seen: dict[str, int] = {}
            renumbered = []
            for tid, _ in pairs:
                seen[tid] = seen.get(tid, 0) + 1
                renumbered.append((tid, seen[tid]))
            pairs = renumbered
        every = max(1, cfg.tape_check_every)
        schedule = [
            {"slot": i + 1, "task_id": tid, "rep": rep, "tape_check": (i % every) == every - 1}
            for i, (tid, rep) in enumerate(pairs)
        ]
        data = {
            "name": name,
            "mode": mode,
            "created": datetime.now().isoformat(timespec="seconds"),
            "task_dir": task_rel,
            "manifest_locked": bool(manifest.get("locked")),
            "manifest_sha256": sha256_file(task_dir / "manifest.json"),
            "manifest_problems": problems,
            "config_sha256": cfg.sha256,
            "config": cfg.raw,
            "seed": cfg.seed,
            "repetitions": reps,
            "verification_default": cfg.verification_default if verification_default is None else bool(verification_default),
            "schedule": schedule,
        }
        if mode == "subsystem":
            data["verification_default"] = False
        root.mkdir(parents=True, exist_ok=False)
        (root / "trials").mkdir()
        (root / "session.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        return cls(root, data)

    @classmethod
    def open(cls, root: str | Path) -> "Session":
        root = Path(root)
        return cls(root, json.loads((root / "session.json").read_text(encoding="utf-8")))

    @staticmethod
    def list_sessions(results_root: Path = RESULTS_ROOT) -> list[Path]:
        if not results_root.exists():
            return []
        return sorted((p for p in results_root.iterdir() if (p / "session.json").exists()), reverse=True)

    def save(self) -> None:
        (self.root / "session.json").write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def record_runtime(self, info: dict[str, Any]) -> None:
        self.data.setdefault("runtime_history", []).append({"at": datetime.now().isoformat(timespec="seconds"), **info})
        self.save()

    # ------------------------------------------------------------------ trials
    def trial_dirs(self, slot: Slot | None = None) -> list[Path]:
        if not self.trials_dir.exists():
            return []
        dirs = sorted(p for p in self.trials_dir.iterdir() if (p / "trial.json").exists())
        if slot is None:
            return dirs
        prefix = f"{slot.slot:02d}_{slot.task_id}_r{slot.rep}"
        return [p for p in dirs if p.name.startswith(prefix)]

    def slot_status(self, slot: Slot) -> str:
        """'pending', 'done', or 'void' (every attempt so far was voided)."""
        dirs = self.trial_dirs(slot)
        if not dirs:
            return "pending"
        for d in dirs:
            rec = json.loads((d / "trial.json").read_text(encoding="utf-8"))
            if not (rec.get("annotations") or {}).get("void"):
                return "done"
        return "void"

    def next_slot(self) -> Slot | None:
        for slot in self.schedule:
            if self.slot_status(slot) in ("pending", "void"):
                return slot
        return None

    def new_trial_dir(self, slot: Slot) -> Path:
        base = f"{slot.slot:02d}_{slot.task_id}_r{slot.rep}"
        attempts = len(self.trial_dirs(slot))
        name = base if attempts == 0 else f"{base}_attempt{attempts + 1}"
        path = self.trials_dir / name
        path.mkdir(parents=True, exist_ok=False)
        return path
