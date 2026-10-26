from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Iterable

import cv2

from ai_learner_host.domain.models import (
    MissionRun,
    RecentRunView,
    TelemetrySample,
    ValidatedPlan,
    VerificationResult,
    WorldStateSnapshot,
    utc_now_iso,
)


class RunRepository:
    def __init__(self, db_path: Path | None = None, snapshot_dir: Path | None = None) -> None:
        self.db_path = db_path or Path("data/ai_learner_host.sqlite3")
        self.snapshot_dir = snapshot_dir or Path("data/snapshots")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS mission_runs (
                    run_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    input_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT
                );
                CREATE TABLE IF NOT EXISTS plan_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    path_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS command_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    path_id INTEGER NOT NULL,
                    command_index INTEGER NOT NULL,
                    acked_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS telemetry_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    timestamp TEXT NOT NULL,
                    vbat REAL,
                    soc REAL,
                    cur REAL,
                    dist REAL
                );
                CREATE TABLE IF NOT EXISTS world_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,
                    raw_image_path TEXT,
                    annotated_image_path TEXT,
                    scene_json TEXT NOT NULL,
                    scene_markdown TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS verification_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    message TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                """
            )

            # Database compatibility: production telemetry uses `dist` for the
            # ultrasonic measurement. Keep the historical `cur` column so an
            # existing database from an older host build remains readable.
            columns = {row[1] for row in con.execute("PRAGMA table_info(telemetry_samples)")}
            if "cur" not in columns:
                con.execute("ALTER TABLE telemetry_samples ADD COLUMN cur REAL")
            if "dist" not in columns:
                con.execute("ALTER TABLE telemetry_samples ADD COLUMN dist REAL")

    async def create_run(self, run: MissionRun) -> None:
        await asyncio.to_thread(self._create_run_sync, run)

    def _create_run_sync(self, run: MissionRun) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO mission_runs(run_id,prompt,input_mode,status,started_at,ended_at) VALUES(?,?,?,?,?,?)",
                (run.run_id, run.prompt, run.input_mode.value, run.status.value, run.started_at, run.ended_at),
            )

    async def finish_run(self, run: MissionRun) -> None:
        await asyncio.to_thread(self._finish_run_sync, run)

    def _finish_run_sync(self, run: MissionRun) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE mission_runs SET status=?, ended_at=? WHERE run_id=?",
                (run.status.value, run.ended_at, run.run_id),
            )

    async def save_plan(self, run_id: str, plan: ValidatedPlan) -> None:
        await asyncio.to_thread(self._save_plan_sync, run_id, plan)

    def _save_plan_sync(self, run_id: str, plan: ValidatedPlan) -> None:
        payload = [[c.op] if c.value is None else [c.op, c.value] for c in plan.commands]
        with self._connect() as con:
            con.execute(
                "INSERT INTO plan_attempts(run_id,revision,path_id,source,plan_json,created_at) VALUES(?,?,?,?,?,?)",
                (run_id, plan.revision, plan.path_id, plan.source, json.dumps(payload), utc_now_iso()),
            )

    async def save_command_ack(self, run_id: str, path_id: int, index: int) -> None:
        await asyncio.to_thread(self._save_command_ack_sync, run_id, path_id, index)

    def _save_command_ack_sync(self, run_id: str, path_id: int, index: int) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO command_executions(run_id,path_id,command_index,acked_at) VALUES(?,?,?,?)",
                (run_id, path_id, index, utc_now_iso()),
            )

    async def save_telemetry(self, run_id: str | None, sample: TelemetrySample) -> None:
        await asyncio.to_thread(self._save_telemetry_sync, run_id, sample)

    def _save_telemetry_sync(self, run_id: str | None, sample: TelemetrySample) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO telemetry_samples(run_id,timestamp,vbat,soc,cur,dist) VALUES(?,?,?,?,?,?)",
                (run_id, sample.timestamp, sample.vbat, sample.soc, sample.cur, sample.dist),
            )

    async def save_snapshot(self, run_id: str, snapshot: WorldStateSnapshot, role: str) -> None:
        await asyncio.to_thread(self._save_snapshot_sync, run_id, snapshot, role)

    def _save_snapshot_sync(self, run_id: str, snapshot: WorldStateSnapshot, role: str) -> None:
        safe_stamp = snapshot.timestamp.replace(":", "-").replace("+", "_")
        raw_path = self.snapshot_dir / f"{run_id}_{safe_stamp}_{role}_raw.jpg"
        annotated_path = self.snapshot_dir / f"{run_id}_{safe_stamp}_{role}_annotated.jpg"
        text_map_path = self.snapshot_dir / f"{run_id}_{safe_stamp}_{role}_world_map.txt"
        raw_saved = cv2.imwrite(str(raw_path), snapshot.raw_frame) if snapshot.raw_frame is not None else False
        ann_saved = cv2.imwrite(str(annotated_path), snapshot.annotated_frame) if snapshot.annotated_frame is not None else False
        if snapshot.scene_text_map:
            text_map_path.write_text(snapshot.scene_text_map, encoding="utf-8")
        with self._connect() as con:
            con.execute(
                """INSERT OR REPLACE INTO world_snapshots
                (snapshot_id,run_id,timestamp,role,raw_image_path,annotated_image_path,scene_json,scene_markdown)
                VALUES(?,?,?,?,?,?,?,?)""",
                (
                    snapshot.snapshot_id,
                    run_id,
                    snapshot.timestamp,
                    role,
                    str(raw_path) if raw_saved else None,
                    str(annotated_path) if ann_saved else None,
                    json.dumps(snapshot.scene_json),
                    snapshot.scene_markdown,
                ),
            )

    async def save_verification(self, run_id: str, round_number: int, result: VerificationResult) -> None:
        await asyncio.to_thread(self._save_verification_sync, run_id, round_number, result)

    def _save_verification_sync(self, run_id: str, round_number: int, result: VerificationResult) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO verification_attempts(run_id,round,success,message,created_at) VALUES(?,?,?,?,?)",
                (run_id, round_number, int(result.success), result.message, utc_now_iso()),
            )

    async def log(self, run_id: str | None, level: str, component: str, message: str) -> None:
        await asyncio.to_thread(self._log_sync, run_id, level, component, message)

    def _log_sync(self, run_id: str | None, level: str, component: str, message: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO event_log(run_id,timestamp,level,component,message) VALUES(?,?,?,?,?)",
                (run_id, utc_now_iso(), level, component, message),
            )

    async def recent_runs(self, limit: int = 24) -> list[RecentRunView]:
        return await asyncio.to_thread(self._recent_runs_sync, limit)

    def _recent_runs_sync(self, limit: int) -> list[RecentRunView]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT run_id,prompt,status,input_mode,started_at FROM mission_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            RecentRunView(
                run_id=row["run_id"],
                prompt=row["prompt"],
                status=row["status"],
                input_mode=row["input_mode"],
                timestamp=row["started_at"],
            )
            for row in rows
        ]
