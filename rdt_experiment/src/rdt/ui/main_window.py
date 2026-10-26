"""Operator window for the Rover Driving Test."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ai_learner_host.perception.service import PerceptionService

from rdt.config import ExperimentConfig
from rdt.session import MODES, Session, Slot
from rdt.tasks import RDTTask
from rdt.trial import StartCapture, TrialRunner, annotate_trial
from rdt.ui.overlay import draw_placement_guide
from rdt.video import Canvas
from rdt.worldstate import parse_scene


def to_pixmap(img: np.ndarray, width: int, height: int) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def open_folder(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


class NewSessionDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Rover Driving Test session")
        form = QFormLayout(self)
        self.name = QLineEdit("rdt")
        self.mode = QComboBox()
        self.mode.addItems(list(MODES))
        self.mode.setCurrentText("dry_run")
        self.tasks = QLineEdit()
        self.tasks.setPlaceholderText("all tasks (or e.g. RDT-01,RDT-05)")
        self.reps = QSpinBox()
        self.reps.setRange(0, 10)
        self.reps.setSpecialValueText("from config")
        self.verify = QCheckBox("Verification loop on by default")
        self.verify.setChecked(True)
        form.addRow("Name", self.name)
        form.addRow("Mode", self.mode)
        form.addRow("Tasks", self.tasks)
        form.addRow("Repetitions", self.reps)
        form.addRow("", self.verify)
        note = QLabel(
            "main: the 24 recorded trials (needs a locked task manifest).\n"
            "dry_run: rehearsal and tolerance setting; not analysed with the main data.\n"
            "subsystem: fixed plans without the LLM (T3 to T5)."
        )
        note.setStyleSheet("color: #888;")
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> dict[str, Any]:
        ids = [t.strip() for t in self.tasks.text().split(",") if t.strip()]
        return {
            "name": self.name.text().strip() or "rdt",
            "mode": self.mode.currentText(),
            "task_ids": ids or None,
            "repetitions": self.reps.value() or None,
            "verification_default": self.verify.isChecked(),
        }


class RDTWindow(QMainWindow):
    LIVE_INTERVAL_S = 0.3

    def __init__(self, app_ctx) -> None:
        super().__init__()
        self.ctx = app_ctx  # rdt.app.AppContext
        self.cfg: ExperimentConfig = app_ctx.cfg
        self.session: Session | None = None
        self.runner: TrialRunner | None = None
        self.slot: Slot | None = None
        self.task: RDTTask | None = None
        self.start_capture: StartCapture | None = None
        self.last_trial_dir: Path | None = None
        self.live_perception: PerceptionService | None = None
        self._recording_frame: np.ndarray | None = None
        self._live_task: asyncio.Task | None = None
        self.setWindowTitle("Rover Driving Test - Experiment 2")
        self.resize(1500, 920)
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start(500)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        self.view = QLabel("Waiting for camera...")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(760, 560)
        self.view.setStyleSheet("background:#111; color:#aaa;")
        lv.addWidget(self.view, 1)
        self.view_caption = QLabel("")
        self.view_caption.setStyleSheet("color:#888;")
        lv.addWidget(self.view_caption)
        splitter.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)

        status = QGroupBox("Rig")
        sg = QGridLayout(status)
        self.lbl_session = QLabel("No session")
        self.lbl_camera = QLabel("camera: -")
        self.lbl_rover = QLabel("rover: -")
        self.lbl_battery = QLabel("battery: -")
        self.lbl_planner = QLabel("planner: -")
        btn_new = QPushButton("New session...")
        btn_open = QPushButton("Open session...")
        btn_analyse = QPushButton("Analyse session")
        btn_new.clicked.connect(self._new_session)
        btn_open.clicked.connect(self._open_session)
        btn_analyse.clicked.connect(self._analyse)
        sg.addWidget(self.lbl_session, 0, 0, 1, 3)
        sg.addWidget(self.lbl_camera, 1, 0)
        sg.addWidget(self.lbl_rover, 1, 1)
        sg.addWidget(self.lbl_battery, 1, 2)
        sg.addWidget(self.lbl_planner, 2, 0, 1, 3)
        sg.addWidget(btn_new, 3, 0)
        sg.addWidget(btn_open, 3, 1)
        sg.addWidget(btn_analyse, 3, 2)
        rv.addWidget(status)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Slot", "Task", "Run", "Tape", "Status", "First pass"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._slot_selected)
        self.table.setMinimumHeight(150)
        self.table.verticalHeader().setVisible(False)
        rv.addWidget(self.table, 1)

        trial = QGroupBox("Trial")
        tv = QVBoxLayout(trial)
        self.lbl_task = QLabel("Select a slot")
        self.lbl_task.setStyleSheet("font-weight:600; font-size:14px;")
        self.lbl_instruction = QLabel("")
        self.lbl_instruction.setWordWrap(True)
        self.lbl_instruction.setStyleSheet("font-size:13px; padding:4px; background:#222; color:#eee;")
        self.lbl_notes = QLabel("")
        self.lbl_notes.setWordWrap(True)
        self.lbl_notes.setStyleSheet("color:#999;")
        tv.addWidget(self.lbl_task)
        tv.addWidget(self.lbl_instruction)
        tv.addWidget(self.lbl_notes)
        row = QHBoxLayout()
        self.chk_verify = QCheckBox("Verification loop (corrections)")
        self.chk_verify.setChecked(self.cfg.verification_default)
        self.spin_battery = QDoubleSpinBox()
        self.spin_battery.setRange(0.0, 30.0)
        self.spin_battery.setDecimals(2)
        self.spin_battery.setSuffix(" V")
        self.spin_battery.setSpecialValueText("battery: from telemetry")
        row.addWidget(self.chk_verify)
        row.addStretch(1)
        row.addWidget(QLabel("Manual battery"))
        row.addWidget(self.spin_battery)
        tv.addLayout(row)
        buttons = QHBoxLayout()
        self.btn_place = QPushButton("Place (simulated)")
        self.btn_place.setVisible(self.ctx.simulated)
        self.btn_place.clicked.connect(self._sim_place)
        self.btn_capture = QPushButton("1  Capture start state")
        self.btn_run = QPushButton("2  Run trial")
        self.btn_abort = QPushButton("Abort")
        self.btn_capture.clicked.connect(lambda: self._spawn(self._capture()))
        self.btn_run.clicked.connect(lambda: self._spawn(self._run()))
        self.btn_abort.clicked.connect(self._abort)
        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(False)
        self.btn_run.setStyleSheet("font-weight:600;")
        for b in (self.btn_place, self.btn_capture, self.btn_run, self.btn_abort):
            buttons.addWidget(b)
        tv.addLayout(buttons)
        self.lbl_placement = QLabel("")
        self.lbl_placement.setWordWrap(True)
        tv.addWidget(self.lbl_placement)
        self.override = QLineEdit()
        self.override.setPlaceholderText("Override reason (only if the placement check fails and you still want to run)")
        tv.addWidget(self.override)
        rv.addWidget(trial)

        result = QGroupBox("Result and observations")
        rf = QGridLayout(result)
        self.lbl_result = QLabel("No trial yet")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setTextFormat(Qt.RichText)
        self.lbl_result.setMinimumHeight(86)
        self.lbl_result.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.tape_x = QLineEdit()
        self.tape_y = QLineEdit()
        self.tape_x.setPlaceholderText("tape x (mm)")
        self.tape_y.setPlaceholderText("tape y (mm)")
        self.chk_contact0 = QCheckBox("Contact seen in first pass")
        self.chk_contact1 = QCheckBox("Contact seen in a correction")
        self.chk_void = QCheckBox("Void this trial (re-run the slot)")
        self.void_reason = QLineEdit()
        self.void_reason.setPlaceholderText("void reason")
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("notes")
        self.btn_save = QPushButton("Save observations and re-score")
        self.btn_save.clicked.connect(self._save_annotations)
        self.btn_folder = QPushButton("Open trial folder")
        self.btn_folder.clicked.connect(lambda: self.last_trial_dir and open_folder(self.last_trial_dir))
        self.btn_next = QPushButton("Next slot")
        self.btn_next.clicked.connect(self._select_next)
        rf.addWidget(self.lbl_result, 0, 0, 1, 4)
        rf.addWidget(QLabel("Final position by tape"), 1, 0)
        rf.addWidget(self.tape_x, 1, 1)
        rf.addWidget(self.tape_y, 1, 2)
        rf.addWidget(self.chk_contact0, 2, 0, 1, 2)
        rf.addWidget(self.chk_contact1, 2, 2, 1, 2)
        rf.addWidget(self.chk_void, 3, 0, 1, 2)
        rf.addWidget(self.void_reason, 3, 2, 1, 2)
        rf.addWidget(self.notes, 4, 0, 1, 4)
        rf.addWidget(self.btn_save, 5, 0, 1, 2)
        rf.addWidget(self.btn_folder, 5, 2)
        rf.addWidget(self.btn_next, 5, 3)
        rv.addWidget(result)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(130)
        rv.addWidget(self.log)
        splitter.addWidget(right)
        splitter.setSizes([860, 640])
        self.setCentralWidget(splitter)

    # ------------------------------------------------------------------ helpers
    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)

        def done(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                self.write_log(f"ERROR {type(exc).__name__}: {exc}")
                QMessageBox.warning(self, "Rover Driving Test", str(exc))
                self._set_busy(False)

        task.add_done_callback(done)

    def write_log(self, message: str) -> None:
        self.log.appendPlainText(f"{datetime.now().strftime('%H:%M:%S')}  {message}")

    def _set_busy(self, busy: bool) -> None:
        self.btn_capture.setEnabled(not busy)
        self.btn_place.setEnabled(not busy)
        self.btn_run.setEnabled((not busy) and self.start_capture is not None)
        self.btn_abort.setEnabled(busy)
        self.table.setEnabled(not busy)
        self.chk_verify.setEnabled(not busy and (self.task is None or self.task.fixed_plan is None))

    # ------------------------------------------------------------------ session
    def attach_session(self, session: Session) -> None:
        self.session = session
        if self.runner is not None:
            self.runner.session = session
        else:
            self.runner = TrialRunner(self.ctx.runtime, session, self.cfg, log=self.write_log)
        session.record_runtime(self.ctx.runtime.describe())
        self.chk_verify.setChecked(session.verification_default)
        locked = "locked" if session.data.get("manifest_locked") else "UNLOCKED"
        self.lbl_session.setText(f"<b>{session.name}</b>  [{session.mode}]  task manifest {locked}  <span style='color:#888'>{session.root}</span>")
        self._fill_table()
        self._select_next()

    def _new_session(self) -> None:
        dlg = NewSessionDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        try:
            session = Session.create(
                v["name"], v["mode"], self.cfg,
                repetitions=v["repetitions"], task_ids=v["task_ids"],
                verification_default=v["verification_default"],
                allow_unlocked=False,
            )
        except RuntimeError as exc:
            if v["mode"] == "main" and "locked" in str(exc):
                QMessageBox.warning(self, "Main session", str(exc))
                return
            raise
        self.write_log(f"session created: {session.root}")
        self.attach_session(session)

    def _open_session(self) -> None:
        from rdt.session import RESULTS_ROOT

        path = QFileDialog.getExistingDirectory(self, "Open session folder", str(RESULTS_ROOT))
        if path:
            self.attach_session(Session.open(path))

    def _analyse(self) -> None:
        if not self.session:
            return
        from rdt.analysis import analyse_session

        out = analyse_session(self.session.root, self.cfg)
        self.write_log(f"analysis written to {out}")
        open_folder(out)

    def _fill_table(self) -> None:
        if not self.session:
            return
        schedule = self.session.schedule
        self.table.blockSignals(True)
        self.table.setRowCount(len(schedule))
        for i, slot in enumerate(schedule):
            status = self.session.slot_status(slot)
            first = ""
            dirs = self.session.trial_dirs(slot)
            if dirs:
                rec = json.loads((dirs[-1] / "trial.json").read_text(encoding="utf-8"))
                fp = (rec.get("scores") or {}).get("first_pass_success")
                first = "yes" if fp else "no"
            values = [str(slot.slot), slot.task_id, str(slot.rep), "tape" if slot.tape_check else "", status, first]
            for j, v in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(v))
        self.table.blockSignals(False)

    def _select_next(self) -> None:
        if not self.session:
            return
        nxt = self.session.next_slot()
        if nxt is None:
            self.write_log("all slots in this session are done")
            return
        self.table.selectRow(nxt.slot - 1)

    def _slot_selected(self) -> None:
        if not self.session:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        slot = self.session.schedule[rows[0].row()]
        self.slot = slot
        self.task = self.session.tasks[slot.task_id]
        self.start_capture = None
        self.btn_run.setEnabled(False)
        self.lbl_placement.setText("")
        self.override.clear()
        g = self.task.goal
        self.lbl_task.setText(f"Slot {slot.slot}: {self.task.task_id} run {slot.rep} - {self.task.title}"
                              + ("   [tape check]" if slot.tape_check else ""))
        self.lbl_instruction.setText(f"Prompt:  {self.task.instruction}")
        self.lbl_notes.setText(self.task.notes)
        fixed = self.task.fixed_plan is not None
        self.chk_verify.setEnabled(not fixed)
        if fixed:
            self.chk_verify.setChecked(False)
        else:
            self.chk_verify.setChecked(self.session.verification_default)
        if self.session.mode == "main" and self.session.next_slot() and slot.slot != self.session.next_slot().slot:
            self.write_log(f"note: slot {slot.slot} is out of the scheduled order")

    # ------------------------------------------------------------------ live view
    async def live_loop(self) -> None:
        self.live_perception = PerceptionService(self.ctx.runtime.frame_buffer, self.ctx.runtime.config)
        while True:
            try:
                if self.runner and self.runner.active:
                    if self._recording_frame is not None:
                        self._show(self._recording_frame)
                    await asyncio.sleep(0.15)
                    continue
                packet = self.ctx.runtime.frame_buffer.latest_packet(max_age_seconds=3.0)
                if packet is None or self.live_perception.readiness_issues():
                    await asyncio.sleep(0.4)
                    continue
                world = await asyncio.to_thread(self.live_perception._process_world, packet.frame)
                mw = parse_scene(world.scene_json)
                canvas = Canvas(mw.arena_width_mm, mw.arena_height_mm, 1280)
                img = canvas.from_birdseye(world.annotated_frame)
                if self.task is not None:
                    report = self.task.placement_report(mw, self.cfg.placement_limits)
                    img = draw_placement_guide(img, canvas, self.task, self.cfg.rover_length_mm, self.cfg.rover_width_mm, report)
                self._show(img)
                self._recording_frame = None
                self.view_caption.setText("Live view with placement guide" if self.task else "Live view")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.view_caption.setText(f"live view: {exc}")
            await asyncio.sleep(self.LIVE_INTERVAL_S)

    def _show(self, img: np.ndarray) -> None:
        self.view.setPixmap(to_pixmap(img, self.view.width(), self.view.height()))

    def _on_recording_frame(self, frame: np.ndarray) -> None:
        self._recording_frame = frame

    # ------------------------------------------------------------------ status
    def _refresh_status(self) -> None:
        rt = self.ctx.runtime
        age = rt.frame_buffer.age_seconds()
        self.lbl_camera.setText("camera: " + ("live" if age is not None and age < 2 else "offline"))
        self.lbl_rover.setText("rover: " + ("connected" if rt.rover.connected else "offline"))
        v = self.runner.latest_vbat if self.runner else None
        self.lbl_battery.setText("battery: " + (f"{v:.2f} V" if v is not None else "no telemetry"))
        ok, msg = rt.llm_readiness()
        self.lbl_planner.setText(f"planner: {rt.planner_label()}  ({'ready' if ok else msg})")
        if self.runner and self.runner.active and self.runner.recorder is not None:
            st = rt.state_store.state.mission_state.name
            self.runner.recorder.status_text = st.replace("_", " ").lower()
            self.view_caption.setText(f"RECORDING  mission state: {st}")

    # ------------------------------------------------------------------ actions
    def _sim_place(self) -> None:
        if self.task is not None and self.ctx.rig is not None:
            self.ctx.rig.place(self.task)
            self.write_log(f"simulated placement for {self.task.task_id}")

    async def _capture(self) -> None:
        if not (self.session and self.task and self.runner):
            raise RuntimeError("Create or open a session and select a slot first")
        self._set_busy(True)
        self.btn_abort.setEnabled(False)
        self.write_log("capturing settled start state...")
        try:
            cap = await self.runner.capture_start(self.task)
        finally:
            self._set_busy(False)
        self.start_capture = cap
        lines = []
        for item in cap.placement.items:
            mark = "<span style='color:#3c3'>OK</span>" if item.ok else "<span style='color:#f55'>FIX</span>"
            lines.append(f"{mark} {item.name}: {item.text}")
        verdict = "Placement OK" if cap.placement.ok else "Placement outside the limits: adjust and capture again, or give an override reason"
        self.lbl_placement.setText(f"<b>{verdict}</b><br>" + "<br>".join(lines))
        self.btn_run.setEnabled(True)
        self.write_log(f"start captured: {verdict}")

    async def _run(self) -> None:
        if not (self.session and self.task and self.runner and self.slot and self.start_capture):
            return
        battery = self.spin_battery.value() or None
        source = "manual"
        if battery is None and self.runner.latest_vbat is not None:
            battery, source = self.runner.latest_vbat, "telemetry"
        problems = self.runner.preflight(battery)
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if self.clock_age_of_capture() > 120:
            raise RuntimeError("The start capture is more than 2 minutes old; capture it again")
        self._set_busy(True)
        self._recording_frame = None
        self.write_log(f"running {self.task.task_id} run {self.slot.rep} (verification {'on' if self.chk_verify.isChecked() else 'off'})")
        try:
            trial_dir = await self.runner.run_trial(
                self.slot, self.task, self.start_capture,
                verification_enabled=self.chk_verify.isChecked(),
                battery_v=battery, battery_source=source,
                override_reason=self.override.text(),
                on_frame=self._on_recording_frame,
            )
        finally:
            self._set_busy(False)
            self.start_capture = None
            self.btn_run.setEnabled(False)
        self.last_trial_dir = trial_dir
        self._show_result(trial_dir)
        self._fill_table()

    def clock_age_of_capture(self) -> float:
        import time

        return time.monotonic() - self.start_capture.captured_at if self.start_capture else 1e9

    def _abort(self) -> None:
        if self.runner and self.runner.active:
            self.runner.abort()
            self.write_log("abort requested")

    def _show_result(self, trial_dir: Path) -> None:
        rec = json.loads((trial_dir / "trial.json").read_text(encoding="utf-8"))
        s = rec.get("scores", {})

        def yn(v):
            if v is None:
                return "<span style='color:#888'>n/a</span>"
            return "<span style='color:#3c3'>SUCCESS</span>" if v else "<span style='color:#f55'>FAIL</span>"

        def f(v, nd=0):
            return "-" if v is None else f"{float(v):.{nd}f}"

        attr = (s.get("attribution") or {})
        self.lbl_result.setText(
            f"<b>{trial_dir.name}</b> ({rec.get('status')})<br>"
            f"First pass {yn(s.get('first_pass_success'))} goal error {f(s.get('first_pass_goal_error_mm'))} mm &nbsp; "
            f"Closed loop {yn(s.get('closed_loop_success'))} ({s.get('corrections_used', 0)} corrections) &nbsp; "
            f"Replay {yn(s.get('replay_success'))}<br>"
            f"e_pos {f(s.get('e_pos_mm'))} mm, e_theta {f(s.get('e_theta_deg'), 1)} deg &nbsp; min clearance {f(s.get('min_block_clearance_mm'))} mm "
            f"&nbsp; planning {f(s.get('planning_latency_s'), 1)} s<br>"
            f"Failure class: {attr.get('first_pass') or '-'} {('(' + attr.get('detail', '') + ')') if attr.get('first_pass') else ''}"
        )
        ann = rec.get("annotations") or {}
        self.tape_x.setText("" if ann.get("tape_x_mm") is None else str(ann["tape_x_mm"]))
        self.tape_y.setText("" if ann.get("tape_y_mm") is None else str(ann["tape_y_mm"]))
        self.chk_contact0.setChecked(bool(ann.get("contact_first_pass")))
        self.chk_contact1.setChecked(bool(ann.get("contact_correction")))
        self.chk_void.setChecked(bool(ann.get("void")))
        self.void_reason.setText(ann.get("void_reason") or "")
        self.notes.setText(ann.get("notes") or "")
        if rec["meta"].get("tape_check"):
            self.write_log("this slot needs a tape measurement of the rover centre")

    def _save_annotations(self) -> None:
        if not (self.last_trial_dir and self.session):
            return
        rec = json.loads((self.last_trial_dir / "trial.json").read_text(encoding="utf-8"))
        task = self.session.tasks[rec["meta"]["task_id"]]

        def num(edit: QLineEdit):
            text = edit.text().strip()
            return float(text) if text else None

        try:
            ann = {
                "tape_x_mm": num(self.tape_x),
                "tape_y_mm": num(self.tape_y),
                "contact_first_pass": self.chk_contact0.isChecked(),
                "contact_correction": self.chk_contact1.isChecked(),
                "void": self.chk_void.isChecked(),
                "void_reason": self.void_reason.text().strip() or None,
                "notes": self.notes.text().strip() or None,
            }
        except ValueError:
            QMessageBox.warning(self, "Observations", "Tape readings must be numbers in millimetres")
            return
        if ann["void"] and not ann["void_reason"]:
            QMessageBox.warning(self, "Observations", "Give a reason for voiding the trial")
            return
        annotate_trial(self.last_trial_dir, task, self.cfg, ann)
        self.write_log(f"observations saved and trial re-scored: {self.last_trial_dir.name}")
        self._show_result(self.last_trial_dir)
        self._fill_table()
