from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ai_learner_host.application.state_store import AppState
from ai_learner_host.domain.enums import MissionState
from ..widgets import CameraView, CommandQueue, LogList, MetricCard


class EngineerPage(QWidget):
    prompt_submitted = Signal(str)
    voice_requested = Signal()
    manual_drive_requested = Signal(str)
    confirm_before_send_changed = Signal(bool)

    def __init__(self, confirm_before_send: bool = True, parent=None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(1)
        grid.setColumnStretch(0, 16)
        grid.setColumnStretch(1, 10)
        grid.setColumnStretch(2, 8)

        # Left: camera + telemetry
        left = QFrame()
        left.setObjectName("panel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)
        camera_header = QHBoxLayout()
        title = QLabel("CEILING CAMERA")
        title.setObjectName("eyebrow")
        self.camera_metrics = QLabel("awaiting stream")
        self.camera_metrics.setObjectName("muted")
        camera_header.addWidget(title)
        camera_header.addStretch(1)
        camera_header.addWidget(self.camera_metrics)
        left_layout.addLayout(camera_header)
        self.camera_view = CameraView()
        self.camera_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.camera_view, 1)

        telemetry_row = QHBoxLayout()
        self.vbat = MetricCard("Battery voltage")
        self.soc = MetricCard("State of charge")
        self.ultrasonic = MetricCard("Ultrasonic distance")
        telemetry_row.addWidget(self.vbat)
        telemetry_row.addWidget(self.soc)
        telemetry_row.addWidget(self.ultrasonic)
        left_layout.addLayout(telemetry_row)
        grid.addWidget(left, 0, 0)

        # Center: queue + run log + prompt
        center = QFrame()
        center.setObjectName("panel")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(14, 14, 14, 12)
        center_layout.setSpacing(10)
        queue_top = QHBoxLayout()
        queue_top.addWidget(QLabel("Command queue"))
        queue_top.addStretch(1)
        self.queue_counter = QLabel("0 / 0")
        self.queue_counter.setObjectName("muted")
        queue_top.addWidget(self.queue_counter)
        center_layout.addLayout(queue_top)
        self.command_queue = CommandQueue()
        center_layout.addWidget(self.command_queue)
        center_layout.addWidget(QLabel("Run log"))
        self.log_list = LogList()
        center_layout.addWidget(self.log_list, 1)

        compose = QHBoxLayout()
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("New instruction…")
        self.prompt_edit.setFixedHeight(52)
        self.mic_button = QPushButton("Mic")
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("primary")
        compose.addWidget(self.prompt_edit, 1)
        compose.addWidget(self.mic_button)
        compose.addWidget(self.send_button)
        center_layout.addLayout(compose)
        self.send_button.clicked.connect(self._send)
        self.mic_button.clicked.connect(self.voice_requested)
        grid.addWidget(center, 0, 1)

        # Right: processed world + manual jog. No emergency STOP control.
        right = QFrame()
        right.setObjectName("panel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)
        processed_header = QHBoxLayout()
        processed_title = QLabel("PROCESSED · ANNOTATED WORLD")
        processed_title.setObjectName("eyebrow")
        self.processed_status = QLabel("waiting")
        self.processed_status.setObjectName("muted")
        processed_header.addWidget(processed_title)
        processed_header.addStretch(1)
        processed_header.addWidget(self.processed_status)
        right_layout.addLayout(processed_header)

        self.processed_view = CameraView("Waiting for calibrated perception…")
        self.processed_view.setMinimumHeight(260)
        self.processed_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        right_layout.addWidget(self.processed_view, 1)
        self.world_summary = QLabel("No processed world-state yet.")
        self.world_summary.setObjectName("muted")
        self.world_summary.setWordWrap(True)
        self.world_summary.setMaximumHeight(100)
        right_layout.addWidget(self.world_summary)

        right_layout.addWidget(QLabel("Manual drive"))
        manual = QGridLayout()
        self.forward = QPushButton("↑")
        self.left = QPushButton("←")
        self.right = QPushButton("→")
        self.back = QPushButton("↓")
        for button in (self.forward, self.left, self.right, self.back):
            button.setFixedHeight(44)
        manual.addWidget(self.forward, 0, 1)
        manual.addWidget(self.left, 1, 0)
        center_label = QLabel("JOG")
        center_label.setObjectName("eyebrow")
        center_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        manual.addWidget(center_label, 1, 1)
        manual.addWidget(self.right, 1, 2)
        manual.addWidget(self.back, 2, 1)
        right_layout.addLayout(manual)
        self.manual_hint = QLabel("Each arrow sends one bounded path ending in STOP.")
        self.manual_hint.setObjectName("muted")
        self.manual_hint.setWordWrap(True)
        right_layout.addWidget(self.manual_hint)
        self.confirm_toggle = QCheckBox("Confirm plan before sending")
        self.confirm_toggle.setChecked(confirm_before_send)
        self.confirm_toggle.toggled.connect(self.confirm_before_send_changed)
        right_layout.addWidget(self.confirm_toggle)
        right_layout.addStretch(1)

        self.forward.clicked.connect(lambda: self.manual_drive_requested.emit("FWD"))
        self.back.clicked.connect(lambda: self.manual_drive_requested.emit("BACK"))
        self.left.clicked.connect(lambda: self.manual_drive_requested.emit("LEFT"))
        self.right.clicked.connect(lambda: self.manual_drive_requested.emit("RIGHT"))
        grid.addWidget(right, 0, 2)

    def set_prompt_text(self, text: str) -> None:
        self.prompt_edit.setPlainText(text)
        self.prompt_edit.setFocus()

    def _send(self) -> None:
        prompt = self.prompt_edit.toPlainText().strip()
        if prompt:
            self.prompt_submitted.emit(prompt)
            self.prompt_edit.clear()

    def set_live_frame(self, frame) -> None:
        self.camera_view.set_frame(frame)

    def set_camera_metrics(self, fps: float, latency_ms: float) -> None:
        self.camera_metrics.setText(f"{fps:.0f} fps · {latency_ms:.0f} ms")

    def set_processed_world(self, frame, summary: str = "") -> None:
        """Display the latest annotated perception frame without waiting for a mission."""
        if frame is not None:
            self.processed_view.set_frame(frame)
        if summary:
            self.world_summary.setText(summary)
        self.processed_status.setText("LIVE")

    def set_processed_status(self, text: str) -> None:
        self.processed_status.setText(text)

    def update_state(self, state: AppState) -> None:
        sample = state.telemetry
        self.vbat.set_value("—" if sample.vbat is None else f"{sample.vbat:.2f} V")
        self.soc.set_value("—" if sample.soc is None else f"{sample.soc:.0f}%")
        self.ultrasonic.set_value("—" if sample.dist is None else f"{sample.dist:g}")
        self.command_queue.set_commands(state.current_plan, state.current_command_index)
        total = len(state.current_plan)
        now = min(max(state.current_command_index + 1, 0), total)
        self.queue_counter.setText(f"{now} / {total}")
        self.log_list.set_logs(state.run_log)

        active = state.mission_state not in {
            MissionState.IDLE,
            MissionState.COMPLETED,
            MissionState.CANCELLED,
            MissionState.FAILED,
            MissionState.NEEDS_ATTENTION,
        }
        for button in (self.forward, self.back, self.left, self.right):
            button.setEnabled(not active and not state.manual_busy and state.rover_connected)
        self.send_button.setEnabled(not active and state.llm_ready)
        self.send_button.setToolTip(state.llm_status_text)
        self.mic_button.setEnabled(not active and state.voice_ready and not state.voice_busy)
        self.mic_button.setText("Listening…" if state.voice_busy else "Mic")
        self.mic_button.setToolTip(state.voice_status_text)
