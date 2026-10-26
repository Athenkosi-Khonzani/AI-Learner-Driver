from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ai_learner_host.application.state_store import AppState, AppStateStore
from ai_learner_host.camera.frame_buffer import FrameBuffer
from ai_learner_host.config import AppConfig
from ai_learner_host.domain.enums import MissionState
from ai_learner_host.perception.service import CalibrationService
from .dialogs import PlanApprovalDialog
from .pages.engineer import EngineerPage
from .pages.operator import OperatorPage
from .pages.settings import SettingsPage
from .theme import APP_QSS
from .widgets import StatusPill


class MainWindow(QMainWindow):
    prompt_submitted = Signal(str)
    voice_requested = Signal()
    manual_drive_requested = Signal(str)
    settings_apply_requested = Signal(dict)
    intrinsics_saved = Signal(object)
    calibration_saved = Signal(object)
    colour_calibration_saved = Signal(object)
    confirm_before_send_changed = Signal(bool)
    plan_approved = Signal()
    plan_rejected = Signal()

    def __init__(
        self,
        state_store: AppStateStore,
        config: AppConfig,
        frame_buffer: FrameBuffer,
        calibration_service: CalibrationService,
    ) -> None:
        super().__init__()
        self.state_store = state_store
        self.config = config
        self._approval_open = False
        self._approval_dialog = None
        self._message_box = None
        self._last_state_processed_frame_id: int | None = None
        self.setWindowTitle("AI Learner · Host")
        self.resize(1240, 820)
        self.setMinimumSize(1020, 700)
        self.setStyleSheet(APP_QSS)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(central)

        top = QFrame()
        top.setObjectName("topBar")
        top.setFixedHeight(50)
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(14, 0, 14, 0)
        top_layout.setSpacing(10)
        dots = QLabel("●  ●  ●")
        dots.setStyleSheet("color:#827e76; letter-spacing:2px;")
        top_layout.addWidget(dots)
        title = QLabel("AI Learner · Host")
        title.setObjectName("appTitle")
        top_layout.addWidget(title)
        top_layout.addSpacing(12)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: list[QPushButton] = []
        for i, name in enumerate(["Operator", "Engineer", "Settings"]):
            button = QPushButton(name)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, idx=i: self.pages.setCurrentIndex(idx))
            self.nav_group.addButton(button, i)
            self.nav_buttons.append(button)
            top_layout.addWidget(button)
        self.nav_buttons[0].setChecked(True)
        top_layout.addStretch(1)
        self.status_pill = StatusPill()
        top_layout.addWidget(self.status_pill)
        root.addWidget(top)

        self.pages = QStackedWidget()
        self.operator = OperatorPage()
        self.engineer = EngineerPage(config.confirm_before_send)
        self.settings = SettingsPage(config, frame_buffer, calibration_service)
        self.pages.addWidget(self.operator)
        self.pages.addWidget(self.engineer)
        self.pages.addWidget(self.settings)
        root.addWidget(self.pages, 1)

        self.operator.prompt_submitted.connect(self.prompt_submitted)
        self.engineer.prompt_submitted.connect(self.prompt_submitted)
        self.operator.voice_requested.connect(self.voice_requested)
        self.engineer.voice_requested.connect(self.voice_requested)
        self.engineer.manual_drive_requested.connect(self.manual_drive_requested)
        self.engineer.confirm_before_send_changed.connect(self.confirm_before_send_changed)
        self.settings.apply_requested.connect(self.settings_apply_requested)
        self.settings.intrinsics_saved.connect(self.intrinsics_saved)
        self.settings.calibration_saved.connect(self.calibration_saved)
        self.settings.colour_calibration_saved.connect(self.colour_calibration_saved)
        self.state_store.subscribe(self.apply_state)

    def apply_state(self, state: AppState) -> None:
        self.status_pill.set_connected(state.rover_connected, state.rover_status_text)
        self.operator.update_state(state)
        self.engineer.update_state(state)

        # Mission captures are still stored in AppState. Only forward a new
        # processed-frame object once, so unrelated telemetry/state updates do
        # not overwrite the continuously refreshed Engineer perception view.
        if state.processed_frame is not None:
            frame_id = id(state.processed_frame)
            if frame_id != self._last_state_processed_frame_id:
                self._last_state_processed_frame_id = frame_id
                self.engineer.set_processed_world(
                    state.processed_frame,
                    state.world_summary,
                )

        if state.pending_approval and not self._approval_open:
            self._approval_open = True
            self._approval_dialog = PlanApprovalDialog(state.current_plan, self)
            self._approval_dialog.finished.connect(self._approval_finished)
            self._approval_dialog.open()
        if state.last_error:
            # Do not repeatedly modal-spam the operator. The run log remains the main diagnostic surface.
            self.statusBar().showMessage(state.last_error, 7000)


    def _approval_finished(self, result: int) -> None:
        self._approval_open = False
        self._approval_dialog = None
        if result == 1:  # QDialog.Accepted
            self.plan_approved.emit()
        else:
            self.plan_rejected.emit()

    def set_live_frame(self, frame) -> None:
        self.engineer.set_live_frame(frame)

    def set_camera_metrics(self, fps: float, latency_ms: float) -> None:
        self.engineer.set_camera_metrics(fps, latency_ms)

    def set_processed_world(self, frame, summary: str = "") -> None:
        self.engineer.set_processed_world(frame, summary)

    def set_processed_status(self, text: str) -> None:
        self.engineer.set_processed_status(text)

    def set_voice_transcript(self, text: str) -> None:
        """Place a voice transcript in both compose boxes for operator review."""
        self.operator.set_prompt_text(text)
        self.engineer.set_prompt_text(text)
        self.statusBar().showMessage("Voice transcribed — review the instruction, then press Send", 7000)

    def show_voice_error(self, message: str) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Voice input")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(message)
        box.finished.connect(lambda _result: setattr(self, "_message_box", None))
        self._message_box = box
        box.open()
