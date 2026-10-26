from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ai_learner_host.camera.frame_buffer import FrameBuffer
from ai_learner_host.config import (
    AppConfig,
    CalibrationProfile,
    CameraIntrinsicsProfile,
    ColourCalibrationProfile,
)
from ai_learner_host.perception.service import CalibrationService, ColourCalibrationService
from ai_learner_host.planning.llm_providers import MODEL_OPTIONS, PROVIDER_KEY_ENV
from ai_learner_host.voice.service import VoiceInputService
from ..dialogs import CalibrationDialog, ColourCalibrationDialog


class SettingsPage(QWidget):
    apply_requested = Signal(dict)
    intrinsics_saved = Signal(object)
    calibration_saved = Signal(object)
    colour_calibration_saved = Signal(object)

    def __init__(
        self,
        config: AppConfig,
        frame_buffer: FrameBuffer,
        calibration_service: CalibrationService,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.frame_buffer = frame_buffer
        self.calibration_service = calibration_service
        self.colour_service = ColourCalibrationService()
        self._calibration_dialog = None
        self._colour_dialog = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        side = QFrame()
        side.setObjectName("panel")
        side.setFixedWidth(230)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 22, 12, 12)
        label = QLabel("SETTINGS")
        label.setObjectName("eyebrow")
        side_layout.addWidget(label)
        self.nav = QListWidget()
        for name in [
            "Connection",
            "Camera",
            "Voice input",
            "Planner",
            "Safety limits",
            "History & logs",
        ]:
            self.nav.addItem(name)
        self.nav.setCurrentRow(0)
        side_layout.addWidget(self.nav, 1)
        version = QLabel("host v0.3.2\nheading handshake fix")
        version.setObjectName("muted")
        side_layout.addWidget(version)
        root.addWidget(side)

        container = QWidget()
        body = QVBoxLayout(container)
        body.setContentsMargins(34, 24, 34, 24)
        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        self.apply_button = QPushButton("Apply settings & reconnect")
        self.apply_button.setObjectName("primary")
        body.addWidget(self.apply_button, 0)
        root.addWidget(container, 1)

        self._build_connection()
        self._build_camera()
        self._build_voice()
        self._build_planner()
        self._build_safety()
        self._build_history()
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.apply_button.clicked.connect(self._emit_apply)

    def _page(self, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 4, 8, 18)
        layout.setSpacing(16)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        desc = QLabel(subtitle)
        desc.setObjectName("muted")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)
        return page, layout

    @staticmethod
    def _card() -> tuple[QFrame, QFormLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        form = QFormLayout(frame)
        form.setContentsMargins(18, 16, 18, 16)
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(14)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return frame, form

    def _build_connection(self) -> None:
        _, layout = self._page(
            "Connection",
            "How the host reaches the rover and ceiling camera over the lab network.",
        )
        card, form = self._card()
        self.rover_uri = QLineEdit(self.config.rover_uri)
        self.camera_url = QLineEdit(self.config.camera_url)
        self.auto_reconnect = QCheckBox("Retry automatically after a dropped link")
        self.auto_reconnect.setChecked(self.config.auto_reconnect)
        self.reconnect_seconds = QDoubleSpinBox()
        self.reconnect_seconds.setRange(0.5, 60.0)
        self.reconnect_seconds.setValue(self.config.reconnect_seconds)
        self.reconnect_seconds.setSuffix(" s")
        form.addRow("Rover WebSocket URI", self.rover_uri)
        form.addRow("Camera stream", self.camera_url)
        form.addRow("Auto reconnect", self.auto_reconnect)
        form.addRow("Retry interval", self.reconnect_seconds)
        layout.addWidget(card)
        layout.addStretch(1)

    def _build_camera(self) -> None:
        _, layout = self._page(
            "Camera",
            "Configure the complete perception chain: lens correction, arena homography and "
            "the red/green/yellow/blue HSV calibration used by the world-state generator.",
        )

        card, form = self._card()

        self.intrinsics_status = QLabel()
        self._refresh_intrinsics_status(self.config.intrinsics)
        self.intrinsics_button = QPushButton("Load calibration JSON")
        self.intrinsics_button.clicked.connect(self._load_intrinsics)

        intrinsics_row = QWidget()
        intrinsics_layout = QHBoxLayout(intrinsics_row)
        intrinsics_layout.setContentsMargins(0, 0, 0, 0)
        intrinsics_layout.addWidget(self.intrinsics_status, 1)
        intrinsics_layout.addWidget(self.intrinsics_button)
        form.addRow("Lens calibration", intrinsics_row)

        self.calibration_status = QLabel()
        self._refresh_calibration_status(self.config.calibration)
        self.calibrate_button = QPushButton("Click to calibrate")
        self.calibrate_button.setObjectName("primary")
        self.calibrate_button.clicked.connect(self._open_calibration)

        arena_row = QWidget()
        arena_layout = QHBoxLayout(arena_row)
        arena_layout.setContentsMargins(0, 0, 0, 0)
        arena_layout.addWidget(self.calibration_status, 1)
        arena_layout.addWidget(self.calibrate_button)
        form.addRow("Arena homography", arena_row)

        self.colour_status = QLabel()
        self._refresh_colour_status(self.config.colour_calibration)
        self.colour_button = QPushButton("Calibrate colours")
        self.colour_button.setObjectName("primary")
        self.colour_button.clicked.connect(self._open_colour_calibration)

        colour_row = QWidget()
        colour_layout = QHBoxLayout(colour_row)
        colour_layout.setContentsMargins(0, 0, 0, 0)
        colour_layout.addWidget(self.colour_status, 1)
        colour_layout.addWidget(self.colour_button)
        form.addRow("Colour calibration", colour_row)

        rover = QLineEdit(
            f"{self.config.rover_marker_dictionary} · ID {self.config.rover_marker_id} · "
            f"centre offset {self.config.rover_centre_left_mm:.0f} mm left / "
            f"{self.config.rover_centre_forward_mm:.0f} mm forward"
        )
        rover.setReadOnly(True)
        form.addRow("Rover localisation", rover)

        note = QLabel(
            "Calibration order: 1) load camera intrinsics, 2) arena homography, "
            "3) colour calibration. Loading new intrinsics invalidates the other two because "
            "their pixels belong to the corrected camera image."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)

        layout.addWidget(card)
        layout.addWidget(note)
        layout.addStretch(1)

    def _build_voice(self) -> None:
        _, layout = self._page(
            "Voice input",
            "The Mic buttons record a short instruction until silence, then OpenAI gpt-transcribe converts it to text for review before sending.",
        )
        card, form = self._card()
        self.microphone = QComboBox()
        values = ["Default system microphone"]
        for name in VoiceInputService.list_input_devices():
            if name not in values:
                values.append(name)
        if self.config.microphone_name not in values:
            values.insert(0, self.config.microphone_name)
        self.microphone.addItems(values)
        index = self.microphone.findText(self.config.microphone_name)
        if index >= 0:
            self.microphone.setCurrentIndex(index)
        self.push_key = QLineEdit(self.config.push_to_talk_key)
        self.push_key.setMaximumWidth(120)
        api_status = QLabel(
            "OPENAI_API_KEY detected" if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY missing"
        )
        api_status.setObjectName("muted")
        form.addRow("Microphone", self.microphone)
        form.addRow("Voice shortcut label", self.push_key)
        form.addRow("Transcription", QLabel(self.config.voice_transcription_model))
        form.addRow("API status", api_status)
        layout.addWidget(card)
        layout.addStretch(1)

    def _build_planner(self) -> None:
        _, layout = self._page(
            "Planner",
            "Choose the strong-tier planning model. GPT-6 Sol is the production default; all three use the same structured command schema and host-side validator.",
        )
        card, form = self._card()
        self.model_selector = QComboBox()
        for label, provider, model in MODEL_OPTIONS:
            self.model_selector.addItem(label, (provider, model))
        selected = 0
        for i in range(self.model_selector.count()):
            provider, model = self.model_selector.itemData(i)
            if provider == self.config.planner_provider and model == self.config.planner_model:
                selected = i
                break
        self.model_selector.setCurrentIndex(selected)
        self.planner_api_status = QLabel()
        self.planner_api_status.setObjectName("muted")
        self.model_selector.currentIndexChanged.connect(self._refresh_planner_api_status)
        self._refresh_planner_api_status()
        form.addRow("Planning model", self.model_selector)
        form.addRow("API key", self.planner_api_status)
        footprint = QLabel(
            f"{self.config.rover_length_mm:.0f} mm long × {self.config.rover_width_mm:.0f} mm wide"
        )
        form.addRow("Rover footprint", footprint)
        layout.addWidget(card)
        layout.addStretch(1)

    def _refresh_planner_api_status(self) -> None:
        data = self.model_selector.currentData()
        if not data:
            self.planner_api_status.setText("No model selected")
            return
        provider, _model = data
        env_name = PROVIDER_KEY_ENV.get(provider, "")
        self.planner_api_status.setText(
            f"{env_name} detected" if env_name and os.getenv(env_name) else f"{env_name or 'API key'} missing"
        )

    def _build_safety(self) -> None:
        _, layout = self._page(
            "Safety limits",
            "Host-side plan limits and operator approval policy. Rover-side ultrasonic safety remains independent.",
        )
        card, form = self._card()
        self.clearance = QSpinBox()
        self.clearance.setRange(1, 100)
        self.clearance.setValue(self.config.planning_clearance_cm)
        self.clearance.setSuffix(" cm")
        self.max_primitives = QSpinBox()
        self.max_primitives.setRange(2, 100)
        self.max_primitives.setValue(self.config.max_primitives)
        self.max_forward = QSpinBox()
        self.max_forward.setRange(10, 10000)
        self.max_forward.setValue(self.config.max_forward_mm)
        self.max_forward.setSuffix(" mm")
        self.max_backward = QSpinBox()
        self.max_backward.setRange(10, 10000)
        self.max_backward.setValue(self.config.max_backward_mm)
        self.max_backward.setSuffix(" mm")
        self.max_turn = QSpinBox()
        self.max_turn.setRange(1, 360)
        self.max_turn.setValue(self.config.max_turn_deg)
        self.max_turn.setSuffix("°")
        self.confirm = QCheckBox("Review the primitive list before rover motion")
        self.confirm.setChecked(self.config.confirm_before_send)
        self.telemetry_on_start = QCheckBox("Send telemetry ON immediately after rover connection")
        self.telemetry_on_start.setChecked(self.config.telemetry_on_start)
        self.manual_jog = QSpinBox()
        self.manual_jog.setRange(10, 1000)
        self.manual_jog.setValue(self.config.manual_jog_mm)
        self.manual_jog.setSuffix(" mm")
        self.manual_turn = QSpinBox()
        self.manual_turn.setRange(1, 90)
        self.manual_turn.setValue(self.config.manual_turn_deg)
        self.manual_turn.setSuffix("°")
        form.addRow("Planning clearance", self.clearance)
        form.addRow("Max primitives", self.max_primitives)
        form.addRow("Max forward", self.max_forward)
        form.addRow("Max backward", self.max_backward)
        form.addRow("Max turn", self.max_turn)
        form.addRow("Confirm before send", self.confirm)
        form.addRow("Telemetry on connect", self.telemetry_on_start)
        form.addRow("Manual jog distance", self.manual_jog)
        form.addRow("Manual turn", self.manual_turn)
        layout.addWidget(card)
        layout.addStretch(1)

    def _build_history(self) -> None:
        _, layout = self._page(
            "History & logs",
            "Mission runs, path attempts, ACKs, telemetry, world-state images, JSON and text maps are stored locally.",
        )
        card, form = self._card()
        db = QLineEdit("data/ai_learner_host.sqlite3")
        db.setReadOnly(True)
        snapshots = QLineEdit("data/snapshots/")
        snapshots.setReadOnly(True)
        form.addRow("Database", db)
        form.addRow("Snapshots", snapshots)
        layout.addWidget(card)
        layout.addStretch(1)

    def _emit_apply(self) -> None:
        provider, model = self.model_selector.currentData()
        self.apply_requested.emit(
            {
                "rover_uri": self.rover_uri.text().strip(),
                "camera_url": self.camera_url.text().strip(),
                "auto_reconnect": self.auto_reconnect.isChecked(),
                "reconnect_seconds": self.reconnect_seconds.value(),
                "microphone_name": self.microphone.currentText(),
                "push_to_talk_key": self.push_key.text().strip() or "Space",
                "planner_provider": provider,
                "planner_model": model,
                "confirm_before_send": self.confirm.isChecked(),
                "telemetry_on_start": self.telemetry_on_start.isChecked(),
                "max_primitives": self.max_primitives.value(),
                "max_forward_mm": self.max_forward.value(),
                "max_backward_mm": self.max_backward.value(),
                "max_turn_deg": self.max_turn.value(),
                "planning_clearance_cm": self.clearance.value(),
                "manual_jog_mm": self.manual_jog.value(),
                "manual_turn_deg": self.manual_turn.value(),
            }
        )

    def _load_intrinsics(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load intrinsic camera calibration",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            profile = self.calibration_service.load_intrinsics_file(path)
        except Exception as exc:
            self.intrinsics_status.setText(f"Load failed · {exc}")
            return

        # A new undistortion mapping changes the pixel coordinates used by both
        # arena and colour calibration, so intentionally invalidate both.
        self.config.intrinsics = profile
        self.config.calibration = CalibrationProfile(
            arena_width_mm=self.config.calibration.arena_width_mm,
            arena_height_mm=self.config.calibration.arena_height_mm,
        )
        self.config.colour_calibration = ColourCalibrationProfile()
        self._refresh_intrinsics_status(profile)
        self._refresh_calibration_status(self.config.calibration)
        self._refresh_colour_status(self.config.colour_calibration)
        self.intrinsics_saved.emit(profile)

    def _open_calibration(self) -> None:
        if self._calibration_dialog is not None:
            self._calibration_dialog.raise_()
            return
        self._calibration_dialog = CalibrationDialog(
            self.frame_buffer,
            self.calibration_service,
            self.config.intrinsics,
            self.config.calibration,
            self,
        )
        self._calibration_dialog.finished.connect(self._calibration_finished)
        self._calibration_dialog.open()

    def _calibration_finished(self, result: int) -> None:
        dialog = self._calibration_dialog
        self._calibration_dialog = None
        if result == 1 and dialog is not None and dialog.profile is not None:
            self.config.calibration = dialog.profile
            # Homography changed, so colour calibration should be redone on the
            # new rectified image.
            self.config.colour_calibration = ColourCalibrationProfile()
            self._refresh_calibration_status(dialog.profile)
            self._refresh_colour_status(self.config.colour_calibration)
            self.calibration_saved.emit(dialog.profile)

    def _open_colour_calibration(self) -> None:
        if self._colour_dialog is not None:
            self._colour_dialog.raise_()
            return
        self._colour_dialog = ColourCalibrationDialog(
            self.frame_buffer,
            self.calibration_service,
            self.colour_service,
            self.config.intrinsics,
            self.config.calibration,
            self.config.colour_calibration,
            self,
        )
        self._colour_dialog.finished.connect(self._colour_calibration_finished)
        self._colour_dialog.open()

    def _colour_calibration_finished(self, result: int) -> None:
        dialog = self._colour_dialog
        self._colour_dialog = None
        if result == 1 and dialog is not None and dialog.profile is not None:
            self.config.colour_calibration = dialog.profile
            self._refresh_colour_status(dialog.profile)
            self.colour_calibration_saved.emit(dialog.profile)

    def _refresh_intrinsics_status(self, profile: CameraIntrinsicsProfile) -> None:
        if profile.valid:
            rms = (
                f" · RMS {profile.rms_reprojection_error:.3f}px"
                if profile.rms_reprojection_error is not None
                else ""
            )
            self.intrinsics_status.setText(
                f"Loaded · {profile.image_width}×{profile.image_height}{rms}"
            )
        else:
            self.intrinsics_status.setText("Not loaded")

    def _refresh_calibration_status(self, profile: CalibrationProfile) -> None:
        if profile.valid:
            self.calibration_status.setText(
                f"Calibrated · {profile.arena_width_mm:.0f}×"
                f"{profile.arena_height_mm:.0f} mm"
            )
        else:
            self.calibration_status.setText("Not calibrated")

    def _refresh_colour_status(self, profile: ColourCalibrationProfile) -> None:
        if profile.valid:
            counts = ", ".join(
                f"{colour[0].upper()}:{profile.colours[colour].get('samples', '✓')}"
                for colour in ("red", "green", "yellow", "blue")
            )
            self.colour_status.setText(f"Calibrated · {counts}")
        else:
            self.colour_status.setText("Not calibrated")
