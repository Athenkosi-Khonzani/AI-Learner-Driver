from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

from ai_learner_host.camera.frame_buffer import FrameBuffer
from ai_learner_host.config import (
    CalibrationProfile,
    CameraIntrinsicsProfile,
    ColourCalibrationProfile,
)
from ai_learner_host.domain.models import DriveCommand
from ai_learner_host.perception.service import CalibrationService, ColourCalibrationService
from .widgets import cv_frame_to_pixmap


class PlanApprovalDialog(QDialog):
    def __init__(self, commands: list[DriveCommand], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm rover plan")
        self.resize(520, 420)
        layout = QVBoxLayout(self)
        title = QLabel("Review the generated driving primitives")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        note = QLabel("The rover will receive this complete path only after confirmation.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        listing = QListWidget()
        for i, command in enumerate(commands):
            listing.addItem(f"{i + 1:02d}    {command.display()}")
        layout.addWidget(listing, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Send to rover")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ClickableCalibrationLabel(QLabel):
    image_clicked = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(700, 460)
        self._image_width = 0
        self._image_height = 0

    def set_image_size(self, width: int, height: int) -> None:
        self._image_width = width
        self._image_height = height

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image_width <= 0 or self._image_height <= 0:
            return
        scale = min(self.width() / self._image_width, self.height() / self._image_height)
        shown_w = self._image_width * scale
        shown_h = self._image_height * scale
        offset_x = (self.width() - shown_w) / 2.0
        offset_y = (self.height() - shown_h) / 2.0
        x = (event.position().x() - offset_x) / scale
        y = (event.position().y() - offset_y) / scale
        if 0 <= x < self._image_width and 0 <= y < self._image_height:
            self.image_clicked.emit(float(x), float(y))


class CalibrationDialog(QDialog):
    """Four-click arena homography calibration on an undistorted frame."""

    def __init__(
        self,
        frame_buffer: FrameBuffer,
        calibration_service: CalibrationService,
        intrinsics: CameraIntrinsicsProfile,
        existing: CalibrationProfile,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.frame_buffer = frame_buffer
        self.calibration_service = calibration_service
        self.intrinsics = intrinsics
        self.existing = existing
        self.profile: CalibrationProfile | None = None
        self.frame = None
        self.points: list[list[float]] = []
        self.setWindowTitle("Arena camera calibration")
        self.resize(920, 700)

        layout = QVBoxLayout(self)
        title = QLabel("Homography calibration")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        instructions = QLabel(
            "The frame is first corrected using the intrinsic camera calibration. "
            "Click arena corners in this order: top-left → top-right → bottom-right → bottom-left. "
            "World coordinates use the bottom-left arena corner as (0,0)."
        )
        instructions.setObjectName("muted")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Arena width (mm)"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(100, 10000)
        self.width_spin.setValue(existing.arena_width_mm)
        self.width_spin.setDecimals(0)
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("Arena height (mm)"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(100, 10000)
        self.height_spin.setValue(existing.arena_height_mm)
        self.height_spin.setDecimals(0)
        size_row.addWidget(self.height_spin)
        size_row.addStretch(1)
        self.capture_button = QPushButton("Capture current frame")
        self.capture_button.setObjectName("primary")
        size_row.addWidget(self.capture_button)
        layout.addLayout(size_row)

        self.image = ClickableCalibrationLabel()
        self.image.setStyleSheet(
            "background:#11110f; border:1px solid #34322e; border-radius:10px;"
        )
        layout.addWidget(self.image, 1)
        self.status = QLabel("No calibration frame captured yet.")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.capture_button.clicked.connect(self.capture_frame)
        self.image.image_clicked.connect(self._add_point)
        self.capture_frame()

    def capture_frame(self) -> None:
        raw = self.frame_buffer.latest(max_age_seconds=10.0)
        if raw is None:
            self.frame = None
            self.points = []
            self.image.clear()
            self.status.setText("No recent camera frame is available. Start the camera first.")
            self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
            return
        if not self.intrinsics.valid:
            self.frame = None
            self.status.setText("Load intrinsic camera calibration before arena calibration.")
            self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
            return
        try:
            self.frame = self.calibration_service.prepare_arena_frame(raw, self.intrinsics)
        except Exception as exc:
            self.frame = None
            self.status.setText(str(exc))
            return
        self.points = []
        self.image.set_image_size(self.frame.shape[1], self.frame.shape[0])
        self._redraw()
        self.status.setText("Click top-left corner (1 of 4).")

    def _add_point(self, x: float, y: float) -> None:
        if self.frame is None or len(self.points) >= 4:
            return
        self.points.append([x, y])
        self._redraw()
        labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
        if len(self.points) < 4:
            self.status.setText(
                f"Click {labels[len(self.points)]} corner ({len(self.points) + 1} of 4)."
            )
        else:
            self.status.setText("Four points selected. Review the order, then save calibration.")
            self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(True)

    def _redraw(self) -> None:
        if self.frame is None:
            return
        annotated = self.frame.copy()
        for i, (x, y) in enumerate(self.points, start=1):
            pt = (int(x), int(y))
            cv2.circle(annotated, pt, 8, (120, 190, 255), -1)
            cv2.putText(
                annotated,
                str(i),
                (pt[0] + 10, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (120, 190, 255),
                2,
                cv2.LINE_AA,
            )
        pixmap = cv_frame_to_pixmap(annotated)
        self.image.setPixmap(
            pixmap.scaled(
                self.image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._redraw()

    def _save(self) -> None:
        if self.frame is None or len(self.points) != 4:
            return
        self.profile = self.calibration_service.compute(
            self.points,
            self.width_spin.value(),
            self.height_spin.value(),
            self.frame.shape,
        )
        self.accept()


class ColourCalibrationDialog(QDialog):
    """Interactive red/green/yellow/blue HSV calibration on the bird's-eye frame."""

    MIN_SAMPLES = 5
    PATCH_RADIUS = 10
    BUTTON_BGR = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "blue": (255, 0, 0),
    }

    def __init__(
        self,
        frame_buffer: FrameBuffer,
        calibration_service: CalibrationService,
        colour_service: ColourCalibrationService,
        intrinsics: CameraIntrinsicsProfile,
        arena: CalibrationProfile,
        existing: ColourCalibrationProfile,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.frame_buffer = frame_buffer
        self.calibration_service = calibration_service
        self.colour_service = colour_service
        self.intrinsics = intrinsics
        self.arena = arena
        self.existing = existing
        self.profile: ColourCalibrationProfile | None = None
        self.frame: np.ndarray | None = None
        self.hsv: np.ndarray | None = None
        self.selected_colour = "red"
        self.sample_patches: dict[str, list[np.ndarray]] = {
            colour: [] for colour in self.BUTTON_BGR
        }
        self.sample_points: dict[str, list[tuple[int, int]]] = {
            colour: [] for colour in self.BUTTON_BGR
        }

        self.setWindowTitle("Colour calibration")
        self.resize(980, 760)
        layout = QVBoxLayout(self)

        title = QLabel("Colour calibration")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        instructions = QLabel(
            "Capture the rectified arena, select a colour, then click at least five places "
            "well inside that coloured block. Avoid the white border. Each click samples a "
            "21×21 pixel patch, so one bad pixel cannot dominate the calibration."
        )
        instructions.setObjectName("muted")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        controls = QHBoxLayout()
        self.colour_group = QButtonGroup(self)
        self.colour_group.setExclusive(True)
        self.colour_buttons: dict[str, QPushButton] = {}
        for colour in self.BUTTON_BGR:
            button = QPushButton(colour.upper())
            button.setCheckable(True)
            if colour == "red":
                button.setChecked(True)
            button.clicked.connect(lambda checked=False, c=colour: self._select_colour(c))
            self.colour_group.addButton(button)
            self.colour_buttons[colour] = button
            controls.addWidget(button)
        controls.addSpacing(14)
        self.undo_button = QPushButton("Undo sample")
        self.clear_button = QPushButton("Clear selected")
        self.capture_button = QPushButton("Capture current frame")
        self.capture_button.setObjectName("primary")
        controls.addWidget(self.undo_button)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        controls.addWidget(self.capture_button)
        layout.addLayout(controls)

        self.image = ClickableCalibrationLabel()
        self.image.setStyleSheet(
            "background:#11110f; border:1px solid #34322e; border-radius:10px;"
        )
        layout.addWidget(self.image, 1)

        self.status = QLabel()
        self.status.setObjectName("muted")
        layout.addWidget(self.status)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.image.image_clicked.connect(self._sample)
        self.undo_button.clicked.connect(self._undo)
        self.clear_button.clicked.connect(self._clear_selected)
        self.capture_button.clicked.connect(self.capture_frame)
        self.capture_frame()

    def _select_colour(self, colour: str) -> None:
        self.selected_colour = colour
        self._refresh_status()
        self._redraw()

    def capture_frame(self) -> None:
        raw = self.frame_buffer.latest(max_age_seconds=10.0)
        if raw is None:
            self.frame = None
            self.hsv = None
            self.image.clear()
            self.status.setText("No recent camera frame is available. Start the camera first.")
            return
        if not self.intrinsics.valid or not self.arena.valid:
            self.frame = None
            self.status.setText(
                "Intrinsic and arena homography calibration must be completed first."
            )
            return
        try:
            self.frame = self.calibration_service.warp_birdseye(
                raw, self.intrinsics, self.arena
            )
        except Exception as exc:
            self.frame = None
            self.hsv = None
            self.status.setText(str(exc))
            return
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.sample_patches = {colour: [] for colour in self.BUTTON_BGR}
        self.sample_points = {colour: [] for colour in self.BUTTON_BGR}
        self.image.set_image_size(self.frame.shape[1], self.frame.shape[0])
        self._redraw()
        self._refresh_status()

    def _sample(self, x: float, y: float) -> None:
        if self.hsv is None:
            return
        ix, iy = int(round(x)), int(round(y))
        patch = self.colour_service.patch_pixels(
            self.hsv, ix, iy, self.PATCH_RADIUS
        )
        self.sample_patches[self.selected_colour].append(patch)
        self.sample_points[self.selected_colour].append((ix, iy))
        self._redraw()
        self._refresh_status()

    def _undo(self) -> None:
        colour = self.selected_colour
        if self.sample_patches[colour]:
            self.sample_patches[colour].pop()
            self.sample_points[colour].pop()
        self._redraw()
        self._refresh_status()

    def _clear_selected(self) -> None:
        colour = self.selected_colour
        self.sample_patches[colour].clear()
        self.sample_points[colour].clear()
        self._redraw()
        self._refresh_status()

    def _refresh_status(self) -> None:
        counts = "   ".join(
            f"{c[0].upper()}:{len(self.sample_patches[c])}" for c in self.BUTTON_BGR
        )
        self.status.setText(
            f"Selected: {self.selected_colour.upper()}   ·   {counts}   ·   "
            f"Need ≥{self.MIN_SAMPLES} samples per colour."
        )
        ready = self.frame is not None and all(
            len(self.sample_patches[c]) >= self.MIN_SAMPLES for c in self.BUTTON_BGR
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(ready)

    def _redraw(self) -> None:
        if self.frame is None:
            return
        annotated = self.frame.copy()
        for colour, points in self.sample_points.items():
            bgr = self.BUTTON_BGR[colour]
            for index, (x, y) in enumerate(points, start=1):
                cv2.rectangle(
                    annotated,
                    (x - self.PATCH_RADIUS, y - self.PATCH_RADIUS),
                    (x + self.PATCH_RADIUS, y + self.PATCH_RADIUS),
                    bgr,
                    2,
                )
                cv2.putText(
                    annotated,
                    str(index),
                    (x + self.PATCH_RADIUS + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    bgr,
                    1,
                    cv2.LINE_AA,
                )
        pixmap = cv_frame_to_pixmap(annotated)
        self.image.setPixmap(
            pixmap.scaled(
                self.image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._redraw()

    def _save(self) -> None:
        try:
            self.profile = self.colour_service.compute(
                self.sample_patches,
                patch_radius_px=self.PATCH_RADIUS,
            )
        except Exception as exc:
            self.status.setText(f"Calibration failed: {exc}")
            return
        self.accept()
