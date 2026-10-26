from __future__ import annotations

from typing import Any

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ai_learner_host.domain.models import DriveCommand
from .theme import ACCENT, DANGER, MUTED, SUCCESS


def cv_frame_to_pixmap(frame: Any) -> QPixmap:
    if frame is None:
        return QPixmap()
    if len(frame.shape) == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class StatusPill(QFrame):
    def __init__(self, text: str = "ROVER OFFLINE", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("statusPill")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(7)
        self.dot = QLabel("●")
        self.label = QLabel(text)
        self.label.setStyleSheet('background:transparent; border:none; font-family: Consolas, "Courier New", monospace; font-size: 10px; font-weight: 600;')
        layout.addWidget(self.dot)
        layout.addWidget(self.label)
        self.set_connected(False)

    def set_connected(self, connected: bool, text: str | None = None) -> None:
        if text is not None:
            self.label.setText(text)
        color = ACCENT if connected else MUTED
        self.dot.setStyleSheet(f"color:{color}; background:transparent;")
        self.setStyleSheet(
            f"QFrame#statusPill {{ background:#202833; border:1px solid {'#446887' if connected else '#34322e'}; border-radius:13px; }}"
        )


class MetricCard(QFrame):
    def __init__(self, label: str, initial: str = "—", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 11, 14, 11)
        layout.setSpacing(4)
        title = QLabel(label.upper())
        title.setObjectName("metricLabel")
        self.value = QLabel(initial)
        self.value.setObjectName("metricValue")
        layout.addWidget(title)
        layout.addWidget(self.value)

    def set_value(self, text: str) -> None:
        self.value.setText(text)


class CameraView(QFrame):
    def __init__(self, placeholder: str = "CEILING CAMERA · awaiting stream", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._pixmap = QPixmap()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.image = QLabel(placeholder)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(320, 220)
        self.image.setStyleSheet(
            "background:#22201b; color:#6f6c65; font-family:Consolas,monospace; border-radius:10px;"
        )
        layout.addWidget(self.image)

    def set_frame(self, frame: Any) -> None:
        pixmap = cv_frame_to_pixmap(frame)
        self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap.isNull():
            return
        self.image.setPixmap(
            self._pixmap.scaled(
                self.image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class CommandQueue(QListWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSpacing(2)
        self.setMinimumHeight(170)

    def set_commands(self, commands: list[DriveCommand], current_index: int) -> None:
        self.clear()
        for index, command in enumerate(commands):
            item = QListWidgetItem()
            if index < current_index:
                suffix = "   ✓ ack"
            elif index == current_index:
                suffix = "   • now"
            else:
                suffix = ""
            item.setText(f"{index + 1:02d}    {command.display()}{suffix}")
            if index == current_index:
                item.setForeground(QColor(ACCENT))
            elif index < current_index:
                item.setForeground(QColor(SUCCESS))
            self.addItem(item)


class LogList(QListWidget):
    def set_logs(self, lines: list[str]) -> None:
        if self.count() == len(lines) and lines:
            return
        self.clear()
        for line in lines:
            item = QListWidgetItem(line)
            item.setForeground(QColor("#8d8a83"))
            self.addItem(item)
        self.scrollToBottom()
