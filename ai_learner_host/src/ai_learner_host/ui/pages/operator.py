from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ai_learner_host.application.state_store import AppState
from ai_learner_host.domain.enums import MissionState
from ..widgets import CommandQueue


class OperatorPage(QWidget):
    prompt_submitted = Signal(str)
    voice_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("panel")
        sidebar.setFixedWidth(300)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 18, 12, 14)
        side.setSpacing(10)
        header = QHBoxLayout()
        title = QLabel("Recent prompts")
        title.setStyleSheet("font-weight:700;")
        self.recent_count = QLabel("0")
        self.recent_count.setObjectName("muted")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.recent_count)
        side.addLayout(header)
        self.recent = QListWidget()
        side.addWidget(self.recent, 1)
        session = QLabel("Run history is persisted locally")
        session.setObjectName("muted")
        side.addWidget(session)
        root.addWidget(sidebar)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(54, 44, 54, 34)
        body_layout.setSpacing(18)

        self.eyebrow = QLabel("READY")
        self.eyebrow.setObjectName("eyebrow")
        self.prompt_label = QLabel("Tell the rover what to do.")
        self.prompt_label.setWordWrap(True)
        self.prompt_label.setStyleSheet("font-size:25px; font-weight:600;")
        body_layout.addWidget(self.eyebrow)
        body_layout.addWidget(self.prompt_label)

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        top = QHBoxLayout()
        top.addWidget(QLabel("DRIVING PRIMITIVES"))
        top.addStretch(1)
        self.step_label = QLabel("—")
        self.step_label.setObjectName("muted")
        top.addWidget(self.step_label)
        card_layout.addLayout(top)
        self.command_queue = CommandQueue()
        card_layout.addWidget(self.command_queue)
        body_layout.addWidget(card, 1)

        input_card = QFrame()
        input_card.setObjectName("card")
        input_layout = QHBoxLayout(input_card)
        input_layout.setContentsMargins(15, 12, 12, 12)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Tell the rover what to do…")
        self.prompt_edit.setFixedHeight(64)
        input_layout.addWidget(self.prompt_edit, 1)
        self.mic_button = QPushButton("Mic")
        self.mic_button.setToolTip("Record a spoken rover instruction")
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("primary")
        input_layout.addWidget(self.mic_button)
        input_layout.addWidget(self.send_button)
        body_layout.addWidget(input_card)
        hint = QLabel("Mic records a short instruction, transcribes it, and places the text here for review before sending.")
        hint.setObjectName("muted")
        body_layout.addWidget(hint)

        self.send_button.clicked.connect(self._send)
        self.mic_button.clicked.connect(self.voice_requested)
        root.addWidget(body, 1)

    def set_prompt_text(self, text: str) -> None:
        self.prompt_edit.setPlainText(text)
        self.prompt_edit.setFocus()

    def _send(self) -> None:
        prompt = self.prompt_edit.toPlainText().strip()
        if prompt:
            self.prompt_submitted.emit(prompt)
            self.prompt_edit.clear()

    def update_state(self, state: AppState) -> None:
        state_name = state.mission_state.name.replace("_", " ")
        self.eyebrow.setText(state_name)
        if state.current_prompt:
            self.prompt_label.setText(state.current_prompt)
        else:
            self.prompt_label.setText("Tell the rover what to do.")

        self.command_queue.set_commands(state.current_plan, state.current_command_index)
        total = len(state.current_plan)
        if total:
            current = min(max(state.current_command_index + 1, 0), total)
            self.step_label.setText(f"step {current} of {total}")
        else:
            self.step_label.setText("—")

        self.recent.clear()
        for run in state.recent_runs:
            stamp = run.timestamp[11:16] if len(run.timestamp) >= 16 else run.timestamp
            item = QListWidgetItem(f"{run.status} · {stamp}\n{run.prompt}")
            self.recent.addItem(item)
        self.recent_count.setText(str(len(state.recent_runs)))

        active = state.mission_state not in {
            MissionState.IDLE,
            MissionState.COMPLETED,
            MissionState.CANCELLED,
            MissionState.FAILED,
            MissionState.NEEDS_ATTENTION,
        }
        self.send_button.setEnabled(not active and state.llm_ready)
        self.send_button.setToolTip(state.llm_status_text)
        self.mic_button.setEnabled(not active and state.voice_ready and not state.voice_busy)
        self.mic_button.setText("Listening…" if state.voice_busy else "Mic")
        self.mic_button.setToolTip(state.voice_status_text)
