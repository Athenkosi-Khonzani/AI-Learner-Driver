from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox
from qasync import QEventLoop

from ai_learner_host.application.connection_manager import ConnectionManager
from ai_learner_host.application.event_bus import EventBus
from ai_learner_host.application.mission_controller import MissionController
from ai_learner_host.application.state_store import AppStateStore
from ai_learner_host.camera.frame_buffer import FrameBuffer
from ai_learner_host.camera.worker import CameraWorker
from ai_learner_host.config import (
    CalibrationProfile,
    ColourCalibrationProfile,
    ConfigRepository,
)
from ai_learner_host.domain.enums import InputMode
from ai_learner_host.perception.service import CalibrationService, PerceptionService
from ai_learner_host.persistence.sqlite_repository import RunRepository
from ai_learner_host.planning.llm_providers import MultiProviderLLMAdapter, display_name
from ai_learner_host.planning.planner import PlannerService
from ai_learner_host.planning.prompt_builder import PromptBuilder
from ai_learner_host.planning.validator import PlanValidator
from ai_learner_host.planning.verification import VerificationService
from ai_learner_host.rover.execution_tracker import ExecutionTracker
from ai_learner_host.rover.gateway import RoverGateway
from ai_learner_host.ui.main_window import MainWindow
from ai_learner_host.voice.service import VoiceInputService


class AppRuntime:
    # The Engineer page gets a continuously refreshed annotated perception view
    # without running CV on the Qt GUI thread. capture_world_state() performs
    # the CPU-heavy OpenCV work via asyncio.to_thread().
    LIVE_PERCEPTION_INTERVAL_SECONDS = 0.25

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        data_dir = project_root / "data"
        self.config_repo = ConfigRepository(data_dir / "settings.json")
        self.config = self.config_repo.load()
        self.state_store = AppStateStore()
        self.event_bus = EventBus()
        self.connections = ConnectionManager()
        self.frame_buffer = FrameBuffer()
        self.camera_worker = CameraWorker(self.frame_buffer)
        self.calibration = CalibrationService()
        self.perception = PerceptionService(self.frame_buffer, self.config)
        self.prompt_builder = PromptBuilder(self.config)
        self.llm = MultiProviderLLMAdapter(self.config)
        self.planner = PlannerService(self.llm, self.prompt_builder)
        self.validator = PlanValidator(self.config)
        self.verifier = VerificationService(self.llm, self.prompt_builder)
        self.voice = VoiceInputService(self.config)
        self.repository = RunRepository(
            data_dir / "ai_learner_host.sqlite3", data_dir / "snapshots"
        )
        self.rover = RoverGateway(
            self.event_bus,
            reconnect_seconds=self.config.reconnect_seconds,
            auto_reconnect=self.config.auto_reconnect,
        )
        self.execution = ExecutionTracker()
        self.controller = MissionController(
            config=self.config,
            event_bus=self.event_bus,
            state_store=self.state_store,
            planner=self.planner,
            perception=self.perception,
            validator=self.validator,
            verifier=self.verifier,
            rover=self.rover,
            execution=self.execution,
            repository=self.repository,
        )
        self.connections.register(self._start_rover, self.rover.stop)
        self.connections.register(self._start_camera, self._stop_camera)
        self.window = MainWindow(
            self.state_store,
            self.config,
            self.frame_buffer,
            self.calibration,
        )
        self._live_perception_task: asyncio.Task | None = None
        self._last_live_perception_status = ""
        self._pending_voice_transcript: str | None = None
        self._wire_ui()
        self._wire_camera()

    def _wire_ui(self) -> None:
        self.window.prompt_submitted.connect(self._submit_prompt_from_ui)
        self.window.manual_drive_requested.connect(
            lambda op: self._spawn(self.controller.manual_drive(op))
        )
        self.window.plan_approved.connect(lambda: self._spawn(self.controller.approve_pending_plan()))
        self.window.plan_rejected.connect(lambda: self._spawn(self.controller.reject_pending_plan()))
        self.window.voice_requested.connect(lambda: self._spawn(self._handle_voice_input()))
        self.window.settings_apply_requested.connect(self.apply_settings)
        self.window.intrinsics_saved.connect(self.save_intrinsics)
        self.window.calibration_saved.connect(self.save_calibration)
        self.window.colour_calibration_saved.connect(self.save_colour_calibration)
        self.window.confirm_before_send_changed.connect(self.set_confirm_before_send)

    def _submit_prompt_from_ui(self, prompt: str) -> None:
        cleaned = prompt.strip()
        input_mode = InputMode.TEXT
        if self._pending_voice_transcript is not None and cleaned == self._pending_voice_transcript.strip():
            input_mode = InputMode.VOICE
        self._pending_voice_transcript = None
        self._spawn(self.controller.submit_prompt(cleaned, input_mode))

    async def _handle_voice_input(self) -> None:
        if self.state_store.state.voice_busy:
            return
        self.state_store.update(voice_busy=True, voice_status_text="LISTENING — speak now")
        self.state_store.append_log("voice capture started")
        try:
            text = await self.voice.transcribe_once()
        except Exception as exc:
            ready, ready_message = self.voice.readiness()
            self.state_store.update(
                voice_ready=ready,
                voice_busy=False,
                voice_status_text=ready_message,
            )
            self.state_store.append_log(f"voice transcription failed · {exc}")
            self.window.show_voice_error(str(exc))
            return

        self._pending_voice_transcript = text
        ready, ready_message = self.voice.readiness()
        self.state_store.update(
            voice_ready=ready,
            voice_busy=False,
            voice_status_text=ready_message,
        )
        self.state_store.append_log(f"voice transcribed · {text}")
        self.window.set_voice_transcript(text)

    def _refresh_ai_readiness(self) -> None:
        llm_ready, llm_message = self.llm.readiness()
        voice_ready, voice_message = self.voice.readiness()
        self.state_store.update(
            llm_ready=llm_ready,
            llm_status_text=llm_message,
            voice_ready=voice_ready,
            voice_status_text=voice_message,
        )

    def _wire_camera(self) -> None:
        self.camera_worker.frame_ready.connect(self.window.set_live_frame)
        self.camera_worker.metrics_changed.connect(self.window.set_camera_metrics)
        self.camera_worker.status_changed.connect(self._camera_status_changed)

    def _spawn(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.state_store.update(last_error=str(exc))
            self.state_store.append_log(f"ERROR  {exc}")

    async def start(self) -> None:
        self._refresh_ai_readiness()
        self.state_store.append_log(
            f"planner · {display_name(self.config.planner_provider, self.config.planner_model)}"
        )
        await self.controller.initialize()
        await self.connections.start_all()
        self._live_perception_task = asyncio.create_task(
            self._live_perception_loop(),
            name="live-engineer-perception",
        )
        issues = self.perception.readiness_issues()
        if issues:
            self.window.pages.setCurrentIndex(2)
            self.window.nav_buttons[2].setChecked(True)
            self.window.settings.nav.setCurrentRow(1)
            self.state_store.append_log(
                "perception setup required · " + ", ".join(issues)
            )

    async def stop(self) -> None:
        if self._live_perception_task is not None:
            self._live_perception_task.cancel()
            try:
                await self._live_perception_task
            except asyncio.CancelledError:
                pass
            self._live_perception_task = None
        await self.connections.stop_all()

    async def _live_perception_loop(self) -> None:
        """Continuously publish annotated CV output to the Engineer page.

        The camera capture itself remains on CameraWorker's QThread. OpenCV
        perception is executed by PerceptionService in a worker thread, and the
        coroutine resumes on the Qt/qasync event loop before touching widgets.
        """
        while True:
            try:
                issues = self.perception.readiness_issues()

                if issues:
                    status = "SETUP REQUIRED"
                    if status != self._last_live_perception_status:
                        self.window.set_processed_status(status)
                        self._last_live_perception_status = status
                    await asyncio.sleep(0.50)
                    continue

                if not self.state_store.state.camera_connected:
                    status = "CAMERA OFFLINE"
                    if status != self._last_live_perception_status:
                        self.window.set_processed_status(status)
                        self._last_live_perception_status = status
                    await asyncio.sleep(0.35)
                    continue

                world = await self.perception.capture_world_state()
                self.window.set_processed_world(
                    world.annotated_frame,
                    self._engineer_world_summary(world),
                )
                self._last_live_perception_status = "LIVE"

                await asyncio.sleep(self.LIVE_PERCEPTION_INTERVAL_SECONDS)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Camera reconnects/calibration changes are normal transient
                # states. Keep the live loop alive and report the condition in
                # the Engineer panel instead of spamming modal errors/logs.
                status = "WAITING FOR PERCEPTION"
                if status != self._last_live_perception_status:
                    self.window.set_processed_status(status)
                    self._last_live_perception_status = status
                await asyncio.sleep(0.50)

    @staticmethod
    def _engineer_world_summary(world) -> str:
        """Compact live summary for the Engineer panel beneath the image."""
        scene = world.scene_json or {}
        rover = scene.get("rover", {})
        pose = rover.get("pose")
        obstacles = scene.get("obstacles", [])

        if pose is None:
            rover_text = "Rover pose unavailable"
        else:
            rover_text = (
                f"Rover ({pose.get('x_mm', 0):.0f}, {pose.get('y_mm', 0):.0f}) mm · "
                f"{pose.get('heading_deg', 0):.1f}° · "
                f"confidence {float(rover.get('confidence', 0.0)):.2f}"
            )

        colours = [str(item.get("colour", "?")) for item in obstacles]
        blocks_text = ", ".join(colours) if colours else "none"
        return f"{rover_text}\nBlocks ({len(obstacles)}): {blocks_text}"

    async def _start_rover(self) -> None:
        await self.rover.start(self.config.rover_uri)

    async def _start_camera(self) -> None:
        self.camera_worker.configure(self.config.camera_url, self.config.reconnect_seconds)
        self.camera_worker.start_capture()

    async def _stop_camera(self) -> None:
        self.camera_worker.stop_capture()

    def apply_settings(self, changes: dict) -> None:
        for key, value in changes.items():
            setattr(self.config, key, value)
        self.config_repo.save(self.config)
        self._refresh_ai_readiness()

        self.rover.auto_reconnect = self.config.auto_reconnect
        self.rover.reconnect_seconds = self.config.reconnect_seconds

        # "Apply settings & reconnect" is deliberately explicit: it restarts both
        # external links even when the URI text did not change.
        self.camera_worker.stop_capture()
        self.camera_worker.configure(self.config.camera_url, self.config.reconnect_seconds)
        self.camera_worker.start_capture()
        self._spawn(self.rover.reconnect(self.config.rover_uri))

        self.window.engineer.confirm_toggle.blockSignals(True)
        self.window.engineer.confirm_toggle.setChecked(self.config.confirm_before_send)
        self.window.engineer.confirm_toggle.blockSignals(False)
        self.state_store.append_log(
            "settings saved · planner="
            + display_name(self.config.planner_provider, self.config.planner_model)
            + " · external links reconnecting"
        )
        self.window.statusBar().showMessage("Settings saved; reconnecting rover and camera", 3000)


    def set_confirm_before_send(self, enabled: bool) -> None:
        self.config.confirm_before_send = bool(enabled)
        self.config_repo.save(self.config)
        if hasattr(self.window.settings, "confirm"):
            self.window.settings.confirm.blockSignals(True)
            self.window.settings.confirm.setChecked(bool(enabled))
            self.window.settings.confirm.blockSignals(False)
        self.state_store.append_log(
            f"confirm-before-send {'enabled' if enabled else 'disabled'}"
        )

    def save_intrinsics(self, profile) -> None:
        # A different lens correction changes the pixel geometry used by every
        # downstream calibration. Deliberately invalidate those profiles.
        arena_width = self.config.calibration.arena_width_mm
        arena_height = self.config.calibration.arena_height_mm
        self.config.intrinsics = profile
        self.config.calibration = CalibrationProfile(
            arena_width_mm=arena_width,
            arena_height_mm=arena_height,
        )
        self.config.colour_calibration = ColourCalibrationProfile()
        self.perception.reset_pose_filter()
        self.config_repo.save(self.config)
        self.state_store.append_log("camera intrinsic calibration loaded · arena/colour calibration reset")

    def save_calibration(self, profile) -> None:
        self.config.calibration = profile
        self.config.colour_calibration = ColourCalibrationProfile()
        self.perception.reset_pose_filter()
        self.config_repo.save(self.config)
        self.state_store.append_log("arena homography calibration saved · colour calibration reset")

    def save_colour_calibration(self, profile) -> None:
        self.config.colour_calibration = profile
        self.config_repo.save(self.config)
        self.state_store.append_log("red/green/yellow/blue colour calibration saved")

    def _camera_status_changed(self, connected: bool, text: str) -> None:
        self.state_store.update(camera_connected=connected, camera_status_text=text)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("AI Learner Host")
    app.setOrganizationName("AI Learner Rover Driver")
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    root = Path(__file__).resolve().parents[2]
    runtime = AppRuntime(root)
    app.aboutToQuit.connect(loop.stop)
    runtime.window.show()

    with loop:
        loop.run_until_complete(runtime.start())
        loop.run_forever()
        loop.run_until_complete(runtime.stop())


if __name__ == "__main__":
    main()
