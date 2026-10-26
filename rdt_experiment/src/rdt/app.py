"""Entry point for the operator application (scripts/run_rdt.py)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdt.config import ExperimentConfig, load_experiment_config
from rdt.runtime import ExperimentRuntime
from rdt.session import RESULTS_ROOT, Session


@dataclass
class AppContext:
    cfg: ExperimentConfig
    runtime: ExperimentRuntime
    simulated: bool
    rig: Any = None
    camera_worker: Any = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rover Driving Test (Experiment 2) operator application")
    parser.add_argument("--simulate", action="store_true", help="rehearse with a simulated camera, rover and planner")
    parser.add_argument("--real-llm", action="store_true", help="with --simulate: use the configured real planner")
    parser.add_argument("--session", help="open an existing session folder")
    parser.add_argument("--config", help="experiment config (default config/rdt_config.json)")
    args = parser.parse_args(argv)

    from PySide6.QtWidgets import QApplication
    from qasync import QEventLoop

    from rdt.ui.main_window import RDTWindow

    app = QApplication(sys.argv if argv is None else ["rdt", *argv])
    app.setApplicationName("Rover Driving Test")
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    cfg = load_experiment_config(args.config)
    staging = RESULTS_ROOT / "_runtime"
    staging.mkdir(parents=True, exist_ok=True)

    rig = None
    if args.simulate:
        from rdt.sim.rig import SimulatedRig

        rig = SimulatedRig(use_real_llm=args.real_llm)
        runtime = ExperimentRuntime(cfg, staging, rig.parts())
    else:
        runtime = ExperimentRuntime(cfg, staging)
    ctx = AppContext(cfg=cfg, runtime=runtime, simulated=bool(args.simulate), rig=rig)
    window = RDTWindow(ctx)
    title = "Rover Driving Test - Experiment 2" + ("   [SIMULATED RIG - not experimental data]" if args.simulate else "")
    window.setWindowTitle(title)
    window.show()

    async def start() -> None:
        if rig is not None:
            uri = await rig.start()
            await runtime.start(uri)
        else:
            from ai_learner_host.camera.worker import CameraWorker

            worker = CameraWorker(runtime.frame_buffer)
            worker.configure(runtime.config.camera_url, runtime.config.reconnect_seconds)
            worker.start_capture()
            ctx.camera_worker = worker
            await runtime.start()
        for key, (old, new) in runtime.overrides_applied.items():
            window.write_log(f"config override {key}: {old} -> {new}")
        if args.session:
            window.attach_session(Session.open(args.session))
        else:
            window.write_log("Create a session (New session...) or open one to begin.")
        asyncio.ensure_future(window.live_loop())

    async def stop() -> None:
        await runtime.stop()
        if rig is not None:
            await rig.stop()
        if ctx.camera_worker is not None:
            ctx.camera_worker.stop_capture()

    app.aboutToQuit.connect(loop.stop)
    with loop:
        loop.run_until_complete(start())
        loop.run_forever()
        loop.run_until_complete(stop())
    return 0
