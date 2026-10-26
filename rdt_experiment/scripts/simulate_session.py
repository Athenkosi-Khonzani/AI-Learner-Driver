"""Run a whole session on the simulated rig without the window, then analyse it.

    python scripts/simulate_session.py                      # 8 tasks x 3 runs
    python scripts/simulate_session.py --reps 1 --tasks RDT-01,RDT-04 --verification off

Output goes to results/ as a dry_run session marked SIMULATED. It shows what
the tables, figures and videos look like; it is not experimental data.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rdt  # noqa: E402,F401
from rdt.analysis import analyse_session  # noqa: E402
from rdt.config import load_experiment_config  # noqa: E402
from rdt.runtime import ExperimentRuntime  # noqa: E402
from rdt.session import RESULTS_ROOT, Session  # noqa: E402
from rdt.sim.rig import SimulatedRig  # noqa: E402
from rdt.trial import TrialRunner  # noqa: E402


async def run(args) -> Path:
    cfg = load_experiment_config()
    ids = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    sess = Session.create("SIMULATED", "dry_run", cfg, results_root=Path(args.out), repetitions=args.reps,
                          task_ids=ids, verification_default=args.verification == "on", allow_unlocked=True)
    rig = SimulatedRig(seed=args.seed, settle_s=args.settle)
    uri = await rig.start()
    rt = ExperimentRuntime(cfg, sess.root, rig.parts())
    await rt.start(uri)
    runner = TrialRunner(rt, sess, cfg, log=print)
    for _ in range(80):
        if rt.rover.connected and rt.frame_buffer.age_seconds() is not None:
            break
        await asyncio.sleep(0.1)
    try:
        while (slot := sess.next_slot()) is not None:
            task = sess.tasks[slot.task_id]
            rig.place(task)
            await asyncio.sleep(1.0)
            start = await runner.capture_start(task)
            await runner.run_trial(
                slot, task, start,
                verification_enabled=args.verification == "on",
                battery_v=runner.latest_vbat,
                override_reason="" if start.placement.ok else "simulated placement outside limits",
            )
    finally:
        await rt.stop()
        await rig.stop()
    return sess.root


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tasks", default="")
    p.add_argument("--verification", choices=["on", "off"], default="on")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--settle", type=float, default=1.0)
    p.add_argument("--out", default=str(RESULTS_ROOT))
    args = p.parse_args()
    root = asyncio.run(run(args))
    out = analyse_session(root, load_experiment_config())
    print((out / "README.md").read_text(encoding="utf-8"))
    print(f"Session: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
