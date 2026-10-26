"""Replay past host missions in the simulator and report execution error.

Reads the host's SQLite database. For every plan that was followed by an
'after' world state, the plan is replayed from the world state it was planned
on (the 'before' state for the first plan, the previous 'after' state for
corrections) and compared with the measured result.

    python scripts/replay_host_runs.py [--db PATH] [--since 2026-09-30]

These runs pre-date Experiment 2, so they are a check of the replay pipeline,
not experimental data.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rdt  # noqa: E402,F401
from llm_test.scenario.scenario_schema import GoalSpec  # noqa: E402

from rdt._bootstrap import host_project_root  # noqa: E402
from rdt.replay import execution_error, predict_pose, replay  # noqa: E402
from rdt.worldstate import parse_scene  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(host_project_root() / "data" / "ai_learner_host.sqlite3"))
    parser.add_argument("--since", default="2026-09-30")
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    runs = con.execute(
        "SELECT run_id, prompt, status FROM mission_runs WHERE started_at >= ? ORDER BY started_at",
        (args.since,),
    ).fetchall()
    rows = []
    for run_id, prompt, status in runs:
        plans = con.execute(
            "SELECT revision, source, plan_json, created_at FROM plan_attempts WHERE run_id=? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        snaps = con.execute(
            "SELECT role, timestamp, scene_json FROM world_snapshots WHERE run_id=? ORDER BY timestamp",
            (run_id,),
        ).fetchall()
        start = next((json.loads(s[2]) for s in snaps if s[0] == "before"), None)
        if start is None:
            continue
        for revision, source, plan_json, created in plans:
            after = next((s for s in snaps if s[0] == "after" and s[1] > created), None)
            if after is None:
                continue
            w0 = parse_scene(start)
            w1 = parse_scene(json.loads(after[2]))
            if not (w0.has_pose and w1.has_pose):
                continue
            cmds = json.loads(plan_json)
            dummy_goal = GoalSpec(target_x_mm=w1.rover_x_mm, target_y_mm=w1.rover_y_mm)
            sim = replay(
                w0, cmds, dummy_goal, name=run_id, instruction=prompt,
                rover_length_mm=w0.rover_length_mm, rover_width_mm=w0.rover_width_mm,
            )
            pred, _ = predict_pose((w0.rover_x_mm, w0.rover_y_mm, w0.rover_heading_deg), cmds)
            e_pos, e_th = execution_error((w1.rover_x_mm, w1.rover_y_mm, w1.rover_heading_deg), pred)
            travel = sum(abs(float(c[1])) for c in cmds if c[0] in ('FWD','BACK'))
            rows.append((run_id, revision, source, len(cmds), travel, sum(abs(float(c[1])) for c in cmds if c[0]=='TURN'), e_pos, e_th,
                         sim.collision or sim.boundary_violation, prompt[:48]))
            start = json.loads(after[2])

    print(f"{'run':<13}{'rev':>4} {'source':<13}{'cmds':>5}{'travel':>8}{'rot':>7}{'e_pos':>8}{'e_th':>7}  sim_unsafe  prompt")
    for r in rows:
        print(f"{r[0]:<13}{r[1]:>4} {r[2]:<13}{r[3]:>5}{r[4]:>8.0f}{r[5]:>7.0f}{r[6]:>8.1f}{r[7]:>7.1f}  {str(r[8]):<10}  {r[9]}")
    if rows:
        import statistics as st

        e = [r[6] for r in rows]
        t = [r[7] for r in rows]
        print(f"\n{len(rows)} plans · e_pos median {st.median(e):.1f} mm (max {max(e):.1f}) · "
              f"e_theta median {st.median(t):.1f}° (max {max(t):.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
