"""End-to-end trials on the simulated rig (real host stack, synthetic camera and rover)."""

import asyncio
import json

import cv2

from rdt.config import load_experiment_config
from rdt.runtime import ExperimentRuntime
from rdt.session import Session
from rdt.sim.rig import SimulatedRig
from rdt.trial import TrialRunner, annotate_trial


def _run(tmp_path, task_ids, verification):
    async def go():
        cfg = load_experiment_config()
        sess = Session.create("e2e", "dry_run", cfg, results_root=tmp_path, task_ids=task_ids, allow_unlocked=True)
        rig = SimulatedRig(seed=5, settle_s=0.5)
        uri = await rig.start()
        rt = ExperimentRuntime(cfg, sess.root, rig.parts())
        await rt.start(uri)
        runner = TrialRunner(rt, sess, cfg)
        for _ in range(60):
            if rt.rover.connected and rt.frame_buffer.age_seconds() is not None:
                break
            await asyncio.sleep(0.1)
        dirs = []
        try:
            while (slot := sess.next_slot()) is not None:
                task = sess.tasks[slot.task_id]
                rig.place(task)
                await asyncio.sleep(1.0)
                start = await runner.capture_start(task)
                assert start.placement.ok, start.placement.to_dict()
                dirs.append(await runner.run_trial(slot, task, start, verification_enabled=verification,
                                                   battery_v=runner.latest_vbat))
        finally:
            await rt.stop()
            await rig.stop()
        return sess, cfg, dirs

    return asyncio.run(go())


def test_trial_files_video_and_scores(tmp_path):
    sess, cfg, dirs = _run(tmp_path, ["RDT-01"], verification=True)
    d = dirs[0]
    for name in ("trial.json", "trail.csv", "events.jsonl", "video.mp4", "final.png", "title.png",
                 "W0_raw.jpg", "W0_annotated.jpg", "after_0_annotated.jpg", "planner_prompt.txt"):
        assert (d / name).exists(), name
    rec = json.loads((d / "trial.json").read_text())
    s = rec["scores"]
    assert rec["status"] == "complete"
    assert s["first_pass_success"] and s["replay_success"]
    assert s["e_pos_mm"] is not None and s["e_pos_mm"] < 60
    assert rec["worlds"]["W0"]["scene"]["rover"]["pose"] is not None
    assert len(rec["trail"]) > 10
    cap = cv2.VideoCapture(str(d / "video.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > fps * (3.0 + 4.0)  # title + final hold + driving
    # annotations re-score the trial
    task = sess.tasks["RDT-01"]
    rec2 = annotate_trial(d, task, cfg, {"contact_first_pass": True})
    assert not rec2["scores"]["first_pass_success"]


def test_verification_off_has_no_corrections(tmp_path):
    _, _, dirs = _run(tmp_path, ["RDT-02"], verification=False)
    rec = json.loads((dirs[0] / "trial.json").read_text())
    assert rec["meta"]["verification_enabled"] is False
    assert len([r for r in rec["rounds"] if r.get("t_path_sent")]) == 1
    assert rec["scores"]["closed_loop_success"] is None
    assert rec["rounds"][0]["verifier"] is not None  # verdict still logged


def test_safety_stop_is_budgeted(tmp_path):
    _, _, dirs = _run(tmp_path, ["RDT-05"], verification=False)
    rec = json.loads((dirs[0] / "trial.json").read_text())
    assert rec["status"] in ("round_limit", "complete")
    assert not rec["scores"]["first_pass_success"]
    assert rec["scores"]["attribution"]["first_pass"] == "planning"
