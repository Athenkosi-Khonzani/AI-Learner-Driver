import math
from pathlib import Path

import pytest

from rdt.analysis import mcnemar_exact, wilson
from rdt.config import load_experiment_config
from rdt.replay import execution_error, predict_pose, replay
from rdt.scoring import score_trial
from rdt.session import Session
from rdt.tasks import RDTTask, load_task_set
from rdt.worldstate import (
    host_orientation_to_sim, parse_scene, polygon_distance, rover_polygon, sim_orientation_to_host,
)
from helpers import record, scene

ROOT = Path(__file__).resolve().parents[1]
CFG = load_experiment_config()


def task(tid):
    return RDTTask.load(ROOT / "tasks" / f"{tid}.json")


def test_orientation_conventions_round_trip():
    for a in (0, 30, 90, 135, 179):
        assert sim_orientation_to_host(host_orientation_to_sim(a)) == pytest.approx(a % 180)
    assert host_orientation_to_sim(0) == pytest.approx(90)   # long side along +X
    assert host_orientation_to_sim(90) == pytest.approx(0)   # long side along +Y


def test_parse_scene_block_polygon_matches_host_polygon():
    w = parse_scene(scene(400, 250, 0, [("red", 855, 620, 280, 200, 90)]))
    b = w.blocks[0]
    from rdt.worldstate import block_polygon
    ours = sorted((round(x), round(y)) for x, y in block_polygon(b))
    host = sorted((round(x), round(y)) for x, y in b.polygon)
    assert ours == host


def test_polygon_distance():
    a = rover_polygon(0, 0, 0, 100, 100)
    b = rover_polygon(300, 0, 0, 100, 100)
    assert polygon_distance(a, b) == pytest.approx(200)
    assert polygon_distance(a, rover_polygon(50, 0, 0, 100, 100)) == 0.0


def test_all_tasks_feasible_with_final_rover():
    tasks, _ = load_task_set(ROOT / "tasks")
    assert len(tasks) == 8
    for t in tasks:
        assert t.feasibility(CFG.rover_length_mm, CFG.rover_width_mm).feasible, t.task_id


def test_block_offset_goal_follows_measured_block():
    t = task("RDT-04")
    nominal = t.resolve_goal(None)
    assert (nominal.target_x_mm, nominal.target_y_mm) == (1100, 450)
    moved = parse_scene(scene(350, 300, 0, [("blue", 1120, 790, 280, 200, 0)]))
    g = t.resolve_goal(moved)
    assert (g.target_x_mm, g.target_y_mm) == (1120, 440)


def test_placement_report():
    t = task("RDT-05")
    good = parse_scene(scene(305, 622, 91, [("red", 860, 615, 280, 200, 90)]))
    assert t.placement_report(good, CFG.placement_limits).ok
    bad = parse_scene(scene(360, 622, 91, [("red", 860, 615, 280, 200, 90)]))
    assert not t.placement_report(bad, CFG.placement_limits).ok
    extra = parse_scene(scene(305, 622, 91, [("red", 860, 615, 280, 200, 90), ("blue", 200, 1000, 280, 200, 0)]))
    assert not t.placement_report(extra, CFG.placement_limits).ok


def test_predict_pose_and_execution_error():
    end, _ = predict_pose((100, 100, 0), [["TURN", 90], ["FWD", 500], ["STOP"]])
    assert end == pytest.approx((600, 100, 90))
    e_pos, e_th = execution_error((610, 95, 92), end)
    assert e_pos == pytest.approx(math.hypot(10, 5))
    assert e_th == pytest.approx(2)


def test_replay_collides_with_block():
    t = task("RDT-05")
    w0 = parse_scene(scene(300, 620, 90, [("red", 855, 620, 280, 200, 90)]))
    res = replay(w0, [["FWD", 1100], ["STOP"]], t.resolve_goal(w0), name="x", instruction="",
                 rover_length_mm=155.5, rover_width_mm=121.3)
    assert res.collision and not res.success


def test_score_success_and_execution_class():
    t = task("RDT-01")
    w0 = scene(400, 250, 0)
    ok = score_trial(record(t, w0, scene(402, 848, 1), [["FWD", 600], ["STOP"]]), t, CFG)
    assert ok["first_pass_success"] and ok["replay_success"]
    assert ok["attribution"]["first_pass"] is None
    miss = score_trial(record(t, w0, scene(400, 780, 0), [["FWD", 600], ["STOP"]]), t, CFG)
    assert not miss["first_pass_success"] and miss["replay_success"]
    assert miss["attribution"]["first_pass"] == "execution"
    assert miss["e_pos_mm"] == pytest.approx(70)


def test_score_planning_protocol_perception_classes():
    t = task("RDT-01")
    w0 = scene(400, 250, 0)
    wrong_plan = score_trial(record(t, w0, scene(400, 550, 0), [["FWD", 300], ["STOP"]]), t, CFG)
    assert wrong_plan["attribution"]["first_pass"] == "planning"
    nack = score_trial(record(t, w0, scene(400, 500, 0), [["FWD", 600], ["STOP"]],
                              nacks=[{"i": 0, "reason": "timeout"}], path_ack=False), t, CFG)
    assert nack["attribution"]["first_pass"] == "protocol"
    assert nack["e_pos_mm"] is None
    tape = score_trial(record(t, w0, scene(400, 780, 0), [["FWD", 600], ["STOP"]],
                              annotations={"tape_x_mm": 400, "tape_y_mm": 850}), t, CFG)
    assert tape["attribution"]["first_pass"] == "perception"
    low_conf = score_trial(record(t, scene(400, 250, 0, conf=0.4), scene(400, 780, 0), [["FWD", 600], ["STOP"]]), t, CFG)
    assert low_conf["attribution"]["first_pass"] == "perception"


def test_collision_from_block_displacement_and_operator():
    t = task("RDT-05")
    blocks = [("red", 855, 620, 280, 200, 90)]
    moved = [("red", 900, 620, 280, 200, 90)]
    w0 = scene(300, 620, 90, blocks)
    plan = [["TURN", -90], ["FWD", 300], ["TURN", 90], ["FWD", 1100], ["TURN", 90], ["FWD", 300], ["STOP"]]
    clean = score_trial(record(t, w0, scene(1400, 620, 180, blocks), plan), t, CFG)
    assert clean["first_pass_success"], clean["rounds"][0]["score"]
    pushed = score_trial(record(t, w0, scene(1400, 620, 180, moved), plan), t, CFG)
    assert not pushed["first_pass_success"]
    assert any(e.startswith("block_displaced") for e in pushed["rounds"][0]["score"]["collision_evidence"])
    seen = score_trial(record(t, w0, scene(1400, 620, 180, blocks), plan, annotations={"contact_first_pass": True}), t, CFG)
    assert not seen["first_pass_success"]


def test_closed_loop_with_correction():
    t = task("RDT-01")
    w0 = scene(400, 250, 0)
    rec = record(t, w0, scene(400, 780, 0), [["FWD", 600], ["STOP"]], verification=True,
                 extra_rounds=[([["FWD", 70], ["STOP"]], scene(401, 849, 0))])
    s = score_trial(rec, t, CFG)
    assert not s["first_pass_success"] and s["closed_loop_success"] and s["corrections_used"] == 1


def test_statistics():
    assert mcnemar_exact(6, 1) == pytest.approx(0.125)
    assert mcnemar_exact(0, 0) == 1.0
    lo, hi = wilson(12, 24)
    assert lo == pytest.approx(0.314, abs=0.002) and hi == pytest.approx(0.686, abs=0.002)


def test_session_schedule(tmp_path):
    s1 = Session.create("a", "dry_run", CFG, results_root=tmp_path, repetitions=3, allow_unlocked=True)
    s2 = Session.create("b", "dry_run", CFG, results_root=tmp_path, repetitions=3, allow_unlocked=True)
    sched1 = [(x.task_id, x.rep) for x in s1.schedule]
    assert sched1 == [(x.task_id, x.rep) for x in s2.schedule]  # seeded
    assert len(sched1) == 24
    for tid in {t for t, _ in sched1}:
        reps = [r for t, r in sched1 if t == tid]
        assert reps == [1, 2, 3]
    assert sum(x.tape_check for x in s1.schedule) == 8


def test_parallax_correction_recovers_true_pose_from_raised_marker():
    """Synthetic camera sees the marker raised 55 mm; the host's correction must undo it."""
    import copy

    from ai_learner_host.camera.frame_buffer import FrameBuffer
    from ai_learner_host.perception.service import PerceptionService

    from rdt.sim.synthetic import SimWorld, SyntheticCamera, synthetic_host_config

    geom = (1549.6, 55.0, 1032.0, -828.0)
    base = synthetic_host_config()
    world = SimWorld()
    cam = SyntheticCamera(world, base, parallax=geom)
    corrected = copy.deepcopy(base)
    corrected.parallax_camera_height_mm, corrected.parallax_marker_height_mm = geom[0], geom[1]
    corrected.parallax_nadir_x_mm, corrected.parallax_nadir_y_mm = geom[2], geom[3]
    for x, y, h in [(300, 250, 0), (855, 627, 90), (1400, 1000, 215)]:
        world.set_pose(x, y, h)
        frame = cam.render()
        raw = parse_scene(PerceptionService(FrameBuffer(), base)._process_world(frame).scene_json)
        fixed = parse_scene(PerceptionService(FrameBuffer(), corrected)._process_world(frame).scene_json)
        raw_err = math.hypot(raw.rover_x_mm - x, raw.rover_y_mm - y)
        fixed_err = math.hypot(fixed.rover_x_mm - x, fixed.rover_y_mm - y)
        assert raw_err > 30            # the uncorrected bias is tens of millimetres
        assert fixed_err < 3           # the correction removes it
        assert abs(((fixed.rover_heading_deg - h + 180) % 360) - 180) < 1.0
