"""Session analysis: tables (CSV and LaTeX) and figures for the Results chapter.

Every trial is re-scored with the current scoring code before aggregation, so
all trials are judged by one version of the rules. Voided trials are listed but
excluded. Outputs go to <session>/analysis/.
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rdt._bootstrap import WORKSPACE_ROOT
from rdt.config import ExperimentConfig
from rdt.scoring import score_trial
from rdt.session import Session

CLASSES = ["protocol", "planning", "perception", "execution"]


# ---------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def rate(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, {100 * lo:.0f}-{100 * hi:.0f}%)"


def med(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return st.median(vals) if vals else None


def fmt(v, nd=0) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


# ---------------------------------------------------------------- loading

def load_trials(session: Session, cfg: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    valid, void = [], []
    for d in session.trial_dirs():
        rec = json.loads((d / "trial.json").read_text(encoding="utf-8"))
        rec["_dir"] = str(d)
        task = session.tasks.get(rec["meta"]["task_id"])
        if task is not None:
            rec["scores"] = score_trial(rec, task, cfg)
        if (rec.get("annotations") or {}).get("void"):
            void.append(rec)
        else:
            valid.append(rec)
    return valid, void


def flatten(rec: dict) -> dict[str, Any]:
    m, s = rec["meta"], rec.get("scores", {})
    attr = s.get("attribution") or {}
    tape = s.get("camera_tape") or {}
    return {
        "trial": Path(rec["_dir"]).name,
        "slot": m["slot"],
        "task_id": m["task_id"],
        "family": m["task_family"],
        "difficulty": m["task_difficulty"],
        "rep": m["rep"],
        "status": rec.get("status"),
        "verification": m["verification_enabled"],
        "first_pass_success": s.get("first_pass_success"),
        "closed_loop_success": s.get("closed_loop_success"),
        "replay_success": s.get("replay_success"),
        "corrections_used": s.get("corrections_used"),
        "first_pass_goal_error_mm": s.get("first_pass_goal_error_mm"),
        "first_pass_heading_error_deg": s.get("first_pass_heading_error_deg"),
        "final_goal_error_mm": s.get("final_goal_error_mm"),
        "e_pos_mm": s.get("e_pos_mm"),
        "e_theta_deg": s.get("e_theta_deg"),
        "collision": s.get("collision_any_round"),
        "boundary_violation": s.get("boundary_violation_any_round"),
        "min_block_clearance_mm": s.get("min_block_clearance_mm"),
        "replay_min_block_clearance_mm": s.get("replay_min_block_clearance_mm"),
        "reference_path_mm": s.get("reference_path_mm"),
        "driven_path_first_pass_mm": s.get("driven_path_first_pass_mm"),
        "path_efficiency_first_pass": s.get("path_efficiency_first_pass"),
        "plan_commands": s.get("plan_commands_first_pass"),
        "plan_rotation_deg": s.get("plan_rotation_first_pass_deg"),
        "planning_latency_s": s.get("planning_latency_s"),
        "drive_time_first_pass_s": s.get("drive_time_first_pass_s"),
        "mission_time_s": s.get("mission_time_s"),
        "failure_class_first_pass": attr.get("first_pass"),
        "failure_class_closed_loop": attr.get("closed_loop"),
        "failure_detail": attr.get("detail"),
        "verifier_agreement": ";".join(str(a) for a in s.get("verifier_agreement", [])),
        "camera_minus_tape_mm": tape.get("camera_minus_tape_mm"),
        "battery_v": m.get("battery_v"),
        "planner": m.get("planner"),
    }


# ---------------------------------------------------------------- latex helper

def latex_table(path: Path, caption: str, label: str, header: list[str], rows: list[list[str]], spec: str) -> None:
    def esc(x: str) -> str:
        return str(x).replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")

    lines = [
        r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\small", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
        " & ".join(esc(h) for h in header) + r" \\", r"\midrule",
    ]
    lines += [" & ".join(esc(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- Experiment 1 bridge

def experiment1_family_rates(provider_key: str = "openai_strong") -> dict[str, float]:
    root = WORKSPACE_ROOT / "llm_tests" / "results" / "strong_tier"
    files = sorted(root.glob("*/pilot_family_summary.csv"))
    if not files:
        return {}
    out = {}
    with files[-1].open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["provider_key"] == provider_key:
                out[row["family"]] = float(row["task_success_rate"])
    return out


# ---------------------------------------------------------------- main

def analyse_session(session_root: str | Path, cfg: ExperimentConfig) -> Path:
    session = Session.open(session_root)
    out = Path(session_root) / "analysis"
    out.mkdir(exist_ok=True)
    trials, void = load_trials(session, cfg)
    rows = [flatten(t) for t in trials]
    write_csv(out / "trials.csv", rows)
    write_csv(out / "void_trials.csv", [{**flatten(t), "void_reason": t["annotations"].get("void_reason")} for t in void])

    notes: list[str] = []
    if session.data.get("config_sha256") != cfg.sha256:
        notes.append("WARNING: the experiment config changed after this session was created.")
    if not session.data.get("manifest_locked"):
        notes.append("NOTE: this session used an unlocked task manifest (rehearsal data).")

    # ---- per task table
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    task_rows = []
    for tid in sorted(by_task):
        rs = by_task[tid]
        n = len(rs)
        fp = sum(bool(r["first_pass_success"]) for r in rs)
        cl_rs = [r for r in rs if r["closed_loop_success"] is not None]
        cl = sum(bool(r["closed_loop_success"]) for r in cl_rs)
        rp = sum(bool(r["replay_success"]) for r in rs)
        task_rows.append({
            "task": tid,
            "family": rs[0]["family"],
            "n": n,
            "first_pass": f"{fp}/{n}",
            "closed_loop": f"{cl}/{len(cl_rs)}" if cl_rs else "-",
            "replay": f"{rp}/{n}",
            "median_goal_error_mm": fmt(med([r["first_pass_goal_error_mm"] for r in rs])),
            "median_e_pos_mm": fmt(med([r["e_pos_mm"] for r in rs])),
            "median_e_theta_deg": fmt(med([r["e_theta_deg"] for r in rs]), 1),
            "min_clearance_mm": fmt(min([r["min_block_clearance_mm"] for r in rs if r["min_block_clearance_mm"] is not None], default=None)),
            "median_path_efficiency": fmt(med([r["path_efficiency_first_pass"] for r in rs]), 2),
        })
    n = len(rows)
    fp = sum(bool(r["first_pass_success"]) for r in rows)
    cl_rows = [r for r in rows if r["closed_loop_success"] is not None]
    cl = sum(bool(r["closed_loop_success"]) for r in cl_rows)
    rp = sum(bool(r["replay_success"]) for r in rows)
    totals = {
        "task": "Total", "family": "", "n": n,
        "first_pass": rate(fp, n), "closed_loop": rate(cl, len(cl_rows)) if cl_rows else "-", "replay": rate(rp, n),
        "median_goal_error_mm": fmt(med([r["first_pass_goal_error_mm"] for r in rows])),
        "median_e_pos_mm": fmt(med([r["e_pos_mm"] for r in rows])),
        "median_e_theta_deg": fmt(med([r["e_theta_deg"] for r in rows]), 1),
        "min_clearance_mm": fmt(min([r["min_block_clearance_mm"] for r in rows if r["min_block_clearance_mm"] is not None], default=None)),
        "median_path_efficiency": fmt(med([r["path_efficiency_first_pass"] for r in rows]), 2),
    }
    write_csv(out / "per_task.csv", task_rows + [totals])
    latex_table(
        out / "table_per_task.tex",
        "Rover Driving Test results for each task. Rates for each task are descriptive; the total row gives Wilson 95\\% intervals.",
        "tab:rdt_per_task",
        ["Task", "First pass", "Closed loop", "Replay", "Goal err. (mm)", "$e_{pos}$ (mm)", "Min. clear. (mm)", "Path eff."],
        [[t["task"], t["first_pass"], t["closed_loop"], t["replay"], t["median_goal_error_mm"], t["median_e_pos_mm"],
          t["min_clearance_mm"], t["median_path_efficiency"]] for t in task_rows + [totals]],
        "lccccccc",
    )

    # ---- McNemar (H1)
    a = sum(1 for r in rows if r["replay_success"] and r["first_pass_success"])
    b = sum(1 for r in rows if r["replay_success"] and not r["first_pass_success"])
    c = sum(1 for r in rows if not r["replay_success"] and r["first_pass_success"])
    d = sum(1 for r in rows if not r["replay_success"] and not r["first_pass_success"])
    p = mcnemar_exact(b, c)
    mc = {"replay_ok_rover_ok": a, "replay_ok_rover_fail": b, "replay_fail_rover_ok": c, "replay_fail_rover_fail": d,
          "discordant": b + c, "exact_mcnemar_p": round(p, 4)}
    (out / "mcnemar.json").write_text(json.dumps(mc, indent=2), encoding="utf-8")
    latex_table(
        out / "table_mcnemar.tex",
        f"Paired outcomes of the first pass on the rover and its simulator replay (exact McNemar test, $p = {p:.3f}$).",
        "tab:rdt_mcnemar",
        ["", "Rover succeeds", "Rover fails"],
        [["Replay succeeds", str(a), str(b)], ["Replay fails", str(c), str(d)]],
        "lcc",
    )

    # ---- attribution (H2)
    failures = [r for r in rows if not r["first_pass_success"]]
    counts = Counter(r["failure_class_first_pass"] for r in failures)
    attr_rows = [{"class": k, "count": counts.get(k, 0),
                  "share": f"{100 * counts.get(k, 0) / len(failures):.0f}%" if failures else "-"} for k in CLASSES]
    cl_fail = sum(1 for r in rows if r["failure_class_closed_loop"] == "verification")
    attr_rows.append({"class": "verification (closed loop not recovered)", "count": cl_fail, "share": "-"})
    write_csv(out / "attribution.csv", attr_rows)
    latex_table(
        out / "table_attribution.tex",
        f"Classes of the {len(failures)} first-pass failures (Table 3.8).",
        "tab:rdt_attribution",
        ["Class", "Count", "Share of failures"],
        [[r["class"], str(r["count"]), r["share"]] for r in attr_rows],
        "lcc",
    )

    # ---- verifier agreement, timing, camera vs tape, repeatability
    agreements = [a for t in trials for a in t["scores"].get("verifier_agreement", []) if a is not None]
    tape = [r["camera_minus_tape_mm"] for r in rows if r["camera_minus_tape_mm"] is not None]
    timing = {
        "planning_latency_s_median": med([r["planning_latency_s"] for r in rows]),
        "drive_time_first_pass_s_median": med([r["drive_time_first_pass_s"] for r in rows]),
        "mission_time_s_median": med([r["mission_time_s"] for r in rows]),
    }
    repeat = []
    for tid, rs in sorted(by_task.items()):
        pts = []
        for t in trials:
            if t["meta"]["task_id"] == tid and t["scores"].get("rounds"):
                s0 = t["scores"]["rounds"][0]["score"]
                if s0.get("final_x_mm") is not None:
                    pts.append((s0["final_x_mm"], s0["final_y_mm"]))
        spread = max((math.dist(p1, p2) for i, p1 in enumerate(pts) for p2 in pts[i + 1:]), default=None)
        repeat.append({"task": tid, "runs": len(pts), "max_spread_of_first_pass_end_mm": fmt(spread)})
    write_csv(out / "repeatability.csv", repeat)
    summary = {
        "session": session.name, "mode": session.mode, "trials": n, "void_trials": len(void),
        "first_pass": rate(fp, n), "closed_loop": rate(cl, len(cl_rows)) if cl_rows else None, "replay": rate(rp, n),
        "mcnemar": mc,
        "failure_classes": dict(counts),
        "verifier_agreement": f"{sum(agreements)}/{len(agreements)}" if agreements else None,
        "camera_minus_tape_mm": {"n": len(tape), "median": med(tape), "max": max(tape) if tape else None},
        "timing": timing,
        "experiment1_gpt6sol_family_success": experiment1_family_rates(),
        "notes": notes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- Experiment 1 bridge by family
    e1 = experiment1_family_rates()
    fam_rows = []
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    for fam, rs in sorted(by_family.items()):
        k = sum(bool(r["first_pass_success"]) for r in rs)
        kr = sum(bool(r["replay_success"]) for r in rs)
        fam_rows.append({"family": fam, "experiment1_sim_success": fmt(100 * e1[fam], 0) + "%" if fam in e1 else "-",
                         "rdt_replay": f"{kr}/{len(rs)}", "rdt_first_pass": f"{k}/{len(rs)}"})
    write_csv(out / "family_bridge.csv", fam_rows)
    latex_table(
        out / "table_family_bridge.tex",
        "Planner success by benchmark family: Experiment 1 (simulation, GPT-6 Sol) against the Rover Driving Test.",
        "tab:rdt_family_bridge",
        ["Family", "Exp. 1 simulation", "RDT replay", "RDT first pass"],
        [[f["family"].replace("_", " "), f["experiment1_sim_success"], f["rdt_replay"], f["rdt_first_pass"]] for f in fam_rows],
        "lccc",
    )

    _figures(out, trials, rows, by_task)
    report = [f"# {session.name} ({session.mode})", "", *notes, "",
              f"Trials: {n} valid, {len(void)} void", f"First pass: {summary['first_pass']}",
              f"Closed loop: {summary['closed_loop']}", f"Replay: {summary['replay']}",
              f"McNemar: b={b}, c={c}, exact p={p:.4f}",
              f"First-pass failure classes: {dict(counts)}",
              f"Verifier agreement with the geometric rule: {summary['verifier_agreement']}",
              f"Camera minus tape: {summary['camera_minus_tape_mm']}",
              f"Timing medians: {timing}"]
    (out / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out


def _figures(out: Path, trials: list[dict], rows: list[dict], by_task: dict[str, list[dict]]) -> None:
    plt.rcParams.update({"font.size": 8})
    tasks = sorted(by_task)

    # 1. closing frames montage (first run of each task)
    frames = []
    for tid in tasks:
        cand = sorted((t for t in trials if t["meta"]["task_id"] == tid), key=lambda t: t["meta"]["rep"])
        if cand:
            img = cv2.imread(str(Path(cand[0]["_dir"]) / "final.png"))
            if img is not None:
                frames.append((tid, img))
    if frames:
        cols = 4 if len(frames) > 3 else len(frames)
        rws = math.ceil(len(frames) / cols)
        fig, axes = plt.subplots(rws, cols, figsize=(3.3 * cols, 2.6 * rws))
        axes = np.atleast_1d(axes).ravel()
        for ax, (tid, img) in zip(axes, frames):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(tid, fontsize=8)
            ax.axis("off")
        for ax in axes[len(frames):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "fig_closing_frames.png", dpi=220)
        fig.savefig(out / "fig_closing_frames.pdf")
        plt.close(fig)

    # 2. final positions around the goal
    if tasks:
        cols = 4 if len(tasks) > 3 else len(tasks)
        rws = math.ceil(len(tasks) / cols)
        fig, axes = plt.subplots(rws, cols, figsize=(2.6 * cols, 2.5 * rws))
        axes = np.atleast_1d(axes).ravel()
        for ax, tid in zip(axes, tasks):
            for t in (t for t in trials if t["meta"]["task_id"] == tid):
                s = t["scores"]
                g = s.get("goal") or {}
                if not s.get("rounds"):
                    continue
                r0 = s["rounds"][0]["score"]
                if r0.get("final_x_mm") is None:
                    continue
                tol_t = g.get("position_tolerance_mm", 50.0)
                lim_t = max(tol_t * 2.2, 120)
                dx, dy = r0["final_x_mm"] - g["x_mm"], r0["final_y_mm"] - g["y_mm"]
                colour = "#2a7" if s.get("first_pass_success") else "#d33"
                far = max(abs(dx), abs(dy))
                if far > lim_t * 0.92:  # pin far misses to the border and label the distance
                    k = lim_t * 0.92 / far
                    ax.plot(dx * k, dy * k, "D", color=colour, ms=5, mfc="none")
                    ax.annotate(f"{math.hypot(dx, dy):.0f} mm", (dx * k, dy * k), fontsize=6,
                                xytext=(3, 3), textcoords="offset points", color=colour)
                else:
                    ax.plot(dx, dy, "o", color=colour, ms=5)
                pred = s["rounds"][0].get("predicted_end_pose")
                if pred and max(abs(pred[0] - g["x_mm"]), abs(pred[1] - g["y_mm"])) <= lim_t:
                    ax.plot(pred[0] - g["x_mm"], pred[1] - g["y_mm"], "x", color="#555", ms=5)
            tol = next((t["scores"]["goal"]["position_tolerance_mm"] for t in trials
                        if t["meta"]["task_id"] == tid and t["scores"].get("goal")), 50.0)
            ax.add_patch(plt.Circle((0, 0), tol, fill=False, ls="--", color="#333"))
            lim = max(tol * 2.2, 120)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.axhline(0, color="#ccc", lw=0.6)
            ax.axvline(0, color="#ccc", lw=0.6)
            ax.set_title(tid)
            ax.set_xlabel("x error (mm)")
            ax.set_ylabel("y error (mm)")
        for ax in axes[len(tasks):]:
            ax.axis("off")
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], marker="o", ls="", color="#2a7", label="first pass success"),
            Line2D([], [], marker="o", ls="", color="#d33", label="first pass fail"),
            Line2D([], [], marker="D", ls="", color="#d33", mfc="none", label="beyond the plot (distance shown)"),
            Line2D([], [], marker="x", ls="", color="#555", label="plan's ideal end pose"),
            Line2D([], [], ls="--", color="#333", label="position tolerance"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out / "fig_final_positions.png", dpi=220)
        fig.savefig(out / "fig_final_positions.pdf")
        plt.close(fig)

    # 3. execution error against commanded distance and rotation
    pts = [(t["scores"]["rounds"][0], t) for t in trials if t["scores"].get("rounds")]
    data = []
    for r0, _t in pts:
        if r0.get("e_pos_mm") is None:
            continue
        dist = sum(abs(float(c[1])) for c in r0["commands"] if c[0] in ("FWD", "BACK"))
        rot = sum(abs(float(c[1])) for c in r0["commands"] if c[0] == "TURN")
        data.append((dist, rot, r0["e_pos_mm"], r0["e_theta_deg"]))
    if data:
        d = np.asarray(data, float)
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        axes[0].scatter(d[:, 0], d[:, 2], s=14, color="#36c")
        axes[0].set_xlabel("Commanded translation (mm)")
        axes[0].set_ylabel("$e_{pos}$ (mm)")
        axes[1].scatter(d[:, 1], d[:, 3], s=14, color="#c63")
        axes[1].set_xlabel("Commanded rotation (deg)")
        axes[1].set_ylabel(r"$e_{\theta}$ (deg)")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig_execution_error.png", dpi=220)
        fig.savefig(out / "fig_execution_error.pdf")
        plt.close(fig)

    # 4. success per task
    if tasks:
        x = np.arange(len(tasks))
        def frac(tid, key):
            rs = by_task[tid]
            vals = [r[key] for r in rs if r[key] is not None]
            return sum(bool(v) for v in vals) / len(vals) if vals else 0.0
        fig, ax = plt.subplots(figsize=(7, 2.6))
        w = 0.27
        ax.bar(x - w, [frac(t, "replay_success") for t in tasks], w, label="Simulator replay", color="#999")
        ax.bar(x, [frac(t, "first_pass_success") for t in tasks], w, label="Rover first pass", color="#36c")
        ax.bar(x + w, [frac(t, "closed_loop_success") for t in tasks], w, label="Rover closed loop", color="#2a7")
        ax.set_xticks(x, tasks)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Success rate")
        ax.legend(ncol=3, fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(out / "fig_success_by_task.png", dpi=220)
        fig.savefig(out / "fig_success_by_task.pdf")
        plt.close(fig)

    # 5. where the time goes
    if rows and tasks:
        plan = [med([r["planning_latency_s"] for r in by_task[t]]) or 0 for t in tasks]
        drive = [med([r["drive_time_first_pass_s"] for r in by_task[t]]) or 0 for t in tasks]
        total = [med([r["mission_time_s"] for r in by_task[t]]) or 0 for t in tasks]
        other = [max(0.0, tt - p - dr) for tt, p, dr in zip(total, plan, drive)]
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.bar(tasks, plan, label="Planning (LLM)", color="#c63")
        ax.bar(tasks, drive, bottom=plan, label="Driving (first pass)", color="#36c")
        ax.bar(tasks, other, bottom=np.add(plan, drive), label="Camera settle, verification, corrections", color="#bbb")
        ax.set_ylabel("Median time (s)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fig_time_breakdown.png", dpi=220)
        fig.savefig(out / "fig_time_breakdown.pdf")
        plt.close(fig)
