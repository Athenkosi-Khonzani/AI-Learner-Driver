# Rover Driving Test (Experiment 2)

This folder runs Experiment 2 of the AI Learner Rover Driver project. The rover drives each task from a frozen start state, and every trial is recorded, replayed in the Experiment 1 simulator, scored and attributed.

The harness does not copy the host application. It imports the production host (`ai_learner_host`) and the Experiment 1 simulator (`llm_test`), so the system under test is the one described in Chapter 4.

## Folder layout

Put the three projects side by side:

```
<parent>/
  ai_learner_host/     production host (with the small Experiment 2 fixes)
  llm_tests/           Experiment 1 harness and simulator (unchanged)
  rdt_experiment/      this folder
```

If the folders live elsewhere, set `RDT_HOST_SRC` and `RDT_LLM_TEST_SRC` to the folders that contain `ai_learner_host` and `llm_test`.

## Install

```powershell
cd rdt_experiment
pip install -r requirements.txt
```

The host must already be calibrated (intrinsics, arena homography, colours) in `ai_learner_host/data/settings.json`. The harness reads that file and applies the frozen values in `config/rdt_config.json` in memory. It never writes them back to disk.

## Commands

| Command | What it does |
|---|---|
| `python scripts/run_rdt.py --simulate` | Rehearse the whole procedure with a simulated camera, rover and planner (no hardware, no API cost) |
| `python scripts/run_rdt.py` | The real rig: rover, IP camera and GPT-6 Sol |
| `python scripts/validate_tasks.py` | Check the eight tasks and redraw Figure 3.4 (`docs/rdt_layouts_main.pdf`) |
| `python scripts/validate_tasks.py --lock` | Lock the task files with SHA-256 hashes (required before the main session) |
| `python scripts/validate_tasks.py --set subsystem` | The fixed-plan dry-run tasks (T3 to T5, no LLM) |
| `python scripts/analyse_rdt.py --latest` | Tables (CSV and LaTeX) and figures for a session |
| `python scripts/fit_parallax.py points.csv --write` | Refit the marker-height parallax correction from tape points (after moving the phone mount) |
| `python scripts/replay_host_runs.py` | Replay past host missions from the host database (checks the replay pipeline) |
| `python -m pytest tests` | 17 tests, including three end-to-end simulated trials |

## What one trial produces

Each trial is saved in `results/<session>/trials/<slot>_<task>_r<run>/`:

| File | Contents |
|---|---|
| `video.mp4` | Title card (prompt, run, planner, verification mode, start frame), the drive with the rover's trail, and a closing frame |
| `title.png`, `final.png` | The first and last frames as images, for the report |
| `trial.json` | Everything else: start state W0, every plan, timings, acknowledgements, NACKs, verifier verdicts, settled end states, trail, observations and scores |
| `trail.csv` | Rover pose from the camera, about 8 times a second, tagged by round |
| `events.jsonl` | Protocol and host events with times |
| `W0_*.jpg`, `after_*_*.jpg` | Raw and annotated camera frames for the start and after each round |
| `planner_prompt.txt` | The exact system and user prompt sent to the planner |

`trial.json` holds only the raw data the scores are computed from, so `analyse_rdt.py` re-scores every trial with the same code. The video plays in VLC. If PowerPoint or another player will not open the `mp4v` codec, convert it with `ffmpeg -i video.mp4 -c:v libx264 out.mp4`.

See `docs/OPERATOR_GUIDE.md` for the procedure in the arena and `docs/METHODOLOGY_UPDATES.md` for what Chapter 3 needs to say.
