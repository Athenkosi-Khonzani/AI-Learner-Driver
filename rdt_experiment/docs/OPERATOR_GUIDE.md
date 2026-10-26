# Operator guide: running the Rover Driving Test

## 0. Before the first session

1. Copy the host fixes into your `ai_learner_host` (see `host_changes/`), then run its tests: `python -m pytest tests` (25 should pass).
2. Rehearse once at home: `python scripts/run_rdt.py --simulate`. Make a `dry_run` session, then click through Place, Capture, Run and Save for two or three tasks. The simulated planner drives in straight lines and ignores blocks, so it fails the obstacle tasks; that is expected and exercises the failure classes.
3. Mark the rover centre on the chassis (it is 63 mm in front of the marker centre), so tape readings measure the same point the camera reports.

## 1. Dry runs: fix the tolerances (once)

The tolerances in the task files (50 to 75 mm, 10 degrees) are provisional. Your past runs show execution errors of about 30 to 55 mm over 700 to 3000 mm of driving, so they need checking on the real rover before anything is locked.

1. New session, mode `subsystem`, repetitions 3. These eight fixed plans (straight 300 and 800 mm, turns of 20, 25, 90 and 180 degrees, a 400 mm square) run without the LLM, so their error is pure execution error. This is also your T3 to T5 data for Section 3.6.
2. New session, mode `dry_run`, all tasks, repetitions 1, with GPT-6 Sol.
3. `python scripts/analyse_rdt.py <session>` for both. Look at `e_pos_mm` and `e_theta_deg` in `trials.csv`.
4. Set each task's `position_tolerance_mm` so that a correct plan executed normally passes. One defensible rule is the 90th percentile of `e_pos` for plans of similar length, plus 20 mm, rounded up to 10 mm. Write that rule in Chapter 3.
5. Set `battery.min_voltage_v` in `config/rdt_config.json`.
6. Lock the task set: `python scripts/validate_tasks.py --lock`. From here on, do not edit a task file or the config. A session records the config hash and warns if it changed.

Dry-run data is rehearsal data. It is not pooled with the main results.

## 1a. If the phone mount has moved: refit the parallax correction

The correction depends on where the camera is: how high it is, and which floor point lies directly under it. Moving the mount changes both, so the stored values become wrong. Refit before any trial you will keep:

1. In the host app, recalibrate the arena homography (and colours if the lighting changed). Close the host app.
2. Reopen the host app just to read positions, or use its live view while it is still open. It applies no correction, so it shows raw camera positions.
3. Put the rover centre on 6 to 9 tape-measured points spread over the arena: corners, middle and edges, not all in one area. For each, write `tape_x, tape_y` and the host's reported `cam_x, cam_y` into a CSV with the header `tape_x,tape_y,cam_x,cam_y`.
4. `python scripts\fit_parallax.py points.csv`. Check that the "after" error is about 10 mm on average and below 30 mm everywhere. Then run it again with `--write`.
5. Close the host app and start `run_rdt.py`. The log shows the new `parallax_*` values.

Do not move the mount again until the main session is finished.

## 1b. After updating the host (parallax correction and verifier fix)

1. Apply the updated `host_changes` patch and run the host tests (32 should pass).
2. Start `run_rdt.py`. The log must list the four `parallax_*` overrides.
3. Camera check in `run_rdt.py`: place the rover centre on 3 or 4 new tape-measured points and compare with the live view. They should agree within about 10 to 20 mm. If they differ by about 50 mm, the correction is not active or the mount moved after the fit.
4. Optional: a `subsystem` session with only SUB-FWD-800 and SUB-SQUARE (type them in "Tasks") to see the corrected execution error.
5. Then lock: `python scripts\validate_tasks.py --lock`.

## 2. The main session (24 trials)

New session, mode `main`. The schedule is 8 tasks × 3 runs in a seeded random order, with every third slot marked for a tape check. Keep the verification loop ON: round 0 of each trial is the first-pass result, so first-pass and closed-loop success come from the same trials.

For each slot:

1. Select the slot (the window picks the next one). Read the prompt in the black box.
2. Place the blocks and the rover on the dashed outlines in the live view. The overlay turns each item green (OK) when it is within the limits: rover 20 mm and 5 degrees, blocks 30 mm and 10 degrees.
3. Check that the battery reads above the minimum, or type a multimeter reading into "Manual battery".
4. Press **1 Capture start state**. The window waits for the camera to settle (10 s), then freezes the frame. That frozen state is exactly what the planner receives and what the simulator replays.
5. Press **2 Run trial**. Do not touch the arena. The window shows the recording, with the trail drawn as the rover moves.
6. Use **Abort** only if the rover is about to leave the arena or damage something. The trial is kept with status `aborted`.
7. When the result appears:
   - On a tape-check slot, measure the rover centre from the arena origin (bottom-left corner) and enter x and y in mm.
   - Tick "Contact seen" if the rover touched a block in that round.
   - Press **Save observations and re-score**.
8. If the trial was invalid for a reason outside the system under test (you bumped the camera, a block was the wrong colour), tick **Void**, give the reason and save. The slot becomes pending and can be re-run, and the void trial stays on file.

Budget about 3 to 4 minutes per trial, so about 1.5 to 2 hours for the 24.

## 3. After the session

`python scripts/analyse_rdt.py --latest` writes `results/<session>/analysis/`:

- `table_per_task.tex`, `table_mcnemar.tex`, `table_attribution.tex`, `table_family_bridge.tex`
- `fig_closing_frames.pdf`, `fig_final_positions.pdf`, `fig_execution_error.pdf`, `fig_success_by_task.pdf`, `fig_time_breakdown.pdf`
- `trials.csv` (one row per trial), `summary.json`, `README.md`

## What the runner does that the host alone does not

- **Frozen start state.** The planner gets the captured W0 instead of a fresh frame, so the replay uses the same state.
- **Round budget.** A NACK makes the host replan without limit. The runner counts NACK replans together with corrections: 0 extra rounds with verification off, 2 with it on. After that it stops sending paths and ends the trial with status `round_limit`.
- **Settled end state.** If the last round did not end with the host's settled capture (after a NACK, abort or timeout), the runner takes one, so every scored end pose is settled.
