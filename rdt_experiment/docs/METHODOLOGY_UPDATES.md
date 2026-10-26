# What Chapters 3 and 4 need to say after this build

These are facts about how Experiment 2 is now run. Each item names the place in the report it affects.

## Values that are now known (replace the red TBCs)

| Where | Value |
|---|---|
| Table 3.9, arena | 1710 × 1254 mm (every host snapshot since 30 September uses this; 1026 mm is out of date) |
| Table 3.9, camera settle | 10 s, then three new frames (set by `rdt_config.json`; the host's own setting was 5 s) |
| Table 3.9, planner | GPT-6 Sol (`gpt-6-sol`), medium reasoning, 8192 output tokens. The host settings file had Gemini 3.1 Pro selected; the runner overrides it |
| Table 3.7, tasks | RDT-01 to RDT-08 as in `tasks/*.json`, Figure 3.4 = `docs/rdt_layouts_main.pdf` |
| Table 3.7, tolerances | 50 to 75 mm and 10°, fixed before the dry run and kept after checking it: every tolerance is at least 3 times the largest subsystem execution error (16 mm) and at least 40% above the median dry-run execution error (35 mm uncorrected, 21 mm with the parallax correction). They were not changed to make any trial pass |
| Table 3.9, battery | Trials start only at 3.65 V or more (the battery fell from 3.9 to 3.64 V over 8 trials with no loss of accuracy) |
| §3.6, T2 camera position accuracy | Against 12 tape points: 55 mm mean error before correction, 7.7 mm mean (22 mm max) after the parallax correction |
| §3.6, T3 to T5 | Subsystem session, one run per motion: straight 300 / 800 mm, e_pos 9 / 16 mm; turns of 20 to 180°, e_θ at most 2.8°; 400 mm square back to start within 7 mm |
| §3.8.3, tape subset | Every third slot of the seeded schedule (8 of 24 trials) |
| Blocks | Nominal 280 × 200 mm (measured by the camera at about 280 × 197 mm) |

## Method changes to describe

1. **Frozen start state.** After placement the host takes a settled capture. That one world state is sent to the planner and replayed in the simulator. The placement limits are 20 mm and 5° for the rover, 30 mm and 10° for blocks, and a pose confidence of at least 0.75.
2. **Goals next to a block** (RDT-04) follow the block's measured position, so placement error is not charged to the planner.
3. **Verification switch.** The 24 main trials run with verification on. Round 0 gives first-pass success, and the last round gives closed-loop success. With verification off, the verifier is still called once so its verdict can be compared with the rule, but no correction is executed.
4. **NACK replans.** The host replans after every NACK, with no limit. In the experiment, NACK replans and verifier corrections share one budget of 2 extra rounds. This is a deliberate limit on the system under test and should be stated.
5. **Execution error** is measured against the ideal kinematic end pose of the whole round-0 plan (the simulator's motion model, ignoring collisions). The simulator itself stops at the first collision, which would make e_pos meaningless for a colliding plan. e_pos is not computed for a round that was NACKed.
6. **Collision on the real rover.** A collision is recorded from any of three sources: the operator saw contact; a block moved 25 mm or more between the start and end captures; or the rover outline from the camera trail overlapped a block outline. All three sources are stored so the Discussion can say which one fired.
7. **New measures**, all from the camera trail:
   - minimum clearance to the blocks (rover and replay);
   - boundary violation, with a 10 mm tolerance for camera noise;
   - driven path length, using a 10 mm deadband so pose jitter does not add length;
   - path efficiency, defined as the conservative shortest path from W0 divided by the first-pass path driven. Because the reference path keeps a circle around the rover, efficiency can slightly exceed 1;
   - planning, driving and total mission time.
8. **Attribution rules** (Table 3.8), applied in this order to a first-pass failure:
   - **Protocol:** a NACK that is not a safety stop, a protocol error, a disconnect, or a path that was never acknowledged.
   - **Planning:** the replay fails.
   - **Perception:** the start state was below the placement or confidence limits, or the camera and tape disagree by more than 30 mm.
   - **Execution:** anything else.

   A closed-loop failure after a first-pass failure is classed as Verification. Verifier verdicts are compared with the geometric rule for every round.
9. **Video record.** Every trial is filmed from the overhead camera at 8 frames per second, with a title card (prompt, run, planner, verification mode, start frame), the rover trail and a closing frame. These are presentation records; scoring uses only the settled captures.
10. **Rehearsal rig.** The procedure and software were rehearsed on a simulated rig. This is a synthetic camera rendered through a known homography and processed by the host's own perception, with a simulated rover speaking the same WebSocket protocol. No simulated data is reported as results.

## Two defects found in the dry runs and fixed before the main session

These are worth describing, because finding them is what the dry runs were for.

**Marker-height parallax (perception).** The tape check showed the camera reporting the rover 45 to 75 mm from its true position, always pushed away from the camera. The homography maps the floor, but the ArUco marker is 55 mm above it. The camera is about 1.59 m high and about 0.8 m outside the arena's bottom edge, so it sees the marker displaced radially away from the point under the camera by H / (H − h). From those measurements that factor is 1.0358. A least-squares fit to 12 tape points gave 1.0368, with the point under the camera at (1032, −828) mm, against the measured position of about −800 mm. The two estimates agree to 0.1%.

The host now corrects the marker position with p_true = n + (p − n)(H − h)/H, using the fitted values, before applying the 63 mm marker-to-centre offset. Headings are unchanged. The blocks are flat and need no correction. After the correction the camera agrees with the tape to 7.7 mm on average (22 mm at most), against 55 mm before.

The bias also stretched every measured movement by 3.7%, which made execution error look larger than it was. Re-scored with the correction, the dry run's median execution error falls from 35 to 21 mm.

**Stale verifier input (verification).** In correction rounds the host gave the verifier the mission's original before-state but only the latest correction plan. So the verifier saw, for example, a 1102 mm displacement after an 89 mm command, and reported the result as uncertain or failed (RDT-05, RDT-08). It now receives every command executed in the mission. For a path stopped part-way by a NACK, that means only the commands the rover acknowledged.

## Host changes (Chapter 4 and the AI-use appendix)

- `perception/service.py` and `config.py`: marker parallax correction (`parallax_*` settings, off unless set; the runner sets the fitted values).
- `application/mission_controller.py`: the verifier receives the cumulative executed plan (`cumulative_executed_plan`).
- `tests/test_parallax.py`, `tests/test_verifier_cumulative_plan.py`: tests for both.

- `planning/prompt_builder.py`: removed the fixed "(26 cm)" and "(20 cm)" text, which contradicted the 155.5 × 121.3 mm footprint in the same prompt.
- `config.py`: defaults updated to the final rover (155.5 × 121.3 mm footprint, rover centre 63 mm in front of the marker centre).
- `tests/test_llm_integration.py`: checks the footprint text and that the stale text is gone.
- `data/settings.json`: planner set to GPT-6 Sol and settle time to 10 s. The runner also enforces these values in memory.

## Observations from the replay of past host runs

Replaying the 31 plans stored in the host database (30 September to 2 October) gave a median e_pos of 30 mm and a median e_θ of 1.8°. In two runs, a commanded turn of about 25 to 28° left the heading almost unchanged, while the same turn executed correctly on another attempt. The subsystem session tested this with SUB-TURN-R20 and SUB-TURN-L25. Both turns executed correctly (e_θ 0.9° and 0.2°), so the fault does not occur with the current firmware.
