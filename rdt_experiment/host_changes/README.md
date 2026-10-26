# Host changes for Experiment 2

## Which patch to apply

Run these from the folder that contains `ai_learner_host`:

| Your host | Command |
|---|---|
| **Already has the first patch** (prompt and footprint fix, applied before the dry runs) | `git apply rdt_experiment/host_changes/ai_learner_host_exp2_update2.patch` |
| Original, unpatched | `git apply rdt_experiment/host_changes/ai_learner_host_exp2.patch` (contains everything) |

If `git apply` refuses, copy the files in `files/` over the same paths in `ai_learner_host/`. That works from either starting point.

Afterwards, run `python -m pytest tests` in `ai_learner_host`. It should report **32 passed**.

## What changes

| File | Change | When |
|---|---|---|
| `src/ai_learner_host/planning/prompt_builder.py` | The planner prompt no longer prints the old rover's "(26 cm)" and "(20 cm)" next to the real footprint | First patch |
| `src/ai_learner_host/config.py` | Defaults match the final rover (155.5 × 121.3 mm, centre 63 mm ahead of the marker); new `parallax_*` settings | First patch + update 2 |
| `src/ai_learner_host/perception/service.py` | Corrects the marker position for its 55 mm height before the 63 mm offset is applied. Off unless the `parallax_*` settings are set | Update 2 |
| `src/ai_learner_host/application/mission_controller.py` | The verifier receives every command executed in the mission, not only the latest correction | Update 2 |
| `tests/test_llm_integration.py`, `tests/test_parallax.py`, `tests/test_verifier_cumulative_plan.py` | Tests for the above | Both |

You do not need to edit `data/settings.json`. The Experiment 2 runner sets the planner (GPT-6 Sol), the 10 s settle and the fitted parallax values in memory, and logs every value it changed.

To use the correction in the normal host app too, add these four lines to `data/settings.json`:

```
"parallax_camera_height_mm": 1549.6,
"parallax_marker_height_mm": 55.0,
"parallax_nadir_x_mm": 1032.0,
"parallax_nadir_y_mm": -828.0,
```

Refit these values if the phone mount is moved.
