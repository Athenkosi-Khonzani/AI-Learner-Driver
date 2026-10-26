# CV Backend Integration — 2026-09

The standalone computer-vision experiments are now integrated into the host application.

## Integrated perception flow

```text
CameraWorker / FrameBuffer
        ↓
intrinsic lens correction
        ↓
arena homography / bird's-eye view
        ↓
├── calibrated HSV → coloured blocks
└── ArUco DICT_4X4_50 / ID 29 → rover pose
        ↓
64 mm left + 90 mm forward rover-centre compensation
        ↓
temporal filtering + confidence
        ↓
WorldStateSnapshot
        ├── annotated image
        ├── scene JSON
        ├── Markdown summary
        └── ASCII/text map
```

The `MissionController` already calls `PerceptionService.capture_world_state()` before planning, after a completed path, and when re-planning. Those calls now use the real perception pipeline rather than the original placeholder. `req_heading` messages now call the same filtered ArUco pose pipeline.

## Settings → Camera

Three calibration steps are available in the desktop GUI:

1. **Lens calibration** — load the intrinsic checkerboard JSON.
2. **Arena homography** — capture the undistorted frame and click TL → TR → BR → BL.
3. **Colour calibration** — select RED/GREEN/YELLOW/BLUE and click at least five patches per colour.

The calibration dependency order is intentional. Loading new intrinsics resets arena + colour calibration. Saving a new homography resets colour calibration.

## Planner context

`PromptBuilder` now includes structured `scene_json` plus the ASCII/text map. `PlannerService` continues to attach the annotated world image to the LLM adapter interface.

## Snapshot output

Each persisted world snapshot saves:

- raw camera image
- annotated bird's-eye image
- JSON world state in SQLite
- Markdown world state in SQLite
- `<snapshot>_world_map.txt` beside the image snapshots

## Current fixed rover localisation configuration

- Dictionary: `DICT_4X4_50`
- Marker ID: `29`
- Rover centre relative to marker at heading 0°: `64 mm left`, `90 mm forward`
- Heading convention: `0° = +Y/up`, `90° = +X/right`

## Engineer live annotated view

The Engineer page now continuously refreshes the **Processed · Annotated World** panel at approximately 4 Hz. Camera capture remains on the dedicated camera QThread, while OpenCV perception runs off the GUI thread through `PerceptionService.capture_world_state()`. The live panel therefore shows the current bird's-eye annotation, rover pose/heading/confidence, and coloured blocks even when no mission is actively being planned.
