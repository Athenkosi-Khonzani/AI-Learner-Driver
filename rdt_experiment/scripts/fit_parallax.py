"""Fit the marker-height parallax correction from tape-measured points.

Do this whenever the phone mount moves (after recalibrating the homography).

1. Make a CSV with one row per point and the header
       tape_x,tape_y,cam_x,cam_y
   tape = rover centre measured with the tape from the arena origin (bottom-left), mm.
   cam  = the rover position the camera reports WITHOUT the correction: read it from
          the host app's live view (the host app has no correction unless you added
          parallax_* to its settings.json). Use at least 6 points spread over the arena.
2. python scripts\\fit_parallax.py points.csv            (shows the fit)
   python scripts\\fit_parallax.py points.csv --write    (also writes it to config/rdt_config.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "rdt_config.json"


def fit(tape: np.ndarray, cam: np.ndarray):
    """cam = k * tape + c, shared k for x and y (radial scaling about the nadir)."""
    n = len(tape)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[0::2, 0], A[0::2, 1], b[0::2] = tape[:, 0], 1.0, cam[:, 0]
    A[1::2, 0], A[1::2, 2], b[1::2] = tape[:, 1], 1.0, cam[:, 1]
    (k, cx, cy), *_ = np.linalg.lstsq(A, b, rcond=None)
    nadir = np.array([cx, cy]) / (1.0 - k)
    corrected = nadir + (cam - nadir) / k
    return float(k), nadir, corrected


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--marker-height", type=float, default=55.0, help="marker height above the floor, mm")
    p.add_argument("--write", action="store_true", help="write the fitted values into config/rdt_config.json")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8-sig")))
    tape = np.array([[float(r["tape_x"]), float(r["tape_y"])] for r in rows])
    cam = np.array([[float(r["cam_x"]), float(r["cam_y"])] for r in rows])
    if len(rows) < 4:
        print("Need at least 4 points (6 or more spread over the arena is better).")
        return 1

    k, nadir, corrected = fit(tape, cam)
    before = np.linalg.norm(cam - tape, axis=1)
    after = np.linalg.norm(corrected - tape, axis=1)
    h = args.marker_height
    H = k * h / (k - 1.0) if k > 1.0 else float("nan")

    print(f"{len(rows)} points")
    print(f"scale k = {k:.4f}   point under the camera = ({nadir[0]:.0f}, {nadir[1]:.0f}) mm")
    print(f"equivalent camera height for a {h:.0f} mm marker: {H:.0f} mm")
    print(f"camera vs tape before: mean {before.mean():.1f} mm, max {before.max():.1f} mm")
    print(f"camera vs tape after : mean {after.mean():.1f} mm, max {after.max():.1f} mm")
    for (tx, ty), e0, e1 in zip(tape, before, after):
        print(f"  tape ({tx:6.0f},{ty:6.0f})  {e0:5.1f} -> {e1:5.1f} mm")
    if not (1.0 < k < 1.2) or not math.isfinite(H):
        print("\nThe fit does not look like a raised-marker parallax (k should be a little above 1).")
        print("Check that the camera readings are uncorrected and the tape origin is the arena's bottom-left corner.")
        return 1
    if after.max() > 30:
        print("\nWARNING: a point is still more than 30 mm off after the fit. Re-measure that point before using these values.")

    if args.write:
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        data["host_overrides"].update({
            "parallax_camera_height_mm": round(H, 1),
            "parallax_marker_height_mm": h,
            "parallax_nadir_x_mm": round(float(nadir[0]), 1),
            "parallax_nadir_y_mm": round(float(nadir[1]), 1),
        })
        data["parallax_note"] = (
            f"Marker-height parallax correction fitted on {date.today().isoformat()} to {len(rows)} tape points "
            f"(scale {k:.4f}, nadir ({nadir[0]:.0f}, {nadir[1]:.0f}) mm, equivalent camera height {H:.0f} mm for "
            f"the {h:.0f} mm marker). Camera vs tape: {before.mean():.1f} mm mean before, {after.mean():.1f} mm mean "
            f"({after.max():.1f} mm max) after. Blocks are flat and need no correction."
        )
        CONFIG.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"\nWritten to {CONFIG}. Restart run_rdt.py so it picks up the new values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
