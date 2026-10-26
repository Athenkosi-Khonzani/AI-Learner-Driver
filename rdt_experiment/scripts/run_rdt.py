"""Start the Rover Driving Test operator application.

    python scripts/run_rdt.py                 # real rover, camera and planner
    python scripts/run_rdt.py --simulate      # rehearsal: simulated rig, scripted planner
    python scripts/run_rdt.py --simulate --real-llm
    python scripts/run_rdt.py --session results/<folder>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rdt  # noqa: E402,F401
from rdt.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
