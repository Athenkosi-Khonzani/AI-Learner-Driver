"""Analyse a Rover Driving Test session: tables (CSV and LaTeX) and figures.

    python scripts/analyse_rdt.py results/<session folder>
    python scripts/analyse_rdt.py --latest
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rdt  # noqa: E402,F401
from rdt.analysis import analyse_session  # noqa: E402
from rdt.config import load_experiment_config  # noqa: E402
from rdt.session import Session  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?")
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    if args.latest or not args.session:
        sessions = Session.list_sessions()
        if not sessions:
            print("No sessions found under results/")
            return 1
        root = sessions[0]
    else:
        root = Path(args.session)
    out = analyse_session(root, load_experiment_config())
    print((out / "README.md").read_text(encoding="utf-8"))
    print(f"Tables and figures: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
