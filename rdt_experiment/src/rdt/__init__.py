"""Experiment 2 (Rover Driving Test) harness for the AI Learner Rover Driver.

The package wraps, and never copies, the production host (``ai_learner_host``)
and the Experiment 1 simulator (``llm_test``). See ``rdt._bootstrap``.
"""

from rdt._bootstrap import ensure_dependency_paths

ensure_dependency_paths()

__version__ = "1.0.0"
