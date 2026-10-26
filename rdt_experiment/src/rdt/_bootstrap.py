"""Locate the two sibling projects this harness depends on.

Expected layout (all three folders side by side)::

    <parent>/ai_learner_host/src/ai_learner_host
    <parent>/llm_tests/src/llm_test
    <parent>/rdt_experiment/src/rdt

If either project is pip-installed, the installed copy is used. Otherwise the
sibling ``src`` folder is added to ``sys.path``. ``RDT_HOST_SRC`` and
``RDT_LLM_TEST_SRC`` override the sibling locations.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # .../rdt_experiment
WORKSPACE_ROOT = PACKAGE_ROOT.parent


def _add(module: str, env_var: str, default: Path) -> None:
    if importlib.util.find_spec(module) is not None:
        return
    candidate = Path(os.environ.get(env_var, default))
    if not (candidate / module).is_dir():
        raise ImportError(
            f"Cannot find '{module}'. Expected it at {candidate / module} "
            f"(set {env_var} to the folder that contains '{module}')."
        )
    sys.path.insert(0, str(candidate))


def ensure_dependency_paths() -> None:
    _add("ai_learner_host", "RDT_HOST_SRC", WORKSPACE_ROOT / "ai_learner_host" / "src")
    _add("llm_test", "RDT_LLM_TEST_SRC", WORKSPACE_ROOT / "llm_tests" / "src")


def host_project_root() -> Path:
    import ai_learner_host

    return Path(ai_learner_host.__file__).resolve().parents[2]
