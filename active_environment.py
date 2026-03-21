"""
Resolve embodied environment: car (hardware) vs game (sim).

Order of precedence:
  1. EXPERIENCE_ENGINE_ACTIVE_ENV (e.g. set in main.ipynb before importing agent)
  2. ACTIVE_ENVIRONMENT= in select_environment.config
  3. game_environment — default without robot hardware (avoids hanging on car init)
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_CONFIG_FILE = _REPO_ROOT / "select_environment.config"
_VALID = frozenset({"car_environment", "game_environment"})


def get_active_environment_name() -> str:
    override = os.environ.get("EXPERIENCE_ENGINE_ACTIVE_ENV", "").strip()
    if override in _VALID:
        return override

    if not _CONFIG_FILE.is_file():
        return "game_environment"

    try:
        text = _CONFIG_FILE.read_text(encoding="utf-8")
    except OSError:
        return "game_environment"

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("ACTIVE_ENVIRONMENT="):
            val = line.split("=", 1)[1].strip()
            if val in _VALID:
                return val

    return "game_environment"
