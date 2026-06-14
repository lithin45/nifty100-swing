"""Canonical filesystem paths and a ``sys.path`` bootstrap.

The repo uses a flat package layout, so the repo root must be importable.
Entry points (``scheduler/run_eod.py``, ``dashboard/app.py``, scripts) call
:func:`bootstrap_sys_path` at the very top so they work whether launched as
``python scheduler/run_eod.py`` or ``python -m scheduler.run_eod``.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = REPO_ROOT / "config"
DATA_DIR: Path = REPO_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"

SETTINGS_FILE: Path = CONFIG_DIR / "settings.yaml"
UNIVERSE_FILE: Path = CONFIG_DIR / "nifty100.csv"
DEFAULT_DB_PATH: Path = DATA_DIR / "swing.db"

BACKTEST_OUTPUT_DIR: Path = REPO_ROOT / "backtest" / "output"


def ensure_data_dir() -> Path:
    """Create the data + cache directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def bootstrap_sys_path() -> None:
    """Ensure the repo root is on ``sys.path`` (flat-layout imports)."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
