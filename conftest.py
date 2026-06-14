"""Pytest bootstrap.

The repository uses a *flat* package layout (``config/``, ``analyzers/`` … are
top-level packages). Placing this ``conftest.py`` at the repo root makes pytest
insert the root onto ``sys.path`` so ``import analyzers.technical`` etc. resolve
without installing the project.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
