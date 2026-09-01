"""Command-line entry point for the trajectory-aligned FM trainer.

The implementation lives in :mod:`fm_training` so trajectory construction,
negative-sampling tests, and ranker-only evaluation can import it without
executing the command-line path. Public names are re-exported for compatibility
with repository tests and analysis utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT))

from fm_training import *  # noqa: F401,F403,E402
from fm_training import main  # noqa: E402


if __name__ == "__main__":
    main()
