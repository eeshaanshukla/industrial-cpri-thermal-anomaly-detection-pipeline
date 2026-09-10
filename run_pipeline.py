"""run_pipeline.py — THE single entry point (Member 4).

    python run_pipeline.py

Regenerates EVERYTHING from the raw xlsx: clean data, trained models,
<TeamName>.csv, summary.json. Zero manual steps. Deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from pipeline.orchestrator import run_full_pipeline  # noqa: E402

if __name__ == "__main__":
    print(f"PowerNext-AI pipeline | team={config.TEAM_NAME} seed={config.RANDOM_STATE}")
    run_full_pipeline(config)
    print("PIPELINE DONE.")
