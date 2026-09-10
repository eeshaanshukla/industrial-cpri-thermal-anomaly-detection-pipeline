"""config.py — OWNER: MEMBER 4 (everyone else READS, never writes).

Single source of truth for seed, flags, team name, and paths.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- team/setup
#: Official team name (file-safe).
#: Drives outputs/<TeamName>.csv.
TEAM_NAME = "5guys1repo"

#: Global seed — imported and applied (numpy + sklearn) in every module.
RANDOM_STATE = 42

#: False during scaffolding, True once supervised M2/M3 models land.
#: NEVER submit while False.
FINAL_MODELS_READY = True

#: Cross-validation folds (M2 = StratifiedKFold, M3 = KFold, same seed/count).
CV_FOLDS = 5

# ------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUTS_DIR = ROOT / "outputs"
MOCKS_DIR = ROOT / "mocks"

#: Official participant workbook (sheet loader has a .csv fallback branch).
WORKBOOK_NAME = "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
SHEET_TRAIN = "Training_Data"
SHEET_TEST = "Test_Data"
