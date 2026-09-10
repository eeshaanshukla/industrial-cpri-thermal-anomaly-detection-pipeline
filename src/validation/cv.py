"""cv.py — shared CV splitters (Member 4 owns; Members 2/3 call these).

Same seed + fold count everywhere for comparability; different splitter
classes by task design (stratified for classification, plain for regression).
"""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from config import CV_FOLDS, RANDOM_STATE  # noqa: E402
except ImportError:  # standalone copy-paste runs
    CV_FOLDS, RANDOM_STATE = 5, 42


def make_classifier_splits(n_splits: int = CV_FOLDS, seed: int = RANDOM_STATE) -> StratifiedKFold:
    """Member 2's splitter: stratified (Invalid ≈13% — plain KFold is unstable)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_regression_splits(n_splits: int = CV_FOLDS, seed: int = RANDOM_STATE) -> KFold:
    """Member 3's splitter: plain KFold (regression targets, no strata)."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
