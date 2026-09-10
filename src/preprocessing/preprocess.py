"""preprocess.py — Member 1 orchestrator (raw xlsx -> clean CSVs + artifacts).

    preprocess_train(raw_path, ...) -> (train_clean_df, artifacts)
    preprocess_test(raw_path, artifacts, ...) -> test_clean_df

Order of operations (missing-flag correctness):
  1. load_raw (xlsx sheet / csv fallback)
  2. snapshot feat_missing_* flags from RAW nulls (pre-imputation)
  3. clean_data (median impute; train-only dedup)
  4. re-attach flags (train: drop flag rows matching deduped rows)
  5. engineer_features (physics + consensus; keeps pre-computed flags)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .cleaner import clean_data
from .feature_engineering import engineer_features, feature_column_list, make_missing_flags
from .loader import load_raw
from .schema import C, SHEET_TEST, SHEET_TRAIN


def _attach_flags(cleaned: pd.DataFrame, raw_flags: pd.DataFrame, raw_ids: pd.Series) -> pd.DataFrame:
    """Re-attach pre-imputation flags to a cleaned frame via Test_ID join.

    Needed because train dedup drops rows — positional concat would misalign.
    Test path is 1:1 so this is a safe no-op join.
    """
    tmp = raw_flags.copy()
    tmp[C.TEST_ID] = raw_ids.astype(str).to_numpy()
    tmp[C.TEST_ID] = tmp[C.TEST_ID].astype(str)
    out = cleaned.copy()
    # map via dict (duplicate-safe, order-preserving)
    for col in raw_flags.columns:
        mapping = dict(zip(raw_ids.astype(str).tolist(), raw_flags[col].tolist()))
        out[col] = out[C.TEST_ID].astype(str).map(mapping).fillna(0).astype(int)
    return out


def preprocess_train(
    raw_path: str | os.PathLike,
    sheet: str = SHEET_TRAIN,
    save_dir: str | os.PathLike | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = load_raw(raw_path, sheet if str(raw_path).lower().endswith((".xlsx", ".xls", ".xlsm")) else None)
    raw_ids = raw[C.TEST_ID] if C.TEST_ID in raw.columns else pd.Series(range(len(raw)))
    flags = make_missing_flags(raw)
    cleaned, artifacts = clean_data(raw, is_train=True, artifacts=None)
    cleaned = _attach_flags(cleaned, flags, raw_ids)
    featurized = engineer_features(cleaned)
    artifacts["feature_columns"] = feature_column_list(featurized)
    artifacts["train_clean_columns"] = list(featurized.columns)
    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        featurized.to_csv(out / "train_clean.csv", index=False)
        with open(out / "feature_columns.json", "w") as f:
            json.dump(artifacts["feature_columns"], f, indent=2)
    return featurized, artifacts


def preprocess_test(
    raw_path: str | os.PathLike,
    artifacts: dict[str, Any],
    sheet: str = SHEET_TEST,
    save_path: str | os.PathLike | None = None,
) -> pd.DataFrame:
    raw = load_raw(raw_path, sheet if str(raw_path).lower().endswith((".xlsx", ".xls", ".xlsm")) else None)
    raw_ids = raw[C.TEST_ID] if C.TEST_ID in raw.columns else pd.Series(range(len(raw)))
    flags = make_missing_flags(raw)
    cleaned, _ = clean_data(raw, is_train=False, artifacts=artifacts)
    cleaned = _attach_flags(cleaned, flags, raw_ids)
    featurized = engineer_features(cleaned)
    # hard guarantees
    assert len(featurized) == len(raw), "test row count changed!"
    assert list(featurized[C.TEST_ID].astype(str)) == list(raw_ids.astype(str)), "test order changed!"
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        featurized.to_csv(save_path, index=False)
    return featurized
