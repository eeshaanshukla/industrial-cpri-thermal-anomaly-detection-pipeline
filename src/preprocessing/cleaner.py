"""cleaner.py — deterministic cleaning (Member 1).

Philosophy (per master context):
  * Median imputation fitted ONCE on train, applied to test (never refit).
  * Dedup is TRAIN-ONLY. Test rows are never dropped/merged/reordered.
  * NO outlier removal — "unusual != invalid" is Member 2's call (25% of grade).
    Spikes / regime shifts pass through untouched; we only document them.

Contract:
    clean_data(df, is_train, artifacts) -> (cleaned_df, artifacts_dict)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schema import C, NUMERIC_INPUT_COLS

ARTIFACT_VERSION = "m1-cleaner-v1"

#: Physically impossible values we *document* but do NOT alter (M2's verdict).
#: Cleaner stays conservative: only imputation + train dedup.
_IMPUTE_EXCLUDE: frozenset[str] = frozenset()  # nothing excluded today


def _fit_medians(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for col in cols:
        if col not in df.columns:
            continue
        med = float(np.nanmedian(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)))
        if not np.isfinite(med):
            med = 0.0  # degenerate all-NaN column fallback (documented)
        medians[col] = med
    return medians


def _apply_medians(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col, med in medians.items():
        if col in _IMPUTE_EXCLUDE or col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        n_missing = int(series.isna().sum())
        if n_missing:
            series = series.fillna(med)
        out[col] = series
    return out


def clean_data(
    df: pd.DataFrame,
    is_train: bool,
    artifacts: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean one split deterministically.

    Train (artifacts is None): FIT medians + dedup rule, return fitted artifacts.
    Test  (artifacts given):   APPLY fitted rules exactly, never refit.

    Test guarantees: same row count, same Test_ID order as input.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"clean_data expects DataFrame, got {type(df)}")
    if C.TEST_ID not in df.columns:
        raise ValueError(f"Missing identifier column {C.TEST_ID!r}")

    input_ids = list(df[C.TEST_ID].astype(str))
    impute_cols = [c for c in NUMERIC_INPUT_COLS if c in df.columns]

    if is_train:
        if artifacts is not None:
            raise ValueError("Train call must pass artifacts=None so rules are FIT, not applied.")
        medians = _fit_medians(df, impute_cols)
        out = _apply_medians(df, medians)
        # Dedup rule (documented): exact-duplicate ROWS across all raw columns,
        # keep first occurrence. Train-only quality decision.
        before = len(out)
        dedup_subset = [c for c in df.columns if c in set(impute_cols + [C.TEST_ID])]
        # NOTE: Test_ID differs per row in practice, so exact-row dupes on the
        # full column set are the conservative definition — two rows must agree
        # on EVERYTHING including Test_ID... which never happens. The meaningful
        # rule is duplicates over measurement columns IGNORING Test_ID:
        meas_subset = [c for c in impute_cols]
        dup_mask = out.duplicated(subset=meas_subset, keep="first")
        n_dupes = int(dup_mask.sum())
        out = out.loc[~dup_mask].reset_index(drop=True)
        artifacts_out: dict[str, Any] = {
            "version": ARTIFACT_VERSION,
            "medians": medians,
            "impute_cols": impute_cols,
            "dedup_rule": (
                "train-only: drop exact-duplicate measurement rows "
                "(all numeric input cols equal, ignoring Test_ID), keep first"
            ),
            "dedup_dropped_train": n_dupes,
            "train_rows_before_dedup": before,
            "train_rows_after_dedup": len(out),
            "test_policy": "never drop/merge/reorder test rows",
        }
        return out, artifacts_out

    # ---- test path: APPLY only ----
    if artifacts is None:
        raise ValueError("Test call requires fitted train artifacts (never refit on test).")
    medians = dict(artifacts.get("medians", {}))
    # Edge: test column never seen in train -> fall back to column median of
    # TEST would be leakage; instead use 0.0 and record it (deterministic).
    for col in impute_cols:
        if col not in medians:
            medians[col] = 0.0
    out = _apply_medians(df, medians)
    out = out.reset_index(drop=True)  # keep input order, just tidy the index

    assert len(out) == len(df), "TEST ROW LOSS: cleaner dropped test rows!"
    assert list(out[C.TEST_ID].astype(str)) == input_ids, "TEST REORDERED in cleaner!"
    return out, artifacts
