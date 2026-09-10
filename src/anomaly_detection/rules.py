"""rules.py — interpretable flag logic + rule baseline (Member 2).

Flags are deterministic diagnostics (Member 4 drops them before submission).
Thresholds are DATA-FITTED percentiles on Valid train rows (saved to
anomaly_thresholds.json) — never hand-picked magic numbers.

Flag vocabulary (canonical order):
  MODEL_INVALID, TRIO_DISAGREE, SENSOR_DEV, MISSING_HEAVY, MISSING_SENSOR,
  ISO_ANOMALY, DUP_MEASUREMENTS, else CONSISTENT
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.schema import C, NUMERIC_INPUT_COLS  # noqa: E402

CLEAN_FLAG = "CONSISTENT"

ISO_FEATURES: list[str] = [
    "feat_s_range",
    "feat_s_consistency",
    "feat_s_cv",
    "feat_abs_s1_s2_gap",
    "feat_abs_s1_s3_gap",
    "feat_abs_s2_s3_gap",
]


def iso_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Disagreement-focused frame for the auxiliary IsolationForest.

    Shared by train and predict so the transform is identical both sides.
    """
    out = pd.DataFrame(index=df.index)
    for c in ISO_FEATURES:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df.columns else 0.0
    for c in ("feat_s1_dev", "feat_s2_dev", "feat_s3_dev"):
        acol = f"abs_{c}"
        out[acol] = pd.to_numeric(df[c], errors="coerce").abs().fillna(0.0) if c in df.columns else 0.0
    out["feat_missing_count"] = (
        pd.to_numeric(df["feat_missing_count"], errors="coerce").fillna(0) if "feat_missing_count" in df.columns else 0
    )
    return out


def fit_flag_thresholds(valid_df: pd.DataFrame, pct: float = 99.0) -> dict[str, float]:
    """Fit flag cutoffs as percentiles over KNOWN-VALID train rows."""
    def _pct(col: str) -> float:
        return float(np.nanpercentile(pd.to_numeric(valid_df[col], errors="coerce").to_numpy(dtype=float), pct))

    devs = pd.concat(
        [pd.to_numeric(valid_df[c], errors="coerce").abs() for c in ("feat_s1_dev", "feat_s2_dev", "feat_s3_dev")],
        axis=1,
    ).max(axis=1)
    return {
        "percentile": pct,
        "fitted_on": "Validity_Label==Valid train rows",
        "trio_range": _pct("feat_s_range"),
        "max_abs_dev": float(np.nanpercentile(devs.to_numpy(dtype=float), pct)),
        "consistency": _pct("feat_s_consistency"),
        "missing_heavy_at": 2,
    }


def _dup_mask(df: pd.DataFrame) -> pd.Series:
    meas = [c for c in NUMERIC_INPUT_COLS if c in df.columns]
    if not meas:
        return pd.Series(False, index=df.index)
    return df.duplicated(subset=meas, keep=False)


def compute_flags(
    df: pd.DataFrame,
    thresholds: dict,
    scores: pd.Series | None = None,
    threshold: float | None = None,
    iso_mask: pd.Series | None = None,
) -> pd.Series:
    """Vectorised flag computation. Returns pipe-joined flag strings."""
    n = len(df)
    trio_range = pd.to_numeric(df["feat_s_range"], errors="coerce").fillna(0)
    consistency = pd.to_numeric(df["feat_s_consistency"], errors="coerce").fillna(0)
    max_dev = pd.concat(
        [pd.to_numeric(df[c], errors="coerce").abs().fillna(0) for c in ("feat_s1_dev", "feat_s2_dev", "feat_s3_dev")],
        axis=1,
    ).max(axis=1)
    miss = pd.to_numeric(df["feat_missing_count"], errors="coerce").fillna(0).astype(int)

    conds: list[tuple[str, pd.Series]] = []
    if scores is not None and threshold is not None:
        conds.append(("MODEL_INVALID", pd.Series(np.asarray(scores) >= threshold, index=df.index)))
    conds += [
        ("TRIO_DISAGREE", trio_range > thresholds["trio_range"]),
        ("SENSOR_DEV", max_dev > thresholds["max_abs_dev"]),
        ("MISSING_HEAVY", miss >= int(thresholds.get("missing_heavy_at", 2))),
        ("MISSING_SENSOR", miss >= 1),
        ("ISO_ANOMALY", iso_mask if iso_mask is not None else pd.Series(False, index=df.index)),
        ("DUP_MEASUREMENTS", _dup_mask(df)),
    ]
    # MISSING_SENSOR subsumed when HEAVY present — drop the milder one there
    flags: list[str] = []
    for i in range(n):
        active = [name for name, m in conds if bool(m.iloc[i])]
        if "MISSING_HEAVY" in active and "MISSING_SENSOR" in active:
            active.remove("MISSING_SENSOR")
        flags.append("|".join(active) if active else CLEAN_FLAG)
    return pd.Series(flags, index=df.index)


def baseline_predict(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Scaffolding rule baseline: Invalid iff strong disagreement/missingness.

    Reference point only — the shipped predictor is the supervised model.
    """
    flags = compute_flags(df, thresholds)
    pred = np.where(
        flags.str.contains("TRIO_DISAGREE|SENSOR_DEV|MISSING_HEAVY"),
        "Invalid",
        "Valid",
    )
    score = np.where(pred == "Invalid", 0.75, 0.25)  # fixed scaffolding scores
    return pd.DataFrame({
        C.TEST_ID: df[C.TEST_ID].astype(str).to_numpy(),
        "Predicted_Validity": pred,
        "Anomaly_Score": score,
        "Reason_Flags": flags.to_numpy(),
    })
