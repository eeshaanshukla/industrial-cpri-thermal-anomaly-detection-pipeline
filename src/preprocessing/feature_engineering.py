"""feature_engineering.py — physically-motivated features (Member 1).

All raw-column references via schema constants (C.*). Engineered names use the
`feat_` prefix. Missing-indicator flags (`feat_missing_*`) are 0/1 and computed
PRE-imputation — the orchestrator (preprocess.py) snapshots nulls from the RAW
frame first, then imputes, then calls engineer_features() on imputed data with
the pre-computed flags attached. Calling engineer_features() standalone on a
raw frame with NaNs also works (flags reflect current NaNs).

No Test_ID / Reference_Parameter / Validity_Label in outputs-as-features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import C, SENSOR_COLS

EPS = 1e-6

MISSING_FLAG_COLS: list[str] = ["feat_missing_count"] + [f"feat_missing_{s}" for s in ("S1", "S2", "S3", "S4")]

# Map flag suffix -> schema sensor constant (no literals elsewhere)
_FLAG_TO_SENSOR: dict[str, str] = {
    "feat_missing_S1": C.S1,
    "feat_missing_S2": C.S2,
    "feat_missing_S3": C.S3,
    "feat_missing_S4": C.S4,
}


def make_missing_flags(raw_df: pd.DataFrame) -> pd.DataFrame:
    """0/1 missing-indicator flags from a PRE-imputation frame (deterministic)."""
    flags = pd.DataFrame(index=raw_df.index)
    count = np.zeros(len(raw_df), dtype=np.int64)
    for flag, sensor in _FLAG_TO_SENSOR.items():
        if sensor in raw_df.columns:
            col_flag = pd.to_numeric(raw_df[sensor], errors="coerce").isna().to_numpy(dtype=np.int64)
        else:
            col_flag = np.zeros(len(raw_df), dtype=np.int64)
        flags[flag] = col_flag
        count = count + col_flag
    flags["feat_missing_count"] = count
    return flags[MISSING_FLAG_COLS]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feat_* columns. Idempotent-ish: existing feat_ cols are refreshed.

    Expected input: imputed frame (no NaNs in model inputs) WITH pre-computed
    feat_missing_* already attached by the orchestrator. If flags are absent
    (standalone call on raw data), they are computed from current NaNs.
    """
    out = df.copy()

    # ---- missing flags (pre-imputation semantics) ----
    if not all(c in out.columns for c in MISSING_FLAG_COLS):
        flags = make_missing_flags(out)
        for c in MISSING_FLAG_COLS:
            out[c] = flags[c].to_numpy()

    volt = _num(out, C.VOLT)
    curr = _num(out, C.CURR)
    amb = _num(out, C.AMB)
    dur = _num(out, C.DUR)
    s1 = _num(out, C.S1)
    s2 = _num(out, C.S2)
    s3 = _num(out, C.S3)
    s4 = _num(out, C.S4)

    # ---- electrical / thermal-load proxies ----
    out["feat_power_input"] = volt * curr                      # V * I
    out["feat_energy_input"] = volt * curr * dur               # V * I * t
    out["feat_i2t"] = curr**2 * dur                            # Joule-heating proxy
    out["feat_i2t_per_volt"] = (curr**2 * dur) / (volt.abs() + EPS)
    out["feat_volt_x_dur"] = volt * dur
    out["feat_curr_x_dur"] = curr * dur
    out["feat_curr_per_volt"] = curr / (volt.abs() + EPS)      # conductance-like

    # ---- sensor consensus (S1..S3 are the trusted thermal trio; S4 auxiliary) ----
    trio = pd.concat([s1, s2, s3], axis=1)
    out["feat_s_mean"] = trio.mean(axis=1)
    out["feat_s_std"] = trio.std(axis=1, ddof=0)
    out["feat_s_max"] = trio.max(axis=1)
    out["feat_s_min"] = trio.min(axis=1)
    out["feat_s_range"] = out["feat_s_max"] - out["feat_s_min"]
    out["feat_s_median"] = trio.median(axis=1)

    # Pairwise gaps (signed + absolute) — sensor-disagreement detectors
    out["feat_s1_s2_gap"] = s1 - s2
    out["feat_s1_s3_gap"] = s1 - s3
    out["feat_s2_s3_gap"] = s2 - s3
    out["feat_abs_s1_s2_gap"] = (s1 - s2).abs()
    out["feat_abs_s1_s3_gap"] = (s1 - s3).abs()
    out["feat_abs_s2_s3_gap"] = (s2 - s3).abs()

    # Consistency ratio: spread relative to level (regime-aware, scale-free)
    out["feat_s_consistency"] = out["feat_s_range"] / (out["feat_s_mean"].abs() + EPS)
    out["feat_s_cv"] = out["feat_s_std"] / (out["feat_s_mean"].abs() + EPS)

    # Per-sensor deviation from trio mean (which sensor disagrees?)
    out["feat_s1_dev"] = s1 - out["feat_s_mean"]
    out["feat_s2_dev"] = s2 - out["feat_s_mean"]
    out["feat_s3_dev"] = s3 - out["feat_s_mean"]

    # ---- S4 relevance probes (S4's usefulness is unproven — quantify, don't assume) ----
    out["feat_s4_minus_smean"] = s4 - out["feat_s_mean"]
    out["feat_s4_abs_dev"] = (s4 - out["feat_s_mean"]).abs()
    out["feat_s4_ratio"] = s4 / (out["feat_s_mean"].abs() + EPS)

    # ---- thermal efficiency: rise per unit input (regime-change lens) ----
    out["feat_temp_per_power"] = out["feat_s_mean"] / (out["feat_power_input"].abs() + EPS)
    out["feat_temp_per_energy"] = out["feat_s_mean"] / (out["feat_energy_input"].abs() + EPS)
    out["feat_temp_per_i2t"] = out["feat_s_mean"] / (out["feat_i2t"].abs() + EPS)

    # ---- absolute-temperature views (rise + ambient) ----
    out["feat_smean_abs"] = out["feat_s_mean"] + amb
    out["feat_smax_abs"] = out["feat_s_max"] + amb

    # ---- guards: engineered cols must be finite post-imputation ----
    feat_cols = [c for c in out.columns if c.startswith("feat_")]
    for c in feat_cols:
        col = pd.to_numeric(out[c], errors="coerce")
        col = col.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[c] = col

    # Missing flags back to strict ints: per-sensor 0/1, count 0..4
    for c in MISSING_FLAG_COLS:
        hi = 4 if c == "feat_missing_count" else 1
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).clip(0, hi).astype(np.int64)

    return out


def feature_column_list(df: pd.DataFrame) -> list[str]:
    """Ordered model-input list: raw numeric inputs + feat_* (excludes ID/targets)."""
    from .schema import FORBIDDEN_FEATURES, NUMERIC_INPUT_COLS

    ordered = [c for c in NUMERIC_INPUT_COLS if c in df.columns]
    ordered += sorted(c for c in df.columns if c.startswith("feat_"))
    assert not (FORBIDDEN_FEATURES & set(ordered)), "Forbidden col leaked into features!"
    return ordered
