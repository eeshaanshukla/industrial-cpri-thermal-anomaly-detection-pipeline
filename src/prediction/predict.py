"""predict.py — shipped 1-arg regressor (Member 3 -> Member 4)."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parents[0]))
try:
    from config import MODEL_DIR as _CFG_MODEL_DIR  # noqa: E402
    DEFAULT_MODEL = Path(_CFG_MODEL_DIR) / "rp_model.pkl"
except ImportError:
    DEFAULT_MODEL = SRC.parents[0] / "models" / "rp_model.pkl"

from preprocessing.schema import C  # noqa: E402


def _load_bundle(model_path: str | Path | None):
    p = Path(model_path) if model_path else DEFAULT_MODEL
    if not p.exists():
        raise FileNotFoundError(f"RP bundle not found: {p}. Run run_member3.py first.")
    return joblib.load(p)


def predict_rp(df: pd.DataFrame, model_path: str | Path | None = None) -> pd.DataFrame:
    """Predict Reference_Parameter for EVERY input row (all rows, any verdict).

    Returns DataFrame[Test_ID, Predicted_Reference_Parameter] in INPUT order.
    Clipped at 0 (physical floor); asserts finite — NaNs fail loudly.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"predict_rp expects DataFrame, got {type(df)}")
    bundle = _load_bundle(model_path)
    feats: list[str] = bundle["feature_columns"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"predict_rp missing features: {missing[:5]}...")
    preds = np.asarray(bundle["model"].predict(df[feats]), dtype=float)
    preds = np.clip(preds, 0.0, None)  # physical floor at the source (M3 owns this)
    assert np.all(np.isfinite(preds)), "NON-FINITE RP predictions — failing loudly"
    return pd.DataFrame({
        C.TEST_ID: df[C.TEST_ID].astype(str).to_numpy(),
        "Predicted_Reference_Parameter": preds,
    })
