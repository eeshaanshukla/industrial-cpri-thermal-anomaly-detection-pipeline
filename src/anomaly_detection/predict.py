"""predict.py — shipped 1-arg predictor (Member 2 -> Member 4)."""

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
    ROOT = SRC.parents[0]
    DEFAULT_MODEL = Path(_CFG_MODEL_DIR) / "anomaly_model.pkl"
except ImportError:
    ROOT = SRC.parents[0]
    DEFAULT_MODEL = ROOT / "models" / "anomaly_model.pkl"

from preprocessing.schema import C  # noqa: E402

from .rules import compute_flags, iso_feature_frame  # noqa: E402

POS, NEG = "Invalid", "Valid"


def _load_bundle(model_path: str | Path | None):
    p = Path(model_path) if model_path else DEFAULT_MODEL
    if not p.exists():
        raise FileNotFoundError(
            f"Anomaly bundle not found: {p}. Run run_member2.py first.")
    return joblib.load(p)


def predict_validity(df: pd.DataFrame, model_path: str | Path | None = None) -> pd.DataFrame:
    """Predict validity for a CLEAN frame (train_clean or test_clean schema).

    Returns DataFrame[Test_ID, Predicted_Validity, Anomaly_Score, Reason_Flags]
    in INPUT order. Anomaly_Score = P(Invalid) in [0, 1].
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"predict_validity expects DataFrame, got {type(df)}")
    bundle = _load_bundle(model_path)
    feats: list[str] = bundle["feature_columns"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"predict_validity missing features: {missing[:5]}...")
    X = df[feats]
    clf = bundle["model"]
    classes = list(clf.classes_)
    scores = clf.predict_proba(X)[:, classes.index(POS)]
    thr = float(bundle["threshold"])
    iso = bundle["iso_forest"]
    iso_pred = iso.predict(iso_feature_frame(df))
    flags = compute_flags(df, bundle["flag_thresholds"], pd.Series(scores, index=df.index),
                          thr, pd.Series(iso_pred == -1, index=df.index))
    return pd.DataFrame({
        C.TEST_ID: df[C.TEST_ID].astype(str).to_numpy(),
        "Predicted_Validity": np.where(scores >= thr, POS, NEG),
        "Anomaly_Score": np.asarray(scores, dtype=float),
        "Reason_Flags": flags.to_numpy(),
    })
