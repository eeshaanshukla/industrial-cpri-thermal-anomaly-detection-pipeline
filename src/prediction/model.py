"""model.py — RP regressor candidates + 1-arg trainer (Member 3).

Candidates (same KFold splits, seed 42 from config):
  linear — LinearRegression on raw inputs (interpretable floor)
  rf     — RandomForestRegressor
  hgb    — HistGradientBoostingRegressor
Winner = lowest OOF RMSE on the winning population.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parents[0]))
try:
    from config import ARTIFACTS_DIR, CV_FOLDS, RANDOM_STATE  # noqa: E402
except ImportError:
    ARTIFACTS_DIR, CV_FOLDS, RANDOM_STATE = Path("artifacts"), 5, 42

from preprocessing.schema import C, FORBIDDEN_FEATURES, NUMERIC_INPUT_COLS  # noqa: E402
from validation.cv import make_regression_splits  # noqa: E402
from validation.metrics import regression_metrics  # noqa: E402

N_SPLITS = CV_FOLDS

#: S4 family for the ablation (raw S4 + S4-probe feats). Decoy hypothesis:
#: corr(S4, RP) ~ 0 — test with-vs-without, don't assume.
S4_FAMILY: list[str] = ["feat_s4_abs_dev", "feat_s4_minus_smean", "feat_s4_ratio"]


def s4_family(feats: list[str]) -> list[str]:
    return [c for c in [C.S4, *S4_FAMILY] if c in feats]


def load_feature_columns() -> list[str]:
    for cand in (ARTIFACTS_DIR / "feature_columns.json", SRC.parents[0] / "feature_columns.json"):
        if Path(cand).exists():
            feats = json.load(open(cand))
            assert not (FORBIDDEN_FEATURES & set(feats)), f"forbidden leak: {FORBIDDEN_FEATURES & set(feats)}"
            return feats
    raise FileNotFoundError("feature_columns.json not found (run run_member1.py first).")


def candidates(feats: list[str]) -> list[tuple[str, object, list[str]]]:
    """(name, estimator, features) — linear uses raw inputs only (clean floor)."""
    raw_only = [c for c in NUMERIC_INPUT_COLS if c in feats]
    full = list(feats)
    return [
        ("linear", LinearRegression(), raw_only),
        ("rf", RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                     random_state=RANDOM_STATE, n_jobs=-1), full),
        ("hgb", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                              random_state=RANDOM_STATE), full),
    ]


def oof_predict(clf, X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, list[dict]]:
    """Out-of-fold predictions + per-fold reports on FIXED team splits."""
    skf = make_regression_splits(N_SPLITS, RANDOM_STATE)
    oof = np.zeros(len(X))
    reports = []
    for tr, va in skf.split(X, y):
        c = clf.__class__(**clf.get_params())
        c.fit(X.iloc[tr], y.iloc[tr])
        p = np.asarray(c.predict(X.iloc[va]), dtype=float)
        oof[va] = p
        reports.append(regression_metrics(y.iloc[va].to_numpy(), p))
    return oof, reports


def feature_importances(model, feats: list[str], X=None, y=None, top_k: int = 15,
                        random_state: int = RANDOM_STATE) -> list[dict]:
    """Native importances, else |coef|, else deterministic permutation fallback."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    elif X is not None and y is not None:
        from sklearn.inspection import permutation_importance

        r = permutation_importance(model, X, y, n_repeats=5,
                                   random_state=random_state, n_jobs=-1)
        imp = np.clip(np.asarray(r.importances_mean, dtype=float), 0, None)
    else:
        return []
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    order = np.argsort(imp)[::-1][:top_k]
    return [{"feature": feats[i], "importance": float(imp[i])} for i in order]


def train_rp_model(train_df_valid_only: pd.DataFrame) -> tuple[object, dict]:
    """Role-doc function contract: 1-arg train -> (fitted_model, info_dict).

    Trains the HGB-vs-RF-vs-linear comparison winner on Valid rows.
    (Full experiment orchestration lives in train.py; this is the callable atom.)
    """
    from .train import compare_models  # deferred: same package, avoids circulars

    feats = load_feature_columns()
    X = train_df_valid_only[feats].copy()
    y = pd.to_numeric(train_df_valid_only[C.RP], errors="coerce").astype(float)
    comp = compare_models(X, y, feats)
    winner = comp["winner"]
    clf = dict((n, c) for n, c, _ in candidates(feats))[winner]
    model = clf.__class__(**clf.get_params())
    model.fit(X, y)
    info = {"model_name": winner, "features": feats,
            "oof_rmse": comp["results"][winner]["oof"]["rmse"]}
    return model, info
