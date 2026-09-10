"""train.py — M3 experiment orchestration (Member 3).

1. Model comparison on Valid-only (linear vs rf vs hgb, same KFold) -> winner.
2. Population experiment on winner: A (Valid-only) vs B (all rows); BOTH
   validated on Valid-only held-out folds (reliable ground truth). Default A
   unless B wins clearly (>=2% RMSE gain AND no worse MAE).
3. S4 ablation on winner+population: with vs without S4 family. Simpler wins
   ties (drop S4 unless it strictly helps RMSE).
4. Edge-robustness proxy: OOF RMSE on edge slices vs overall.
5. Final refit on full winning population; bundle -> models/rp_model.pkl.
"""

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
    from config import RANDOM_STATE  # noqa: E402
except ImportError:
    RANDOM_STATE = 42

from preprocessing.schema import C, FORBIDDEN_FEATURES  # noqa: E402
from validation.cv import make_regression_splits  # noqa: E402
from validation.metrics import regression_metrics  # noqa: E402

from .model import N_SPLITS, candidates, feature_importances, oof_predict, s4_family  # noqa: E402


def compare_models(X: pd.DataFrame, y: pd.Series, feats: list[str]) -> dict:
    results: dict[str, dict] = {}
    for name, clf, use_feats in candidates(feats):
        cols = [c for c in use_feats if c in X.columns]
        oof, reports = oof_predict(clf, X[cols], y)
        results[name] = {
            "features_used": cols, "n_features": len(cols),
            "oof": regression_metrics(y.to_numpy(), oof),
            "fold_mean_std": _agg(reports),
        }
    winner = min(results, key=lambda k: results[k]["oof"]["rmse"])
    return {"results": results, "winner": winner}


def _agg(reports: list[dict]) -> dict:
    return {k: {"mean": float(np.mean([r[k] for r in reports])),
                "std": float(np.std([r[k] for r in reports]))}
            for k in ("mae", "rmse", "r2")}


def population_experiment(train_clean: pd.DataFrame, feats: list[str], winner: str) -> dict:
    """A (Valid-only) vs B (all rows), same folds, Valid-only held-out scoring."""
    clf0 = dict((n, c) for n, c, _ in candidates(feats))[winner]
    cols = [c for c in dict((n, f) for n, _, f in candidates(feats))[winner] if c in train_clean.columns]
    X = train_clean[cols]
    y_all = pd.to_numeric(train_clean[C.RP], errors="coerce").astype(float)
    is_valid = (train_clean[C.VALID].astype(str) == "Valid").to_numpy()
    skf = make_regression_splits(N_SPLITS, RANDOM_STATE)
    oof_a = np.full(len(X), np.nan)
    oof_b = np.full(len(X), np.nan)
    for tr, va in skf.split(X, y_all):
        va_valid = va[is_valid[va]]  # held-out scoring: Valid rows only, same set
        tr_a = tr[is_valid[tr]]
        ca = clf0.__class__(**clf0.get_params())
        ca.fit(X.iloc[tr_a], y_all.iloc[tr_a])
        oof_a[va_valid] = ca.predict(X.iloc[va_valid])
        cb = clf0.__class__(**clf0.get_params())
        cb.fit(X.iloc[tr], y_all.iloc[tr])
        oof_b[va_valid] = cb.predict(X.iloc[va_valid])
    mask = is_valid  # both OOF arrays defined exactly on Valid rows
    m_a = regression_metrics(y_all.to_numpy()[mask], oof_a[mask])
    m_b = regression_metrics(y_all.to_numpy()[mask], oof_b[mask])
    b_wins = (m_b["rmse"] < m_a["rmse"] * 0.98) and (m_b["mae"] <= m_a["mae"])
    return {
        "rule": "B wins iff RMSE_B < 0.98*RMSE_A AND MAE_B <= MAE_A; else default A (Valid-only)",
        "A_valid_only": m_a, "B_all_rows": m_b,
        "chosen_population": "B_all_rows" if b_wins else "A_valid_only",
    }


def s4_ablation(X: pd.DataFrame, y: pd.Series, feats: list[str], winner: str) -> dict:
    clf0 = dict((n, c) for n, c, _ in candidates(feats))[winner]
    fam = s4_family(feats)
    cols_full = [c for c in feats if c in X.columns]
    cols_no_s4 = [c for c in cols_full if c not in fam]
    oof_full, _ = oof_predict(clf0, X[cols_full], y)
    oof_no, _ = oof_predict(clf0, X[cols_no_s4], y)
    m_full = regression_metrics(y.to_numpy(), oof_full)
    m_no = regression_metrics(y.to_numpy(), oof_no)
    drop = m_no["rmse"] <= m_full["rmse"]  # simpler wins ties
    return {
        "rule": "drop S4 family unless it strictly improves OOF RMSE (simpler wins ties)",
        "s4_family": fam,
        "with_s4": m_full, "without_s4": m_no,
        "decision": "drop_s4" if drop else "keep_s4",
    }


def edge_check(train_clean: pd.DataFrame, feats: list[str], winner: str,
               oof: np.ndarray, y: pd.Series) -> dict:
    """OOF error on edge slices (robustness proxy for the unseen 2nd dataset)."""
    out: dict[str, dict] = {"overall_oof": regression_metrics(y.to_numpy(), oof)}
    curr = pd.to_numeric(train_clean[C.CURR], errors="coerce")
    volt = pd.to_numeric(train_clean[C.VOLT], errors="coerce")
    slices = {
        "edge_high_current": curr >= curr.quantile(0.95),
        "edge_high_voltage": volt >= volt.quantile(0.95),
        "edge_long_duration": pd.to_numeric(train_clean[C.DUR], errors="coerce") >= 120,
    }
    for name, mask in slices.items():
        m = mask.to_numpy() & y.index.isin(y.index)
        idx = np.where(mask.to_numpy())[0]
        idx = idx[np.isin(train_clean.index.to_numpy()[idx], y.index.to_numpy())]
        if len(idx) >= 5:
            # map clean-frame positions -> y positions
            pos = train_clean.index.get_indexer_for(y.index)
            inv = {p: i for i, p in enumerate(pos)}
            take = np.array([inv[i] for i in idx if i in inv])
            out[name] = {"n": int(len(take)),
                         **regression_metrics(y.to_numpy()[take], oof[take])}
        else:
            out[name] = {"n": int(len(idx)), "note": "too few rows — skipped"}
    return out


def build_bundle(train_clean: pd.DataFrame, feature_columns: list[str]) -> tuple[dict, dict]:
    assert not (FORBIDDEN_FEATURES & set(feature_columns)), "forbidden col in features!"
    feats = list(feature_columns)
    valid_mask = train_clean[C.VALID].astype(str) == "Valid"
    yv = pd.to_numeric(train_clean.loc[valid_mask, C.RP], errors="coerce").astype(float)
    Xv = train_clean.loc[valid_mask, feats]

    comp = compare_models(Xv, yv, feats)
    winner = comp["winner"]
    pop = population_experiment(train_clean, feats, winner)
    use_valid_only = pop["chosen_population"] == "A_valid_only"

    y_fit = yv if use_valid_only else pd.to_numeric(train_clean[C.RP], errors="coerce").astype(float)
    X_fit = Xv if use_valid_only else train_clean[feats]
    abl = s4_ablation(X_fit, y_fit, feats, winner)
    final_cols = ([c for c in feats if c not in abl["s4_family"]]
                  if abl["decision"] == "drop_s4" else list(feats))

    clf0 = dict((n, c) for n, c, _ in candidates(feats))[winner]
    oof_final, fold_reports = oof_predict(clf0, X_fit[final_cols], y_fit)
    final_model = clf0.__class__(**clf0.get_params())
    final_model.fit(X_fit[final_cols], y_fit)

    bundle = {"model_name": winner, "model": final_model,
              "feature_columns": final_cols,
              "population": pop["chosen_population"],
              "s4_decision": abl["decision"], "random_state": RANDOM_STATE}
    yv_np = yv.to_numpy()
    metrics_json = {
        "task": "reference_parameter_regression",
        "cv": f"KFold(n={N_SPLITS}, seed={RANDOM_STATE})",
        "model_comparison_oof": {k: v["oof"] for k, v in comp["results"].items()},
        "model_comparison_winner": winner,
        "population_experiment": pop,
        "s4_ablation": abl,
        "final_oof": regression_metrics(
            y_fit.to_numpy(), np.clip(oof_final, 0, None)),
        "final_fold_mean_std": _agg(fold_reports),
        "final_features_n": len(final_cols),
        "top_features": feature_importances(final_model, final_cols,
                                            X_fit[final_cols], y_fit),
        "edge_robustness": edge_check(train_clean if not use_valid_only else
                                      train_clean.loc[valid_mask].copy(), feats, winner,
                                      oof_final, y_fit),
        "train_target_stats": {"min": float(np.min(yv_np)), "max": float(np.max(yv_np)),
                               "mean": float(np.mean(yv_np)), "std": float(np.std(yv_np))},
        "n_fit_rows": int(len(X_fit)),
        "clip_rule": "predictions clipped at 0 (temperature rise above ambient)",
    }
    return bundle, metrics_json


def save_bundle(bundle: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
