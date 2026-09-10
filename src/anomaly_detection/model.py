"""model.py — supervised validity classifier (Member 2).

- 5-fold StratifiedKFold, seed 42 from config (classification, ~13% Invalid).
- Candidates: RandomForest vs HistGradientBoosting, identical folds; winner =
  best OOF Invalid-F1. class_weight balanced (no SMOTE — reproducibility).
- Threshold tuned on OOF P(Invalid) for Invalid-F1 (tie-break: precision).
- Final model refit on full train; bundle saved to models/anomaly_model.pkl.
- Aux IsolationForest on Valid-only disagreement features -> ISO_ANOMALY flag
  (diagnostic only — the verdict is purely the supervised signal).
- Regime check: OOF false-positive rate on high-current VALID edge rows must
  stay low (the "unusual != invalid" 25%-grade behaviour, made measurable).

Anomaly_Score DEFINITION (also stored in anomaly_thresholds.json):
  P(Invalid) from the winning classifier; float in [0, 1];
  higher = more suspicious. NOT blended with unsupervised scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parents[0]))
try:
    from config import ARTIFACTS_DIR, CV_FOLDS, RANDOM_STATE  # noqa: E402
except ImportError:
    ARTIFACTS_DIR, CV_FOLDS, RANDOM_STATE = Path("artifacts"), 5, 42

from preprocessing.schema import C, FORBIDDEN_FEATURES  # noqa: E402
from validation.cv import make_classifier_splits  # noqa: E402
from validation.metrics import (  # noqa: E402
    classification_metrics,
    confusion_matrix_nested,
)

from .rules import fit_flag_thresholds, iso_feature_frame  # noqa: E402

N_SPLITS = CV_FOLDS
POS, NEG = "Invalid", "Valid"


def load_feature_columns() -> list[str]:
    """Feature list with the MANDATORY forbidden-name assertion (never trust)."""
    for cand in (ARTIFACTS_DIR / "feature_columns.json", SRC.parents[0] / "feature_columns.json"):
        if Path(cand).exists():
            feats = json.load(open(cand))
            assert not (FORBIDDEN_FEATURES & set(feats)), f"forbidden leak: {FORBIDDEN_FEATURES & set(feats)}"
            return feats
    raise FileNotFoundError("feature_columns.json not found (run run_member1.py first).")


def candidates() -> list[tuple[str, object]]:
    cands: list[tuple[str, object]] = [
        ("rf", RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                      min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
    ]
    try:
        cands.append(("hgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=RANDOM_STATE)))
    except TypeError:  # older sklearn without HGB class_weight
        cands.append(("hgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE)))
    return cands


def tune_threshold(y_true_bin: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Grid-search P(Invalid) cutoff for Invalid-F1; tie-break on precision."""
    best = (0.5, -1.0, -1.0)  # (thr, f1, prec)
    for thr in np.arange(0.05, 0.955, 0.05):
        pred = (scores >= thr).astype(int)
        tp = int(((pred == 1) & (y_true_bin == 1)).sum())
        fp = int(((pred == 1) & (y_true_bin == 0)).sum())
        fn = int(((pred == 0) & (y_true_bin == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if (f1, prec) > (best[1], best[2]):
            best = (float(thr), f1, prec)
    return best[0], best[1]


def run_cv(X: pd.DataFrame, y: pd.Series) -> dict:
    """OOF evaluation for every candidate on identical stratified folds."""
    skf = make_classifier_splits(N_SPLITS, RANDOM_STATE)
    y_bin = (y == POS).astype(int).to_numpy()
    results: dict[str, dict] = {}
    for name, clf in candidates():
        oof = np.zeros(len(X))
        fold_reports = []
        for tr, va in skf.split(X, y):
            c = clf.__class__(**clf.get_params())
            c.fit(X.iloc[tr], y.iloc[tr])
            s = c.predict_proba(X.iloc[va])[:, list(c.classes_).index(POS)]
            oof[va] = s
            fold_reports.append(classification_metrics(y.iloc[va].to_numpy(), c.predict(X.iloc[va]), s))
        thr, thr_f1 = tune_threshold(y_bin, oof)
        oof_pred = np.where(oof >= thr, POS, NEG)
        results[name] = {
            "oof_scores": oof, "oof_pred_at_tuned_thr": oof_pred,
            "fold_reports": fold_reports,
            "oof_report_at_05": classification_metrics(y.to_numpy(), np.where(oof >= 0.5, POS, NEG), oof),
            "tuned_threshold": thr,
            "oof_report_at_tuned": classification_metrics(y.to_numpy(), oof_pred, oof),
            "tuned_oof_f1": float(thr_f1),
        }
    return results


def regime_check(df: pd.DataFrame, y: pd.Series, oof_scores: np.ndarray, threshold: float) -> dict:
    """OOF FPR on VALID edge-of-distribution rows (high current / high power)."""
    curr = pd.to_numeric(df[C.CURR], errors="coerce")
    cut = float(curr.quantile(0.95))
    edge_valid = (y == NEG) & (curr >= cut)
    oof_pred_invalid = oof_scores >= threshold
    fpr = float(oof_pred_invalid[edge_valid.to_numpy()].mean()) if edge_valid.sum() else float("nan")
    return {
        "edge_definition": "train rows with Load_Current_A >= p95 AND Validity_Label==Valid",
        "current_p95": cut,
        "n_edge_valid_rows": int(edge_valid.sum()),
        "oof_fpr_on_edge_valid": fpr,
        "oof_invalid_recall": float(oof_pred_invalid[(y == POS).to_numpy()].mean()),
    }


def worked_valid_but_unusual(df: pd.DataFrame, y: pd.Series, oof_scores: np.ndarray,
                             threshold: float) -> dict:
    """One worked example: statistically unusual yet correctly kept Valid.

    Picks the highest-current Valid row that OOF scoring keeps below threshold.
    """
    curr = pd.to_numeric(df[C.CURR], errors="coerce")
    p99 = float(curr.quantile(0.99))
    cands = df[(y == NEG) & (curr >= p99)].copy()
    cands["_oof"] = oof_scores[cands.index.to_numpy()]
    kept = cands[cands["_oof"] < threshold].sort_values(C.CURR, ascending=False)
    if kept.empty:  # fallback: highest-current Valid row regardless
        kept = cands.sort_values(C.CURR, ascending=False)
    row = kept.iloc[0]
    return {
        "test_id": str(row[C.TEST_ID]),
        "why_unusual": f"Load_Current_A={row[C.CURR]:.1f} (>= p99={p99:.1f}); "
                       f"Applied_Voltage_kV={row[C.VOLT]:.1f}; S1/S2/S3="
                       f"{row[C.S1]:.1f}/{row[C.S2]:.1f}/{row[C.S3]:.1f}",
        "true_label": NEG, "oof_anomaly_score": float(row["_oof"]),
        "threshold": float(threshold),
        "oof_verdict": NEG if float(row["_oof"]) < threshold else POS,
    }


def fit_final(X: pd.DataFrame, y: pd.Series, model_name: str):
    clf = dict(candidates())[model_name]
    final_model = clf.__class__(**clf.get_params())
    final_model.fit(X, y)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_STATE)
    iso.fit(iso_feature_frame(X[y == NEG]))
    return final_model, iso


def aggregate(reports: list[dict]) -> dict:
    keys = [k for k in reports[0] if isinstance(reports[0][k], float)]
    return {k: {"mean": float(np.nanmean([r[k] for r in reports])),
                "std": float(np.nanstd([r[k] for r in reports]))} for k in keys}


def score_definition(model_name: str, threshold: float) -> str:
    return (f"Anomaly_Score = P(Invalid) from {model_name} classifier "
            f"(predict_proba positive class); range [0,1]; higher = more suspicious; "
            f"decision threshold {threshold:.2f} tuned for Invalid-F1 on OOF scores "
            f"(tie-break: precision). NOT blended with unsupervised signals.")


def build_bundle(train_clean: pd.DataFrame, feature_columns: list[str]) -> tuple[dict, dict, dict]:
    """Train everything. Returns (joblib_bundle, thresholds_json, cv_metrics_json)."""
    assert not (FORBIDDEN_FEATURES & set(feature_columns)), "forbidden col in features!"
    X = train_clean[feature_columns].copy()
    y = train_clean[C.VALID].astype(str)
    assert set(y.unique()) <= {NEG, POS}, f"unexpected labels: {y.unique()}"

    cv = run_cv(X, y)
    winner = max(cv, key=lambda k: cv[k]["tuned_oof_f1"])
    w = cv[winner]
    final_model, iso = fit_final(X, y, winner)
    flag_thr = fit_flag_thresholds(train_clean[y == NEG])
    regime = regime_check(train_clean, y, w["oof_scores"], w["tuned_threshold"])
    worked = worked_valid_but_unusual(train_clean, y, w["oof_scores"], w["tuned_threshold"])

    bundle = {"model_name": winner, "model": final_model, "iso_forest": iso,
              "feature_columns": feature_columns, "threshold": w["tuned_threshold"],
              "flag_thresholds": flag_thr, "random_state": RANDOM_STATE}
    thresholds_json = {"threshold": w["tuned_threshold"],
                       "tuned_for": "Invalid-class F1 on OOF scores (tie-break: precision)",
                       "anomaly_score_definition": score_definition(winner, w["tuned_threshold"]),
                       "flag_thresholds": flag_thr,
                       "model": winner, "random_state": RANDOM_STATE}
    cv_metrics_json = {
        "task": "validity_classification",
        "cv": f"StratifiedKFold(n={N_SPLITS}, seed={RANDOM_STATE})",
        "model_comparison_oof_invalid_f1": {k: v["tuned_oof_f1"] for k, v in cv.items()},
        "winner": winner,
        "winner_fold_mean_std_at_05": aggregate(w["fold_reports"]),
        "winner_oof_at_tuned_threshold": w["oof_report_at_tuned"],
        "winner_oof_confusion_matrix": confusion_matrix_nested(
            y.to_numpy(), w["oof_pred_at_tuned_thr"]),
        "tuned_threshold": w["tuned_threshold"],
        "regime_check": regime,
        "worked_example_valid_but_unusual": worked,
        "class_balance_train": {str(k): int(v) for k, v in y.value_counts().items()},
        "n_train_rows": len(X), "n_features": len(feature_columns),
    }
    return bundle, thresholds_json, cv_metrics_json


def train_anomaly_model(train_df: pd.DataFrame) -> tuple[object, dict]:
    """Role-doc function contract: 1-arg train -> (fitted_model, info_dict).

    Loads the frozen feature list from disk (asserting the forbidden names),
    runs selection + threshold tuning + final refit. info_dict carries the
    threshold, flag thresholds, winner name, and OOF summary for the caller.
    """
    feats = load_feature_columns()
    bundle, thr_json, cv_json = build_bundle(train_df, feats)
    info = {"model_name": bundle["model_name"], "threshold": bundle["threshold"],
            "flag_thresholds": bundle["flag_thresholds"],
            "oof_invalid_f1": cv_json["winner_oof_at_tuned_threshold"]["pos_f1"],
            "cv_metrics": cv_json, "thresholds": thr_json}
    return bundle["model"], info


def save_bundle(bundle: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
