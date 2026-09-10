"""run_member2.py — Member 2 end-to-end (dev entry point).

train_clean.csv -> CV model selection -> models/anomaly_model.pkl,
artifacts/anomaly_thresholds.json, artifacts/anomaly_cv_metrics.json

Usage:  python run_member2.py
Needs:  run_member1.py outputs (data/processed/train_clean.csv, feature_columns).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from anomaly_detection.predict import predict_validity  # noqa: E402
from anomaly_detection.rules import baseline_predict, fit_flag_thresholds  # noqa: E402
from anomaly_detection.model import POS, build_bundle, save_bundle  # noqa: E402
from preprocessing.schema import C  # noqa: E402


def main() -> None:
    train_path = ROOT / "data" / "processed" / "train_clean.csv"
    test_path = ROOT / "data" / "processed" / "test_clean.csv"
    if not train_path.exists():
        sys.exit("Missing data/processed/train_clean.csv — run run_member1.py first.")
    feats = json.load(open(ROOT / "artifacts" / "feature_columns.json"))
    train = pd.read_csv(train_path)

    # scaffolding baseline (in-sample reference — optimistic, for comparison only)
    base = baseline_predict(train, fit_flag_thresholds(train[train[C.VALID] == "Valid"]))
    base_hit = (base["Predicted_Validity"].to_numpy() == train[C.VALID].to_numpy()).mean()

    bundle, thr_json, cv_json = build_bundle(train, feats)
    save_bundle(bundle, ROOT / "models" / "anomaly_model.pkl")
    json.dump(thr_json, open(ROOT / "artifacts" / "anomaly_thresholds.json", "w"), indent=2)
    json.dump(cv_json, open(ROOT / "artifacts" / "anomaly_cv_metrics.json", "w"), indent=2)

    print(f"Baseline (rule, in-sample acc, optimistic): {base_hit:.3f}")
    print(f"Winner: {cv_json['winner']}  threshold={cv_json['tuned_threshold']:.2f}")
    print("OOF @ tuned:", {k: (round(v, 3) if isinstance(v, float) else v)
          for k, v in cv_json["winner_oof_at_tuned_threshold"].items()})
    print("Regime check:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cv_json["regime_check"].items()})

    if test_path.exists():
        test = pd.read_csv(test_path)
        pv = predict_validity(test)
        n_inv = int((pv["Predicted_Validity"] == POS).sum())
        print(f"Test prediction: {n_inv}/{len(pv)} Invalid ({n_inv / len(pv):.1%})")
        print("Flag mix:", pv["Reason_Flags"].str.split("|").explode().value_counts().head(8).to_dict())
        assert list(pv[C.TEST_ID]) == list(test[C.TEST_ID].astype(str)), "order break!"
    print("MEMBER-2 BUILD: OK -> models/anomaly_model.pkl + artifacts/anomaly_*.json")


if __name__ == "__main__":
    main()
