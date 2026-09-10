"""run_member3.py — Member 3 end-to-end (dev entry point).

train_clean.csv -> model/population/S4 experiments -> models/rp_model.pkl,
artifacts/rp_cv_metrics.json

Usage:  python run_member3.py
Needs:  run_member1.py outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from prediction.model import load_feature_columns  # noqa: E402
from prediction.predict import predict_rp  # noqa: E402
from prediction.train import build_bundle, save_bundle  # noqa: E402
from preprocessing.schema import C  # noqa: E402


def main() -> None:
    train_path = ROOT / "data" / "processed" / "train_clean.csv"
    test_path = ROOT / "data" / "processed" / "test_clean.csv"
    if not train_path.exists():
        sys.exit("Missing data/processed/train_clean.csv — run run_member1.py first.")
    feats = load_feature_columns()
    train = pd.read_csv(train_path)

    bundle, metrics_json = build_bundle(train, feats)
    save_bundle(bundle, ROOT / "models" / "rp_model.pkl")
    json.dump(metrics_json, open(ROOT / "artifacts" / "rp_cv_metrics.json", "w"), indent=2)

    print("Model OOF RMSE:",
          {k: round(v["rmse"], 3) for k, v in metrics_json["model_comparison_oof"].items()},
          f"-> winner {metrics_json['model_comparison_winner']}")
    pe = metrics_json["population_experiment"]
    print(f"Population: A rmse={pe['A_valid_only']['rmse']:.3f} vs "
          f"B rmse={pe['B_all_rows']['rmse']:.3f} -> {pe['chosen_population']}")
    ab = metrics_json["s4_ablation"]
    print(f"S4 ablation: with={ab['with_s4']['rmse']:.3f} vs "
          f"without={ab['without_s4']['rmse']:.3f} -> {ab['decision']}")
    fo = metrics_json["final_oof"]
    print(f"FINAL OOF: MAE={fo['mae']:.3f} RMSE={fo['rmse']:.3f} R2={fo['r2']:.3f} "
          f"(n={metrics_json['n_fit_rows']}, feats={metrics_json['final_features_n']})")
    print("Top-5 feats:", [(d["feature"], round(d["importance"], 3))
                           for d in metrics_json["top_features"][:5]])

    if test_path.exists():
        rp = predict_rp(pd.read_csv(test_path))
        ts = metrics_json["train_target_stats"]
        print(f"Test preds: min={rp['Predicted_Reference_Parameter'].min():.2f} "
              f"max={rp['Predicted_Reference_Parameter'].max():.2f} "
              f"(train range {ts['min']:.1f}–{ts['max']:.1f})")
        assert (rp["Predicted_Reference_Parameter"] >= 0).all() and \
            rp["Predicted_Reference_Parameter"].notna().all()
    print("MEMBER-3 BUILD: OK -> models/rp_model.pkl + artifacts/rp_cv_metrics.json")


if __name__ == "__main__":
    main()
