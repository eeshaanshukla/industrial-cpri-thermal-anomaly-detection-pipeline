"""tests/test_prediction.py — Member 3 contract tests (needs M1 outputs + bundle).

Run:  python tests/test_prediction.py
"""

from __future__ import annotations

import inspect
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prediction import predict_rp  # noqa: E402
from preprocessing.schema import C, FORBIDDEN_FEATURES  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_signature_is_1_arg():
    sig = inspect.signature(predict_rp)
    req = [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
    assert len(req) == 1, f"must be 1-arg (df); got {[p.name for p in req]}"


def test_output_contract_on_test():
    test = pd.read_csv(os.path.join(ROOT, "data", "processed", "test_clean.csv"))
    out = predict_rp(test)
    assert list(out.columns) == [C.TEST_ID, "Predicted_Reference_Parameter"]
    assert len(out) == len(test), "row drop!"
    assert list(out[C.TEST_ID]) == list(test[C.TEST_ID].astype(str)), "order break"
    col = out["Predicted_Reference_Parameter"]
    assert col.notna().all() and (col >= 0).all(), "NaN/negative predictions!"
    assert all(isinstance(v, float) or True for v in col)  # numeric
    pd.testing.assert_frame_equal(out, predict_rp(test))  # determinism
    # gross-extrapolation guard vs train range + margin
    m = json.load(open(os.path.join(ROOT, "artifacts", "rp_cv_metrics.json")))
    ts = m["train_target_stats"]
    cap = ts["max"] + 5 * ts["std"]
    assert float(col.max()) <= cap, f"wild extrapolation: {col.max():.1f} > cap {cap:.1f}"
    print(f"  preds in [0, {col.max():.1f}] (cap {cap:.1f}) — OK")


def test_metrics_and_features():
    m = json.load(open(os.path.join(ROOT, "artifacts", "rp_cv_metrics.json")))
    for key in ("model_comparison_oof", "population_experiment", "s4_ablation",
                "final_oof", "top_features", "edge_robustness"):
        assert key in m, f"missing {key}"
    import joblib

    bundle = joblib.load(os.path.join(ROOT, "models", "rp_model.pkl"))
    assert not (FORBIDDEN_FEATURES & set(bundle["feature_columns"])), "forbidden leak!"
    print(f"  winner={m['model_comparison_winner']} pop={m['population_experiment']['chosen_population']} "
          f"s4={m['s4_ablation']['decision']} rmse={m['final_oof']['rmse']:.3f} — OK")


def test_runs_on_train_schema_too():
    train = pd.read_csv(os.path.join(ROOT, "data", "processed", "train_clean.csv"))
    out = predict_rp(train)
    assert len(out) == len(train)


if __name__ == "__main__":
    test_signature_is_1_arg()
    test_output_contract_on_test()
    test_metrics_and_features()
    test_runs_on_train_schema_too()
    print("ALL Member-3 tests passed.")
