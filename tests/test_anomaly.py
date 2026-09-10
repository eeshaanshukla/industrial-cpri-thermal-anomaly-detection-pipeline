"""tests/test_anomaly.py — Member 2 contract tests (needs M1 outputs + bundle).

Run:  python tests/test_anomaly.py
"""

from __future__ import annotations

import inspect
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anomaly_detection import predict_validity  # noqa: E402
from preprocessing.schema import C  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_signature_is_1_arg():
    sig = inspect.signature(predict_validity)
    req = [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
    assert len(req) == 1, f"must be 1-arg (df); got {[p.name for p in req]}"


def test_output_contract():
    test = pd.read_csv(os.path.join(ROOT, "data", "processed", "test_clean.csv"))
    out = predict_validity(test)
    assert list(out.columns) == [C.TEST_ID, "Predicted_Validity", "Anomaly_Score", "Reason_Flags"]
    assert len(out) == len(test)
    assert list(out[C.TEST_ID]) == list(test[C.TEST_ID].astype(str)), "order/count break"
    assert set(out["Predicted_Validity"].unique()) <= {"Valid", "Invalid"}
    assert out["Anomaly_Score"].between(0, 1).all() and out["Anomaly_Score"].isna().sum() == 0
    assert (out["Reason_Flags"].astype(str).str.len() > 0).all()
    # determinism
    pd.testing.assert_frame_equal(out, predict_validity(test))
    # score/threshold consistency with saved bundle threshold
    import json

    thr = json.load(open(os.path.join(ROOT, "artifacts", "anomaly_thresholds.json")))["threshold"]
    expect_invalid = out["Anomaly_Score"] >= thr
    assert ((out["Predicted_Validity"] == "Invalid") == expect_invalid).all()
    rate = expect_invalid.mean()
    assert 0.0 < rate < 0.5, f"implausible invalid rate {rate:.2%}"
    print(f"  test invalid rate: {rate:.1%} (threshold {thr}) — OK")


def test_runs_on_train_schema_too():
    train = pd.read_csv(os.path.join(ROOT, "data", "processed", "train_clean.csv"))
    out = predict_validity(train)
    assert len(out) == len(train)


if __name__ == "__main__":
    test_signature_is_1_arg()
    test_output_contract()
    test_runs_on_train_schema_too()
    print("ALL Member-2 tests passed.")
