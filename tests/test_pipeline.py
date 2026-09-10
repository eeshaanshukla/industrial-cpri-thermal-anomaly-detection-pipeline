"""tests/test_pipeline.py — Member 4 tests (mock units + end-to-end on outputs/).

Run:  python tests/test_pipeline.py   (after run_pipeline.py for the e2e part)
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.output_writer import build_final_output  # noqa: E402
from pipeline.summary import attention_top3, generate_summary  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _mocks():
    test_df = pd.DataFrame({"Test_ID": ["T-3", "T-1", "T-2", "T-4"]})
    validity_df = pd.DataFrame({
        "Test_ID": ["T-3", "T-1", "T-2", "T-4"],
        "Predicted_Validity": ["Valid", "Invalid", "Valid", "Valid"],
        "Anomaly_Score": [0.1, 0.9, 0.45, 0.2],
        "Reason_Flags": ["CONSISTENT", "MODEL_INVALID", "CONSISTENT", "CONSISTENT"],
    })
    rp_df = pd.DataFrame({"Test_ID": ["T-3", "T-1", "T-2", "T-4"],
                          "Predicted_Reference_Parameter": [20.0, 30.0, 25.0, 22.0]})
    return test_df, validity_df, rp_df


def test_builder_with_mocks():
    test_df, v, r = _mocks()
    final = build_final_output(test_df, v, r)
    assert list(final.columns) == ["Test_ID", "Predicted_Reference_Parameter", "Validity_Label"]
    assert list(final["Test_ID"]) == ["T-3", "T-1", "T-2", "T-4"], "order must follow input"
    assert len(final) == 4 and final["Test_ID"].is_unique
    assert list(final["Validity_Label"]) == ["Valid", "Invalid", "Valid", "Valid"]
    # order violation must fail loudly
    bad = v.iloc[::-1].reset_index(drop=True)
    try:
        build_final_output(test_df, bad, r)
        raise AssertionError("reordered input should raise")
    except ValueError:
        pass


def test_summary_with_mocks():
    test_df, v, r = _mocks()
    final = build_final_output(test_df, v, r)
    s = generate_summary(final, v, r, "MockTeam", 0.5, 0.9, 1.1, 45)
    assert s["records_analysed"] == 4 and s["abnormal_invalid_count"] == 1
    assert s["min_predicted_reference_parameter"] <= s["avg_predicted_reference_parameter"] \
        <= s["max_predicted_reference_parameter"]
    assert len(s["top_3_attention_test_ids"]) == 3 and len(set(s["top_3_attention_test_ids"])) == 3
    assert s["approach_explanation_word_count"] <= 100
    # determinism of pure functions
    s2 = generate_summary(final, v, r, "MockTeam", 0.5, 0.9, 1.1, 45)
    assert s == s2
    # borderline+extreme blend: T-1 (0.9, far above thr) and T-2 (0.45, near thr) rank high
    top = attention_top3(v, 0.5)
    assert "T-1" in top and "T-2" in top, top


def test_end_to_end_outputs():
    import config  # noqa: E402

    csv_path = os.path.join(ROOT, "outputs", f"{config.TEAM_NAME}.csv")
    js_path = os.path.join(ROOT, "outputs", "summary.json")
    assert os.path.exists(csv_path) and os.path.exists(js_path), "run run_pipeline.py first"
    assert config.FINAL_MODELS_READY, "not submittable while scaffolding!"
    final = pd.read_csv(csv_path)
    assert list(final.columns) == ["Test_ID", "Predicted_Reference_Parameter", "Validity_Label"]
    raw_test = pd.read_excel(
        os.path.join(ROOT, "data", "raw", config.WORKBOOK_NAME), sheet_name=config.SHEET_TEST,
        engine="openpyxl")
    assert len(final) == len(raw_test) == 350
    assert list(final["Test_ID"].astype(str)) == list(raw_test["Test_ID"].astype(str))
    assert final["Predicted_Reference_Parameter"].notna().all()
    assert (final["Predicted_Reference_Parameter"] >= 0).all()
    assert set(final["Validity_Label"].unique()) <= {"Valid", "Invalid"}
    s = json.load(open(js_path))
    assert s["records_analysed"] == len(raw_test)
    assert s["abnormal_invalid_count"] == int((final["Validity_Label"] == "Invalid").sum())
    assert abs(s["avg_predicted_reference_parameter"] - final["Predicted_Reference_Parameter"].mean()) < 1e-9
    assert len(s["top_3_attention_test_ids"]) == 3
    assert s["approach_explanation_word_count"] <= 100
    print(f"  e2e OK: {len(final)} rows, {s['abnormal_invalid_count']} invalid, "
          f"top3={s['top_3_attention_test_ids']}")


if __name__ == "__main__":
    test_builder_with_mocks()
    test_summary_with_mocks()
    test_end_to_end_outputs()
    print("ALL Member-4 tests passed.")
