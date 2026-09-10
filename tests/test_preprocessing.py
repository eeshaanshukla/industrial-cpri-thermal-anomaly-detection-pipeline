"""tests/test_preprocessing.py — Member 1 acceptance tests (no dependency on M2/M3/M4).

Run:  python -m pytest tests/test_preprocessing.py -q
   or python tests/test_preprocessing.py  (stdlib-only runner below)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import (  # noqa: E402
    C,
    engineer_features,
    feature_column_list,
    make_missing_flags,
    preprocess_test,
    preprocess_train,
)
from preprocessing.schema import FORBIDDEN_FEATURES  # noqa: E402

RAW_XLSX = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def _find_workbook() -> str | None:
    if not os.path.isdir(RAW_XLSX):
        return None
    for f in sorted(os.listdir(RAW_XLSX)):
        if f.lower().endswith((".xlsx", ".xlsm")):
            return os.path.join(RAW_XLSX, f)
    return None


def test_schema_forbiddens():
    assert FORBIDDEN_FEATURES == frozenset({"Test_ID", "Reference_Parameter", "Validity_Label"})


def test_missing_flags_binary_and_preimputation():
    df = pd.DataFrame({
        C.S1: [1.0, None, 3.0],
        C.S2: [None, None, 3.0],
        C.S3: [1.0, 2.0, 3.0],
        C.S4: [0.5, 0.6, None],
    })
    flags = make_missing_flags(df)
    assert list(flags["feat_missing_count"]) == [1, 2, 1]
    for c in flags.columns:
        if c == "feat_missing_count":
            assert set(flags[c].unique()) <= {0, 1, 2, 3, 4}, c
        else:
            assert set(flags[c].unique()) <= {0, 1}, c
    per_sensor = [c for c in flags.columns if c != "feat_missing_count"]
    assert (flags["feat_missing_count"] == flags[per_sensor].sum(axis=1)).all()


def test_end_to_end_if_workbook_present():
    xlsx = _find_workbook()
    if xlsx is None:
        print("SKIP end-to-end: no workbook in data/raw/")
        return
    with tempfile.TemporaryDirectory() as td:
        train_clean, arts = preprocess_train(xlsx, save_dir=td)
        test_clean = preprocess_test(xlsx, arts, save_path=os.path.join(td, "t.csv"))
        # re-run determinism
        test_clean2 = preprocess_test(xlsx, arts)
        pd.testing.assert_frame_equal(test_clean, test_clean2)
    # acceptance criteria
    from preprocessing import load_raw
    raw_test = load_raw(xlsx, "Test_Data")
    assert len(test_clean) == len(raw_test), "test rows dropped!"
    assert list(test_clean[C.TEST_ID].astype(str)) == list(raw_test[C.TEST_ID].astype(str))
    feat_cols = feature_column_list(test_clean)
    assert not (FORBIDDEN_FEATURES & set(feat_cols))
    assert any(c.startswith("feat_missing_") for c in feat_cols)
    assert int(test_clean[feat_cols].isna().sum().sum()) == 0, "NaNs remain in features"
    train_cols = set(train_clean.columns) - {C.RP, C.VALID}
    test_cols = set(test_clean.columns)
    assert train_cols == test_cols, f"schema drift: {train_cols ^ test_cols}"


def test_saved_artifacts_if_present():
    feat_json = os.path.join(os.path.dirname(__file__), "..", "artifacts", "feature_columns.json")
    if not os.path.exists(feat_json):
        print("SKIP saved-artifact check: artifacts not built yet")
        return
    feats = json.load(open(feat_json))
    assert not (FORBIDDEN_FEATURES & set(feats)), "forbidden col in feature_columns.json"


if __name__ == "__main__":
    test_schema_forbiddens()
    test_missing_flags_binary_and_preimputation()
    test_end_to_end_if_workbook_present()
    test_saved_artifacts_if_present()
    print("ALL Member-1 tests passed.")
