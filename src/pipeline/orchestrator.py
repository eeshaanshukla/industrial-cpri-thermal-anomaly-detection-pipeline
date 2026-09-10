"""orchestrator.py — run_full_pipeline (Member 4).

FINAL path (FINAL_MODELS_READY=True):  M1 -> train M2 -> train M3 -> join -> write.
SCAFFOLD path (False):                  M1 -> rule baseline + median RP -> join -> write.
Both deterministic; generation is printed and stamped into the run log.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parents[0]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from anomaly_detection.predict import predict_validity  # noqa: E402
from anomaly_detection.rules import baseline_predict, fit_flag_thresholds  # noqa: E402
from pipeline.output_writer import build_final_output, write_submission_csv  # noqa: E402
from pipeline.summary import generate_summary  # noqa: E402
from prediction.predict import predict_rp  # noqa: E402
from preprocessing import load_raw, preprocess_test, preprocess_train  # noqa: E402
from preprocessing.schema import C, SHEET_TEST  # noqa: E402


def _find_workbook(cfg) -> Path:
    preferred = cfg.RAW_DIR / cfg.WORKBOOK_NAME
    if preferred.exists():
        return preferred
    cands = sorted(cfg.RAW_DIR.glob("*.xlsx"))
    if not cands:
        raise FileNotFoundError(f"No .xlsx in {cfg.RAW_DIR}")
    return cands[0]


def run_full_pipeline(cfg) -> dict[str, Any]:
    generation = "final" if cfg.FINAL_MODELS_READY else "scaffolding"
    print(f"MODEL GENERATION: {generation} (FINAL_MODELS_READY={cfg.FINAL_MODELS_READY})")
    if not cfg.FINAL_MODELS_READY:
        print("WARNING: scaffolding outputs are NOT submittable.")

    xlsx = _find_workbook(cfg)
    print(f"Raw workbook: {xlsx.name}")
    raw_test = load_raw(xlsx, cfg.SHEET_TEST)
    raw_ids = raw_test[C.TEST_ID].astype(str).tolist()
    print(f"Raw test rows: {len(raw_test)}")

    # ---- Member 1 (full build: processed + artifacts + mocks + quality report)
    print("[M1] preprocessing...")
    import run_member1
    run_member1.main(str(xlsx))
    train_clean = pd.read_csv(cfg.PROCESSED_DIR / "train_clean.csv")
    test_clean = pd.read_csv(cfg.PROCESSED_DIR / "test_clean.csv")
    feats = json.load(open(cfg.ARTIFACTS_DIR / "feature_columns.json"))
    if generation == "final":
        # ---- Member 2 (train supervised) ----
        print("[M2] training validity classifier...")
        import run_member2
        run_member2.main()
        # ---- Member 3 (train regressor) ----
        print("[M3] training RP regressor...")
        import run_member3
        run_member3.main()
        validity_df = predict_validity(test_clean)
        rp_df = predict_rp(test_clean)
        t1_f1 = json.load(open(cfg.ARTIFACTS_DIR / "anomaly_cv_metrics.json"))[
            "winner_oof_at_tuned_threshold"]["pos_f1"]
        rp_rmse = json.load(open(cfg.ARTIFACTS_DIR / "rp_cv_metrics.json"))["final_oof"]["rmse"]
        thr = json.load(open(cfg.ARTIFACTS_DIR / "anomaly_thresholds.json"))["threshold"]
    else:
        print("[M2/M3] scaffolding predictors (rule baseline + median RP)...")
        valid_only = train_clean[train_clean[C.VALID].astype(str) == "Valid"]
        flag_thr = fit_flag_thresholds(valid_only)
        validity_df = baseline_predict(test_clean, flag_thr)
        sys.path.insert(0, str(ROOT / "mocks"))
        from mock_rp_helper import mock_predict_rp
        rp_df = mock_predict_rp(test_clean)
        t1_f1, rp_rmse, thr = 0.0, 0.0, 0.5

    # ---- Member 4 (join + write) ----
    print("[M4] writing outputs...")
    final_df = build_final_output(test_clean, validity_df, rp_df)
    assert list(final_df["Test_ID"]) == raw_ids, "output IDs != RAW test input order!"
    assert len(final_df) == len(raw_test), "row count != raw test count!"
    csv_path = write_submission_csv(final_df, cfg.OUTPUTS_DIR / f"{cfg.TEAM_NAME}.csv")
    summary = generate_summary(final_df, validity_df, rp_df, cfg.TEAM_NAME, thr,
                               t1_f1, rp_rmse, len(feats))
    assert summary["records_analysed"] == len(raw_test)
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(cfg.OUTPUTS_DIR / "summary.json", "w"), indent=2)

    print(f"WROTE {csv_path} ({len(final_df)} rows)")
    print(f"WROTE {cfg.OUTPUTS_DIR / 'summary.json'}")
    print(f"Invalid: {summary['abnormal_invalid_count']} | "
          f"RP range {summary['min_predicted_reference_parameter']:.1f}–"
          f"{summary['max_predicted_reference_parameter']:.1f} "
          f"avg {summary['avg_predicted_reference_parameter']:.1f}")
    print(f"Top-3 attention: {summary['top_3_attention_test_ids']}")
    return {"generation": generation, "csv": str(csv_path),
            "summary": str(cfg.OUTPUTS_DIR / "summary.json")}
