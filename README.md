# PowerNext-AI 2026 — Screening Round (Team 5guys1repo)

Black-box test-bench challenge: detect invalid records (Task 1), predict
`Reference_Parameter` (Task 2), auto-generate the test summary (Task 3).

## Run (the only command that matters)

```bash
pip install -r requirements.txt
python run_pipeline.py
```

This regenerates **everything** from the raw workbook in `data/raw/` —
clean data, both trained models, `outputs/5guys1repo.csv`, `outputs/summary.json`.
Zero manual steps, deterministic (seed 42). It prints
`MODEL GENERATION: final` (never submit on `scaffolding`).

## Outputs

| File | Content |
|---|---|
| `outputs/<TeamName>.csv` | `Test_ID,Predicted_Reference_Parameter,Validity_Label` — 350 rows, raw input order |
| `outputs/summary.json` | counts, min/max/avg (over ALL rows), top-3 attention IDs, ≤100-word generated explanation |
| `docs/methodology_note.pdf` | ≤2-page methodology note incl. digital-twin steps |

## Layout

```text
config.py                 seed 42, TEAM_NAME, FINAL_MODELS_READY, paths (M4 owns)
run_pipeline.py           single entry point -> src/pipeline/orchestrator.py
run_member1/2/3.py        per-module dev entry points (also called by the pipeline)
src/preprocessing/        M1: loader, cleaner, features, schema (column-name truth)
src/anomaly_detection/    M2: rules, model (HGB classifier), predict_validity
src/prediction/           M3: model, train (experiments), predict_rp
src/validation/           M4: shared CV splitters + metric helpers
src/pipeline/             M4: orchestrator, summary, output_writer
models/ artifacts/        trained bundles + CV metrics/thresholds (generated)
tests/                    one suite per member; run any with python tests/test_*.py
```

## Results (on the official participant file)

- **Task 1:** HistGradientBoosting, OOF Invalid P/R/F1 = 1.00/0.86/0.925;
  0% FPR on high-current-but-Valid edge rows; 38/350 test rows flagged Invalid.
- **Task 2:** HistGradientBoosting on Valid-only rows, OOF MAE 0.67 / RMSE 1.07 /
  R² 0.990; S4 ablation → dropped (decoy confirmed); test preds 13.1–59.4.
- **Task 3:** fully code-generated `summary.json` (attention = anomaly score ×
  decision-borderline closeness).

## Reproduce / verify

```bash
python tests/test_preprocessing.py && python tests/test_anomaly.py \
  && python tests/test_prediction.py && python tests/test_pipeline.py
```

Clean-checkout check: delete `data/processed/ models/ artifacts/*.json
artifacts/*.md outputs/` (keep `data/raw/*.xlsx`), re-run `run_pipeline.py`,
diff `outputs/` — byte-identical.
