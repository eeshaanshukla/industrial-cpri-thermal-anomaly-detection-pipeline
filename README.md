# Industrial Thermal Hotspot Prediction & Anomaly Engine

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Architecture-Modular_Pipeline-orange.svg?style=flat)](https://github.com/)
[![Testing](https://img.shields.io/badge/Testing-Pytest-green.svg?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)


> **PowerNext-AI Hackathon 2026 — Screening Round Submission** 

An end-to-end, production grade machine learning and data engineering pipeline built for Central Power Research Institute (CPRI) laboratory test-bench data. The system implements a **dual-tier anomaly detection engine**, a **physics-guided thermal hotspot regressor**, an **automated evaluation report builder** and an architectural blueprint for **Digital Twin edge deployment**.

---

## 📌 Executive Summary & Key Results

| Task / Module | Methodology | Key Metrics / Benchmarks |
| :--- | :--- | :--- |
| **Task 01: Anomaly Engine** | Domain Rules + `HistGradientBoostingClassifier` | **F1-Score: 0.925** (Precision: 1.00, Recall: 0.86)<br>• 0% False Positive Rate on valid high-load regimes<br>• 38 / 350 test records isolated as `Invalid` |
| **Task 02: Reference Prediction** | `HistGradientBoostingRegressor` (Valid-only training) | **OOF MAE: 0.67°C** \| **RMSE: 1.07°C** \| **$R^2$: 0.990**<br>• `Sensor_S4` feature ablation (decoy feature removed)<br>• Inferred test range: 13.1°C to 59.4°C above ambient |
| **Task 03: Automated Summary** | Dynamic programmatic JSON builder | Automatically exports counts, statistics, top-3 attention IDs, and a concise methodology note |

---

## 🏗️ System Architecture

```text
                                 ┌─────────────────────────────────────────┐
                                 │ CPRI Raw Test Telemetry (.xlsx / .csv)  │
                                 └────────────────────┬────────────────────┘
                                                      │
                                                      ▼
                                   ┌─────────────────────────────────────┐
                                   │   Preprocessing & Cleaning Module   │
                                   │   - Outlier / Noise Handling        │
                                   │   - Feature Engineering & Schema    │
                                   └──────────────────┬──────────────────┘
                                                      │
                                                      ▼
                                   ┌─────────────────────────────────────┐
                                   │   Dual-Tier Anomaly Engine          │
                                   │   - Tier 1: Physical Rule Checks    │
                                   │   - Tier 2: HGB Classification      │
                                   └──────────┬──────────────────┬───────┘
                                              │                  │
                                 [Invalid]    │                  │    [Valid]
                                 ┌────────────▼───┐          ┌───▼──────────────────────────┐
                                 │ Flag & Isolate │          │ Physics-Guided Hotspot Regr. │
                                 └────────────────┘          │ (Reference Parameter Pred.)  │
                                                             └──────────────┬───────────────┘
                                                                            │
                                                      ┌─────────────────────┘
                                                      ▼
                                   ┌─────────────────────────────────────┐
                                   │   Automated Output & Summary Engine │
                                   │   - 5guys1repo.csv                  │
                                   │   - summary.json                    │
                                   └─────────────────────────────────────┘
# PowerNext-AI Hackathon 2026 — Screening Round



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

