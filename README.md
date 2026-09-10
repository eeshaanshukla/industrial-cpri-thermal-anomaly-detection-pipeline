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
