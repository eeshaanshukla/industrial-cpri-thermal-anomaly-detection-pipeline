# Methodology Note — Team 5guys1repo — PowerNext-AI 2026 Screening Round

## 1. Approach

**Preprocessing (M1).** We load the participant workbook (`Training_Data` /
`Test_Data`) with a CSV fallback, freeze exact headers in `schema.py`, impute
with train-fitted medians (sensors only — the sole missing columns), and
deduplicate training rows only (12 dropped, keep-first). All 350 test rows are
preserved in input order. We add 37 `feat_*` columns: power/energy/Joule-heating
proxies, S1–S3 consensus (mean, spread, pairwise gaps, per-sensor deviations),
S4-relevance probes, rise-per-input ratios, and pre-imputation `feat_missing_*`
flags — 45 model features total, with ID/targets excluded by construction.

**Validity detection (M2).** A supervised HistGradientBoosting classifier on
engineer labels (StratifiedKFold ×5, seed 42, balanced classes) beats a rule
baseline and RandomForest on out-of-fold Invalid-F1 (0.925; P 1.00 / R 0.86).
The decision threshold (0.15) is tuned on OOF scores. `Anomaly_Score` is exactly
P(Invalid) in [0,1]; `Reason_Flags` (trio disagreement, sensor deviation,
missingness, duplicates) are deterministic diagnostics. 38/350 test rows are
flagged Invalid.

**Reference prediction (M3).** HistGradientBoosting regression trained on Valid
rows only (866), plain KFold ×5: OOF MAE 0.67, RMSE 1.07, R² 0.990 — far above
the linear floor (RMSE 3.98). A Valid-only-vs-all-rows experiment (both scored
on Valid held-out folds) keeps Valid-only; an S4 ablation drops the S4 family.
Predictions are clipped at 0 at source and asserted finite; test range
13.1–59.4 sits inside the training range (11.9–61.6).

**Integration (M4).** `python run_pipeline.py` reproduces everything from the
raw xlsx. The writer renames `Predicted_Validity`→`Validity_Label`, drops
diagnostics, and asserts 350/350 rows in input order with finite, non-negative
predictions. `summary.json` is fully code-generated (template + fitted numbers).

## 2. Important parameters

Load_Current_A dominates (0.92 permutation importance — current drives heating),
refined by sensor consensus (S2 most trusted, corr 0.70 with target; S1 0.54,
S3 0.48), absolute-temperature views, and missingness flags. Sensor_S4 is the
planted decoy (corr 0.002; ablation confirms dropping it helps). Voltage,
duration, and ambient enter through power/energy interaction terms.

## 3. Abnormal-data method

We deliberately do NOT equate outlier with invalid: the classifier learns from
real labels which unusual patterns engineers accepted. Evidence: 0% OOF
false-positive rate on high-current-but-Valid edge rows, e.g. TRN-0762
(110 A, ≥p99) scores 0.0006 — correctly kept Valid. Flags explain each verdict
without overriding the supervised signal.

## 4. Assumptions

Training on Valid rows only is our engineering decision (Invalid rows carry
unreliable measurements), disclosed not mandated. Median imputation fitted once
on train leaks negligibly. Summary min/max/avg cover ALL predicted rows.
Threshold 0.15 and feature medians transfer to the second unseen dataset, which
we assume shares the schema; edge-slice OOF checks (high-current RMSE 1.21 vs
1.07 overall) are our robustness proxy.

## 5. Digital Twin Automation Steps

1. **Connect:** stream test-bench logs into a validated schema service (this
   pipeline's loader + `schema.py`, rising to a message queue in production).
2. **Preprocess live:** apply frozen medians/features per record; quarantine
   schema violations for engineer review instead of crashing.
3. **Twin inference:** serve the validity + RP models behind a versioned API;
   log every score, flag, and prediction with model version.
4. **Attention queue:** rank live records with the §1 attention blend so
   engineers review the riskiest tests first.
5. **Feedback loop:** record engineer Valid/Invalid verdicts as new labels.
6. **Retrain & gate:** nightly retraining with the frozen CV scheme; promote
   only on better OOF metrics plus the regime-edge checks.
7. **Monitor:** track input drift, invalid-rate shifts, and prediction-range
   excursions with alerts; keep deterministic reruns for audit.
