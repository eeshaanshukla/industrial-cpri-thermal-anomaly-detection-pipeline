"""summary.py — Task 3 automated summary (Member 4).

Scope (documented): min/max/avg over ALL predicted test rows (not Valid-only) —
the official text doesn't restrict them.

Attention rule (M4 engineering decision, M2 signals only — never RP outliers,
never M3 uncertainty):
  attention = 0.5 * score01 + 0.5 * closeness
  score01   = Anomaly_Score (already in [0,1], higher = more suspicious)
  closeness = 1 - |score - threshold| / max(threshold, 1 - threshold)
Ranked over ALL rows (not Invalid-only) so exactly 3 unique IDs always result.
Ties broken by Test_ID for determinism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXPLANATION_TEMPLATE = (
    "We cleaned {n} test records with train-fitted medians and kept every row in "
    "input order, adding {n_feats} thermal-consistency features. A gradient-boosting "
    "validity classifier (Invalid F1 {t1_f1}) flagged {n_inv} records as Invalid, "
    "separating sensor faults from genuine regime shifts. A second gradient-boosting "
    "model (RMSE {rmse}) predicted Reference_Parameter for all rows from {lo} to {hi} "
    "(avg {avg}). Attention blends anomaly score with decision-borderline closeness; "
    "highest attention: {top3}."
)


def attention_top3(validity_df: pd.DataFrame, threshold: float) -> list[str]:
    scores = pd.to_numeric(validity_df["Anomaly_Score"], errors="coerce").to_numpy(dtype=float)
    denom = max(float(threshold), 1.0 - float(threshold), 1e-9)
    closeness = 1.0 - np.abs(scores - float(threshold)) / denom
    attention = 0.5 * scores + 0.5 * closeness
    order = sorted(range(len(validity_df)), key=lambda i: (-attention[i], str(validity_df["Test_ID"].iloc[i])))
    top = [str(validity_df["Test_ID"].iloc[i]) for i in order[:3]]
    assert len(set(top)) == 3, "top-3 must be 3 unique IDs"
    return top


def count_words(text: str) -> int:
    return len(text.split())


def generate_summary(final_df: pd.DataFrame, validity_df: pd.DataFrame, rp_df: pd.DataFrame,
                     team_name: str, threshold: float, t1_f1: float,
                     rp_rmse: float, n_feats: int) -> dict:
    """Build the Task 3 summary dict. Pure function — explanation is templated."""
    rp = pd.to_numeric(final_df["Predicted_Reference_Parameter"], errors="coerce").to_numpy(dtype=float)
    n_inv = int((final_df["Validity_Label"].astype(str) == "Invalid").sum())
    top3 = attention_top3(validity_df, threshold)
    explanation = EXPLANATION_TEMPLATE.format(
        n=len(final_df), n_feats=n_feats, t1_f1=f"{t1_f1:.3f}", n_inv=n_inv,
        rmse=f"{rp_rmse:.2f}", lo=f"{float(np.min(rp)):.1f}",
        hi=f"{float(np.max(rp)):.1f}", avg=f"{float(np.mean(rp)):.1f}",
        top3=", ".join(top3))
    words = count_words(explanation)
    assert words <= 100, f"explanation {words} words — over the 100-word cap!"
    assert float(np.min(rp)) <= float(np.mean(rp)) <= float(np.max(rp))
    return {
        "team": team_name,
        "records_analysed": int(len(final_df)),
        "abnormal_invalid_count": n_inv,
        "min_predicted_reference_parameter": float(np.min(rp)),
        "max_predicted_reference_parameter": float(np.max(rp)),
        "avg_predicted_reference_parameter": float(np.mean(rp)),
        "top_3_attention_test_ids": top3,
        "approach_explanation": explanation,
        "approach_explanation_word_count": words,
    }
