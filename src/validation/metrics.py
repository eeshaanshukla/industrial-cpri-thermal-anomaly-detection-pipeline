"""metrics.py — shared metric helpers (Member 4 owns; all modules call these).

Internal validation metrics (organizers publish weights, not formulas).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_scores=None, pos_label: str = "Invalid",
                           neg_label: str = "Valid") -> dict:
    """Per-class precision/recall/F1 (+ ROC-AUC when scores given). Labels explicit."""
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[neg_label, pos_label], zero_division=0)
    out = {
        "neg_label": neg_label, "pos_label": pos_label,
        "neg_precision": float(p[0]), "neg_recall": float(r[0]), "neg_f1": float(f[0]),
        "neg_support": int(s[0]),
        "pos_precision": float(p[1]), "pos_recall": float(r[1]), "pos_f1": float(f[1]),
        "pos_support": int(s[1]),
    }
    if y_scores is not None:
        y_bin = (y_true == pos_label).astype(int)
        out["roc_auc"] = (float(roc_auc_score(y_bin, np.asarray(y_scores, dtype=float)))
                          if len(np.unique(y_bin)) > 1 else float("nan"))
    return out


def confusion_matrix_nested(y_true, y_pred, neg_label: str = "Valid",
                            pos_label: str = "Invalid") -> dict:
    """OOF confusion matrix as nested lists, label order documented."""
    cm = confusion_matrix(np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str),
                          labels=[neg_label, pos_label]).tolist()
    return {"labels_order": [neg_label, pos_label],
            "matrix": [[int(v) for v in row] for row in cm],
            "layout": "rows=true [neg,pos], cols=pred [neg,pos] (TN,FP / FN,TP)"}


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE (primary), R² (secondary) for Reference_Parameter validation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": float(mean_absolute_error(y_true, y_pred)), "rmse": rmse,
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
            "n": int(len(y_true))}
