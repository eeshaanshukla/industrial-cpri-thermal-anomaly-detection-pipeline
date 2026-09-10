"""Member 4: shared CV/metric utilities (M2/M3 own their task splitters, built here)."""

from .cv import make_classifier_splits, make_regression_splits
from .metrics import classification_metrics, confusion_matrix_nested, regression_metrics

__all__ = [
    "make_classifier_splits",
    "make_regression_splits",
    "classification_metrics",
    "confusion_matrix_nested",
    "regression_metrics",
]
