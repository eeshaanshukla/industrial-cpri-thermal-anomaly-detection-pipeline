"""Member 2: abnormality / validity detection (Task 1).

Contract: predict_validity(df) -> DataFrame[Test_ID, Predicted_Validity,
           Anomaly_Score, Reason_Flags]  (1-arg; model loaded inside)
"""

from .model import train_anomaly_model
from .predict import predict_validity
from .rules import baseline_predict, compute_flags, fit_flag_thresholds, iso_feature_frame

__all__ = ["predict_validity", "train_anomaly_model", "baseline_predict", "compute_flags",
           "fit_flag_thresholds", "iso_feature_frame"]
