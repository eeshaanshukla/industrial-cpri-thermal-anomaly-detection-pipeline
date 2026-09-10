"""Member 3: Reference_Parameter modeling (Task 2, 35% of grade).

Contract: predict_rp(df) -> DataFrame[Test_ID, Predicted_Reference_Parameter]
(1-arg; model loaded inside; clipped >= 0; finite everywhere).
"""

from .model import train_rp_model
from .predict import predict_rp

__all__ = ["predict_rp", "train_rp_model"]
