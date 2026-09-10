"""output_writer.py — the ONLY internal->final mapping (Member 4).

INTERNAL (never graded): Predicted_Validity, Anomaly_Score, Reason_Flags,
  Predicted_Reference_Parameter
FINAL (FROZEN): Test_ID, Predicted_Reference_Parameter, Validity_Label
in RAW TEST INPUT ORDER, one row per raw test ID — index-join, never sort.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_final_output(test_df: pd.DataFrame, validity_df: pd.DataFrame,
                       rp_df: pd.DataFrame) -> pd.DataFrame:
    """Join M2 + M3 outputs onto the clean-test frame order. Pure function."""
    for name, df in (("validity_df", validity_df), ("rp_df", rp_df)):
        if "Test_ID" not in df.columns:
            raise ValueError(f"{name} missing Test_ID")
    want_ids = test_df["Test_ID"].astype(str).tolist()
    v_ids = validity_df["Test_ID"].astype(str).tolist()
    r_ids = rp_df["Test_ID"].astype(str).tolist()
    if v_ids != want_ids:
        raise ValueError("validity_df IDs/order != test input order")
    if r_ids != want_ids:
        raise ValueError("rp_df IDs/order != test input order")

    rp = pd.to_numeric(rp_df["Predicted_Reference_Parameter"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(rp)):
        raise ValueError("non-finite Predicted_Reference_Parameter at writer")
    if np.any(rp < 0):
        raise ValueError("negative Predicted_Reference_Parameter at writer (M3 must clip)")
    labels = validity_df["Predicted_Validity"].astype(str).to_numpy()
    if set(np.unique(labels)) - {"Valid", "Invalid"}:
        raise ValueError(f"unexpected validity labels: {np.unique(labels)}")

    final = pd.DataFrame({
        "Test_ID": want_ids,
        "Predicted_Reference_Parameter": rp,
        "Validity_Label": labels,  # FROZEN official header (renamed here, only here)
    })
    assert len(final) == len(test_df) and list(final["Test_ID"]) == want_ids
    return final


def write_submission_csv(final_df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(p, index=False)
    return p
