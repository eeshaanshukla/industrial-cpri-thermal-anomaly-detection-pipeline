"""schema.py — SINGLE SOURCE OF TRUTH for column names (Member 1 WRITES, all READ).

Frozen hour 0 from the official participant .xlsx (byte-verified headers).
NOBODY uses literal column strings outside this file — import these constants.

Real file headers carry unit suffixes (Applied_Voltage_kV etc.); the website
shows them without suffixes. Always use the suffixed (real) names below.

Final submission header for validity is FROZEN to `Validity_Label`
(Sample_Submission sheet literal + README Submission row).
"""

from __future__ import annotations


class C:
    """Raw column-name constants — exact participant-file headers."""

    TEST_ID = "Test_ID"
    VOLT = "Applied_Voltage_kV"
    CURR = "Load_Current_A"
    AMB = "Ambient_Temperature_C"
    DUR = "Test_Duration_min"
    S1 = "Sensor_S1"
    S2 = "Sensor_S2"
    S3 = "Sensor_S3"
    S4 = "Sensor_S4"
    RP = "Reference_Parameter"
    VALID = "Validity_Label"


class Sub:
    """FINAL submission headers (outputs/<TeamName>.csv) — FROZEN.

    Internal pipeline columns (Predicted_Validity, Anomaly_Score, Reason_Flags)
    are mapped/dropped by Member 4's output_writer.py — the ONLY place that
    maps internal -> final.
    """

    TEST_ID = "Test_ID"
    PRED_RP = "Predicted_Reference_Parameter"
    VALID = "Validity_Label"  # <-- frozen, NOT Predicted_Validity


# ---- Convenience groups (all derived from C, never literals) ----

#: Every raw input column present in BOTH train and test sheets.
INPUT_COLS: list[str] = [
    C.TEST_ID,
    C.VOLT,
    C.CURR,
    C.AMB,
    C.DUR,
    C.S1,
    C.S2,
    C.S3,
    C.S4,
]

#: Numeric model-input columns (everything except the Test_ID identifier).
NUMERIC_INPUT_COLS: list[str] = [
    C.VOLT,
    C.CURR,
    C.AMB,
    C.DUR,
    C.S1,
    C.S2,
    C.S3,
    C.S4,
]

#: Sensor columns (the only columns with measured missingness).
SENSOR_COLS: list[str] = [C.S1, C.S2, C.S3, C.S4]

#: Target columns (train only — NEVER in test data, NEVER in feature list).
TARGET_COLS: list[str] = [C.RP, C.VALID]

#: Forbidden in feature_columns.json — identifier + targets.
FORBIDDEN_FEATURES: frozenset[str] = frozenset({C.TEST_ID, C.RP, C.VALID})

#: Full expected train-sheet column order (raw file).
TRAIN_SHEET_COLS: list[str] = INPUT_COLS + TARGET_COLS

#: Full expected test-sheet column order (raw file).
TEST_SHEET_COLS: list[str] = list(INPUT_COLS)

#: Sheet names in the participant workbook.
SHEET_TRAIN = "Training_Data"
SHEET_TEST = "Test_Data"
SHEET_README = "README"
SHEET_SAMPLE_SUB = "Sample_Submission"

#: Internal pipeline columns (Members 2/3/4 — never in final CSV except via mapping).
INTERNAL_VALID = "Predicted_Validity"
INTERNAL_SCORE = "Anomaly_Score"
INTERNAL_REASONS = "Reason_Flags"
INTERNAL_RP = "Predicted_Reference_Parameter"
