"""make_synthetic.py — DEV-ONLY synthetic participant workbook.

The REAL .xlsx comes from the organizers. This generator builds a
spec-faithful stand-in (1000 train / 350 test rows, 866/134 Valid/Invalid,
sensor-only missingness, 4 exact-duplicate test pairs, S4 ~uncorrelated)
so Member 1's pipeline is testable before the real file lands.
Run:  python -m preprocessing.make_synthetic   (from src/)
Deterministic (seed 42). Output: data/raw/SYNTHETIC_PARTICIPANT.xlsx
"""

from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_TRAIN, N_TEST = 1000, 350
N_INVALID_TRAIN = 134


def _base_inputs(n: int, start_id: int, prefix: str) -> pd.DataFrame:
    # Two operating regimes: normal (93%) + high-current regime (7%, still Valid-able)
    regime = rng.random(n) < 0.07
    volt = np.where(regime, rng.uniform(220, 400, n), rng.uniform(66, 230, n))
    curr = np.where(regime, rng.uniform(1200, 2000, n), rng.uniform(100, 1200, n))
    amb = rng.uniform(15, 45, n)
    dur = rng.choice([15, 30, 60, 120, 240], n, p=[0.15, 0.25, 0.3, 0.2, 0.1]).astype(float)
    return pd.DataFrame({
        "Test_ID": [f"{prefix}-{i:04d}" for i in range(start_id, start_id + n)],
        "Applied_Voltage_kV": np.round(volt, 1),
        "Load_Current_A": np.round(curr, 1),
        "Ambient_Temperature_C": np.round(amb, 1),
        "Test_Duration_min": dur,
        "_regime": regime.astype(int),
    })


def _true_rp(df: pd.DataFrame) -> np.ndarray:
    v = df["Applied_Voltage_kV"].to_numpy()
    i = df["Load_Current_A"].to_numpy()
    d = df["Test_Duration_min"].to_numpy()
    rp = 4.0 + 0.028 * i * np.sqrt(d / 60.0) + 0.045 * v + rng.normal(0, 1.5, len(df))
    return np.clip(rp, 1.0, None)


def _sensors(df: pd.DataFrame, rp: np.ndarray, invalid_mask: np.ndarray) -> pd.DataFrame:
    n = len(df)
    s1 = rp + rng.normal(0, 1.5, n)
    s2 = rp + rng.normal(0, 1.5, n)
    s3 = 0.92 * rp + rng.normal(0, 2.0, n)
    # S4: auxiliary, ~uncorrelated by design
    s4 = rng.uniform(-5, 55, n)
    # Fault injection on Invalid rows (each a different failure mode)
    idx = np.where(invalid_mask)[0]
    modes = rng.choice(["spike", "stuck", "drop", "swap", "bias"], len(idx))
    for k, m in zip(idx, modes):
        if m == "spike":
            s1[k] += rng.uniform(30, 80)
        elif m == "stuck":
            s2[k] = 25.0  # frozen sensor
        elif m == "drop":
            s3[k] = np.nan  # will also count as missing
        elif m == "swap":
            s1[k], s2[k] = s2[k], s1[k]
            s3[k] += rng.uniform(-25, -10)
        else:
            s1[k] += rng.uniform(12, 20)
            s3[k] += rng.uniform(-20, -12)
    out = df.copy()
    out["Sensor_S1"] = np.round(s1, 2)
    out["Sensor_S2"] = np.round(s2, 2)
    out["Sensor_S3"] = np.round(s3, 2)
    out["Sensor_S4"] = np.round(s4, 2)
    return out


def _sprinkle_missing(df: pd.DataFrame, rate: float = 0.03) -> pd.DataFrame:
    out = df.copy()
    for c in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]:
        m = rng.random(len(out)) < rate
        out.loc[m, c] = np.nan
    return out


def build() -> dict[str, pd.DataFrame]:
    train = _base_inputs(N_TRAIN, 1, "TR")
    test = _base_inputs(N_TEST, 1, "TE")
    rp_train = _true_rp(train)
    rp_test_hidden = _true_rp(test)

    invalid_train = np.zeros(N_TRAIN, bool)
    invalid_train[rng.choice(N_TRAIN, N_INVALID_TRAIN, replace=False)] = True
    # hidden test labels (for OUR validation only — never shipped in Test_Data)
    invalid_test = rng.random(N_TEST) < (N_INVALID_TRAIN / N_TRAIN)

    train = _sensors(train, rp_train, invalid_train)
    test = _sensors(test, rp_test_hidden, invalid_test)
    train = _sprinkle_missing(train)
    test = _sprinkle_missing(test)

    train["Reference_Parameter"] = np.round(rp_train, 2)
    train["Validity_Label"] = np.where(invalid_train, "Invalid", "Valid")

    # 4 exact-duplicate test pairs (identical measurements, distinct Test_IDs)
    for j in range(4):
        src = test.iloc[j * 10].copy()
        dup_idx = j * 10 + 5
        for c in test.columns:
            if c != "Test_ID":
                test.loc[test.index[dup_idx], c] = src[c]

    train = train.drop(columns=["_regime"])
    test_pub = test.drop(columns=["_regime"])
    hidden = test[["Test_ID"]].copy()
    hidden["Hidden_Reference_Parameter"] = np.round(rp_test_hidden, 2)
    hidden["Hidden_Validity_Label"] = np.where(invalid_test, "Invalid", "Valid")

    readme = pd.DataFrame({"Item": ["Submission"], "Detail": ["<TeamName>.csv: Test_ID,Predicted_Reference_Parameter,Validity_Label"]})
    sample = pd.DataFrame({
        "Test_ID": test_pub["Test_ID"].head(3),
        "Predicted_Reference_Parameter": [0.0, 0.0, 0.0],
        "Validity_Label": ["Valid", "Valid", "Valid"],
    })
    return {"README": readme, "Training_Data": train, "Test_Data": test_pub,
            "Sample_Submission": sample, "_hidden_test_labels": hidden}


def main(out_path: str = "data/raw/SYNTHETIC_PARTICIPANT.xlsx") -> None:
    import os
    sheets = build()
    hidden = sheets.pop("_hidden_test_labels")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    hpath = os.path.join(os.path.dirname(out_path), "_HIDDEN_test_labels.csv")
    hidden.to_csv(hpath, index=False)
    tr, te = sheets["Training_Data"], sheets["Test_Data"]
    print(f"wrote {out_path}: train={len(tr)} test={len(te)}")
    print("train validity:", tr["Validity_Label"].value_counts().to_dict())
    print("train missing cells:", int(tr.filter(like='Sensor').isna().sum().sum()))
    print("test missing cells:", int(te.filter(like='Sensor').isna().sum().sum()))
    print(f"(dev-only hidden labels -> {hpath}; NEVER feed to models)")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/SYNTHETIC_PARTICIPANT.xlsx")
