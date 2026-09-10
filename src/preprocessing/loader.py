"""loader.py — raw participant-file loader (Member 1).

INPUT:  participant .xlsx (sheets Training_Data / Test_Data), unmodified,
        in data/raw/. `.csv` fallback branch for hidden/second-round data.
OUTPUT: in-memory pandas.DataFrame (callers also checkpoint CSVs).

Contract:
    load_raw(path, sheet=None) -> pd.DataFrame
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .schema import SHEET_TEST, SHEET_TRAIN


def _is_excel(path: str | os.PathLike) -> bool:
    return str(path).lower().endswith((".xlsx", ".xls", ".xlsm"))


def load_raw(path: str | os.PathLike, sheet: str | None = None) -> pd.DataFrame:
    """Load raw train or test data from .xlsx (primary) or .csv (fallback).

    Args:
        path: path to the participant .xlsx or a fallback .csv.
        sheet: sheet name for workbooks. If None, defaults to Training_Data
            when the filename hints 'train', Test_Data when it hints 'test',
            else Training_Data.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if the sheet is missing from the workbook.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raw data file not found: {p.resolve()}")

    if _is_excel(p):
        if sheet is None:
            lname = p.name.lower()
            if "test" in lname and "train" not in lname:
                sheet = SHEET_TEST
            else:
                sheet = SHEET_TRAIN
        try:
            df = pd.read_excel(p, sheet_name=sheet, engine="openpyxl")
        except ValueError as exc:  # bad sheet name -> list what's there
            xls = pd.ExcelFile(p, engine="openpyxl")
            raise ValueError(
                f"Sheet {sheet!r} not in {p.name}. Available: {xls.sheet_names}"
            ) from exc
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Sheet {sheet!r} did not return a DataFrame.")
        return df

    # .csv fallback branch (hidden/second-round data may ship as CSVs)
    return pd.read_csv(p)


def load_train_test(
    raw_path: str | os.PathLike,
    train_sheet: str = SHEET_TRAIN,
    test_sheet: str = SHEET_TEST,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both sheets from one participant workbook.

    If `raw_path` is a directory, looks for the first *.xlsx inside it;
    if it is a CSV path, raises (use load_raw per-file for the CSV case).
    """
    p = Path(raw_path)
    if p.is_dir():
        cands = sorted(p.glob("*.xlsx")) + sorted(p.glob("*.XLSX"))
        if not cands:
            raise FileNotFoundError(f"No .xlsx found in directory {p.resolve()}")
        p = cands[0]
    if not _is_excel(p):
        raise ValueError(
            "load_train_test expects the participant .xlsx; "
            f"got {p.name}. Use load_raw() per-file for CSV fallbacks."
        )
    return load_raw(p, train_sheet), load_raw(p, test_sheet)
