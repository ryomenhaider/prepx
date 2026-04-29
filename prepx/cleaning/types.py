"""
prepx.cleaning.types — type coercion before missing value handling.

This module handles converting string columns to proper types BEFORE any missing
value processing. This is critical because strings like "N/A", "null", "-"
need to become NaN before missing handling can work correctly.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

NULL_VALUES = {
    "nan", "na", "n/a", "null", "none", "nil", "n", "none", "-", "--", "---",
    "", ".", " ", "  ", "null", "nulls", "na", "n a", "n.a.", "n.a", "na ",
    " not available", "not available", "not applicable", "missing", "unknown",
}

NUMERIC_PATTERNS = [
    r"^-?\d+$",
    r"^-?\d+\.?\d*$",
    r"^[\$€£¥]?\s*-?\d{1,3}(,\d{3})*(\.\d+)?$",
    r"^[\$€£¥]?\s*-?\d+(\.\d+)?\s*[\$€£¥]?$",
    r"^[\$€£¥]\s*\d+",
    r"^-\$[\d,]+\.?\d*$",
    r"^\d+\.\d+$",
    r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$",
]

NULL_PATTERN = re.compile(
    r"^(nan|na|n/?a|null|none|nil|n/?a\.?|missing|unknown|-|\\.\\.?|\\s*)$",
    re.IGNORECASE,
)


def coerce_types(
    df: pd.DataFrame,
    strip_whitespace: bool = True,
    normalize_nulls: bool = True,
    numeric_threshold: float = 0.8,
    datetime_threshold: float = 0.8,
) -> dict:
    """
    Coerce object(string) columns to numeric or datetime types.

    This runs BEFORE missing value handling so "N/A" strings become NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    strip_whitespace : bool
        Strip leading/trailing whitespace from strings.
    normalize_nulls : bool
        Convert common null representations to np.nan.
    numeric_threshold : float
        If >= this fraction of values parse as numeric, convert column.
    datetime_threshold : float
        If >= this fraction of values parse as datetime, convert column.

    Returns
    -------
    dict
        Report of coerces: {"numeric": [...], "datetime": [...], "nulls_normalized": [...]}
    """
    report = {"numeric": [], "datetime": [], "nulls_normalized": []}
    df = df.copy()

    obj_cols = df.select_dtypes(include="object").columns.tolist()

    for col in obj_cols:
        series = df[col]

        if strip_whitespace:
            series = series.apply(lambda x: x.strip() if isinstance(x, str) else x)
            df[col] = series

        if normalize_nulls:
            null_indicators = series.astype(str).str.lower().isin(NULL_VALUES)
            null_count = null_indicators.sum()
            if null_count > 0:
                df.loc[null_indicators, col] = np.nan
                report["nulls_normalized"].append(col)

    for col in df.select_dtypes(include="object").columns:
        if _try_numeric(df, col, numeric_threshold):
            report["numeric"].append(col)
        else:
            if _try_datetime(df, col, datetime_threshold):
                report["datetime"].append(col)

    return report


def _try_numeric(df: pd.DataFrame, col: str, threshold: float) -> bool:
    """Try to coerce a column to numeric. Returns True if successful."""
    original = df[col].copy()
    non_null = df[col].dropna()

    if len(non_null) == 0:
        return False

    cleaned = (
        non_null.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("¥", "", regex=False)
        .str.strip()
    )

    converted = pd.to_numeric(cleaned, errors="coerce")
    success_rate = converted.notna().sum() / len(non_null)

    if success_rate >= threshold:
        df[col] = pd.to_numeric(
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"[\$€£¥]", "", regex=True)
            .str.strip(),
            errors="coerce",
        )
        return True

    return False


def _try_datetime(df: pd.DataFrame, col: str, threshold: float) -> bool:
    """Try to coerce a column to datetime. Returns True if successful."""
    non_null = df[col].dropna()

    if len(non_null) == 0:
        return False

    try:
        converted = pd.to_datetime(non_null, errors="coerce")
        success_rate = converted.notna().sum() / len(non_null)

        if success_rate >= threshold:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            return True
    except Exception:
        pass

    return False


def detect_mixed_types(df: pd.DataFrame, col: str) -> dict:
    """
    Detect if a column has mixed types (e.g., some numeric, some strings).

    Returns
    -------
    dict
        {"type_counts": {str: 10, int: 5, float: 2, "mixed": 3}, "is_mixed": bool}
    """
    series = df[col].dropna()
    type_counts = {}

    for val in series:
        if isinstance(val, bool):
            t = "bool"
        elif isinstance(val, (int, np.integer)):
            t = "int"
        elif isinstance(val, (float, np.floating)):
            t = "float"
        elif isinstance(val, str):
            t = "str"
        else:
            t = str(type(val).__name__)

        type_counts[t] = type_counts.get(t, 0) + 1

    is_mixed = len(type_counts) > 1

    return {"type_counts": type_counts, "is_mixed": is_mixed}


def parse_mixed_numeric(
    df: pd.DataFrame,
    col: str,
    default_dtype: Optional[str] = "float",
) -> pd.Series:
    """
    Parse a column with mixed numeric formats ("1,200", "$100", "N/A").

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name.
    default_dtype : str
        "float" or "int".

    Returns
    -------
    pd.Series
        Parsed series with NaN for unparseable values.
    """
    s = df[col].astype(str)

    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[\$€£¥]", "", regex=True)
    s = s.str.strip()

    s = s.replace("nan", np.nan)
    s = s.replace("na", np.nan)
    s = s.replace("null", np.nan)
    s = s.replace("none", np.nan)
    s = s.replace("", np.nan)

    if default_dtype == "int":
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    else:
        return pd.to_numeric(s, errors="coerce")