"""
prepx.cleaning.missing — missing value handling strategies.

Supports: ffill, bfill, median, mode, KNN imputation, MICE (multiple imputation),
and missing indicators.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from sklearn.impute import KNNImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


impute_engine = None


def handle_missing(
    df: pd.DataFrame,
    method: str = "auto",
    missing_indicators: bool = False,
    ffill_direction: str = "forward",
) -> dict:
    """
    Handle missing values using various strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (should already have types coerced).
    method : str
        Strategy: "auto", "ffill", "bfill", "drop", "median", "mode",
        "knn", "mice".
    missing_indicators : bool
        Add indicator columns for each column with missing values.
    ffill_direction : str
        "forward" or "backward" for ffill/bfill.

    Returns
    -------
    dict
        Report of missing handling actions.
    """
    report = {"action": method, "columns": {}, "indicators_added": []}
    df = df.copy()

    missing_cols = df.columns[df.isnull().any()].tolist()

    if not missing_cols:
        report["action"] = "none"
        return report

    if method == "drop":
        before = len(df)
        df = df.dropna()
        report["rows_removed"] = before - len(df)

    elif method == "ffill":
        df = df.ffill() if ffill_direction == "forward" else df.bfill()
        for col in missing_cols:
            report["columns"][col] = {"filled": ffill_direction}

    elif method == "bfill":
        df = df.bfill()
        for col in missing_cols:
            report["columns"][col] = {"filled": "backward"}

    elif method == "auto":
        report = _auto_fill(df, missing_cols, report)

    elif method == "median":
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in missing_cols:
            if col in num_cols:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                report["columns"][col] = {"filled": f"median ({median_val})"}
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                    report["columns"][col] = {"filled": f"mode ({mode_val[0]})"}

    elif method == "mode":
        for col in missing_cols:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                report["columns"][col] = {"filled": f"mode ({mode_val[0]})"}

    elif method == "knn":
        if not HAS_SKLEARN:
            raise ImportError(
                "sklearn required for KNN imputation. Install: pip install scikit-learn"
            )
        report = _knn_fill(df, missing_cols, report)

    elif method == "mice":
        if not HAS_SKLEARN:
            raise ImportError(
                "sklearn required for MICE imputation. Install: pip install scikit-learn"
            )
        report = _mice_fill(df, missing_cols, report)

    if missing_indicators:
        df, indicators = _add_missing_indicators(df, missing_cols)
        report["indicators_added"] = indicators

    return report


def _auto_fill(df: pd.DataFrame, missing_cols: list, report: dict) -> dict:
    """Auto-fill: median/mode for numeric, mode for categorical."""
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in missing_cols:
        if col in num_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report["columns"][col] = {"filled": f"median ({median_val})"}
        else:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                report["columns"][col] = {"filled": f"mode ({mode_val[0]})"}

    return report


def _knn_fill(df: pd.DataFrame, missing_cols: list, report: dict) -> dict:
    """KNN imputation using sklearn."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return report

    cols_to_impute = [c for c in missing_cols if c in numeric_cols]

    if not cols_to_impute:
        return report

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    for col in cols_to_impute:
        n_filled = df[col].isnull().sum()
        report["columns"][col] = {"filled": "knn", "filled_count": n_filled}

    return report


def _mice_fill(df: pd.DataFrame, missing_cols: list, report: dict) -> dict:
    """MICE (Multiple Imputation by Chained Equations) using sklearn."""
    from sklearn.impute import IterativeImputer

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_impute = [c for c in missing_cols if c in numeric_cols]

    if not cols_to_impute:
        return report

    imputer = IterativeImputer(max_iter=10, random_state=42)
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    for col in cols_to_impute:
        n_filled = df[col].isnull().sum()
        report["columns"][col] = {"filled": "mice", "filled_count": n_filled}

    return report


def _add_missing_indicators(
    df: pd.DataFrame,
    missing_cols: list,
) -> tuple[pd.DataFrame, list]:
    """Add binary indicator columns for missing values."""
    indicators = []

    for col in missing_cols:
        indicator_name = f"{col}_missing"
        df[indicator_name] = df[col].isnull().astype(int)
        indicators.append(indicator_name)

    return df, indicators


def compute_missing_summary(df: pd.DataFrame) -> dict:
    """Compute detailed missing value statistics."""
    total = df.size
    missing = df.isnull().sum().sum()
    missing_pct = (missing / total * 100) if total > 0 else 0

    cols_with_missing = df.columns[df.isnull().any()].tolist()

    per_col = {}
    for col in cols_with_missing:
        col_missing = df[col].isnull().sum()
        per_col[col] = {
            "count": int(col_missing),
            "pct": round(col_missing / len(df) * 100, 2),
        }

    return {
        "total_missing": int(missing),
        "pct_missing": round(missing_pct, 2),
        "cols_with_missing": len(cols_with_missing),
        "per_column": per_col,
    }