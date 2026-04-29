"""
prepx.cleaning.outliers — outlier detection and handling.

Supports: IQR, Z-score, Modified Z-score, Isolation Forest, winsorizing.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    action: str = "capped",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    modified_z_threshold: float = 3.5,
    contamination: float = 0.05,
    columns: Optional[list] = None,
) -> dict:
    """
    Detect and handle outliers in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    method : str
        Detection method: "iqr", "zscore", "modified_zscore", "isolation_forest".
    action : str
        How to handle: "capped", "removed", "flagged".
    iqr_multiplier : float
        IQR multiplier for fence calculation (default 1.5).
    z_threshold : float
        Z-score threshold for outlier detection.
    modified_z_threshold : float
        Modified Z-score threshold (robust to outliers).
    contamination : float
        Expected proportion of outliers (for IsolationForest).
    columns : list
        Specific columns to process. If None, all numeric columns.

    Returns
    -------
    dict
        Report of outlier actions.
    """
    report = {"method": method, "action": action, "columns": {}}
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        col_data = df[col].dropna()
        if len(col_data) < 4:
            continue

        if method == "iqr":
            detected = _iqr_outliers(col_data, iqr_multiplier)
            lo, hi = detected["bounds"]
        elif method == "zscore":
            detected = _zscore_outliers(col_data, z_threshold)
            lo, hi = detected["bounds"]
        elif method == "modified_zscore":
            detected = _modified_zscore_outliers(col_data, modified_z_threshold)
            lo, hi = detected["bounds"]
        elif method == "isolation_forest":
            detected = _isolation_forest_outliers(df[[col]], contamination)
            lo, hi = detected["bounds"]
        else:
            raise ValueError(f"Unknown method: {method}")

        n_outliers = detected["count"]
        if n_outliers == 0:
            continue

        if action == "capped":
            df[col] = df[col].clip(lo, hi)
            report["columns"][col] = {"outliers": n_outliers, "bounds": (lo, hi), "action": "capped"}

        elif action == "removed":
            mask = (df[col] >= lo) & (df[col] <= hi)
            df = df[mask]
            report["columns"][col] = {"outliers": n_outliers, "action": "removed"}

        elif action == "flagged":
            df[f"{col}_outlier"] = ~((df[col] >= lo) & (df[col] <= hi))
            df[f"{col}_outlier"] = df[f"{col}_outlier"].fillna(False).astype(int)
            report["columns"][col] = {
                "outliers": n_outliers,
                "indicator": f"{col}_outlier",
                "action": "flagged",
            }

    return report


def _iqr_outliers(
    series: pd.Series,
    multiplier: float = 1.5,
) -> dict:
    """Detect outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - multiplier * iqr
    hi = q3 + multiplier * iqr

    mask = (series < lo) | (series > hi)
    count = int(mask.sum())

    return {"count": count, "bounds": (lo, hi), "indices": series[mask].index.tolist()}


def _zscore_outliers(
    series: pd.Series,
    threshold: float = 3.0,
) -> dict:
    """Detect outliers using Z-score method."""
    mu = series.mean()
    sigma = series.std()

    if sigma == 0:
        return {"count": 0, "bounds": (mu, mu), "indices": []}

    z_scores = (series - mu) / sigma
    mask = z_scores.abs() > threshold
    count = int(mask.sum())

    lo = mu - threshold * sigma
    hi = mu + threshold * sigma

    return {"count": count, "bounds": (lo, hi), "indices": series[mask].index.tolist()}


def _modified_zscore_outliers(
    series: pd.Series,
    threshold: float = 3.5,
) -> dict:
    """Detect outliers using Modified Z-score (MAD-based)."""
    median = series.median()
    mad = (series - median).abs().median()

    if mad == 0:
        return {"count": 0, "bounds": (median, median), "indices": []}

    modified_z = 0.6745 * (series - median) / mad
    mask = modified_z.abs() > threshold
    count = int(mask.sum())

    lo = median - threshold * mad / 0.6745
    hi = median + threshold * mad / 0.6745

    return {"count": count, "bounds": (lo, hi), "indices": series[mask].index.tolist()}


def _isolation_forest_outliers(
    df: pd.DataFrame,
    contamination: float = 0.05,
) -> dict:
    """Detect outliers using Isolation Forest."""
    if not HAS_SKLEARN:
        raise ImportError(
            "sklearn required for IsolationForest. Install: pip install scikit-learn"
        )

    iso = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso.fit_predict(df)
    scores = iso.score_samples(df)

    lo = df.min().min()
    hi = df.max().max()

    mask = predictions == -1
    count = int(mask.sum())

    return {"count": count, "bounds": (lo, hi), "indices": df[mask].index.tolist()}


def winsorize(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> pd.DataFrame:
    """
    Winsorize values at specified percentiles (cap extremes).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list
        Columns to winsorize. If None, all numeric.
    lower_percentile : float
        Lower percentile for capping.
    upper_percentile : float
        Upper percentile for capping.

    Returns
    -------
    pd.DataFrame
        Winsorized DataFrame.
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        lo = df[col].quantile(lower_percentile)
        hi = df[col].quantile(upper_percentile)
        df[col] = df[col].clip(lo, hi)

    return df


def detect_outlier_summary(df: pd.DataFrame) -> dict:
    """Run all outlier detection methods and summarize."""
    results = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        iqr = _iqr_outliers(series)
        zscore = _zscore_outliers(series)
        mzscore = _modified_zscore_outliers(series)

        results[col] = {
            "iqr_outliers": iqr["count"],
            "zscore_outliers": zscore["count"],
            "modified_zscore_outliers": mzscore["count"],
        }

    return results