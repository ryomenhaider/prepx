"""
prepx.eda.stats — numeric and categorical statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_numeric_stats(df: pd.DataFrame) -> dict:
    """Compute comprehensive numeric statistics."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = {}

    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 1:
            continue

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

        stats[col] = {
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "p25": round(float(q1), 4),
            "median": round(float(s.median()), 4),
            "p75": round(float(q3), 4),
            "max": round(float(s.max()), 4),
            "skewness": round(float(s.skew()), 4) if len(s) > 2 else 0.0,
            "kurtosis": round(float(s.kurtosis()), 4) if len(s) > 3 else 0.0,
            "outliers_iqr": n_out,
            "unique": int(s.nunique()),
            "zeros": int((s == 0).sum()),
            "negatives": int((s < 0).sum()),
        }

    return stats


def compute_categorical_stats(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Compute comprehensive categorical statistics."""
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    stats = {}

    for col in cat_cols:
        s = df[col].dropna()
        if len(s) < 1:
            continue

        vc = s.value_counts()
        probs = vc / vc.sum()
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        stats[col] = {
            "count": int(s.count()),
            "unique": int(s.nunique()),
            "top_categories": vc.head(top_n).to_dict(),
            "mode": str(vc.index[0]) if len(vc) > 0 else None,
            "mode_freq_pct": round(vc.iloc[0] / len(s) * 100, 2) if len(vc) > 0 else 0,
            "entropy": round(entropy, 4),
        }

    return stats


def compute_dtypes_summary(df: pd.DataFrame) -> dict:
    """Compute data types summary."""
    return {
        "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
        "datetime": df.select_dtypes(include=["datetime"]).columns.tolist(),
        "object": df.select_dtypes(include="object").columns.tolist(),
        "bool": df.select_dtypes(include="bool").columns.tolist(),
    }


def compute_overview(df: pd.DataFrame) -> dict:
    """Compute basic overview statistics."""
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "total_cells": df.size,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "duplicate_rows": int(df.duplicated().sum()),
    }