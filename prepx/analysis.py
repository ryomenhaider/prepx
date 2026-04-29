"""
prepx.eda — main EDA function using modular architecture.

This module provides the main eda() function using the modular EDA components.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from prepx.eda.stats import (
    compute_numeric_stats,
    compute_categorical_stats,
    compute_dtypes_summary,
    compute_overview,
)
from prepx.eda.distributions import fit_distributions, detect_multimodality
from prepx.eda.correlations import compute_correlations, compute_vif, compute_correlation_with_target
from prepx.eda.drifts import compute_drift, compute_drift_summary


def eda(
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    test_data: Optional[pd.DataFrame] = None,
    correlations: Optional[list] = None,
    check_drift: bool = False,
    detect_multimodality: bool = False,
    fit_distributions: bool = False,
    top_n_categories: int = 10,
    correlation_threshold: float = 0.7,
    verbose: bool = True,
) -> dict:
    """
    Perform comprehensive EDA.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Target column for analysis.
    test_data : pd.DataFrame
        Test data for drift detection.
    correlations : list
        Correlation methods: "pearson", "spearman", "kendall".
    check_drift : bool
        Check for train/test drift.
    detect_multimodality : bool
        Detect multimodal distributions.
    fit_distributions : bool
        Fit distributions (requires scipy).
    top_n_categories : int
        Number of top categories to show.
    correlation_threshold : float
        Threshold for high correlation detection.
    verbose : bool
        Print report.

    Returns
    -------
    dict
        Comprehensive EDA report.
    """
    report = {}

    report["overview"] = compute_overview(df)
    report["dtypes"] = compute_dtypes_summary(df)
    report["numeric_stats"] = compute_numeric_stats(df)
    report["categorical_stats"] = compute_categorical_stats(df, top_n=top_n_categories)

    missing = df.isnull().sum()
    report["missing"] = {
        "total": int(missing.sum()),
        "per_column": {col: int(missing[col]) for col in df.columns if missing[col] > 0},
    }

    if correlations is None:
        correlations = ["pearson"]
    report["correlations"] = compute_correlations(
        df,
        methods=correlations,
        threshold=correlation_threshold,
    )

    if target and target in df.columns:
        report["target_analysis"] = _analyze_target(df, target, correlations)

    if test_data is not None and check_drift:
        report["drift_analysis"] = compute_drift(df, test_data)

    if detect_multimodality:
        report["multimodality"] = _detect_all_multimodal(df)

    if fit_distributions:
        try:
            report["distributions"] = _fit_all_distributions(df)
        except ImportError as e:
            report["distributions"] = {"error": str(e)}

    report["warnings"] = _build_warnings(report, correlation_threshold)

    if verbose:
        _print_eda_report(report, target)

    return report


def _analyze_target(df: pd.DataFrame, target: str, methods: list) -> dict:
    """Analyze target column."""
    result = {}
    t = df[target]

    if t.dtype == "object" or t.nunique() < 20:
        vc = t.value_counts()
        result["class_balance"] = vc.to_dict()
        if len(vc) > 1:
            result["imbalance_ratio"] = round(float(vc.iloc[0] / vc.iloc[-1]), 2)

    if pd.api.types.is_numeric_dtype(t):
        for method in methods:
            if method in ["pearson", "spearman", "kendall"]:
                corrs = compute_correlation_with_target(df, target, methods=[method])
                result[f"feature_correlations_{method}"] = corrs.get(method, {})

    return result


def _detect_all_multimodal(df: pd.DataFrame) -> dict:
    """Detect multimodality for all numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results = {}

    for col in num_cols:
        results[col] = detect_multimodality(df[col])

    multimodal = [k for k, v in results.items() if v.get("is_multimodal")]
    return {"results": results, "multimodal_columns": multimodal}


def _fit_all_distributions(df: pd.DataFrame) -> dict:
    """Fit distributions to all numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results = {}

    for col in num_cols:
        try:
            results[col] = fit_distributions(df[col])
        except Exception as e:
            results[col] = {"error": str(e)}

    return results


def _build_warnings(report: dict, corr_threshold: float) -> list:
    """Build diagnostic warnings."""
    warnings_list = []

    if report["overview"].get("duplicate_rows", 0) > 0:
        warnings_list.append({
            "level": "warn",
            "message": f"{report['overview']['duplicate_rows']} duplicate rows detected",
        })

    missing = report.get("missing", {})
    for col, count in missing.get("per_column", {}).items():
        if count / report["overview"]["rows"] > 0.3:
            warnings_list.append({
                "level": "warn",
                "message": f"Column '{col}' has >30% missing values",
            })

    for method, corr_data in report.get("correlations", {}).items():
        high_pairs = corr_data.get("high_pairs", [])
        for pair in high_pairs:
            warnings_list.append({
                "level": "warn",
                "message": f"High correlation ({pair['r']}) between '{pair['col_a']}' and '{pair['col_b']}'",
            })

    for col, stats in report.get("numeric_stats", {}).items():
        if abs(stats.get("skewness", 0)) > 1.5:
            warnings_list.append({
                "level": "info",
                "message": f"Column '{col}' is highly skewed (skew={stats['skewness']})",
            })

        if stats.get("outliers_iqr", 0) > 0:
            pct = round(stats["outliers_iqr"] / stats["count"] * 100, 1)
            warnings_list.append({
                "level": "info",
                "message": f"Column '{col}' has {stats['outliers_iqr']} outliers ({pct}%)",
            })

    if "drift_analysis" in report:
        drift = report["drift_analysis"]
        drifted = len(drift.get("drifted", []))
        if drifted > 0:
            warnings_list.append({
                "level": "warn",
                "message": f"{drifted} columns show drift between train/test",
            })

    if "multimodal_columns" in report.get("multimodality", {}):
        multimodal = report["multimodality"]["multimodal_columns"]
        if multimodal:
            warnings_list.append({
                "level": "info",
                "message": f"{len(multimodal)} columns have multimodal distributions",
            })

    return warnings_list


def _print_eda_report(report: dict, target: Optional[str] = None):
    """Print human-readable EDA report."""
    w = 62
    sep = "─" * w

    print(f"\n{'━' * w}")
    print(f"  prepx  ·  EDA Report")
    print(f"{'━' * w}")

    ov = report["overview"]
    print(f"\n  OVERVIEW")
    print(sep)
    print(f"  {'Rows':30s} {ov['rows']:>10,}")
    print(f"  {'Columns':30s} {ov['cols']:>10,}")
    print(f"  {'Duplicate rows':30s} {ov['duplicate_rows']:>10,}")
    print(f"  {'Memory':30s} {ov['memory_mb']:>9.3f} MB")

    dt = report["dtypes"]
    print(f"\n  DATA TYPES")
    print(sep)
    for dtype, cols in dt.items():
        if dtype != "per_column" and cols:
            print(f"  {dtype:10s} ({len(cols)}): {', '.join(cols[:5])}{'…' if len(cols) > 5 else ''}")

    miss = report["missing"]
    if miss["total"] > 0:
        print(f"\n  MISSING  —  {miss['total']} cells")
        print(sep)
        for col, count in list(miss["per_column"].items())[:5]:
            print(f"    {col:30s} {count:>6}")

    if report["numeric_stats"]:
        print(f"\n  NUMERIC STATS")
        print(sep)
        hdr = f"  {'Column':20s} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Skew':>8}"
        print(hdr)
        for col, stats in list(report["numeric_stats"].items())[:8]:
            print(f"  {col:20s} {stats['mean']:>8.3f} {stats['std']:>8.3f} {stats['min']:>8.3f} {stats['max']:>8.3f} {stats['skewness']:>8.3f}")

    if report["warnings"]:
        print(f"\n  WARNINGS  ({len(report['warnings'])})")
        print(sep)
        for w_item in report["warnings"]:
            icon = "⚠" if w_item["level"] == "warn" else "ℹ"
            print(f"  {icon} {w_item['message']}")

    print(f"\n{'━' * w}\n")