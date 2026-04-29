"""
prepx.cleaner — main clean() function using modular cleaning pipeline.

This module provides the main clean() function using the modular architecture:
1. Type coercion (strings → numeric/datetime) - FIRST
2. Missing value handling
3. Outlier handling
4. Deduplication last (after data is clean)
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from prepx.cleaning import (
    coerce_types,
    handle_missing as handle_miss,
    handle_outliers as handle_outs,
    deduplicate as dedupe,
    standardize_columns as std_cols,
    compute_missing_summary,
)


def clean(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    dedupe_method: str = "exact",
    dedupe_threshold: float = 0.85,
    handle_missing: str = "auto",
    missing_method: str = "auto",
    missing_indicators: bool = False,
    missing_threshold: float = 0.6,
    fix_dtypes: bool = True,
    strip_whitespace: bool = True,
    standardize_columns: bool = True,
    naming_style: str = "snake",
    remove_outliers: bool = False,
    outlier_method: str = "iqr",
    outlier_action: str = "capped",
    outlier_threshold: float = 3.0,
    drop_constant_cols: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean a DataFrame and return ``(cleaned_df, report_dict)``.

    Pipeline Order (important):
    1. Type coercion (strings → numeric/datetime) - enables missing value detection
    2. Column standardization (rename to snake_case)
    3. Whitespace stripping
    4. Duplicate removal
    5. Constant column removal
    6. High-missing column removal
    7. Missing value handling
    8. Outlier handling

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    drop_duplicates : bool
        Remove duplicate rows.
    dedupe_method : str
        "exact" or "fuzzy".
    dedupe_threshold : float
        Similarity threshold for fuzzy dedupe.
    handle_missing : str
        Deprecated, use missing_method.
    missing_method : str
        Strategy: "auto", "drop", "ffill", "bfill", "median", "mode", "knn", "mice".
    missing_indicators : bool
        Add indicator columns for missing values.
    missing_threshold : float
        Drop columns with > this fraction missing.
    fix_dtypes : bool
        Coerce types.
    strip_whitespace : bool
        Strip whitespace from strings.
    standardize_columns : bool
        Rename columns to snake_case.
    naming_style : str
        "snake", "camel", "pascal", "kebab".
    remove_outliers : bool
        Handle outliers.
    outlier_method : str
        "iqr", "zscore", "modified_zscore", "isolation_forest".
    outlier_action : str
        "capped", "removed", "flagged".
    outlier_threshold : float
        Threshold for zscore method.
    drop_constant_cols : bool
        Drop constant columns.
    verbose : bool
        Print report.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Cleaned DataFrame and report.
    """
    if handle_missing != "auto":
        import warnings
        warnings.warn(
            "handle_missing is deprecated, use missing_method",
            DeprecationWarning,
            stacklevel=2,
        )
        missing_method = handle_missing

    report = {"initial_shape": df.shape, "steps": []}
    df = df.copy()

    if standardize_columns:
        df, col_report = std_cols(df, naming_style=naming_style)
        if col_report.get("renamed"):
            report["steps"].append({"step": "standardize_columns", **col_report})

    if fix_dtypes:
        type_report = coerce_types(df, strip_whitespace=strip_whitespace)
        if type_report["numeric"] or type_report["datetime"] or type_report["nulls_normalized"]:
            report["steps"].append({"step": "fix_dtypes", **type_report})

    if drop_duplicates:
        before = len(df)
        df, dup_report = dedupe(df, method=dedupe_method, threshold=dedupe_threshold)
        dup_report.pop("method", None)
        dup_report.pop("original_rows", None)
        if dup_report.get("duplicates_removed", 0) > 0:
            report["steps"].append({"step": "deduplicate", **dup_report})

    if drop_constant_cols:
        const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if const_cols:
            df = df.drop(columns=const_cols)
            report["steps"].append({"step": "drop_constant_cols", "dropped": const_cols})

    missing_rates = df.isnull().mean()
    high_miss = missing_rates[missing_rates > missing_threshold].index.tolist()
    if high_miss:
        df = df.drop(columns=high_miss)
        report["steps"].append({
            "step": "drop_high_missing_cols",
            "threshold": missing_threshold,
            "dropped": high_miss,
        })

    miss_report = handle_miss(df, method=missing_method, missing_indicators=missing_indicators)
    if miss_report.get("columns"):
        report["steps"].append({"step": "handle_missing", **miss_report})

    if remove_outliers:
        out_report = handle_outs(
            df,
            method=outlier_method,
            action=outlier_action,
            z_threshold=outlier_threshold,
        )
        if out_report.get("columns"):
            report["steps"].append({"step": "handle_outliers", **out_report})

    report["final_shape"] = df.shape
    report["rows_removed"] = report["initial_shape"][0] - df.shape[0]
    report["cols_removed"] = report["initial_shape"][1] - df.shape[1]

    if verbose:
        _print_clean_report(report)

    return df, report


def _print_clean_report(report: dict):
    """Print human-readable cleaning report."""
    w = 62
    sep = "─" * w

    print(f"\n{'━' * w}")
    print(f"  prepx  ·  Cleaning Report")
    print(f"{'━' * w}")
    print(f"  {'Initial shape':30s} {report['initial_shape'][0]:>6} rows × {report['initial_shape'][1]} cols")
    print(f"  {'Final shape':30s} {report['final_shape'][0]:>6} rows × {report['final_shape'][1]} cols")
    print(f"  {'Rows removed':30s} {report['rows_removed']:>6}")
    print(f"  {'Columns removed':30s} {report['cols_removed']:>6}")
    print(sep)

    for step in report["steps"]:
        name = step.get("step", "")

        if name == "standardize_columns":
            renamed = step.get("renamed", {})
            count = step.get("count", len(renamed))
            print(f"\n  ✦ Column names → {count} renamed")

        elif name == "fix_dtypes":
            total = (
                len(step.get("numeric", []))
                + len(step.get("datetime", []))
                + len(step.get("nulls_normalized", []))
            )
            if total > 0:
                print(f"\n  ✦ Type coercion  ({total} columns)")
                if step.get("numeric"):
                    print(f"      Numeric  : {', '.join(step['numeric'])}")
                if step.get("datetime"):
                    print(f"      Datetime : {', '.join(step['datetime'])}")
                if step.get("nulls_normalized"):
                    print(f"      Nulls normalized: {', '.join(step['nulls_normalized'])}")

        elif name == "deduplicate":
            print(f"\n  ✦ Duplicates removed  : {step.get('duplicates_removed', 0)}")

        elif name == "drop_constant_cols":
            dropped = step.get("dropped", [])
            if dropped:
                print(f"\n  ✦ Constant columns dropped  ({len(dropped)}): {', '.join(dropped[:5])}")

        elif name == "drop_high_missing_cols":
            dropped = step.get("dropped", [])
            if dropped:
                print(f"\n  ✦ High-missing columns dropped  ({len(dropped)}): {', '.join(dropped[:5])}")

        elif name == "handle_missing":
            cols = step.get("columns", {})
            if cols:
                print(f"\n  ✦ Missing values handled  ({len(cols)} columns)")
                for col, info in list(cols.items())[:5]:
                    print(f"      {col:25s} → {info.get('filled', 'dropped')}")

        elif name == "handle_outliers":
            cols = step.get("columns", {})
            if cols:
                print(f"\n  ✦ Outliers handled  ({len(cols)} columns)")
                for col, info in list(cols.items())[:5]:
                    action = info.get("action", "capped")
                    n = info.get("outliers", 0)
                    print(f"      {col:25s} {n:>4} {action}")

    print(f"\n{'━' * w}\n")