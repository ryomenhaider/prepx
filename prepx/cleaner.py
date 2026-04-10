"""
dataprep.cleaner — auto-clean any pandas DataFrame and get a full report.
"""

from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def clean(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    handle_missing: str = "auto",        # "auto" | "drop" | "fill" | "none"
    missing_threshold: float = 0.6,      # drop column if > threshold% missing
    fix_dtypes: bool = True,
    strip_whitespace: bool = True,
    standardize_columns: bool = True,
    remove_outliers: bool = False,
    outlier_method: str = "iqr",         # "iqr" | "zscore"
    outlier_threshold: float = 3.0,
    drop_constant_cols: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean a DataFrame and return ``(cleaned_df, report_dict)``.

    Parameters
    ----------
    df                  : Input DataFrame.
    drop_duplicates     : Remove exact duplicate rows.
    handle_missing      : Strategy for missing values.
                          ``"auto"``  — median/mode fill for numeric/categorical;
                          ``"drop"``  — drop rows with any NaN;
                          ``"fill"``  — median/mode fill (same as auto but explicit);
                          ``"none"``  — leave missing values untouched.
    missing_threshold   : Drop columns whose missing-rate exceeds this value (0–1).
    fix_dtypes          : Attempt numeric and datetime coercions.
    strip_whitespace    : Strip leading/trailing whitespace from string columns.
    standardize_columns : Convert column names to snake_case.
    remove_outliers     : Cap or remove outlier rows (numeric columns only).
    outlier_method      : ``"iqr"`` fences or ``"zscore"`` threshold.
    outlier_threshold   : Z-score cut-off (used when ``outlier_method="zscore"``).
    drop_constant_cols  : Drop columns that contain only one unique value.
    verbose             : Print a human-readable summary to stdout.

    Returns
    -------
    cleaned_df : pd.DataFrame
    report     : dict  — machine-readable record of every action taken.
    """
    report: dict = {
        "initial_shape": df.shape,
        "steps": [],
    }

    df = df.copy()

    # ── 1. Column name standardisation ──────────────────────
    if standardize_columns:
        old_cols = list(df.columns)
        df.columns = [_snake(c) for c in df.columns]
        renamed = {o: n for o, n in zip(old_cols, df.columns) if o != n}
        if renamed:
            report["steps"].append({
                "step": "standardize_columns",
                "renamed": renamed,
                "count": len(renamed),
            })

    # ── 2. Whitespace stripping ──────────────────────────────
    if strip_whitespace:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        for col in str_cols:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df[col] = df[col].replace("", np.nan)
        if str_cols:
            report["steps"].append({
                "step": "strip_whitespace",
                "columns": str_cols,
            })

    # ── 3. Type coercion ─────────────────────────────────────
    if fix_dtypes:
        coerced = _fix_dtypes(df)
        if coerced:
            report["steps"].append({"step": "fix_dtypes", "coerced": coerced})

    # ── 4. Duplicate removal ─────────────────────────────────
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        report["steps"].append({
            "step": "drop_duplicates",
            "removed": removed,
            "remaining": len(df),
        })

    # ── 5. Constant column removal ───────────────────────────
    if drop_constant_cols:
        const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if const_cols:
            df = df.drop(columns=const_cols)
            report["steps"].append({
                "step": "drop_constant_cols",
                "dropped": const_cols,
            })

    # ── 6. High-missingness column removal ───────────────────
    missing_rates = df.isnull().mean()
    high_miss = missing_rates[missing_rates > missing_threshold].index.tolist()
    if high_miss:
        df = df.drop(columns=high_miss)
        report["steps"].append({
            "step": "drop_high_missing_cols",
            "threshold": missing_threshold,
            "dropped": high_miss,
        })

    # ── 7. Missing value handling ────────────────────────────
    if handle_missing in ("auto", "fill"):
        fill_report = _fill_missing(df)
        report["steps"].append({"step": "fill_missing", **fill_report})
    elif handle_missing == "drop":
        before = len(df)
        df = df.dropna()
        report["steps"].append({
            "step": "drop_missing_rows",
            "removed": before - len(df),
            "remaining": len(df),
        })

    # ── 8. Outlier handling ──────────────────────────────────
    if remove_outliers:
        out_report = _handle_outliers(df, outlier_method, outlier_threshold)
        report["steps"].append({"step": "outliers", **out_report})

    # ── Finalize ─────────────────────────────────────────────
    report["final_shape"] = df.shape
    report["rows_removed"] = report["initial_shape"][0] - df.shape[0]
    report["cols_removed"] = report["initial_shape"][1] - df.shape[1]

    if verbose:
        _print_clean_report(report)

    return df, report


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _snake(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\s\-\.]+", "_", name)
    name = re.sub(r"[^\w]", "", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower().strip("_")


def _fix_dtypes(df: pd.DataFrame) -> dict:
    coerced: dict = {"numeric": [], "datetime": []}
    for col in df.select_dtypes(include="object").columns:
        # Try numeric
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() / max(df[col].notna().sum(), 1) > 0.8:
            df[col] = converted
            coerced["numeric"].append(col)
            continue
        # Try datetime
        try:
            converted_dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if converted_dt.notna().sum() / max(df[col].notna().sum(), 1) > 0.8:
                df[col] = converted_dt
                coerced["datetime"].append(col)
        except Exception:
            pass
    return coerced


def _fill_missing(df: pd.DataFrame) -> dict:
    num_fills = {}
    cat_fills = {}
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            val = df[col].median()
            df[col] = df[col].fillna(val)
            num_fills[col] = {"missing": int(n_missing), "filled_with": f"median ({val:.4g})"}
        else:
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                val = mode_vals[0]
                df[col] = df[col].fillna(val)
                cat_fills[col] = {"missing": int(n_missing), "filled_with": f"mode ({val!r})"}
    return {"numeric_fills": num_fills, "categorical_fills": cat_fills}


def _handle_outliers(df: pd.DataFrame, method: str, threshold: float) -> dict:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    capped = {}
    for col in num_cols:
        if method == "iqr":
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            mu, sigma = df[col].mean(), df[col].std()
            lo = mu - threshold * sigma
            hi = mu + threshold * sigma
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out:
            df[col] = df[col].clip(lo, hi)
            capped[col] = {"outliers_capped": int(n_out), "lo": round(lo, 4), "hi": round(hi, 4)}
    return {"method": method, "capped": capped}


def _print_clean_report(report: dict):
    w = 62
    sep = "─" * w
    print(f"\n{'━' * w}")
    print(f"  dataprep  ·  Cleaning Report")
    print(f"{'━' * w}")
    print(f"  {'Initial shape':30s} {report['initial_shape'][0]:>6} rows × {report['initial_shape'][1]} cols")
    print(f"  {'Final shape':30s} {report['final_shape'][0]:>6} rows × {report['final_shape'][1]} cols")
    print(f"  {'Rows removed':30s} {report['rows_removed']:>6}")
    print(f"  {'Columns removed':30s} {report['cols_removed']:>6}")
    print(sep)

    for step in report["steps"]:
        name = step["step"]

        if name == "standardize_columns":
            print(f"\n  ✦ Column names → snake_case  ({step['count']} renamed)")
            for old, new in list(step["renamed"].items())[:8]:
                print(f"      {old!r:25s} → {new!r}")
            if step["count"] > 8:
                print(f"      … and {step['count'] - 8} more")

        elif name == "strip_whitespace":
            print(f"\n  ✦ Whitespace stripped  ({len(step['columns'])} string columns)")

        elif name == "fix_dtypes":
            n = len(step["coerced"]["numeric"]) + len(step["coerced"]["datetime"])
            print(f"\n  ✦ Type coercion  ({n} columns updated)")
            if step["coerced"]["numeric"]:
                print(f"      Numeric  : {', '.join(step['coerced']['numeric'])}")
            if step["coerced"]["datetime"]:
                print(f"      Datetime : {', '.join(step['coerced']['datetime'])}")

        elif name == "drop_duplicates":
            print(f"\n  ✦ Duplicate rows removed  : {step['removed']}")

        elif name == "drop_constant_cols":
            print(f"\n  ✦ Constant columns dropped : {', '.join(step['dropped'])}")

        elif name == "drop_high_missing_cols":
            pct = int(step["threshold"] * 100)
            print(f"\n  ✦ High-missing columns dropped  (>{pct}% NaN)")
            for col in step["dropped"]:
                print(f"      {col}")

        elif name == "fill_missing":
            total = len(step["numeric_fills"]) + len(step["categorical_fills"])
            print(f"\n  ✦ Missing values filled  ({total} columns)")
            for col, info in step["numeric_fills"].items():
                print(f"      {col:30s} {info['missing']:4d} NaN → {info['filled_with']}")
            for col, info in step["categorical_fills"].items():
                print(f"      {col:30s} {info['missing']:4d} NaN → {info['filled_with']}")

        elif name == "drop_missing_rows":
            print(f"\n  ✦ Rows with any NaN dropped  : {step['removed']}")

        elif name == "outliers":
            print(f"\n  ✦ Outliers capped  (method: {step['method']})")
            for col, info in step["capped"].items():
                print(f"      {col:30s} {info['outliers_capped']:4d} values capped to [{info['lo']}, {info['hi']}]")

    print(f"\n{'━' * w}\n")