"""
dataprep.eda — exploratory data analysis in one function call.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def eda(
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    top_n_categories: int = 10,
    correlation_method: str = "pearson",   # "pearson" | "spearman" | "kendall"
    verbose: bool = True,
) -> dict:
    """
    Run a full exploratory analysis on *df* and return a rich report dict.

    Parameters
    ----------
    df                  : DataFrame to analyse.
    target              : Optional target/label column — enables class-balance
                          stats and per-feature correlation with the target.
    top_n_categories    : How many top categories to show for object columns.
    correlation_method  : Method for the correlation matrix.
    verbose             : Print a human-readable summary to stdout.

    Returns
    -------
    report : dict with keys:
        overview, dtypes, missing, numeric_stats, categorical_stats,
        correlations, target_analysis (if target supplied), warnings
    """
    report: dict = {}

    # ── Overview ─────────────────────────────────────────────
    report["overview"] = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "total_cells": df.size,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
    }

    # ── Data types ───────────────────────────────────────────
    report["dtypes"] = {
        "numeric":   df.select_dtypes(include=[np.number]).columns.tolist(),
        "datetime":  df.select_dtypes(include=["datetime"]).columns.tolist(),
        "object":    df.select_dtypes(include="object").columns.tolist(),
        "boolean":   df.select_dtypes(include="bool").columns.tolist(),
        "per_column": df.dtypes.astype(str).to_dict(),
    }

    # ── Missing values ───────────────────────────────────────
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    report["missing"] = {
        "total_missing_cells": int(miss.sum()),
        "pct_missing_overall": round(miss.sum() / df.size * 100, 2),
        "per_column": {
            col: {"count": int(miss[col]), "pct": float(miss_pct[col])}
            for col in df.columns
            if miss[col] > 0
        },
        "complete_columns": int((miss == 0).sum()),
        "columns_with_missing": int((miss > 0).sum()),
    }

    # ── Numeric stats ────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_stats = {}
    for col in num_cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        skew_val = float(s.skew()) if len(s) > 2 else 0.0
        num_stats[col] = {
            "count":    int(s.count()),
            "mean":     round(float(s.mean()), 4),
            "std":      round(float(s.std()), 4),
            "min":      round(float(s.min()), 4),
            "p25":      round(float(q1), 4),
            "median":   round(float(s.median()), 4),
            "p75":      round(float(q3), 4),
            "max":      round(float(s.max()), 4),
            "skewness": round(skew_val, 4),
            "kurtosis": round(float(s.kurtosis()), 4) if len(s) > 3 else None,
            "outliers_iqr": n_out,
            "unique":   int(s.nunique()),
            "zeros":    int((s == 0).sum()),
            "negatives": int((s < 0).sum()),
        }
    report["numeric_stats"] = num_stats

    # ── Categorical stats ────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_stats = {}
    for col in cat_cols:
        s = df[col].dropna()
        vc = s.value_counts()
        entropy = _entropy(vc)
        cat_stats[col] = {
            "count":           int(s.count()),
            "unique":          int(s.nunique()),
            "top_categories":  vc.head(top_n_categories).to_dict(),
            "mode":            str(vc.index[0]) if len(vc) else None,
            "mode_freq_pct":   round(vc.iloc[0] / len(s) * 100, 2) if len(vc) else None,
            "entropy":         round(entropy, 4),
        }
    report["categorical_stats"] = cat_stats

    # ── Correlations ─────────────────────────────────────────
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr(method=correlation_method).round(4)
        # Find pairs with |r| > 0.7 (high correlation)
        high_corr_pairs = []
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i + 1:]:
                val = corr_matrix.loc[c1, c2]
                if abs(val) >= 0.7:
                    high_corr_pairs.append({"col_a": c1, "col_b": c2, "r": round(float(val), 4)})
        report["correlations"] = {
            "method":          correlation_method,
            "matrix":          corr_matrix.to_dict(),
            "high_corr_pairs": sorted(high_corr_pairs, key=lambda x: abs(x["r"]), reverse=True),
        }
    else:
        report["correlations"] = {"method": correlation_method, "matrix": {}, "high_corr_pairs": []}

    # ── Target analysis ──────────────────────────────────────
    if target and target in df.columns:
        report["target_analysis"] = _target_analysis(df, target, num_cols, correlation_method)

    # ── Auto-generated warnings ──────────────────────────────
    report["warnings"] = _build_warnings(df, report)

    if verbose:
        _print_eda_report(report, target)

    return report


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _entropy(value_counts: pd.Series) -> float:
    probs = value_counts / value_counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _target_analysis(df, target, num_cols, method):
    result: dict = {}
    t = df[target]

    # Class balance (categorical target)
    if t.dtype == "object" or t.nunique() < 20:
        vc = t.value_counts()
        result["class_balance"] = vc.to_dict()
        result["imbalance_ratio"] = round(float(vc.iloc[0] / vc.iloc[-1]), 2) if len(vc) > 1 else 1.0

    # Numeric correlations with target (if target is numeric)
    if pd.api.types.is_numeric_dtype(t):
        other_num = [c for c in num_cols if c != target]
        corrs = {
            col: round(float(df[[col, target]].corr(method=method).iloc[0, 1]), 4)
            for col in other_num
        }
        result["feature_correlations"] = dict(
            sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        )

    return result


def _build_warnings(df: pd.DataFrame, report: dict) -> list[dict]:
    ws = []

    # Duplicate rows
    n_dup = report["overview"]["duplicate_rows"]
    if n_dup > 0:
        ws.append({"level": "warn", "message": f"{n_dup} duplicate rows detected."})

    # High missing
    for col, info in report["missing"]["per_column"].items():
        if info["pct"] > 30:
            ws.append({"level": "warn", "message": f"Column '{col}' has {info['pct']}% missing values."})

    # High cardinality
    for col, info in report["categorical_stats"].items():
        if info["unique"] > 50:
            ws.append({"level": "info", "message": f"Column '{col}' has high cardinality ({info['unique']} unique values)."})

    # Skewed numerics
    for col, info in report["numeric_stats"].items():
        if abs(info["skewness"]) > 1.5:
            ws.append({"level": "info", "message": f"Column '{col}' is highly skewed (skewness={info['skewness']})."})

    # Outliers
    for col, info in report["numeric_stats"].items():
        if info["outliers_iqr"] > 0:
            pct = round(info["outliers_iqr"] / info["count"] * 100, 1)
            ws.append({"level": "info", "message": f"Column '{col}' has {info['outliers_iqr']} IQR outliers ({pct}%)."})

    # Multi-collinearity
    for pair in report["correlations"].get("high_corr_pairs", []):
        ws.append({"level": "warn", "message": f"High correlation ({pair['r']}) between '{pair['col_a']}' and '{pair['col_b']}'."})

    return ws


# ─────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────

def _print_eda_report(report: dict, target: Optional[str]):
    w = 62
    sep = "─" * w

    print(f"\n{'━' * w}")
    print(f"  dataprep  ·  EDA Report")
    print(f"{'━' * w}")

    # Overview
    ov = report["overview"]
    print(f"\n  OVERVIEW")
    print(sep)
    print(f"  {'Rows':30s} {ov['rows']:>10,}")
    print(f"  {'Columns':30s} {ov['cols']:>10,}")
    print(f"  {'Total cells':30s} {ov['total_cells']:>10,}")
    print(f"  {'Duplicate rows':30s} {ov['duplicate_rows']:>10,}")
    print(f"  {'Memory usage':30s} {ov['memory_mb']:>9.3f} MB")

    # Types
    dt = report["dtypes"]
    print(f"\n  DATA TYPES")
    print(sep)
    if dt["numeric"]:   print(f"  Numeric  ({len(dt['numeric'])})  : {', '.join(dt['numeric'][:6])}{'…' if len(dt['numeric'])>6 else ''}")
    if dt["object"]:    print(f"  Object   ({len(dt['object'])})  : {', '.join(dt['object'][:6])}{'…' if len(dt['object'])>6 else ''}")
    if dt["datetime"]:  print(f"  Datetime ({len(dt['datetime'])}) : {', '.join(dt['datetime'])}")
    if dt["boolean"]:   print(f"  Boolean  ({len(dt['boolean'])})  : {', '.join(dt['boolean'])}")

    # Missing
    miss = report["missing"]
    print(f"\n  MISSING VALUES  —  {miss['total_missing_cells']} cells ({miss['pct_missing_overall']}% of total)")
    print(sep)
    if miss["per_column"]:
        for col, info in miss["per_column"].items():
            bar = _bar(info["pct"] / 100, 20)
            print(f"  {col:30s} {info['count']:>6}  {info['pct']:>5.1f}%  {bar}")
    else:
        print("  No missing values — dataset is complete.")

    # Numeric stats
    if report["numeric_stats"]:
        print(f"\n  NUMERIC COLUMNS")
        print(sep)
        hdr = f"  {'Column':22s} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Outliers':>9}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for col, s in report["numeric_stats"].items():
            print(f"  {col:22s} {s['mean']:>10.4g} {s['std']:>10.4g} {s['min']:>10.4g} {s['max']:>10.4g} {s['outliers_iqr']:>9}")

    # Categorical stats
    if report["categorical_stats"]:
        print(f"\n  CATEGORICAL COLUMNS")
        print(sep)
        for col, s in report["categorical_stats"].items():
            top = list(s["top_categories"].items())[:5]
            top_str = ", ".join(f"{k!r}({v})" for k, v in top)
            print(f"  {col:28s} {s['unique']:>4} unique  mode={s['mode']!r} ({s['mode_freq_pct']}%)")
            print(f"    Top: {top_str}")

    # Correlations
    hcp = report["correlations"]["high_corr_pairs"]
    if hcp:
        print(f"\n  HIGH CORRELATIONS  (|r| ≥ 0.70)  [{report['correlations']['method']}]")
        print(sep)
        for p in hcp[:10]:
            print(f"  {p['col_a']:25s}  ↔  {p['col_b']:25s}  r = {p['r']:+.4f}")

    # Target analysis
    if "target_analysis" in report:
        ta = report["target_analysis"]
        print(f"\n  TARGET: '{target}'")
        print(sep)
        if "class_balance" in ta:
            print(f"  Class balance  (imbalance ratio {ta['imbalance_ratio']:.2f}x)")
            for cls, cnt in list(ta["class_balance"].items())[:8]:
                bar = _bar(cnt / report["overview"]["rows"], 16)
                print(f"    {str(cls):25s} {cnt:>7,}  {bar}")
        if "feature_correlations" in ta:
            print(f"  Correlation with target (top 8):")
            for col, r in list(ta["feature_correlations"].items())[:8]:
                print(f"    {col:30s}  r = {r:+.4f}")

    # Warnings
    ws = report["warnings"]
    if ws:
        print(f"\n  WARNINGS & NOTES  ({len(ws)})")
        print(sep)
        icons = {"warn": "⚠", "info": "ℹ"}
        for w_item in ws:
            print(f"  {icons.get(w_item['level'], '·')} {w_item['message']}")

    print(f"\n{'━' * w}\n")


def _bar(ratio: float, width: int = 20) -> str:
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)