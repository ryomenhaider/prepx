"""
prepx.cleaning.leakage — detect potential data leakage.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


LEAKAGE_PATTERNS = [
    "id", "uuid", "key", "pk", "primary_key",
    "created_at", "updated_at", "modified_at", "timestamp", "date", "datetime",
    "_at", "_on", "_time",
    "password", "passwd", "pwd", "token", "secret",
    "ip_", "ip_address", "user_agent",
    "session", "cookie",
]


def detect_leakage_columns(
    df: pd.DataFrame,
    target: Optional[str] = None,
    patterns: Optional[list] = None,
) -> list:
    """
    Detect columns that should likely not be used as features.

    These include ID columns, temporal columns that would cause look-ahead bias,
    and sensitive columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Target column name.
    patterns : list
        Additional patterns to check.

    Returns
    -------
    list
        List of potentially leaky column names.
    """
    if patterns is None:
        patterns = LEAKAGE_PATTERNS

    cols = df.columns.tolist()
    leaky = []

    target_excluded = [target] if target else []

    for col in cols:
        if col in target_excluded:
            continue

        col_lower = col.lower()

        for pattern in patterns:
            if pattern in col_lower:
                leaky.append({
                    "column": col,
                    "reason": f"matches pattern '{pattern}'",
                    "suggestion": _get_suggestion(pattern),
                })
                break

        if _looks_like_id(col):
            if col not in leaky:
                leaky.append({
                    "column": col,
                    "reason": "appears to be an ID column",
                    "suggestion": "exclude from modeling",
                })

    return leaky


def _looks_like_id(col: str) -> bool:
    """Check if column looks like an ID."""
    col_lower = col.lower()

    if col_lower.endswith("_id") or col_lower == "id":
        return True

    if len(col) < 6:
        return False

    if re.match(r"^[a-f0-9]+$", col_lower):
        return True

    if re.match(r"^\d+$", col):
        return True

    return False


def _get_suggestion(pattern: str) -> str:
    """Get suggestion for handling leaky column."""
    suggestions = {
        "id": "exclude from modeling - unique identifier",
        "uuid": "exclude from modeling - unique identifier",
        "key": "exclude from modeling - may be unique key",
        "timestamp": "use carefully - may cause look-ahead bias",
        "datetime": "use carefully - may cause look-ahead bias",
        "created_at": "use carefully - may cause look-ahead bias",
        "updated_at": "use carefully - may cause data leakage",
        "password": "never include - security risk",
        "passwd": "never include - security risk",
        "token": "never include - security risk",
        "secret": "never include - security risk",
    }
    return suggestions.get(pattern, "review before modeling")


def check_target_leakage(
    df: pd.DataFrame,
    target: str,
    correlation_threshold: float = 0.95,
) -> dict:
    """
    Check if any feature is suspiciously correlated with target.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Target column name.
    correlation_threshold : float
        Threshold for flagging suspicious correlation.

    Returns
    -------
    dict
        Report with suspicious feature correlations.
    """
    if target not in df.columns:
        return {"error": f"Target '{target}' not found"}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]

    suspicious = []

    for col in numeric_cols:
        try:
            corr = df[[col, target]].corr().iloc[0, 1]
            if abs(corr) >= correlation_threshold:
                suspicious.append({
                    "feature": col,
                    "correlation": round(corr, 3),
                    "warning": "suspiciously high correlation with target",
                })
        except Exception:
            pass

    return {
        "suspicious_features": suspicious,
        "threshold": correlation_threshold,
    }