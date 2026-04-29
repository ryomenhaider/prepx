"""
prepx.cleaning.dedupe — deduplication strategies.

Supports: exact and fuzzy deduplication using rapidfuzz.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    warnings.warn(
        "rapidfuzz not installed. Fuzzy deduplication disabled. "
        "Install: pip install rapidfuzz"
    )


def deduplicate(
    df: pd.DataFrame,
    method: str = "exact",
    threshold: float = 0.85,
    columns: Optional[list] = None,
    keep: str = "first",
) -> dict:
    """
    Remove duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    method : str
        "exact" or "fuzzy".
    threshold : float
        Similarity threshold for fuzzy matching (0-1).
    columns : list
        Columns to consider for deduplication. If None, all columns.
    keep : str
        Which to keep: "first", "last", or False (drop all duplicates).

    Returns
    -------
    dict
        Report of deduplication.
    """
    report = {"method": method, "original_rows": len(df)}
    df = df.copy()

    if columns is None:
        columns = df.columns.tolist()

    if method == "exact":
        before = len(df)
        df = df.drop_duplicates(subset=columns, keep=keep)
        removed = before - len(df)
        report["duplicates_removed"] = removed
        report["remaining_rows"] = len(df)

    elif method == "fuzzy":
        if not HAS_RAPIDFUZZ:
            raise ImportError(
                "rapidfuzz required for fuzzy deduplication. Install: pip install rapidfuzz"
            )
        before = len(df)
        df = _fuzzy_dedupe(df, columns, threshold, keep)
        removed = before - len(df)
        report["duplicates_removed"] = removed
        report["remaining_rows"] = len(df)
        report["threshold"] = threshold

    return df, report


def _fuzzy_dedupe(
    df: pd.DataFrame,
    columns: list,
    threshold: float,
    keep: str,
) -> pd.DataFrame:
    """Fuzzy deduplication using rapidfuzz."""
    df = df.copy()
    df["__dupe_key__"] = df[columns].astype(str).apply(lambda x: " | ".join(x), axis=1)

    to_remove = set()

    values = df["__dupe_key__"].tolist()

    for i, val in enumerate(values):
        if i in to_remove:
            continue

        if i >= len(values):
            break

        remaining = values[i + 1:]
        if not remaining:
            continue

        matches = process.extract(
            val,
            remaining,
            scorer=fuzz.token_sort_ratio,
            limit=len(remaining),
            score_cutoff=threshold * 100,
        )

        for match in matches:
            match_idx = values.index(match[0], i + 1)
            to_remove.add(match_idx)

    df = df.drop(index=list(to_remove))
    df = df.drop(columns=["__dupe_key__"])

    if keep == "last":
        df = df.reverse().drop_duplicates(subset=columns, keep="first").reverse()

    return df


def normalize_categoricals(
    df: pd.DataFrame,
    column: str,
    mappings: Optional[dict] = None,
    normalize_spacing: bool = True,
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """
    Normalize categorical values (e.g., "USA" = "U.S.A" = "United States").

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name.
    mappings : dict
        Custom mappings dict = {"value": "canonical_form"}.
    normalize_spacing : bool
        Strip whitespace and normalize spacing.
    case_sensitive : bool
        Whether to lowercase.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized values.
    """
    df = df.copy()
    series = df[column].astype(str)

    if normalize_spacing:
        series = series.str.strip().str.replace(r"\s+", " ", regex=True)

    if not case_sensitive:
        series = series.str.lower()

    if mappings:
        series = series.replace(mappings)

    df[column] = series
    return df


def detect_potential_duplicates(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    threshold: float = 0.9,
) -> list:
    """
    Detect potential duplicates without removing them.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list
        Columns to check.
    threshold : float
        Similarity threshold.

    Returns
    -------
    list
        List of potential duplicate pairs.
    """
    if not HAS_RAPIDFUZZ:
        warnings.warn("rapidfuzz not installed. Install: pip install rapidfuzz")
        return []

    if columns is None:
        columns = df.columns.tolist()

    df["__key__"] = df[columns].astype(str).apply(lambda x: " | ".join(x), axis=1)
    values = df["__key__"].tolist()

    duplicates = []

    for i in range(len(values) - 1):
        remaining = values[i + 1:]
        matches = process.extract(
            values[i],
            remaining,
            scorer=fuzz.token_sort_ratio,
            limit=5,
            score_cutoff=threshold * 100,
        )

        for match in matches:
            duplicates.append({
                "row_a": i,
                "row_b": values.index(match[0], i + 1),
                "similarity": match[1] / 100,
                "value": match[0],
            })

    df = df.drop(columns=["__key__"])
    return duplicates


def compute_dupe_stats(df: pd.DataFrame) -> dict:
    """Compute duplicate statistics."""
    exact_dupes = df.duplicated().sum()

    return {
        "exact_duplicates": int(exact_dupes),
        "pct_duplicates": round(exact_dupes / len(df) * 100, 2) if len(df) > 0 else 0,
    }