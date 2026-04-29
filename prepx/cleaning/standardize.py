"""
prepx.cleaning.standardize — column standardization and naming conventions.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


def standardize_columns(
    df: pd.DataFrame,
    naming_style: str = "snake",
    lowercase: bool = True,
    remove_special_chars: bool = True,
) -> dict:
    """
    Standardize column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    naming_style : str
        "snake", "camel", "pascal", or "kebab".
    lowercase : bool
        Convert to lowercase.
    remove_special_chars : bool
        Replace special characters with underscores.

    Returns
    -------
    dict
        Report of column renames.
    """
    report = {"renamed": {}}
    df = df.copy()

    old_cols = df.columns.tolist()
    new_cols = []

    for col in old_cols:
        new_col = _convert_name(col, naming_style, lowercase, remove_special_chars)
        new_cols.append(new_col)

    df.columns = new_cols
    renamed = {o: n for o, n in zip(old_cols, new_cols) if o != n}

    if renamed:
        report["renamed"] = renamed
        report["count"] = len(renamed)

    return df, report


def _convert_name(
    name: str,
    style: str,
    lowercase: bool,
    remove_special: bool,
) -> str:
    """Convert column name to specified style."""
    name = str(name).strip()

    if remove_special:
        name = re.sub(r"[^\w\s]", "", name)
        name = re.sub(r"[\s\-]+", "_", name)

    if lowercase:
        name = name.lower()

    if style == "snake":
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
        name = re.sub(r"[\-\.]+", "_", name)
        name = name.lower().strip("_")

    elif style == "camel":
        name = _to_camel(name)

    elif style == "pascal":
        name = _to_pascal(name)

    elif style == "kebab":
        name = name.replace("_", "-")

    return name


def _to_camel(name: str) -> str:
    """Convert to camelCase."""
    parts = re.split(r"[\s_\-]+", name)
    if not parts:
        return name
    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])


def _to_pascal(name: str) -> str:
    """Convert to PascalCase."""
    parts = re.split(r"[\s_\-]+", name)
    return "".join(p.capitalize() for p in parts)


def normalize_categories(
    df: pd.DataFrame,
    col_list: Optional[list] = None,
    strip_whitespace: bool = True,
    lowercase: bool = False,
    remove_duplicates: bool = True,
) -> dict:
    """
    Normalize categorical/string columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col_list : list
        Columns to normalize. If None, all object columns.
    strip_whitespace : bool
        Strip whitespace.
    lowercase : bool
        Convert to lowercase.
    remove_duplicates : bool
        Remove duplicate values (e.g., "A" vs "a" if lowercase).

    Returns
    -------
    dict
        Report of normalizations.
    """
    report = {"columns": {}}
    df = df.copy()

    if col_list is None:
        col_list = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in col_list:
        changes = 0

        if strip_whitespace:
            before = df[col].copy()
            df[col] = df[col].astype(str).str.strip()
            changes += (before != df[col]).sum()

        if lowercase:
            before = df[col].copy()
            df[col] = df[col].str.lower()
            changes += (before != df[col]).sum()

        if remove_duplicates:
            unique_before = df[col].nunique()
            df[col] = df[col].astype(str).str.strip().str.lower()
            unique_after = df[col].nunique()
            if unique_after > unique_before:
                changes += unique_after - unique_before

        if changes > 0:
            report["columns"][col] = {"changes": changes}

    return df, report


def detect_similar_columns(df: pd.DataFrame, threshold: float = 0.9) -> list:
    """
    Detect columns with similar names (potential data leakage/redundancy).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float
        Similarity threshold (0-1).

    Returns
    -------
    list
        List of similar column pairs.
    """
    from rapidfuzz import fuzz, process

    cols = df.columns.tolist()
    similar = []

    for i, col_a in enumerate(cols):
        remaining = cols[i + 1:]
        for col_b in remaining:
            ratio = fuzz.ratio(col_a.lower(), col_b.lower()) / 100
            if ratio >= threshold:
                similar.append({"col_a": col_a, "col_b": col_b, "similarity": round(ratio, 2)})

    return similar