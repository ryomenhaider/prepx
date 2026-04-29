"""
prepx — One-line DataFrame cleaning and EDA with advanced imputation, outlier detection, and drift analysis.
"""

from prepx.cleaner import clean as _clean

from prepx.cleaning.types import coerce_types
from prepx.cleaning.missing import handle_missing
from prepx.cleaning.outliers import handle_outliers
from prepx.cleaning.dedupe import deduplicate
from prepx.cleaning.standardize import standardize_columns
from prepx.cleaning.leakage import detect_leakage_columns

from prepx.eda.stats import (
    compute_numeric_stats,
    compute_categorical_stats,
    compute_dtypes_summary,
    compute_overview,
)

__version__ = "1.0.0"


def clean(df, **kwargs):
    return _clean(df, **kwargs)


def eda(df, **kwargs):
    from prepx.analysis import eda as _eda
    return _eda(df, **kwargs)


__all__ = [
    "clean",
    "eda",
    "coerce_types",
    "handle_missing",
    "handle_outliers",
    "deduplicate",
    "standardize_columns",
    "detect_leakage_columns",
    "compute_numeric_stats",
    "compute_categorical_stats",
    "compute_dtypes_summary",
    "compute_overview",
]