"""
prepx.cleaning — data cleaning module.

Submodules:
- types: type coercion before missing handling
- missing: missing value imputation strategies
- outliers: outlier detection and handling
- dedupe: deduplication (exact + fuzzy)
- standardize: column naming and categorical normalization
- leakage: data leakage detection
"""

from prepx.cleaning.types import coerce_types, detect_mixed_types, parse_mixed_numeric
from prepx.cleaning.missing import handle_missing, compute_missing_summary
from prepx.cleaning.outliers import handle_outliers, winsorize, detect_outlier_summary
from prepx.cleaning.dedupe import deduplicate, normalize_categoricals, detect_potential_duplicates
from prepx.cleaning.standardize import standardize_columns, normalize_categories, detect_similar_columns
from prepx.cleaning.leakage import detect_leakage_columns, check_target_leakage

__all__ = [
    "coerce_types",
    "detect_mixed_types",
    "parse_mixed_numeric",
    "handle_missing",
    "compute_missing_summary",
    "handle_outliers",
    "winsorize",
    "detect_outlier_summary",
    "deduplicate",
    "normalize_categoricals",
    "detect_potential_duplicates",
    "standardize_columns",
    "normalize_categories",
    "detect_similar_columns",
    "detect_leakage_columns",
    "check_target_leakage",
]