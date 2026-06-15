"""Filtering of comparison results.

Each criterion is optional and they combine with logical AND. Rows with ``NaN``
in a filtered column are dropped by that criterion (they cannot satisfy a
threshold). ``# Unique Peptides`` is an annotation column carried through from
the source data set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_UNIQUE_PEPTIDES_COL = "# Unique Peptides"


def filter_results(
    results: pd.DataFrame,
    *,
    max_pvalue: float | None = None,
    max_fdr: float | None = None,
    min_abs_log2fc: float | None = None,
    min_unique_peptides: int | None = None,
    unique_peptides_col: str = DEFAULT_UNIQUE_PEPTIDES_COL,
    pvalue_col: str = "p_value",
    fdr_col: str = "adj_p_value",
    log2fc_col: str = "log2_fold_change",
) -> pd.DataFrame:
    """Filter a comparison results table by any combination of criteria.

    Args:
        max_pvalue: Keep rows with ``p_value`` <= this.
        max_fdr: Keep rows with adjusted p-value / FDR <= this.
        min_abs_log2fc: Keep rows with ``|log2_fold_change|`` >= this.
        min_unique_peptides: Keep rows with ``# Unique Peptides`` >= this.
        unique_peptides_col: Name of the unique-peptides column.
        pvalue_col, fdr_col, log2fc_col: Column names (override if customised).

    Returns:
        A filtered copy of ``results`` (original order preserved).
    """
    mask = pd.Series(True, index=results.index)

    if max_pvalue is not None:
        mask &= results[pvalue_col] <= max_pvalue
    if max_fdr is not None:
        mask &= results[fdr_col] <= max_fdr
    if min_abs_log2fc is not None:
        mask &= results[log2fc_col].abs() >= min_abs_log2fc
    if min_unique_peptides is not None:
        peptides = pd.to_numeric(results[unique_peptides_col], errors="coerce")
        mask &= peptides >= min_unique_peptides

    # NaN comparisons yield False already; make that explicit and safe.
    mask = mask.fillna(False).astype(bool)
    return results[mask].copy()
