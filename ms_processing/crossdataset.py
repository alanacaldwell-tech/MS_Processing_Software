"""Compare proteomics results across multiple data sets (Excel files).

Given the comparison-results tables from several data sets (each produced by
:func:`ms_processing.stats.compare_conditions`), this aligns them by protein
identifier and lets you:

  * build a wide table of a chosen metric (e.g. log2 fold change) side by side,
  * compute the overlap of significant proteins between data sets,
  * correlate a metric between two data sets.

Each input table just needs a shared identifier column (default ``Accession``).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .filtering import filter_results


def align_results(
    results_by_name: Mapping[str, pd.DataFrame],
    *,
    value_cols: tuple[str, ...] = ("log2_fold_change", "adj_p_value"),
    id_col: str = "Accession",
    label_cols: tuple[str, ...] = ("Gene",),
) -> pd.DataFrame:
    """Align several results tables into one wide table, keyed by ``id_col``.

    For each data set ``name``, the requested ``value_cols`` are suffixed with
    ``__name`` (e.g. ``log2_fold_change__DatasetA``). ``label_cols`` (e.g. Gene)
    are carried through once, taken from the first data set that has them.

    Returns:
        A DataFrame indexed by ``id_col`` (outer join across data sets), so a
        protein present in any data set appears, with NaN where it is absent.
    """
    if not results_by_name:
        raise ValueError("No data sets provided to align.")

    merged: pd.DataFrame | None = None
    labels: pd.DataFrame | None = None

    for name, df in results_by_name.items():
        if id_col not in df.columns:
            raise KeyError(f"Data set {name!r} has no {id_col!r} column.")
        missing = [c for c in value_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Data set {name!r} is missing columns: {missing}.")

        sub = df[[id_col, *value_cols]].copy()
        sub = sub.rename(columns={c: f"{c}__{name}" for c in value_cols})
        sub = sub.drop_duplicates(subset=id_col).set_index(id_col)
        merged = sub if merged is None else merged.join(sub, how="outer")

        # Collect labels once (first data set that provides them wins).
        present_labels = [c for c in label_cols if c in df.columns]
        if labels is None and present_labels:
            labels = df[[id_col, *present_labels]].drop_duplicates(subset=id_col).set_index(id_col)

    if labels is not None:
        merged = labels.join(merged, how="right")

    return merged.reset_index()


def significant_sets(
    results_by_name: Mapping[str, pd.DataFrame],
    *,
    id_col: str = "Accession",
    **filter_kwargs,
) -> dict[str, set]:
    """Return, per data set, the set of significant protein IDs.

    ``filter_kwargs`` are passed straight to
    :func:`ms_processing.filtering.filter_results` (e.g. ``max_fdr=0.05,
    min_abs_log2fc=1.0``), so significance is defined consistently across sets.
    """
    out: dict[str, set] = {}
    for name, df in results_by_name.items():
        hits = filter_results(df, **filter_kwargs)
        out[name] = set(hits[id_col].dropna())
    return out


def overlap_summary(significant: Mapping[str, set]) -> dict:
    """Summarize overlap between per-data-set significant sets.

    Returns a dict with each set's size, the size of the intersection across all
    sets, the union size, and the shared protein IDs.
    """
    sets = list(significant.values())
    intersection = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    return {
        "per_dataset_counts": {name: len(ids) for name, ids in significant.items()},
        "n_shared": len(intersection),
        "n_union": len(union),
        "shared_ids": intersection,
    }


def correlate_metric(
    aligned: pd.DataFrame,
    name_x: str,
    name_y: str,
    *,
    metric: str = "log2_fold_change",
    method: str = "pearson",
) -> float:
    """Correlation of ``metric`` between two data sets in an aligned table.

    Uses only proteins measured in both data sets (rows where neither value is
    NaN). ``method`` is any pandas correlation method (pearson/spearman/kendall).
    """
    x_col, y_col = f"{metric}__{name_x}", f"{metric}__{name_y}"
    for col in (x_col, y_col):
        if col not in aligned.columns:
            raise KeyError(f"Column {col!r} not found in aligned table.")
    pair = aligned[[x_col, y_col]].dropna()
    return float(pair[x_col].corr(pair[y_col], method=method))
