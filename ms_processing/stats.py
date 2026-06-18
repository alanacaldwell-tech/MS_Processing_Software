"""Statistical analysis: condition means, fold change, and t-tests.

Functions here are pure: they take a :class:`ProteomicsDataset` plus a
:class:`ConditionMap` and explicit arguments, and return DataFrames. The same
calls back a notebook cell today and a GUI action later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from .conditions import ConditionMap
from .dataset import ProteomicsDataset

# Multiple-testing correction methods exposed to the user. Maps a friendly label
# to the method string understood by statsmodels' ``multipletests``.
CORRECTION_METHODS: dict[str, str] = {
    "fdr_bh": "Benjamini-Hochberg (FDR)",
    "fdr_by": "Benjamini-Yekutieli (FDR)",
    "bonferroni": "Bonferroni",
    "holm": "Holm-Bonferroni",
    "sidak": "Sidak",
}

DEFAULT_CORRECTION = "fdr_bh"


def adjust_pvalues(pvalues, method: str = DEFAULT_CORRECTION) -> np.ndarray:
    """Adjust p-values for multiple comparisons.

    ``NaN`` p-values (e.g. proteins with too few/zero-variance replicates) are
    excluded from the correction and remain ``NaN`` in the result.
    """
    if method not in CORRECTION_METHODS:
        valid = ", ".join(sorted(CORRECTION_METHODS))
        raise ValueError(f"Unknown correction method {method!r}. Choose from: {valid}.")

    pvals = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvals.shape, np.nan)
    mask = ~np.isnan(pvals)
    if mask.any():
        adjusted[mask] = multipletests(pvals[mask], method=method)[1]
    return adjusted


def condition_means(dataset: ProteomicsDataset, condition_map: ConditionMap) -> pd.DataFrame:
    """Mean protein abundance per protein (row) per condition (column).

    The data set's annotation columns are preserved alongside the means.
    """
    condition_map.validate(dataset)
    data = dataset.data

    means = {c.name: data[list(c.replicate_columns)].mean(axis=1) for c in condition_map}
    means_df = pd.DataFrame(means, index=dataset.raw.index)
    return pd.concat([dataset.annotations, means_df], axis=1)


def compare_conditions(
    dataset: ProteomicsDataset,
    condition_map: ConditionMap,
    numerator: str,
    denominator: str,
    *,
    correction: str = DEFAULT_CORRECTION,
    equal_var: bool = True,
    data_is_log: bool | None = None,
) -> pd.DataFrame:
    """Compare two conditions (``numerator`` vs ``denominator``).

    Computes, per protein:
      * mean abundance of each condition,
      * fold change and its log2,
      * two-sided independent t-test p-value (``scipy.stats.ttest_ind``),
      * adjusted p-value / FDR using ``correction``,
      * -log10(p-value).

    Fold change respects the data scale:
      * **linear** data → fold change = mean(numerator) / mean(denominator),
        log2 fold change = log2 of that;
      * **log2** data → log2 fold change = mean(numerator) − mean(denominator),
        fold change = 2 ** that, and the t-test runs on the log values.

    Args:
        equal_var: If True, Student's t-test; if False, Welch's t-test.
        data_is_log: Whether the data is log-scaled. Defaults to the dataset's
            ``is_log_transformed`` flag (set by normalization).

    The data set's annotation columns are preserved in the output.
    """
    condition_map.validate(dataset)
    cond_a = condition_map[numerator]
    cond_b = condition_map[denominator]
    data = dataset.data
    is_log = dataset.is_log_transformed if data_is_log is None else data_is_log

    a = data[list(cond_a.replicate_columns)]
    b = data[list(cond_b.replicate_columns)]

    mean_a = a.mean(axis=1)
    mean_b = b.mean(axis=1)
    if is_log:
        log2_fc = mean_a - mean_b
        fold_change = np.power(2.0, log2_fc)
    else:
        fold_change = mean_a / mean_b
        log2_fc = np.log2(fold_change.where(fold_change > 0))

    # Row-wise two-sided independent t-test; NaN where it cannot be computed.
    with np.errstate(invalid="ignore"):
        _, pvalues = scipy_stats.ttest_ind(
            a.to_numpy(dtype=float),
            b.to_numpy(dtype=float),
            axis=1,
            equal_var=equal_var,
            nan_policy="omit",
        )
    pvalues = np.asarray(pvalues, dtype=float)

    adjusted = adjust_pvalues(pvalues, method=correction)

    results = pd.DataFrame(
        {
            f"mean_{numerator}": mean_a,
            f"mean_{denominator}": mean_b,
            "fold_change": fold_change,
            "log2_fold_change": log2_fc,
            "p_value": pvalues,
            "adj_p_value": adjusted,
            "neg_log10_p_value": -np.log10(pd.Series(pvalues, index=data.index)),
        },
        index=data.index,
    )
    output = pd.concat([dataset.annotations, results], axis=1)
    output.attrs["correction"] = correction
    output.attrs["comparison"] = f"{numerator} vs {denominator}"
    return output
