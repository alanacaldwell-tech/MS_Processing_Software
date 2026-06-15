"""Visualizations: volcano plot, abundance heatmap, enrichment bar plot.

All functions return a matplotlib ``Axes`` (or ``Figure`` for the clustered
heatmap) so they render inline in the notebook and can be embedded by a future
GUI. They take the same DataFrames produced by :mod:`ms_processing.stats`,
:mod:`ms_processing.enrichment` and a :class:`~ms_processing.dataset.ProteomicsDataset`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .conditions import ConditionMap
from .dataset import ProteomicsDataset


def volcano_plot(
    results: pd.DataFrame,
    *,
    log2fc_col: str = "log2_fold_change",
    pvalue_col: str = "p_value",
    use_adjusted: bool = False,
    adj_pvalue_col: str = "adj_p_value",
    fc_threshold: float = 1.0,
    p_threshold: float = 0.05,
    label_col: str | None = "Gene",
    n_labels: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Volcano plot: log2 fold change (x) vs -log10 p-value (y).

    Points are colored as up-regulated, down-regulated, or not significant based
    on ``fc_threshold`` (|log2FC|) and ``p_threshold``. Optionally labels the top
    ``n_labels`` most significant hits using ``label_col``.

    Args:
        use_adjusted: If True, use the adjusted p-value column for the y-axis and
            the significance cutoff; otherwise use the raw p-value.
    """
    yp_col = adj_pvalue_col if use_adjusted else pvalue_col
    df = results[[log2fc_col, yp_col] + ([label_col] if label_col else [])].copy()
    df = df.dropna(subset=[log2fc_col, yp_col])
    df["_neglog10p"] = -np.log10(df[yp_col].clip(lower=np.finfo(float).tiny))

    sig = df[yp_col] <= p_threshold
    up = sig & (df[log2fc_col] >= fc_threshold)
    down = sig & (df[log2fc_col] <= -fc_threshold)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        df.loc[~(up | down), log2fc_col], df.loc[~(up | down), "_neglog10p"],
        s=12, c="lightgrey", label="ns", alpha=0.6, edgecolors="none",
    )
    ax.scatter(
        df.loc[up, log2fc_col], df.loc[up, "_neglog10p"],
        s=14, c="#c0392b", label="up", alpha=0.8, edgecolors="none",
    )
    ax.scatter(
        df.loc[down, log2fc_col], df.loc[down, "_neglog10p"],
        s=14, c="#2471a3", label="down", alpha=0.8, edgecolors="none",
    )

    ax.axhline(-np.log10(p_threshold), color="grey", ls="--", lw=0.8)
    ax.axvline(fc_threshold, color="grey", ls="--", lw=0.8)
    ax.axvline(-fc_threshold, color="grey", ls="--", lw=0.8)

    if label_col and n_labels > 0:
        top = df.loc[up | down].nlargest(n_labels, "_neglog10p")
        for _, row in top.iterrows():
            text = str(row[label_col])
            if text and text.lower() != "nan":
                ax.annotate(
                    text, (row[log2fc_col], row["_neglog10p"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points",
                )

    ylabel = "-log10(adj p-value)" if use_adjusted else "-log10(p-value)"
    ax.set_xlabel("log2(fold change)")
    ax.set_ylabel(ylabel)
    ax.set_title("Volcano plot")
    ax.legend(frameon=False, fontsize=8, loc="best")
    return ax


def abundance_heatmap(
    dataset: ProteomicsDataset,
    condition_map: ConditionMap,
    *,
    proteins: pd.Index | list | None = None,
    label_col: str = "Gene",
    zscore_rows: bool = True,
    cluster: bool = True,
    max_proteins: int = 50,
):
    """Heatmap of protein abundance across replicate columns.

    Args:
        proteins: Row index (subset) of proteins to show, e.g. the significant
            hits. Defaults to the first ``max_proteins`` rows.
        label_col: Annotation column used to label rows (falls back to index).
        zscore_rows: Z-score each protein (row) so relative patterns are visible.
        cluster: If True, use ``seaborn.clustermap`` (clusters rows & columns);
            otherwise a plain ``seaborn.heatmap``.

    Returns:
        A seaborn ``ClusterGrid`` when ``cluster`` is True, else a matplotlib ``Axes``.
    """
    condition_map.validate(dataset)
    cols = condition_map.assigned_columns
    data = dataset.data[cols]

    if proteins is not None:
        data = data.loc[proteins]
    if len(data) > max_proteins:
        data = data.iloc[:max_proteins]

    data = data.dropna(how="any")
    if data.empty:
        raise ValueError("No proteins with complete data to plot in the heatmap.")

    # Row labels from an annotation column if available.
    if label_col in dataset.raw.columns:
        labels = dataset.raw.loc[data.index, label_col].astype(str)
        data = data.set_axis(labels, axis=0)

    if zscore_rows:
        row_std = data.std(axis=1).replace(0, np.nan)
        data = data.sub(data.mean(axis=1), axis=0).div(row_std, axis=0).dropna(how="any")

    cmap = "vlag" if zscore_rows else "viridis"
    if cluster:
        grid = sns.clustermap(
            data, cmap=cmap, center=0 if zscore_rows else None,
            figsize=(max(6, 0.4 * data.shape[1] + 4), max(6, 0.25 * len(data) + 2)),
            xticklabels=True, yticklabels=True, cbar_kws={"label": "z-score" if zscore_rows else "abundance"},
        )
        grid.ax_heatmap.set_xlabel("Replicate")
        return grid

    _, ax = plt.subplots(figsize=(max(6, 0.5 * data.shape[1] + 3), max(5, 0.25 * len(data) + 2)))
    sns.heatmap(
        data, cmap=cmap, center=0 if zscore_rows else None, ax=ax,
        cbar_kws={"label": "z-score" if zscore_rows else "abundance"},
    )
    ax.set_xlabel("Replicate")
    ax.set_ylabel("Protein")
    return ax


def enrichment_barplot(
    enrichment: pd.DataFrame,
    *,
    top_n: int = 15,
    term_col: str = "Term",
    score_col: str = "Adj_P_value",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal bar plot of the top enriched terms by -log10(adjusted p-value)."""
    if enrichment.empty:
        raise ValueError("No enrichment results to plot.")

    df = enrichment.nsmallest(top_n, score_col).copy()
    df["_neglog10"] = -np.log10(df[score_col].clip(lower=np.finfo(float).tiny))
    df = df.iloc[::-1]  # largest bar on top

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(df) + 1)))

    ax.barh(df[term_col].astype(str), df["_neglog10"], color="#2c7fb8")
    ax.set_xlabel("-log10(adjusted p-value)")
    ax.set_title(f"Top {min(top_n, len(df))} enriched terms")
    ax.tick_params(axis="y", labelsize=8)
    return ax
