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
from .dataset import DEFAULT_GENE_COLUMN, ProteomicsDataset


def volcano_plot(
    results: pd.DataFrame,
    *,
    log2fc_col: str = "log2_fold_change",
    pvalue_col: str = "p_value",
    use_adjusted: bool = False,
    adj_pvalue_col: str = "adj_p_value",
    fc_threshold: float = 1.0,
    p_threshold: float = 0.05,
    label_col: str | None = DEFAULT_GENE_COLUMN,
    n_labels: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Volcano plot: log2 fold change (x) vs -log10 p-value (y).

    Points are colored as up-regulated, down-regulated, or not significant based
    on ``fc_threshold`` (|log2FC|) and ``p_threshold``.

    Labelling targets the most interesting hits by taking the union of the top
    ``n_labels`` most significant and the top ``n_labels`` largest |log2FC|
    (restricted to significant points). When the ``adjustText`` package is
    installed, labels are repelled so they don't overlap each other or the
    points; otherwise a simple offset is used.

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
        candidates = df.loc[up | down]
        # Union of the most significant and the largest-magnitude fold changes.
        by_sig = candidates.nlargest(n_labels, "_neglog10p")
        by_fc = candidates.reindex(candidates[log2fc_col].abs().sort_values(ascending=False).index).head(n_labels)
        to_label = pd.concat([by_sig, by_fc])
        to_label = to_label[~to_label.index.duplicated()]

        texts = []
        for _, row in to_label.iterrows():
            text = str(row[label_col])
            if text and text.lower() != "nan":
                texts.append(
                    ax.text(row[log2fc_col], row["_neglog10p"], text, fontsize=7)
                )

        try:  # repel labels so they don't overlap, if adjustText is available
            from adjustText import adjust_text

            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
                expand=(1.2, 1.4),
            )
        except ImportError:
            # Fallback: nudge labels off their points (may overlap on dense plots).
            for t in texts:
                t.set_position((t.get_position()[0] + 0.03, t.get_position()[1] + 0.03))

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
    label_col: str = DEFAULT_GENE_COLUMN,
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

    if data.empty:
        raise ValueError(
            "No proteins left to plot in the heatmap after preparing the data. "
            "If z-scoring is on, every selected protein may have had zero variance "
            "across replicates; try zscore_rows=False or widen the protein selection."
        )

    cmap = "vlag" if zscore_rows else "viridis"

    # Clustering needs >= 2 observations along a dimension. Fall back gracefully
    # when only one protein/replicate is present so a 1-row heatmap still renders.
    can_cluster_rows = data.shape[0] >= 2
    can_cluster_cols = data.shape[1] >= 2
    if cluster and (can_cluster_rows or can_cluster_cols):
        grid = sns.clustermap(
            data, cmap=cmap, center=0 if zscore_rows else None,
            row_cluster=can_cluster_rows, col_cluster=can_cluster_cols,
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


def _embedding_scatter(
    scores: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    label_samples: bool,
    draw_origin: bool,
    ax: plt.Axes | None,
) -> plt.Axes:
    """Shared sample-scatter used by the PCA plot, colored by condition."""
    for col in (x_col, y_col):
        if col not in scores.columns:
            raise ValueError(f"{col} not available in the embedding.")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    if "Condition" in scores.columns:
        for condition, sub in scores.groupby("Condition"):
            ax.scatter(sub[x_col], sub[y_col], s=60, alpha=0.85, label=str(condition))
        ax.legend(frameon=False, fontsize=8, title="Condition")
    else:
        ax.scatter(scores[x_col], scores[y_col], s=60, alpha=0.85)

    if label_samples:
        for sample, row in scores.iterrows():
            ax.annotate(
                str(sample), (row[x_col], row[y_col]),
                fontsize=7, xytext=(4, 4), textcoords="offset points",
            )

    if draw_origin:
        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def pca_plot(
    pca_result,
    *,
    pc_x: int = 1,
    pc_y: int = 2,
    label_samples: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of PCA sample scores, colored by condition.

    Args:
        pca_result: A :class:`~ms_processing.multivariate.PCAResult`.
        pc_x, pc_y: 1-based component numbers for the x and y axes.
        label_samples: Annotate each point with its sample (replicate) name.
    """
    return _embedding_scatter(
        pca_result.scores, f"PC{pc_x}", f"PC{pc_y}",
        xlabel=pca_result.variance_label(pc_x),
        ylabel=pca_result.variance_label(pc_y),
        title="PCA of samples",
        label_samples=label_samples, draw_origin=True, ax=ax,
    )


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


def compare_scatter(
    aligned: pd.DataFrame,
    name_x: str,
    name_y: str,
    *,
    metric: str = "log2_fold_change",
    label_col: str | None = DEFAULT_GENE_COLUMN,
    annotate_corr: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter of a metric (e.g. log2 fold change) between two data sets.

    Takes the wide table from :func:`ms_processing.crossdataset.align_results` and
    plots ``metric__name_x`` vs ``metric__name_y`` for proteins measured in both,
    with a y=x reference line and (optionally) the Pearson correlation.
    """
    x_col, y_col = f"{metric}__{name_x}", f"{metric}__{name_y}"
    for col in (x_col, y_col):
        if col not in aligned.columns:
            raise KeyError(f"Column {col!r} not found in aligned table.")

    pair = aligned[[c for c in (x_col, y_col, label_col) if c]].dropna(subset=[x_col, y_col])
    if pair.empty:
        raise ValueError("No proteins measured in both data sets to compare.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(pair[x_col], pair[y_col], s=14, alpha=0.6, edgecolors="none", c="#34495e")

    lo = float(min(pair[x_col].min(), pair[y_col].min()))
    hi = float(max(pair[x_col].max(), pair[y_col].max()))
    ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=0.8)

    if annotate_corr:
        r = pair[x_col].corr(pair[y_col])
        ax.text(0.05, 0.95, f"r = {r:.2f}\nn = {len(pair)}",
                transform=ax.transAxes, va="top", fontsize=9)

    ax.set_xlabel(f"{metric} ({name_x})")
    ax.set_ylabel(f"{metric} ({name_y})")
    ax.set_title(f"{metric}: {name_x} vs {name_y}")
    return ax
