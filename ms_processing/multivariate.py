"""Principal component analysis (PCA) on a proteomics data set.

PCA treats each **biological replicate (data column) as a sample** and each
**protein as a feature** — the usual orientation for proteomics QC, where you
want to see whether replicates of the same condition cluster together and
whether conditions separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .conditions import ConditionMap
from .dataset import ProteomicsDataset


@dataclass
class PCAResult:
    """Result of a PCA run.

    Attributes:
        scores: Sample x component DataFrame (PC coordinates), with a
            ``Condition`` column when a condition map was supplied.
        explained_variance_ratio: Fraction of variance per component.
        loadings: Protein x component DataFrame (feature contributions).
        n_components: Number of components computed.
    """

    scores: pd.DataFrame
    explained_variance_ratio: np.ndarray
    loadings: pd.DataFrame
    n_components: int

    def variance_label(self, component: int) -> str:
        """Axis label like ``"PC1 (42.1%)"`` for a 1-based component number."""
        pct = self.explained_variance_ratio[component - 1] * 100
        return f"PC{component} ({pct:.1f}%)"


def _prepare_matrix(
    dataset: ProteomicsDataset,
    *,
    scale: bool,
    log_transform: bool,
    method: str,
):
    """Build the samples x proteins matrix used by PCA.

    Returns ``(matrix, x)`` where ``matrix`` is the (index-preserving) DataFrame
    and ``x`` is the numeric array after optional log/scale transforms.
    """
    matrix = dataset.data.T  # samples (replicates) x proteins
    matrix = matrix.dropna(axis=1, how="any")
    if matrix.shape[0] < 2:
        raise ValueError(f"{method} needs at least 2 samples (data columns).")
    if matrix.shape[1] < 2:
        raise ValueError(f"{method} needs at least 2 proteins with complete data.")

    if log_transform:
        matrix = np.log2(matrix.clip(lower=0) + 1)

    x = matrix.to_numpy(dtype=float)
    if scale:
        x = StandardScaler().fit_transform(x)
    return matrix, x


def _attach_conditions(
    scores: pd.DataFrame,
    dataset: ProteomicsDataset,
    condition_map: ConditionMap | None,
) -> pd.DataFrame:
    """Insert a ``Condition`` column mapping each sample to its condition."""
    if condition_map is None:
        return scores
    condition_map.validate(dataset)
    col_to_condition = {
        col: cond.name for cond in condition_map for col in cond.replicate_columns
    }
    scores.insert(0, "Condition", [col_to_condition.get(s, "unassigned") for s in scores.index])
    return scores


def run_pca(
    dataset: ProteomicsDataset,
    condition_map: ConditionMap | None = None,
    *,
    n_components: int = 2,
    scale: bool = True,
    log_transform: bool | None = None,
) -> PCAResult:
    """Run PCA over the quantitative data of ``dataset``.

    Samples (data columns) are the observations and proteins are the features.
    Proteins with any missing value across samples are dropped (PCA needs a
    complete matrix).

    Args:
        condition_map: Optional; if given, each sample's condition is attached to
            the scores so plots can color by condition. Replicate columns not
            assigned to a condition are labelled ``"unassigned"``.
        n_components: Number of principal components (capped at the smaller of the
            sample/feature counts).
        scale: Standardize each protein (zero mean, unit variance) before PCA so
            high-abundance proteins don't dominate. Recommended.
        log_transform: Apply ``log2(x + 1)`` first. Defaults to ``None``, which
            logs only when the dataset is not already log-scaled (so a normalized
            dataset isn't logged twice, and raw data still gets logged for PCA).

    Returns:
        A :class:`PCAResult`.
    """
    if log_transform is None:
        log_transform = not dataset.is_log_transformed
    matrix, x = _prepare_matrix(
        dataset, scale=scale, log_transform=log_transform, method="PCA"
    )

    k = min(n_components, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=k)
    scores_arr = pca.fit_transform(x)

    component_names = [f"PC{i}" for i in range(1, k + 1)]
    scores = pd.DataFrame(scores_arr, index=matrix.index, columns=component_names)
    scores = _attach_conditions(scores, dataset, condition_map)

    loadings = pd.DataFrame(pca.components_.T, index=matrix.columns, columns=component_names)

    return PCAResult(
        scores=scores,
        explained_variance_ratio=pca.explained_variance_ratio_,
        loadings=loadings,
        n_components=k,
    )
