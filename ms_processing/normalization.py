"""Normalization of quantitative proteomics data.

Following common best practice (and the Perseus convention), the recommended
path is **log2 transform followed by per-sample median centering**:

  1. ``log2`` the intensities — stabilizes variance and makes the right-skewed
     distribution approximately Gaussian. Non-positive values become missing.
  2. subtract each sample's (column's) median so every sample is centered at 0,
     correcting for differences in total load / instrument sensitivity.

Once data is log-scaled, downstream fold change is a *difference* of means
(handled automatically by :func:`ms_processing.stats.compare_conditions` via the
dataset's ``is_log_transformed`` flag).

Methods are selectable; ``vsn`` (variance-stabilizing normalization) is reserved
as a future alternative and currently raises ``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import ProteomicsDataset

# method key -> (human label, applies a log transform?)
NORMALIZATION_METHODS: dict[str, str] = {
    "none": "No normalization (leave data as-is)",
    "log2": "log2 transform only",
    "log2_median": "log2 transform + per-sample median centering (recommended)",
    "median": "per-sample median centering only (assumes data already log-scaled)",
    "vsn": "Variance-stabilizing normalization (not yet implemented)",
}
DEFAULT_NORMALIZATION = "log2_median"

# Methods that put the data on a log2 scale.
_LOG_METHODS = {"log2", "log2_median"}


def log2_transform(data: pd.DataFrame) -> pd.DataFrame:
    """log2-transform a numeric frame; non-positive values become NaN."""
    numeric = data.apply(pd.to_numeric, errors="coerce")
    return np.log2(numeric.where(numeric > 0))


def median_center(data: pd.DataFrame) -> pd.DataFrame:
    """Subtract each column's (sample's) median from that column."""
    numeric = data.apply(pd.to_numeric, errors="coerce")
    return numeric.sub(numeric.median(axis=0, skipna=True), axis=1)


def normalize_data(data: pd.DataFrame, method: str = DEFAULT_NORMALIZATION) -> tuple[pd.DataFrame, bool]:
    """Normalize a data frame (samples in columns).

    Returns ``(normalized, is_log)`` where ``is_log`` says whether the result is
    on a log2 scale.

    Raises:
        ValueError: for an unknown method.
        NotImplementedError: for ``vsn`` (reserved for a future release).
    """
    if method not in NORMALIZATION_METHODS:
        valid = ", ".join(sorted(NORMALIZATION_METHODS))
        raise ValueError(f"Unknown normalization {method!r}. Choose from: {valid}.")

    if method == "vsn":
        raise NotImplementedError(
            "VSN normalization is not implemented yet. Use 'log2_median' (recommended) "
            "or 'median'. VSN is reserved as a future selectable alternative."
        )

    if method == "none":
        return data.apply(pd.to_numeric, errors="coerce"), False
    if method == "log2":
        return log2_transform(data), True
    if method == "median":
        return median_center(data), False
    # log2_median
    return median_center(log2_transform(data)), True


def normalize_dataset(
    dataset: ProteomicsDataset, method: str = DEFAULT_NORMALIZATION
) -> ProteomicsDataset:
    """Return a new :class:`ProteomicsDataset` with normalized quantitative data.

    Annotation columns are preserved; only the data columns are replaced. The
    returned dataset records the ``normalization`` used and its
    ``is_log_transformed`` flag, which the stats/PCA steps read automatically.
    """
    normalized, is_log = normalize_data(dataset.data, method)

    new_raw = dataset.raw.copy()
    new_raw[dataset.data_columns] = normalized.to_numpy()

    return ProteomicsDataset(
        raw=new_raw,
        experiment_type=dataset.experiment_type,
        plex=dataset.plex,
        data_start_column=dataset.data_start_column,
        is_log_transformed=is_log,
        normalization=method,
    )
