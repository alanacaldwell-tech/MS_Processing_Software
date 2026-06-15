"""MS Processing Software.

In-house processing of bottom-up proteomics data (Perseus-like), designed to
handle multiple data sets exported as Excel files.

This package holds the reusable processing logic. The notebook(s) under
``notebooks/`` demonstrate the workflow, and a GUI/web layer can later import
these same functions so behaviour stays consistent between the two.
"""

from .columns import (
    DEFAULT_DATA_START_COLUMN,
    column_index_to_letter,
    column_letter_to_index,
)
from .experiments import EXPERIMENT_TYPES, ExperimentType, get_experiment_type
from .plex import STANDARD_PLEXES, PlexConfig
from .dataset import ProteomicsDataset, load_dataset
from .conditions import Condition, ConditionMap
from .stats import (
    CORRECTION_METHODS,
    DEFAULT_CORRECTION,
    adjust_pvalues,
    compare_conditions,
    condition_means,
)
from .filtering import DEFAULT_UNIQUE_PEPTIDES_COL, filter_results

__all__ = [
    "DEFAULT_DATA_START_COLUMN",
    "column_index_to_letter",
    "column_letter_to_index",
    "EXPERIMENT_TYPES",
    "ExperimentType",
    "get_experiment_type",
    "STANDARD_PLEXES",
    "PlexConfig",
    "ProteomicsDataset",
    "load_dataset",
    "Condition",
    "ConditionMap",
    "CORRECTION_METHODS",
    "DEFAULT_CORRECTION",
    "adjust_pvalues",
    "compare_conditions",
    "condition_means",
    "DEFAULT_UNIQUE_PEPTIDES_COL",
    "filter_results",
]

__version__ = "0.1.0"
