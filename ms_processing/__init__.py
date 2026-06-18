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
from .dataset import (
    DEFAULT_GENE_COLUMN,
    DatasetSpec,
    ProteomicsDataset,
    load_dataset,
    load_datasets,
)
from .conditions import Condition, ConditionMap
from .mapping import fill_gene_symbols
from .stats import (
    CORRECTION_METHODS,
    DEFAULT_CORRECTION,
    adjust_pvalues,
    compare_conditions,
    condition_means,
)
from .filtering import DEFAULT_UNIQUE_PEPTIDES_COL, filter_results
from .annotations import (
    AnnotationServiceError,
    annotate_results,
    fetch_annotations,
    parse_uniprot_entry,
)
from .enrichment import (
    DEFAULT_LIBRARY,
    GO_LIBRARIES,
    EnrichmentServiceError,
    enrich_genes,
    parse_enrichr_results,
)
from .plots import (
    abundance_heatmap,
    compare_scatter,
    enrichment_barplot,
    pca_plot,
    volcano_plot,
)
from .multivariate import PCAResult, run_pca
from .crossdataset import (
    align_results,
    compare_across_datasets,
    correlate_metric,
    overlap_summary,
    significant_sets,
)

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
    "DatasetSpec",
    "load_datasets",
    "DEFAULT_GENE_COLUMN",
    "fill_gene_symbols",
    "Condition",
    "ConditionMap",
    "CORRECTION_METHODS",
    "DEFAULT_CORRECTION",
    "adjust_pvalues",
    "compare_conditions",
    "condition_means",
    "DEFAULT_UNIQUE_PEPTIDES_COL",
    "filter_results",
    "AnnotationServiceError",
    "annotate_results",
    "fetch_annotations",
    "parse_uniprot_entry",
    "DEFAULT_LIBRARY",
    "GO_LIBRARIES",
    "EnrichmentServiceError",
    "enrich_genes",
    "parse_enrichr_results",
    "abundance_heatmap",
    "compare_scatter",
    "enrichment_barplot",
    "pca_plot",
    "volcano_plot",
    "PCAResult",
    "run_pca",
    "align_results",
    "compare_across_datasets",
    "correlate_metric",
    "overlap_summary",
    "significant_sets",
]

__version__ = "0.1.0"
