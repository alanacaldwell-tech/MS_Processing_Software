"""Loading and structuring a proteomics data set from Excel.

A data set is one Excel sheet where:
  * the first row holds column headers,
  * each subsequent row is one identified protein,
  * columns A-Y are annotation/metadata (protein IDs, gene names, ``# Unique
    Peptides``, ...),
  * quantitative data begins at column Z (configurable) and spans ``plex``
    columns, one per biological replicate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .columns import (
    DEFAULT_DATA_START_COLUMN,
    column_index_to_letter,
    column_letter_to_index,
)
from .experiments import ExperimentType, get_experiment_type
from .plex import PlexConfig

# Default annotation column holding the gene symbol for each protein.
DEFAULT_GENE_COLUMN = "Gene Symbol"


@dataclass
class ProteomicsDataset:
    """A loaded proteomics data set plus its experimental metadata.

    Attributes:
        raw: The full sheet as read from Excel (headers as columns).
        experiment_type: The declared experiment type.
        plex: Multiplexing configuration (number of data columns).
        data_start_column: Excel column letter where data begins (default "Z").
        is_log_transformed: True once the quantitative data has been log-scaled
            (downstream fold change is then a difference of means, not a ratio).
        normalization: Name of the normalization applied (None if raw).
    """

    raw: pd.DataFrame
    experiment_type: ExperimentType
    plex: PlexConfig
    data_start_column: str = DEFAULT_DATA_START_COLUMN
    is_log_transformed: bool = False
    normalization: str | None = None

    def __post_init__(self) -> None:
        start = self.data_start_index
        needed = start + self.plex.n_channels
        if needed > self.raw.shape[1]:
            have = self.raw.shape[1]
            raise ValueError(
                f"Sheet has {have} columns but a {self.plex.n_channels}-plex data set "
                f"starting at column {self.data_start_column} "
                f"(index {start}) requires at least {needed} columns."
            )

    @property
    def data_start_index(self) -> int:
        """0-based index of the first data column."""
        return column_letter_to_index(self.data_start_column)

    @property
    def annotation_columns(self) -> list[str]:
        """Names of the annotation/metadata columns (everything before data)."""
        return list(self.raw.columns[: self.data_start_index])

    @property
    def data_columns(self) -> list[str]:
        """Names of the quantitative data columns (one per biological replicate)."""
        start = self.data_start_index
        return list(self.raw.columns[start : start + self.plex.n_channels])

    @property
    def annotations(self) -> pd.DataFrame:
        """The annotation/metadata sub-frame."""
        return self.raw[self.annotation_columns]

    @property
    def data(self) -> pd.DataFrame:
        """The quantitative data sub-frame (numeric coercion applied)."""
        return self.raw[self.data_columns].apply(pd.to_numeric, errors="coerce")

    def describe(self) -> dict:
        """A small summary dict useful for display / debugging."""
        first = self.data_start_column
        last = column_index_to_letter(self.data_start_index + self.plex.n_channels - 1)
        return {
            "experiment_type": self.experiment_type.name,
            "plex": self.plex.n_channels,
            "n_proteins": int(self.raw.shape[0]),
            "n_annotation_columns": len(self.annotation_columns),
            "data_column_range": f"{first}-{last}",
            "data_columns": self.data_columns,
            "normalization": self.normalization,
            "is_log_transformed": self.is_log_transformed,
        }


def load_dataset(
    path: str | Path,
    experiment_type: str | ExperimentType,
    plex: int | PlexConfig,
    *,
    sheet_name: str | int = 0,
    data_start_column: str = DEFAULT_DATA_START_COLUMN,
) -> ProteomicsDataset:
    """Read an Excel file into a :class:`ProteomicsDataset`.

    Args:
        path: Path to the ``.xlsx`` file.
        experiment_type: Experiment type key/name/alias or an ``ExperimentType``.
        plex: Plex count (int) or a ``PlexConfig``.
        sheet_name: Sheet to read (name or index). Defaults to the first sheet.
        data_start_column: Excel column letter where data begins. Defaults to "Z".

    The first row of the sheet is always treated as the header.
    """
    exp = experiment_type if isinstance(experiment_type, ExperimentType) else get_experiment_type(experiment_type)
    plex_cfg = plex if isinstance(plex, PlexConfig) else PlexConfig(plex)

    raw = pd.read_excel(path, sheet_name=sheet_name, header=0, engine="openpyxl")

    return ProteomicsDataset(
        raw=raw,
        experiment_type=exp,
        plex=plex_cfg,
        data_start_column=data_start_column,
    )


@dataclass
class DatasetSpec:
    """Per-file loading configuration for a single Excel data set.

    Use this when files do NOT share the same layout — each file carries its own
    experiment type, plex, data-start column and sheet. ``name`` defaults to the
    file stem and is used as the key in the loaded-datasets mapping.
    """

    path: str | Path
    experiment_type: str | ExperimentType
    plex: int | PlexConfig
    name: str | None = None
    data_start_column: str = DEFAULT_DATA_START_COLUMN
    sheet_name: str | int = 0

    @property
    def resolved_name(self) -> str:
        return self.name or Path(self.path).stem


def load_datasets(
    sources: Sequence[str | Path | DatasetSpec],
    experiment_type: str | ExperimentType | None = None,
    plex: int | PlexConfig | None = None,
    *,
    data_start_column: str = DEFAULT_DATA_START_COLUMN,
    sheet_name: str | int = 0,
    names: Sequence[str] | None = None,
) -> dict[str, ProteomicsDataset]:
    """Load one or more Excel files into a name -> :class:`ProteomicsDataset` map.

    Two modes, matching "are the columns identical across files?":

    * **Identical columns** — pass a list of file paths plus shared
      ``experiment_type`` and ``plex`` (and optionally ``data_start_column`` /
      ``sheet_name``). The same configuration is applied to every file. Provide
      ``names`` to label them; otherwise file stems are used.
    * **Adjusted per file** — pass a list of :class:`DatasetSpec`, each carrying
      its own configuration. ``experiment_type`` / ``plex`` here are ignored.

    Raises:
        ValueError: if paths are given without shared ``experiment_type``/``plex``,
            if ``names`` length mismatches, or if two data sets resolve to the
            same name.
    """
    if not sources:
        raise ValueError("No files provided to load.")

    all_specs = all(isinstance(s, DatasetSpec) for s in sources)
    any_specs = any(isinstance(s, DatasetSpec) for s in sources)
    if any_specs and not all_specs:
        raise ValueError("Mix of paths and DatasetSpec is not supported; use one or the other.")

    if all_specs:
        specs: list[DatasetSpec] = list(sources)  # type: ignore[arg-type]
    else:
        if experiment_type is None or plex is None:
            raise ValueError(
                "When passing file paths, 'experiment_type' and 'plex' are required "
                "(shared across all files). For per-file settings pass DatasetSpec objects."
            )
        if names is not None and len(names) != len(sources):
            raise ValueError(f"Got {len(names)} names for {len(sources)} files.")
        specs = [
            DatasetSpec(
                path=path,
                experiment_type=experiment_type,
                plex=plex,
                name=(names[i] if names is not None else None),
                data_start_column=data_start_column,
                sheet_name=sheet_name,
            )
            for i, path in enumerate(sources)
        ]

    datasets: dict[str, ProteomicsDataset] = {}
    for spec in specs:
        name = spec.resolved_name
        if name in datasets:
            raise ValueError(f"Duplicate data set name {name!r}; give explicit unique names.")
        datasets[name] = load_dataset(
            spec.path,
            experiment_type=spec.experiment_type,
            plex=spec.plex,
            sheet_name=spec.sheet_name,
            data_start_column=spec.data_start_column,
        )
    return datasets
