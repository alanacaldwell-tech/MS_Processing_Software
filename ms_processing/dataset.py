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


@dataclass
class ProteomicsDataset:
    """A loaded proteomics data set plus its experimental metadata.

    Attributes:
        raw: The full sheet as read from Excel (headers as columns).
        experiment_type: The declared experiment type.
        plex: Multiplexing configuration (number of data columns).
        data_start_column: Excel column letter where data begins (default "Z").
    """

    raw: pd.DataFrame
    experiment_type: ExperimentType
    plex: PlexConfig
    data_start_column: str = DEFAULT_DATA_START_COLUMN

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
