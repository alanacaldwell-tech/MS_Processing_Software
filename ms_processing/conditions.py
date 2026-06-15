"""Grouping biological-replicate columns into named experimental conditions.

Each quantitative data column is a separate biological replicate. The user
groups replicate columns that belong to the same experimental condition
(e.g. columns Z-AB are three replicates of a "Treatment" condition). This module
captures that mapping and validates it against a data set. It is exactly the
information a GUI condition-assignment form would collect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .dataset import ProteomicsDataset


@dataclass(frozen=True)
class Condition:
    """One experimental condition and the replicate columns that make it up.

    Attributes:
        name: Condition label (e.g. "Treatment", "Mock").
        replicate_columns: The data-column names that are replicates of it.
    """

    name: str
    replicate_columns: tuple[str, ...]

    @property
    def n_replicates(self) -> int:
        return len(self.replicate_columns)


class ConditionMap:
    """An ordered collection of :class:`Condition` objects for a data set."""

    def __init__(self, conditions: Sequence[Condition]):
        names = [c.name for c in conditions]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate condition names: {sorted(dupes)}")
        self._conditions: tuple[Condition, ...] = tuple(conditions)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Sequence[str]]) -> "ConditionMap":
        """Build from ``{condition_name: [replicate_column, ...]}``."""
        return cls([Condition(name, tuple(cols)) for name, cols in mapping.items()])

    def __iter__(self):
        return iter(self._conditions)

    def __len__(self) -> int:
        return len(self._conditions)

    def __getitem__(self, name: str) -> Condition:
        for c in self._conditions:
            if c.name == name:
                return c
        raise KeyError(f"No condition named {name!r}. Available: {self.names}.")

    @property
    def names(self) -> list[str]:
        return [c.name for c in self._conditions]

    @property
    def assigned_columns(self) -> list[str]:
        """All replicate columns assigned across every condition (in order)."""
        return [col for c in self._conditions for col in c.replicate_columns]

    def unassigned_columns(self, dataset: ProteomicsDataset) -> list[str]:
        """Data columns of ``dataset`` not assigned to any condition."""
        assigned = set(self.assigned_columns)
        return [col for col in dataset.data_columns if col not in assigned]

    def validate(self, dataset: ProteomicsDataset) -> None:
        """Check the mapping is consistent with ``dataset``.

        Raises ``ValueError`` if a referenced column is not a data column of the
        data set, or if a column is assigned to more than one condition.
        """
        valid_cols = set(dataset.data_columns)
        assigned = self.assigned_columns

        unknown = [c for c in assigned if c not in valid_cols]
        if unknown:
            raise ValueError(
                f"Columns assigned to conditions but not data columns of the "
                f"data set: {unknown}. Data columns are {dataset.data_columns}."
            )

        seen: set[str] = set()
        overlap = {c for c in assigned if c in seen or seen.add(c)}
        if overlap:
            raise ValueError(f"Columns assigned to more than one condition: {sorted(overlap)}")
