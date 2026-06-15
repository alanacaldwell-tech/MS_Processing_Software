"""Experiment-type definitions.

When a user uploads a data set they declare what kind of experiment produced it
(ABPP, AP-MS, whole proteome, ...). The experiment type carries metadata about
the experimental design that downstream steps can use. The registry is
intentionally easy to extend as new assay types are supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentType:
    """Describes a category of proteomics experiment.

    Attributes:
        key: Short machine-friendly identifier (e.g. ``"abpp"``).
        name: Human-readable name shown in the UI.
        description: One-line explanation of the assay.
        aliases: Alternative names/abbreviations that should resolve to this type.
    """

    key: str
    name: str
    description: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


# Registry of supported experiment types. Extend this list to add more.
EXPERIMENT_TYPES: dict[str, ExperimentType] = {
    exp.key: exp
    for exp in (
        ExperimentType(
            key="abpp",
            name="ABPP",
            description="Activity-based protein profiling.",
            aliases=("activity-based protein profiling",),
        ),
        ExperimentType(
            key="ap_ms",
            name="AP-MS",
            description="Affinity purification mass spectrometry (interactomics).",
            aliases=("apms", "ap-ms", "affinity purification"),
        ),
        ExperimentType(
            key="whole_proteome",
            name="Whole proteome",
            description="Global / whole-proteome expression profiling.",
            aliases=("whole proteome", "global", "expression", "total proteome"),
        ),
    )
}


def get_experiment_type(name: str) -> ExperimentType:
    """Resolve an experiment type by key, name, or alias (case-insensitive).

    Raises:
        KeyError: if no matching experiment type is registered.
    """
    needle = name.strip().lower()
    for exp in EXPERIMENT_TYPES.values():
        candidates = {exp.key.lower(), exp.name.lower(), *(a.lower() for a in exp.aliases)}
        if needle in candidates:
            return exp
    valid = ", ".join(sorted(EXPERIMENT_TYPES)) or "(none)"
    raise KeyError(f"Unknown experiment type {name!r}. Registered keys: {valid}.")
