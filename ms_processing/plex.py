"""Multiplexing configuration.

Isobaric labelling (e.g. TMT) multiplexes several samples into one run. The plex
count tells the program how many quantitative data columns to expect. Standard
plexes are 6, 10 and 16, but any custom positive integer is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

# Standard isobaric-label plex sizes offered as defaults in the UI.
STANDARD_PLEXES: tuple[int, ...] = (6, 10, 16)


@dataclass(frozen=True)
class PlexConfig:
    """How many quantitative channels (data columns) the data set contains.

    Attributes:
        n_channels: Number of quantitative data columns / biological replicates.
    """

    n_channels: int

    def __post_init__(self) -> None:
        if not isinstance(self.n_channels, int) or isinstance(self.n_channels, bool):
            raise TypeError(f"n_channels must be an int, got {type(self.n_channels).__name__}")
        if self.n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {self.n_channels}")

    @property
    def is_standard(self) -> bool:
        """True if this plex matches one of the standard sizes (6/10/16)."""
        return self.n_channels in STANDARD_PLEXES

    @classmethod
    def standard(cls, n_channels: int) -> "PlexConfig":
        """Create a config from one of the standard plex sizes (6/10/16)."""
        if n_channels not in STANDARD_PLEXES:
            allowed = ", ".join(str(p) for p in STANDARD_PLEXES)
            raise ValueError(
                f"{n_channels} is not a standard plex ({allowed}). "
                f"Use PlexConfig({n_channels}) for a custom plex."
            )
        return cls(n_channels)
