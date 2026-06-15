"""Helpers for translating between Excel column letters and indices.

The application assumes that the quantitative data columns begin at Excel
column ``Z`` (the 26th column, 1-based). Columns A-Y are therefore treated as
annotation/metadata (protein IDs, gene names, descriptions, etc.).
"""

from __future__ import annotations

# Excel column where the quantitative data is assumed to start (1-based 26).
DEFAULT_DATA_START_COLUMN = "Z"


def column_letter_to_index(letter: str) -> int:
    """Convert an Excel column letter (e.g. ``"A"``, ``"Z"``, ``"AA"``) to a
    0-based column index.

    >>> column_letter_to_index("A")
    0
    >>> column_letter_to_index("Z")
    25
    >>> column_letter_to_index("AA")
    26
    """
    letter = letter.strip().upper()
    if not letter or not letter.isalpha():
        raise ValueError(f"Invalid Excel column letter: {letter!r}")

    index = 0
    for char in letter:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1  # convert from 1-based to 0-based


def column_index_to_letter(index: int) -> str:
    """Convert a 0-based column index to an Excel column letter.

    >>> column_index_to_letter(0)
    'A'
    >>> column_index_to_letter(25)
    'Z'
    >>> column_index_to_letter(26)
    'AA'
    """
    if index < 0:
        raise ValueError(f"Column index must be non-negative, got {index}")

    letters = ""
    index += 1  # convert to 1-based for the math below
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters
