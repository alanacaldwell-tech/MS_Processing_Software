"""Identifier mapping: fill missing gene symbols from UniProt accessions.

The pipeline keys proteins on a gene-symbol column (default ``"Gene Symbol"``).
When a row has no gene symbol, its UniProt accession (from the ``"Accession"``
column) is looked up via the UniProt REST API and the empty cell is filled with
the gene name UniProt reports.

This reuses :func:`ms_processing.annotations.fetch_annotations`, so it needs
outbound access to ``rest.uniprot.org`` and raises ``AnnotationServiceError`` if
the service is unreachable.
"""

from __future__ import annotations

import pandas as pd

from .annotations import fetch_annotations
from .dataset import DEFAULT_GENE_COLUMN


def _is_missing(series: pd.Series) -> pd.Series:
    """Boolean mask of cells that are NaN, empty, or the string 'nan'."""
    text = series.astype(str).str.strip()
    return series.isna() | (text == "") | (text.str.lower() == "nan")


def fill_gene_symbols(
    df: pd.DataFrame,
    *,
    gene_col: str = DEFAULT_GENE_COLUMN,
    accession_col: str = "Accession",
    session=None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Return a copy of ``df`` with missing gene symbols filled from UniProt.

    For every row whose ``gene_col`` is blank, the ``accession_col`` value is
    looked up in UniProt and the gene name is written into ``gene_col``. Rows that
    already have a gene symbol are left untouched. If ``gene_col`` is absent it is
    created (so every row is resolved from its accession).

    Args:
        gene_col: Gene-symbol column to fill (default ``"Gene Symbol"``).
        accession_col: Column holding the UniProt accession.

    Raises:
        KeyError: if ``accession_col`` is not present.
        AnnotationServiceError: if the UniProt lookup fails.
    """
    if accession_col not in df.columns:
        raise KeyError(
            f"Column {accession_col!r} not found; cannot map gene symbols. "
            f"Available: {list(df.columns)}."
        )

    out = df.copy()
    if gene_col not in out.columns:
        out[gene_col] = pd.NA

    missing = _is_missing(out[gene_col])
    if not missing.any():
        return out  # nothing to do (no network call)

    accessions = [
        a for a in out.loc[missing, accession_col].astype(str).str.strip() if a and a.lower() != "nan"
    ]
    if not accessions:
        return out

    annotations = fetch_annotations(accessions, session=session, timeout=timeout)
    lookup = {
        acc: gene
        for acc, gene in zip(annotations["Accession"], annotations["UniProt_Gene"])
        if isinstance(gene, str) and gene.strip()
    }

    def resolve(acc) -> object:
        return lookup.get(str(acc).strip(), pd.NA)

    filled = out.loc[missing, accession_col].map(resolve)
    out.loc[missing, gene_col] = filled.where(filled.notna(), out.loc[missing, gene_col])
    return out
