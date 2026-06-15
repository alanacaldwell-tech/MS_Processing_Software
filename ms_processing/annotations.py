"""Protein annotation lookups via the UniProt REST API.

For a set of UniProt accessions this fetches, per protein:
  * the recommended protein name,
  * a snapshot of the protein's function (the UniProt FUNCTION comment),
  * its subcellular localization.

Network access to ``rest.uniprot.org`` is required at runtime. The HTTP layer is
isolated from the parsing layer (:func:`parse_uniprot_entry`) so the parsing can
be unit-tested offline and so an alternative backend could be slotted in later.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

try:  # requests is only needed for the live fetch, not for parsing.
    import requests
except ImportError:  # pragma: no cover - exercised only without requests installed
    requests = None  # type: ignore[assignment]

UNIPROT_ACCESSIONS_URL = "https://rest.uniprot.org/uniprotkb/accessions"

# Fields requested from UniProt. Kept small for speed.
UNIPROT_FIELDS = (
    "accession",
    "id",
    "protein_name",
    "gene_names",
    "cc_function",
    "cc_subcellular_location",
)

# Max accessions per request (UniProt accepts a generous list; chunk to be safe).
_CHUNK_SIZE = 100


class AnnotationServiceError(RuntimeError):
    """Raised when the annotation service cannot be reached or returns an error."""


def parse_uniprot_entry(entry: dict) -> dict:
    """Extract the fields we care about from one UniProt JSON entry.

    Pure function (no network) so it can be tested with captured responses.
    Missing pieces come back as empty strings rather than raising.
    """
    accession = entry.get("primaryAccession", "")

    # Protein name: recommended -> submitted -> "".
    desc = entry.get("proteinDescription", {})
    protein_name = ""
    rec = desc.get("recommendedName") or {}
    if rec:
        protein_name = rec.get("fullName", {}).get("value", "")
    if not protein_name:
        submitted = desc.get("submissionNames") or []
        if submitted:
            protein_name = submitted[0].get("fullName", {}).get("value", "")

    # Gene name: first gene's primary name.
    genes = entry.get("genes") or []
    gene_name = ""
    if genes:
        gene_name = genes[0].get("geneName", {}).get("value", "")

    # Function + subcellular location come from typed comments.
    function = ""
    locations: list[str] = []
    for comment in entry.get("comments", []):
        ctype = comment.get("commentType")
        if ctype == "FUNCTION" and not function:
            texts = comment.get("texts") or []
            if texts:
                function = texts[0].get("value", "")
        elif ctype == "SUBCELLULAR LOCATION":
            for loc in comment.get("subcellularLocations", []):
                value = loc.get("location", {}).get("value")
                if value:
                    locations.append(value)

    return {
        "Accession": accession,
        "UniProt_Protein_Name": protein_name,
        "UniProt_Gene": gene_name,
        "Function": function,
        "Subcellular_Location": "; ".join(dict.fromkeys(locations)),  # de-dup, keep order
    }


def _chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_annotations(
    accessions: Iterable[str],
    *,
    timeout: float = 30.0,
    session: "requests.Session | None" = None,
) -> pd.DataFrame:
    """Fetch annotations for ``accessions`` from UniProt.

    Returns a DataFrame with columns ``Accession``, ``UniProt_Protein_Name``,
    ``UniProt_Gene``, ``Function`` and ``Subcellular_Location``. Accessions with
    no UniProt entry are still represented (with empty annotation fields) so the
    result aligns 1:1 with the input.

    Raises:
        AnnotationServiceError: if ``requests`` is unavailable or the service
            cannot be reached / returns an error.
    """
    if requests is None:
        raise AnnotationServiceError(
            "The 'requests' package is required for UniProt lookups. "
            "Install it with `pip install requests`."
        )

    # De-duplicate while preserving order, drop blanks.
    unique = list(dict.fromkeys(a for a in (str(x).strip() for x in accessions) if a))
    owns_session = session is None
    session = session or requests.Session()

    parsed: dict[str, dict] = {}
    try:
        for chunk in _chunks(unique, _CHUNK_SIZE):
            params = {
                "accessions": ",".join(chunk),
                "fields": ",".join(UNIPROT_FIELDS),
                "format": "json",
            }
            resp = session.get(UNIPROT_ACCESSIONS_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            for entry in resp.json().get("results", []):
                record = parse_uniprot_entry(entry)
                if record["Accession"]:
                    parsed[record["Accession"]] = record
    except requests.RequestException as exc:  # pragma: no cover - needs network
        raise AnnotationServiceError(f"UniProt request failed: {exc}") from exc
    finally:
        if owns_session:
            session.close()

    empty = {
        "UniProt_Protein_Name": "",
        "UniProt_Gene": "",
        "Function": "",
        "Subcellular_Location": "",
    }
    rows = [parsed.get(acc, {"Accession": acc, **empty}) for acc in unique]
    return pd.DataFrame(rows, columns=["Accession", *empty.keys()])


def annotate_results(
    results: pd.DataFrame,
    *,
    accession_col: str = "Accession",
    timeout: float = 30.0,
    session: "requests.Session | None" = None,
) -> pd.DataFrame:
    """Add UniProt annotation columns to a results/dataset table.

    Looks up the accessions in ``results[accession_col]`` and left-merges the
    annotation columns (protein name, function, subcellular location) onto it.
    """
    if accession_col not in results.columns:
        raise KeyError(
            f"Column {accession_col!r} not found. Available: {list(results.columns)}."
        )
    annotations = fetch_annotations(
        results[accession_col], timeout=timeout, session=session
    )
    return results.merge(annotations, on=accession_col, how="left")
