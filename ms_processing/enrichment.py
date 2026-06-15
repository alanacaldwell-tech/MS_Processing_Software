"""GO term (and other) enrichment analysis via the Enrichr API.

Given a list of genes (typically the significant hits from a comparison), Enrichr
performs an over-representation test against a chosen gene-set library and returns
enriched terms with p-values and Benjamini-Hochberg adjusted p-values.

Network access to ``maayanlab.cloud`` is required at runtime. The HTTP layer is
isolated from the parsing layer (:func:`parse_enrichr_results`) so parsing can be
unit-tested offline.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

ENRICHR_ADD_LIST_URL = "https://maayanlab.cloud/Enrichr/addList"
ENRICHR_ENRICH_URL = "https://maayanlab.cloud/Enrichr/enrich"

# Common GO gene-set libraries offered by Enrichr.
GO_LIBRARIES: dict[str, str] = {
    "BP": "GO_Biological_Process_2023",
    "MF": "GO_Molecular_Function_2023",
    "CC": "GO_Cellular_Component_2023",
}
DEFAULT_LIBRARY = GO_LIBRARIES["BP"]

# Column order for the parsed results table.
_RESULT_COLUMNS = [
    "Rank",
    "Term",
    "P_value",
    "Adj_P_value",
    "Z_score",
    "Combined_score",
    "Overlapping_genes",
    "N_overlap",
]


class EnrichmentServiceError(RuntimeError):
    """Raised when the enrichment service cannot be reached or returns an error."""


def parse_enrichr_results(payload: dict, library: str) -> pd.DataFrame:
    """Parse an Enrichr ``/enrich`` JSON payload into a tidy DataFrame.

    Pure function (no network). Enrichr returns, per term, a list shaped like::

        [rank, term, p_value, z_score, combined_score, [genes], adj_p_value, ...]

    Results are sorted by adjusted p-value ascending.
    """
    rows = []
    for item in payload.get(library, []):
        genes = item[5] if len(item) > 5 and isinstance(item[5], list) else []
        rows.append(
            {
                "Rank": item[0] if len(item) > 0 else None,
                "Term": item[1] if len(item) > 1 else "",
                "P_value": item[2] if len(item) > 2 else float("nan"),
                "Z_score": item[3] if len(item) > 3 else float("nan"),
                "Combined_score": item[4] if len(item) > 4 else float("nan"),
                "Overlapping_genes": genes,
                "Adj_P_value": item[6] if len(item) > 6 else float("nan"),
                "N_overlap": len(genes),
            }
        )
    df = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    if not df.empty:
        df = df.sort_values("Adj_P_value", kind="stable").reset_index(drop=True)
    return df


def enrich_genes(
    genes: Iterable[str],
    *,
    library: str = DEFAULT_LIBRARY,
    description: str = "ms_processing gene list",
    timeout: float = 30.0,
    session: "requests.Session | None" = None,
) -> pd.DataFrame:
    """Run an Enrichr over-representation test for ``genes`` against ``library``.

    Args:
        genes: Gene symbols to test (e.g. significant hits from a comparison).
        library: Enrichr gene-set library name (see :data:`GO_LIBRARIES`).
        description: Label sent to Enrichr for the submitted list.

    Returns:
        DataFrame of enriched terms (Term, P_value, Adj_P_value, Combined_score,
        Overlapping_genes, ...) sorted by adjusted p-value.

    Raises:
        EnrichmentServiceError: if ``requests`` is unavailable, no valid genes are
            provided, or the service cannot be reached / returns an error.
    """
    if requests is None:
        raise EnrichmentServiceError(
            "The 'requests' package is required for Enrichr. "
            "Install it with `pip install requests`."
        )

    gene_list = [g for g in (str(x).strip() for x in genes) if g]
    if not gene_list:
        raise EnrichmentServiceError("No valid gene symbols supplied for enrichment.")

    owns_session = session is None
    session = session or requests.Session()
    try:
        add_resp = session.post(
            ENRICHR_ADD_LIST_URL,
            files={"list": (None, "\n".join(gene_list)), "description": (None, description)},
            timeout=timeout,
        )
        add_resp.raise_for_status()
        user_list_id = add_resp.json().get("userListId")
        if user_list_id is None:
            raise EnrichmentServiceError("Enrichr did not return a userListId.")

        enrich_resp = session.get(
            ENRICHR_ENRICH_URL,
            params={"userListId": user_list_id, "backgroundType": library},
            timeout=timeout,
        )
        enrich_resp.raise_for_status()
        payload = enrich_resp.json()
    except requests.RequestException as exc:  # pragma: no cover - needs network
        raise EnrichmentServiceError(f"Enrichr request failed: {exc}") from exc
    finally:
        if owns_session:
            session.close()

    return parse_enrichr_results(payload, library)
