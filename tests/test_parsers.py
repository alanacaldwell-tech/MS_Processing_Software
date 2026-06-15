"""Offline tests for the annotation/enrichment parsers and plotting.

These exercise the parsing and plotting logic without any network access, using
captured-style payloads. Run with:  python -m pytest tests/  (or python tests/test_parsers.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

# Make the package importable whether run from the repo root or tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ms_processing as msp


def test_parse_uniprot_entry():
    entry = {
        "primaryAccession": "P04637",
        "proteinDescription": {
            "recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}}
        },
        "genes": [{"geneName": {"value": "TP53"}}],
        "comments": [
            {"commentType": "FUNCTION", "texts": [{"value": "Acts as a tumor suppressor."}]},
            {
                "commentType": "SUBCELLULAR LOCATION",
                "subcellularLocations": [
                    {"location": {"value": "Nucleus"}},
                    {"location": {"value": "Cytoplasm"}},
                    {"location": {"value": "Nucleus"}},  # duplicate -> de-duped
                ],
            },
        ],
    }
    rec = msp.parse_uniprot_entry(entry)
    assert rec["Accession"] == "P04637"
    assert rec["UniProt_Protein_Name"] == "Cellular tumor antigen p53"
    assert rec["UniProt_Gene"] == "TP53"
    assert rec["Subcellular_Location"] == "Nucleus; Cytoplasm"
    assert "tumor suppressor" in rec["Function"]


def test_parse_uniprot_entry_missing_fields():
    rec = msp.parse_uniprot_entry({"primaryAccession": "X"})
    assert rec["Accession"] == "X"
    assert rec["Function"] == ""
    assert rec["Subcellular_Location"] == ""


def test_parse_enrichr_results_sorted():
    lib = msp.DEFAULT_LIBRARY
    payload = {
        lib: [
            [2, "cell cycle arrest", 3.4e-3, 2.0, 11.2, ["TP53", "CDKN1A"], 2.1e-2],
            [1, "apoptotic process", 1.2e-5, 3.1, 25.4, ["TP53", "BAX", "CASP3"], 4.0e-4],
        ]
    }
    df = msp.parse_enrichr_results(payload, lib)
    # sorted by adjusted p-value ascending
    assert df["Term"].iloc[0] == "apoptotic process"
    assert df["N_overlap"].iloc[0] == 3
    assert df["Adj_P_value"].iloc[0] < df["Adj_P_value"].iloc[1]


def test_plots_render():
    ds = msp.load_dataset("sample_data/sample_dataset.xlsx", "ABPP", 6)
    cmap = msp.ConditionMap.from_mapping(
        {
            "Treatment": ["Treatment_1", "Treatment_2", "Treatment_3"],
            "Mock": ["Mock_1", "Mock_2", "Mock_3"],
        }
    )
    res = msp.compare_conditions(ds, cmap, "Treatment", "Mock")
    assert msp.volcano_plot(res) is not None
    hits = msp.filter_results(res, max_fdr=0.05, min_abs_log2fc=1.0)
    assert msp.abundance_heatmap(ds, cmap, proteins=hits.index) is not None
    # Single-protein selection must not crash (clustermap empty-matrix guard).
    assert msp.abundance_heatmap(ds, cmap, proteins=res.index[:1]) is not None


def test_pca_runs_and_plots():
    ds = msp.load_dataset("sample_data/sample_dataset.xlsx", "ABPP", 6)
    cmap = msp.ConditionMap.from_mapping(
        {
            "Treatment": ["Treatment_1", "Treatment_2", "Treatment_3"],
            "Mock": ["Mock_1", "Mock_2", "Mock_3"],
        }
    )
    pca = msp.run_pca(ds, cmap, n_components=2)
    # 6 samples (replicates), Condition + PC1 + PC2 columns.
    assert pca.scores.shape[0] == 6
    assert {"Condition", "PC1", "PC2"}.issubset(pca.scores.columns)
    assert len(pca.explained_variance_ratio) == 2
    assert pca.loadings.shape[1] == 2
    assert msp.pca_plot(pca) is not None


if __name__ == "__main__":
    test_parse_uniprot_entry()
    test_parse_uniprot_entry_missing_fields()
    test_parse_enrichr_results_sorted()
    test_plots_render()
    test_pca_runs_and_plots()
    print("All tests passed.")
