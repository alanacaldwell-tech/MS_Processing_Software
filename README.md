# MS Processing Software

In-house, Perseus-like processing of **bottom-up proteomics** data, designed to
handle multiple data sets exported as Excel files.

This first iteration is a **Python notebook** backed by a reusable package
(`ms_processing/`). The processing logic lives in importable functions so a
future GUI / web front end can call the exact same code — keeping behaviour
consistent between the notebook and the eventual app.

## Pipeline

**ingest → group biological replicates into conditions → average → compare
conditions (fold change + statistics) → filter → visualize → annotate → GO enrichment**

1. **Ingest** an Excel file. First row = headers; each row = one identified
   protein. Columns A–Y are annotation/metadata (protein IDs, gene names,
   `# Unique Peptides`, …); quantitative data starts at column **Z** and spans
   the plex count (6 / 10 / 16 / custom), one column per biological replicate.
2. **Group** replicate columns into named experimental conditions
   (e.g. Z–AB = "Treatment").
3. **Average** protein abundance per protein per condition.
4. **Compare** two conditions (numerator vs denominator): fold change,
   log2(fold change), two-sided independent t-test p-value, adjusted p-value /
   FDR (selectable correction method), and −log10(p-value).
5. **Filter** by p-value, FDR, |log2 fold change|, and `# Unique Peptides`.
6. **Visualize**: volcano plot (log2FC vs −log10 p, with non-overlapping labels
   for the top significant / largest fold-change hits) and a clustered abundance
   heatmap of the significant proteins.
7. **PCA**: principal component analysis over the samples (replicates as
   observations, proteins as features) to assess replicate clustering and
   condition separation.
8. **Annotate** each protein with its **subcellular localization** and a snapshot
   of its **function** via the UniProt REST API.
9. **GO enrichment**: over-representation test of the significant genes against a
   GO gene-set library via the Enrichr API.
10. **Compare across data sets**: load **one or more** Excel files (with columns
    either identical across files or configured per file via `DatasetSpec`),
    align results by protein, build a side-by-side metric table, quantify the
    overlap of significant hits, and correlate fold changes between data sets.

### Network note

Steps 7–8 call external services (`rest.uniprot.org`, `maayanlab.cloud`) and need
outbound internet access at runtime. They raise `AnnotationServiceError` /
`EnrichmentServiceError` if unreachable, and the notebook degrades gracefully so
the rest of the pipeline still runs offline.

## Data assumptions

- First Excel row is the header.
- Each row is one protein.
- Proteins are keyed on a `Gene Symbol` column; blanks are filled from the
  `Accession` (UniProt) column via `fill_gene_symbols`.
- Quantitative data begins at column **Z** (configurable via `data_start_column`).
- Plex count = number of data columns (biological replicates).

## Setup

```bash
pip install -r requirements.txt
```

## Run the demo

The repo ships a synthetic data set so everything runs without real data:

```bash
python sample_data/generate_sample.py        # creates sample_data/sample_dataset.xlsx
jupyter notebook notebooks/proteomics_processing.ipynb
```

## Using the package directly

```python
import ms_processing as msp

ds = msp.load_dataset("sample_data/sample_dataset.xlsx", "ABPP", plex=6)

conditions = msp.ConditionMap.from_mapping({
    "Treatment": ["Treatment_1", "Treatment_2", "Treatment_3"],
    "Mock":      ["Mock_1", "Mock_2", "Mock_3"],
})

results = msp.compare_conditions(ds, conditions, "Treatment", "Mock", correction="fdr_bh")
hits = msp.filter_results(results, max_fdr=0.05, min_abs_log2fc=1.0, min_unique_peptides=2)
```

## Package layout

| Module | Responsibility |
|---|---|
| `ms_processing/columns.py` | Excel column-letter ↔ index helpers (`Z` default data start) |
| `ms_processing/experiments.py` | Experiment-type registry (ABPP, AP-MS, whole proteome, …) |
| `ms_processing/plex.py` | Multiplex config (6/10/16/custom) |
| `ms_processing/dataset.py` | Load Excel (one or many); split annotation vs. data columns |
| `ms_processing/mapping.py` | Fill missing `Gene Symbol`s from UniProt accessions |
| `ms_processing/conditions.py` | Assign replicate columns to conditions |
| `ms_processing/stats.py` | Means, fold change, t-tests, multiple-testing correction |
| `ms_processing/filtering.py` | Filter results by p-value / FDR / fold change / peptides |
| `ms_processing/plots.py` | Volcano, heatmap, PCA, enrichment bar, cross-data-set scatter |
| `ms_processing/multivariate.py` | PCA over samples (scikit-learn) |
| `ms_processing/crossdataset.py` | Load & compare results across multiple data sets |
| `ms_processing/annotations.py` | UniProt lookups: subcellular localization + function |
| `ms_processing/enrichment.py` | GO term enrichment via Enrichr |

## Tests

```bash
python tests/test_parsers.py        # offline parser + plotting checks
```

## Roadmap

- Remote file storage for uploaded data sets.
- Web / GUI front end on top of `ms_processing`.
- Normalization & imputation steps.
