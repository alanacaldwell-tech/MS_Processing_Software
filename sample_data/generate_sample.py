"""Generate small synthetic proteomics Excel files for demos/tests.

The layout mirrors the real expectation:
  * columns A-Y are annotation/metadata (only a few are meaningful here, the
    rest are filler so that data still starts at column Z),
  * column Z onward holds the quantitative data (one column per biological
    replicate) for a 6-plex example: 3 "Treatment" + 3 "Mock" replicates.

Two files are written over the same protein list so cross-data-set comparison can
be demonstrated:
  * ``sample_dataset.xlsx``   — the primary demo data set,
  * ``sample_dataset_2.xlsx`` — a second data set with correlated-but-different
    treatment effects (a "replication" experiment).

Run:  python sample_data/generate_sample.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N_PROTEINS = 200
# Shared protein list so the two data sets are comparable by Accession/Gene.
ACCESSIONS = [f"P{i:05d}" for i in range(N_PROTEINS)]
GENES = [f"GENE{i}" for i in range(N_PROTEINS)]


def _annotations(rng: np.random.Generator) -> dict:
    # Leave a few gene symbols blank to demonstrate UniProt-accession mapping.
    gene_symbols = list(GENES)
    for i in (3, 17, 42):
        if i < len(gene_symbols):
            gene_symbols[i] = ""

    annotations = {
        "Accession": ACCESSIONS,
        "Gene Symbol": gene_symbols,
        "Description": ["Example protein"] * N_PROTEINS,
        "# Unique Peptides": rng.integers(1, 25, size=N_PROTEINS),
    }
    # Pad with filler columns so real data lands exactly on column Z (26th).
    for j in range(len(annotations), 25):
        annotations[f"Filler_{j}"] = ["" for _ in range(N_PROTEINS)]
    return annotations


def make_dataset(path: Path, *, seed: int, effect_noise: float = 0.0) -> None:
    """Write one synthetic data set.

    ``effect_noise`` perturbs the per-protein treatment effect so a second data
    set is correlated with, but not identical to, the first.
    """
    rng = np.random.default_rng(seed)
    annotations = _annotations(rng)

    baseline = rng.lognormal(mean=10, sigma=1.0, size=N_PROTEINS)

    # Deterministic effect (shared across data sets via this fixed RNG), so the
    # two data sets share the same up/down proteins; effect_noise adds variation.
    effect_rng = np.random.default_rng(2024)
    effect = np.ones(N_PROTEINS)
    up = effect_rng.choice(N_PROTEINS, size=30, replace=False)
    down = effect_rng.choice(np.setdiff1d(np.arange(N_PROTEINS), up), size=30, replace=False)
    effect[up] = effect_rng.uniform(2.0, 4.0, size=up.size)
    effect[down] = effect_rng.uniform(0.25, 0.5, size=down.size)
    if effect_noise:
        effect = effect * rng.lognormal(mean=0, sigma=effect_noise, size=N_PROTEINS)

    def replicates(scale: np.ndarray, n: int) -> list[np.ndarray]:
        return [scale * rng.lognormal(mean=0, sigma=0.15, size=N_PROTEINS) for _ in range(n)]

    data = {}
    for i, col in enumerate(replicates(baseline * effect, 3), start=1):
        data[f"Treatment_{i}"] = col
    for i, col in enumerate(replicates(baseline, 3), start=1):
        data[f"Mock_{i}"] = col

    frame = pd.DataFrame({**annotations, **data})
    frame.to_excel(path, index=False, engine="openpyxl")
    print(f"Wrote {path} ({frame.shape[0]} proteins x {frame.shape[1]} columns)")


def main() -> None:
    make_dataset(HERE / "sample_dataset.xlsx", seed=42)
    make_dataset(HERE / "sample_dataset_2.xlsx", seed=7, effect_noise=0.25)


if __name__ == "__main__":
    main()
