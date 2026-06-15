"""Generate a small synthetic proteomics Excel file for demos/tests.

The layout mirrors the real expectation:
  * columns A-Y are annotation/metadata (only a few are meaningful here, the
    rest are filler so that data still starts at column Z),
  * column Z onward holds the quantitative data (one column per biological
    replicate) for a 6-plex example: 3 "Treatment" + 3 "Mock" replicates.

Run:  python sample_data/generate_sample.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent / "sample_dataset.xlsx"

N_PROTEINS = 200
RNG = np.random.default_rng(42)


def main() -> None:
    # --- Annotation columns A-Y (25 columns) ---------------------------------
    annotations = {
        "Accession": [f"P{i:05d}" for i in range(N_PROTEINS)],
        "Gene": [f"GENE{i}" for i in range(N_PROTEINS)],
        "Description": ["Example protein"] * N_PROTEINS,
        "# Unique Peptides": RNG.integers(1, 25, size=N_PROTEINS),
    }
    # Pad with filler columns so real data lands exactly on column Z (26th).
    for j in range(len(annotations), 25):
        annotations[f"Filler_{j}"] = ["" for _ in range(N_PROTEINS)]

    # --- Quantitative data columns Z onward (6-plex) -------------------------
    # Baseline abundance per protein; Treatment gets a multiplicative effect on
    # a subset of proteins so there is real signal to detect.
    baseline = RNG.lognormal(mean=10, sigma=1.0, size=N_PROTEINS)
    effect = np.ones(N_PROTEINS)
    up = RNG.choice(N_PROTEINS, size=30, replace=False)
    down = RNG.choice(np.setdiff1d(np.arange(N_PROTEINS), up), size=30, replace=False)
    effect[up] = RNG.uniform(2.0, 4.0, size=up.size)
    effect[down] = RNG.uniform(0.25, 0.5, size=down.size)

    def replicates(scale: np.ndarray, n: int) -> list[np.ndarray]:
        return [scale * RNG.lognormal(mean=0, sigma=0.15, size=N_PROTEINS) for _ in range(n)]

    data = {}
    for i, col in enumerate(replicates(baseline * effect, 3), start=1):
        data[f"Treatment_{i}"] = col
    for i, col in enumerate(replicates(baseline, 3), start=1):
        data[f"Mock_{i}"] = col

    frame = pd.DataFrame({**annotations, **data})
    frame.to_excel(OUTPUT, index=False, engine="openpyxl")
    print(f"Wrote {OUTPUT} ({frame.shape[0]} proteins x {frame.shape[1]} columns)")


if __name__ == "__main__":
    main()
