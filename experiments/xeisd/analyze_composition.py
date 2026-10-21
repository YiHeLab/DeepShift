#!/usr/bin/env python3
"""Which conformers chemical-shift-only X-EISD selects from the 50:50 MIXED pool.

Reads ``<runs>/mixed/<model>/optimize_cs/indices.csv`` (``run_paper_protocol.py`` output)
for SHIFTX2 and DeepShift: 1,000 restarts of 100 selected conformers each, where an index
below 1,700 is an ENSEMBLE conformer and the rest are RANDOM. Writes one JSON with each
model's ENSEMBLE fraction (mean, standard deviation across restarts, median, 2.5-97.5%
quantiles), the DeepShift-minus-SHIFTX2 difference with a bootstrap interval over
independent restarts, and the probability that a random DeepShift restart selects a higher
ENSEMBLE fraction than a random SHIFTX2 restart. The reference is the 76% ENSEMBLE the
X-EISD paper reached optimizing all eight observables jointly.

    python analyze_composition.py --runs <runs> --out <mixed_composition.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = ("shiftx2", "deepshift")
N_ENSEMBLE_POOL = 1_700
N_MIXED_POOL = 3_400
ENSEMBLE_SIZE = 100
EXPECTED_RESTARTS = 1_000
PAPER_ENSEMBLE_FRACTION = 0.76
INITIAL_ENSEMBLE_FRACTION = 0.50
BOOTSTRAP_SEED = 20260731
BOOTSTRAP_RESAMPLES = 20_000


def load_fraction(runs: Path, model: str) -> np.ndarray:
    path = runs / "mixed" / model / "optimize_cs" / "indices.csv"
    indices = np.loadtxt(path, delimiter=",", dtype=np.int64)
    expected = (EXPECTED_RESTARTS, ENSEMBLE_SIZE)
    if indices.shape != expected:
        raise ValueError(f"{path}: expected {expected}, found {indices.shape}")
    if indices.min() < 0 or indices.max() >= N_MIXED_POOL:
        raise ValueError(f"{path}: index outside 0..{N_MIXED_POOL - 1}")
    if any(len(np.unique(row)) != ENSEMBLE_SIZE for row in indices):
        raise ValueError(f"{path}: at least one optimized ensemble has duplicates")
    # The parser discards the source IDs and addresses rows by position:
    # positions 0:1700 are ENSEMBLE; 1700:3400 are RANDOM.
    return (indices < N_ENSEMBLE_POOL).mean(axis=1)


def summarize(values: np.ndarray) -> dict[str, object]:
    return {
        "mean_ensemble_fraction": float(values.mean()),
        "mean_random_fraction": float(1.0 - values.mean()),
        "sd_across_restarts": float(values.std(ddof=1)),
        "median_ensemble_fraction": float(np.median(values)),
        "restart_quantile_2p5_97p5": [
            float(x) for x in np.quantile(values, [0.025, 0.975])
        ],
        "absolute_distance_from_paper_0p76": float(
            abs(values.mean() - PAPER_ENSEMBLE_FRACTION)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True,
                        help="the run_paper_protocol.py output directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fractions = {model: load_fraction(args.runs, model) for model in MODELS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = EXPECTED_RESTARTS
    difference = np.empty(BOOTSTRAP_RESAMPLES)
    for draw in range(BOOTSTRAP_RESAMPLES):
        deep = fractions["deepshift"][rng.integers(0, n, size=n)].mean()
        shiftx = fractions["shiftx2"][rng.integers(0, n, size=n)].mean()
        difference[draw] = deep - shiftx

    deep = fractions["deepshift"]
    shiftx = fractions["shiftx2"]
    pairwise = deep[:, None] - shiftx[None, :]
    probability = float(
        ((pairwise > 0).sum() + 0.5 * (pairwise == 0).sum()) / pairwise.size
    )
    output = {
        "question": (
            "composition selected by chemical shifts alone from the 50:50 "
            "X-EISD MIXED pool"
        ),
        "published_reference": {
            "optimization": "joint optimization using all experimental data types",
            "ensemble_fraction": PAPER_ENSEMBLE_FRACTION,
            "random_fraction": 1.0 - PAPER_ENSEMBLE_FRACTION,
            "source": "Lincoff et al., Communications Chemistry 3, 74 (2020)",
            "doi": "10.1038/s42004-020-0323-0",
        },
        "current_protocol": {
            "optimization": "chemical shifts only",
            "initial_ensemble_fraction": INITIAL_ENSEMBLE_FRACTION,
            "restarts_per_model": EXPECTED_RESTARTS,
            "conformers_per_optimized_ensemble": ENSEMBLE_SIZE,
            "important_scope": (
                "optimizer-restart evidence on one protein; not protein-level "
                "generalization and not a known ground-truth population"
            ),
        },
        "models": {model: summarize(fractions[model]) for model in MODELS},
        "deepshift_minus_shiftx2": {
            "mean_ensemble_fraction": float(deep.mean() - shiftx.mean()),
            "independent_restart_bootstrap_ci95": [
                float(x) for x in np.quantile(difference, [0.025, 0.975])
            ],
            "probability_random_deepshift_restart_has_higher_ensemble_fraction": (
                probability
            ),
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
