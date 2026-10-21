#!/usr/bin/env python3
"""Paired DeepShift-versus-SHIFTX2 comparison over every X-EISD restart.

Reads ``<run-root>/<pool>/<model>/rescore_all/results.csv`` (``run_paper_protocol.py``
output) for each pool and both models, and writes one JSON report: per pool, the paired
improvement (mean, median, percentile-bootstrap 95% interval of the mean, fraction of
restarts improved) in the sum of the seven non-chemical-shift log scores, in each of the
eight log scores, and in each of the eight RMSEs. Chemical shifts are the optimized
observable, so the seven others are the held-out endpoint. Restarts are paired by index.

    python analyze_pools.py --run-root <runs> --out <results.json>

Pools are bootstrapped in the order ENSEMBLE, MIXED, RANDOM from one seeded generator, so
the default order is the one the reported intervals come from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POOLS = ("ENSEMBLE", "MIXED", "RANDOM")
SCORE_COLUMNS = {
    "saxs": 2,
    "cs": 3,
    "fret": 4,
    "jc": 5,
    "noe": 6,
    "pre": 7,
    "rdc": 8,
    "rh": 9,
}
RMSE_COLUMNS = {
    "saxs": 10,
    "cs": 11,
    "fret": 12,
    "jc": 13,
    "noe": 14,
    "pre": 15,
    "rdc": 16,
    "rh": 17,
}
OFF_DIAGONAL = ("saxs", "fret", "jc", "noe", "pre", "rdc", "rh")


def bootstrap_mean_ci(
    differences: np.ndarray, rng: np.random.Generator, samples: int
) -> tuple[float, float]:
    estimates = np.empty(samples, dtype=np.float64)
    chunk = 500
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        estimates[start:stop] = differences[indices].mean(axis=1)
    return tuple(float(x) for x in np.percentile(estimates, [2.5, 97.5]))


def summarize(
    differences: np.ndarray, rng: np.random.Generator, samples: int
) -> dict:
    low, high = bootstrap_mean_ci(differences, rng, samples)
    return {
        "mean_improvement": float(np.mean(differences)),
        "median_improvement": float(np.median(differences)),
        "ci95_low": low,
        "ci95_high": high,
        "fraction_improved": float(np.mean(differences > 0)),
        "n_pairs": len(differences),
    }


def read_arm(root: Path, pool: str, model: str, epochs: int) -> np.ndarray:
    path = root / pool.lower() / model.lower() / "rescore_all/results.csv"
    values = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if values.shape != (epochs, 18):
        raise ValueError(f"{path}: expected {(epochs, 18)}, found {values.shape}")
    if not np.array_equal(values[:, 0], np.arange(epochs)):
        raise ValueError(f"{path}: restart indices are not 0..{epochs - 1}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--pools", nargs="+", default=list(POOLS), choices=list(POOLS),
                        help="pools with both arms present; the headline rule is "
                             "evaluated over these")
    args = parser.parse_args()

    rng = np.random.default_rng(args.bootstrap_seed)
    result = {
        "definition": {
            "improvement_direction": (
                "DeepShift-SHIFTX2 for log scores; SHIFTX2-DeepShift for RMSE"
            ),
            "primary": "sum of seven non-CS log scores",
            "bootstrap": "paired percentile bootstrap over all independent restarts",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "headline_pass_rule": (
                f"primary ci95_low > 0 in every evaluated pool: {list(args.pools)}"
            ),
        },
        "pools": {},
    }

    pools = tuple(args.pools)
    for pool in pools:
        baseline = read_arm(args.run_root, pool, "shiftx2", args.epochs)
        deep = read_arm(args.run_root, pool, "deepshift", args.epochs)
        primary_columns = [SCORE_COLUMNS[name] for name in OFF_DIAGONAL]
        primary_difference = (
            deep[:, primary_columns].sum(axis=1)
            - baseline[:, primary_columns].sum(axis=1)
        )
        pool_result = {
            "primary_non_cs_score": summarize(
                primary_difference, rng, args.bootstrap_samples
            ),
            "score": {},
            "rmse": {},
        }
        for name in SCORE_COLUMNS:
            difference = deep[:, SCORE_COLUMNS[name]] - baseline[:, SCORE_COLUMNS[name]]
            pool_result["score"][name] = summarize(
                difference, rng, args.bootstrap_samples
            )
        for name in RMSE_COLUMNS:
            difference = baseline[:, RMSE_COLUMNS[name]] - deep[:, RMSE_COLUMNS[name]]
            pool_result["rmse"][name] = summarize(
                difference, rng, args.bootstrap_samples
            )
        result["pools"][pool] = pool_result

    passed = all(
        result["pools"][pool]["primary_non_cs_score"]["ci95_low"] > 0
        for pool in pools
    )
    result["headline_pass"] = passed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"headline_pass={passed} -> {args.out}")
    for pool in pools:
        item = result["pools"][pool]["primary_non_cs_score"]
        print(
            f"{pool}: mean={item['mean_improvement']:.6g}, "
            f"95% CI [{item['ci95_low']:.6g}, {item['ci95_high']:.6g}], "
            f"fraction={item['fraction_improved']:.3f}"
        )


if __name__ == "__main__":
    main()
