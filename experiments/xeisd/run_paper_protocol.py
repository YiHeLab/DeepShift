#!/usr/bin/env python3
"""Run the X-EISD paper's chemical-shift-only protocol and rescore all eight observables.

For each pool and forward model, this performs 1,000 independent optimizations of
100-member ensembles. Each optimization uses the repository's 10,000 exchange proposals
and maximizes only the chemical-shift score. The saved final indices are then rescored in
a separate, non-optimizing pass on all eight observables; the second pass is needed
because the optimizer does not update scores for disabled observables.

SHIFTX2 is the X-EISD repository's own back-calculated shift table for each pool;
DeepShift is the table ``run_shift_tables.py`` and ``build_mixed.py`` wrote. Each arm
reseeds NumPy with ``--seed`` before its optimization, so arms are independent of the
order they run in. Writes, per arm, ``<out>/<pool>/<model>/optimize_cs/`` and
``rescore_all/`` (``indices.csv``, ``results.csv``) plus ``run.json``. The ``eisd`` package
is imported from ``XEISD_HOME``, unmodified.

    XEISD_HOME=<checkout> python run_paper_protocol.py --tables <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pools import xeisd_home

POOL_KEYS = {
    "ENSEMBLE": "ensemble",
    "RANDOM": "trades_uf",
    "MIXED": "mixed",
}


def run_arm(xeisd: Path, pool: str, model: str, cs_path: Path, out: Path,
            epochs: int, seed: int) -> None:
    from eisd.optimizer import main as eisd_main
    from eisd.parser import read_data
    from eisd.utils import meta_data

    files = meta_data(str(xeisd / "data") + "/")
    key = POOL_KEYS[pool]
    pool_files = dict(files[key])
    pool_files["cs"] = str(cs_path)
    exp_data = read_data(files["exp"], mode="exp")
    bc_data = read_data(pool_files, mode=key)

    optimize = out / "optimize_cs"
    rescore = out / "rescore_all"
    optimize.mkdir(parents=True, exist_ok=False)
    rescore.mkdir(parents=True, exist_ok=False)

    np.random.seed(seed)
    eisd_main(
        exp_data,
        bc_data,
        ens_size=100,
        epochs=epochs,
        mode="cs",
        beta=0.1,
        opt_type="max",
        output_dir=str(optimize),
        verbose=False,
    )
    indices = pd.read_csv(optimize / "indices.csv", header=None).to_numpy(dtype=int)
    if indices.shape != (epochs, 100):
        raise RuntimeError(f"unexpected optimized-index shape: {indices.shape}")
    eisd_main(
        exp_data,
        bc_data,
        pre_indices=indices,
        output_dir=str(rescore),
        verbose=False,
    )
    record = {
        "pool": pool,
        "model": model,
        "cs_path": str(cs_path.resolve()),
        "epochs": epochs,
        "ensemble_size": 100,
        "exchange_proposals_per_ensemble": 10000,
        "optimization_mode": "cs",
        "optimization_type": "max",
        "seed": seed,
        "rescore_mode": "all",
    }
    (out / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True,
                        help="directory holding drksh3_{ENSEMBLE,MIXED,RANDOM}_CS_deepshift.txt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000, help="independent restarts")
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--pools", nargs="+", choices=sorted(POOL_KEYS),
                        default=sorted(POOL_KEYS))
    parser.add_argument("--models", nargs="+", choices=["SHIFTX2", "DeepShift"],
                        default=["SHIFTX2", "DeepShift"])
    args = parser.parse_args()

    xeisd = xeisd_home()
    sys.path.insert(0, str(xeisd))
    paper_cs = {
        "ENSEMBLE": xeisd / "data/back_calc_data/ENSEMBLE_PED/drksh3_ENSEMBLE_CS.txt",
        "RANDOM": xeisd / "data/back_calc_data/Trades/drksh3_CS_structdata.txt",
        "MIXED": xeisd / "data/back_calc_data/mixpool/drksh3_mixpool_CS.txt",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for pool in args.pools:
        for model in args.models:
            out = args.out / pool.lower() / model.lower()
            if out.exists():
                raise FileExistsError(f"refusing to overwrite completed/partial arm: {out}")
            cs_path = (paper_cs[pool] if model == "SHIFTX2"
                       else args.tables / f"drksh3_{pool}_CS_deepshift.txt")
            print(f"=== {pool} / {model} ===", flush=True)
            run_arm(xeisd, pool, model, cs_path, out, args.epochs, args.seed)


if __name__ == "__main__":
    main()
