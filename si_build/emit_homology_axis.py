#!/usr/bin/env python3
"""The homology axis Figure S2 plots, from the run named by ``$DEEPSHIFT_RUN``.

Writes, into ``--out-dir``:

  sid_to_training_identity.json   structure id -> best MMseqs2 identity of its entity to
                                  any training-fold entity (0.0 when no hit reaches the
                                  coverage floor), the instrument Table S8 uses
                                  (``run_best_hit_identity``, coverage 0.50, cov-mode 2)

Usage:
    DEEPSHIFT_RUN=/path/to/run DEEPSHIFT_BENCH_OUT=/path/to/run/benchmark \\
        python emit_homology_axis.py --out-dir /path/to/run/final/homology_axis
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import si_common

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from deepshift.data import census as corpus
from deepshift.data import folds as fold_stage
from deepshift.data.discover import load_fasta
from deepshift.data.sequence_clustering import run_best_hit_identity

ENTITY_FASTA = "entity_query.fasta"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=si_common.root("DEEPSHIFT_RUN"))
    ap.add_argument("--bench", type=Path, default=None,
                    help="benchmark root holding fold6_tasks.tsv (default $DEEPSHIFT_BENCH_OUT)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--coverage", type=float, default=0.50)
    ap.add_argument("--cov-mode", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    bench = args.bench or Path(os.environ.get("DEEPSHIFT_BENCH_OUT") or args.run / "benchmark")
    data_dir = args.run / "data"
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    ledger = corpus.read_realizations(data_dir)
    fold_of = corpus.read_fold_map(data_dir)
    corpus.bind_fold_map(fold_of, set(ledger["label_set_id"]))
    sequences = load_fasta(data_dir / ENTITY_FASTA)
    entity_of = dict(zip(ledger["label_set_id"], ledger["logical_entity_id"], strict=True))
    training_folds = tuple(range(1, fold_stage.N_FOLDS + 1))
    training = {e for ls, e in entity_of.items() if fold_of[ls] in training_folds}
    held = {e for ls, e in entity_of.items() if fold_of[ls] == fold_stage.HOLDOUT_FOLD}
    if training & held:
        raise SystemExit("an entity is both held out and in training")
    queries = {e: sequences[e] for e in sorted(held) if e in sequences}
    targets = {e: sequences[e] for e in sorted(training) if e in sequences}
    hits, _ = run_best_hit_identity(queries, targets, coverage=args.coverage,
                                    cov_mode=args.cov_mode, threads=args.threads)

    with open(bench / "fold6_tasks.tsv") as fh:
        tasks = list(csv.DictReader(fh, delimiter="\t"))
    axis, unmeasured = {}, []
    for t in tasks:
        entity = entity_of.get(t["label_set_id"])
        if entity is None or entity not in queries:
            unmeasured.append(t["structure_id"])
            axis[t["structure_id"]] = 0.0
            continue
        hit = hits.get(entity)
        axis[t["structure_id"]] = float(hit["identity"]) if hit else 0.0
    (out / "sid_to_training_identity.json").write_text(json.dumps(axis, indent=1) + "\n")

    n_with_hit = sum(1 for v in axis.values() if v > 0)
    print(f"wrote {out}: {len(tasks)} structures, {n_with_hit} with a hit, "
          f"{len(unmeasured)} unmeasured")


if __name__ == "__main__":
    main()
