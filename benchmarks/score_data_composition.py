#!/usr/bin/env python3
"""The two training-composition measurements Table S22 prints, from held-out dumps.

The table asks two different questions and its two blocks score different cohorts, so
they are measured here in one program but never pooled.

  block 1  At a fixed number of training rows, is a row of a new protein worth more than
           a row of another structure of a protein already in the set? Three arms grown
           from one floor -- floor, redundant, diverse -- compared pairwise. The record is
           ``arms_comparison.json``: the arm composition read back from the construction
           report, and three paired contrasts.

  block 2  Once every protein is in the training set, do its extra deposited structures
           still reduce error? One structure per protein against every structure, on the
           source-matched cohort -- the label sets an AlphaFold model, a crystal structure
           and a solution-NMR conformer all realize -- so the three test lines score the
           same proteins seen three ways. The record is ``source_matched_report.json``.

Every arm holds out fold 1 and trains on folds 2-5, and a materialization manifest never
restricts the held-out fold, so all five arms describe the same rows. They are aligned on
their shared rows before any comparison rather than trusted to line up, and
``deepshift.evaluate`` does the aligning, the scoring and the clustered bootstrap, so no
statistic here is computed by a rule the rest of the project does not use.

Usage:
  python -m benchmarks.score_data_composition \\
      --floor <npz> --redundant <npz> --diverse <npz> --arms-report <json> \\
      --one-per-protein <npz> --shipped-all <npz> \\
      --out-comparison <json> --out-source-matched <json>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from deepshift import evaluate as ev
from deepshift.domain import VALID_SOURCES
from deepshift.normalization import BACKBONE_SHIFTS
from deepshift.statistics import DEFAULT_SEED

#: Block 1's three contrasts, as ``(baseline, candidate)``. Both growth paths start at the
#: floor, and the third is the headline: equal rows, more proteins.
CONTRASTS = (("floor", "redundant"), ("floor", "diverse"), ("redundant", "diverse"))
#: Block 2's contrast, one per test source. Positive delta favours keeping every structure.
BLOCK2_BASELINE, BLOCK2_CANDIDATE = "one_per_protein", "shipped_all"
DEFAULT_RESAMPLES = 2000


def common_rows(arms: dict) -> dict:
    """Restrict every arm to the rows and nuclei all of them share, in one order.

    ``align`` settles two arms; a table needs one common row set or its columns are not
    comparable to each other. The first pass shrinks a reference to the rows every arm
    holds, the second brings each arm onto that reference.
    """
    names = list(arms)
    reference = arms[names[0]]
    for name in names[1:]:
        reference, _ = ev.align(reference, arms[name])
    return {name: ev.align(reference, arms[name])[1] for name in names}


def contrast_block(arms: dict, columns, *, boots, seed) -> dict:
    """Block 1: each pair as ``baseline -> candidate``, positive delta favouring candidate."""
    return {f"{base}__to__{cand}":
            ev.compare(arms[base], arms[cand], columns=columns, boots=boots, seed=seed)
            for base, cand in CONTRASTS}


def source_block(arms: dict, columns, *, boots, seed) -> dict:
    """Block 2: per test source, on the marginal and the source-matched cohorts.

    The matched cohort is what the table reads. The marginal one is reported beside it
    because matching shrinks the denominator, and by how much is a finding rather than a
    detail.
    """
    reference = next(iter(arms.values()))
    matched = ev.matched_rows(reference)
    source = reference.source
    report = {"n_common_rows": int(reference.n_rows), "arms": list(arms), "cohorts": {}}
    for code, name in enumerate(VALID_SOURCES):
        marginal = source == code
        report["cohorts"][name] = {
            "marginal_rows": int(marginal.sum()),
            "marginal_label_sets": int(np.unique(reference.label_set[marginal]).size),
            "matched_rows": int((marginal & matched).sum()),
            "matched_label_sets": int(
                np.unique(reference.label_set[marginal & matched]).size)}
    report["table_a"] = {}
    for cohort, keep in (("matched", matched), ("marginal", np.ones_like(matched))):
        report["table_a"][cohort] = {
            name: {arm: ev.score(preds, columns=columns, rows=keep & (source == code),
                                 boots=boots, seed=seed)
                   for arm, preds in arms.items()}
            for code, name in enumerate(VALID_SOURCES)}
    report["table_b"] = {}
    for code, name in enumerate(VALID_SOURCES):
        rows = matched & (source == code)
        key = f"{name}__{BLOCK2_BASELINE}__vs__{BLOCK2_CANDIDATE}"
        report["table_b"][key] = {
            "size_matched": False,
            "cohort": "source-matched label sets of the held-out fold",
            "nuclei": ev.compare(_restrict(arms[BLOCK2_BASELINE], rows),
                                 _restrict(arms[BLOCK2_CANDIDATE], rows),
                                 columns=columns, boots=boots, seed=seed)}
    return report


def _restrict(preds, rows):
    """The same arm with every cell outside ``rows`` unscorable, row order untouched.

    ``compare`` pairs two arms positionally and refuses arms whose rows do not line up,
    so a cohort is applied as a mask over the cells rather than as a row filter.
    """
    return dataclasses.replace(preds, effective_mask=preds.effective_mask & rows[:, None])


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, role in (("floor", "block 1, the shared origin"),
                       ("redundant", "block 1, the same proteins seen three ways"),
                       ("diverse", "block 1, three times the proteins at equal rows"),
                       ("one-per-protein", "block 2, one structure per protein"),
                       ("shipped-all", "block 2, the paper member that holds out fold 1")):
        ap.add_argument(f"--{name}", required=True, type=Path,
                        help=f"held-out prediction dump: {role}")
    ap.add_argument("--arms-report", required=True, type=Path,
                    help="build_arms.py's composition report, carried into block 1")
    ap.add_argument("--out-comparison", required=True, type=Path)
    ap.add_argument("--out-source-matched", required=True, type=Path)
    ap.add_argument("--boots", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap


def main(argv=None):
    ns = build_parser().parse_args(argv)
    columns = list(BACKBONE_SHIFTS)

    paths = {"floor": ns.floor, "redundant": ns.redundant, "diverse": ns.diverse,
             BLOCK2_BASELINE: ns.one_per_protein, BLOCK2_CANDIDATE: ns.shipped_all}
    print(f"reading {len(paths)} held-out dumps")
    arms = common_rows({name: ev.read_predictions(path) for name, path in paths.items()})
    rows = next(iter(arms.values())).n_rows
    print(f"common rows across every arm: {rows:,}")

    comparison = {"arms": json.loads(ns.arms_report.read_text())["arms"],
                  "contrasts": contrast_block(
                      {k: arms[k] for k in ("floor", "redundant", "diverse")},
                      columns, boots=ns.boots, seed=ns.seed)}
    ns.out_comparison.parent.mkdir(parents=True, exist_ok=True)
    ns.out_comparison.write_text(json.dumps(comparison, indent=1) + "\n")

    matched = source_block({k: arms[k] for k in (BLOCK2_BASELINE, BLOCK2_CANDIDATE)},
                           columns, boots=ns.boots, seed=ns.seed)
    ns.out_source_matched.parent.mkdir(parents=True, exist_ok=True)
    ns.out_source_matched.write_text(json.dumps(matched, indent=1) + "\n")

    for name, entry in comparison["contrasts"].items():
        print(f"\n{name}  (positive = the candidate is better)")
        for column in columns:
            if column in entry:
                low, high = entry[column]["ci"]
                print(f"  {column:9s}{entry[column]['rmse_baseline']:9.4f}"
                      f"{entry[column]['rmse_candidate']:10.4f}"
                      f"{entry[column]['delta_rmse']:+9.4f}  [{low:+7.4f}, {high:+7.4f}]")
    print(f"\nwrote {ns.out_comparison}\nwrote {ns.out_source_matched}")
    return comparison, matched


if __name__ == "__main__":
    main()
