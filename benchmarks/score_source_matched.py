#!/usr/bin/env python3
"""Backbone accuracy from three kinds of coordinates, on the sites all three realize.

Cohort: the five cross-validation members' held-out predictions, folds 1-5 pooled, so
every protein is scored by the one member that held its fold out.
``deepshift.evaluate.pool`` concatenates the dumps and refuses shared rows.

Matching: a cell -- one (label residue, nucleus) -- enters only where an AlphaFold model,
a crystal structure and a solution-NMR conformer all make it scorable, which is
``deepshift.cohorts.matched_cells``. The three arms are then scored against the same
deposited value and the coordinates are the only thing that differs between them, where
the marginal per-source comparison is confounded with which proteins are crystallized.

Residual: prediction minus the resolved observed value, in ppm, nothing centred. The
exclusion fence reaches every arm through one application, the effective mask the dumps
were written under.

Statistics: per nucleus (``deepshift.normalization.BACKBONE_SHIFTS``) and per source,
RMSE and MAE with 95 percent percentile intervals from a cluster bootstrap over label
sets, through ``deepshift.statistics``; and for each pair of sources a paired difference
on one shared resample, with the verdict of ``deepshift.statistics.compare`` -- an
interval on the RMSE difference that excludes zero names a source, anything else is a tie.

Usage:
  python -m benchmarks.score_source_matched --out <out>/source_matched.json
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import fields, replace
from itertools import combinations
from pathlib import Path

import numpy as np

from deepshift import evaluate
from deepshift.calibration import MEMBER_FOLDS
from deepshift.cohorts import matched_cells
from deepshift.domain import VALID_SOURCES
from deepshift.normalization import BACKBONE_SHIFTS
from deepshift.quality_control import FINAL_QC_POLICY
from deepshift.statistics import (
    DEFAULT_SEED,
    MIN_SCORABLE_CELLS,
    BootstrapSpec,
    bonferroni,
    bootstrap_indices,
    compare,
    error_replicates,
    percentile_interval,
    unit_codes,
)

from . import config

#: The held-out dump the trainer writes for each cross-validation member.
DUMP_NAME = "final/fold%d.preds.npz"
BOOTSTRAP_UNIT = "label_set"
DEFAULT_RESAMPLES = 2000
#: The family-wise level of the source comparisons' Bonferroni correction.
ALPHA = 0.05

_FOLD = re.compile(r"fold(\d+)")


def default_dumps() -> list[Path]:
    """The five held-out cross-validation dumps inside this run root."""
    run = config.current().run
    return [run / (DUMP_NAME % fold) for fold in MEMBER_FOLDS]


def _fold_in(text) -> int | None:
    found = _FOLD.search(str(text))
    return int(found.group(1)) if found else None


def check_dumps(paths) -> None:
    """The five held-out dumps, each holding the fold its file name says."""
    if len(paths) != len(MEMBER_FOLDS):
        raise SystemExit(f"the matched cohort is the {len(MEMBER_FOLDS)} held-out folds "
                         f"{MEMBER_FOLDS}; {len(paths)} dumps were given")
    for path in paths:
        with np.load(path, allow_pickle=False) as dump:
            held = (_fold_in(dump["checkpoint_name"])
                    if "checkpoint_name" in dump.files else None)
        named = _fold_in(Path(path).name)
        if held is not None and named is not None and held != named:
            raise SystemExit(f"{path} is named for fold {named} and holds fold {held}")


def _arrays(predictions, ndim=None) -> list[str]:
    """The arrays of a population, optionally only those with ``ndim`` axes."""
    return [field.name for field in fields(predictions)
            if isinstance(getattr(predictions, field.name), np.ndarray)
            and (ndim is None or getattr(predictions, field.name).ndim == ndim)]


def keep_columns(predictions, columns):
    """Six nuclei of the sixty-four, taken before five populations are pooled."""
    take = [predictions.shift_cols.index(name) for name in columns]
    return replace(predictions, shift_cols=tuple(columns),
                   **{name: getattr(predictions, name)[:, take]
                      for name in _arrays(predictions, ndim=2)})


def keep_rows(predictions, rows):
    """A population restricted to ``rows``; every array carries one entry per row."""
    return replace(predictions, **{name: getattr(predictions, name)[rows]
                                   for name in _arrays(predictions)})


def read(paths, columns):
    """Pool the held-out folds into one population, carrying each row's label residue."""
    parts, label_residue = [], []
    for path in paths:
        parts.append(keep_columns(evaluate.read_predictions(path), columns))
        label_residue.append(evaluate.read_label_residues(path))
    return evaluate.pool(parts), np.concatenate(label_residue)


def source_arms(predictions, cells, label_residue) -> dict:
    """One population per structure source, over the same label residues in one order.

    Re-keying the rows on the label residue is what lets ``compare`` pair the arms: it
    refuses arms that do not describe the same rows, and its label-agreement check then
    confirms that the three are scored against one deposited value.
    """
    label_residue = np.asarray(label_residue)
    matched = replace(predictions, effective_mask=predictions.effective_mask & cells)
    realized = cells.any(axis=1)
    arms = {}
    for code, name in enumerate(VALID_SOURCES):
        rows = np.flatnonzero(realized & (predictions.source == code))
        rows = rows[np.argsort(label_residue[rows], kind="stable")]
        arms[name] = replace(keep_rows(matched, rows),
                             row_key=label_residue[rows].astype(str))
    residues = [arms[name].row_key for name in VALID_SOURCES]
    if not all(np.array_equal(residues[0], other) for other in residues[1:]):
        raise SystemExit("the three arms do not carry the same label residues")
    return arms


def _error(arm, selected, column, codes, n_units, draws) -> dict:
    """One source's error on one nucleus, with intervals from the shared resample."""
    residual = arm.pred_ppm[selected, column] - arm.resolved_ppm[selected, column]
    rmse, rmse_draws, mae, mae_draws = error_replicates(residual, codes, n_units, draws)
    return {"n_cells": int(residual.size),
            "rmse": rmse, "rmse_ci95": percentile_interval(rmse_draws),
            "mae": mae, "mae_ci95": percentile_interval(mae_draws)}


def paired_verdict(first, second, entry) -> dict:
    """One pair of sources, with ``compare``'s verdict named by source."""
    return {"sources": [first, second], "n_cells": entry["n"],
            "rmse": {first: entry["rmse_baseline"], second: entry["rmse_candidate"]},
            "mae": {first: entry["mae_baseline"], second: entry["mae_candidate"]},
            "delta_rmse": entry["delta_rmse"], "delta_rmse_ci95": entry["ci"],
            "verdict": {"baseline": first,
                        "candidate": second}.get(entry["verdict"], entry["verdict"]),
            "k_tail": entry["k_tail"], "resamples": entry["resamples"]}


def source_family(nuclei) -> dict:
    """Every pair of sources on every nucleus, Bonferroni-corrected over that family
    (``deepshift.statistics.bonferroni``): per pair, how many differences are significant
    and which are not."""
    tests = [(name, pair, entry) for name, node in nuclei.items()
             for pair, entry in node["paired"].items()]
    out = {"n_tests": len(tests), "alpha": ALPHA}
    for pair in sorted({pair for _, pair, _ in tests}):
        verdicts = {name: bonferroni(entry["k_tail"], entry["resamples"], len(tests), ALPHA)
                    for name, other, entry in tests if other == pair}
        out[pair] = {
            "n_significant": sum(test["significant"] for test in verdicts.values()),
            "n_significant_plus_one": sum(test["significant_plus_one"]
                                          for test in verdicts.values()),
            "not_significant": sorted(name for name, test in verdicts.items()
                                      if not test["significant"])}
    return out


def score(arms, columns, *, bootstrap) -> tuple[dict, dict]:
    """Per-nucleus per-source error and the paired verdicts, on the matched cells."""
    head = arms[VALID_SOURCES[0]]
    nuclei, thin = {}, {}
    for name in columns:
        column = head.shift_cols.index(name)
        selected = head.effective_mask[:, column]
        n_cells = int(selected.sum())
        if n_cells < MIN_SCORABLE_CELLS:
            thin[name] = {"n_matched_cells": n_cells, "floor": MIN_SCORABLE_CELLS}
            continue
        codes, n_units = unit_codes(head.label_set[selected])
        draws = bootstrap_indices(n_units, bootstrap, stream=column)
        nuclei[name] = {
            "per_source": {source: _error(arm, selected, column, codes, n_units, draws)
                           for source, arm in arms.items()},
            "n_matched_sites": int(np.unique(head.row_key[selected]).size),
            "n_label_sets": n_units,
            "paired": {},
        }
    for first, second in combinations(VALID_SOURCES, 2):
        differences = compare(arms[first], arms[second], bootstrap=bootstrap,
                              columns=list(columns))
        for name, entry in differences.items():
            nuclei[name]["paired"][f"{first}_vs_{second}"] = paired_verdict(first, second, entry)
    return nuclei, thin


def report(predictions, label_residue, *, columns, bootstrap) -> dict:
    """The three-source matched report, in the schema ``fig_source_matched.py`` draws."""
    cells, _ = matched_cells(predictions, label_residue)
    arms = source_arms(predictions, cells, label_residue)
    nuclei, thin = score(arms, columns, bootstrap=bootstrap)
    head = arms[VALID_SOURCES[0]]
    return {
        "cohort": f"the held-out predictions of the {len(MEMBER_FOLDS)} cross-validation "
                  f"members, folds {list(MEMBER_FOLDS)} pooled, every protein scored by "
                  f"the member that held its fold out",
        "matching": "a (label residue, nucleus) cell scorable from an AlphaFold model, a "
                    "crystal structure and a solution-NMR conformer alike, so the three "
                    "arms are scored against one deposited value and only the coordinates "
                    "differ; each (label residue, source) is realized once",
        "residual": "prediction minus the resolved observed value, in ppm; nothing is "
                    "centred",
        "exclusions_on_observed": f"the {FINAL_QC_POLICY} fence and the prochiral "
                                  f"resolution the dumps' effective mask already carries, "
                                  f"applied identically to the three arms; this report "
                                  f"adds no exclusion of its own",
        "bootstrap": {"unit": bootstrap.unit,
                      "n_units": int(np.unique(head.label_set).size),
                      "resamples": bootstrap.resamples, "seed": bootstrap.seed},
        "paired": "delta_rmse is the first source's RMSE minus the second's, in ppm; the "
                  "verdict names the source with the lower error when the interval on the "
                  "difference excludes zero, and is a tie when it does not",
        "n_matched_sites": int(head.n_rows),
        "nuclei": nuclei,
        "bonferroni": source_family(nuclei),
        "not_scored": thin,
    }


def print_report(result) -> None:
    print(f"\n{'nucleus':10s}{'sites':>9s}{'label sets':>12s}"
          + "".join(f"{source + ' RMSE':>14s}" for source in VALID_SOURCES))
    for name, node in result["nuclei"].items():
        print(f"{name:10s}{node['n_matched_sites']:9,}{node['n_label_sets']:12,}"
              + "".join(f"{node['per_source'][source]['rmse']:14.4f}"
                        for source in VALID_SOURCES))
    print(f"\n{'nucleus':10s}{'pair':16s}{'delta RMSE':>11s}{'95% CI':>22s}  verdict")
    for name, node in result["nuclei"].items():
        for pair, entry in node["paired"].items():
            low, high = entry["delta_rmse_ci95"]
            print(f"{name:10s}{pair:16s}{entry['delta_rmse']:+11.4f}"
                  f"{f'[{low:+.4f}, {high:+.4f}]':>22s}  {entry['verdict']}")
    for name, node in result["not_scored"].items():
        print(f"not scored (under {node['floor']} matched cells): {name} "
              f"{node['n_matched_cells']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", nargs="+", metavar="NPZ",
                        help="the five held-out fold dumps; the run root's own by default")
    parser.add_argument("--out", required=True, help="write the report as JSON")
    parser.add_argument("--boots", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    paths = [Path(path) for path in args.preds] if args.preds else default_dumps()
    check_dumps(paths)
    bootstrap = BootstrapSpec(unit=BOOTSTRAP_UNIT, resamples=args.boots, seed=args.seed)
    predictions, label_residue = read(paths, list(BACKBONE_SHIFTS))
    print(f"{predictions.n_rows:,} pooled rows x {len(predictions.shift_cols)} nuclei, "
          f"{len(np.unique(predictions.label_set)):,} label sets")

    result = report(predictions, label_residue, columns=list(BACKBONE_SHIFTS),
                    bootstrap=bootstrap)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print_report(result)
    print(f"\nwrote {args.out}")
    return result


if __name__ == "__main__":
    main()
