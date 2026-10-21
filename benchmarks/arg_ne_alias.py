#!/usr/bin/env python3
"""The arginine N-epsilon sweep-width alias fence, measured on the fold-6 benchmark cells.

A subset of BMRB depositions record the arginine guanidinium N-epsilon resonance folded
by one 15N spectral width, so the deposited value sits about 30 ppm above the true one.
The fence is model-free: an observed arginine ``ne_shift`` outside mean +- 3.5 sd of the
BMRB population (``deepshift.quality_control.ARG_NE_OBSERVED_RANGE_PPM``) leaves the
benchmark for every arm. It reads the observed value alone.

This script counts what the fence removes among the cells the benchmark scorers see
before it: the fold-6 rows (``fold6_rows``, as ``score_fence_effect`` reads them) with
the cells the composite fence removed put back from the cache, as its ``unfenced_rows``
does for the backbone.

``at one sweep width above`` is an excluded value 25 to 40 ppm above the median of the
in-band values: 15N spectral widths of triple-resonance practice are 30-40 ppm and the
recorded displacement on this corpus is 29-31 ppm.

The band is mean +- 3.5 sd of BMRB's filtered statistics for arginine N-epsilon, as the code
carries them (``deepshift.data.norm.BMRB_FILT``); ``rule.basis`` records them and checks
that they give the band. The later BMRB statistics row shipped in
``bmrb_arg_ne_statistics.csv`` is recomputed as a cross-check.

    python -m benchmarks.arg_ne_alias --out figures/reference/ne_alias_fence.json

Everything but ``rule`` is recomputed from this run: ``benchmark_cells``,
``fold6_cache_labels`` (the label census the manuscript quotes: every observed Arg
N-epsilon of the fold-6 cache), ``training_cache_labels`` (the same census per fold),
DeepShift's ``per_arm`` entry and ``rule.basis``.
``n_excluded_above_band`` is the count above the upper edge, which is what the manuscript's
"above the band" is.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from deepshift.chemistry import STANDARD_RESIDUES
from deepshift.data.norm import BMRB_FILT
from deepshift.quality_control import ARG_NE_OBSERVED_RANGE_PPM, derive_structural_flags

from . import cells as common
from . import config
from .score_validation import fold6_rows

HERE = Path(__file__).resolve().parent
STATISTICS = HERE.parent / "figures" / "reference" / "bmrb_arg_ne_statistics.csv"
RECORD = HERE.parent / "figures" / "reference" / "ne_alias_fence.json"
SWEEP_RANGE_PPM = (25.0, 40.0)
SD_MULTIPLE = 3.5


def code_basis() -> dict:
    """The band's moments as the code carries them, BMRB's filtered statistics, and the band
    they give; refuses moments that do not give the fence's band."""
    mean, sd = BMRB_FILT["ARG"]["NE"]
    band = [mean - SD_MULTIPLE * sd, mean + SD_MULTIPLE * sd]
    if max(abs(b - f) for b, f in zip(band, ARG_NE_OBSERVED_RANGE_PPM)) > 1e-3:
        raise SystemExit(f"BMRB_FILT ARG NE {mean} +- {SD_MULTIPLE} x {sd} gives {band}, not "
                         f"the fence's {ARG_NE_OBSERVED_RANGE_PPM}")
    return {"bmrb_mean": mean, "bmrb_std": sd,
            "source": "deepshift.data.norm.BMRB_FILT['ARG']['NE'], the BMRB filtered "
                      "chemical-shift statistics the build fitted the band from",
            "band_ppm": [round(b, 4) for b in band]}


def bmrb_basis() -> dict:
    """The mean, sd and count of the shipped BMRB statistics row, and the band they give."""
    with open(STATISTICS) as fh:
        row = next(csv.DictReader(line for line in fh if not line.startswith("#")))
    mean, sd = float(row["avg"]), float(row["std"])
    return {"bmrb_mean": mean, "bmrb_std": sd, "n": int(row["count"]),
            "band_ppm": [round(mean - SD_MULTIPLE * sd, 3),
                                       round(mean + SD_MULTIPLE * sd, 3)]}


def fold6_arg_ne():
    """``(observed, prediction, label_set)`` over the Arg N-epsilon cells before the fence.

    The fold-6 rows carry the run's composite fence, whose p100 half already contains
    this band, so the cells it removed are read back from the cache: observed there, not
    removed by the structural rule (``quality_control.derive_structural_flags``), and
    kept by the rows in every other respect. Cells the rows kept must equal the cache's
    re-referenced values, or nothing is counted.
    """
    with open(common.OUT_ROOT / "fold6_tasks.tsv") as fh:
        sids = [t["structure_id"] for t in csv.DictReader(fh, delimiter="\t")]
    rows, cols, _atoms, _partner = fold6_rows(sids)
    column, arginine = cols.index("ne_shift"), STANDARD_RESIDUES.index("ARG")
    cache = Path(common.CACHE) / "structural"
    shifts = np.load(cache / "shifts.npy", mmap_mode="r")
    observed_in_cache = np.load(cache / "shift_mask.npy", mmap_mode="r")
    structural = derive_structural_flags(Path(common.CACHE).parent)[0]
    index = {(common.structure_id(r), r["struct_resnum"]): r for r in common.build_row_index()[1]}
    observed, predicted, label_set = [], [], []
    for sid in sids:
        for row in rows[sid]:
            if row["aa"] != arginine:
                continue
            source = index[(sid, row["struct_resnum"])]
            g, label_residue = source["g"], source["label_residue_id"]
            if not observed_in_cache[g, column] or "ne_shift" in structural.get(label_residue, ()):
                continue
            value = float(shifts[g, column])
            if column in row["obs_cols"] and abs(value - row["true_raw"][column]) > 1e-4:
                raise SystemExit(f"{sid} {row['struct_resnum']}: the rows' Arg NE target is "
                                 "not the cache's re-referenced shift")
            observed.append(value)
            predicted.append(float(row["our_pred"][column]))
            label_set.append(row["label_set_id"])
    return np.array(observed), np.array(predicted), np.array(label_set)


def cache_arg_ne(cache=None):
    """Every observed Arg N-epsilon label of one fold's cache (the fold-6 cache by
    default), as the manuscript counts them.

    This is the label census (one cell per cache row), which includes rows the prediction
    dump does not carry; the benchmark cells above are the subset DeepShift predicted.
    """
    cache = Path(cache or common.CACHE)
    column = json.loads((cache / "config.json").read_text())["shift_cols"].index("ne_shift")
    shifts = np.load(cache / "structural" / "shifts.npy", mmap_mode="r")
    observed = np.load(cache / "structural" / "shift_mask.npy", mmap_mode="r")
    residue = json.loads((cache / "structural" / "id_maps.json").read_text())["comp_id"]
    rows = [g for g in np.flatnonzero(observed[:, column]) if residue.get(str(g)) == "ARG"]
    return np.array([shifts[g, column] for g in rows], dtype=float)


def band_report(observed, low, high) -> dict:
    inside = (observed >= low) & (observed <= high)
    mode = float(np.median(observed[inside]))
    displacement = observed[~inside] - mode
    above = (displacement >= SWEEP_RANGE_PPM[0]) & (displacement <= SWEEP_RANGE_PPM[1])
    return {
        "n": int(observed.size), "n_in_band": int(inside.sum()),
        "n_excluded": int((~inside).sum()),
        "pct_excluded": round(100 * float((~inside).mean()), 3),
        "in_band_median_ppm": round(mode, 3),
        "excluded_displacement_from_in_band_median": {
            "median": round(float(np.median(displacement)), 3),
            "p25": round(float(np.percentile(displacement, 25)), 3),
            "p75": round(float(np.percentile(displacement, 75)), 3)},
        "n_excluded_above_band": int((observed > high).sum()),
        "n_excluded_at_one_sweep_width_above": int(above.sum()),
        "n_excluded_other": int((~above).sum()),
        "excluded_other_values": sorted(round(float(v), 3)
                                        for v in observed[~inside][~above])[:40],
    }


def label_set_character(observed, label_set, low, high) -> dict:
    """Aliasing is an instrument setting, so a deposition is all in band or all out."""
    counts = {"clean": 0, "wholly_out_of_band": 0, "mixed": 0}
    for key in np.unique(label_set):
        out = (observed[label_set == key] < low) | (observed[label_set == key] > high)
        counts["wholly_out_of_band" if out.all() else "clean" if not out.any() else "mixed"] += 1
    return {**counts, "note": "an instrument setting should be all-or-nothing within a deposition"}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", type=Path, default=RECORD, help="the record to carry over from")
    ap.add_argument("--out", type=Path, default=RECORD)
    ns = ap.parse_args(argv)
    low, high = ARG_NE_OBSERVED_RANGE_PPM
    record = json.loads(ns.record.read_text())
    observed, predicted, label_set = fold6_arg_ne()

    record["rule"]["basis"] = {**code_basis(), "cross_check": {
        "source": "figures/reference/bmrb_arg_ne_statistics.csv (BMRB filtered statistics, 2026-08-05)",
        **bmrb_basis()}}
    report = band_report(observed, low, high)
    report["label_sets"] = label_set_character(observed, label_set, low, high)
    record["benchmark_cells"] = report
    record["fold6_cache_labels"] = band_report(cache_arg_ne(), low, high)
    folds = Path(common.CACHE).parent
    record["training_cache_labels"] = {
        f"fold_{k}": band_report(cache_arg_ne(folds / f"fold_{k}"), low, high)
        for k in range(7) if (folds / f"fold_{k}" / "config.json").is_file()}
    # A test once run on other new nuclei and never applied, and arms that predict no
    # N-epsilon: neither is read, and neither was measured on this run.
    record.pop("other_new_nuclei_same_test_NOT_APPLIED", None)
    record["per_arm_ne_uncentred"] = {}

    residual, kept = predicted - observed, (observed >= low) & (observed <= high)
    record["per_arm_ne_uncentred"][config.DEEPSHIFT_ARM] = {
        "n_before": int(residual.size),
        "rmse_before": round(float(np.sqrt((residual ** 2).mean())), 4),
        "mae_before": round(float(np.abs(residual).mean()), 4),
        "n_after": int(kept.sum()),
        "rmse_after": round(float(np.sqrt((residual[kept] ** 2).mean())), 4),
        "mae_after": round(float(np.abs(residual[kept]).mean()), 4)}
    ns.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"basis": record["rule"]["basis"], "benchmark_cells": {
        k: v for k, v in report.items() if k != "excluded_other_values"}}, indent=1))
    return record


if __name__ == "__main__":
    main()
