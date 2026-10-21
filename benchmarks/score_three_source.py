#!/usr/bin/env python3
"""Each predictor's backbone accuracy from three kinds of coordinates of one deposition.

On the master cells, the label sets that have a crystal, an AlphaFold and a solution-NMR
structure instance. Within them, a site is one (label set, label residue, nucleus) that
all three sources realize, so a predictor's three arms are scored against the same
deposited values and only the coordinates differ. Per predictor and nucleus: the RMSE
from each source over the cells it produced at the sites it produced on all three
sources, which source is least accurate, and the paired differences with a cluster
bootstrap over label sets (one resample drives all three sources).

    python -m benchmarks.score_three_source --tier 2 \\
        --master-cells <bench>/master_cells/master_cells_pf14.npz \\
        --tools ucbshift_full_sameprot ucbshift_x ... --out <dir>/three_source.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from deepshift.statistics import (
    DEFAULT_SEED,
    BootstrapSpec,
    bootstrap_indices,
    error_replicates,
    paired_verdict,
    percentile_interval,
    unit_codes,
)

from . import cells as common
from . import score_by_source_tier as sbt
from .score_by_source import BACKBONE, build_cohort

SOURCES = ("xtal", "af", "nmr")
REFERENCE = "deepshift"


def site_keys(cohort) -> np.ndarray:
    """(label set, label residue, nucleus) of every cohort cell, as one string."""
    _go, rows = common.build_row_index()
    label_residue = {(common.structure_id(r), r["struct_resnum"]): r["label_residue_id"]
                     for r in rows}
    return np.array([f"{c[1]}|{label_residue[(c[0], c[3][0])]}|{c[2]}" for c in cohort.cells])


def score_arm(arm, cohort, sites, label_sets, triple, spec) -> dict:
    """One predictor: per nucleus, each source's error at the sites it covers on all three."""
    finite = np.isfinite(cohort.resid[arm])
    out = {}
    for stream, nucleus in enumerate(BACKBONE):
        base = (cohort.atom == nucleus) & np.isin(label_sets, triple) & finite
        covered = [set(sites[base & (cohort.src == s)]) for s in SOURCES]
        common_sites = set.intersection(*covered)
        keep = base & np.isin(sites, list(common_sites))
        if not keep.any():
            continue
        codes, n_units = unit_codes(label_sets[keep])
        draws = bootstrap_indices(n_units, spec, stream=stream)
        per_source, replicates = {}, {}
        for s in SOURCES:
            mine = cohort.src[keep] == s
            rmse, rmse_draws, mae, _ = error_replicates(
                cohort.resid[arm][keep][mine], codes[mine], n_units, draws)
            per_source[s] = {"n_cells": int(mine.sum()), "rmse": rmse,
                             "rmse_ci95": percentile_interval(rmse_draws), "mae": mae}
            replicates[s] = rmse_draws
        paired = {}
        for first, second in combinations(SOURCES, 2):
            delta = replicates[first] - replicates[second]
            interval = percentile_interval(delta)
            paired[f"{first}_minus_{second}"] = {
                "delta_rmse": per_source[first]["rmse"] - per_source[second]["rmse"],
                "delta_rmse_ci95": interval,
                "verdict": {"baseline": first, "candidate": second}.get(
                    paired_verdict(interval), "tie")}
        worst = max(SOURCES, key=lambda s: per_source[s]["rmse"])
        out[nucleus] = {"n_sites": len(common_sites), "n_label_sets": n_units,
                        "per_source": per_source, "least_accurate_source": worst,
                        "paired": paired}
    return {"nuclei": out,
            "n_nuclei_scored": len(out),
            "n_nuclei_nmr_least_accurate": sum(v["least_accurate_source"] == "nmr"
                                               for v in out.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, choices=sorted(sbt.LEVEL_BY_TIER), required=True)
    ap.add_argument("--tools", nargs="+", required=True)
    ap.add_argument("--master-cells", dest="master_cells", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)
    if ns.out.exists():
        raise SystemExit(f"refusing to overwrite {ns.out}")

    with open(common.OUT_ROOT / "fold6_tasks.tsv") as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}
    sbt.check_arm_policy(ns.tools, [], ns.tier, tasks)
    cohort = build_cohort(
        argparse.Namespace(tools=ns.tools, leaky=[], master_cells=ns.master_cells),
        verify=lambda tool, sid, task, exc: sbt.verify_tiered(tool, sid, task, exc, ns.tier))
    sites = site_keys(cohort)
    label_sets = np.array([c[1] for c in cohort.cells])
    present = {s: set(label_sets[cohort.src == s]) for s in SOURCES}
    triple = sorted(set.intersection(*present.values()))
    in_triple = np.isin(label_sets, triple)
    bmrb = {tasks[s]["label_set_id"]: tasks[s]["bmrb_id"] for s in tasks}
    spec = BootstrapSpec(unit="label_set", resamples=ns.boot, seed=ns.seed)

    arms = {arm: score_arm(arm, cohort, sites, label_sets, triple, spec)
            for arm in [REFERENCE, *ns.tools]}
    report = {
        "what": "each predictor's backbone RMSE from a crystal, an AlphaFold and a "
                "solution-NMR structure of the same deposition, at the sites all three "
                "realize",
        "cohort": {"master_cells": ns.master_cells,
                   "tier": ns.tier,
                   "n_label_sets_with_all_three_sources": len(triple),
                   "n_bmrb_depositions_with_all_three_sources":
                       len({bmrb[ls] for ls in triple}),
                   "n_master_cells_on_those_label_sets": int(in_triple.sum()),
                   "n_structure_instances_on_those_label_sets":
                       len(set(cohort.sid_arr[in_triple]))},
        "site": "(label set, label residue, nucleus) realized by all three sources; a "
                "source with several structure instances contributes each of their cells",
        "per_arm_sites": "an arm is scored at the sites it produced a value for on all three "
                         "sources, so every arm's three RMSEs share their sites",
        "residual": "prediction minus the resolved observed value, ppm; nothing is centred",
        "bootstrap": {"unit": spec.unit, "resamples": spec.resamples, "seed": spec.seed,
                      "paired": "one resample of label sets drives all three sources"},
        "arms": arms,
        "summary": {arm: {"n_nuclei_scored": a["n_nuclei_scored"],
                          "n_nuclei_nmr_least_accurate": a["n_nuclei_nmr_least_accurate"]}
                    for arm, a in arms.items()},
    }
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(report, indent=1) + "\n")

    print(f"wrote {ns.out}: {len(triple)} label sets with all three sources")
    for arm, a in arms.items():
        print(f"  {arm:24s} NMR least accurate on {a['n_nuclei_nmr_least_accurate']}"
              f"/{a['n_nuclei_scored']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
