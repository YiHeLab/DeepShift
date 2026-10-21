#!/usr/bin/env python3
"""How much of DeepShift's fold-6 cell set each competitor arm actually produced.

The universe ``U(nuc)`` is what DeepShift can score: the cells of that nucleus the
fold-6 ensemble declares observed -- the run's composite p100 + structural fence was
applied to the observed labels when those predictions were written -- after the
ambiguity drop and prochiral resolution, less the arginine N-epsilon labels outside
``deepshift.quality_control.ARG_NE_OBSERVED_RANGE_PPM``, which are a sweep-width fold in
the deposited label and leave the benchmark for every arm.

Coverage of an arm on a nucleus is the share of ``U(nuc)`` that arm emitted a finite
value for. A structure the arm crashed on, declined, or lost to self-exclusion
contributes no cell, so its cells count as uncovered: coverage is coverage of the
comparison, not of the tool's good intentions. Each arm is verified at the exclusion
level its own name declares (``config.arm_exclusion_tier``), which is the level every
table reports it at.

Writes, into ``--out-dir``:

  coverage_census.json  per arm and nucleus: cells covered, ``|U(nuc)|``, coverage
  arm_coverage.npz      the measurement behind it -- the universe as one record per
                        cell, and one boolean row per arm -- which
                        ``benchmarks.master_cells`` intersects into the frozen set

    python -m benchmarks.coverage_census --out-dir $DEEPSHIFT_BENCH_OUT/master_cells
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from deepshift.chemistry import STANDARD_RESIDUES
from deepshift.quality_control import ARG_NE_OBSERVED_RANGE_PPM

from . import cells as common
from . import config
from . import score_by_source as sbs
from . import score_by_source_tier as sbt
from .master_cells import MASTER_COVERAGE_FLOOR
from .score_validation import _resolved_cells, fold6_rows

ARG_NE_COLUMN = "ne_shift"
ARGININE = "ARG"
COVERAGE_NPZ = "arm_coverage.npz"
CENSUS_JSON = "coverage_census.json"

UNIVERSE_RULE = (
    "the cells of held-out fold 6 DeepShift can score: observed after the run's "
    f"composite p100 + structural fence, ambiguity-dropped and prochiral-resolved, less "
    f"arginine {ARG_NE_COLUMN} labels outside {list(ARG_NE_OBSERVED_RANGE_PPM)} ppm")


def universe(sids, rows_by_sid, cols, partner):
    """-> (the master-cell arrays, {(sid, resnum, column): position}, n_arg_ne).

    One record per scorable cell, in task order, so a boolean row per arm indexes it.
    """
    low, high = ARG_NE_OBSERVED_RANGE_PPM
    arginine = STANDARD_RESIDUES.index(ARGININE)
    ne_column = cols.index(ARG_NE_COLUMN) if ARG_NE_COLUMN in cols else None
    atom, sid_of, label_set, residue, index, aliased = [], [], [], [], {}, 0
    for sid in sids:
        for row in rows_by_sid[sid]:
            for j in _resolved_cells(row["our_pred"], row["true_raw"], row["obs_cols"],
                                     row["aa"], partner, row["amb"]):
                if (j == ne_column and row["aa"] == arginine
                        and not low <= row["true_raw"][j] <= high):
                    aliased += 1
                    continue
                index[(sid, row["struct_resnum"], j)] = len(atom)
                atom.append(cols[j])
                sid_of.append(sid)
                label_set.append(row["label_set_id"])
                residue.append(int(row["struct_resnum"]))
    arrays = {"atom": np.asarray(atom), "sid": np.asarray(sid_of),
              "label_set": np.asarray(label_set), "residue": np.asarray(residue, np.int64)}
    return arrays, index, aliased


def arm_coverage(arm, sids, rows_by_sid, cols, atom_of_col, partner, tasks, exc, index):
    """A boolean row over the universe: the cells this arm emitted a finite value for."""
    usable, dropped = sbs.tool_cells(
        arm, sids, rows_by_sid, cols, atom_of_col, partner, tasks, exc,
        enforce_selfexcl=config.arm_exclusion_tier(arm) is not None,
        verify=sbt.verify_at_own_level)
    covered = np.zeros(len(index), dtype=bool)
    for sid, found in usable.items():
        for (resnum, j) in found:
            position = index.get((sid, resnum, j))
            if position is not None:
                covered[position] = True
    return covered, len(usable), len(dropped)


def census(covered, arrays, columns) -> dict:
    """Per arm, per nucleus: cells covered out of the universe, and the coverage."""
    atom = arrays["atom"]
    sizes = {column: int((atom == column).sum()) for column in columns}
    out = {}
    for arm, row in covered.items():
        by_nucleus = {}
        for column, total in sizes.items():
            if not total:
                continue
            n = int(row[atom == column].sum())
            by_nucleus[column] = {"n_cells": n, "n_universe": total,
                                  "coverage": n / total,
                                  "joins": n / total >= MASTER_COVERAGE_FLOOR}
        out[arm] = {"n_cells": int(row.sum()), "by_nucleus": by_nucleus}
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(config.BENCHMARK_ARMS),
                    help="the competitor arms to census (default: benchmarks.config)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ns = ap.parse_args(argv)
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    census_path, coverage_path = ns.out_dir / CENSUS_JSON, ns.out_dir / COVERAGE_NPZ
    for path in (census_path, coverage_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")

    tasks_tsv = common.OUT_ROOT / "fold6_tasks.tsv"
    with open(tasks_tsv) as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}
    sids = list(tasks)
    print(f"fold-6 rows for {len(sids)} structure instances (fenced observed cells)...")
    rows_by_sid, cols, atom_of_col, partner = fold6_rows(sids)
    arrays, index, aliased = universe(sids, rows_by_sid, cols, partner)
    print(f"universe: {len(index):,} cells, {aliased} aliased arginine labels excluded")

    excluder = common.SelfExcluder()
    covered, structures = {}, {}
    for arm in ns.arms:
        row, usable, dropped = arm_coverage(arm, sids, rows_by_sid, cols, atom_of_col,
                                            partner, tasks, excluder, index)
        covered[arm] = row
        structures[arm] = usable
        print(f"  {arm:26s} structures={usable:5d} dropped={dropped:4d} "
              f"cells={int(row.sum()):9,d} ({row.mean():6.2%} of the universe)")
    covered[config.DEEPSHIFT_ARM] = np.ones(len(index), dtype=bool)
    structures[config.DEEPSHIFT_ARM] = len(sids)

    columns = sorted(set(arrays["atom"].astype(str)))
    np.savez_compressed(coverage_path, arms=np.asarray(list(covered)),
                        covered=np.stack(list(covered.values())), **arrays)
    report = {
        "what": "per arm and nucleus, the share of DeepShift's fold-6 scorable cells the "
                "arm emitted a finite value for",
        "universe_rule": UNIVERSE_RULE,
        "coverage_rule": "a structure an arm failed, declined or lost to self-exclusion "
                         "contributes no cell; each arm is verified at the exclusion level "
                         "its own name declares",
        "coverage_floor": MASTER_COVERAGE_FLOOR,
        "exclusion_tier_by_arm": {arm: config.arm_exclusion_tier(arm) for arm in covered},
        "n_structure_instances": len(sids),
        "n_universe_cells": len(index),
        "n_arg_ne_aliased_excluded": aliased,
        "universe_by_nucleus": {c: int((arrays["atom"] == c).sum()) for c in columns},
        "structures_per_arm": structures,
        "arms_scanned": list(covered),
        "arms": census(covered, arrays, columns),
        "coverage_npz": COVERAGE_NPZ,
        "tasks_tsv": str(tasks_tsv),
    }
    census_path.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"wrote {census_path} and {coverage_path}")
    for arm, node in report["arms"].items():
        short = sorted(c for c, v in node["by_nucleus"].items() if not v["joins"])
        if short:
            print(f"  below the floor: {arm:26s} {len(short)} nuclei "
                  f"(worst {min(node['by_nucleus'][c]['coverage'] for c in short):.3f})")
    return report


if __name__ == "__main__":
    main()
