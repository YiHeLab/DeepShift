#!/usr/bin/env python3
"""What the p100 fence removes: every arm's backbone error with it and without it.

Methods says that not excluding unphysical values raises backbone MAE by some per cent
and multiplies backbone RMSE by some factor, for DeepShift and every other evaluated
predictor alike. The unphysical-value exclusion is the p100 rule: the fitted accepted
span per residue type and nucleus (with the arginine N-epsilon interval, which touches no
backbone cell). This scorer produces the two ranges from this run.

**Fenced** is the benchmark as the tables score it: the frozen master cell set (the
fourteen anticorrelated channels already out), each arm on the master cells it emitted,
PLM-CS withheld on the proteins it trained on, every retrieval arm verified at its own
exclusion level. The run's composite fence is decomposed by the package's own rules
(``deepshift.quality_control``): the p100 flags and the structural flags are derived
again from the cache and the p100 authority, and their union must equal the run's flag
file, or nothing is scored.

**Unfenced** lifts the p100 flags alone. A backbone cell comes back when p100 flagged it,
the structural rule did not, the cache observed it, and it is not in an anticorrelated
channel; it joins the cohort under the master rule itself -- on a structure of the
master set, emitted by every arm that joined that nucleus's master set. Nothing else
changes: same structures, same predictions, same exclusions, the cells differing only by
what p100 removed.

Per arm, both conditions pool the six backbone nuclei in raw ppm (prediction minus the
re-referenced observed value), which is what "backbone MAE" in the sentence means; the
per-nucleus effects are reported beside them. The effect of NOT applying the fence is

    MAE change, %  = 100 * (MAE_unfenced / MAE_fenced - 1)
    RMSE ratio     = RMSE_unfenced / RMSE_fenced

and the sentence's ranges are the minimum and maximum over the sixteen predictors the
manuscript evaluates (every arm of ``benchmarks.config.BENCHMARK_ARMS`` but the
``*_selfexcl`` arms, which only the exclusion ladder prints, plus DeepShift). The fenced
side is checked, arm by arm and nucleus by nucleus, against the benchmark's own scoring
report when ``--check`` names it.

    python -m benchmarks.score_fence_effect \\
        --master-cells <master>/master_cells_pf14.npz \\
        --p100-cutoffs $SI_P100_CUTOFFS \\
        --check <scored>/score_tier2_master.json \\
        --out <scored>/fence_effect.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from deepshift.quality_control import (
    derive_p100_flags,
    derive_structural_flags,
    union_flags,
)

from . import cells as common
from . import config
from . import master_cells as master
from . import score_by_source_tier as sbt
from .score_by_source import BACKBONE, drop_trained_on, tool_cells
from .score_validation import _resolved_cells, fold6_rows

DEEPSHIFT = config.DEEPSHIFT_ARM

#: The predictors the sentence speaks for: the sixteen main.tex lists as evaluated.
SENTENCE_ARMS = (*(a for a in config.BENCHMARK_ARMS if not a.endswith("_selfexcl")),
                 DEEPSHIFT)


def decompose_fence(p100_cutoffs: Path) -> tuple[dict, dict]:
    """``(p100 flags, structural flags)``, refused unless their union is the run's fence."""
    cache_root = Path(common.CACHE).parent
    p100, p100_report = derive_p100_flags(cache_root, p100_cutoffs)
    structural, structural_report = derive_structural_flags(cache_root)
    run_fence = json.loads(Path(common.FLAGS).read_text())
    if union_flags(p100, structural) != union_flags(run_fence):
        raise SystemExit(f"p100 plus structural flags derived from {cache_root} are not "
                         f"the run's fence {common.FLAGS}; refusing to attribute its cells")
    return ({"p100": p100, "structural": structural},
            {"p100": p100_report, "structural": structural_report})


def unfenced_rows(fenced_rows, flags, cols):
    """The fenced rows with the p100-only backbone cells put back.

    Returns ``(rows, n_lifted)``. A cell is lifted where p100 flagged its label residue
    and column, the structural rule did not, and the cache observed it; its target is
    the cache's re-referenced shift, which is checked equal to the packet's target on
    every cell the packet kept.
    """
    structural = Path(common.CACHE) / "structural"
    shifts = np.load(structural / "shifts.npy", mmap_mode="r")
    observed = np.load(structural / "shift_mask.npy", mmap_mode="r")
    _go, index_rows = common.build_row_index()
    row_of = {(common.structure_id(r), r["struct_resnum"]): r for r in index_rows}
    backbone = {j for j, c in enumerate(cols) if c[:-6].upper() in BACKBONE}
    p100, structural_flags = flags["p100"], flags["structural"]

    out, lifted = {}, 0
    for sid, rows in fenced_rows.items():
        out[sid] = []
        for row in rows:
            source = row_of[(sid, row["struct_resnum"])]
            g, lrid = source["g"], source["label_residue_id"]
            kept = row["obs_cols"]
            if kept and not np.allclose(np.asarray(shifts[g])[kept],
                                        np.asarray(row["true_raw"])[kept], atol=1e-4):
                raise SystemExit(f"{sid} {row['struct_resnum']}: the packet's targets are "
                                 "not the cache's re-referenced shifts")
            back = sorted(
                j for j in backbone
                if cols[j] in p100.get(lrid, ()) and cols[j] not in structural_flags.get(lrid, ())
                and observed[g, j] and j not in kept)
            if not back:
                out[sid].append(row)
                continue
            true_raw = np.array(row["true_raw"], dtype=np.float64)
            true_raw[back] = np.asarray(shifts[g], dtype=np.float64)[back]
            out[sid].append({**row, "obs_cols": sorted([*kept, *back]), "true_raw": true_raw})
            lifted += len(back)
    return out, lifted


def arm_cells(rows_by_sid, cols, atom_of_col, partner, tasks, exc):
    """``{arm: {sid: {(resnum, col): (prediction, target)}}}`` for every arm and DeepShift."""
    arms = {}
    for arm in config.BENCHMARK_ARMS:
        found, _bad = tool_cells(arm, list(tasks), rows_by_sid, cols, atom_of_col, partner,
                                 tasks, exc,
                                 enforce_selfexcl=config.arm_exclusion_tier(arm) is not None,
                                 verify=sbt.verify_at_own_level)
        arms[arm], _withheld = drop_trained_on(arm, found, tasks)
    arms[DEEPSHIFT] = {sid: {(row["struct_resnum"], j): pair
                             for row in rows
                             for j, pair in _resolved_cells(row["our_pred"], row["true_raw"],
                                                            row["obs_cols"], row["aa"],
                                                            partner, row["amb"]).items()}
                       for sid, rows in rows_by_sid.items()}
    return arms


def score(arms, cohort, atom_of_col):
    """Per arm: per backbone nucleus and pooled, the cells it emitted in ``cohort``."""
    out = {}
    for arm, found in arms.items():
        err = {a: [] for a in BACKBONE}
        for sid, keys in cohort.items():
            mine = found.get(sid, {})
            for key in keys:
                if atom_of_col[key[1]] in err and key in mine:
                    prediction, target = mine[key]
                    err[atom_of_col[key[1]]].append(prediction - target)
        node = {}
        for a in [*BACKBONE, "backbone"]:
            e = np.asarray([x for v in err.values() for x in v] if a == "backbone"
                           else err[a], float)
            if e.size:
                node[a] = {"n_cells": int(e.size), "mae": float(np.abs(e).mean()),
                           "rmse": float(np.sqrt((e ** 2).mean()))}
        out[arm] = node
    return out


def effect(fenced, unfenced):
    out = {}
    for arm, nodes in fenced.items():
        out[arm] = {}
        for a, f in nodes.items():
            u = unfenced[arm][a]
            out[arm][a] = {"cells_fenced": f["n_cells"], "cells_unfenced": u["n_cells"],
                           "mae_fenced": f["mae"], "mae_unfenced": u["mae"],
                           "mae_change_pct": 100.0 * (u["mae"] / f["mae"] - 1.0),
                           "rmse_fenced": f["rmse"], "rmse_unfenced": u["rmse"],
                           "rmse_ratio": u["rmse"] / f["rmse"]}
    return out


def check_against_report(fenced, report_path):
    """The fenced side must be the benchmark table's own numbers, cell count and all."""
    report = json.loads(Path(report_path).read_text())["by_source"]["all"]["nuclei"]
    checked = 0
    for arm, nodes in fenced.items():
        for a in BACKBONE:
            theirs = report.get(a, {}).get(arm)
            if theirs is None:
                continue
            mine = nodes[a]
            if (mine["n_cells"] != theirs["n_cells"]
                    or not np.isclose(mine["rmse"], theirs["rmse"], rtol=0, atol=1e-9)
                    or not np.isclose(mine["mae"], theirs["mae"], rtol=0, atol=1e-9)):
                raise SystemExit(f"fenced {arm} {a}: {mine} is not the scoring report's "
                                 f"n={theirs['n_cells']} rmse={theirs['rmse']} "
                                 f"mae={theirs['mae']}")
            checked += 1
    return checked


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-cells", type=Path, required=True)
    ap.add_argument("--p100-cutoffs", type=Path, required=True,
                    help="the run's p100 authority (SI_P100_CUTOFFS)")
    ap.add_argument("--check", type=Path, default=None,
                    help="the benchmark's scoring report the fenced side must reproduce")
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)

    with open(common.OUT_ROOT / "fold6_tasks.tsv") as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}
    exc = common.SelfExcluder()

    flags, flag_reports = decompose_fence(ns.p100_cutoffs)
    rows, cols, atom_of_col, partner = fold6_rows(list(tasks))
    rows_unfenced, n_lifted = unfenced_rows(rows, flags, cols)

    frame = master.load_master(ns.master_cells)
    fenced_cohort = master.master_index(frame, cols)
    arms_f = arm_cells(rows, cols, atom_of_col, partner, tasks, exc)
    arms_u = arm_cells(rows_unfenced, cols, atom_of_col, partner, tasks, exc)

    # The lifted cells enter under the master rule: a master structure, every arm that
    # joined the nucleus emitted it, and no anticorrelated channel.
    joined = {column: node["joined"] for column, node in
              master.read_report(ns.master_cells)["by_nucleus"].items()}
    channels = {(ls, master._ATOM_COLUMN[nuc]) for ls, nuc in
                master.anticorrelated_channels()[0]}
    label_set = {sid: t["label_set_id"] for sid, t in tasks.items()}
    unfenced_cohort, added, refused = {}, 0, 0
    for sid, keys in fenced_cohort.items():
        cohort = set(keys)
        for before, row in zip(rows[sid], rows_unfenced[sid], strict=True):
            for j in sorted(set(row["obs_cols"]) - set(before["obs_cols"])):
                key, column = (row["struct_resnum"], j), cols[j]
                if (label_set[sid], column) in channels:
                    refused += 1
                    continue
                if all(key in arms_u[arm].get(sid, {}) for arm in joined[column]):
                    cohort.add(key)
                    added += 1
                else:
                    refused += 1
        unfenced_cohort[sid] = cohort

    fenced = score(arms_f, fenced_cohort, atom_of_col)
    unfenced = score(arms_u, unfenced_cohort, atom_of_col)
    checked = check_against_report(fenced, ns.check) if ns.check else 0
    per_arm = effect(fenced, unfenced)

    pooled = {arm: per_arm[arm]["backbone"] for arm in SENTENCE_ARMS}
    per_nucleus = [per_arm[arm][a] for arm in SENTENCE_ARMS for a in BACKBONE
                   if a in per_arm[arm]]

    def span(values):
        return [min(values), max(values)]

    summary = {
        "definition": "the effect of NOT applying the p100 fence: MAE change % = "
                      "100*(MAE_unfenced/MAE_fenced - 1), RMSE ratio = "
                      "RMSE_unfenced/RMSE_fenced; each arm on the cells it emitted",
        "arms": list(SENTENCE_ARMS),
        "backbone_pooled": {
            "mae_change_pct_range": span([e["mae_change_pct"] for e in pooled.values()]),
            "rmse_ratio_range": span([e["rmse_ratio"] for e in pooled.values()]),
        },
        "per_nucleus": {
            "mae_change_pct_range": span([e["mae_change_pct"] for e in per_nucleus]),
            "rmse_ratio_range": span([e["rmse_ratio"] for e in per_nucleus]),
        },
    }
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps({
        "what": "backbone MAE and RMSE of every fold-6 arm with the p100 fence and without "
                "it, same structures, predictions and exclusions",
        "fence_lifted": "p100: the fitted accepted span per residue type and nucleus "
                        "(Methods, 'Masking and outlier control'); the structural Cys/Pro "
                        "flags, the perturber mask and the anticorrelated channels stay",
        "cohort": {
            "fenced": "the frozen master cell set",
            "unfenced": "the master set plus the p100-only backbone cells of its "
                        "structures that every arm joining the nucleus emitted",
            "fenced_backbone_cells": sum(atom_of_col[j] in BACKBONE
                                         for keys in fenced_cohort.values()
                                         for _resnum, j in keys),
            "unfenced_backbone_cells": sum(atom_of_col[j] in BACKBONE
                                           for keys in unfenced_cohort.values()
                                           for _resnum, j in keys),
            "p100_cells_lifted_on_deepshift_rows": n_lifted,
            "lifted_cells_admitted": added,
            "lifted_cells_not_admitted": refused,
            "lifted_cells_on_structures_outside_the_master_set": n_lifted - added - refused,
        },
        "fence_decomposition": {"union_equals_run_fence": True,
                                "p100_flags": flag_reports["p100"]["unique_flags"],
                                "structural_flags": flag_reports["structural"]["cells_flagged"]},
        "fenced_side_checked_against_report": {"report": str(ns.check) if ns.check else None,
                                               "arm_nuclei_matched": checked},
        "summary": summary,
        "per_arm": per_arm,
    }, indent=1, default=str) + "\n")
    print(f"wrote {ns.out}")
    print(f"lifted {n_lifted:,} p100 cells on DeepShift's rows; {added:,} enter the cohort")
    for arm in SENTENCE_ARMS:
        e = per_arm[arm]["backbone"]
        print(f"{arm:24s} cells {e['cells_fenced']:,} -> {e['cells_unfenced']:,}  "
              f"MAE {e['mae_fenced']:.4f} -> {e['mae_unfenced']:.4f} "
              f"({e['mae_change_pct']:+.2f}%)  RMSE {e['rmse_fenced']:.4f} -> "
              f"{e['rmse_unfenced']:.4f} (x{e['rmse_ratio']:.3f})")
    b = summary["backbone_pooled"]
    print("backbone pooled: MAE +{:.1f}-{:.1f}%, RMSE x{:.2f}-{:.2f}".format(
        *b["mae_change_pct_range"], *b["rmse_ratio_range"]))
    p = summary["per_nucleus"]
    print("per nucleus:     MAE {:+.1f} to {:+.1f}%, RMSE x{:.2f}-{:.2f}".format(
        *p["mae_change_pct_range"], *p["rmse_ratio_range"]))


if __name__ == "__main__":
    main()
