#!/usr/bin/env python3
"""Cysteine C-beta, resolved by disulfide state, on this run's held-out fold 6.

A free thiol sits near 28 ppm and a disulfide near 41, and which state a cysteine is in
is not a sequence property. A predictor can therefore look accurate on cysteine C-beta
by placing each residue in the right state while getting nothing right inside a state,
so every quantity here is computed inside one state.

MODE RULE. Oxidised when the residue's own SG has another SG within
``geometry_context.DISULFIDE_BOND_MAX_ANGSTROM`` in the 8 A cloud the model was given,
reduced otherwise, through ``geometry_context.cysteine_state``. The state is read from
the coordinates, never from the shift; the deposited C-beta value is used only for the
independent check the report carries.

CELLS. The master cell set's ``cb_shift`` cells (``--master-cells``), so this table scores
the cells Table 1 and Figure 2 score. With no master file the producer falls back to its
own arms: the universe is the cysteine C-beta cells the fold-6 ensemble scored, every arm
covering at least ``MIN_COVERAGE`` of it joins, and the cell set is what the joining arms
all emitted. An arm below that coverage is scored on the cell set intersected with its own
cells and carries ``partial_coverage``.

ARMS. ``benchmarks/config.py``'s roster, less the arms this run's coverage census says
return no C-beta; an arm is otherwise dropped here only by its own coverage. The leaky
arms are scored beside the rest and named as leaky in the report.

PER ARM AND MODE. RMSE of the raw residual, the within-mode amplitude (sd of the
predictions over sd of the observations, which pooled over both modes measures nothing
because the ~11 ppm gap dominates the variance), the within-mode correlation, and the
recovered split -- mean predicted disulfide value minus mean predicted free thiol -- against
the same difference in the observed values. Intervals are a cluster bootstrap over label
sets through ``deepshift.statistics``.

    python -m benchmarks.score_cys_modes --out <dir>/si_cys_modes.json \
        --cells-out <dir>/cys_cb_cells.npz [--master-cells <master>.npz]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from deepshift import geometry_context
from deepshift.statistics import (
    DEFAULT_SEED,
    BootstrapSpec,
    bootstrap_indices,
    error_replicates,
    percentile_interval,
    secondary_shift_agreement,
    secondary_shift_replicates,
    unit_codes,
)

from . import cells as common
from . import master_cells
from .config import BENCHMARK_ARMS, arm_exclusion_tier
from .score_by_source import drop_trained_on, tool_cells
from .score_by_source_tier import LEVEL_BY_TIER, verify_tiered
from .score_validation import _resolved_cells, fold6_rows

#: The nucleus this table is about.
NUCLEUS = "cb_shift"
#: Arms scored with no self-exclusion at all, and so leaky, read off the one roster.
LEAKY = tuple(arm for arm in BENCHMARK_ARMS if arm_exclusion_tier(arm) is None)
#: What an arm must cover of the universe to join the fallback cell set.
MIN_COVERAGE = 0.90
#: The panels the table prints, and the panels the report carries beside them.
SOURCES = ("all", "af", "nmr", "xtal")
BOOTSTRAP = BootstrapSpec(unit="label_set", resamples=2000, seed=DEFAULT_SEED)
#: Below this a mode has no reportable statistic.
MIN_MODE_CELLS = 20


def disulfide_states(rows):
    """``(sid, residue number) -> StructuralState`` for every cysteine row of fold 6.

    The cloud is the one the model read: the stored atom types, offsets and coordinates
    of that row, at the row's own atom count.
    """
    atom_type = np.load(common.AD / "atom_type.npy", mmap_mode="r")
    atom_offset = np.load(common.AD / "atom_offset.npy", mmap_mode="r")
    atom_xyz = np.load(common.AD / "atom_xyz.npy", mmap_mode="r")
    atom_count = np.load(common.AD / "atom_count.npy", mmap_mode="r")
    observed = common.load_pred_npz()
    column = observed["cols"].index(NUCLEUS)
    states = {}
    for index, row in enumerate(rows):
        if row["resname"] != "CYS" or not observed["mask"][index, column]:
            continue
        g, n = row["g"], int(atom_count[row["g"]])
        states[(common.structure_id(row), row["struct_resnum"])] = geometry_context.cysteine_state(
            np.asarray(atom_type[g, :n]), np.asarray(atom_offset[g, :n]),
            np.asarray(atom_xyz[g, :n], dtype=np.float64),
            float(observed["true_resolved"][index, column]))
    return states


def arm_cells(ns, tasks, rows_by_sid, cols, atom_of_col, partner):
    """``arm -> {(sid, residue number): (prediction, target)}`` on the C-beta column, and
    what each arm had withheld as its own training data (``drop_trained_on``)."""
    column = cols.index(NUCLEUS)
    sids = list(tasks)
    exc = common.SelfExcluder()
    out, withheld = {}, {}
    for arm in ns.arms:
        found, dropped = tool_cells(
            arm, sids, rows_by_sid, cols, atom_of_col, partner, tasks, exc,
            enforce_selfexcl=arm not in ns.leaky,
            verify=lambda tool, sid, task, authority:
                verify_tiered(tool, sid, task, authority, ns.tier))
        found, withheld[arm] = drop_trained_on(arm, found, tasks)
        out[arm] = {(sid, key[0]): pair for sid, cells in found.items()
                    for key, pair in cells.items() if key[1] == column}
        print(f"  {arm:24s} structures usable={len(found):5d} dropped={len(dropped):4d} "
              f"C-beta cells={len(out[arm]):6d}")
    out["deepshift"] = {(sid, row["struct_resnum"]): pair
                        for sid, rows in rows_by_sid.items() for row in rows
                        for col, pair in _resolved_cells(
                            row["our_pred"], row["true_raw"], row["obs_cols"], row["aa"],
                            partner, row["amb"]).items() if col == column}
    return out, {arm: record for arm, record in withheld.items() if record}


def choose_cells(universe, arms, master_path):
    """The cells every arm is scored on, and the record of how they were chosen."""
    if master_path is not None:
        master = master_cells.load_master(master_path)
        cells = sorted(universe)
        keys = master_cells.cell_keys([NUCLEUS] * len(cells), [sid for sid, _r in cells],
                                      [resnum for _s, resnum in cells])
        keep = master_cells.restrict(keys, master, what=f"fold-6 {NUCLEUS}")
        chosen = [cell for cell, inside in zip(cells, keep) if inside]
        return chosen, {"source": "master cell file", "path": str(master_path),
                        "rule": "the master set's cells of this nucleus, joined on "
                                "(nucleus column, structure instance, residue number)",
                        "n_master_cells_of_this_nucleus":
                            int((master["atom"] == NUCLEUS).sum()),
                        "n_in_this_run_universe": len(chosen)}
    coverage = {arm: len(universe & set(cells)) / len(universe) for arm, cells in arms.items()}
    joining = sorted(arm for arm, value in coverage.items() if value >= MIN_COVERAGE)
    chosen = sorted(universe.intersection(*(set(arms[arm]) for arm in joining)))
    return chosen, {
        "source": "this producer's own intersection; no master cell file was given",
        "rule": f"the universe is the cysteine {NUCLEUS} cells the fold-6 ensemble "
                f"scored; an arm covering at least {MIN_COVERAGE:.0%} of it joins, and "
                "the cell set is what the joining arms all emitted",
        "n_universe": len(universe), "joining_arms": joining,
        "coverage": {arm: round(value, 4) for arm, value in sorted(coverage.items())}}


def mode_block(observed, predicted, unit, n_units, draws):
    """RMSE, within-mode amplitude and correlation for one arm inside one mode."""
    residual = predicted - observed
    rmse, rmse_draws, mae, _mae_draws = error_replicates(residual, unit, n_units, draws)
    residue = np.zeros(observed.size, dtype=int)          # one residue type: cysteine
    correlation, amplitude, _n = secondary_shift_agreement(predicted, observed, residue)
    r_draws, amplitude_draws = secondary_shift_replicates(predicted, observed, residue,
                                                          unit, draws)
    return {
        "n_cells": int(observed.size), "rmse": rmse, "mae": mae,
        "rmse_ci95": percentile_interval(rmse_draws),
        "amplitude": amplitude, "r": correlation,
        "amplitude_ci95": percentile_interval(amplitude_draws) if amplitude_draws.size else None,
        "r_ci95": percentile_interval(r_draws) if r_draws.size else None,
        "mean_predicted_ppm": float(predicted.mean()),
        "mean_observed_ppm": float(observed.mean()),
    }


def score_panel(table, arms, partial, draws_of):
    """One source panel: the observed split, then every arm inside each mode."""
    oxidized = table["state"] == "oxidized"
    unit, n_units = unit_codes(table["label_set"])
    draws = draws_of(n_units)
    observed = table["true_ppm"]
    node = {"n_cells": int(observed.size), "n_label_sets": n_units,
            "n_disulfide": int(oxidized.sum()), "n_free_thiol": int((~oxidized).sum()),
            "observed_mean_disulfide_ppm": float(observed[oxidized].mean()),
            "observed_mean_free_thiol_ppm": float(observed[~oxidized].mean()),
            "arms": {}}
    node["observed_split_ppm"] = (node["observed_mean_disulfide_ppm"]
                                  - node["observed_mean_free_thiol_ppm"])
    for arm in arms:
        finite = np.isfinite(table[f"pred_{arm}"])
        if (finite & oxidized).sum() < MIN_MODE_CELLS or (finite & ~oxidized).sum() < MIN_MODE_CELLS:
            continue
        entry = {"label": arm, "partial_coverage": arm in partial,
                 "n_cells": int(finite.sum())}
        for name, selected in (("disulfide", finite & oxidized),
                               ("free_thiol", finite & ~oxidized),
                               ("pooled", finite)):
            entry[name] = mode_block(observed[selected], table[f"pred_{arm}"][selected],
                                     unit[selected], n_units, draws)
        entry["pooled"]["amplitude_note"] = (
            "DO NOT QUOTE. Pooled over both modes the gap between the modes dominates "
            "the variance and every arm scores near 1.00 whatever it does inside a mode.")
        entry["recovered_split_ppm"] = (entry["disulfide"]["mean_predicted_ppm"]
                                        - entry["free_thiol"]["mean_predicted_ppm"])
        entry["split_recovered_pct"] = (100.0 * entry["recovered_split_ppm"]
                                        / node["observed_split_ppm"])
        node["arms"][arm] = entry
    return node


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="the arms to score; the default is every arm of "
                         "config.BENCHMARK_ARMS this run's census says returns C-beta")
    ap.add_argument("--leaky", nargs="*", default=list(LEAKY))
    ap.add_argument("--tier", type=int, choices=sorted(LEVEL_BY_TIER), default=2)
    ap.add_argument("--master-cells", type=Path, default=None,
                    help="the frozen master cell file; without it the producer falls "
                         "back to its own arms' intersection and says so")
    ap.add_argument("--out", required=True, help="the report Table S23 is rendered from")
    ap.add_argument("--cells-out", required=True,
                    help="the per-cell dump Figure 4 panel (a) draws")
    return ap


def main(argv=None):
    ns = build_parser().parse_args(argv)
    if ns.arms is None:
        ns.arms = master_cells.arms_returning(NUCLEUS)
    with open(common.OUT_ROOT / "fold6_tasks.tsv") as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}

    _go, rows = common.build_row_index()
    states = disulfide_states(rows)
    print(f"cysteine C-beta cells the ensemble scored: {len(states):,}")
    rows_by_sid, cols, atom_of_col, partner = fold6_rows(list(tasks))
    arms, withheld = arm_cells(ns, tasks, rows_by_sid, cols, atom_of_col, partner)

    with_state = {key for key, state in states.items() if state.geometry_state is not None}
    universe = with_state & set(arms["deepshift"])
    chosen, cell_record = choose_cells(universe, {a: arms[a] for a in ns.arms},
                                       ns.master_cells)
    scored = [arm for arm in ["deepshift", *ns.arms]
              if len(set(arms[arm]) & set(chosen)) >= MIN_MODE_CELLS]
    partial = {arm for arm in scored
               if len(set(arms[arm]) & set(chosen)) < len(chosen)}
    print(f"cells scored: {len(chosen):,}; arms {scored}; partial {sorted(partial)}")

    source_of = {sid: tasks[sid]["source"] for sid in tasks}
    label_of = {sid: tasks[sid]["label_set_id"] for sid in tasks}
    table = {
        "sid": np.array([sid for sid, _r in chosen]),
        "resnum": np.array([resnum for _s, resnum in chosen]),
        "label_set": np.array([label_of[sid] for sid, _r in chosen]),
        "source": np.array([source_of[sid] for sid, _r in chosen]),
        "state": np.array([states[cell].geometry_state for cell in chosen]),
        "true_ppm": np.array([arms["deepshift"][cell][1] for cell in chosen], float),
    }
    for arm in scored:
        table[f"pred_{arm}"] = np.array(
            [arms[arm][cell][0] if cell in arms[arm] else np.nan for cell in chosen], float)

    def draws_of(n_units):
        return bootstrap_indices(n_units, BOOTSTRAP)

    report = {
        "what": "cysteine C-beta accuracy inside each disulfide state, held-out fold 6",
        "nucleus": NUCLEUS,
        "residual": "arm prediction minus the re-referenced observed label; no centring",
        "mode_rule": {
            "predicate": "another SG within "
                         f"{geometry_context.DISULFIDE_BOND_MAX_ANGSTROM} A of the "
                         "residue's own SG in the 8 A cloud the model was given",
            "code": "deepshift.geometry_context.cysteine_state, geometry_state",
            "note": "read from the coordinates, never from the shift"},
        "cell_set": cell_record,
        "exclusion_tier": ns.tier,
        "exclusion_level": LEVEL_BY_TIER[ns.tier].value,
        "leaky_arms": [arm for arm in ns.leaky if arm in scored],
        "partial_coverage_arms": sorted(partial),
        "withheld_as_training_data": withheld,
        "bootstrap": {"unit": BOOTSTRAP.unit, "resamples": BOOTSTRAP.resamples,
                      "seed": BOOTSTRAP.seed},
        "cohort": {
            "cys_cb_cells_the_ensemble_scored": len(states),
            "cells_with_a_disulfide_state": len(with_state),
            "cells_scored": len(chosen),
            "n_structure_instances": len(set(table["sid"])),
            "n_label_sets": len(set(table["label_set"]))},
        "arms": {arm: arm for arm in scored},
        "by_source": {}, "checks": {},
    }
    for source in SOURCES:
        keep = np.ones(len(chosen), bool) if source == "all" else (table["source"] == source)
        if keep.sum() < MIN_MODE_CELLS:
            continue
        report["by_source"][source] = score_panel(
            {k: v[keep] for k, v in table.items()}, scored, partial, draws_of)

    # The mode rule, checked against the one other thing that knows the state: the
    # deposited C-beta value itself, classified by the package's own deadband.
    shift_state = np.array([states[cell].shift_state or "" for cell in chosen])
    both = shift_state != ""
    report["checks"]["geometry_state_vs_deposited_shift_state"] = {
        "rule": "geometry_context.cysteine_shift_state on the deposited C-beta value, "
                f"outside the {geometry_context.CYS_REDUCED_MAX_PPM}-"
                f"{geometry_context.CYS_OXIDIZED_MIN_PPM} ppm deadband",
        "n_cells_with_both": int(both.sum()),
        "n_agree": int((shift_state[both] == table["state"][both]).sum()),
        "n_disagree": int((shift_state[both] != table["state"][both]).sum()),
        "reading": "the two assignments are independent -- one reads coordinates, the "
                   "other the shift -- so their agreement rate measures the mode rule"}

    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ns.cells_out, arms=np.array(scored),
                        ss_state=np.where(table["state"] == "oxidized", "ss", "free"),
                        **{k: v for k, v in table.items() if k != "state"})
    Path(ns.out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"wrote {ns.out} and {ns.cells_out}")
    for source, node in report["by_source"].items():
        print(f"--- {source}: n={node['n_cells']} ss={node['n_disulfide']} "
              f"free={node['n_free_thiol']} observed split={node['observed_split_ppm']:.3f}")
        for arm, entry in node["arms"].items():
            print(f"    {arm:24s} rmse ss/free {entry['disulfide']['rmse']:.3f}/"
                  f"{entry['free_thiol']['rmse']:.3f}  amp "
                  f"{entry['disulfide']['amplitude']:.3f}/{entry['free_thiol']['amplitude']:.3f}"
                  f"  split {entry['recovered_split_ppm']:.3f}")
    return report


if __name__ == "__main__":
    main()
