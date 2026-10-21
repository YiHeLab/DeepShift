#!/usr/bin/env python3
"""Aromatic 1H sites of fold 6, with the ring-current field their structure fixes.

The question Figure 4 asks with these cells is whether a predictor responds to a field the
coordinates already determine. The field is therefore computed here from the structure each
prediction was made from -- ``deepshift.ring_current``, a sum over every aromatic ring
except the proton's own residue's -- and never from a shift. Each proton is placed from its
heavy parent, because the pipeline's clouds carry no hydrogens.

SITES. The seventeen aromatic CH and NH sites of Phe, Tyr, Trp and His
(``ring_current.AROMATIC_PROTONS``) that the fold-6 ensemble scored: the exclusion fence was
applied when those predictions were written.

ARMS. ``benchmarks/config.py``'s roster at the reported exclusion, less the arms this
run's coverage census says return no aromatic 1H. Every listed arm's own value on the
same cells, resolved by the benchmark's per-arm prochiral rule. An arm that does not
emit a site leaves NaN there and is not dropped: SHIFTX2 emits no tryptophan six-ring
protons at all, and blanking those cells for every arm would change the cohort rather
than record the gap.

``ring_ppm`` is stored as a SHIELDING, positive above a ring face; the ppm shift a nucleus
sees is its negative, which is the convention the figure draws.

    python -m benchmarks.dump_ring_cells --out <dir>/ring_cells.npz
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from pathlib import Path
from typing import NamedTuple

import numpy as np

from deepshift import ring_current, structure
from deepshift.chemistry import AA_3_TO_1

from . import cells as common
from .master_cells import arms_returning
from .score_by_source import tool_cells
from .score_by_source_tier import LEVEL_BY_TIER, verify_tiered
from .score_validation import _resolved_cells, fold6_rows

#: One-letter to three-letter, for naming the residue a site sits on.
AA_1_TO_3 = {one: three for three, one in AA_3_TO_1.items()}


class Site(NamedTuple):
    """One scored aromatic proton: where it is, what it reads, and its field."""

    sid: str
    resnum: str
    column: int
    proton: str
    residue: str
    label_set: str
    observed: float
    predicted: float
    shielding: float

#: The nucleus columns of the aromatic sites, which name the arms that can be compared here.
RING_COLUMNS = tuple(sorted({f"{proton.lower()}_shift"
                             for protons in ring_current.AROMATIC_PROTONS.values()
                             for proton in protons}))


def one_structure(task):
    """``[(residue number, residue name, proton, shielding ppm)]`` for one structure."""
    structure_id, path, chain, model_index = task
    try:
        models = structure.parse_models(path, chain)
        if not models:
            return structure_id, [], {"structure_id": structure_id, "error": "no models"}
        model = models[min(int(model_index), len(models) - 1)]
        residues = structure.to_int_resid(
            structure.canonical_residues(model, drop_hydrogens=True))
    except (OSError, ValueError) as error:
        return structure_id, [], {"structure_id": structure_id, "error": str(error)}

    rings = ring_current.ring_geometry(residues)
    own: dict[int, list] = {}
    for ring in rings:
        own.setdefault(ring.residue, []).append(ring)
    rows = []
    for number, residue in residues.items():
        name = residue["residue_name"]
        for proton, parent in ring_current.AROMATIC_PROTONS.get(name, {}).items():
            point = residue["atoms"].get(parent)
            if point is None:
                continue
            placed = ring_current.proton_position(point, own.get(number, []))
            if placed is None:
                continue
            rows.append((number, name, proton,
                         ring_current.ring_current_shielding(placed, rings, skip=number)))
    return structure_id, rows, None


def ring_field(tasks, processes: int):
    """``(structure instance, residue number, proton) -> shielding ppm``."""
    jobs = [(task["structure_id"], task["pdb_path"], task["chain"], task["model_index"])
            for task in tasks.values()]
    with mp.Pool(processes) as pool:
        results = pool.map(one_structure, jobs, chunksize=8)
    field, failed = {}, []
    for structure_id, rows, error in results:
        if error:
            failed.append(error)
        for number, _name, proton, shielding in rows:
            field[(structure_id, str(number), proton)] = shielding
    return field, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="the arms to record beside DeepShift; the default is every arm "
                         "of config.BENCHMARK_ARMS at the exclusion the paper reports "
                         "that this run's census says returns an aromatic 1H")
    ap.add_argument("--tier", type=int, choices=sorted(LEVEL_BY_TIER), default=2)
    ap.add_argument("--processes", type=int, default=min(12, mp.cpu_count()))
    ap.add_argument("--out", required=True)
    ns = ap.parse_args(argv)
    if ns.arms is None:
        ns.arms = arms_returning(*RING_COLUMNS, tiers=(ns.tier,))

    with open(common.OUT_ROOT / "fold6_tasks.tsv") as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}
    field, failed = ring_field(tasks, ns.processes)
    print(f"ring field at {len(field):,} aromatic proton sites "
          f"({len(failed)} structures failed)")

    rows_by_sid, cols, atom_of_col, partner = fold6_rows(list(tasks))
    column_of = {proton: cols.index(f"{proton.lower()}_shift")
                 for protons in ring_current.AROMATIC_PROTONS.values()
                 for proton in protons}
    proton_of_column = {column: proton for proton, column in column_of.items()}

    exc = common.SelfExcluder()
    arms = {}
    for arm in ns.arms:
        found, dropped = tool_cells(
            arm, list(tasks), rows_by_sid, cols, atom_of_col, partner, tasks, exc,
            verify=lambda tool, sid, task, authority:
                verify_tiered(tool, sid, task, authority, ns.tier))
        arms[arm] = {(sid, key[0], key[1]): pair for sid, found_cells in found.items()
                     for key, pair in found_cells.items() if key[1] in proton_of_column}
        print(f"  {arm:24s} structures usable={len(found):5d} dropped={len(dropped):4d} "
              f"aromatic 1H cells={len(arms[arm]):6d}")

    cells = []
    for sid, rows in rows_by_sid.items():
        for row in rows:
            for column, (prediction, target) in _resolved_cells(
                    row["our_pred"], row["true_raw"], row["obs_cols"], row["aa"],
                    partner, row["amb"]).items():
                proton = proton_of_column.get(column)
                if proton is None:
                    continue
                shielding = field.get((sid, row["struct_resnum"], proton))
                if shielding is None:
                    continue
                cells.append(Site(sid, row["struct_resnum"], column, proton,
                                  AA_1_TO_3[row["res1"]], row["label_set_id"],
                                  target, prediction, shielding))
    print(f"aromatic 1H cells with a field: {len(cells):,}")

    source_of = {sid: tasks[sid]["source"] for sid in tasks}
    payload = {
        "arms": np.asarray(["deepshift", *ns.arms]),
        "sid": np.asarray([c.sid for c in cells]),
        "resnum": np.asarray([c.resnum for c in cells]),
        "atom": np.asarray([c.proton for c in cells]),
        "residue": np.asarray([c.residue for c in cells]),
        "label_set": np.asarray([c.label_set for c in cells]),
        "source": np.asarray([source_of[c.sid] for c in cells]),
        "true_ppm": np.asarray([c.observed for c in cells], dtype=np.float64),
        "ring_ppm": np.asarray([c.shielding for c in cells], dtype=np.float64),
        "pred_deepshift": np.asarray([c.predicted for c in cells], dtype=np.float64),
    }
    for arm in ns.arms:
        payload[f"pred_{arm}"] = np.asarray(
            [arms[arm].get((c.sid, c.resnum, c.column), (np.nan, np.nan))[0]
             for c in cells],
            dtype=np.float64)

    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    n_structures = len(set(payload["sid"].tolist()))
    print(f"wrote {out}: {len(cells):,} cells over {n_structures} structures")


if __name__ == "__main__":
    main()
