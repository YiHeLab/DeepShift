#!/usr/bin/env python3
"""Out-of-fold backbone error with the residue's own (phi, psi), for the error surface.

COHORT. The five cross-validation members' held-out dumps, ``final/fold{k}.preds.npz``:
member k is trained on folds {1..5} minus k and predicts only its own held-out fold, so
pooling the five is out-of-fold everywhere and gives the cell count 5-degree bins need.
Fold 6 is not in it: fold 6 is the benchmark's held-out fold and the five members were
never trained without it in the way this pooling requires.

TORSIONS. phi and psi of each row's own centre residue, from the 8 A cloud the model read,
through ``geometry_context.backbone_torsions_degrees``. The cloud carries no residue
numbering, so the preceding C and the following N are located by their peptide bonds; a
row whose neighbour the cloud does not carry has no angle and is dropped, not guessed at.
``--validate`` re-derives the same angles from the structure files themselves, through the
release's PDB parser, on a sample of fold 6 and records the disagreement.

CELLS. Every backbone cell the dump declares effective -- the exclusion fence was applied
when the predictions were written -- with its absolute error in ppm.

    python -m benchmarks.dump_cv_torsions --out <dir>/cv_cells.npz [--validate 150]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deepshift import evaluate, geometry_context, structure
from deepshift.calibration import MEMBER_FOLDS
from deepshift.normalization import BACKBONE_SHIFTS

from . import config
from .score_source_matched import check_dumps

#: Where the trainer leaves each member's held-out predictions inside the run.
DUMP_NAME = "final/fold%d.preds.npz"


def fold_cache(build, fold: int) -> Path:
    """The cache directory of one fold of this run."""
    return build.run / f"work/unified_cache/fold_{fold}"


def eligible_rows(cache: Path):
    """The cache rows a prediction dump carries, and their identities, in dump order."""
    atom_count = np.load(cache / "atom/atom_count.npy")
    shift_mask = np.load(cache / "structural/shift_mask.npy")
    order = np.flatnonzero((atom_count > 0) & shift_mask.any(1))
    with open(cache / "structural/id_maps.json") as fh:
        id_maps = json.load(fh)
    keys = np.asarray([f"{id_maps['materialization_id'][str(int(g))]}|"
                       f"{id_maps['label_residue_id'][str(int(g))]}" for g in order])
    return order, keys


def torsions(cache: Path, order) -> np.ndarray:
    """``(phi, psi)`` per eligible row of one fold, in degrees."""
    atom_type = np.load(cache / "atom/atom_type.npy", mmap_mode="r")
    atom_offset = np.load(cache / "atom/atom_offset.npy", mmap_mode="r")
    atom_xyz = np.load(cache / "atom/atom_xyz.npy", mmap_mode="r")
    atom_count = np.load(cache / "atom/atom_count.npy", mmap_mode="r")
    out = np.empty((len(order), 2), dtype=np.float64)
    for row, g in enumerate(order):
        n = int(atom_count[g])
        out[row] = geometry_context.backbone_torsions_degrees(
            np.asarray(atom_type[g, :n]), np.asarray(atom_offset[g, :n]),
            np.asarray(atom_xyz[g, :n], dtype=np.float64))
    return out


def validate_against_structures(build, n_structures: int) -> dict:
    """The same angles from the coordinate files, on a sample of the fold-6 roster.

    The cloud is a gathered neighbourhood; the file is the structure it was gathered
    from. Agreement between the two is what says the peptide-bond search found the real
    neighbours, so it is measured rather than asserted.
    """
    from . import cells as common  # fold-6 bound, which this is

    _order, rows = common.build_row_index()
    build_logs = common.load_build_logs()
    by_structure: dict[str, list] = {}
    for index, row in enumerate(rows):
        by_structure.setdefault(row["materialization_id"], []).append((index, row))
    sampled = sorted(by_structure)[:n_structures]

    cache = build.cache
    order = common.g_order()
    cloud = torsions(cache, order)
    differences, missing = [], 0
    for materialization_id in sampled:
        log = build_logs[materialization_id]
        path = common.resolve_pdb(by_structure[materialization_id][0][1]["source"], log["pdb_path"])
        models = structure.parse_models(path, log["chain_id"])
        if not models:
            continue
        model_index = int(by_structure[materialization_id][0][1]["model_index"])
        residues = structure.to_int_resid(
            structure.canonical_residues(models[min(model_index, len(models) - 1)]))
        for index, row in by_structure[materialization_id]:
            number = int(row["struct_resnum"])
            here, before, after = (residues.get(number), residues.get(number - 1),
                                   residues.get(number + 1))
            if here is None or before is None or after is None:
                missing += 1
                continue
            try:
                parsed = (geometry_context.dihedral_degrees(
                              before["atoms"]["C"], here["atoms"]["N"],
                              here["atoms"]["CA"], here["atoms"]["C"]),
                          geometry_context.dihedral_degrees(
                              here["atoms"]["N"], here["atoms"]["CA"],
                              here["atoms"]["C"], after["atoms"]["N"]))
            except KeyError:
                missing += 1
                continue
            for parsed_angle, cloud_angle in zip(parsed, cloud[index]):
                if np.isfinite(parsed_angle) and np.isfinite(cloud_angle):
                    difference = abs(parsed_angle - cloud_angle)
                    differences.append(min(difference, 360.0 - difference))
    differences = np.asarray(differences)
    return {
        "against": "deepshift.structure.parse_models + canonical_residues on the "
                   "coordinate file, with the sequence neighbours taken by residue number",
        "cohort": f"{len(sampled)} structures of the fold-6 roster, "
                  f"{int(differences.size)} angles compared",
        "median_abs_diff_deg": float(np.median(differences)) if differences.size else None,
        "p99_abs_diff_deg": float(np.percentile(differences, 99)) if differences.size else None,
        "frac_beyond_1deg": float((differences > 1.0).mean()) if differences.size else None,
        "n_rows_without_a_sequence_neighbour_in_the_file": missing,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", type=int, default=150,
                    help="structures of the fold-6 roster to re-derive the angles for; "
                         "0 skips the check")
    ns = ap.parse_args(argv)

    build = config.current()
    dumps = {fold: build.run / (DUMP_NAME % fold) for fold in MEMBER_FOLDS}
    missing = [str(path) for path in dumps.values() if not path.exists()]
    if missing:
        raise SystemExit("missing held-out member dumps:\n  " + "\n  ".join(missing))
    check_dumps(list(dumps.values()))

    columns = list(BACKBONE_SHIFTS)
    atom, phi, psi, error, fold_of = [], [], [], [], []
    for fold, path in dumps.items():
        predictions = evaluate.read_predictions(path)
        cache = fold_cache(build, fold)
        order, keys = eligible_rows(cache)
        if not np.array_equal(predictions.row_key, keys):
            raise SystemExit(f"{path} does not carry the eligible rows of {cache}, in order")
        angles = torsions(cache, order)
        defined = np.isfinite(angles).all(1)
        absolute = np.abs(predictions.pred_ppm - predictions.resolved_ppm)
        cells = 0
        for index, column in enumerate(columns):
            j = predictions.shift_cols.index(column)
            selected = predictions.effective_mask[:, j] & defined
            rows = np.flatnonzero(selected)
            atom.append(np.full(rows.size, index, dtype=np.int16))
            phi.append(angles[rows, 0])
            psi.append(angles[rows, 1])
            error.append(absolute[rows, j])
            fold_of.append(np.full(rows.size, fold, dtype=np.int16))
            cells += rows.size
        print(f"fold {fold}: {defined.sum():,}/{defined.size:,} rows with (phi, psi), "
              f"{cells:,} backbone cells")

    payload = {"atom_names": np.asarray(columns), "atom": np.concatenate(atom),
               "phi": np.concatenate(phi), "psi": np.concatenate(psi),
               "abs_err": np.concatenate(error), "fold": np.concatenate(fold_of)}
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)

    if ns.validate:
        print("validation:", json.dumps(validate_against_structures(build, ns.validate),
                                        indent=1))
    print(f"wrote {out}: {payload['atom'].size:,} cells")


if __name__ == "__main__":
    main()
