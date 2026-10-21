#!/usr/bin/env python3
"""DeepShift's held-out fold-6 error against the properties taken to measure a structure.

One question: does the error track the things usually taken to say how good a structure
is? Six covariates cut the same cells six ways -- AlphaFold's own per-residue confidence,
crystallographic resolution, how close the residue sits to an aromatic ring, how buried it
is, which kind of structure it came from, and how many kinds the deposition has.

This is not a predictor comparison, so it keeps its own cohort and states it: every
backbone cell the fold-6 five-member ensemble scored, on the whole fold-6 roster, less the
anticorrelated channels every float excludes (``benchmarks/reference/
anticorrelated_channels.json``). The composite exclusion fence was applied when those
predictions were written. Residuals are raw -- prediction minus the re-referenced observed
value, nothing removed.

The covariates are computed here, from the structure each prediction was made from, on the
chain and model the benchmark gave the predictors:

  plddt            AlphaFold's per-residue confidence, the CA temperature factor of the
                   model file. AlphaFold structures only.
  resolution       the crystallographic resolution in the coordinate file's own
                   ``REMARK   2 RESOLUTION`` record. X-ray structures only.
  ring_distance_A  CA to the centre of the nearest aromatic ring of a different residue,
                   through ``deepshift.ring_current.ring_geometry``; infinite where the
                   structure has no such ring, which the farthest band holds.
  rsa              the residue's solvent accessible area over the accessible area of that
                   residue type in an extended Gly-X-Gly tripeptide (Tien et al. 2013
                   theoretical maxima); 0 buried, 1 exposed. Areas from freesasa on heavy
                   atoms with element van der Waals radii.

Blocks are not comparable to each other: the confidence block is AlphaFold structures only
and the resolution block X-ray structures only.

    python -m benchmarks.score_error_covariates --out <dir>/si_error_covariates.json
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np

from deepshift import ring_current, structure
from deepshift.statistics import (
    DEFAULT_SEED,
    BootstrapSpec,
    bootstrap_indices,
    error_replicates,
    percentile_interval,
    unit_codes,
)

from . import cells as common
from .master_cells import anticorrelated_channels
from .score_validation import _resolved_cells, fold6_rows

BACKBONE = ["HA", "H", "CA", "CB", "C", "N"]
BOOTSTRAP = BootstrapSpec(unit="label_set", resamples=2000, seed=DEFAULT_SEED)
MIN_STRATUM_CELLS = 30

#: Element van der Waals radii, Angstrom, for the accessibility calculation.
RADIUS = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}
#: Tien et al. 2013 (PLoS ONE 8:e80635) theoretical Gly-X-Gly maxima, square Angstrom.
MAXASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLU": 223,
          "GLN": 225, "GLY": 104, "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236,
          "MET": 224, "PHE": 240, "PRO": 159, "SER": 155, "THR": 172, "TRP": 285,
          "TYR": 263, "VAL": 174}
RESOLUTION_RECORD = re.compile(r"^REMARK   2 RESOLUTION\.\s+([0-9.]+)\s+ANGSTROM")


def resolution_from_header(path) -> float:
    """The resolution the coordinate file itself states, NaN when it states none."""
    with open(path, errors="ignore") as stream:
        for line in stream:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                break
            found = RESOLUTION_RECORD.match(line)
            if found:
                return float(found.group(1))
    return float("nan")


def one_structure(task):
    """Per-residue covariates of one structure instance, on its own chain and model."""
    import freesasa

    structure_id, path, chain, model_index, source = task
    try:
        models = structure.parse_models(path, chain)
        if not models:
            return structure_id, [], {"structure_id": structure_id, "error": "no models"}
        model = models[min(int(model_index), len(models) - 1)]
        residues = structure.to_int_resid(
            structure.canonical_residues(model, drop_hydrogens=True))
    except (OSError, ValueError) as error:
        return structure_id, [], {"structure_id": structure_id, "error": str(error)}
    if not residues:
        return structure_id, [], {"structure_id": structure_id,
                                  "error": "no residues for that chain and model"}

    coordinates, radii, owner = [], [], []
    for number, residue in residues.items():
        for name, point in residue["atoms"].items():
            element = next((character for character in name if character.isalpha()), "C")
            coordinates.extend(float(value) for value in point)
            radii.append(RADIUS.get(element.upper(), 1.80))
            owner.append(number)
    try:
        areas = freesasa.calcCoord(coordinates, radii)
        per_atom = np.asarray([areas.atomArea(index) for index in range(len(radii))])
    except Exception as error:                            # freesasa raises bare exceptions
        return structure_id, [], {"structure_id": structure_id,
                                  "error": f"accessibility: {error}"}
    accessible: dict[int, float] = {}
    for number, area in zip(owner, per_atom):
        accessible[number] = accessible.get(number, 0.0) + float(area)

    rings = ring_current.ring_geometry(residues)
    centers = np.asarray([ring.center for ring in rings]) if rings else np.zeros((0, 3))
    ring_owner = np.asarray([ring.residue for ring in rings])
    plddt = (structure.ca_temperature_factors(path, chain) if source == "af" else {})

    rows = []
    for number, residue in residues.items():
        alpha = residue["atoms"].get("CA")
        if alpha is None:
            continue
        alpha = np.asarray(alpha, dtype=np.float64)
        other = ring_owner != number
        distance = (float(np.min(np.linalg.norm(centers[other] - alpha, axis=1)))
                    if len(centers) and other.any() else np.inf)
        maximum = MAXASA.get(residue["residue_name"])
        rows.append((number, residue["residue_name"], distance,
                     accessible.get(number, 0.0) / maximum if maximum else np.nan,
                     plddt.get(number, (np.nan, ""))[0]))
    return structure_id, rows, None


def geometry_pass(tasks, processes: int):
    """``(structure instance, residue number) -> (ring distance, rsa, plddt)``."""
    jobs = [(task["structure_id"], task["pdb_path"], task["chain"], task["model_index"],
             task["source"]) for task in tasks.values()]
    with mp.Pool(processes) as pool:
        results = pool.map(one_structure, jobs, chunksize=8)
    covariates, failed = {}, []
    for structure_id, rows, error in results:
        if error:
            failed.append(error)
        for number, _name, distance, rsa, plddt in rows:
            covariates[(structure_id, str(number))] = (distance, rsa, plddt)
    return covariates, failed


def deepshift_cells(tasks):
    """Every backbone cell the fold-6 ensemble scored, with its residual and identity,
    less the anticorrelated channels. -> (cells, number of channel cells dropped)."""
    rows_by_sid, _cols, atom_of_col, partner = fold6_rows(list(tasks))
    keep = {index for index, name in enumerate(atom_of_col) if name in BACKBONE}
    channels, _record = anticorrelated_channels()
    cells, dropped = [], 0
    for sid, rows in rows_by_sid.items():
        for row in rows:
            for column, (prediction, target) in _resolved_cells(
                    row["our_pred"], row["true_raw"], row["obs_cols"], row["aa"],
                    partner, row["amb"]).items():
                if column not in keep:
                    continue
                if (row["label_set_id"], atom_of_col[column]) in channels:
                    dropped += 1
                    continue
                cells.append((sid, row["struct_resnum"], atom_of_col[column],
                              row["label_set_id"], prediction - target))
    return cells, dropped


def strata(source, plddt, resolution, ring, rsa, n_sources):
    """The six blocks, as the table prints them."""
    af, xtal = source == "af", source == "xtal"
    return [
        ("plddt", "AlphaFold confidence (pLDDT), AlphaFold structures", [
            ("$<$50", af & (plddt < 50)),
            ("50--70", af & (plddt >= 50) & (plddt < 70)),
            ("70--80", af & (plddt >= 70) & (plddt < 80)),
            ("80--90", af & (plddt >= 80) & (plddt < 90)),
            ("90--95", af & (plddt >= 90) & (plddt < 95)),
            (r"$\geq$95", af & (plddt >= 95))]),
        ("resolution", r"Crystallographic resolution (\AA), X-ray structures", [
            (r"$\leq$1.5", xtal & (resolution <= 1.5)),
            ("1.5--2.0", xtal & (resolution > 1.5) & (resolution <= 2.0)),
            ("2.0--2.5", xtal & (resolution > 2.0) & (resolution <= 2.5)),
            ("$>$2.5", xtal & (resolution > 2.5))]),
        ("ring", r"Distance to the nearest aromatic ring (\AA)", [
            ("$<$5", ring < 5),
            ("5--6", (ring >= 5) & (ring < 6)),
            ("6--8", (ring >= 6) & (ring < 8)),
            ("8--12", (ring >= 8) & (ring < 12)),
            # a structure with no aromatic ring of another residue is farther than any
            # band edge, so it belongs to the farthest band, not to none
            (r"$\geq$12 or none", ring >= 12)]),
        ("rsa", "Relative solvent accessibility", [
            ("$<$0.10 (buried)", rsa < 0.10),
            ("0.10--0.25", (rsa >= 0.10) & (rsa < 0.25)),
            ("0.25--0.50", (rsa >= 0.25) & (rsa < 0.50)),
            (r"$\geq$0.50 (exposed)", rsa >= 0.50)]),
        ("source", "Structure source", [
            ("AlphaFold", af), ("solution NMR", source == "nmr"), ("X-ray crystal", xtal)]),
        ("n_sources", "Structure sources available for the deposition", [
            ("one", n_sources == 1), ("two", n_sources == 2), ("three", n_sources == 3)]),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--processes", type=int, default=min(12, mp.cpu_count()))
    ns = ap.parse_args(argv)

    with open(common.OUT_ROOT / "fold6_tasks.tsv") as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}
    print(f"fold-6 roster: {len(tasks)} structure instances")
    cells, n_anticorrelated = deepshift_cells(tasks)
    print(f"DeepShift backbone cells: {len(cells):,} "
          f"({n_anticorrelated:,} anticorrelated-channel cells dropped)")
    covariates, failed = geometry_pass(tasks, ns.processes)
    print(f"per-residue covariates: {len(covariates):,} residues, "
          f"{len(failed)} structures failed")

    sid = np.asarray([cell[0] for cell in cells])
    resnum = np.asarray([cell[1] for cell in cells])
    atom = np.asarray([cell[2] for cell in cells])
    label_set = np.asarray([cell[3] for cell in cells])
    residual = np.asarray([cell[4] for cell in cells], dtype=np.float64)
    joined = np.asarray([covariates.get((s, r), (np.nan, np.nan, np.nan))
                         for s, r in zip(sid, resnum)], dtype=np.float64)
    ring, rsa, plddt = joined[:, 0], joined[:, 1], joined[:, 2]
    hit = np.asarray([(s, r) in covariates for s, r in zip(sid, resnum)])

    source_of = {s: tasks[s]["source"] for s in tasks}
    source = np.asarray([source_of[s] for s in sid])
    resolution_of = {s: resolution_from_header(tasks[s]["pdb_path"])
                     for s in tasks if tasks[s]["source"] == "xtal"}
    resolution = np.asarray([resolution_of.get(s, np.nan) for s in sid])
    sources_of_label: dict[str, set] = {}
    for task in tasks.values():
        sources_of_label.setdefault(task["label_set_id"], set()).add(task["source"])
    n_sources = np.asarray([len(sources_of_label[ls]) for ls in label_set])

    unit, n_units = unit_codes(label_set)
    draws = bootstrap_indices(n_units, BOOTSTRAP)
    report = {
        "what": "DeepShift held-out fold-6 backbone RMSE stratified by six structure "
                "covariates, measured on this run",
        "cohort": "every backbone cell the fold-6 five-member ensemble scored, on the "
                  "whole fold-6 roster, less the anticorrelated channels; not a predictor "
                  "comparison, so not the master set",
        "anticorrelated_cells_dropped": n_anticorrelated,
        "residual": "DeepShift prediction minus the re-referenced observed label; "
                    "no centring",
        "covariate_definitions": {
            "plddt": "AlphaFold per-residue confidence, the CA temperature factor of the "
                     "model file the cache was built from",
            "resolution": "the resolution stated in the coordinate file's own "
                          "REMARK 2 RESOLUTION record",
            "ring": "CA to the centre of the nearest aromatic ring (Phe, Tyr, Trp, His) "
                    "of a different residue, ring sets from deepshift.ring_current; a "
                    "residue in a structure with no such ring is in the farthest band",
            "rsa": "solvent accessible area over the Tien et al. 2013 Gly-X-Gly maximum "
                   "for that residue type; freesasa on heavy atoms",
            "source": "the kind of structure the prediction was made from",
            "n_sources": "how many of AlphaFold, solution NMR and X-ray the deposition "
                         "has a structure for on the fold-6 roster"},
        "bootstrap": {"unit": BOOTSTRAP.unit, "n_units": n_units,
                      "resamples": BOOTSTRAP.resamples, "seed": BOOTSTRAP.seed,
                      "scope": "every stratum and nucleus; the table prints point "
                               "estimates and the intervals live here"},
        "n_cells": int(residual.size),
        "n_structure_instances": len(set(sid.tolist())),
        "structures_that_failed": failed,
        "join_coverage": {
            "cells_with_a_geometry_covariate": int(hit.sum()),
            "cells_without": int((~hit).sum()),
            "alphafold_cells_with_a_plddt": int(np.isfinite(plddt).sum()),
            "alphafold_cells_total": int((source == "af").sum()),
            "crystal_cells_with_a_resolution": int(np.isfinite(resolution).sum()),
            "crystal_cells_total": int((source == "xtal").sum())},
        "blocks": {}, "checks": {},
    }

    for key, title, block in strata(source, plddt, resolution, ring, rsa, n_sources):
        node = {"title": title, "strata": {}}
        for name, selected in block:
            entry = {"n_cells": int(selected.sum()),
                     "n_structure_instances": len(set(sid[selected].tolist())),
                     "n_label_sets": len(set(label_set[selected].tolist())),
                     "nuclei": {}}
            for nucleus in BACKBONE:
                inside = selected & (atom == nucleus)
                if inside.sum() < MIN_STRATUM_CELLS:
                    entry["nuclei"][nucleus] = {"n_cells": int(inside.sum()), "rmse": None}
                    continue
                rmse, rmse_draws, mae, _mae = error_replicates(
                    residual[inside], unit[inside], n_units, draws)
                entry["nuclei"][nucleus] = {"n_cells": int(inside.sum()), "rmse": rmse,
                                            "rmse_ci95": percentile_interval(rmse_draws),
                                            "mae": mae}
            node["strata"][name] = entry
        report["blocks"][key] = node

    bands = list(report["blocks"]["plddt"]["strata"])
    inversion = {"claim": "the lowest confidence band is the best stratum on all six "
                          "backbone nuclei", "per_nucleus": {}}
    best_low = 0
    for nucleus in BACKBONE:
        by_band = {band: report["blocks"]["plddt"]["strata"][band]["nuclei"][nucleus]["rmse"]
                   for band in bands}
        present = {band: value for band, value in by_band.items() if value is not None}
        by_mae = {band: report["blocks"]["plddt"]["strata"][band]["nuclei"][nucleus].get("mae")
                  for band in bands}
        present_mae = {band: value for band, value in by_mae.items() if value is not None}
        best = min(present, key=present.get)
        best_low += best == bands[0]
        inversion["per_nucleus"][nucleus] = {
            "rmse_by_band": present, "best_band_by_rmse": best,
            "best_band_by_mae": min(present_mae, key=present_mae.get),
            "lowest_band_ci95":
                report["blocks"]["plddt"]["strata"][bands[0]]["nuclei"][nucleus].get("rmse_ci95"),
            "highest_band_ci95":
                report["blocks"]["plddt"]["strata"][bands[-1]]["nuclei"][nucleus].get("rmse_ci95")}
    inversion["n_nuclei_where_the_lowest_band_is_best_by_rmse"] = int(best_low)
    inversion["n_nuclei_where_the_lowest_band_is_best_by_mae"] = int(sum(
        inversion["per_nucleus"][n]["best_band_by_mae"] == bands[0] for n in BACKBONE))
    inversion["verdict"] = (
        "reproduced" if best_low == len(BACKBONE) else
        f"NOT reproduced as stated: the lowest band is best on {best_low} of the "
        f"{len(BACKBONE)} nuclei on this cohort")
    report["checks"]["plddt_inversion"] = inversion
    report["checks"]["source_block_partitions_the_cohort"] = {
        "n_cells_in_the_three_source_strata": int(sum(
            report["blocks"]["source"]["strata"][name]["n_cells"]
            for name in ("AlphaFold", "solution NMR", "X-ray crystal"))),
        "n_cells_in_the_cohort": int(residual.size),
        "reading": "the source block is a partition, so its strata must sum to the cohort"}
    # Every block but the two source-restricted ones partitions the cohort; a stratum
    # rule that drops a residue (as an infinite ring distance once did) is caught here.
    for key in ("ring", "rsa", "source", "n_sources"):
        total = sum(entry["n_cells"] for entry in report["blocks"][key]["strata"].values())
        report["checks"][f"{key}_block_partitions_the_cohort"] = {
            "n_cells_in_the_strata": total, "n_cells_in_the_cohort": int(residual.size)}
        if total != residual.size:
            raise SystemExit(f"the {key} block holds {total:,} cells of the cohort's "
                             f"{residual.size:,}: its strata do not partition it")

    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("pLDDT inversion:", inversion["verdict"])
    print("wrote", out)
    return report


if __name__ == "__main__":
    main()
