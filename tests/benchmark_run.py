"""A run root the fold-6 competitor benchmark can be pointed at.

Small and synthetic: one structure instance per source, three observed nuclei each, five
member prediction dumps in the schema ``deepshift evaluate --save-npz`` writes, and the
authorities the benchmark resolves from a run root. A test that needs other residues,
other nuclei or more label sets hands ``build_run`` its own rows: a row names the columns
it observes, the deposited label of each ``labels`` entry, and the error every member
makes on each ``errors`` entry.
"""
import csv
import json
from pathlib import Path

import numpy as np

from deepshift import evaluate
from deepshift.domain import SHIFT_COLUMNS, STANDARD_RESIDUES, VALID_SOURCES
from deepshift.quality_control import ARG_NE_OBSERVED_RANGE_PPM

COLUMNS = tuple(SHIFT_COLUMNS)
OBSERVED = ("ca_shift", "cb_shift", "n_shift")
MEMBER_FOLDS = (1, 2, 3, 4, 5)

# A cohort wide enough to compare one nucleus with another: eight label sets to resample,
# two thirds of the residues arginine so its N-epsilon column carries cells of its own.
SIDE_CHAIN_STRUCTURES = 8
SIDE_CHAIN_RESIDUES = 24
SIDE_CHAIN_ARGININES = 16
SIDE_CHAIN_ALIASED = 4
SIDE_CHAIN_OBSERVED = ("ca_shift", "cb_shift", "cg_shift", "cd_shift")
ARG_NE_COLUMN = "ne_shift"


def _rows():
    """One structure instance per source, four label residues in cache order."""
    return [
        {"g": 0, "bmrb": "11", "source": "af", "struct_resnum": "5", "comp_id": "ALA",
         "accession": "P00001", "chain": "A", "observed": OBSERVED},
        {"g": 1, "bmrb": "11", "source": "af", "struct_resnum": "6", "comp_id": "SER",
         "accession": "P00001", "chain": "A", "observed": OBSERVED},
        {"g": 2, "bmrb": "22", "source": "nmr", "struct_resnum": "7", "comp_id": "LEU",
         "accession": "1abc", "chain": "A", "observed": OBSERVED},
        {"g": 3, "bmrb": "33", "source": "xtal", "struct_resnum": "8", "comp_id": "VAL",
         "accession": "2abc", "chain": "A", "observed": OBSERVED},
    ]


def side_chain_rows():
    """Rows carrying side-chain nuclei: arginine and lysine, and an aliased N-epsilon.

    Every arginine N-epsilon label sits mid-way through the range the package fences on,
    except the first residue of the first few structures, which sit an interval width
    above it. Each member's error is a fraction of the offset between the members, so an
    arm's error is dominated by the offset it carries and its interval is narrow.
    """
    low, high = ARG_NE_OBSERVED_RANGE_PPM
    rows = []
    for structure in range(SIDE_CHAIN_STRUCTURES):
        for residue in range(SIDE_CHAIN_RESIDUES):
            arginine = residue < SIDE_CHAIN_ARGININES
            observed = SIDE_CHAIN_OBSERVED + ((ARG_NE_COLUMN,) if arginine else ())
            row = {"g": len(rows), "bmrb": str(100 + structure), "source": "af",
                   "struct_resnum": str(residue + 1),
                   "comp_id": "ARG" if arginine else "LYS",
                   "accession": f"P{structure:05d}", "chain": "A", "observed": observed,
                   "errors": dict.fromkeys(observed, 0.02 * ((structure + residue) % 5 - 2))}
            if arginine:
                aliased = residue == 0 and structure < SIDE_CHAIN_ALIASED
                row["labels"] = {ARG_NE_COLUMN: 2 * high - low if aliased
                                 else (low + high) / 2}
            rows.append(row)
    return rows


def observed_ppm(row, column):
    """The deposited label one cell carries: the column's own value unless the row names one."""
    return row.get("labels", {}).get(column, 50.0 + COLUMNS.index(column))


def _write_cache(run: Path, flagged_cell, rows):
    """The fold-6 cache the benchmark takes its rows from, and the row identities."""
    fold = run / "work/unified_cache/fold_6"
    structural, atom = fold / "structural", fold / "atom"
    structural.mkdir(parents=True)
    atom.mkdir(parents=True)
    maps = {name: {} for name in ("bmrb_id", "source", "materialization_id",
                                  "label_residue_id", "label_set_id", "comp_id",
                                  "accession", "chain", "model_index", "entity_id",
                                  "shift_list_id", "entity_assembly_id")}
    struct_resid = {}
    for row in rows:
        key = str(row["g"])
        maps["bmrb_id"][key] = row["bmrb"]
        maps["source"][key] = row["source"]
        maps["materialization_id"][key] = f"mz_{row['bmrb']}{row['source']}"
        maps["label_residue_id"][key] = f"lr_{row['bmrb']}_{row['struct_resnum']}"
        maps["label_set_id"][key] = f"ls_{row['bmrb']}"
        maps["comp_id"][key] = row["comp_id"]
        maps["accession"][key] = row["accession"]
        maps["chain"][key] = row["chain"]
        maps["model_index"][key] = "0"
        maps["entity_id"][key] = "1"
        maps["shift_list_id"][key] = "1"
        maps["entity_assembly_id"][key] = "1"
        struct_resid[key] = row["struct_resnum"]
    (structural / "id_maps.json").write_text(json.dumps(maps))
    (structural / "global_to_struct_resid.json").write_text(json.dumps(struct_resid))

    observed = np.zeros((len(rows), len(COLUMNS)), dtype=bool)
    for position, row in enumerate(rows):
        for column in row["observed"]:
            observed[position, COLUMNS.index(column)] = True
    np.save(structural / "shift_mask.npy", observed)
    np.save(atom / "atom_count.npy", np.full(len(rows), 40))

    # The fence the run fitted, as the flag artifact carries it: label cells by identity.
    flag_row, flag_column = flagged_cell
    flags = {maps["label_residue_id"][str(flag_row)]: [flag_column]}
    fenced = observed.copy()
    fenced[flag_row, COLUMNS.index(flag_column)] = False
    return maps, fenced, flags


def _write_authorities(run: Path, flags, rows, maps):
    data = run / "data"
    for name in ("alphafold", "pdbs", "crystal_pdbs"):
        (data / name).mkdir(parents=True)
    fold_map = data / "fold_assignments_labelset_identity90.json"
    fold_map.write_text(json.dumps(
        dict.fromkeys(dict.fromkeys(maps["label_set_id"].values()), 6)))
    for name in ("labelset_edges_uniprot.csv", "labelset_edges_same_bmrb.csv"):
        with open(data / name, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["edge_group", "label_set_id", "bmrb_id", "fold"])
    with open(data / "uniprot_mapping_candidates.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bmrb_id", "accession", "mapping_source", "source_final",
                         "source_digest_file", "source_sha256"])
        for bmrb, accession in dict.fromkeys(
                (row["bmrb"], row["accession"]) for row in rows if row["source"] == "af"):
            writer.writerow([bmrb, accession, "baseline", "True", "baseline.json", "a" * 64])
    for source in VALID_SOURCES:
        with open(data / f"structure_build_log_{source}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["materialization_id", "pdb_path", "chain_id"])
            for materialization, pdb, chain in dict.fromkeys(
                    (maps["materialization_id"][str(row["g"])], f"{row['bmrb']}.pdb",
                     row["chain"]) for row in rows if row["source"] == source):
                writer.writerow([materialization, pdb, chain])
    flags_path = run / "p100_fit/qc_flags_p100_structural_oof64_all7.json"
    flags_path.parent.mkdir(parents=True)
    flags_path.write_text(json.dumps(flags, indent=1, sort_keys=True) + "\n")
    return fold_map, flags_path


def _predictions(rows, maps, mask, offset):
    true_ppm = np.zeros((len(rows), len(COLUMNS)))
    pred_ppm = np.zeros((len(rows), len(COLUMNS)))
    for position, row in enumerate(rows):
        for column in row["observed"]:
            index = COLUMNS.index(column)
            true_ppm[position, index] = observed_ppm(row, column)
            pred_ppm[position, index] = (observed_ppm(row, column) + offset
                                         + row.get("errors", {}).get(column, 0.0))
    ambiguity = np.zeros((len(rows), len(COLUMNS)), dtype=np.int64)
    residue_code = np.asarray([STANDARD_RESIDUES.index(row["comp_id"]) for row in rows])
    resolved, effective = evaluate.resolve(pred_ppm, true_ppm, mask, ambiguity,
                                           residue_code, COLUMNS)
    keys = [f'{maps["materialization_id"][str(row["g"])]}|'
            f'{maps["label_residue_id"][str(row["g"])]}' for row in rows]
    return evaluate.Predictions(
        shift_cols=COLUMNS, pred_ppm=pred_ppm, true_ppm=true_ppm, resolved_ppm=resolved,
        mask=mask, effective_mask=effective, ambiguity=ambiguity,
        residue_code=residue_code, row_key=np.asarray(keys),
        label_set=np.asarray([maps["label_set_id"][str(row["g"])] for row in rows]),
        bmrb=np.asarray([row["bmrb"] for row in rows]),
        source=np.asarray([VALID_SOURCES.index(row["source"]) for row in rows]))


def build_run(tmp_path, *, rows=None, flagged_cell=(1, "cb_shift"), mask=None):
    """A run root the benchmark can read, and the pieces a test needs to check it."""
    run = tmp_path / "run"
    rows = _rows() if rows is None else rows
    maps, fenced, flags = _write_cache(run, flagged_cell, rows)
    fold_map, flags_path = _write_authorities(run, flags, rows, maps)
    dumps = []
    (run / "final/intervals").mkdir(parents=True)
    for member in MEMBER_FOLDS:
        path = run / f"final/intervals/fold6_member{member}.npz"
        evaluate.write_predictions(
            _predictions(rows, maps, fenced if mask is None else mask, 0.1 * member),
            path)
        dumps.append(path)
    return {"run": run, "rows": rows, "maps": maps, "fenced": fenced, "flags": flags,
            "fold_map": fold_map, "flags_path": flags_path, "dumps": dumps}


def write_competitor(root, tool, structure_id, predictions, *, mode=None, hitlog=None):
    """One arm's output for one structure instance, in the schema the scorers read."""
    directory = Path(root) / tool
    directory.mkdir(parents=True, exist_ok=True)
    record = {"status": "ok", "runtime_sec": 1.0, "predictions": predictions}
    if mode:
        record["self_exclusion"] = {"mode": mode, "max_kept_ref_identity": 0.4}
    (directory / f"{structure_id}.json").write_text(json.dumps(record))
    if hitlog is not None:
        log_dir = directory / "hitlog"
        log_dir.mkdir(exist_ok=True)
        (log_dir / f"{structure_id}.log").write_text(hitlog)


def bind(monkeypatch, built):
    """Point the benchmark modules at one built run root."""
    monkeypatch.setenv("DEEPSHIFT_RUN", str(built["run"]))
    monkeypatch.setenv("DEEPSHIFT_SIFTS_CSV", str(built["run"] / "sifts.csv"))
    monkeypatch.setenv("DEEPSHIFT_REFDB_SHIFTS", str(built["run"] / "refdb"))
    monkeypatch.setenv("DEEPSHIFT_BENCH_OUT", str(built["run"] / "benchmark"))
    from benchmarks import cells, config, score_validation

    build = config.current()
    for name, value in (("BUILD", build), ("RUN", build.run), ("CACHE", build.cache),
                        ("SD", build.structural), ("AD", build.atom),
                        ("DATA", build.data), ("MEMBERS", build.members),
                        ("FOLD_MAP", build.fold_map), ("FLAGS", build.flags),
                        ("OUT_ROOT", build.out_root), ("PDB_DIRS", build.pdb_dirs),
                        ("UNIPROT_LEDGER", build.uniprot_ledger),
                        ("SIFTS", build.sifts), ("REFDB_SHIFTS", build.refdb_shifts)):
        monkeypatch.setattr(cells, name, value)
    monkeypatch.setattr(score_validation, "OUT_ROOT", build.out_root)
    cells.load_pred_npz.cache_clear()
    cells.build_row_index.cache_clear()
    return built | {"cells": cells, "config": config,
                    "score_validation": score_validation, "build": build}
