"""Score four predictors on the Stöckelmaier tau benchmark and write ``common_cells.csv``.

The statistic is the authors': per (residue, atom), the absolute difference between the
mean over the five globular and the mean over the five stretched conformers, divided by
the pooled within-state sample standard deviation; S >= 3.29 marks a sensitive site.
Rows are kept for the sites all four predictors cover:

* DeepShift: ``deepshift_long.csv`` from ``predict_deepshift.py``.
* SHIFTX2: the authors' released predictions (``shiftX2/cs_*.dat`` in the archive).
* UCBShift 2.0 (self-excluded combined output): the CSVs ``ucbshift2_arm.py`` writes.
* MLFF (MACE, Amber-relaxed input): the JSONs ``mlff_arm.py`` writes.

    python score_tau.py --benchmark-dir <archive> --deepshift <dir>/deepshift_long.csv \\
        --ucbshift2 <dir>/ucbshift2_csv --mlff <dir>/mlff --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from deepshift.domain import AA_3_TO_1

THRESHOLD = 3.29
SEQUENCE = "SRTPSLPTPPTREPKKVAVVRTPPKSPSSAK"
GROUPS = {
    "globular": {f"tau_apo_openmm_{i}" for i in range(42690, 42695)},
    "stretched": {f"tau_apo_openmm_{i}" for i in range(44595, 44600)},
}
METHODS = ("DeepShift_original", "SHIFTX2_release", "UCBShift2_combined_selfexcl",
           "MLFF_MACE_relaxed")

# BMRB names put the within-group proton index last; the released GROMACS PDBs number the
# protons of the three-proton groups from 2 and put the number first.
METHYL_TARGETS = {
    "ALA": {"HB1", "HB2", "HB3"},
    "VAL": {"HG11", "HG12", "HG13", "HG21", "HG22", "HG23"},
    "LEU": {"HD11", "HD12", "HD13", "HD21", "HD22", "HD23"},
    "ILE": {"HG21", "HG22", "HG23", "HD11", "HD12", "HD13"},
    "THR": {"HG21", "HG22", "HG23"},
    "MET": {"HE1", "HE2", "HE3"},
}


def pdb_atoms(path: Path) -> dict[tuple[int, str], set[str]]:
    atoms: dict[tuple[int, str], set[str]] = defaultdict(set)
    with path.open() as stream:
        for line in stream:
            if line.startswith(("ATOM  ", "HETATM")):
                atoms[(int(line[22:26]), line[17:20].strip())].add(line[12:16].strip())
    return atoms


def predictor_atom_to_pdb(atom: str, residue_name: str) -> str:
    """Map a BMRB atom name to the released PDB/SHIFTX2 name."""
    atom = atom.upper()
    if len(atom) >= 4 and atom[-2:].isdigit() and atom.startswith("H"):
        # two numeric suffixes: group number, then proton number (HD11 -> 2HD1)
        return f"{int(atom[-1]) + 1}{atom[:-2]}{atom[-2]}"
    if atom in METHYL_TARGETS.get(residue_name, set()):
        return f"{int(atom[-1]) + 1}{atom[:-1]}"
    return atom


def alternate_pdb_atom(atom: str) -> str:
    """Map a one-digit BMRB prochiral name (HB2) to PDB form (2HB)."""
    atom = atom.upper()
    if atom.startswith("H") and atom[-1:].isdigit():
        return f"{atom[-1]}{atom[:-1]}"
    return atom


def deepshift_atom_table(long: pd.DataFrame, valid_atoms) -> pd.DataFrame:
    rows = []
    for row in long.itertuples(index=False):
        atom = row.column[: -len("_shift")].upper()
        residue_id, residue_name = int(row.residue_id), str(row.residue_name)
        mapped = predictor_atom_to_pdb(atom, residue_name)
        present = valid_atoms[(residue_id, residue_name)]
        if mapped not in present:
            mapped = alternate_pdb_atom(atom)
            if mapped not in present:
                continue
        rows.append({"resnum": residue_id, "resname": residue_name,
                     "atomname_deepshift": atom, "atomname_shiftx2": mapped,
                     "structure": row.structure, "group": row.group,
                     "value": float(row.value)})
    return pd.DataFrame(rows)


def sensitivity_from_long(table: pd.DataFrame, atom_column: str) -> pd.DataFrame:
    stats = (table.groupby(["resnum", "resname", atom_column, "group"])["value"]
             .agg(["mean", "std", "count"]).unstack("group"))
    n_g, n_s = stats[("count", "globular")], stats[("count", "stretched")]
    if set(n_g) != {5} or set(n_s) != {5}:
        raise SystemExit("a site does not have five conformers per state")
    pooled_var = ((n_g - 1) * stats[("std", "globular")] ** 2
                  + (n_s - 1) * stats[("std", "stretched")] ** 2) / (n_g + n_s - 2)
    pooled = np.sqrt(pooled_var)
    distance = (stats[("mean", "globular")] - stats[("mean", "stretched")]).abs()
    index = stats.index
    return pd.DataFrame({
        "resnum": index.get_level_values("resnum"),
        "resname": index.get_level_values("resname"),
        "atomname": index.get_level_values(atom_column),
        "globular_mean": stats[("mean", "globular")].to_numpy(),
        "stretched_mean": stats[("mean", "stretched")].to_numpy(),
        "globular_std": stats[("std", "globular")].to_numpy(),
        "stretched_std": stats[("std", "stretched")].to_numpy(),
        "pooled_std": pooled.to_numpy(),
        "mean_difference": distance.to_numpy(),
        "sensitivity": (distance / pooled).to_numpy(),
    })


def read_shiftx2(directory: Path) -> pd.DataFrame:
    frames = []
    for group in GROUPS:
        table = pd.read_csv(directory / f"cs_{group}.dat")
        columns = [c for c in table if c.startswith(group)]
        long = table[["resnum", "resname", "atomname", *columns]].melt(
            id_vars=["resnum", "resname", "atomname"], value_vars=columns, value_name="value")
        long["group"] = group
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def row_from_stats(method, position, residue, atom, observed) -> dict:
    """Sensitivity row from two five-element samples (UCBShift 2.0 and MLFF arms)."""
    globular = np.asarray(observed["globular"], dtype=float)
    stretched = np.asarray(observed["stretched"], dtype=float)
    g_std, s_std = float(np.std(globular, ddof=1)), float(np.std(stretched, ddof=1))
    pooled = math.sqrt((4 * g_std * g_std + 4 * s_std * s_std) / 8)
    difference = abs(float(np.mean(globular) - np.mean(stretched)))
    sensitivity = difference / pooled if pooled else (math.inf if difference else math.nan)
    return {
        "method": method, "position": position, "resnum": position + 1,
        "residue": residue, "atom": atom,
        "globular_mean": float(np.mean(globular)),
        "stretched_mean": float(np.mean(stretched)),
        "globular_std": g_std, "stretched_std": s_std,
        "pooled_within_state_sd": pooled, "between_state_difference": difference,
        "sensitivity": sensitivity, "sensitive_ge_3.29": bool(sensitivity >= THRESHOLD),
    }


def state_of(stem: str) -> str:
    return next(name for name, members in GROUPS.items() if stem in members)


def sensitivity_rows(method: str, observed: dict) -> list[dict]:
    return [row_from_stats(method, position, SEQUENCE[position - 1], atom, groups)
            for (position, atom), groups in sorted(observed.items())
            if len(groups["globular"]) == 5 and len(groups["stretched"]) == 5]


def new_observed() -> dict:
    return defaultdict(lambda: {"globular": [], "stretched": []})


def ucbshift2_rows(directory: Path) -> list[dict]:
    """The combined (``*_UCBShift``) output. Its atoms are those with a ``*_X`` column."""
    observed = new_observed()
    for stem in sorted(s for members in GROUPS.values() for s in members):
        table = pd.read_csv(directory / f"{stem}.csv")
        sequence = "".join(AA_3_TO_1[str(name).upper()] for name in table["RESNAME"])
        if sequence != SEQUENCE:
            raise SystemExit(f"UCBShift 2.0 sequence changed for {stem}: {sequence}")
        atoms = sorted(c[:-2] for c in table if c.endswith("_X"))
        for row in table.to_dict(orient="records"):
            position = int(row["RESNUM"]) - 1
            for atom in atoms:
                value = row.get(atom + "_UCBShift")
                if value is not None and math.isfinite(float(value)):
                    observed[(position, atom)][state_of(stem)].append(float(value))
    return sensitivity_rows("UCBShift2_combined_selfexcl", observed)


def mlff_rows(directory: Path) -> list[dict]:
    observed = new_observed()
    for stem in sorted(s for members in GROUPS.values() for s in members):
        record = json.loads((directory / f"{stem}.json").read_text())
        residues = sorted(record["predictions"], key=int)
        if len(residues) != 31:
            raise SystemExit(f"MLFF {stem}: {len(residues)} residues")
        for position, residue_id in enumerate(residues, 1):
            for atom, value in record["predictions"][residue_id].items():
                if atom != "resname":
                    observed[(position, atom)][state_of(stem)].append(float(value))
    return sensitivity_rows("MLFF_MACE_relaxed", observed)


def stats_rows(frame: pd.DataFrame, method: str, atom_of) -> list[dict]:
    rows = []
    for row in frame.itertuples(index=False):
        atom = atom_of(row)
        if atom is None:
            continue
        rows.append({
            "method": method, "position": int(row.resnum) - 1, "resnum": int(row.resnum),
            "residue": AA_3_TO_1[str(row.resname)], "atom": atom,
            "globular_mean": float(row.globular_mean),
            "stretched_mean": float(row.stretched_mean),
            "globular_std": float(row.globular_std),
            "stretched_std": float(row.stretched_std),
            "pooled_within_state_sd": float(row.pooled_std),
            "between_state_difference": float(row.mean_difference),
            "sensitivity": float(row.sensitivity),
            "sensitive_ge_3.29": bool(float(row.sensitivity) >= THRESHOLD),
        })
    return rows


def deepshift_and_shiftx2_rows(benchmark: Path, long_csv: Path) -> tuple[list[dict], list[dict]]:
    pdb_dir = benchmark / "PDB_files/04_freesasa"
    valid_atoms = pdb_atoms(sorted(pdb_dir.glob("*.pdb"))[0])
    long = pd.read_csv(long_csv)
    table = deepshift_atom_table(long, valid_atoms)
    deep = sensitivity_from_long(table, "atomname_deepshift")
    mapping = table[["resnum", "resname", "atomname_deepshift", "atomname_shiftx2"]
                    ].drop_duplicates()
    deep = deep.merge(mapping, left_on=["resnum", "resname", "atomname"],
                      right_on=["resnum", "resname", "atomname_deepshift"],
                      validate="one_to_one")
    shiftx2 = sensitivity_from_long(read_shiftx2(benchmark / "shiftX2"), "atomname")
    to_deepshift = {(int(r.resnum), str(r.atomname_shiftx2)): str(r.atomname)
                    for r in deep.itertuples(index=False)}
    return (stats_rows(deep, "DeepShift_original", lambda r: str(r.atomname)),
            stats_rows(shiftx2, "SHIFTX2_release",
                       lambda r: to_deepshift.get((int(r.resnum), str(r.atomname)))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--deepshift", type=Path, required=True, help="deepshift_long.csv")
    parser.add_argument("--ucbshift2", type=Path, required=True)
    parser.add_argument("--mlff", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    deep, shiftx2 = deepshift_and_shiftx2_rows(args.benchmark_dir, args.deepshift)
    methods = {"DeepShift_original": deep, "SHIFTX2_release": shiftx2,
               "UCBShift2_combined_selfexcl": ucbshift2_rows(args.ucbshift2),
               "MLFF_MACE_relaxed": mlff_rows(args.mlff)}
    common = set.intersection(*({(r["position"], r["atom"]) for r in methods[m]}
                                for m in METHODS))
    rows = [r for m in METHODS for r in methods[m] if (r["position"], r["atom"]) in common]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "common_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for method in METHODS:
        values = np.array([r["sensitivity"] for r in rows if r["method"] == method])
        print(f"{method}: {int((values >= THRESHOLD).sum())}/{len(values)} sites with "
              f"S >= {THRESHOLD} ({100 * np.mean(values >= THRESHOLD):.2f}%)")


if __name__ == "__main__":
    main()
