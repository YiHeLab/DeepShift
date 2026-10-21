"""Resolve the 52 3DRobot targets onto the released folds and pick one scoring member each.

A member k never saw a target when every label set of the target's BMRB entry sits in
fold k, 0 or 6, or has no fold (members train on folds 1-5 except their own). A target in
fold k is therefore predicted by member k alone, and a target in fold 0, fold 6 or without
a fold by any of the five. One out-of-fold member is chosen per target, deterministically
from the target name, so that all 52 targets are scored by one single-member predictor.
The scored label set is the one that member held out (or the only mapped one). Candidates
are each target's native structure plus 300 decoys, with the C-alpha RMSD from 3DRobot's
``list.txt``.

    python build_cohort.py --candidates <3DRobot targets> --out cohort.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SHIFTS = HERE.parents[1] / "data" / "deepshift_shifts.parquet"
MEMBERS = (1, 2, 3, 4, 5)
NEUTRAL_FOLDS = {0, 6, None}
N_CANDIDATES = 301


def candidates(directory: Path) -> list[dict]:
    """The native structure first, then the decoys, each with its C-alpha RMSD."""
    rmsd = {"native.pdb": 0.0}
    for line in (directory / "list.txt").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].endswith(".pdb"):
            rmsd[fields[0]] = float(fields[1])
    on_disk = {path.name for path in directory.glob("*.pdb")}
    if len(on_disk) != N_CANDIDATES or set(rmsd) != on_disk:
        raise SystemExit(f"{directory.name}: {len(on_disk)} PDBs on disk, {len(rmsd)} listed; "
                         f"expected the same {N_CANDIDATES}")
    ordered = ["native.pdb", *sorted(on_disk - {"native.pdb"})]
    return [{"name": name, "rmsd": rmsd[name]} for name in ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidates", type=Path, required=True,
                        help="directory with one folder per target (3DRobot_set, unpacked)")
    parser.add_argument("--roster", type=Path, default=HERE / "roster.tsv")
    parser.add_argument("--shifts", type=Path, default=SHIFTS)
    parser.add_argument("--out", type=Path, default=HERE / "cohort.json")
    args = parser.parse_args()

    label_sets = pd.read_parquet(args.shifts, columns=["bmrb_id", "label_set_id", "fold"])
    label_sets = label_sets.drop_duplicates().astype({"bmrb_id": str})
    with args.roster.open() as stream:
        roster = list(csv.DictReader(stream, delimiter="\t"))

    targets = []
    for entry in sorted(roster, key=lambda e: e["target"]):
        rows = label_sets[label_sets.bmrb_id == entry["bmrb"]]
        folds = {ls: (None if pd.isna(f) else int(f))
                 for ls, f in zip(rows.label_set_id, rows.fold, strict=True)}
        blocking = {f for f in folds.values() if f not in NEUTRAL_FOLDS}
        members = [k for k in MEMBERS if not blocking - {k}]
        if not members:
            raise SystemExit(f"{entry['target']}: no member is out of fold")
        digest = int(hashlib.sha256(entry["target"].encode()).hexdigest(), 16)
        single = members[digest % len(members)]
        mapped = [ls for ls, f in folds.items() if f is not None]
        pick = [ls for ls in mapped if folds[ls] == single] or mapped or list(folds)
        if len(pick) != 1:
            raise SystemExit(f"{entry['target']}: scored label set is ambiguous: {pick}")
        targets.append({
            "target": entry["target"], "bmrb": entry["bmrb"], "chain": entry["chain"],
            "label_sets": folds, "scored_label_set_id": pick[0],
            "members": members, "single_member": single,
            "clean_for_all_members": len(members) == len(MEMBERS),
            "candidates": candidates(args.candidates / entry["target"]),
        })
    args.out.write_text(json.dumps({"targets": targets}, indent=1) + "\n")
    print(f"{len(targets)} targets; single-member counts "
          f"{ {k: sum(t['single_member'] == k for t in targets) for k in MEMBERS} }")


if __name__ == "__main__":
    main()
