#!/usr/bin/env python3
"""DeepShift back-calculated chemical-shift tables for the X-EISD drkN SH3 pools.

X-EISD (Lincoff et al., Commun. Chem. 2020) consumes pre-computed back-calculated
observables, one file per observable, so swapping the chemical-shift forward model means
regenerating exactly one file and leaving the other seven untouched. The ensemble, the
experimental data, the error model, the Bayesian scoring and the optimizer stay theirs.

Their file is a whitespace table with no header: one row per structure, first column the
structure's source index, then 262 columns in the row order of ``drksh3_CS_exp_mod.txt``
(one per resnum/atom-name experimental observation over residues 3-57 and atoms
C/CA/CB/H/HA). This writes a drop-in replacement of the same shape and column order.
Every row is the mean, in ppm, of the five released members (``models/``): drkN SH3 is
BMRB 25500, which belongs to the case-study hold-out (fold 0) that no member trains on.

Pools (``--pool``):
  ensemble       the 1,700-conformer PED deposit, read from its tarball
  random-mixed   only the 1,700 RANDOM conformers the paper's MIXED pool embeds
  random         all 100,000 RANDOM conformers (hours; resumable)

MIXED is then assembled on the CPU by ``build_mixed.py``.

    XEISD_HOME=<checkout> python run_shift_tables.py --pool ensemble --out-dir <dir>
"""
from __future__ import annotations

import argparse
import bz2
import csv
import tarfile
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from pools import N_ENSEMBLE, N_RANDOM, mixed_random_indices, xeisd_home

from deepshift.model import load_checkpoint
from deepshift.prediction import predict_pdb

REPO = Path(__file__).resolve().parents[2]
N_OBSERVATIONS = 262
ATOM_TO_COLUMN = {"C": "c_shift", "CA": "ca_shift", "CB": "cb_shift",
                  "H": "h_shift", "HA": "ha_shift"}
POOLS = {
    "ensemble": ("drksh3_ENSEMBLE_CS_deepshift.txt", N_ENSEMBLE),
    "random-mixed": ("drksh3_RANDOM_MIXEDROWS_CS_deepshift.txt", N_ENSEMBLE),
    "random": ("drksh3_RANDOM_CS_deepshift.txt", N_RANDOM),
}


def observation_order(path: Path) -> list[tuple[int, str]]:
    """The 262 (resnum, shift column) pairs, in the row order their scorer expects."""
    with path.open(newline="") as stream:
        order = [(int(row["resnum"]), ATOM_TO_COLUMN[row["atomname"]])
                 for row in csv.DictReader(stream)]
    if len(order) != N_OBSERVATIONS:
        raise ValueError(f"expected {N_OBSERVATIONS} CS observations, found {len(order)}")
    return order


def models_from_tar(path: Path) -> Iterator[tuple[int, str]]:
    """Tar members in basename order, the order the paper's row indices assume."""
    with tarfile.open(path, "r:*") as archive:
        members = sorted((m for m in archive.getmembers()
                          if m.isfile() and m.name.lower().endswith(".pdb")),
                         key=lambda m: Path(m.name).name)
        for index, member in enumerate(members):
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"could not read {member.name} from {path}")
            yield index, handle.read().decode("ascii", "replace")


def models_from_bz2(path: Path) -> Iterator[tuple[int, str]]:
    """Sequentially numbered MODELs of the compressed stream, index = number - 1."""
    current, expected = None, 1
    with bz2.open(path, "rt", encoding="ascii", errors="replace") as stream:
        for line in stream:
            record = line[:6].strip()
            if record == "MODEL":
                current = []
                continue
            if current is None:
                continue
            if record == "ENDMDL":
                yield expected - 1, "".join(current)
                expected += 1
                current = None
                continue
            current.append(line)
    if current is not None:
        raise ValueError("truncated final MODEL in the RANDOM stream")


def row_values(frame, order) -> list[float]:
    """One table row: the 262 observations, in their order, from one prediction.

    Every observation must land: a missing residue or a blanked cell would shift every
    later column by one and silently misalign the table with their scorer.
    """
    position = {int(resid): i for i, resid in enumerate(frame.residue_id)}
    values = []
    for resnum, column in order:
        if resnum not in position:
            raise RuntimeError(f"prediction lacks experimental residue {resnum}")
        value = float(frame[column].iloc[position[resnum]])
        if not np.isfinite(value):
            raise RuntimeError(f"non-finite {column} at residue {resnum}")
        values.append(value)
    return values


def written_indices(path: Path) -> set[int]:
    """Indices already in a partial table, so a restart resumes rather than repeats."""
    if not path.exists():
        return set()
    return {int(line.split(maxsplit=1)[0]) for line in path.read_text().splitlines()
            if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", choices=sorted(POOLS), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=REPO / "models",
                        help="directory holding deepshift_fold1.pt .. deepshift_fold5.pt")
    parser.add_argument("--random-stream", type=Path,
                        help="the RANDOM pool's bz2 PDB stream (default: "
                             "$XEISD_HOME/data/structures/Trades/all_structs_cat.pdb.bz2)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    xeisd = xeisd_home()
    name, expected = POOLS[args.pool]
    if args.pool == "ensemble":
        source = models_from_tar(xeisd / "data/structures/ENSEMBLE_PED/PED8AAC-pdb.tar.gz")
    else:
        source = models_from_bz2(args.random_stream or
                                 xeisd / "data/structures/Trades/all_structs_cat.pdb.bz2")
    wanted = set(mixed_random_indices(xeisd)) if args.pool == "random-mixed" else None

    order = observation_order(xeisd / "data/experimental_data/drksh3_CS_exp_mod.txt")
    members = [load_checkpoint(args.models_dir / f"deepshift_fold{k}.pt", device=args.device)
               for k in range(1, 6)]
    available = set(members[0].shift_columns)
    absent = sorted({column for _, column in order} - available)
    if absent:
        raise ValueError(f"checkpoint lacks required columns: {absent}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / name
    part = destination.with_name(destination.name + ".part")
    done = written_indices(part)
    started, written = time.monotonic(), len(done)

    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "conformer.pdb"
        with part.open("a") as output:
            for index, text in source:
                if wanted is not None and index not in wanted:
                    continue
                if index in done:
                    continue
                scratch.write_text(text)
                # Chain "_" keeps every chain: the conformers are single-chain, unnamed.
                frame = predict_pdb(scratch, "_", members, batch_size=args.batch_size)
                if frame is None:
                    raise RuntimeError(f"conformer {index}: no usable residues")
                output.write(f"{index} "
                             + " ".join(f"{v:.4f}" for v in row_values(frame, order))
                             + "\n")
                written += 1
                if written % 200 == 0:
                    output.flush()
                    rate = (written - len(done)) / max(time.monotonic() - started, 1e-9)
                    print(f"{args.pool} {written}/{expected} rows; {rate:.2f} conformers/s; "
                          f"ETA {(expected - written) / max(rate, 1e-9) / 60:.1f} min",
                          flush=True)
                if written >= expected:
                    break

    if written != expected:
        raise RuntimeError(f"{args.pool}: wrote {written} rows, expected {expected}")
    # The scorer indexes structures by row order, so the table is promoted only once
    # every row exists and the rows are sorted back into source order.
    rows = sorted((int(line.split(maxsplit=1)[0]), line)
                  for line in part.read_text().splitlines() if line.strip())
    destination.write_text("".join(line + "\n" for _, line in rows))
    part.unlink()
    print(f"{args.pool}: {len(rows)} rows x {N_OBSERVATIONS} observations -> {destination} "
          f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
