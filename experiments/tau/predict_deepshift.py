"""DeepShift on the ten Stöckelmaier tau conformers.

The benchmark (Stöckelmaier and Oostenbrink, PCCP 2024) holds five conformers of a
31-residue tau fragment in a globular state and five in a stretched state. All five
released members predict every conformer: the fragment's label sets sit in fold 0, which
no member trained on. Writes ``deepshift_long.csv``, one row per (structure, residue,
shift column), which ``score_tau.py`` turns into the state-separation statistic.

    python predict_deepshift.py --benchmark-dir <unzipped archive> --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deepshift.model import load_checkpoint
from deepshift.prediction import predict_pdb

MODELS = Path(__file__).resolve().parents[2] / "models"
PDB_SUBDIR = Path("PDB_files/04_freesasa")
# Deposited conformer ids: 4269x are the globular state, 4459x the stretched state.
STATES = {"4269": "globular", "4459": "stretched"}
CHAIN = "X"
N_RESIDUES = 31


def state_of(stem: str) -> str:
    for prefix, name in STATES.items():
        if prefix in stem:
            return name
    raise ValueError(f"{stem}: carries neither state id {sorted(STATES)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark-dir", type=Path, required=True,
                        help="the unpacked Stöckelmaier archive (holds PDB_files/)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=MODELS,
                        help="directory holding deepshift_fold{1..5}.pt")
    parser.add_argument("--device", default="cuda",
                        help="the reported numbers were computed on cuda (fp16 autocast)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    members = [load_checkpoint(args.models / f"deepshift_fold{k}.pt", device=args.device)
               for k in range(1, 6)]
    structures = sorted((args.benchmark_dir / PDB_SUBDIR).glob("tau_apo_openmm_*.pdb"))
    if len(structures) != 10:
        raise SystemExit(f"expected ten tau conformers, found {len(structures)}")

    rows = []
    for pdb in structures:
        frame = predict_pdb(pdb, CHAIN, members, batch_size=args.batch_size)
        if frame is None or len(frame) != N_RESIDUES:
            raise SystemExit(f"{pdb.name}: expected {N_RESIDUES} residues")
        melted = frame.drop(columns="atom_cloud_truncated").melt(
            id_vars=["residue_id", "residue_name"], var_name="column", value_name="value")
        melted["structure"] = pdb.stem
        melted["group"] = state_of(pdb.stem)
        rows.append(melted)
        print(f"{pdb.stem} ok", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.out / "deepshift_long.csv", index=False)


if __name__ == "__main__":
    main()
