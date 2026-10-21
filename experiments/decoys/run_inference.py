"""DeepShift over the 3DRobot pool: 52 targets x 301 candidates, one out-of-fold member each.

``build_cohort.py`` names, per target, the released member that never saw it. Each
candidate (the native structure and 300 decoys) is predicted by that member alone, on its
chain of the target. Writes ``<out>/<target>.parquet``: one row per (candidate, residue)
with the target, the member, the candidate's name and C-alpha RMSD, and the shift columns
in ppm. A target whose file already holds every candidate for its member is skipped.

    python run_inference.py --cohort cohort.json --candidates <3DRobot targets> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from deepshift.model import load_checkpoint
from deepshift.prediction import predict_pdb

MODELS = Path(__file__).resolve().parents[2] / "models"


def complete(path: Path, member: int, n_candidates: int) -> bool:
    if not path.is_file():
        return False
    frame = pd.read_parquet(path, columns=["members", "candidate"])
    return set(frame.members.astype(str)) == {str(member)} and frame.candidate.nunique() == n_candidates


def chain_of(pdb: Path, preferred: str) -> str:
    """The target's chain when the file has it, else its only chain."""
    present = []
    for line in pdb.read_text(errors="replace").splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 22:
            chain = line[21].strip()
            if chain not in present:
                present.append(chain)
    if preferred in present:
        return preferred
    if len(present) == 1:
        return present[0]
    raise SystemExit(f"{pdb}: chain {preferred!r} absent; chains present {present}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--targets", default="", help="comma-separated subset")
    parser.add_argument("--device", default="cuda",
                        help="the reported numbers were computed on cuda (fp16 autocast)")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    entries = {t["target"]: t for t in json.loads(args.cohort.read_text())["targets"]}
    selected = [t for t in args.targets.split(",") if t] or sorted(entries)
    members = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for position, target in enumerate(selected, 1):
        entry = entries[target]
        member = entry["single_member"]
        destination = args.out / f"{target}.parquet"
        if complete(destination, member, len(entry["candidates"])):
            print(f"{target}: complete, skipping", flush=True)
            continue
        if member not in members:
            members[member] = load_checkpoint(
                args.models / f"deepshift_fold{member}.pt", device=args.device)
        frames = []
        for candidate in entry["candidates"]:
            path = args.candidates / target / candidate["name"]
            frame = predict_pdb(path, chain_of(path, entry["chain"]), [members[member]],
                                batch_size=args.batch_size)
            if frame is None or frame.empty:
                raise SystemExit(f"{target}/{candidate['name']}: no usable prediction")
            frame.insert(0, "rmsd", float(candidate["rmsd"]))
            frame.insert(0, "candidate", candidate["name"])
            frame.insert(0, "n_members", 1)
            frame.insert(0, "members", str(member))
            frame.insert(0, "target", target)
            frames.append(frame)
        temporary = destination.with_suffix(".parquet.tmp")
        pd.concat(frames, ignore_index=True).to_parquet(temporary, index=False,
                                                        compression="zstd")
        os.replace(temporary, destination)
        print(f"[{position}/{len(selected)}] {target}: member {member}", flush=True)


if __name__ == "__main__":
    main()
