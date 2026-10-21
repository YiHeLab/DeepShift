#!/usr/bin/env python3
"""DeepShift over the nine public Nagy IDP8 ensembles.

Nagy et al. (Appl Spectrosc 2024) back-calculate CA/CB/C' with SPARTA+ over each deposited
PED ensemble, average over conformers, and report the RMSD against BMRB shifts. Exactly one
thing changes here: the predictor. Each conformer is predicted by the mean, in ppm, of the
five released members (``models/``) and no re-referencing is applied, as Nagy applied none.
All nine proteins belong to the case-study hold-out (fold 0) or to no fold, so no member
trained on them.

Reads ``manifest.json`` written by ``fetch.py`` and writes one
``<out>/<label>.parquet`` per ensemble (one row per conformer residue); an ensemble whose
file exists is skipped, so an interrupted run resumes. ``score.py`` scores the result.

    python run_inference.py --data <idp8 data directory> --out <predictions directory>
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import pandas as pd

from deepshift.model import load_checkpoint
from deepshift.prediction import predict_pdb

REPO = Path(__file__).resolve().parents[2]


def split_models(path: Path, workdir: Path) -> list[Path]:
    """One file per MODEL, in deposited order; predict_pdb parses a single structure."""
    out, current, index = [], [], 0
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("MODEL "):
            current = []
        elif line.startswith("ENDMDL"):
            index += 1
            destination = workdir / f"model_{index:05d}.pdb"
            destination.write_text("".join(current) + "END\n")
            out.append(destination)
            current = []
        elif line.startswith(("ATOM", "HETATM", "TER")):
            current.append(line)
    if not out and current:
        destination = workdir / "model_00001.pdb"
        destination.write_text("".join(current) + "END\n")
        out.append(destination)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True,
                        help="directory holding manifest.json (written by fetch.py)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=REPO / "models",
                        help="directory holding deepshift_fold1.pt .. deepshift_fold5.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--labels", default="", help="comma-separated subset of ensembles")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = json.loads((args.data / "manifest.json").read_text())
    wanted = [name for name in args.labels.split(",") if name]
    ensembles = [e for e in manifest["ensembles"] if not wanted or e["label"] in wanted]
    members = [load_checkpoint(args.models_dir / f"deepshift_fold{k}.pt", device=args.device)
               for k in range(1, 6)]
    args.out.mkdir(parents=True, exist_ok=True)

    started, total_models = time.monotonic(), 0
    for entry in ensembles:
        label = entry["label"]
        destination = args.out / f"{label}.parquet"
        if destination.exists():
            print(f"{label}: complete, skipping", flush=True)
            continue
        began = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            models = split_models(args.data / entry["path"], Path(tmp))
            frames = []
            for index, model in enumerate(models, 1):
                frame = predict_pdb(model, "A", members, batch_size=args.batch_size)
                if frame is None:
                    raise RuntimeError(f"{label} model {index}: no usable residues")
                frame.insert(0, "model_index", index)
                frames.append(frame)
                if index % 100 == 0 or index == len(models):
                    rate = index / max(time.monotonic() - began, 1e-9)
                    print(f"{label} [{index:4d}/{len(models)}] {rate:.1f} models/s",
                          flush=True)
        frame = pd.concat(frames, ignore_index=True)
        frame.insert(0, "members", "1,2,3,4,5")
        frame.insert(0, "label", label)
        frame.to_parquet(destination, index=False)
        total_models += len(models)
        print(f"{label}: {len(models)} models, {len(frame):,} rows -> {destination} "
              f"{(time.monotonic() - began) / 60:.1f} min", flush=True)

    print(f"idp8 finished: {len(ensembles)} ensembles, {total_models:,} conformers, "
          f"{(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
