"""MLFF (MACE) shift predictions for the Amber-relaxed 3DRobot candidates.

The predictor takes relaxed input only, so this reads ``relax_decoys.py``'s output. Each
relaxed candidate is polished with REDUCE (hydrogens kept, flips optimised) and passed to
the MLFF predictor; its secondary shifts are converted to ppm with the predictor's own
random-coil table (``utils/get_rc_shifts``), exactly as its ``predict`` does. Writes
``<out>/<target>/<candidate>.json``, resumable per candidate.

    MACE_PYTHON mlff_decoys.py --cohort cohort.json --relaxed <dir> --out <dir> \\
        [--device cuda] [--shard 0 --nshards 2]

Needs MLFF_HOME, MLFF_MODELS and REDUCE_BIN (see ``mlff_predictor.py``).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from mlff_predictor import build_predictor, descriptors_fast, env_path, inside, reduce_polish

ATOMS = ("N", "CA", "CB", "C", "H", "HA")
ATTEMPTS = 3     # the MACE neighbour list occasionally fails a host allocation


def predict_ppm(predictor, run, pdb: Path, batch_size: int) -> dict:
    """{residue number: {"resname": ..., ATOM: shift in ppm}}."""
    from utils.get_rc_shifts import get_rc_shifts
    from utils.process_data import load_struct, struct_to_dataframe

    with inside(env_path("MLFF_HOME")):
        frame = struct_to_dataframe(load_struct(str(pdb)))
        sequence, res_ids = predictor.extract_sequence(frame)
        rc = get_rc_shifts(sequence, res_ids=res_ids, format_1letter=True).set_index("res_id")
        secondary = predictor.predict_cs(descriptors_fast(predictor, run, frame, batch_size))

    out = {}
    for res_id, per_atom in secondary.items():
        if res_id not in rc.index:
            continue
        row = rc.loc[res_id]
        names = frame[frame["res_id"] == res_id]["name"]
        cell = {"resname": str(names.iloc[0]) if len(names) else ""}
        for atom, value in per_atom.items():
            if atom not in ATOMS:
                continue
            reference = row.get(atom)
            if hasattr(reference, "iloc"):      # a duplicated res_id gives a Series
                reference = reference.iloc[0]
            if reference is None or reference != reference:     # NaN: no reference
                continue
            cell[atom] = float(value) + float(reference)
        if len(cell) > 1:
            out[str(res_id)] = cell
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--relaxed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", default="", help="comma-separated subset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()

    entries = {t["target"]: t for t in json.loads(args.cohort.read_text())["targets"]}
    chosen = [t for t in args.targets.split(",") if t] or sorted(entries)
    predictor, run = build_predictor(args.device)
    failed = []
    with tempfile.TemporaryDirectory(prefix="mlff-decoys-") as work:
        for target in sorted(chosen)[args.shard::args.nshards]:
            for candidate in entries[target]["candidates"]:
                stem = Path(candidate["name"]).stem
                destination = args.out / target / f"{stem}.json"
                if destination.exists() and destination.stat().st_size:
                    continue
                relaxed = args.relaxed / target / candidate["name"]
                if not relaxed.exists():
                    raise SystemExit(f"no relaxed structure at {relaxed}; the predictor "
                                     "takes relaxed input only")
                for attempt in range(ATTEMPTS):
                    try:
                        polished = reduce_polish(relaxed, Path(work) / f"{target}_{stem}.pdb")
                        predictions = predict_ppm(predictor, run, polished, args.batch_size)
                        break
                    except Exception as error:
                        if attempt == ATTEMPTS - 1:
                            failed.append(f"{target}/{stem}: {type(error).__name__}: {error}")
                else:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(
                    {"target": target, "candidate": candidate["name"],
                     "predictions": predictions}))
            print(f"{target}: done", flush=True)
    if failed:
        raise SystemExit("failed candidates:\n" + "\n".join(failed))


if __name__ == "__main__":
    main()
