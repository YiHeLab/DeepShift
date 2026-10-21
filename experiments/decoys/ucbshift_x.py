"""UCBShift (original, Li et al. 2020) on every 3DRobot candidate; the ``*_X`` columns are scored.

Each candidate is stripped of hydrogens (its chain set to A), hydrogenated with REDUCE, and
passed to ``CSpred.py --batch --test`` (test mode excludes same-protein transfer references;
only the structure-only ``H_X``, ``HA_X`` and ``N_X`` columns are scored, so transfer
plays no part). One CSpred process per target, so the trained models load once per target.
Writes ``<out>/<target>/<candidate>.csv``; resumable per target.

    UCBSHIFT_HOME    checkout of THGLab/CSpred (UCBShift 1.0) with its models and bins
    UCBSHIFT_PYTHON  Python of the UCBShift environment
    REDUCE_BIN       REDUCE executable

    python ucbshift_x.py --cohort cohort.json --candidates <3DRobot targets> --out <dir> \\
        [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"set {name}")
    return Path(value)


def strip_hydrogens(source: Path, destination: Path) -> None:
    with source.open() as reader, destination.open("w") as writer:
        for line in reader:
            if line.startswith(("ATOM", "HETATM")):
                name = line[12:16].strip()
                if name[:1] == "H" or (len(name) > 1 and name[0].isdigit() and name[1] == "H"):
                    continue
                line = line[:21] + "A" + line[22:]
            writer.write(line)


def run_target(directory: Path, hydrogenated: Path, out: Path) -> str:
    target = directory.name
    candidates = sorted(directory.glob("*.pdb"))
    result_dir = out / target
    result_dir.mkdir(parents=True, exist_ok=True)
    if len(list(result_dir.glob("*.csv"))) >= len(candidates):
        return f"{target}: complete"
    hydrogen_dir = hydrogenated / target
    hydrogen_dir.mkdir(parents=True, exist_ok=True)
    reduce_bin, home, python = (env_path("REDUCE_BIN"), env_path("UCBSHIFT_HOME"),
                                env_path("UCBSHIFT_PYTHON"))
    for pdb in candidates:
        target_file = hydrogen_dir / pdb.name
        if target_file.exists() and target_file.stat().st_size:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".pdb") as stripped:
            strip_hydrogens(pdb, Path(stripped.name))
            with target_file.open("w") as stream:
                subprocess.run([str(reduce_bin), stripped.name, "-Quiet"], stdout=stream,
                               stderr=subprocess.DEVNULL, check=False)
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as listing:
        listing.write("\n".join(str(p) for p in sorted(hydrogen_dir.glob("*.pdb"))) + "\n")
        listing.flush()
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           MKL_NUM_THREADS="1",
                           PATH=os.pathsep.join([str(python.parent), os.environ["PATH"]]))
        log = result_dir / "_run.log"
        with log.open("w") as stream:
            subprocess.run([str(python), "CSpred.py", "--batch", "--test", "-o",
                            str(result_dir), listing.name],
                           cwd=home, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                           check=False)
    done = len(list(result_dir.glob("*.csv")))
    return f"{target}: {done}/{len(candidates)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    hydrogenated = args.out.parent / (args.out.name + "_hydrogenated")
    targets = sorted(t["target"] for t in json.loads(args.cohort.read_text())["targets"])
    directories = [args.candidates / target for target in targets]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for message in pool.map(lambda d: run_target(d, hydrogenated, args.out), directories):
            print(message, flush=True)


if __name__ == "__main__":
    main()
