"""UCBShift 2.0 on the ten Stöckelmaier tau conformers.

``prepare`` strips the ACE/NME caps and the released hydrogens from each conformer and
rebuilds hydrogens with REDUCE; ``run`` runs UCBShift 2.0's ``CSpred.py`` in test mode
(``-t``, transfer references at or above 90 % sequence identity excluded) and keeps one CSV
per conformer. ``score_tau.py`` reads the ``*_UCBShift`` columns.

Needs a UCBShift 2.0 checkout with ``../competitors/patches/ucbshift2.patch`` applied (see ``README.md``):

    UCBSHIFT2_HOME  checkout of THGLab/CSpred, 2.0 branch, with bins/ installed
    UCBSHIFT2_MODELS  directory of the 141 downloaded ``.sav`` models
    UCBSHIFT_PYTHON  Python of the UCBShift environment
    REDUCE_BIN      REDUCE executable

    python ucbshift2_arm.py prepare --benchmark-dir <archive> --out <dir>
    python ucbshift2_arm.py run --benchmark-dir <archive> --out <dir>
"""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from deepshift.domain import AA_3_TO_1

SEQUENCE = "SRTPSLPTPPTREPKKVAVVRTPPKSPSSAK"
SELF_IDENT_CUTOFF = "0.90"


def env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"set {name}")
    return Path(value)


def conformers(benchmark: Path) -> list[Path]:
    paths = sorted((benchmark / "PDB_files/04_freesasa").glob("tau_apo_openmm_*.pdb"))
    if len(paths) != 10:
        raise SystemExit(f"expected ten tau conformers, found {len(paths)}")
    return paths


def is_hydrogen(line: str) -> bool:
    atom = line[12:16].strip().upper().lstrip("1234")
    return line[76:78].strip().upper() == "H" or atom.startswith("H")


def standard_heavy_lines(source: Path) -> list[str]:
    """Standard-residue heavy-atom records; caps and hydrogens (GROMACS-numbered) go."""
    kept = []
    for line in source.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[17:20].strip().upper() not in AA_3_TO_1 or is_hydrogen(line):
            continue
        kept.append("ATOM  " + line[6:])
    return kept


def sequence_and_numbers(path: Path) -> tuple[str, list[int]]:
    seen, residues = set(), []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and line[17:20].strip().upper() in AA_3_TO_1:
            key = (line[21:22], int(line[22:26]), line[26:27], line[17:20].strip().upper())
            if key not in seen:
                seen.add(key)
                residues.append(key)
    return "".join(AA_3_TO_1[r[3]] for r in residues), [r[1] for r in residues]


def prepare(benchmark: Path, out: Path) -> None:
    reduce_bin = env_path("REDUCE_BIN")
    (out / "standard_heavy").mkdir(parents=True, exist_ok=True)
    (out / "prepared_reduce").mkdir(parents=True, exist_ok=True)
    for source in conformers(benchmark):
        heavy = out / "standard_heavy" / source.name
        prepared = out / "prepared_reduce" / source.name
        heavy.write_text("\n".join([*standard_heavy_lines(source), "END"]) + "\n")
        with prepared.open("w") as stream:
            proc = subprocess.run([str(reduce_bin), "-build", "-Quiet", str(heavy)],
                                  stdout=stream, stderr=subprocess.PIPE, text=True)
        if proc.returncode or not prepared.stat().st_size:
            raise SystemExit(f"REDUCE failed for {source.name}: {proc.stderr}")
        sequence, numbers = sequence_and_numbers(prepared)
        if sequence != SEQUENCE or numbers != list(range(2, 33)):
            raise SystemExit(f"{source.name}: prepared sequence or numbering changed")
        print(f"prepared {source.name}", flush=True)


def run_one(source: Path, out: Path) -> str:
    prepared = out / "prepared_reduce" / source.name
    output = out / "ucbshift2_csv" / f"{source.stem}.csv"
    if output.exists() and output.stat().st_size:
        return f"skipped {source.stem}"
    home, python = env_path("UCBSHIFT2_HOME"), env_path("UCBSHIFT_PYTHON")
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join([
        str(python.parent), str(home / "bins/ncbi-blast-2.9.0+/bin"),
        str(home / "bins/mTM-align"), environment.get("PATH", "")])
    environment["UCB_SELF_IDENT_CUTOFF"] = SELF_IDENT_CUTOFF
    command = [str(python), "CSpred.py", str(prepared), "-o", str(output),
               "--models", str(env_path("UCBSHIFT2_MODELS")) + "/", "-t"]
    proc = subprocess.run(command, cwd=home, env=environment, capture_output=True,
                          text=True, timeout=3600)
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / f"{source.stem}.log").write_text(
        proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    if proc.returncode or not output.exists() or not output.stat().st_size:
        output.unlink(missing_ok=True)
        raise SystemExit(f"UCBShift 2.0 failed on {source.stem}; see logs/")
    table = pd.read_csv(output)
    if len(table) != 31 or table["RESNUM"].astype(int).tolist() != list(range(2, 33)):
        output.unlink()
        raise SystemExit(f"{source.stem}: unexpected residues in the UCBShift 2.0 output")
    return f"ok {source.stem}"


def run(benchmark: Path, out: Path, workers: int) -> None:
    if not all((out / "prepared_reduce" / p.name).exists() for p in conformers(benchmark)):
        raise SystemExit("run `prepare` first")
    (out / "ucbshift2_csv").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, source, out) for source in conformers(benchmark)]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.benchmark_dir, args.out)
    else:
        run(args.benchmark_dir, args.out, args.workers)


if __name__ == "__main__":
    main()
