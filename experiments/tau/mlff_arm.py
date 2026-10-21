"""MLFF (MACE) on the ten Stöckelmaier tau conformers, from Amber-relaxed coordinates.

``relax`` removes the ACE/NME caps and every released hydrogen, lets PDBFixer add
hydrogens (pH 7) and relaxes each conformer in Amber99SB with OBC implicit solvent, every
heavy atom restrained to its released position at 10 kcal/mol/A^2. ``predict`` polishes the
hydrogens with REDUCE and runs the MLFF predictor, writing one JSON per conformer of
secondary shifts (a per-residue, per-atom random-coil constant cancels from the benchmark
statistic). ``score_tau.py`` reads those JSONs. The MLFF predictor takes relaxed input only.

    OPENMM_PYTHON relax --benchmark-dir <archive> --out <dir> [--platform OpenCL]
    MACE_PYTHON   predict --out <dir> [--device cuda]

``predict`` needs MLFF_HOME, MLFF_MODELS and REDUCE_BIN (see ``mlff_predictor.py``).
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

SEQUENCE = "SRTPSLPTPPTREPKKVAVVRTPPKSPSSAK"
STIFFNESS = 10.0     # kcal/mol/A^2
TOLERANCE = 2.39     # kcal/mol/nm
ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
STATES = {"globular": range(42690, 42695), "stretched": range(44595, 44600)}


def state_of(stem: str) -> str:
    return next(name for name, ids in STATES.items()
                if any(stem == f"tau_apo_openmm_{i}" for i in ids))


def conformers(benchmark: Path) -> list[Path]:
    paths = sorted((benchmark / "PDB_files/04_freesasa").glob("tau_apo_openmm_*.pdb"))
    if len(paths) != 10:
        raise SystemExit(f"expected ten tau conformers, found {len(paths)}")
    return paths


def standard_heavy(source: Path, destination: Path) -> None:
    """Protein heavy atoms only: caps and every released hydrogen are removed."""
    kept = []
    for line in source.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[17:20].strip().upper() not in ONE_LETTER:
            continue
        atom = line[12:16].strip().upper().lstrip("1234")
        if line[76:78].strip().upper() == "H" or atom.startswith("H"):
            continue
        kept.append(line)
    destination.write_text("\n".join([*kept, "END"]) + "\n")


def relax_one(heavy: Path, destination: Path, platform: str) -> None:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(heavy))
    fixer.findMissingResidues()
    fixer.missingResidues = {}         # complete atoms only, never residues
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    forcefield = app.ForceField("amber99sb.xml", "amber99_obc.xml")
    system = forcefield.createSystem(fixer.topology, constraints=app.HBonds)
    restraint = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter("k", STIFFNESS * unit.kilocalories_per_mole / unit.angstroms**2)
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    for atom in fixer.topology.atoms():
        if atom.element is not None and atom.element.symbol != "H":
            restraint.addParticle(atom.index,
                                  fixer.positions[atom.index].value_in_unit(unit.nanometers))
    system.addForce(restraint)
    simulation = app.Simulation(fixer.topology, system, openmm.LangevinIntegrator(0, 0.01, 0.0),
                                openmm.Platform.getPlatformByName(platform))
    simulation.context.setPositions(fixer.positions)
    simulation.minimizeEnergy(
        maxIterations=0, tolerance=TOLERANCE * unit.kilocalories_per_mole / unit.nanometer)
    positions = simulation.context.getState(getPositions=True).getPositions()
    with destination.open("w") as stream:
        app.PDBFile.writeFile(simulation.topology, positions, stream, keepIds=True)


def relax(benchmark: Path, out: Path, platform: str) -> None:
    (out / "relaxed").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tau-relax-") as work:
        for source in conformers(benchmark):
            heavy = Path(work) / source.name
            standard_heavy(source, heavy)
            relax_one(heavy, out / "relaxed" / source.name, platform)
            print(f"relaxed {source.name}", flush=True)


def secondary_shifts(predictor, run, pdb: Path, batch_size: int) -> dict:
    """{residue number: {"resname": ..., ATOM: secondary shift in ppm}}."""
    from mlff_predictor import descriptors_fast, env_path, inside
    from utils.process_data import load_struct, struct_to_dataframe

    with inside(env_path("MLFF_HOME")):
        frame = struct_to_dataframe(load_struct(str(pdb)))
        secondary = predictor.predict_cs(descriptors_fast(predictor, run, frame, batch_size))
    predictions = {}
    for res_id, shifts in secondary.items():
        names = frame[frame["res_id"] == res_id]["name"]
        cell = {"resname": str(names.iloc[0]) if len(names) else ""}
        for atom, value in shifts.items():
            if math.isfinite(float(value)):
                cell[str(atom).upper()] = float(value)
        if len(cell) > 1:
            predictions[str(int(res_id))] = cell
    return predictions


def predict(out: Path, device: str, batch_size: int) -> None:
    from mlff_predictor import build_predictor, reduce_polish

    relaxed = sorted((out / "relaxed").glob("tau_apo_openmm_*.pdb"))
    if len(relaxed) != 10:
        raise SystemExit("run `relax` first")
    (out / "mlff").mkdir(exist_ok=True)
    predictor, run = build_predictor(device)
    with tempfile.TemporaryDirectory(prefix="tau-mlff-") as work:
        for pdb in relaxed:
            polished = reduce_polish(pdb, Path(work) / f"{pdb.stem}.reduce.pdb")
            predictions = secondary_shifts(predictor, run, polished, batch_size)
            sequence = "".join(ONE_LETTER[predictions[r]["resname"]]
                               for r in sorted(predictions, key=int))
            if sequence != SEQUENCE:
                raise SystemExit(f"{pdb.name}: MLFF parsed sequence {sequence}")
            (out / "mlff" / f"{pdb.stem}.json").write_text(json.dumps(
                {"structure": pdb.stem, "group": state_of(pdb.stem),
                 "sequence": sequence, "predictions": predictions}, indent=1))
            print(f"MLFF {pdb.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    relax_parser = sub.add_parser("relax")
    relax_parser.add_argument("--benchmark-dir", type=Path, required=True)
    relax_parser.add_argument("--out", type=Path, required=True)
    relax_parser.add_argument("--platform", default="OpenCL")
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--out", type=Path, required=True)
    predict_parser.add_argument("--device", default="cuda")
    predict_parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.command == "relax":
        relax(args.benchmark_dir, args.out, args.platform)
    else:
        predict(args.out, args.device, args.batch_size)


if __name__ == "__main__":
    main()
