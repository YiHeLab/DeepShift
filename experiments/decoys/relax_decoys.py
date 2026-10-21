"""Amber-relax the 3DRobot candidates (52 targets x 301) for the MLFF predictor.

PDBFixer completes missing atoms (never missing residues) and adds hydrogens at pH 7, then
Amber99SB with OBC implicit solvent minimises the structure with every heavy atom
restrained to its input position at 10 kcal/mol/A^2. The 301 candidates of a target share
chain, residues and atom names, so one OpenMM System and Context is built per target and
reused, the restraint centres rewritten per candidate; a candidate whose (chain, residue,
atom) signature differs rebuilds it. Writes ``<out>/<target>/<candidate>.pdb``; resumable
per candidate.

    OPENMM_PYTHON relax_decoys.py --cohort cohort.json --candidates <3DRobot targets> \\
        --out <dir> [--platform OpenCL] [--shard 0 --nshards 4]
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

STIFFNESS = 10.0     # kcal/mol/A^2
TOLERANCE = 2.39     # kcal/mol/nm


def fix(pdb: Path):
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(pdb))
    fixer.findMissingResidues()
    fixer.missingResidues = {}         # complete atoms only, never residues
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    return fixer


def signature(topology) -> tuple:
    return tuple((chain.id, residue.id, residue.name, atom.name)
                 for chain in topology.chains()
                 for residue in chain.residues() for atom in residue.atoms())


class Engine:
    """One OpenMM System and Context, rebuilt only when the topology changes."""

    def __init__(self, platform: str):
        import openmm
        import openmm.app as app

        self.forcefield = app.ForceField("amber99sb.xml", "amber99_obc.xml")
        self.platform = openmm.Platform.getPlatformByName(platform)
        self.signature = None
        self.simulation = None

    def build(self, fixer) -> None:
        import openmm
        import openmm.app as app
        import openmm.unit as unit

        self.dispose()
        system = self.forcefield.createSystem(fixer.topology, constraints=app.HBonds)
        force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", STIFFNESS * unit.kilocalories_per_mole / unit.angstroms**2)
        for name in ("x0", "y0", "z0"):
            force.addPerParticleParameter(name)
        self.heavy = [a.index for a in fixer.topology.atoms()
                      if a.element is not None and a.element.symbol != "H"]
        for index in self.heavy:
            force.addParticle(index, fixer.positions[index].value_in_unit(unit.nanometers))
        system.addForce(force)
        self.simulation = app.Simulation(
            fixer.topology, system, openmm.LangevinIntegrator(0, 0.01, 0.0), self.platform)
        self.force = force
        self.signature = signature(fixer.topology)

    def dispose(self) -> None:
        # An OpenCL context lives until its Python object is collected; without the
        # explicit collection a run exhausts the driver's contexts.
        if self.simulation is not None:
            del self.simulation.context
            del self.simulation
            self.simulation = None
            gc.collect()
        self.signature = None

    def relax(self, fixer, destination: Path) -> None:
        import openmm.app as app
        import openmm.unit as unit

        if self.simulation is None or signature(fixer.topology) != self.signature:
            self.build(fixer)
        else:
            # The restraint centres are this candidate's own heavy-atom positions.
            for slot, index in enumerate(self.heavy):
                self.force.setParticleParameters(
                    slot, index, fixer.positions[index].value_in_unit(unit.nanometers))
            self.force.updateParametersInContext(self.simulation.context)
        self.simulation.context.setPositions(fixer.positions)
        self.simulation.minimizeEnergy(
            maxIterations=0, tolerance=TOLERANCE * unit.kilocalories_per_mole / unit.nanometer)
        positions = self.simulation.context.getState(getPositions=True).getPositions()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pdb.part")
        with temporary.open("w") as stream:
            app.PDBFile.writeFile(self.simulation.topology, positions, stream, keepIds=True)
        temporary.replace(destination)     # a partial file is never taken as finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--platform", default="OpenCL")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--targets", default="", help="comma-separated subset")
    args = parser.parse_args()

    entries = {t["target"]: t for t in json.loads(args.cohort.read_text())["targets"]}
    chosen = [t for t in args.targets.split(",") if t] or sorted(entries)
    engine, failed = Engine(args.platform), []
    for target in sorted(chosen)[args.shard::args.nshards]:
        for candidate in entries[target]["candidates"]:
            destination = args.out / target / candidate["name"]
            if destination.exists() and destination.stat().st_size:
                continue
            try:
                engine.relax(fix(args.candidates / target / candidate["name"]), destination)
            except Exception as error:    
                engine.dispose()            # a failed context is never reused
                failed.append(f"{target}/{candidate['name']}: {type(error).__name__}: {error}")
        print(f"{target}: relaxed", flush=True)
    engine.dispose()
    if failed:
        raise SystemExit("failed candidates:\n" + "\n".join(failed))


if __name__ == "__main__":
    main()
