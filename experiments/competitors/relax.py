"""Amber99SB relaxation of the fold-6 chains: the input the MLFF arm is run on.

PDBFixer completes missing atoms (never missing residues) and adds hydrogens at pH 7, then
OpenMM minimises in implicit solvent (``amber99sb.xml`` + ``amber99_obc.xml``) with every
heavy atom restrained to its input position at 10 kcal/mol/A^2, so the chain is relaxed
rather than refolded. Ligands, ions and water have no Amber99 template: they are left out of
the minimisation and their original records are appended to the output unchanged.

A chain that PDBFixer completes but the force field has no template for is retried once with
the deposited atoms no Amber99SB template admits removed (rules R1-R4 in ``repair``).

Reads ``<chains>/<structure_id>.pdb`` (extracted from the task's structure when absent) and
writes ``<out>/<structure_id>.pdb``. Run in an interpreter that has OpenMM and PDBFixer:

    python experiments/competitors/relax.py --chains CHAINS --out RELAXED
"""
import argparse
import csv
import math
import os
import tempfile
import time
from pathlib import Path

import openmm
import openmm.app as app
import openmm.unit as unit
from common import extract_chain
from pdbfixer import PDBFixer

STIFFNESS = 10.0   # kcal/mol/A^2
TOLERANCE = 2.39   # kcal/mol/nm
WATER_ION = {'HOH', 'WAT', 'DOD'}


def relax(pdb_in, pdb_out, platform_name=None):
    """Minimise ``pdb_in`` with restrained heavy atoms and write the result to ``pdb_out``."""
    heteroatoms = [line for line in Path(pdb_in).read_text().splitlines()
                   if line.startswith('HETATM')]

    # The order is load-bearing: heterogens are removed and nonstandard residues replaced
    # before missing atoms are sought, or a standard residue reaches the force field with an
    # incomplete topology and has no template. Missing residues are never built, so the
    # relaxed chain keeps exactly the residues that were deposited.
    fixer = PDBFixer(filename=str(pdb_in))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    forcefield = app.ForceField('amber99sb.xml', 'amber99_obc.xml')
    system = forcefield.createSystem(fixer.topology, constraints=app.HBonds)
    force = openmm.CustomExternalForce('0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
    force.addGlobalParameter('k', STIFFNESS * unit.kilocalories_per_mole / unit.angstroms ** 2)
    for name in ('x0', 'y0', 'z0'):
        force.addPerParticleParameter(name)
    for atom in fixer.topology.atoms():
        if atom.element is not None and atom.element.symbol != 'H':
            force.addParticle(atom.index,
                              fixer.positions[atom.index].value_in_unit(unit.nanometers))
    system.addForce(force)

    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    platform = (openmm.Platform.getPlatformByName(platform_name),) if platform_name else ()
    simulation = app.Simulation(fixer.topology, system, integrator, *platform)
    simulation.context.setPositions(fixer.positions)
    simulation.minimizeEnergy(
        maxIterations=0,
        tolerance=TOLERANCE * unit.kilocalories_per_mole / unit.nanometer)
    state = simulation.context.getState(getPositions=True)

    partial = Path(pdb_out).with_suffix('.pdb.part')
    with open(partial, 'w') as handle:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), handle, keepIds=True)
    if heteroatoms:
        body = [ln for ln in partial.read_text().splitlines() if ln.strip() != 'END']
        partial.write_text('\n'.join([*body, *heteroatoms, 'END']) + '\n')
    partial.replace(pdb_out)           # a partial file is never taken for a finished one


def repair(lines):
    """The chain's PDB lines without the deposited atoms no Amber99SB template admits.

    R1  CYS whose SG lies within 2.5 A of another CYS SG: HG removed (disulfide)
    R2  C-terminal residue: HXT removed (Amber C-termini are carboxylate)
    R3  N-terminal residue carrying H1/H2/H3: backbone H removed (NH3+ has three H)
    R4  N-terminal GLU HE2 / ASP HD2 removed (no N-terminal GLH / ASH template)
    """
    def key(line):
        return line[21], line[22:27]       # chain, residue number + insertion code

    by_res, order = {}, []
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            if key(line) not in by_res:
                by_res[key(line)] = []
                order.append(key(line))
            by_res[key(line)].append(line)
    names = {k: {ln[12:16].strip() for ln in v} for k, v in by_res.items()}
    resname = {k: v[0][17:20] for k, v in by_res.items()}
    # Polymer residues: every ATOM residue, plus HETATM residues PDBFixer converts (they
    # carry backbone N/CA/C), never waters or ligands.
    poly = [k for k in order if resname[k] not in WATER_ION
            and ({'N', 'CA', 'C'} <= names[k] or by_res[k][0].startswith('ATOM'))]
    first, last = poly[0], poly[-1]
    drop = set()

    def remove(k, atom):
        for i, line in enumerate(lines):
            if line.startswith(('ATOM', 'HETATM')) and key(line) == k \
                    and line[12:16].strip() == atom:
                drop.add(i)

    sulfur = {k: tuple(float(ln[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
              for k in poly if resname[k] == 'CYS'
              for ln in by_res[k] if ln[12:16].strip() == 'SG'}
    cysteines = list(sulfur)
    for i, ki in enumerate(cysteines):
        for kj in cysteines[i + 1:]:
            if math.dist(sulfur[ki], sulfur[kj]) <= 2.5:
                for k in (ki, kj):
                    if 'HG' in names[k]:
                        remove(k, 'HG')
    if 'HXT' in names[last]:
        remove(last, 'HXT')
    if 'H' in names[first] and {'H1', 'H2', 'H3'} & names[first]:
        remove(first, 'H')
    if resname[first] == 'GLU' and 'HE2' in names[first]:
        remove(first, 'HE2')
    if resname[first] == 'ASP' and 'HD2' in names[first]:
        remove(first, 'HD2')
    return [ln for i, ln in enumerate(lines) if i not in drop]


def relax_with_repair(src, dst, platform_name):
    """``relax``; on a missing-template failure, once more on the repaired input."""
    try:
        relax(src, dst, platform_name)
    except ValueError as error:
        if 'No template found' not in str(error):
            raise
        with tempfile.TemporaryDirectory() as td:
            fixed = Path(td) / 'repaired.pdb'
            fixed.write_text('\n'.join(repair(Path(src).read_text().splitlines())) + '\n')
            relax(fixed, dst, platform_name)


def main():
    bench_out = os.environ.get('DEEPSHIFT_BENCH_OUT', '')
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tasks', type=Path, default=Path(bench_out) / 'fold6_tasks.tsv',
                        help='fold-6 task table (default $DEEPSHIFT_BENCH_OUT/fold6_tasks.tsv)')
    parser.add_argument('--chains', type=Path, required=True,
                        help='directory of extracted chains, <structure_id>.pdb')
    parser.add_argument('--out', type=Path, required=True,
                        help='directory the relaxed chains are written to')
    parser.add_argument('--platform', default=None,
                        help='OpenMM platform (CUDA, OpenCL, CPU); default is the fastest')
    parser.add_argument('--only', nargs='+', default=None, metavar='STRUCTURE_ID',
                        help='relax just these structure ids')
    args = parser.parse_args()

    with open(args.tasks) as stream:
        tasks = list(csv.DictReader(stream, delimiter='\t'))
    if args.only:
        tasks = [t for t in tasks if t['structure_id'] in set(args.only)]
    args.chains.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    failed = {}
    began = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        sid = task['structure_id']
        dst = args.out / f'{sid}.pdb'
        if dst.exists() and dst.stat().st_size:
            skipped += 1
            continue
        src = args.chains / f'{sid}.pdb'
        try:
            if not src.is_file():
                extract_chain(task['pdb_path'], task['chain'], src, task.get('model_index') or 0)
            relax_with_repair(src, dst, args.platform)
            done += 1
        except Exception as error:  # one bad chain must not stop the run
            failed[sid] = f'{type(error).__name__}: {error}'
            print(f'FAILED {sid}: {failed[sid]}', flush=True)
        if i % 25 == 0 or i == len(tasks):
            print(f'[{i}/{len(tasks)}] relaxed {done}, present {skipped}, failed {len(failed)}, '
                  f'{time.perf_counter() - began:.0f} s', flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
