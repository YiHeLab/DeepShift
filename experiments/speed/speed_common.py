"""Paths, cohort and adapter loading shared by the speed-census scripts.

Everything the census reads or writes lives under ``$SI_SPEED_CENSUS``:

  chain_manifest.json      structure id -> chain, BMRB id, atom count (240 fold-6 crystal chains)
  cohort_fold6_xtal.json   ``xtal_structure_ids``: the 203 chains that are timed
  chains/<id>.pdb          the single-chain PDB every method is timed on
  out/                     one timing record per pass (``emit_speed.py`` reads these)
  batch/, work/            scratch directories of the passes

The 37 chains outside the cohort supply the discarded warm-up structures.
"""
import importlib.util
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ADAPTERS = REPO / 'experiments' / 'competitors' / 'adapters'

STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
    'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}


def root():
    value = os.environ.get('SI_SPEED_CENSUS')
    if not value:
        raise SystemExit('SI_SPEED_CENSUS is not set (the census directory)')
    return Path(value)


def out_dir():
    return root() / 'out'


def manifest():
    return json.loads((root() / 'chain_manifest.json').read_text())


def cohort_ids():
    return json.loads((root() / 'cohort_fold6_xtal.json').read_text())['xtal_structure_ids']


def chain_pdb(structure_id):
    return root() / 'chains' / f'{structure_id}.pdb'


def cohort():
    """[(structure id, chain PDB, chain id)] of the timed cohort, in cohort order."""
    chains = manifest()
    return [(sid, chain_pdb(sid), chains[sid]['chain']) for sid in cohort_ids()]


def aa_residue_count(pdb):
    """The one size denominator for every method: standard amino acids with a CA."""
    n = 0
    with open(pdb) as fh:
        for line in fh:
            if (line[:6].strip() in ('ATOM', 'HETATM') and line[12:16].strip() == 'CA'
                    and line[17:20].strip() in STANDARD_AA):
                n += 1
    return n


def load_module(path, name):
    """Import one file of ``experiments/competitors`` without putting its directory first
    on ``sys.path`` (its ``common`` module would shadow the adapters' own)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_adapter(name):
    return load_module(ADAPTERS / f'{name}.py', f'adapter_{name}')
