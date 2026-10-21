"""LEGOLAS adapter.

LEGOLAS is a TorchANI/AEV machine-learning model that reads atom coordinates only:
structure-only, no shift database. It needs explicit hydrogens, placed with REDUCE, and is
forced onto the CPU.

Output CSV columns: ATOM_TYPE, SEQ_ID, RES_TYPE, CHEMICAL_SHIFT, CHEMICAL_SHIFT_STD, for
N, CA, CB, C, HA and the amide H. Glycine emits two HA rows, written as HA2 and HA3 in
file order (the scorer resolves the prochiral swap). The N-terminal residue emits several
H rows (NH3+); their mean is written.

Reads ``LEGOLAS_PYTHON`` (interpreter of LEGOLAS's environment) and ``LEGOLAS_HOME`` (the
checkout holding ``test/legolas.py``).
"""
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import hydrogenate, resname_map, tool_path


def predict(tool, chain_pdb, chain, bmrb_id, exclude_bmrbs='', exclude_uniprots='',
            identity_cutoff=0.90, workdir=None, hitlog=None):
    python = tool_path('LEGOLAS_PYTHON')
    script = tool_path('LEGOLAS_HOME') + '/test/legolas.py'
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    hyd = workdir / 'hyd.pdb'
    if not hydrogenate(chain_pdb, hyd):
        return {}, {'mode': 'error', 'error': 'reduce_no_output'}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    with open(workdir / 'legolas.log', 'w') as log:
        subprocess.run([python, script, str(hyd), '-o', 'csv'],
                       cwd=str(workdir), env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=3600, check=False)
    csv_out = workdir / 'hyd_cs.csv'
    if not csv_out.exists():
        return {}, {'mode': 'structure_only', 'excluded': []}
    grouped = defaultdict(lambda: defaultdict(list))
    with open(csv_out) as fh:
        for row in csv.DictReader(fh):
            try:
                rid = str(int(row['SEQ_ID']))
                value = float(row['CHEMICAL_SHIFT'])
            except (ValueError, KeyError):
                continue
            grouped[rid][row['ATOM_TYPE'].strip()].append(value)
    names = resname_map(chain_pdb)
    preds = {}
    for rid, atoms in grouped.items():
        cell = {'resname': names.get(rid, '')}
        for atom, values in atoms.items():
            if atom == 'HA' and cell['resname'] == 'GLY' and len(values) >= 2:
                cell['HA2'], cell['HA3'] = values[0], values[1]
            else:
                cell[atom] = sum(values) / len(values)
        preds[rid] = cell
    return preds, {'mode': 'structure_only', 'excluded': []}
