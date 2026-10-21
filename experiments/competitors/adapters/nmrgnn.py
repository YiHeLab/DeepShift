"""NMRGNN adapter.

NMRGNN predicts H, C and N shifts with a graph neural network over the atoms of the query
structure: structure-only, no shift database. It needs explicit hydrogens (REDUCE) and an
input file with a ``.pdb`` suffix, and is forced onto the CPU.

``nmrgnn eval-struct <pdb> <out.csv>`` writes ``index, residues, resids, names, peaks,
confident, ...``. Atom names are PDB names, so N, CA, CB, C, HA, H and the side-chain
protons map straight to the scorer's columns. Every finite peak is kept; the scorer
ignores names it does not score.

Reads ``NMRGNN_BIN``, the ``nmrgnn`` executable of NMRGNN's environment.
"""
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import hydrogenate, resname_map, tool_path


def predict(tool, chain_pdb, chain, bmrb_id, exclude_bmrbs='', exclude_uniprots='',
            identity_cutoff=0.90, workdir=None, hitlog=None):
    binary = tool_path('NMRGNN_BIN')
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    hyd = workdir / 'hyd.pdb'
    if not hydrogenate(chain_pdb, hyd):
        return {}, {'mode': 'error', 'error': 'reduce_no_output'}
    out_csv = workdir / 'nmrgnn.csv'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    with open(workdir / 'nmrgnn.log', 'w') as log:
        subprocess.run([binary, 'eval-struct', str(hyd), str(out_csv)],
                       cwd=str(workdir), env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=3600, check=False)
    if not out_csv.exists():
        return {}, {'mode': 'structure_only', 'excluded': []}
    names = resname_map(chain_pdb)
    preds = {}
    n_confident = n_total = 0
    with open(out_csv) as fh:
        for row in csv.DictReader(fh):
            try:
                rid = str(int(row['resids']))
                value = float(row['peaks'])
            except (ValueError, KeyError):
                continue
            if not math.isfinite(value):
                continue
            n_total += 1
            if str(row.get('confident', '')).strip().lower() == 'true':
                n_confident += 1
            cell = preds.setdefault(rid, {'resname': names.get(rid, '')})
            cell[row['names'].strip().upper()] = value
    return preds, {'mode': 'structure_only', 'excluded': [],
                   'n_peaks': n_total, 'n_confident': n_confident}
