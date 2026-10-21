"""NapShift adapter.

NapShift is an artificial neural network over structural descriptors with a camcoil
random-coil reference: structure-only, no shift database. It reads the chain PDB as
extracted (no hydrogens are added) and writes a TALOS table, one ``RESID RESNAME ATOMNAME
SHIFT`` row per predicted cell. Nuclei: N, C, CA, CB, HA, HN (the scorer maps HN to H).
Forced onto the CPU.

Reads ``NAPSHIFT_PYTHON`` (interpreter of NapShift's environment) and ``NAPSHIFT_HOME``
(the checkout holding ``napshift.py``).
"""
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import resname_map, tool_path


def predict(tool, chain_pdb, chain, bmrb_id, exclude_bmrbs='', exclude_uniprots='',
            identity_cutoff=0.90, workdir=None, hitlog=None):
    python = tool_path('NAPSHIFT_PYTHON')
    script = tool_path('NAPSHIFT_HOME') + '/napshift.py'
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    outdir = workdir / 'napout'
    outdir.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    with open(workdir / 'napshift.log', 'w') as log:
        subprocess.run([python, script, '-f', str(chain_pdb), '-o', str(outdir)],
                       cwd=str(workdir), env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=1800, check=False)
    names = resname_map(chain_pdb)
    preds = {}
    for tab in sorted(outdir.glob('prediction_*.tab')):
        with open(tab) as fh:
            for line in fh:
                fields = line.split()
                if len(fields) < 4:
                    continue
                try:
                    rid = str(int(fields[0]))
                    value = float(fields[3])
                except ValueError:
                    continue                       # REMARK/DATA/VARS/FORMAT headers
                if not math.isfinite(value):
                    continue
                cell = preds.setdefault(rid, {'resname': names.get(rid, '')})
                cell[fields[2]] = value
    return preds, {'mode': 'structure_only', 'excluded': []}
