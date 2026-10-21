"""SPARTA+ adapter.

SPARTA+ predicts backbone shifts from a neural network over local sequence plus
ring-current, electric-field and hydrogen-bond terms computed from the query structure. It
transfers no shift from the query protein or its homologs, so the arm is structure-only.
Hydrogens are placed with REDUCE. Nuclei written: N, CA, CB, C, HA, HN (the scorer maps
HN to H).

Reads ``SPARTA_HOME``, the SPARTA+ distribution directory (it holds ``bin/SPARTA+``).
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import hydrogenate, resname_map, tool_path


def parse_pred_tab(path):
    """``pred.tab`` -> {residue number: {atom: ppm}}; atoms N, CA, CB, C, HA, HN."""
    out = {}
    in_data = False
    with open(path) as fh:
        for line in fh:
            if line.startswith('FORMAT'):
                in_data = True
                continue
            if not in_data:
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                resid = int(fields[0])
            except ValueError:
                continue
            out.setdefault(str(resid), {})[fields[2]] = float(fields[4])
    return out


def predict(tool, chain_pdb, chain, bmrb_id, exclude_bmrbs='', exclude_uniprots='',
            identity_cutoff=0.90, workdir=None, hitlog=None):
    home = tool_path('SPARTA_HOME')
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if not hydrogenate(chain_pdb, workdir / 'in.pdb'):
        return {}, {'mode': 'error', 'error': 'reduce_no_output'}
    env = dict(os.environ, SPARTAP_DIR=home)
    with open(workdir / 'sparta.log', 'w') as log:
        subprocess.run([home + '/bin/SPARTA+', '-spartaDir', home, '-in', 'in.pdb'],
                       cwd=str(workdir), env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=1800, check=False)
    tab = workdir / 'pred.tab'
    raw = parse_pred_tab(tab) if tab.exists() else {}
    names = resname_map(chain_pdb)
    preds = {rid: {'resname': names.get(rid, ''), **atoms} for rid, atoms in raw.items()}
    return preds, {'mode': 'structure_only', 'excluded': []}
