#!/usr/bin/env python3
"""DeepShift's chains-finished count over the census's 240 fold-6 crystal chains.

Runs ``deepshift.prediction.predict_pdb`` on every chain of ``chain_manifest.json`` with the
first ``--members`` shipped checkpoints, on the CPU, and records which chains it finishes.
``--members 1`` writes ``$SI_SPEED_CENSUS/deepshift_coverage240_final.json`` (the
single-model row), ``--members 5`` ``deepshift_coverage240_final_ensemble.json`` (the
ensemble row). ``summarize.py`` reads both; ``emit_speed.py`` checks them against the census.

    python coverage240.py --members 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import speed_common as sc
import torch

sys.path.insert(0, str(sc.REPO / 'src'))
from deepshift.prediction import load_checkpoint, predict_pdb


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--members', type=int, required=True, choices=(1, 5))
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--models', type=Path, default=sc.REPO / 'models',
                    help='directory of deepshift_fold{1..5}.pt')
    ns = ap.parse_args()

    torch.set_num_threads(ns.threads)
    members = [load_checkpoint(ns.models / f'deepshift_fold{k}.pt')
               for k in range(1, ns.members + 1)]
    manifest = sc.manifest()
    rows, t0 = [], time.time()
    for sid in sorted(manifest):
        try:
            frame = predict_pdb(sc.chain_pdb(sid), manifest[sid]['chain'], members)
            rows.append({'structure_id': sid, 'ok': True,
                         'n_residues_predicted': len(frame)})
        except Exception as exc:
            rows.append({'structure_id': sid, 'ok': False,
                         'error': f'{type(exc).__name__}: {exc}'[:300]})
    n_ok = sum(r['ok'] for r in rows)
    name = 'deepshift_coverage240_final' + ('' if ns.members == 1 else '_ensemble') + '.json'
    out = sc.root() / name
    out.write_text(json.dumps({
        'what': f'predict_pdb with {ns.members} shipped checkpoint(s) over every chain of the '
                'census manifest: finishes or fails',
        'device': 'cpu', 'seconds': round(time.time() - t0, 1),
        'n_tasks': len(rows), 'n_ok': n_ok, 'failures': [r for r in rows if not r['ok']],
        'per_chain': rows}, indent=1) + '\n')
    print(f'{n_ok}/{len(rows)} finished; wrote {out}')


if __name__ == '__main__':
    main()
