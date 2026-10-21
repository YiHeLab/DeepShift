#!/usr/bin/env python3
"""Per-invocation inference speed of one predictor over the 203-chain fold-6 crystal cohort.

One arm per invocation, never two at once (a timing taken under contention is not
reportable). Writes ``$SI_SPEED_CENSUS/out/<arm>__full__r0.json``, the per-structure records
``summarize.py`` aggregates.

Clock, identical for every arm. The single-chain PDB (``chains/``) is made once and is
not timed. For a competitor the clock spans ``adapter.predict()``, the same interval
``competitors/run_arm.py`` records as ``runtime_sec``: the tool's own hydrogenation (REDUCE,
where it needs explicit H), process and model start, features, evaluation and output
parsing. Every competitor is a per-file command-line tool and pays its full startup on
every structure. For DeepShift the clock spans a fresh ``deepshift predict`` process per
structure, the like-for-like row against them. Two off-cohort chains are run and discarded
before the clock starts, so every arm has the same page-cache and lazy-initialisation
warm-up.

    python run_speed_census.py --arm ucbshift_x
    python run_speed_census.py --arm deepshift_final_gpu_ens_coldproc
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import speed_common as sc

REPO = sc.REPO
WORK = 'work'
MODELS = REPO / 'models'

# arm -> (adapter module, tool name the adapter is called with)
COMPETITORS = {
    'ucbshift_x': ('ucbshift', 'ucbshift_x'),
    'shiftx2_noshifty': ('shiftx2', 'shiftx2_noshifty'),
    'sparta': ('sparta', 'sparta'),
    'procs15': ('procs15', 'procs15'),
    'legolas_cpu': ('legolas', 'legolas'),
    'napshift': ('napshift', 'napshift'),
    'nmrgnn_cpu': ('nmrgnn', 'nmrgnn'),
}
# arm -> number of ensemble members it loads
DEEPSHIFT_ARMS = {
    'deepshift_final_gpu_single_coldproc': 1,
    'deepshift_final_gpu_ens_coldproc': 5,
}


def load_cohort(warmup):
    """(timed structure ids, warm-up ids, manifest).

    Warm-up chains come from the 37 fold-6 crystal chains outside the cohort, so priming a
    tool never touches a timed structure.
    """
    manifest = sc.manifest()
    sids = sc.cohort_ids()
    warm = sorted(set(manifest) - set(sids))[:warmup]
    return sids, warm, manifest


def record(arm, sid, info, wall, n_predicted, status):
    return {'arm': arm, 'structure_id': sid, 'n_aa': info['n_aa'], 'wall_sec': wall,
            'n_residues_predicted': n_predicted, 'status': status}


def run_competitor(arm, sids, manifest, records):
    module, tool = COMPETITORS[arm]
    adapter = sc.load_adapter(module)
    for i, sid in enumerate(sids, 1):
        info = manifest[sid]
        work = sc.root() / WORK / arm / sid
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        t0 = time.perf_counter()
        try:
            preds, _ = adapter.predict(tool, str(sc.chain_pdb(sid)), info['chain'],
                                       info['bmrb_id'], workdir=str(work))
            status = 'ok' if preds else 'no_output'
        except Exception as exc:
            preds, status = {}, f'error:{type(exc).__name__}'
        wall = time.perf_counter() - t0
        records.append(record(arm, sid, info, wall, len(preds), status))
        shutil.rmtree(work, ignore_errors=True)
        if i % 20 == 0 or i == len(sids):
            print(f'  [{i}/{len(sids)}] {arm} last={wall:.2f}s', flush=True)


def run_deepshift(arm, sids, manifest, records, models):
    """A fresh ``deepshift predict`` process per structure on the shipped checkpoints."""
    checkpoints = [models / f'deepshift_fold{i}.pt' for i in range(1, DEEPSHIFT_ARMS[arm] + 1)]
    env = dict(os.environ, PYTHONPATH=str(REPO / 'src'))
    for i, sid in enumerate(sids, 1):
        info = manifest[sid]
        out = sc.root() / WORK / arm / f'{sid}.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-m', 'deepshift.pipeline', 'predict', '--pdb',
               str(sc.chain_pdb(sid)), '--chain', info['chain'], '--out', str(out)]
        for checkpoint in checkpoints:
            cmd += ['--checkpoint', str(checkpoint)]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        wall = time.perf_counter() - t0
        n = 0
        if out.exists():
            with open(out) as fh:
                n = sum(1 for _ in fh) - 1
        status = 'ok' if proc.returncode == 0 and n else f'error:rc{proc.returncode}'
        records.append(record(arm, sid, info, wall, n, status))
        out.unlink(missing_ok=True)
        if i % 20 == 0 or i == len(sids):
            print(f'  [{i}/{len(sids)}] {arm} last={wall:.3f}s', flush=True)
    return {'mode': 'fresh process per structure', 'n_models': len(checkpoints)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--arm', required=True, choices=sorted(COMPETITORS) + sorted(DEEPSHIFT_ARMS))
    ap.add_argument('--warmup', type=int, default=2,
                    help='off-cohort structures run and discarded before the clock starts')
    ap.add_argument('--models', type=Path, default=MODELS,
                    help='directory of deepshift_fold{1..5}.pt (DeepShift arms)')
    ns = ap.parse_args()

    sids, warm, manifest = load_cohort(ns.warmup)
    for sid in list(sids) + list(warm):
        manifest[sid]['n_aa'] = sc.aa_residue_count(sc.chain_pdb(sid))

    tag = f'{ns.arm}__full__r0'
    print(f'=== {tag}: warmup={len(warm)} timed={len(sids)} ===', flush=True)
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    everything, extra = [], {}
    if ns.arm in DEEPSHIFT_ARMS:
        extra = run_deepshift(ns.arm, list(warm) + list(sids), manifest, everything, ns.models)
    else:
        run_competitor(ns.arm, list(warm) + list(sids), manifest, everything)
    warm_records, records = everything[:len(warm)], everything[len(warm):]

    payload = {'arm': ns.arm, 'n_structures': len(records), 'started_utc': started,
               'sum_wall_sec': sum(r['wall_sec'] for r in records),
               'warmup_records': warm_records, 'extra': extra, 'records': records}
    sc.out_dir().mkdir(parents=True, exist_ok=True)
    (sc.out_dir() / (tag + '.json')).write_text(json.dumps(payload, indent=1) + '\n')
    ok = [r for r in records if r['status'] == 'ok']
    print(f"{tag}: ok={len(ok)}/{len(records)} sum_wall={payload['sum_wall_sec']:.1f}s")


if __name__ == '__main__':
    main()
