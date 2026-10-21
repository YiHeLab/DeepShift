#!/usr/bin/env python3
"""Concurrent GPU throughput: DeepShift over all 203 cohort chains in ONE invocation.

A per-protein loop in a warm process measures latency, not what a user processing many
structures gets. This arm is the throughput regime: a CPU worker pool parses and builds
atom clouds for many proteins concurrently, residues from different proteins are packed
into shared GPU batches, every member of the ensemble runs on each batch, and per-protein
CSVs are written as proteins complete.

The clock is charged like ``run_speed_batch.py``'s competitor arms: the parent times a
child process from launch to the last CSV on disk -- interpreter start, imports,
checkpoint load, featurization, forward passes, denormalization, output. A timed run
refuses to start unless the GPU is idle. ``--check N`` compares the first N proteins with
``deepshift.prediction.predict_pdb`` and makes no timing claim.

Writes ``$SI_SPEED_CENSUS/out/deepshift_throughput_{single,ens}__batch__r<repeat>.json``.

    python run_speed_throughput.py --members 1 --check 5
    python run_speed_throughput.py --members 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import speed_common as sc

SRC = sc.REPO / 'src'
MODELS = sc.REPO / 'models'


def checkpoints(ns):
    return [str(Path(ns.models) / f'deepshift_fold{i}.pt') for i in range(1, ns.members + 1)]


def cohort():
    return [(sid, str(pdb), chain) for sid, pdb, chain in sc.cohort()]


# ------------------------------------------------------------------ child: the timed tool

_RECIPE = None


def _init_worker(recipe):
    global _RECIPE
    _RECIPE = recipe


def _featurize(item):
    """predict_pdb's first half, verbatim in effect: parse, insertion siblings, cloud."""
    from deepshift.atom_cloud import build_atom_cloud
    from deepshift.data.structures import retained_insertion_partner_residues
    from deepshift.structure import is_hydrogen, parse_structure
    sid, pdb, chain = item
    aa = parse_structure(pdb, chain, keep_nonstandard=True)
    if aa is None:
        return sid, None
    sib = {k: {**v, 'atoms': {n: x for n, x in v['atoms'].items() if not is_hydrogen(n)}}
           for k, v in retained_insertion_partner_residues(pdb, chain).items()}
    built = build_atom_cloud(aa, recipe=_RECIPE, partner_residues=sib or None)
    if built is None:
        return sid, None
    return sid, (built.model_arrays, built.residue_codes, built.residue_ids,
                 built.residue_names)


def child(ns):
    sys.path.insert(0, str(SRC))
    import multiprocessing as mp

    import numpy as np
    import pandas as pd
    import torch

    from deepshift.model import load_checkpoint
    from deepshift.normalization import denormalize
    from deepshift.prediction import checkpoint_recipe, nucleus_exists

    items = cohort()[:ns.limit or None]
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fork the featurization pool BEFORE CUDA is initialised; the recipe is read from the
    # checkpoint on CPU first so workers never touch the GPU.
    ckpts = checkpoints(ns)
    recipe = checkpoint_recipe(load_checkpoint(ckpts[0]))
    pool = mp.get_context('fork').Pool(ns.workers, _init_worker, (recipe,))
    stream = pool.imap_unordered(_featurize, items, chunksize=1)

    loaded = [load_checkpoint(c, device='cuda') for c in ckpts]
    columns = loaded[0].shift_columns

    pending, done = [], {}
    n_rows = 0
    results = {}          # sid -> list of per-member z chunks, filled as batches return

    def flush(final=False):
        nonlocal pending, n_rows
        while n_rows >= ns.batch or (final and n_rows):
            take, rows, used = [], 0, 0
            for (arrs, sid, lo, hi) in pending:
                if rows >= ns.batch:
                    break
                k = min(hi - lo, ns.batch - rows)
                take.append((arrs, sid, lo, lo + k))
                rows += k
                used += 1
            # split the last partially-consumed protein back onto the queue
            hi = take[-1][3]
            orig = pending[used - 1]
            rest = [(orig[0], orig[1], hi, orig[3])] if hi < orig[3] else []
            pending = rest + pending[used:]
            n_rows -= rows
            batch = [np.concatenate([a[j][lo:hi] for a, _, lo, hi in take])
                     for j in range(4)]
            tensors = [torch.from_numpy(x).pin_memory().to('cuda', non_blocking=True)
                       for x in batch]
            with torch.inference_mode(), torch.autocast('cuda'):
                z = [m.model(*tensors)[0].float() for m in loaded]
            z = torch.stack(z).cpu().numpy()          # members x rows x columns
            at = 0
            for _, sid, lo, hi in take:
                results[sid][1].append((lo, z[:, at:at + hi - lo]))
                at += hi - lo
                maybe_write(sid)

    def maybe_write(sid):
        codes, chunks, ids, names, n = results[sid]
        if sum(c.shape[1] for _, c in chunks) < n:
            return
        z = np.concatenate([c for _, c in sorted(chunks, key=lambda t: t[0])], axis=1)
        ppm = np.mean([denormalize(z[i], codes, m.stats, m.shift_columns)
                       for i, m in enumerate(loaded)], axis=0)
        frame = pd.DataFrame({'residue_id': ids, 'residue_name': names})
        for i, col in enumerate(columns):
            v = ppm[:, i].copy()
            v[np.array([not nucleus_exists(r, col) for r in names])] = np.nan
            frame[col] = v
        frame.to_csv(out_dir / f'{sid}.csv', index=False)
        done[sid] = len(ids)
        del results[sid]

    failed = []
    for sid, feat in stream:
        if feat is None:
            failed.append(sid)
            continue
        arrs, codes, ids, names = feat
        n = len(arrs[0])
        results[sid] = (codes, [], ids, names, n)
        pending.append((arrs, sid, 0, n))
        n_rows += n
        flush()
    flush(final=True)
    torch.cuda.synchronize()
    pool.close()
    pool.join()
    print(json.dumps({'n_outputs': len(done), 'n_residues': sum(done.values()),
                      'failed': failed}))


# ------------------------------------------------------------------ parent: the clock

def gpu_idle():
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                        '--format=csv,noheader'], capture_output=True, text=True).stdout
    return not q.strip(), q.strip()


def check(ns, out_dir):
    """Every checked protein must match the shipped predict_pdb to fp16-autocast noise."""
    sys.path.insert(0, str(SRC))
    import numpy as np
    import pandas as pd

    from deepshift.prediction import predict_pdb
    worst = 0.0
    for sid, pdb, chain in cohort()[:ns.check]:
        ref = predict_pdb(pdb, chain, checkpoints(ns))
        got = pd.read_csv(out_dir / f'{sid}.csv')
        assert list(ref.residue_id) == list(got.residue_id), sid
        cols = [c for c in ref.columns if c.endswith('_shift')]
        d = np.nanmax(np.abs(ref[cols].to_numpy(float) - got[cols].to_numpy(float)))
        worst = max(worst, float(d))
        print(f'  {sid}: {len(ref)} residues, max |diff| {d:.2e} ppm')
    print(f'check: worst |diff| {worst:.2e} ppm over {ns.check} proteins')
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', type=int, default=5, choices=(1, 5))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--batch', type=int, default=256, help='residues per GPU batch')
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--check', type=int, default=0, help='untimed: verify N vs predict_pdb')
    ap.add_argument('--models', type=Path, default=MODELS,
                    help='directory of deepshift_fold{1..5}.pt')
    ap.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--limit', type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument('--out-dir', help=argparse.SUPPRESS)
    ns = ap.parse_args()
    if ns.child:
        return child(ns)

    arm = f'deepshift_throughput_{"single" if ns.members == 1 else "ens"}'
    kind = f'check{ns.check}' if ns.check else 'batch'
    tag = f'{arm}__{kind}__r{ns.repeat}'
    out_dir = sc.root() / 'batch' / tag
    if not ns.check:
        idle, who = gpu_idle()
        if not idle:
            raise SystemExit(f'REFUSING a timed run: GPU busy\n{who}')
    subprocess.run(['rm', '-rf', str(out_dir)], check=True)

    cmd = [sys.executable, __file__, '--child', '--members', str(ns.members),
           '--workers', str(ns.workers), '--batch', str(ns.batch),
           '--limit', str(ns.check), '--out-dir', str(out_dir), '--models', str(ns.models)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    total = time.perf_counter() - t0
    if proc.returncode:
        sys.exit(proc.stderr)
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    n = ns.check or len(cohort())
    if ns.check:
        res['check_worst_abs_diff_ppm'] = check(ns, out_dir)
    payload = {'arm': arm, 'mode': 'throughput', 'members': ns.members,
               'workers': ns.workers, 'batch_residues': ns.batch, 'n_chains': n, **res,
               'total_sec': total, 'per_protein_sec': total / n,
               'proteins_per_sec': n / total, 'residues_per_sec': res['n_residues'] / total,
               'gpu': subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                     capture_output=True, text=True).stdout.strip(),
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    sc.out_dir().mkdir(parents=True, exist_ok=True)
    (sc.out_dir() / f'{tag}.json').write_text(json.dumps(payload, indent=1) + '\n')
    print(f'{tag}: {res["n_outputs"]}/{n} outputs, {res["n_residues"]} residues, '
          f'{total:.1f} s = {payload["per_protein_sec"]:.3f} s/protein, '
          f'{payload["residues_per_sec"]:.0f} residues/s')


if __name__ == '__main__':
    main()
