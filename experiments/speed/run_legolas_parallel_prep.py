#!/usr/bin/env python3
"""LEGOLAS GPU batch with its REDUCE hydrogenation spread over worker threads, the worker
count DeepShift's throughput arm featurizes with (``run_speed_throughput.py --workers 8``).
``run_speed_batch.py`` hydrogenates the 203 chains one after another, which charges LEGOLAS
54 s of serial preparation that DeepShift's parallel featurization does not pay.

Writes ``$SI_SPEED_CENSUS/out/legolas_batch_gpu_prep<workers>__batch__r<repeat>.json``.

    python run_legolas_parallel_prep.py 8 1
"""
import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import run_speed_batch as B
import speed_common as sc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('workers', type=int, nargs='?', default=8)
    ap.add_argument('repeat', type=int, nargs='?', default=0)
    ns = ap.parse_args()

    tag = f'legolas_batch_gpu_prep{ns.workers}__batch__r{ns.repeat}'
    d = B.batch_dir() / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    items = sc.cohort()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(ns.workers) as ex:      # each task is two REDUCE subprocesses
        ok = list(ex.map(lambda it: B.hydrogenate(it[1], d / f'{it[0]}.pdb'), items))
    files = [d / f'{sid}.pdb' for (sid, _, _), good in zip(items, ok) if good]
    prep = time.perf_counter() - t0
    t1 = time.perf_counter()
    n_out = B.invoke('legolas_batch_gpu', files, d, {}, ns.workers)
    tool = time.perf_counter() - t1
    p = {'arm': 'legolas_batch_gpu', 'prep_workers': ns.workers, 'n_chains': len(items),
         'n_prepared': len(files), 'n_outputs': n_out, 'prep_sec': prep, 'tool_sec': tool,
         'total_sec': prep + tool, 'per_protein_sec': (prep + tool) / len(items),
         'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    sc.out_dir().mkdir(parents=True, exist_ok=True)
    (sc.out_dir() / f'{tag}.json').write_text(json.dumps(p, indent=1) + '\n')
    print(f'{tag}: {n_out}/{len(items)}, prep {prep:.1f}s + tool {tool:.1f}s = '
          f'{p["per_protein_sec"]:.3f} s/protein')


if __name__ == '__main__':
    main()
