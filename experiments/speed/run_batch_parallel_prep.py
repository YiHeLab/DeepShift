#!/usr/bin/env python3
"""A batch arm of run_speed_batch.py with its per-file preparation spread over N worker
threads, the worker count DeepShift's throughput arm featurizes with
(``run_speed_throughput.py --workers 8``). ``run_speed_batch.py`` prepares the 203 chains one
after another, which charges every hydrogenating tool about 54 s of serial REDUCE that
DeepShift's parallel featurization does not pay. The tool invocation itself is unchanged.

Writes ``$SI_SPEED_CENSUS/out/<arm>_prep<workers>__batch__r<repeat>.json``.

    python run_batch_parallel_prep.py sparta_batch 8 3
"""
import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import run_speed_batch as B
import speed_common as sc

ARMS = ('sparta_batch', 'legolas_batch_cpu', 'ucbshift_x_batch')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('arm', choices=ARMS)
    ap.add_argument('workers', type=int, nargs='?', default=8)
    ap.add_argument('repeat', type=int, nargs='?', default=0)
    ap.add_argument('--smoke', type=int, default=0, help='run only the first N chains')
    ns = ap.parse_args()

    items = sc.cohort()[:ns.smoke or None]
    kind = 'smoke' if ns.smoke else 'batch'
    tag = f'{ns.arm}_prep{ns.workers}__{kind}__r{ns.repeat}'
    d = B.batch_dir() / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(ns.workers) as ex:      # each task is the arm's own per-file prep
        parts = list(ex.map(lambda it: B.PREP[ns.arm]([it], d)[0], items))
    files = [f for part in parts for f in part]
    state = {}
    if ns.arm == 'ucbshift_x_batch':                # prep_ucbshift's list file, in cohort order
        state['list'] = d / 'batch_list.txt'
        state['list'].write_text('\n'.join(str(p) for p in files) + '\n')
    prep = time.perf_counter() - t0
    t1 = time.perf_counter()
    n_out = B.invoke(ns.arm, files, d, state, B.UCBSHIFT_WORKERS)
    tool = time.perf_counter() - t1
    p = {'arm': ns.arm, 'prep_workers': ns.workers, 'n_chains': len(items),
         'n_prepared': len(files), 'n_outputs': n_out, 'prep_sec': prep, 'tool_sec': tool,
         'total_sec': prep + tool, 'per_protein_sec': (prep + tool) / len(items),
         'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    sc.out_dir().mkdir(parents=True, exist_ok=True)
    (sc.out_dir() / f'{tag}.json').write_text(json.dumps(p, indent=1) + '\n')
    print(f'{tag}: {n_out}/{len(items)}, prep {prep:.1f}s + tool {tool:.1f}s = '
          f'{p["per_protein_sec"]:.3f} s/protein')


if __name__ == '__main__':
    main()
