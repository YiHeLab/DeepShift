"""Run MLFF over the fold-6 task table on Amber99SB-relaxed chains.

Reads ``$DEEPSHIFT_BENCH_OUT/fold6_tasks.tsv`` and ``<relaxed>/<structure_id>.pdb`` (written
by ``relax.py``), runs ``adapters/mlff.py`` on each chain, and writes
``$DEEPSHIFT_BENCH_OUT/mlff/<structure_id>.json``. A structure with a non-empty output is
skipped, so a run resumes where it stopped. Run in the interpreter of the environment that
holds mace-torch, torch_geometric and ase (``$MACE_PYTHON``):

    $MACE_PYTHON experiments/competitors/run_mlff.py --relaxed RELAXED
"""
import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from adapters import mlff


def main():
    bench_out = Path(os.environ.get('DEEPSHIFT_BENCH_OUT', ''))
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tasks', type=Path, default=bench_out / 'fold6_tasks.tsv',
                        help='fold-6 task table (default $DEEPSHIFT_BENCH_OUT/fold6_tasks.tsv)')
    parser.add_argument('--relaxed', type=Path, required=True,
                        help='directory of relaxed chains, <structure_id>.pdb')
    parser.add_argument('--out', type=Path, default=bench_out / 'mlff',
                        help='output directory (default $DEEPSHIFT_BENCH_OUT/mlff)')
    parser.add_argument('--device', default=None, help='cuda or cpu (default: cuda if present)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='cut-outs per MACE forward; 8 reproduces one-per-forward to 1e-4 ppm')
    parser.add_argument('--only', nargs='+', default=None, metavar='STRUCTURE_ID',
                        help='run just these structure ids')
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    with open(args.tasks) as stream:
        tasks = list(csv.DictReader(stream, delimiter='\t'))
    if args.only:
        tasks = [t for t in tasks if t['structure_id'] in set(args.only)]
    args.out.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    began = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        sid = task['structure_id']
        out = args.out / f'{sid}.json'
        if out.exists() and out.stat().st_size:
            continue
        started = time.perf_counter()
        record = {'self_exclusion': {'mode': 'structure_only', 'excluded': []}}
        try:
            with tempfile.TemporaryDirectory() as td:
                record['predictions'] = mlff.predict(args.relaxed / f'{sid}.pdb', td,
                                                     args.batch_size, args.device)
            record['status'] = 'ok'
            done += 1
        except Exception as error:  # one bad chain must not stop the run
            record.update(status='failed', predictions={},
                          error=f'{type(error).__name__}: {error}')
            failed += 1
            print(f'FAILED {sid}: {record["error"]}', flush=True)
        record['runtime_sec'] = round(time.perf_counter() - started, 2)
        partial = out.with_suffix('.json.part')
        partial.write_text(json.dumps(record, indent=1))
        partial.replace(out)
        if i % 25 == 0 or i == len(tasks):
            print(f'[{i}/{len(tasks)}] ok {done}, failed {failed}, '
                  f'{time.perf_counter() - began:.0f} s', flush=True)


if __name__ == '__main__':
    main()
