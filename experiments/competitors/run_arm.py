"""Run one structure-only or retrieval competitor arm over the fold-6 task table.

Reads ``$DEEPSHIFT_BENCH_OUT/fold6_tasks.tsv`` (written by ``python -m benchmarks.cells``),
extracts each structure's chain from the model the cache selected, runs the arm's adapter on
it, and writes ``$DEEPSHIFT_BENCH_OUT/<arm>/<structure_id>.json`` plus, for every arm that
retrieves reference shifts, ``<arm>/hitlog/<structure_id>.log``. Structures with a non-empty
output are skipped, so a run resumes where it stopped.

Arms: sparta, napshift, legolas, procs15, nmrgnn, camshift (structure-only);
ucbshift_x, ucbshift_full_leaky, ucbshift_full_sameprot, ucbshift_full_selfexcl,
ucbshift2_sameprot, ucbshift2_selfexcl, shiftx2_leaky, shiftx2_sameprot, shiftx2_selfexcl,
shiftx2_noshifty. MLFF and PLM-CS have their own scripts.

The ``*_sameprot`` and ``*_selfexcl`` arms block, per structure, the BMRB entries listed in
``benchmarks/reference/competitor_same_protein.json`` for the arm's transfer library; a
structure that file does not list keeps the task table's ``exclude_bmrbs`` and its own BMRB.
``*_selfexcl`` arms also drop every reference at >=90 % identity.

    python run_arm.py --arm sparta --all --workers 12
    python run_arm.py --arm ucbshift_full_sameprot --structures b10002_nmr_l1_a1_e1
    python run_arm.py --arm procs15 --all --batch 250
"""
import argparse
import csv
import importlib.util
import json
import multiprocessing
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from common import count_h, extract_chain

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent.parent / 'benchmarks' / 'reference' / 'competitor_same_protein.json'

# arm -> (adapter module, transfer library whose blocking list the arm uses or None)
ARMS = {
    'sparta': ('sparta', None), 'napshift': ('napshift', None), 'legolas': ('legolas', None),
    'procs15': ('procs15', None), 'nmrgnn': ('nmrgnn', None), 'camshift': ('camshift', None),
    'ucbshift_x': ('ucbshift', None), 'ucbshift_full_leaky': ('ucbshift', None),
    'ucbshift_full_sameprot': ('ucbshift', 'ucbshift'),
    'ucbshift_full_selfexcl': ('ucbshift', 'ucbshift'),
    'ucbshift2_sameprot': ('ucbshift', 'ucbshift'),
    'ucbshift2_selfexcl': ('ucbshift', 'ucbshift'),
    'shiftx2_leaky': ('shiftx2', None), 'shiftx2_noshifty': ('shiftx2', None),
    'shiftx2_sameprot': ('shiftx2', 'shifty'), 'shiftx2_selfexcl': ('shiftx2', 'shifty'),
}
# Arms that write a hit log: those that retrieve reference shifts.
RETRIEVING = {arm for arm in ARMS if arm.startswith(('ucbshift_full', 'ucbshift2', 'shiftx2'))
              and arm != 'shiftx2_noshifty'}
# The file each hydrogen-requiring arm feeds its predictor. A file without hydrogens
# fails the task instead of yielding predictions for an input the tool cannot use.
PREPARED = {'sparta': 'in.pdb', 'legolas': 'hyd.pdb', 'nmrgnn': 'hyd.pdb',
            'camshift': 'reduced.pdb'}


def out_root():
    value = os.environ.get('DEEPSHIFT_BENCH_OUT')
    if not value:
        raise SystemExit('DEEPSHIFT_BENCH_OUT is not set')
    return Path(value)


def load_adapter(arm):
    path = HERE / 'adapters' / f'{ARMS[arm][0]}.py'
    spec = importlib.util.spec_from_file_location(f'adapter_{path.stem}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blocked_bmrbs(arm, task, frozen):
    """The BMRB ids ``arm`` blocks for ``task``, comma-separated; '' for an unblocked arm."""
    library = ARMS[arm][1]
    if library is None:
        return ''
    listed = frozen['exclude_bmrbs'][library].get(task['structure_id'])
    if listed is not None:
        return ','.join(listed)
    own = {task['bmrb_id']} | {b for b in task['exclude_bmrbs'].split(',') if b}
    return ','.join(sorted(own, key=lambda x: (len(x), x)))


def write_record(arm, task, preds, selfexcl, seconds):
    record = {
        'schema_version': 1, 'tool': arm, 'structure_id': task['structure_id'],
        'bmrb_id': task['bmrb_id'], 'source': task['source'],
        'structure_file': Path(task['pdb_path']).name, 'chain': task['chain'],
        'self_exclusion': selfexcl, 'predictions': preds, 'runtime_sec': round(seconds, 2),
        'status': 'ok' if preds else 'no_output',
    }
    path = out_root() / arm / f"{task['structure_id']}.json"
    tmp = path.with_suffix('.json.part')
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(path)
    return record['status']


def prepare_chain(task, work):
    work.mkdir(parents=True, exist_ok=True)
    chain_pdb = work / 'chain.pdb'
    extract_chain(task['pdb_path'], task['chain'], chain_pdb, task.get('model_index') or 0)
    return chain_pdb


def run_one(job):
    arm, task, frozen, identity_cutoff = job
    sid = task['structure_id']
    out = out_root() / arm / f'{sid}.json'
    if out.exists() and out.stat().st_size:
        return sid, 'cached', None
    work = out_root() / '_work' / arm / sid
    try:
        chain_pdb = prepare_chain(task, work)
        hitlog = out_root() / arm / 'hitlog' / f'{sid}.log' if arm in RETRIEVING else None
        started = time.time()
        preds, selfexcl = load_adapter(arm).predict(
            arm, str(chain_pdb), task['chain'], task['bmrb_id'],
            exclude_bmrbs=blocked_bmrbs(arm, task, frozen),
            exclude_uniprots=task['uniprots'], identity_cutoff=identity_cutoff,
            workdir=str(work), hitlog=str(hitlog) if hitlog else None)
        seconds = time.time() - started
        if arm in PREPARED and not count_h(work / PREPARED[arm])[0]:
            return sid, 'error:NoHydrogensInPreparedFile', f'{arm} got no hydrogens'
        status = write_record(arm, task, preds, selfexcl, seconds)
        return sid, status, None
    except Exception as exc:
        return sid, f'error:{type(exc).__name__}', str(exc)[:400]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_procs15_batches(tasks, batch):
    """ProCS15 pays its data load once per call, so structures run in batches."""
    adapter = load_adapter('procs15')
    counts = Counter()
    for start in range(0, len(tasks), batch):
        chunk = tasks[start:start + batch]
        work = out_root() / '_work' / 'procs15' / f'batch{start // batch:05d}'
        shutil.rmtree(work, ignore_errors=True)
        chains = {t['structure_id']: prepare_chain(t, work / 'chains' / t['structure_id'])
                  for t in chunk}
        started = time.time()
        done = adapter.predict_many(chains, work / 'run')
        seconds = (time.time() - started) / max(len(chunk), 1)
        for task in chunk:
            sid = task['structure_id']
            if sid in done:
                counts[write_record('procs15', task, done[sid],
                                    {'mode': 'structure_only', 'excluded': []}, seconds)] += 1
            else:
                counts['no_output'] += 1
                print(f'  FAIL {sid} predictor produced no .procs', flush=True)
        shutil.rmtree(work, ignore_errors=True)
        print(f'[{min(start + batch, len(tasks))}/{len(tasks)}] {dict(counts)}', flush=True)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--arm', required=True, choices=sorted(ARMS))
    ap.add_argument('--structures', nargs='*', help='structure ids to run')
    ap.add_argument('--all', action='store_true', help='every structure of the task table')
    ap.add_argument('--source', choices=['af', 'nmr', 'xtal'])
    ap.add_argument('--tasks', type=Path, help='task table (default fold6_tasks.tsv)')
    ap.add_argument('--identity_cutoff', type=float, default=0.90,
                    help='*_selfexcl arms: references at or above this identity are dropped')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--batch', type=int, default=250, help='procs15 only: structures per call')
    ap.add_argument('--shard', help='i/n: run only the tasks whose index %% n == i')
    ns = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    with open(ns.tasks or out_root() / 'fold6_tasks.tsv') as fh:
        tasks = list(csv.DictReader(fh, delimiter='\t'))
    by_id = {t['structure_id']: t for t in tasks}
    if ns.structures:
        selected = [by_id[s] for s in ns.structures]
    elif ns.all:
        selected = [t for t in tasks if not ns.source or t['source'] == ns.source]
    else:
        raise SystemExit('specify --structures or --all')
    if ns.shard:
        i, n = (int(x) for x in ns.shard.split('/'))
        selected = [t for k, t in enumerate(selected) if k % n == i]

    arm_dir = out_root() / ns.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    done = {f[:-5] for f in os.listdir(arm_dir) if f.endswith('.json')}
    todo = [t for t in selected if t['structure_id'] not in done]
    print(f'arm={ns.arm} selected={len(selected)} todo={len(todo)} workers={ns.workers}',
          flush=True)
    if ns.arm == 'procs15':
        print('STATUS', dict(run_procs15_batches(todo, ns.batch)), flush=True)
        return

    frozen = json.loads(FROZEN.read_text()) if ARMS[ns.arm][1] else None
    jobs = [(ns.arm, t, frozen, ns.identity_cutoff) for t in todo]
    counts = Counter()
    started = time.time()

    def handle(i, result):
        sid, status, detail = result
        counts[status.split(':')[0]] += 1
        if status not in ('ok', 'cached'):
            print(f'  FAIL {sid} {status} {(detail or "")[:120]}', flush=True)
        if i % 25 == 0 or i == len(jobs):
            print(f'[{i}/{len(jobs)}] {(time.time() - started) / 60:.1f} min {dict(counts)}',
                  flush=True)

    if ns.workers > 1:
        with multiprocessing.Pool(ns.workers, maxtasksperchild=200) as pool:
            for i, result in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1), 1):
                handle(i, result)
    else:
        for i, job in enumerate(jobs, 1):
            handle(i, run_one(job))
    print('STATUS', dict(counts), flush=True)


if __name__ == '__main__':
    main()
