"""Regenerate the same-protein UCBShift and SHIFTX2 stores of UCBShift's 200-protein test set.

Table S19 reads cells made from these two stores. Each structure is run with the blocking
list ``benchmarks/reference/ucb200_competitor_same_protein.json`` holds for it under the
tool's transfer library (``ucbshift``, ``shifty``); a structure the file does not list
stops the run, since an unlisted structure would be run with nothing blocked. Output goes
under ``$DEEPSHIFT_BENCH_OUT``:

  ucb200_ucbshift_full_sameprot_sameprotein_seq_20261002/<id>.json + hitlog/<id>.log
  ucb200_shiftx2_sameprot_sameprotein_seq_20261002/<id>.cs + hitlog/<id>.log

The test chains are ``$UCBSHIFT_HOME/train_model/pdbs/test/<id>.pdb``, the ones that have a
file in ``test_targets`` there; <id> is the PDB id plus the chain letter, ``_`` for the first
chain. UCBShift runs on the REDUCE-hydrogenated chain of model 0 and writes the six backbone
nuclei; SHIFTX2 reads the chain with waters dropped and writes its CSV, and skips the three
chains the SHIFTX2 run left out (2GGM and 2HZI sit in the training folds, 1TCF has no BMRB
entry). Structures with a non-empty output are skipped, so a run resumes where it stopped.
Tool locations are the variables of ``adapters/ucbshift.py`` and ``adapters/shiftx2.py``.

    python run_ucb200.py --store ucbshift --workers 12
    python run_ucb200.py --store shiftx2 --structures 109M_ 1AE3_
"""
import argparse
import json
import multiprocessing
import shutil
import time
from collections import Counter
from pathlib import Path

from common import extract_chain, tool_path
from run_arm import load_adapter, out_root

FROZEN = (Path(__file__).resolve().parent.parent.parent / 'benchmarks' / 'reference'
          / 'ucb200_competitor_same_protein.json')
TAG = 'sameprotein_seq_20261002'
# store -> (arm, transfer library of the blocking list, output directory name)
STORES = {
    'ucbshift': ('ucbshift_full_sameprot', 'ucbshift', f'ucb200_ucbshift_full_sameprot_{TAG}'),
    'shiftx2': ('shiftx2_sameprot', 'shifty', f'ucb200_shiftx2_sameprot_{TAG}'),
}
SHIFTX2_SKIPPED = {'2GGM', '2HZI', '1TCF'}


def structures_of(store):
    """{id: PDB path} of the test chains the store covers."""
    root = Path(tool_path('UCBSHIFT_HOME')) / 'train_model'
    labelled = {p.stem for p in (root / 'test_targets').glob('*.csv')}
    found = {p.stem: p for p in (root / 'pdbs' / 'test').glob('*.pdb') if p.stem in labelled}
    if store == 'shiftx2':
        found = {k: v for k, v in found.items() if k[:4].upper() not in SHIFTX2_SKIPPED}
    return dict(sorted(found.items()))


def run_one(job):
    store, sid, pdb, blocked = job
    arm, _library, name = STORES[store]
    root = out_root() / name
    suffix = '.json' if store == 'ucbshift' else '.cs'
    out = root / f'{sid}{suffix}'
    if out.exists() and out.stat().st_size:
        return sid, 'cached', None
    work = out_root() / '_work' / name / sid
    chain = sid[4]
    try:
        work.mkdir(parents=True, exist_ok=True)
        predictor = load_adapter(arm)
        if store == 'ucbshift':
            chain_pdb = work / 'chain.pdb'
            extract_chain(pdb, chain, chain_pdb)
        else:
            chain_pdb = pdb
        preds, selfexcl = predictor.predict(
            arm, str(chain_pdb), chain, '', exclude_bmrbs=','.join(blocked),
            workdir=str(work), hitlog=str(root / 'hitlog' / f'{sid}.log'))
        if store == 'ucbshift':
            record = {'tool': arm, 'structure_id': sid, 'chain': chain,
                      'self_exclusion': selfexcl, 'predictions': preds,
                      'status': 'ok' if preds else 'no_output'}
            out.write_text(json.dumps(record))
        elif preds:
            shutil.copyfile(work / 'query.pdb.cs', out)
        return sid, 'ok' if preds else 'no_output', None
    except Exception as exc:
        return sid, f'error:{type(exc).__name__}', str(exc)[:400]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--store', required=True, choices=sorted(STORES))
    ap.add_argument('--structures', nargs='*', help='structure ids to run (default: all)')
    ap.add_argument('--workers', type=int, default=1)
    ns = ap.parse_args()

    structures = structures_of(ns.store)
    lists = json.loads(FROZEN.read_text())['exclude_bmrbs'][STORES[ns.store][1]]
    selected = ns.structures or list(structures)
    unknown = [s for s in selected if s not in structures]
    unlisted = [s for s in selected if s not in lists]
    if unknown or unlisted:
        raise SystemExit(f'not a {ns.store} test structure: {unknown}; '
                         f'no blocking list in {FROZEN.name}: {unlisted}')
    root = out_root() / STORES[ns.store][2]
    (root / 'hitlog').mkdir(parents=True, exist_ok=True)
    jobs = [(ns.store, s, structures[s], lists[s]) for s in selected]
    print(f'store={root.name} structures={len(jobs)} workers={ns.workers}', flush=True)

    counts = Counter()
    started = time.time()
    with multiprocessing.Pool(ns.workers, maxtasksperchild=50) as pool:
        for i, (sid, status, detail) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            counts[status.split(':')[0]] += 1
            if status not in ('ok', 'cached'):
                print(f'  FAIL {sid} {status} {(detail or "")[:120]}', flush=True)
            if i % 10 == 0 or i == len(jobs):
                print(f'[{i}/{len(jobs)}] {(time.time() - started) / 60:.1f} min '
                      f'{dict(counts)}', flush=True)
    print('STATUS', dict(counts), flush=True)


if __name__ == '__main__':
    main()
