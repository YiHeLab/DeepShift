"""SHIFTX2 on its own 61-protein test set, in the three settings Table S20 prints.

Writes SHIFTX2's CSV for every protein under ``$DEEPSHIFT_BENCH_OUT/shiftx2_61/``:

  asdistributed/<id>.cs   SHIFTX+ with SHIFTY+ homology transfer, nothing blocked
  structonly/<id>.cs      SHIFTX+ only
  selfexcl/<id>.cs        as distributed, with the query's own depositions blocked
  hitlog/<id>.log         the references SHIFTY+ transferred from in the as-distributed run

The blocked depositions are read off that hit log, as the same-BMRB rows of Table S20 take
them: the library entry of the highest identity and every entry at or above 95 % identity.
The 61 protein ids are those of ``benchmarks/reference/shiftx2_test61_ucbshift_overlap.csv``
(<PDB id><chain letter>). The query chain is ``$UCBSHIFT_HOME/refDB/pdbs/<id>.pdb`` when
UCBShift's RefDB has it; otherwise the first model of the RCSB entry, downloaded into
``<out>/rcsb/`` unless the file is already there. Proteins with all three outputs are
skipped, so a run resumes where it stopped. SHIFTX2 is located by ``SHIFTX2_HOME`` and
``SHIFTX2_PYTHON`` (``adapters/shiftx2.py``).

    python run_shiftx2_61.py --workers 12
    python run_shiftx2_61.py --structures 1KF3A 1XMTA
"""
import argparse
import csv
import multiprocessing
import shutil
import time
import urllib.request
from collections import Counter
from pathlib import Path

from common import tool_path
from run_arm import load_adapter, out_root

ROSTER = (Path(__file__).resolve().parent.parent.parent / 'benchmarks' / 'reference'
          / 'shiftx2_test61_ucbshift_overlap.csv')
MODES = ('asdistributed', 'structonly', 'selfexcl')
SAME_ENTRY_IDENTITY = 95.0    # percent


def roster():
    with open(ROSTER) as fh:
        return [row['pdb'] for row in csv.DictReader(line for line in fh if line[0] != '#')]


def query_source(pid, root):
    """The file the chain is cut from: RefDB's structure, else the RCSB entry's first model."""
    refdb = Path(tool_path('UCBSHIFT_HOME')) / 'refDB' / 'pdbs' / f'{pid}.pdb'
    if refdb.exists():
        return refdb
    entry = root / 'rcsb' / f'{pid[:4]}.pdb'
    model1 = root / 'rcsb' / f'{pid[:4]}_model1.pdb'
    if not entry.exists():
        entry.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f'https://files.rcsb.org/download/{pid[:4]}.pdb', entry)
    lines = []
    for line in entry.read_text().splitlines(keepends=True):
        if line.startswith('ENDMDL'):
            break
        lines.append(line)
    model1.write_text(''.join(lines))
    return model1


def own_depositions(hitlog):
    """BMRB ids of the as-distributed run's top hit and of every hit at >=95 % identity."""
    hits = []
    for line in Path(hitlog).read_text().splitlines():
        field = line.split()
        if len(field) == 4 and field[0] == 'HITLOG':
            hits.append((field[2].removeprefix('bmr').removesuffix('.cs'), float(field[3])))
    if not hits:
        return []
    top = max(hits, key=lambda hit: hit[1])[0]
    return sorted({b for b, identity in hits if identity >= SAME_ENTRY_IDENTITY} | {top})


def run_mode(tool, mode, source, chain, work, root, pid, **kwargs):
    """One SHIFTX2 run; its CSV is kept as ``<mode>/<pid>.cs``. False when it wrote none."""
    preds, _ = load_adapter(tool).predict(tool, source, chain, '', workdir=str(work / mode),
                                          **kwargs)
    if preds:
        shutil.copyfile(work / mode / 'query.pdb.cs', root / mode / f'{pid}.cs')
    return bool(preds)


def run_one(pid):
    root = out_root() / 'shiftx2_61'
    if all((root / mode / f'{pid}.cs').exists() for mode in MODES):
        return pid, 'cached', None
    work = out_root() / '_work' / 'shiftx2_61' / pid
    hitlog = root / 'hitlog' / f'{pid}.log'
    try:
        source, chain = str(query_source(pid, root)), pid[4]
        runs = (
            ('shiftx2_leaky', 'asdistributed', {'hitlog': str(hitlog)}),
            ('shiftx2_noshifty', 'structonly', {}),
            # the blocked depositions are read when this run starts, off the first run's log
            ('shiftx2_sameprot', 'selfexcl', None))
        for tool, mode, kwargs in runs:
            if kwargs is None:
                kwargs = {'exclude_bmrbs': ','.join(own_depositions(hitlog))}
            if not run_mode(tool, mode, source, chain, work, root, pid, **kwargs):
                return pid, f'no_output:{mode}', None
        return pid, 'ok', None
    except Exception as exc:
        return pid, f'error:{type(exc).__name__}', str(exc)[:400]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--structures', nargs='*', help='protein ids to run (default: all 61)')
    ap.add_argument('--workers', type=int, default=1)
    ns = ap.parse_args()

    proteins = roster()
    selected = ns.structures or proteins
    unknown = [p for p in selected if p not in proteins]
    if unknown:
        raise SystemExit(f'not in the 61-protein roster: {unknown}')
    root = out_root() / 'shiftx2_61'
    for sub in (*MODES, 'hitlog'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    print(f'shiftx2_61 proteins={len(selected)} workers={ns.workers}', flush=True)

    counts = Counter()
    started = time.time()
    with multiprocessing.Pool(ns.workers) as pool:
        for i, (pid, status, detail) in enumerate(pool.imap_unordered(run_one, selected), 1):
            counts[status.split(':')[0]] += 1
            if status not in ('ok', 'cached'):
                print(f'  FAIL {pid} {status} {(detail or "")[:120]}', flush=True)
            if i % 10 == 0 or i == len(selected):
                print(f'[{i}/{len(selected)}] {(time.time() - started) / 60:.1f} min '
                      f'{dict(counts)}', flush=True)
    print('STATUS', dict(counts), flush=True)


if __name__ == '__main__':
    main()
