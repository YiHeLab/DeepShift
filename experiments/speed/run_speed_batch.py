#!/usr/bin/env python3
"""Batch inference speed: ONE invocation of a tool over all 203 cohort chains, through the
tool's own shipped batch mode. Companion to run_speed_census.py, which times the same
tools one process per structure.

  SPARTA+   SPARTA+ -in f1 f2 ...
  LEGOLAS   legolas.py f1 f2 ... -o csv
  SHIFTX2   shiftx2.py -b '<glob>' -n          (-n: SHIFTY+ off, the printed arm)
  NapShift  napshift.py -f f1 f2 ...
  ProCS15   procs15_predictor --pdb-files f1 f2 ...
  UCBShift  CSpred.py list.txt --batch --shiftx_only -w N
  NMRGNN    none: eval-struct reads extra files as trajectory frames of the first, so
            there is no per-structure batch mode and no arm

The clock covers everything from the shared single-chain PDBs to predictions on disk:
per-file preparation (REDUCE where the tool's adapter hydrogenates, water stripping for
SHIFTX2, the REDUCE -Trim/-build pair for UCBShift) plus the single tool invocation. This
script prepares the chains one after another and is used directly for the tools with no
preparation step to parallelize (SHIFTX2, NapShift, ProCS15); ``run_batch_parallel_prep.py``
spreads the preparation of the others over worker threads.

A timed pass refuses to start unless the machine is quiet (CPU idle >= 90 % over three
samples). ``--smoke N`` runs the first N chains with no gate and no timing claim, to show
that each invocation produces outputs.

    python run_speed_batch.py --arm shiftx2_batch
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import speed_common as sc

SERIAL_ARMS = ('shiftx2_batch', 'napshift_batch', 'procs15_batch')
UCBSHIFT_WORKERS = 4        # CSpred's own --batch worker count

COMMON = sc.load_module(sc.REPO / 'experiments' / 'competitors' / 'common.py',
                        'competitors_common')


tool = COMMON.tool_path
hydrogenate = COMMON.hydrogenate


def batch_dir():
    return sc.root() / 'batch'


def box_is_quiet():
    for _ in range(3):
        line = subprocess.run(['vmstat', '1', '2'], capture_output=True, text=True
                              ).stdout.strip().splitlines()[-1]
        if int(line.split()[14]) < 90:                              # vmstat "id" column
            return False, f'CPU idle {line.split()[14]}% < 90%'
        time.sleep(1)
    return True, ''


# ------------------------------------------------------------------ per-arm preparation
# -> (input paths for the one invocation, extra state the invocation needs)

def prep_reduce(items, d):
    out = []
    for sid, pdb, _ in items:
        hyd = d / f'{sid}.pdb'
        if hydrogenate(pdb, hyd):
            out.append(hyd)
    return out, {}


def prep_copy(items, d):
    out = []
    for sid, pdb, _ in items:
        local = d / f'{sid}.pdb'
        shutil.copy(pdb, local)
        out.append(local)
    return out, {}


def prep_waterless(items, d):
    extract = sc.load_adapter('shiftx2').extract_without_waters
    out = []
    for sid, pdb, chain in items:
        local = d / f'{sid}.pdb'
        extract(pdb, chain, local)
        out.append(local)
    return out, {}


def prep_ucbshift(items, d):
    """REDUCE -Trim then -build, exactly as the adapter prepares a single file."""
    reduce_bin = os.environ.get('REDUCE_BIN', 'reduce')
    out = []
    for sid, pdb, _ in items:
        local = d / f'{sid}.pdb'
        shutil.copy(pdb, local)
        with open(d / f'{sid}.trim', 'w') as fh:
            subprocess.run([reduce_bin, '-Trim', '-Quiet', str(local)],
                           stdout=fh, stderr=subprocess.DEVNULL)
        with open(local, 'w') as fh:
            subprocess.run([reduce_bin, '-build', '-Quiet', str(d / f'{sid}.trim')],
                           stdout=fh, stderr=subprocess.DEVNULL)
        out.append(local)
    lst = d / 'batch_list.txt'
    lst.write_text('\n'.join(str(p) for p in out) + '\n')
    return out, {'list': lst}


PREP = {'sparta_batch': prep_reduce, 'legolas_batch_cpu': prep_reduce,
        'legolas_batch_gpu': prep_reduce, 'shiftx2_batch': prep_waterless,
        'napshift_batch': prep_copy, 'procs15_batch': prep_copy,
        'ucbshift_x_batch': prep_ucbshift}


# ------------------------------------------------------------------ the one invocation

def cpu_env(**extra):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    e.update(extra)
    return e


def invoke(arm, files, d, state, workers):
    """Run the tool once over ``files``; the number of outputs it wrote."""
    with open(d / 'tool.log', 'w') as log:
        return _invoke(arm, files, d, state, workers, log)


def _invoke(arm, files, d, state, workers, log):
    run = lambda cmd, **kw: subprocess.run(         # noqa: E731
        [str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, check=False, **kw)
    if arm == 'sparta_batch':
        home = tool('SPARTA_HOME')
        run([home + '/bin/SPARTA+', '-spartaDir', home, '-in', *files],
            cwd=str(d), env=dict(os.environ, SPARTAP_DIR=home))
        return len(list(d.glob('*_pred.tab'))) or len(list(d.glob('pred*.tab')))
    if arm in ('legolas_batch_cpu', 'legolas_batch_gpu'):
        env = cpu_env() if arm.endswith('cpu') else dict(os.environ)
        run([tool('LEGOLAS_PYTHON'), tool('LEGOLAS_HOME') + '/test/legolas.py', *files,
             '-o', 'csv'], cwd=str(d), env=env)
        return len(list(d.glob('*_cs.csv')))
    if arm == 'shiftx2_batch':
        home = Path(tool('SHIFTX2_HOME'))
        # the SHIFTX2 environment leads PATH: shiftx2.py's batch path is Python-2 code
        env = dict(os.environ)
        env['PATH'] = f"{Path(tool('SHIFTX2_PYTHON')).parent}{os.pathsep}{env.get('PATH', '')}"
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        tmp = d / 'sxtmp'
        tmp.mkdir(exist_ok=True)
        run(['python', home / 'shiftx2.py', '-b', str(d / '*.pdb'), '-z', tmp,
             '-f', 'CSV', '-a', 'ALL', '-n'], cwd=str(home), env=env)
        return len(list(d.glob('*.cs')) + list(d.glob('*.csv')))
    if arm == 'napshift_batch':
        outdir = d / 'napout'
        outdir.mkdir(exist_ok=True)
        run([tool('NAPSHIFT_PYTHON'), tool('NAPSHIFT_HOME') + '/napshift.py', '-f', *files,
             '-o', outdir], cwd=str(d), env=cpu_env())
        return len(list(outdir.iterdir()))
    if arm == 'procs15_batch':
        run([tool('PROCS15_BIN'), '--data-folder', tool('PROCS15_DATA'), '--pdb-files', *files],
            cwd=str(d))
        return len(list(d.glob('*.procs')))
    if arm == 'ucbshift_x_batch':
        python = tool('UCBSHIFT_PYTHON')
        home = tool('UCBSHIFT_HOME')
        outdir = d / 'ucbout'
        outdir.mkdir(exist_ok=True)
        env = dict(os.environ)
        env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
        env['PYTHONPATH'] = home
        env['UCBY_EXCLUDE_BMRBS'] = ''
        run([python, home + '/CSpred.py', state['list'], '--batch', '--shiftx_only',
             '--output', str(outdir) + '/', '-w', workers], cwd=str(d), env=env)
        return len(list(outdir.glob('*.csv')))
    raise SystemExit(f'unknown arm {arm}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--arm', required=True, choices=SERIAL_ARMS)
    ap.add_argument('--smoke', type=int, default=0, help='run only the first N chains; no gate')
    ns = ap.parse_args()

    items = sc.cohort()
    if ns.smoke:
        items = items[:ns.smoke]
    else:
        quiet, why = box_is_quiet()
        if not quiet:
            raise SystemExit(f'REFUSING a timed run: {why}')

    kind = f'smoke{ns.smoke}' if ns.smoke else 'batch'
    tag = f'{ns.arm}__{kind}__r0'
    d = batch_dir() / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)

    t0 = time.perf_counter()
    files, state = PREP[ns.arm](items, d)
    prep_sec = time.perf_counter() - t0
    t1 = time.perf_counter()
    n_out = invoke(ns.arm, files, d, state, UCBSHIFT_WORKERS)
    tool_sec = time.perf_counter() - t1

    payload = {'arm': ns.arm, 'mode': 'batch', 'smoke': ns.smoke,
               'n_chains': len(items), 'n_prepared': len(files), 'n_outputs': n_out,
               'prep_sec': prep_sec, 'tool_sec': tool_sec,
               'total_sec': prep_sec + tool_sec,
               'per_protein_sec': (prep_sec + tool_sec) / len(items),
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    sc.out_dir().mkdir(parents=True, exist_ok=True)
    (sc.out_dir() / (tag + '.json')).write_text(json.dumps(payload, indent=1) + '\n')
    print(f'{tag}: {n_out}/{len(items)} outputs, prep {prep_sec:.1f}s + tool {tool_sec:.1f}s '
          f'= {payload["per_protein_sec"]:.3f} s/protein')


if __name__ == '__main__':
    main()
