#!/usr/bin/env python3
"""Run the retimed census passes one at a time, each on an idle machine, and record that.

Before every pass: wait until no GPU compute process is running and both the 1- and
5-minute load averages have stayed below 1.0 for three consecutive minutes. During the
pass, every 5 s, the CPU used by processes OUTSIDE the timed job's process tree is read
from /proc. A pass in which that foreign CPU ever exceeds MAX_FOREIGN cores was contended
-- something started underneath it -- and is rerun (up to 3 attempts). Every attempt, kept
or rejected, is appended to ``$SI_SPEED_CENSUS/out/ledger.tsv`` with its peak foreign CPU,
so which timing files are reportable is recorded rather than remembered; ``emit_speed.py``
prints a pass only if the ledger kept it.

The queue is the REDUCE-using batch tools with eight preparation workers (two repeats each)
and the two per-invocation DeepShift rows.

    python retime_idle_ledger.py
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import speed_common as sc

HERE = Path(__file__).resolve().parent
MAX_FOREIGN = 1.0     # cores; an idle desktop uses ~0.1
HZ = 100

PASSES = (
    [('run_batch_parallel_prep.py', a, '8', r)
     for r in ('3', '4') for a in ('sparta_batch', 'legolas_batch_cpu', 'ucbshift_x_batch')]
    + [('run_speed_census.py', '--arm', arm)
       for arm in ('deepshift_final_gpu_single_coldproc', 'deepshift_final_gpu_ens_coldproc')]
)


def load():
    one, five = Path('/proc/loadavg').read_text().split()[:2]
    return float(one), float(five)


def gpu_busy():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout
    return bool(out.strip())


def wait_idle():
    quiet = 0
    while quiet < 3:
        one, five = load()
        quiet = quiet + 1 if (one < 1.0 and five < 1.0 and not gpu_busy()) else 0
        time.sleep(60)


def cpu_by_pid():
    """pid -> (ppid, utime+stime jiffies) for every live process."""
    out = {}
    for d in Path('/proc').iterdir():
        if d.name.isdigit():
            try:
                f = (d / 'stat').read_text().rsplit(')', 1)[1].split()
                out[int(d.name)] = (int(f[1]), int(f[11]) + int(f[12]))
            except (OSError, IndexError):
                pass
    return out


def tree(root, procs):
    kids, todo = {root}, [root]
    while todo:
        p = todo.pop()
        for pid, (ppid, _) in procs.items():
            if ppid == p and pid not in kids:
                kids.add(pid)
                todo.append(pid)
    return kids


def timed(cmd):
    """Run one pass; return its clock and the peak CPU (cores) used outside its tree."""
    started = time.strftime('%Y-%m-%dT%H:%M:%S')
    child = subprocess.Popen([sys.executable, *cmd], cwd=HERE)
    me = {os.getpid()}
    before, peak = cpu_by_pid(), 0.0
    while child.poll() is None:
        time.sleep(5)
        now = cpu_by_pid()
        mine = tree(child.pid, now) | me
        foreign = sum(j - before[p][1] for p, (_, j) in now.items()
                      if p not in mine and p in before and j >= before[p][1])
        peak = max(peak, foreign / HZ / 5.0)
        before = now
    return started, time.strftime('%Y-%m-%dT%H:%M:%S'), child.returncode, peak


def main():
    argparse.ArgumentParser(description=__doc__.split('\n\n')[0]).parse_args()
    ledger = sc.out_dir() / 'ledger.tsv'
    sc.out_dir().mkdir(parents=True, exist_ok=True)
    if not ledger.exists():
        ledger.write_text('pass\tattempt\tstarted\tfinished\treturncode\tpeak_foreign_cores\tverdict\n')
    for cmd in PASSES:
        for attempt in (1, 2, 3):
            wait_idle()
            started, finished, rc, peak = timed(cmd)
            verdict = 'kept' if rc == 0 and peak <= MAX_FOREIGN else 'rejected'
            with ledger.open('a') as f:
                f.write(f"{' '.join(cmd)}\t{attempt}\t{started}\t{finished}\t{rc}\t{peak:.2f}\t{verdict}\n")
            print(f"{' '.join(cmd)} attempt {attempt}: peak foreign CPU {peak:.2f} cores -> {verdict}", flush=True)
            if verdict == 'kept':
                break
    print('done', flush=True)


if __name__ == '__main__':
    sys.exit(main())
