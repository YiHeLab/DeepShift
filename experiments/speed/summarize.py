#!/usr/bin/env python3
"""Aggregate the census timing records into ``$SI_SPEED_CENSUS/speed_census.json``.

Reads, per printed per-invocation arm, ``out/<arm>__full__r0.json`` (``run_speed_census.py``);
the DeepShift chains-finished counts in ``deepshift_coverage240_final.json`` and
``deepshift_coverage240_final_ensemble.json`` (``coverage240.py``); and, for each competitor,
the fold-6 benchmark records ``<bench>/<tool>/<structure_id>.json`` that ``run_arm.py``
wrote. ``--bench`` (default ``$DEEPSHIFT_BENCH_OUT``) names that directory; its
``fold6_tasks.tsv`` defines the crystal chains the finish rate is taken over.

Per arm, over the 203-chain cohort: the cohort mean wall clock per protein with a bootstrap
95 % interval over structures; the decomposition wall = a + b * residues, where ``a`` is the
fixed cost every invocation pays (process start, model load, hydrogenation) and ``b`` the
marginal cost of another residue, each with a protein-bootstrap interval; the cohort residue
count; and the chains the method finishes.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import speed_common as sc

SEED = 20260731
BOOT = 2000

# arm -> (display name, hardware, fold-6 benchmark directory of the tool)
ARMS = {
    'legolas_cpu': ('LEGOLAS', '20 CPU', 'legolas'),
    'nmrgnn_cpu': ('NMRGNN$^{\\dagger}$', '20 CPU', 'nmrgnn'),
    'sparta': ('SPARTA+', '20 CPU', 'sparta'),
    'shiftx2_noshifty': ('SHIFTX2, SHIFTY+ disabled', '20 CPU', 'shiftx2_noshifty'),
    'napshift': ('NapShift', '20 CPU', 'napshift'),
    'ucbshift_x': ('UCBShift-X (structure only)', '20 CPU', 'ucbshift_x'),
    'procs15': ('PROCS15', '20 CPU', 'procs15'),
    'deepshift_final_gpu_single_coldproc':
        ('DeepShift, single model, per-invocation', 'GPU', None),
    'deepshift_final_gpu_ens_coldproc': ('DeepShift, 5-model, per-invocation', 'GPU', None),
}
COVERAGE_FILES = {'deepshift_final_gpu_single_coldproc': 'deepshift_coverage240_final.json',
                  'deepshift_final_gpu_ens_coldproc': 'deepshift_coverage240_final_ensemble.json'}

# The bootstrap draws come from one generator that is consumed arm by arm, in the order
# the census summarized its arms: (arm, structures timed), the arm None where the census
# also timed a configuration this table does not print. Each such draw is taken and
# discarded so that the printed intervals do not depend on which arms are summarized.
STREAM = ([(None, 203)] * 5 + [(None, 60)] * 2
          + [(None, 203), ('legolas_cpu', 203), (None, 203), ('nmrgnn_cpu', 203),
             ('sparta', 203), ('shiftx2_noshifty', 203), (None, 203), ('napshift', 203),
             ('ucbshift_x', 203), (None, 203), ('procs15', 203), (None, 203),
             ('deepshift_final_gpu_single_coldproc', 203),
             ('deepshift_final_gpu_ens_coldproc', 203)])


def finish_rate(tool, bench):
    """n_ok / n_tasks over every crystal task of the fold-6 table, from the tool's records."""
    csv.field_size_limit(sys.maxsize)
    with open(Path(bench) / 'fold6_tasks.tsv') as fh:
        tasks = {t['structure_id'] for t in csv.DictReader(fh, delimiter='\t')
                 if t['source'] == 'xtal'}
    ok = found = 0
    for sid in tasks:
        path = Path(bench) / tool / f'{sid}.json'
        if path.exists():
            found += 1
            ok += json.loads(path.read_text()).get('status') == 'ok'
    return {'n_ok': ok, 'n_tasks': len(tasks), 'n_found': found}


def ols(x, y):
    a = np.vstack([np.ones_like(x), x]).T
    (b0, b1), *_ = np.linalg.lstsq(a, y, rcond=None)
    return float(b0), float(b1)


def summarize_arm(arm, idx, bench):
    name, hardware, tool = ARMS[arm]
    path = sc.out_dir() / f'{arm}__full__r0.json'
    records = [r for r in json.loads(path.read_text())['records'] if r['status'] == 'ok']
    wall = np.array([r['wall_sec'] for r in records])
    residues = np.array([r['n_aa'] for r in records], float)
    if len(wall) != idx.shape[1]:
        raise SystemExit(f'{path.name}: {len(wall)} timed structures, expected {idx.shape[1]}')

    boot_mean = np.sort(wall[idx].mean(1))
    a, b = ols(residues, wall)
    boot_ab = np.array([ols(residues[i], wall[i]) for i in idx])
    if tool:
        coverage = finish_rate(tool, bench)
    else:
        coverage = json.loads((sc.root() / COVERAGE_FILES[arm]).read_text())
        coverage = {'n_ok': coverage['n_ok'], 'n_tasks': coverage['n_tasks']}
    return {
        'arm': arm, 'method': name, 'hardware': hardware,
        'n_structures_timed': len(wall),
        'per_protein_mean_sec': float(wall.mean()),
        'per_protein_mean_ci95': [float(boot_mean[int(.025 * BOOT)]),
                                  float(boot_mean[int(.975 * BOOT)])],
        'fixed_cost_intercept_sec': a,
        'fixed_cost_intercept_ci95': [float(np.percentile(boot_ab[:, 0], 2.5)),
                                      float(np.percentile(boot_ab[:, 0], 97.5))],
        'marginal_ms_per_residue': 1000 * b,
        'marginal_ms_per_residue_ci95': [float(1000 * np.percentile(boot_ab[:, 1], 2.5)),
                                         float(1000 * np.percentile(boot_ab[:, 1], 97.5))],
        'n_residues_total': int(residues.sum()),
        'coverage_all240_fold6_xtal': coverage,
        'source_file': f'$SI_SPEED_CENSUS/out/{path.name}',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--bench', default=os.environ.get('DEEPSHIFT_BENCH_OUT'),
                    help='competitor benchmark directory (default $DEEPSHIFT_BENCH_OUT)')
    ns = ap.parse_args()
    if not ns.bench:
        raise SystemExit('--bench or DEEPSHIFT_BENCH_OUT is required')

    rng = np.random.default_rng(SEED)
    rows = {}
    for arm, n in STREAM:
        idx = rng.integers(0, n, size=(BOOT, n))
        if arm:
            rows[arm] = summarize_arm(arm, idx, ns.bench)
    census = {'artifact': 'deepshift inference-speed census', 'cohort_n': len(sc.cohort_ids()),
              'seed': SEED, 'bootstrap': BOOT, 'arms': list(rows.values())}
    (sc.root() / 'speed_census.json').write_text(json.dumps(census, indent=1) + '\n')
    for r in rows.values():
        print(f"{r['method'][:40]:40s} per_protein={r['per_protein_mean_sec']:7.3f}s  "
              f"fixed={r['fixed_cost_intercept_sec']:6.2f}s  n={r['n_structures_timed']}")
    print(f"\nwrote {sc.root() / 'speed_census.json'}")


if __name__ == '__main__':
    main()
