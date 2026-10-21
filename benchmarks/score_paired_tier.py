#!/usr/bin/env python3
"""Paired cluster bootstrap for the fold-6 competitor benchmark (any tier).

score_by_source.py and score_by_source_tier.py resample each arm independently, so two
overlapping marginal intervals say nothing about the difference. The manuscript's
significance claims are about the difference, which needs one shared resample driving
DeepShift and the competitor together. That is all this script adds.

Nothing about the cohort is re-implemented: it calls score_by_source.build_cohort, the
one place the common-cell rule and the residual convention live, with
score_by_source_tier's tiered verifier so `--tier` means what it does there. Point
estimates are therefore identical cell for cell to the matching score_*.json, which
--check_marginal asserts.

Every difference is DeepShift minus a competitor, DeepShift being the run's own fold-6
evaluation packet.

The resampling unit is the label set, matching score_by_source.py.

Master cells. With ``--master-cells`` the cohort is the frozen master set, and each
difference is taken on the cells BOTH arms of that pair emitted -- the reference covers
every master cell, so a pair's cells are the competitor's own. The competitor's point
estimate therefore still equals the marginal report's, and the reference's is its error
on that competitor's cells, which ``--check_marginal`` compares only where the two arms
cover the nucleus equally.

Output: {source: {nucleus: {arm: {deepshift, competitor, diff, ci, p_deepshift_better}}}}
with diff = deepshift - competitor in ppm RMSE (negative = DeepShift better).

Usage (the same-protein tier the manuscript's Table 1 reports):
  python -m benchmarks.score_paired_tier --tier 2 \
      --tools ucbshift_full_sameprot ucbshift_x shiftx2_sameprot sparta \
      --out <out>/score_tier2_paired.json --check_marginal <out>/score_tier2.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from deepshift.statistics import DEFAULT_SEED

from . import cells as common
from . import score_by_source as sbs
from . import score_by_source_tier as sbt
from .score_by_source import BACKBONE, SOURCES

REFERENCE = 'deepshift'


def paired_bootstrap(cohort, boot, seed):
    """{source: {nucleus: {arm: {...}}}} -- one shared resample per source drives every arm."""
    cells, resid, atom, src = cohort.cells, cohort.resid, cohort.atom, cohort.src
    names = [n for n in cohort.names if n != REFERENCE]
    clus = np.array([str(c[1]) for c in cells])      # the label set of every cell

    rng = np.random.default_rng(seed)
    out = {}
    for source in [*SOURCES, 'all']:
        smask = np.ones(len(cells), bool) if source == 'all' else (src == source)
        keys = sorted(set(clus[smask]))
        index = {k: i for i, k in enumerate(keys)}
        unit_a = np.array([index[k] for k in clus[smask]])
        boots = rng.integers(0, len(keys), size=(boot, len(keys)))

        entry = {}
        for a in BACKBONE:
            amask = (atom[smask] == a)
            if not amask.any():
                continue
            u_a = unit_a[amask]

            def rmse(name, pair, *, smask=smask, amask=amask, u_a=u_a, boots=boots):
                """RMSE and its replicates for one arm on the cells of one pair."""
                v = resid[name][smask][amask][pair]
                sq_u = np.bincount(u_a[pair], weights=v ** 2, minlength=boots.shape[1])
                ct_u = np.bincount(u_a[pair], minlength=boots.shape[1]).astype(float)
                bs_ct = ct_u[boots].sum(1)
                ok = bs_ct > 0                  # same replicates for both arms of the pair
                bs_sq = sq_u[boots].sum(1)
                return float(np.sqrt(sq_u.sum() / ct_u.sum())), np.sqrt(bs_sq[ok] / bs_ct[ok])

            emitted = {name: np.isfinite(resid[name][smask][amask])
                       for name in (REFERENCE, *names)}
            nuc = {}
            for n in names:
                pair = emitted[REFERENCE] & emitted[n]
                if not pair.any():
                    continue          # no cell of this nucleus both arms of the pair have
                ds, ds_b = rmse(REFERENCE, pair)
                comp, comp_b = rmse(n, pair)
                d_b = ds_b - comp_b             # one resample, both arms: paired difference
                nuc[n] = {'deepshift': ds, 'competitor': comp, 'diff': ds - comp,
                          'n_cells': int(pair.sum()),
                          'ci': [float(np.percentile(d_b, 2.5)),
                                 float(np.percentile(d_b, 97.5))],
                          'p_deepshift_better': float((d_b < 0).mean())}
            entry[a] = nuc
        out[source] = entry
        n_units = len(keys)
        print(f'--- {source} (n_{sbs.UNIT}={n_units}) : DeepShift minus competitor RMSE, '
              f'negative = DeepShift better')
        for a in BACKBONE:
            for n in names:
                if a not in entry:
                    continue
                if n not in entry[a]:
                    continue
                e = entry[a][n]
                star = '*' if e['ci'][1] < 0 or e['ci'][0] > 0 else ' '
                print(f"  {a:3s} vs {n:26s} {e['deepshift']:6.3f} vs {e['competitor']:6.3f} "
                      f"diff {e['diff']:+.3f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]{star} "
                      f"P(DS better)={e['p_deepshift_better']:.3f}")
    return out


def check_marginal(report, path, tol=1e-9):
    """The point estimates must equal the marginal run's: same cells, same residuals.

    A competitor is always checked. The reference is checked where the pair covers the
    nucleus equally, which is every nucleus of a run on one intersection and every
    nucleus of a master run bar those an arm covers less of: there the reference's
    paired estimate is its error on the competitor's cells, which the marginal report
    does not carry.
    """
    with open(path) as fh:
        m = json.load(fh)
    if (m['n_common_structures'], m['n_common_cells']) != \
       (report['n_common_structures'], report['n_common_cells']):
        raise SystemExit(f'cohort differs from {path}: '
                         f"{report['n_common_structures']}/{report['n_common_cells']} vs "
                         f"{m['n_common_structures']}/{m['n_common_cells']}")
    worst, unpaired = 0.0, 0
    for source, srep in m['by_source'].items():
        for a, nuc in srep['nuclei'].items():
            for n, d in nuc.items():
                if n == 'n_cells' or n == REFERENCE:
                    continue
                got = report[source][a].get(n)
                if got is None:
                    continue          # a pair with no shared cell carries no difference
                worst = max(worst, abs(got['competitor'] - d['rmse']))
                if d.get('n_cells', nuc['n_cells']) == nuc[REFERENCE].get('n_cells',
                                                                         nuc['n_cells']):
                    worst = max(worst, abs(got['deepshift'] - nuc[REFERENCE]['rmse']))
                else:
                    unpaired += 1
    if worst > tol:
        raise SystemExit(f'point estimates differ from {path} by {worst:.3e}')
    print(f'point estimates match {path} (max |delta| = {worst:.2e}; {unpaired} '
          f'reference estimates on a partial arm\'s own cells, not compared)')


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', type=int, choices=sorted(sbt.LEVEL_BY_TIER), required=True)
    ap.add_argument('--tools', nargs='+', required=True)
    ap.add_argument('--leaky', nargs='*', default=[])
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--master-cells', dest='master_cells', default=None,
                    help='the frozen master cell set (benchmarks.master_cells)')
    ap.add_argument('--allow-anticorrelated', action='store_true',
                    help='score the cells Methods excludes, for its with/without comparison; '
                         'never for a deployed float')
    ap.add_argument('--out', required=True)
    ap.add_argument('--check_marginal', help='score_*.json whose point estimates must match')
    return ap


def main():
    ns = build_parser().parse_args()

    if Path(ns.out).exists():
        raise SystemExit(f'refusing to overwrite existing {ns.out}')

    with open(common.OUT_ROOT / 'fold6_tasks.tsv') as fh:
        tasks = {t['structure_id']: t for t in csv.DictReader(fh, delimiter='\t')}
    print(f'TIER {ns.tier}  ({len(tasks)} structure instances, OUT_ROOT={common.OUT_ROOT})')
    sbt.check_arm_policy(ns.tools, ns.leaky, ns.tier, tasks)

    fwd = ['--tools', *ns.tools, '--boot', str(ns.boot), '--seed', str(ns.seed),
           '--out', '/dev/null']            # never read: only the cohort is built here
    if ns.master_cells:
        fwd += ['--master-cells', ns.master_cells]
    if ns.allow_anticorrelated:
        fwd += ['--allow-anticorrelated']
    if ns.leaky:
        fwd += ['--leaky', *ns.leaky]
    cohort = sbs.build_cohort(
        sbs.build_parser().parse_args(fwd),
        verify=lambda tool, sid, task, exc: sbt.verify_tiered(tool, sid, task, exc, ns.tier))

    report = {'tier': ns.tier, 'unit': sbs.UNIT, 'boot': ns.boot, 'seed': ns.seed,
              'reference_arm': REFERENCE, 'methods': list(cohort.names),
              'residual_convention': sbs.RESIDUAL_CONVENTION,
              'cohort_rule': cohort.cohort_rule, 'master_cells': cohort.master_cells,
              'pair_rule': 'each difference is taken on the cells both arms of the pair '
                           'emitted',
              'n_common_structures': len(cohort.common_sids),
              'n_common_cells': len(cohort.cells),
              'diff_convention': 'diff = deepshift - competitor (ppm RMSE); negative = DeepShift better'}
    print(f"\ncommon structures {report['n_common_structures']}  "
          f"common cells {report['n_common_cells']:,}")
    report.update(paired_bootstrap(cohort, ns.boot, ns.seed))

    if ns.check_marginal:
        check_marginal(report, ns.check_marginal)

    Path(ns.out).write_text(json.dumps(report, indent=2))
    print('wrote', ns.out)


if __name__ == '__main__':
    main()
