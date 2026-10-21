#!/usr/bin/env python3
"""Per-nucleus accuracy of DeepShift and the tier's side-chain competitors on fold 6.

Every nucleus the fold-6 ensemble emits is scored, not the six backbone nuclei
score_by_source.py compares, and each nucleus is scored on the cells its own arms share:
a nucleus both incumbents emit carries three arms, a nucleus one of them emits carries
two, and a nucleus neither emits carries DeepShift alone. No nucleus is dropped for want
of a comparator.

``--master-cells`` replaces that per-run intersection with the frozen master cell set
(:mod:`benchmarks.master_cells`): every arm of a nucleus is then scored on Master(nucleus),
an arm below the coverage floor on the master cells it emitted, with its own ``n_cells``
and a ``below_master_coverage`` flag beside its errors. A difference is always taken on
the cells the pair shares, so it stays paired.

Residuals are raw -- an arm's prediction minus the observed label that arm's own
prediction resolved, by ``deepshift.evaluate.resolve`` through
``score_validation._resolved_cells`` -- and nothing is centred. The scored cells are the
ones the run's exclusion fence left in the fold-6 dumps, less the arginine N-epsilon
labels outside ``deepshift.quality_control.ARG_NE_OBSERVED_RANGE_PPM``. That fence reads
the observed value alone, so no prediction decides which labels are scored, and the cell
leaves the benchmark rather than one arm.

Per nucleus and arm, RMSE and MAE carry a 95% percentile interval from a cluster
bootstrap over label sets (``deepshift.statistics``), and one resample drives every arm
of a nucleus, so the interval on DeepShift minus a competitor is an interval on their
difference. The verdict is ``deepshift.statistics.paired_verdict``, read from DeepShift's
side.

Self-exclusion is the tier's, and nothing else here differs by tier:
``score_by_source_tier.verify_tiered`` per structure and ``check_arm_policy`` on each
arm's declared mode. A structure that fails closed for an arm is absent from that arm.

Usage (the same-protein tier the manuscript's Figure 3 (lower) reports):
  python -m benchmarks.score_by_nucleus --tier 2 \
      --tools ucbshift2_sameprot shiftx2_sameprot --out <out>/score_by_nucleus_tier2.json
"""
import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepshift.chemistry import STANDARD_RESIDUES
from deepshift.normalization import BACKBONE_SHIFTS
from deepshift.quality_control import ARG_NE_OBSERVED_RANGE_PPM
from deepshift.statistics import (
    DEFAULT_SEED,
    BootstrapSpec,
    bootstrap_indices,
    error_replicates,
    paired_verdict,
    percentile_interval,
    unit_codes,
)

from . import cells as common
from . import master_cells as master
from . import score_by_source as sbs
from . import score_by_source_tier as sbt
from .score_validation import _resolved_cells, fold6_rows

DEEPSHIFT = 'deepshift'
VERDICTS = ('win', 'tie', 'loss')

# A nucleus enters a comparison on evidence, not on existence: under a hundred cells an
# RMSE and its interval are a handful of depositions, and the bar would report which
# proteins happen to carry that atom rather than how well an arm predicts it. The floor
# is read on each arm's own coverage, so a competitor that emits a nucleus for a few
# residues cannot shrink the cell set the other arms are scored on.
MIN_CELLS = 100

# A spectrometer fold by one 15N sweep width lands part of an arginine N-epsilon
# deposition under the backbone nitrogen density. The interval those labels are held to
# is the package's.
ARG_NE_COLUMN = 'ne_shift'
ARGININE = 'ARG'

# statistics.compare's verdict is named for the arm it favours, with DeepShift the
# baseline of every difference here. The figure prints these words.
VERDICT_FROM_DEEPSHIFT = {'baseline': 'win', 'candidate': 'loss', 'tie': 'tie'}

RESIDUAL = ("raw: an arm's prediction minus the observed label its own prediction "
            "resolved, with no per-arm and no per-protein constant removed")
SCOPE_RULE = (
    f"every nucleus DeepShift emits, an arm emitting a nucleus when it covers at least "
    f"{MIN_CELLS} cells of it: the arms scored are DeepShift and the competitors that "
    f"emit the nucleus, on the cells all of them cover. Where that common set falls "
    f"below {MIN_CELLS} cells only the competitor covering the most cells is kept, and "
    f"where it is still below, DeepShift is scored alone: a nucleus in scope is never "
    f"dropped for want of a comparator")
MASTER_SCOPE_RULE = (
    f"every nucleus DeepShift emits, scored on the frozen master cell set for that "
    f"nucleus: every arm covering at least {MIN_CELLS} of its master cells is scored, on "
    f"the master cells it emitted, and an arm that covers fewer of them does not shrink "
    f"the set the others are scored on. Each arm reports its own n_cells and carries "
    f"below_master_coverage where it is under the floor the master set was frozen at")
VERDICT_RULE = (
    "per nucleus one cluster bootstrap over label sets drives every arm, so the "
    "percentile interval on DeepShift minus a competitor is an interval on their "
    "difference: win where that interval lies below zero, loss where it lies above, tie "
    "where it spans zero")


@dataclass(frozen=True)
class Cohort:
    """One residual per arm per scored cell, with the identities a bootstrap needs."""

    column: np.ndarray          # position of the cell's nucleus in the dump's columns
    label_set: np.ndarray
    structure: np.ndarray
    resid: dict                 # arm -> ppm residual per cell, nan where the arm is silent
    arms: list
    structures_per_arm: dict
    n_arg_ne_excluded: int
    cols: list
    atom_of_col: list
    observed: np.ndarray = None   # the observed ppm per scored cell, for the spread column
    in_master: np.ndarray = None  # cell is in the frozen master set, or None when unused
    master_cells: str = None      # the frozen set scored, or None for the arm intersection
    below_master: dict = None     # arm -> {nucleus} it covers less than the floor of


@dataclass(frozen=True)
class Errors:
    """One arm's error on one nucleus, with the replicates behind its intervals."""

    rmse: float
    mae: float
    rmse_draws: np.ndarray
    mae_draws: np.ndarray

    @classmethod
    def measure(cls, residual, unit, n_units, draws):
        """Both errors and their replicates, from one arm's residuals on one nucleus."""
        rmse, rmse_draws, mae, mae_draws = error_replicates(residual, unit, n_units, draws)
        return cls(rmse=rmse, mae=mae, rmse_draws=rmse_draws, mae_draws=mae_draws)

    def report(self) -> dict:
        return {'rmse': self.rmse, 'rmse_ci95': percentile_interval(self.rmse_draws),
                'mae': self.mae, 'mae_ci95': percentile_interval(self.mae_draws)}


def build_cohort(tier, tools, frozen=None) -> Cohort:
    """Load every arm over the fold-6 task table and lay their cells side by side.

    The cells are DeepShift's: it predicts every nucleus of every structure, so a
    competitor's coverage is a subset and enters as a residual beside ours or as nan.
    """
    with open(common.OUT_ROOT / 'fold6_tasks.tsv') as fh:
        tasks = {t['structure_id']: t for t in csv.DictReader(fh, delimiter='\t')}
    sids = list(tasks)
    excluder = common.SelfExcluder()

    print(f'fold-6 rows for {len(sids)} structure instances (fenced observed cells)...')
    rows_by_sid, cols, atom_of_col, partner = fold6_rows(sids)

    def verify(tool, sid, task, exc):
        return sbt.verify_tiered(tool, sid, task, exc, tier)

    competitors = {}
    for tool in tools:
        usable, dropped = sbs.tool_cells(tool, sids, rows_by_sid, cols, atom_of_col,
                                         partner, tasks, excluder, verify=verify)
        competitors[tool] = usable
        print(f'  {tool:26s} usable={len(usable):5d} dropped={len(dropped):4d}')

    low, high = ARG_NE_OBSERVED_RANGE_PPM
    arginine = STANDARD_RESIDUES.index(ARGININE)
    ne_column = cols.index(ARG_NE_COLUMN) if ARG_NE_COLUMN in cols else None
    scorable = (master.master_index(master.load_master(frozen), cols) if frozen else None)
    arms = [DEEPSHIFT, *tools]
    column, label_set, structure, observed, in_master = [], [], [], [], []
    resid = {arm: [] for arm in arms}
    excluded, silent = 0, {}
    for sid in sids:
        for row in rows_by_sid[sid]:
            ours = _resolved_cells(row['our_pred'], row['true_raw'], row['obs_cols'],
                                   row['aa'], partner, row['amb'])
            for j, (prediction, target) in ours.items():
                if (j == ne_column and row['aa'] == arginine
                        and not low <= row['true_raw'][j] <= high):
                    excluded += 1
                    continue
                column.append(j)
                label_set.append(row['label_set_id'])
                structure.append(sid)
                observed.append(target)
                if scorable is not None:
                    in_master.append((row['struct_resnum'], j) in scorable.get(sid, ()))
                resid[DEEPSHIFT].append(prediction - target)
                for tool in tools:
                    cell = competitors[tool].get(sid, silent).get((row['struct_resnum'], j))
                    resid[tool].append(np.nan if cell is None else cell[0] - cell[1])
    if not column:
        raise SystemExit(f'the {len(sids)} structure instances in '
                         f'{common.OUT_ROOT / "fold6_tasks.tsv"} name no cell of the '
                         f'fold-6 dumps, so there is nothing to score')
    print(f'{len(column):,} scored cells; {excluded} arginine {ARG_NE_COLUMN} labels '
          f'outside {low}-{high} ppm excluded from every arm')
    return Cohort(
        column=np.asarray(column), label_set=np.asarray(label_set),
        structure=np.asarray(structure),
        observed=np.asarray(observed, float),
        resid={arm: np.asarray(values, float) for arm, values in resid.items()},
        arms=arms,
        structures_per_arm={DEEPSHIFT: len(sids),
                            **{tool: len(usable) for tool, usable in competitors.items()}},
        n_arg_ne_excluded=excluded, cols=list(cols), atom_of_col=list(atom_of_col),
        in_master=np.asarray(in_master, bool) if frozen else None,
        master_cells=frozen,
        below_master=master.below_coverage(master.read_report(frozen)) if frozen else {})


def shared_cells(covered, chosen) -> np.ndarray:
    """The cells DeepShift and every chosen competitor cover."""
    keep = covered[DEEPSHIFT].copy()
    for tool in chosen:
        keep &= covered[tool]
    return keep


def choose_arms(covered, counts, competitors):
    """The competitors one nucleus is scored against, and the cells those arms share."""
    chosen = [tool for tool in competitors if counts[tool] >= MIN_CELLS]
    keep = shared_cells(covered, chosen)
    if keep.sum() < MIN_CELLS and len(chosen) > 1:
        chosen = [max(chosen, key=lambda tool: counts[tool])]
        keep = shared_cells(covered, chosen)
    if keep.sum() < MIN_CELLS and chosen:
        chosen = []
        keep = shared_cells(covered, chosen)
    return chosen, keep


def difference(ours: Errors, theirs: Errors) -> dict:
    """DeepShift minus one competitor on the cells they share, metric by metric.

    ``ours`` is DeepShift measured on those same cells, so the two sets of replicates
    come from one resample and their difference carries a paired interval.
    """
    out = {}
    for metric in ('rmse', 'mae'):
        draws = getattr(ours, f'{metric}_draws') - getattr(theirs, f'{metric}_draws')
        interval = percentile_interval(draws)
        out[metric] = {'deepshift': getattr(ours, metric),
                       'competitor': getattr(theirs, metric),
                       'diff': getattr(ours, metric) - getattr(theirs, metric),
                       'ci95': interval,
                       'verdict': VERDICT_FROM_DEEPSHIFT[paired_verdict(interval)]}
    return out


def score_nuclei(cohort: Cohort, competitors, spec: BootstrapSpec):
    """Score every nucleus DeepShift emits; return the nuclei and those below the floor."""
    unit, n_units = unit_codes(cohort.label_set)
    draws = bootstrap_indices(n_units, spec)
    by_nucleus, below_floor = {}, {}
    for j, column in enumerate(cohort.cols):
        seen = cohort.column == j
        covered = {arm: seen & np.isfinite(cohort.resid[arm]) for arm in cohort.arms}
        counts = {arm: int(mask.sum()) for arm, mask in covered.items()}
        if not counts[DEEPSHIFT]:
            continue                          # a nucleus this benchmark holds no cell of
        if counts[DEEPSHIFT] < MIN_CELLS:
            below_floor[cohort.atom_of_col[j]] = counts
            continue
        if cohort.in_master is None:
            chosen, keep = choose_arms(covered, counts, competitors)
        else:
            # The frozen set decides the cells; an arm that covers less of it is still
            # scored, on the master cells it emitted, and never shrinks the others'.
            keep = seen & cohort.in_master
            chosen = [tool for tool in competitors
                      if int((covered[tool] & keep).sum()) >= MIN_CELLS]
        # Every arm on the cells it emitted, and every difference on the cells its pair
        # shares: DeepShift covers the whole set, so a pair's cells are the competitor's.
        cells = {arm: np.flatnonzero(keep & covered[arm]) for arm in (DEEPSHIFT, *chosen)}
        scored = {arm: Errors.measure(cohort.resid[arm][rows], unit[rows], n_units, draws)
                  for arm, rows in cells.items()}
        ours_on = {tool: Errors.measure(cohort.resid[DEEPSHIFT][cells[tool]],
                                        unit[cells[tool]], n_units, draws)
                   for tool in chosen}
        whole = cells[DEEPSHIFT]
        by_nucleus[cohort.atom_of_col[j]] = {
            'backbone': column in BACKBONE_SHIFTS,
            'column': column,
            'n_cells': int(whole.size),
            'n_label_sets': int(np.unique(unit[whole]).size),
            'n_structures': int(np.unique(cohort.structure[whole]).size),
            'observed_sd': (float(np.std(cohort.observed[whole]))
                            if cohort.observed is not None else None),
            'cells_each_arm_covers': counts,
            'emitting_competitors': chosen,
            'arms': {arm: {**errors.report(), 'n_cells': int(cells[arm].size),
                           'below_master_coverage': cohort.atom_of_col[j] in
                           (cohort.below_master or {}).get(arm, ())}
                     for arm, errors in scored.items()},
            'vs_deepshift': {tool: difference(ours_on[tool], scored[tool])
                             for tool in chosen},
        }
    return by_nucleus, below_floor


def tally(by_nucleus, competitors) -> dict:
    """Wins, ties and losses per competitor, over every nucleus and over side chains."""
    out = {}
    for tool in competitors:
        counted = {}
        for scope, nodes in (('all_nuclei', list(by_nucleus.values())),
                             ('side_chain', [n for n in by_nucleus.values()
                                             if not n['backbone']])):
            seen = Counter(node['vs_deepshift'][tool]['rmse']['verdict'] for node in nodes
                           if tool in node['vs_deepshift'])
            counted[scope] = {verdict: int(seen[verdict]) for verdict in VERDICTS}
        out[tool] = counted
    return out


def build_report(cohort: Cohort, competitors, spec: BootstrapSpec, tier, by_nucleus,
                 below_floor) -> dict:
    """The report the figure draws, every field of it measured on this cohort."""
    side = [node for node in by_nucleus.values() if not node['backbone']]
    incumbents = Counter(len(node['emitting_competitors']) for node in side)
    low, high = ARG_NE_OBSERVED_RANGE_PPM
    return {
        'exclusion_tier': tier,
        'exclusion_policy': sbt.TIER_PREDICATE[tier],
        'scope_rule': MASTER_SCOPE_RULE if cohort.master_cells else SCOPE_RULE,
        'master_cells': cohort.master_cells,
        'arms_below_master_coverage': {a: sorted(n) for a, n in
                                       (cohort.below_master or {}).items()},
        'scope_counts': {
            'nuclei_deepshift_emits': len(by_nucleus),
            'nuclei_below_the_floor_for_deepshift': len(below_floor),
            'side_chain_nuclei': len(side),
            'side_chain_by_emitting_competitors': {str(number): int(count) for number, count
                                                   in sorted(incumbents.items())},
            'side_chain_with_no_incumbent': sorted(
                name for name, node in by_nucleus.items()
                if not node['backbone'] and not node['emitting_competitors']),
        },
        'arms': cohort.arms,
        'competitors': list(competitors),
        'residual': RESIDUAL,
        'exclusions_on_observed': {
            'run_fence': "the run's fitted exclusion fence, applied to the observed cells "
                         'when the fold-6 prediction dumps were written',
            'arg_ne_sweep_width': {
                'column': ARG_NE_COLUMN,
                'residue': ARGININE,
                'accepted_ppm': [low, high],
                'rule': f'an observed {ARGININE} {ARG_NE_COLUMN} outside '
                        f'[{low}, {high}] ppm is a sweep-width fold in the deposited '
                        f'label, and its cell leaves the benchmark for every arm',
                'n_cells_removed': cohort.n_arg_ne_excluded,
            },
        },
        'verdict_rule': VERDICT_RULE,
        'bootstrap': {'unit': spec.unit, 'resamples': spec.resamples, 'seed': spec.seed},
        'n_scored_cells': int(cohort.column.size),
        'n_structures_per_arm': cohort.structures_per_arm,
        'tally': tally(by_nucleus, competitors),
        'by_nucleus': by_nucleus,
        'nuclei_below_the_floor': below_floor,
    }


def print_report(report):
    arms = report['arms']
    print(f"\n{'nucleus':>8s} {'cells':>9s} " + ' '.join(f'{arm:>14s}' for arm in arms)
          + '   incumbents')
    for name, node in report['by_nucleus'].items():
        values = ' '.join(f"{node['arms'][arm]['rmse']:14.3f}" if arm in node['arms']
                          else ' ' * 14 for arm in arms)
        print(f"{name:>8s} {node['n_cells']:9,d} {values}   "
              f"{','.join(node['emitting_competitors']) or '-'}")
    for tool, counts in report['tally'].items():
        print(f"{tool:26s} side chain {counts['side_chain']}  "
              f"all nuclei {counts['all_nuclei']}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tier', type=int, choices=sorted(sbt.LEVEL_BY_TIER), required=True)
    ap.add_argument('--tools', nargs='+', required=True,
                    help="the tier's side-chain competitor arms")
    ap.add_argument('--boots', dest='boot', type=int, default=2000,
                    help='cluster-bootstrap resamples per nucleus')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--master-cells', dest='master_cells', default=None,
                    help='the frozen master cell set (benchmarks.master_cells)')
    ap.add_argument('--out', required=True)
    return ap


def main(argv=None):
    ns = build_parser().parse_args(argv)

    if Path(ns.out).exists():
        raise SystemExit(f'refusing to overwrite existing {ns.out}')

    with open(common.OUT_ROOT / 'fold6_tasks.tsv') as fh:
        tasks = {t['structure_id']: t for t in csv.DictReader(fh, delimiter='\t')}
    print(f'TIER {ns.tier}  ({len(tasks)} structure instances, OUT_ROOT={common.OUT_ROOT})')
    sbt.check_arm_policy(ns.tools, [], ns.tier, tasks)

    spec = BootstrapSpec(unit='label_set', resamples=ns.boot, seed=ns.seed)
    cohort = build_cohort(ns.tier, ns.tools, ns.master_cells)
    by_nucleus, below_floor = score_nuclei(cohort, ns.tools, spec)
    report = build_report(cohort, ns.tools, spec, ns.tier, by_nucleus, below_floor)
    Path(ns.out).write_text(json.dumps(report, indent=2))
    print_report(report)
    print('\nwrote', ns.out)
    return report


if __name__ == '__main__':
    main()
