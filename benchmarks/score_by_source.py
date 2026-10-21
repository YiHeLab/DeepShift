#!/usr/bin/env python3
"""Per-source (af / nmr / xtal) x per-nucleus backbone RMSE+MAE for every fold-6 arm.

The scoring rules:

  * One common cell set: a cell is scored only if every listed method predicted it and
    the protein survived self-exclusion for every fair homology arm. Otherwise the
    per-method n differ and the rows are not comparable.
  * ``--master-cells`` replaces that per-run intersection with the frozen master cell
    set (:mod:`benchmarks.master_cells`), so every float scores the same cells whichever
    arms it happens to print. An arm below the coverage floor does not shrink the set:
    it is scored on the master cells it did emit, its ``n_cells`` is reported per
    nucleus, and its rows carry ``below_master_coverage``.
  * Split by structure source, so the three manuscript sub-tables come from one pass.
  * Bootstrap CI over label sets (one deposited protein each), not cells.

The fenced observed cells, raw deposited targets and per-method prochiral resolution are
score_validation.py's. Residuals are raw: prediction minus the re-referenced observed
value. Beside RMSE and MAE each arm carries the secondary-shift correlation, with a
cluster-bootstrap interval from the same draws. The bootstrap resamples whole label sets.

Leaky arms (`*_leaky`) are scored without the self-exclusion drop, on the same
proteins and cells as the fair arms.

`--metrics rmse,mae,medae` adds the median absolute error.
"""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepshift.statistics import (
    DEFAULT_SEED,
    secondary_shift_agreement,
    secondary_shift_replicates,
    weighted_median_replicates,
)

from . import cells as common
from . import config
from . import master_cells as master
from .score_validation import (
    _resolved_cells,
    fold6_rows,
    load_comp,
    one_letter,
    verify_self_exclusion,
)

#: Frozen benchmark authorities shipped with the code (rosters, excluded channels).
REFERENCE = Path(__file__).resolve().parent / 'reference'

BACKBONE = ['N', 'H', 'CA', 'CB', 'C', 'HA']
# The intervals are reported at the label set: structure instances of one label set
# share the deposited shifts, so resampling them separately narrows every interval.
UNIT = 'labelset'
#: How every residual is formed; the report records it.
RESIDUAL_CONVENTION = 'raw: prediction minus the re-referenced observed value'
SOURCES = ['xtal', 'af', 'nmr']
METRICS = ['rmse', 'mae', 'medae']
DEFAULT_METRICS = ['rmse', 'mae']


def tool_cells(tool, sids, rows_by_sid, cols, atom_of_col, partner, tasks, exc,
               enforce_selfexcl=True, verify=verify_self_exclusion):
    """-> (sid -> {(struct_resnum, col): (prediction, target)}, set of unusable sids).

    Both sides of every cell are kept: the error is their difference, and the
    secondary-shift agreement needs the two values themselves.

    `verify` is the self-exclusion predicate. A tiered scorer passes its own; nobody
    rebinds a name in this module to change what it means.
    """
    out, bad = {}, set()
    for sid in sids:
        comp = load_comp(tool, sid)
        if comp is None or comp.get('status') != 'ok':
            bad.add(sid)
            continue
        ok, _ = verify(tool, sid, tasks[sid], exc)
        if enforce_selfexcl and not ok:
            bad.add(sid)
            continue
        cpred = comp.get('predictions', {})
        d = {}
        for row in rows_by_sid[sid]:
            vec = np.full(len(cols), np.nan)
            c = cpred.get(row['struct_resnum'])
            if c is not None and one_letter(c.get('resname')) == row['res1']:
                for j in row['obs_cols']:
                    v = c.get(atom_of_col[j])
                    if v is not None:
                        vec[j] = float(v)
            for j, pair in _resolved_cells(vec, row['true_raw'], row['obs_cols'],
                                           row['aa'], partner, row['amb']).items():
                d[(row['struct_resnum'], j)] = pair
        if not d:
            bad.add(sid)          # status-ok but zero matched cells (multi-entity chain)
            continue
        out[sid] = d
    return out, bad


@dataclass(frozen=True)
class Cohort:
    """The common cell set every arm is scored on, ready to resample."""

    cells: list          # (structure_id, label_set_id, atom, (resnum, col), residue code)
    resid: dict          # arm -> residual per cell, ppm
    pred: dict           # arm -> prediction per cell, ppm
    target: dict         # arm -> the target that arm resolved, ppm
    residue: np.ndarray  # residue type per cell, for the secondary-shift centring
    atom: np.ndarray
    sid_arr: np.ndarray
    src: np.ndarray
    names: list
    common_sids: list
    master_cells: str = None   # the frozen cell set scored, or None for the arm intersection
    below_master: dict = None  # arm -> {nucleus} it covers less than the floor of
    trained_on: dict = None    # arm -> what was withheld as that arm's own training data

    @property
    def cohort_rule(self) -> str:
        return ("the frozen master cell set: every arm is scored on Master(nucleus), and "
                "an arm below the coverage floor on the master cells it emitted"
                if self.master_cells else
                "the cells every listed arm predicted, on the structures all of them kept")

    def flagged(self, arm: str, nucleus: str) -> bool:
        """Whether one arm covers less of this nucleus than the master floor asks."""
        return nucleus in (self.below_master or {}).get(arm, ())


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tools', nargs='+', required=True)
    ap.add_argument('--leaky', nargs='*', default=[],
                    help='arms scored without the self-exclusion drop, and so leaky')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--metrics', default=','.join(DEFAULT_METRICS),
                    help=f'comma list from {METRICS}')
    ap.add_argument('--allow-anticorrelated', action='store_true',
                    help='score the cells Methods excludes; for a deliberate comparison '
                         'of the two cohorts, never for a deployed float')
    ap.add_argument('--master-cells', dest='master_cells', default=None,
                    help='the frozen master cell set (benchmarks.master_cells): score every '
                         'arm on Master(nucleus) instead of this run of arms intersected')
    ap.add_argument('--out', default=str(common.OUT_ROOT / 'score_by_source.json'))
    return ap


def training_roster(arm: str) -> set:
    """The BMRB accessions ``arm`` trained on, or an empty set if it declares none.

    Read from the released corpus itself rather than a transcribed list, so the roster
    cannot drift from the files the competitor actually shipped.
    """
    name = config.ARM_TRAINING_ROSTERS.get(arm)
    if not name:
        return set()
    root = getattr(config.current(), name)
    listing = REFERENCE / config.ARM_CORPUS_LISTINGS.get(arm, "")
    if not root.is_dir() and listing.is_file():
        # The corpus is third-party and not redistributed: its frozen file listing stands
        # in for it.
        return set(json.loads(listing.read_text())["bmrb_ids"])
    return {f.name[3:-3] for f in root.iterdir()
            if f.name.startswith("bmr") and f.name.endswith(".cs")}


def same_protein_roster(arm: str) -> dict:
    """The frozen same-protein roster of ``arm``."""
    return json.loads((REFERENCE / config.ARM_SAME_PROTEIN_ROSTERS[arm]).read_text())


def drop_trained_on(arm, cells_by_sid, tasks) -> tuple:
    """Remove the structures of every protein ``arm`` trained on, and say how many went.

    A structure is withheld when its label set is in the arm's frozen same-protein roster
    (same BMRB, shared UniProt, or >=95 % identity over >=90 % of the shorter sequence).
    Every label set whose own BMRB is in the corpus must be in that roster, or the
    roster is stale and nothing is scored. Returns ``(kept, dropped)``; ``dropped``
    records what was withheld, for the report and the caption that quotes it.
    """
    corpus = training_roster(arm)
    if not corpus:
        return cells_by_sid, None
    roster = same_protein_roster(arm)
    same = set(roster["label_set_ids"])
    floor = {t["label_set_id"] for t in tasks.values() if t["bmrb_id"] in corpus}
    if floor - same:
        raise SystemExit(f"{len(floor - same)} label sets whose own BMRB is in the {arm} "
                         "training corpus are missing from its same-protein roster")
    hit = [s for s in cells_by_sid if tasks[s]["label_set_id"] in same]
    kept = {s: v for s, v in cells_by_sid.items() if s not in hit}
    return kept, {"roster": config.ARM_TRAINING_ROSTERS[arm],
                  "roster_size": len(corpus),
                  "same_protein_roster": config.ARM_SAME_PROTEIN_ROSTERS[arm],
                  "structure_instances": len(hit),
                  "label_sets": len({tasks[s]["label_set_id"] for s in hit}),
                  "depositions": len({tasks[s]["bmrb_id"] for s in hit}),
                  "same_bmrb_depositions": len({tasks[s]["bmrb_id"] for s in hit
                                                if tasks[s]["bmrb_id"] in corpus}),
                  "rule": "a structure of the same protein as any entry of this arm's "
                          "training corpus (same BMRB, shared UniProt, or >=95% identity "
                          "over >=90% of the shorter sequence) is dropped from this arm "
                          "before scoring; every other arm keeps the whole master set"}


def build_cohort(ns, *, verify=verify_self_exclusion) -> Cohort:
    """Load every arm and lay them on one cell set: the frozen master cells when
    ``--master-cells`` names them, otherwise the cells every listed arm predicted.

    This is the whole cohort definition, and the only place it exists. A scorer that
    needs it -- marginal intervals here, paired intervals in score_paired_tier -- calls
    this and gets the same cells, in the same order, from the same rule. On the master
    cells an arm carries nan where it produced nothing, so it is scored on what it
    produced and every other arm keeps the whole set.
    """
    with open(common.OUT_ROOT / 'fold6_tasks.tsv') as fh:
        tasks = {t['structure_id']: t for t in csv.DictReader(fh, delimiter='\t')}
    sids = list(tasks)
    source_of = {s: tasks[s]['source'] for s in sids}
    exc = common.SelfExcluder()

    print(f'fold-6 rows for {len(sids)} structure instances (fenced observed cells)...')
    rows_by_sid, cols, atom_of_col, partner = fold6_rows(sids)
    col_of_atom = {a: [j for j in range(len(cols)) if atom_of_col[j] == a] for a in BACKBONE}
    keep_cols = {j for a in BACKBONE for j in col_of_atom[a]}

    methods = {}
    bad_union = set()
    trained_exclusions = {}
    for tool in ns.tools:
        leaky = tool in ns.leaky
        r, bad = tool_cells(tool, sids, rows_by_sid, cols, atom_of_col, partner,
                            tasks, exc, enforce_selfexcl=not leaky, verify=verify)
        r, trained_on = drop_trained_on(tool, r, tasks)
        if trained_on:
            trained_exclusions[tool] = trained_on
            print(f'  {tool:26s} withheld {trained_on["structure_instances"]:4d} structure '
                  f'instances on {trained_on["depositions"]} depositions in its own '
                  f'training set')
        methods[tool] = r
        if not leaky:                      # leaky arms never define the common set
            bad_union |= bad
        else:
            bad_union |= (bad - set(r))
        print(f'  {tool:26s} usable={len(r):5d} dropped={len(bad):4d}'
              f'{"  [leaky: self-exclusion not enforced]" if leaky else ""}')

    ours = {sid: {(row['struct_resnum'], j): pair
                  for row in rows_by_sid[sid]
                  for j, pair in _resolved_cells(row['our_pred'], row['true_raw'],
                                                 row['obs_cols'], row['aa'], partner,
                                                 row['amb']).items()}
            for sid in sids}
    methods['deepshift'] = ours

    frozen = getattr(ns, 'master_cells', None)
    if frozen:
        scorable = master.master_index(
            master.load_master(frozen,
                               allow_anticorrelated=getattr(ns, 'allow_anticorrelated', False)),
            cols)
        below = master.below_coverage(master.read_report(frozen))
        common_sids = [s for s in sids if s in scorable]
    else:
        scorable, below = None, {}
        common_sids = [s for s in sids if s not in bad_union
                       and all(s in m for m in methods.values())]
    print(f'\ncommon structure instances: {len(common_sids)} / {len(sids)}')

    # ---- the scored cells: the frozen master set, or every method's own intersection ----
    cells = []                     # (sid, label_set_id, atom, (resnum, col), residue code)
    for sid in common_sids:
        ls = tasks[sid]['label_set_id']
        for row in rows_by_sid[sid]:
            for j in row['obs_cols']:
                if j not in keep_cols:
                    continue
                k = (row['struct_resnum'], j)
                inside = (k in scorable[sid] if scorable is not None
                          else all(k in m[sid] for m in methods.values()))
                if inside:
                    cells.append((sid, ls, atom_of_col[j], k, row['aa']))
    # A structure carrying no scored cell of these nuclei is not in this cohort, whatever
    # else it carries: the counts a caption prints are of the cells below.
    scored_sids = {c[0] for c in cells}
    common_sids = [s for s in common_sids if s in scored_sids]
    print(f'scored backbone cells: {len(cells):,} '
          f'over {len(common_sids)} structure instances')

    names = list(methods)
    # An arm that did not emit a master cell carries nan there, so it is absent from that
    # cell and from no other arm's count.
    absent = (float('nan'), float('nan'))
    pred, target = {}, {}
    for n in names:
        found = methods[n]
        pairs = [found.get(c[0], {}).get(c[3], absent) for c in cells]
        pred[n] = np.array([p for p, _t in pairs], float)
        target[n] = np.array([t for _p, t in pairs], float)
    resid = {n: pred[n] - target[n] for n in names}
    residue = np.array([c[4] for c in cells])
    atom = np.array([c[2] for c in cells])
    sid_arr = np.array([c[0] for c in cells])
    src = np.array([source_of[c[0]] for c in cells])
    return Cohort(cells=cells, resid=resid, pred=pred, target=target, residue=residue,
                  atom=atom, sid_arr=sid_arr, src=src, names=names,
                  common_sids=common_sids,
                  master_cells=frozen, below_master=below,
                  trained_on=trained_exclusions or None)


def main(argv=None, *, verify=verify_self_exclusion):
    ns = build_parser().parse_args(argv)
    metrics = [m.strip() for m in ns.metrics.split(',') if m.strip()]
    if not metrics or set(metrics) - set(METRICS):
        raise SystemExit(f'--metrics {ns.metrics!r}: choose from {METRICS}')

    cohort = build_cohort(ns, verify=verify)
    cells, resid, atom, residue = cohort.cells, cohort.resid, cohort.atom, cohort.residue
    sid_arr, src, names, common_sids = cohort.sid_arr, cohort.src, cohort.names, cohort.common_sids

    rng = np.random.default_rng(ns.seed)
    key_arr = np.array([c[1] for c in cells])
    report = {'n_common_structures': len(common_sids), 'n_common_cells': len(cells),
              'unit': UNIT, 'boot': ns.boot, 'seed': ns.seed,
              'methods': names, 'residual_convention': RESIDUAL_CONVENTION,
              'cohort_rule': cohort.cohort_rule, 'master_cells': cohort.master_cells,
              'arms_withheld_own_training_data': cohort.trained_on or {},
              'arms_below_master_coverage': {a: sorted(n) for a, n in
                                             (cohort.below_master or {}).items()},
              'secondary_shift_convention': "r is on the secondary shift: "
              "each residue type's observed mean removed within the scored cohort",
              'metrics': metrics, 'by_source': {}}

    for source in [*SOURCES, 'all']:
        smask = np.ones(len(cells), bool) if source == 'all' else (src == source)
        s_sids = sorted(set(sid_arr[smask]))
        keys = sorted(set(key_arr[smask]))
        key_index = {k: i for i, k in enumerate(keys)}
        unit = np.array([key_index[k] for k in key_arr[smask]])
        boots = rng.integers(0, len(keys), size=(ns.boot, len(keys)))
        # how many times each draw uses each unit -- the same resample matrix as `boots`,
        # in the form a weighted median needs
        mult = (np.stack([np.bincount(b, minlength=len(keys)) for b in boots])
                if 'medae' in metrics else None)
        entry = {'n_structures': len(s_sids), 'n_units': len(keys), 'nuclei': {}}
        for a in BACKBONE:
            amask = (atom[smask] == a)
            n_cells = int(amask.sum())
            if n_cells == 0:
                continue
            unit_a = unit[amask]
            nuc = {'n_cells': n_cells}
            for n in names:
                # An arm is scored on the cells it emitted: every scored cell where the
                # set is one intersection, the master cells it covers where it is not.
                v = resid[n][smask][amask]
                emitted = np.isfinite(v)
                if not emitted.any():
                    continue         # an arm with no cell of this nucleus has no entry
                u_e, v_e = unit_a[emitted], v[emitted]
                sq_u = np.bincount(u_e, weights=v_e ** 2, minlength=len(keys))
                ab_u = np.bincount(u_e, weights=np.abs(v_e), minlength=len(keys))
                ct_u = np.bincount(u_e, minlength=len(keys)).astype(float)
                bs_sq = sq_u[boots].sum(1)
                bs_ab = ab_u[boots].sum(1)
                bs_ct = ct_u[boots].sum(1)
                ok = bs_ct > 0
                # metric -> (point estimate, bootstrap replicates)
                stat = {'rmse': (float(np.sqrt(sq_u.sum() / ct_u.sum())),
                                 np.sqrt(bs_sq[ok] / bs_ct[ok])),
                        'mae': (float(ab_u.sum() / ct_u.sum()), bs_ab[ok] / bs_ct[ok])}
                if 'medae' in metrics:
                    av = np.abs(v_e)
                    stat['medae'] = (float(np.median(av)),
                                     weighted_median_replicates(
                                         av, u_e, mult, even="mean"
                                     )[ok])
                # Secondary-shift agreement: each residue type's observed mean removed
                # from that arm's prediction and from the target it resolved, by the
                # package rule, so a per-residue-type offset cannot flatter a correlation.
                pred_a = cohort.pred[n][smask][amask][emitted]
                target_a = cohort.target[n][smask][amask][emitted]
                residue_a = residue[smask][amask][emitted]
                correlation, _, n_secondary = secondary_shift_agreement(
                    pred_a, target_a, residue_a)
                r_draws, _ = secondary_shift_replicates(
                    pred_a, target_a, residue_a, u_e, boots)
                nuc[n] = {
                    'n_cells': int(emitted.sum()),
                    'below_master_coverage': cohort.flagged(n, a),
                    **{m: stat[m][0] for m in metrics},
                    **{f'{m}_ci95': [float(np.percentile(stat[m][1], 2.5)),
                                     float(np.percentile(stat[m][1], 97.5))] for m in metrics},
                    **{f'{m}_se': float(stat[m][1].std(ddof=1)) for m in metrics},
                    'r': correlation, 'n_secondary': n_secondary,
                    'r_ci95': [float(np.percentile(r_draws, 2.5)),
                               float(np.percentile(r_draws, 97.5))] if r_draws.size else None,
                }
            entry['nuclei'][a] = nuc
        report['by_source'][source] = entry
        head = metrics[0]
        print(f"\n=== {source}  (n_struct={len(s_sids)}, {head}) ===")
        hdr = ' '.join(f'{a:>16s}' for a in BACKBONE)
        print(f"{'method':26s} {hdr}")
        for n in names:
            cellstr = ' '.join(
                f"{entry['nuclei'][a][n][head]:8.3f}+-{entry['nuclei'][a][n][head + '_se']:5.3f}"
                if n in entry['nuclei'].get(a, {}) else ' ' * 16 for a in BACKBONE)
            partial = [a for a in BACKBONE if n in entry['nuclei'].get(a, {})
                       and entry['nuclei'][a][n]['n_cells'] < entry['nuclei'][a]['n_cells']]
            print(f'{n:26s} {cellstr}'
                  + (f'   [fewer cells on {",".join(partial)}]' if partial else ''))

    Path(ns.out).write_text(json.dumps(report, indent=2))
    print('\nwrote', ns.out)


if __name__ == '__main__':
    main()
