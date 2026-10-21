#!/usr/bin/env python3
"""Entity-grained (label-set) CV fold assignment.

Union-find nodes are label sets. Hard leakage edges: same BMRB, shared UniProt
(all sources), exact/alignment token-hash, MMseqs 90% bidirectional/
containment, and 90% modeled-region. UCB-200 exclusion (same-BMRB/UniProt/
homology, at fold-0's own thresholds) runs before clustering. CV-eligible
components are bin-packed by a seeded greedy pass drawing a component-safe
fold-6 holdout and balancing residue mass jointly with source counts.
Folds: 0=excluded, 1-5=CV, 6=holdout.

Usage: python -m deepshift.data.folds --data_dir <dir> ..."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from collections.abc import Iterable

import pandas as pd

from .paths import DATA_DIR as _DATA_ROOT
from .paths import UCB_TEST_DIR

_ROOT = os.path.dirname(os.path.abspath(__file__))

N_FOLDS = 5
EXCLUDED_FOLD = 0   # fold-0 = held out from training entirely

# Honor SHIFT_DATA_ROOT (paths.py) while keeping the repo-relative default absolute (robust to cwd).
DATA_DIR    = _DATA_ROOT if os.path.isabs(_DATA_ROOT) else os.path.join(_ROOT, _DATA_ROOT)
# UCB_TEST_DIR (paths.py) is only a fallback to regenerate the shipped list.
UCB_EXCLUDED_BASENAME = 'ucbshift200_excluded_bmrbs.json'
# The UCBShift-200 test targets as JSON (<PDB4><CHAIN> -> one-letter sequence): the
# sequence authority of the fold-0 homology arm, named by SHIFT_UCB_TEST_TARGETS.
UCB_TEST_TARGETS_BASENAME = 'ucbshift200_test_targets.json'
# External case-study proteins (IDP8, tau, drkN SH3) held out of every training fold by
# the same direct rule as the UCBShift-200 boundary. They are pinned to fold 0 AFTER the
# seeded draw, so adding one leaves every other assignment, fold 6 included, unchanged.
CASE_STUDY_EXCLUDED_BASENAME = 'case_study_excluded_bmrbs.json'
# A near-identical segment is the same local structure whatever share of the construct it
# is, and the coverage floor above lets a long one through (a 76-residue tau segment at 99 %
# identity covers 59 % of K18). A training record also leaves with a case-study protein
# when an alignment of at least CASE_STUDY_SEGMENT_MIN_ALIGNED residues reaches
# CASE_STUDY_SEGMENT_IDENTITY, the identity the fold map itself treats as one protein.
CASE_STUDY_SEGMENT_IDENTITY = 0.90
CASE_STUDY_SEGMENT_MIN_ALIGNED = 40
HOLDOUT_FOLD = 6

# The benchmark-exclusion arm runs wider than the training-fold axis
# (cluster.MMSEQS_IDENTITY/COVERAGE): a homolog of a benchmark protein leaves training at
# identity >= 0.65 when the shorter sequence is covered >= 0.60. Coverage mode 1 covers
# the target and mode 2 the query; the union over both orderings is the coverage of the
# shorter side, so a fragment partner is found.
UCB_HOMOLOGY_IDENTITY = 0.65
UCB_HOMOLOGY_COVERAGE = 0.60
UCB_HOMOLOGY_COVERAGE_MODES = (1, 2)

# Seeded random split (an id hash can correlate with deposition order); assignment
# is at the identity/UniProt component level, so a draw never splits a twin.
SPLIT_SEED   = int(os.environ.get('SHIFT_SPLIT_SEED', '20260701'))
HOLDOUT_FRAC = float(os.environ.get('SHIFT_HOLDOUT_FRAC', '0.10'))
# Fold-6 gate: the drawn holdout must land in a 9-11% band of every mass dimension
# (label sets/BMRBs/residue) and hold each af/nmr/xtal source out representatively.
# The seeded draw is refused rather than redrawn on a miss: searching for a passing
# draw would select the split on the gate's own outcome.
HOLDOUT_MIN_FRAC = float(os.environ.get('SHIFT_HOLDOUT_MIN_FRAC', '0.09'))
HOLDOUT_MAX_FRAC = float(os.environ.get('SHIFT_HOLDOUT_MAX_FRAC', '0.11'))
# Fold-size balance gate: max fractional deviation of any CV fold's BMRB count from
# the mean, dealt to the least-loaded fold; a broken split blows past it and refuses
# to write. Row counts belong to the split stage, not here.
FOLD_BALANCE_TOL = float(os.environ.get('SHIFT_FOLD_BALANCE_TOL', '0.15'))


class UnionFind:
    """Iterative union-find with path compression + rank."""

    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {x: x for x in items}
        self.rank = dict.fromkeys(items, 0)

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def load_bmrb_to_pdbs(pairs_csv: str) -> dict[str, set[str]]:
    """data/pairs.csv: Entry_ID, pdb_ids (comma-separated, 4-char each).
    Returns: bmrb_id -> set of uppercased 4-char PDB IDs.
    """
    df = pd.read_csv(pairs_csv, dtype={'Entry_ID': str})
    out: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        bmrb = str(r['Entry_ID'])
        raw = str(r['pdb_ids']) if not pd.isna(r['pdb_ids']) else ''
        pdbs = {p.strip().upper() for p in raw.split(',') if p.strip()}
        if pdbs:
            out[bmrb] = pdbs
    return out


def load_ucb_test_pdb4(test_dir: str) -> set[str]:
    """4-char PDB IDs from the UCBShift test dir (filenames <PDB4><CHAIN>.pdb).
    Empty set if absent; callers refuse an empty set (see main())."""
    pdbs: set[str] = set()
    if not os.path.isdir(test_dir):
        print(f'WARN: UCB test dir not found: {test_dir}')
        return pdbs
    for fn in sorted(os.listdir(test_dir)):
        if not fn.endswith('.pdb'):
            continue
        stem = fn[:-4]                  # e.g. "1AM7A" or "109M_"
        if len(stem) >= 4:
            pdbs.add(stem[:4].upper())
    return pdbs


def load_shipped_excluded_bmrbs(json_path: str) -> set[str] | None:
    """Shipped, machine-independent UCBShift-200 excluded-BMRB list.
    Returns a set of BMRB ids, or None if absent (caller falls back to scraping
    UCB_TEST_DIR). Accepts a bare JSON list or a dict with a "bmrbs" key, optionally
    also "test_200_bmrbs" (the 200 benchmark BMRBs themselves; checked to be a
    subset of "bmrbs", so a declared benchmark entry cannot be left unexcluded and
    silently disable its same-BMRB/UniProt arms)."""
    if not os.path.isfile(json_path):
        return None
    with open(json_path) as fh:
        obj = json.load(fh)
    if isinstance(obj, dict):
        if 'bmrbs' not in obj:
            raise ValueError('UCB exclusion object is missing bmrbs')
        declared = {str(b).strip() for b in obj.get('test_200_bmrbs', ())}
        missing_benchmark = declared - {str(b).strip() for b in obj['bmrbs']}
        if missing_benchmark:
            raise ValueError(
                'UCB exclusion omits benchmark proteins it declares in '
                f'test_200_bmrbs: {sorted(missing_benchmark)}')
        obj = obj['bmrbs']
    if not isinstance(obj, list):
        raise ValueError('UCB exclusion must be a JSON list or {"bmrbs": [...]}')
    values = [str(b).strip() for b in obj]
    if any(not value or not value.isdigit() for value in values):
        raise ValueError('UCB exclusion contains a nonnumeric/blank BMRB id')
    if len(values) != len(set(values)):
        raise ValueError('UCB exclusion contains duplicate BMRB ids')
    return set(values)


def _packaged_ucb_excluded_bmrbs() -> set[str]:
    """Tracked, code-hashed authority for the definitive UCBShift-200 boundary."""
    packaged = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'ucbshift200_excluded_bmrbs.json')
    expected = load_shipped_excluded_bmrbs(packaged)
    if not expected:
        raise RuntimeError(f'packaged UCBShift-200 authority is empty: {packaged}')
    return expected


def load_case_study_exclusion(json_path: str) -> tuple[set[str], dict[str, str]]:
    """The packaged case-study hold-out: ``{"bmrbs": [...], "sequences": {name: seq}}``.
    BMRBs seed the same-deposition and same-UniProt arms; the sequences seed the homology
    arm (so a protein absent from the corpus, or a peptide, still excludes its homologs)."""
    with open(json_path) as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict) or 'bmrbs' not in obj or 'sequences' not in obj:
        raise ValueError(f'{json_path}: case-study exclusion needs "bmrbs" and "sequences"')
    bmrbs = [str(b).strip() for b in obj['bmrbs']]
    if not bmrbs or any(not b.isdigit() for b in bmrbs) or len(bmrbs) != len(set(bmrbs)):
        raise ValueError(f'{json_path}: BMRB ids must be unique numeric strings')
    sequences = {str(k): str(v).strip().upper() for k, v in obj['sequences'].items()}
    bad = [k for k, v in sequences.items()
           if not v or any(c not in 'ACDEFGHIKLMNPQRSTVWYX' for c in v)]
    if bad:
        raise ValueError(f'{json_path}: invalid one-letter sequences for {bad}')
    return set(bmrbs), sequences


def pin_after_draw(ls_to_fold: dict, seed) -> dict:
    """Move ``seed`` to fold 0 in a finished assignment; returns
    ``{label_set: previous fold}`` for every label set that moved.

    The case-study exclusion is applied after the draw rather than through it so that
    every other assignment stays byte-identical. A seed member the draw put in fold 6 is
    refused: moving it would change the held-out benchmark, which is the operator's call.
    """
    in_holdout = sorted(ls for ls in seed if ls_to_fold.get(ls) == HOLDOUT_FOLD)
    if in_holdout:
        raise RuntimeError(
            f'{len(in_holdout)} case-study label sets were drawn into fold {HOLDOUT_FOLD}; '
            f'pinning them would change the held-out benchmark: {in_holdout[:5]}')
    moved = {ls: ls_to_fold[ls] for ls in sorted(seed)
             if ls in ls_to_fold and ls_to_fold[ls] != EXCLUDED_FOLD}
    for ls in moved:
        ls_to_fold[ls] = EXCLUDED_FOLD
    return moved


def _uniprots_of(bmrbs, uniprot_map) -> set[str]:
    """Every validated UniProt of these depositions."""
    out = set()
    for b in bmrbs:
        ups = uniprot_map.get(str(b))
        if not ups:
            continue
        out |= {str(u) for u in (ups if isinstance(ups, (list, tuple, set)) else [ups]) if u}
    return out


def case_study_segment_matched_labelsets(ls_rec, case_sequences, *, mmseqs_binary=None,
                                         identity=CASE_STUDY_SEGMENT_IDENTITY,
                                         min_aligned=CASE_STUDY_SEGMENT_MIN_ALIGNED,
                                         search_fn=None) -> dict:
    """``{label_set_id: {'case', 'identity', 'aligned'}}`` for every universe label set
    sharing a segment of at least ``min_aligned`` aligned residues at ``identity`` or more
    with a case-study sequence, whatever the coverage. ``search_fn`` replaces MMseqs."""
    targets = {ls: (rec.get('entity_sequence') or rec.get('modeled_sequence') or '')
               for ls, rec in ls_rec.items()}
    targets = {ls: seq for ls, seq in targets.items() if seq}
    queries = {f'__case__{name}': seq for name, seq in case_sequences.items() if seq}
    if not targets or not queries:
        return {}
    if search_fn is None:
        from . import sequence_clustering as _search
        hits = _search.run_segment_search(queries, targets, identity=identity,
                                          binary=mmseqs_binary)
    else:
        hits = search_fn(queries, targets)
    matched = {}
    for (query, ls), hit in hits.items():
        if hit['aligned'] < min_aligned or hit['identity'] < identity:
            continue
        record = {'case': query.replace('__case__', '', 1),
                  'identity': hit['identity'], 'aligned': hit['aligned']}
        if ls not in matched or hit['aligned'] > matched[ls]['aligned']:
            matched[ls] = record
    return matched


def case_study_boundary(seed, routes, direct, ls_to_fold, ls_rec, full_edge_families,
                        training_folds=None) -> dict:
    """The external-boundary report for the case-study proteins.

    ``prohibited`` is the gate: a label set that satisfies the direct rule itself
    (``direct``: same deposition, same UniProt, homolog or shared segment) and is still in
    a training fold. It is empty unless the seed or its closure has a bug.

    ``retained_training_relatives`` is for inspection, not a gate: the training label sets
    that share a full-graph edge GROUP with a seed record. A group is transitive, so it
    chains through partner chains and fragments to proteins with no relation to the case
    study; discarding every group would remove most of the corpus. Each entry names the
    seed record it is grouped with, so a reader can judge the relation.
    """
    training = set(TRAINING_FOLDS if training_folds is None else training_folds)
    retained = {}
    for family, groups in full_edge_families.items():
        for group in groups:
            members = set(group)
            if not members & seed:
                continue
            for ls in sorted(members - seed):
                if ls_to_fold.get(ls) in training:
                    retained.setdefault(family, {})[ls] = {
                        'bmrb': str(ls_rec[ls]['bmrb_id']), 'fold': ls_to_fold[ls],
                        'with_seed': sorted(members & seed)[:3]}
    prohibited = {ls: {'bmrb': str(ls_rec[ls]['bmrb_id']), 'fold': ls_to_fold[ls],
                       'routes': sorted(routes.get(ls, ()))}
                  for ls in sorted(direct) if ls_to_fold.get(ls) in training}
    return {
        'rule': {
            'direct': f'same deposition, same UniProt, or homolog at identity >= '
                      f'{UCB_HOMOLOGY_IDENTITY} over coverage >= {UCB_HOMOLOGY_COVERAGE} '
                      'of the shorter sequence; same-deposition closure',
            'segment': f'alignment of >= {CASE_STUDY_SEGMENT_MIN_ALIGNED} residues at '
                       f'identity >= {CASE_STUDY_SEGMENT_IDENTITY}, any coverage',
            'gate': 'no label set satisfying the rule itself may stay in a training fold',
            'retained_relatives': 'transitive edge groups, reported for inspection only'},
        'seed': {ls: {'bmrb': str(ls_rec[ls]['bmrb_id']), 'routes': sorted(routes.get(ls, ()))}
                 for ls in sorted(seed) if ls in ls_rec},
        'retained_training_relatives': {family: dict(sorted(hits.items()))
                                        for family, hits in sorted(retained.items())},
        'prohibited': prohibited,
    }


def case_study_boundary_summary(boundary) -> str:
    """One line for the build log: the training relatives retained by family, and how
    many label sets the gate will refuse. ``prohibited`` holds one record per label set,
    so it is counted, never summed."""
    retained = {family: len(hits)
                for family, hits in boundary['retained_training_relatives'].items()}
    return (f'case-study boundary: retained training relatives by family '
            f'{json.dumps(retained)}; prohibited {len(boundary["prohibited"])}')


def resolve_ucb_excluded_bmrbs(
    json_path: str,
    bmrb_pool: set[str],
    *,
    pairs_csv: str,
    test_dir: str,
    allow_empty: bool = False,
) -> set[str]:
    """Resolve the UCBShift-200 boundary, preferring the shipped BMRB list; an
    absent/empty list falls back to scraping the UCB coordinate dir and projecting
    PDB ids through pairs.csv. The result must contain the packaged benchmark
    authority; empty/stale is permitted only via an explicit CLI override."""
    try:
        shipped = load_shipped_excluded_bmrbs(json_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        if not allow_empty:
            raise RuntimeError(
                f'UCBShift-200 exclusion is stale or wrong: invalid authority '
                f'{json_path}: {exc}') from exc
        print(f'WARN: explicitly overriding invalid UCB exclusion {json_path}: {exc}')
        shipped = None
    excluded = set(shipped or ())
    source = f'shipped JSON {json_path}'
    if not excluded:
        test_pdbs = load_ucb_test_pdb4(test_dir)
        pairs = load_bmrb_to_pdbs(pairs_csv) if os.path.isfile(pairs_csv) else {}
        excluded = {
            bmrb for bmrb, pdbs in pairs.items() if pdbs & test_pdbs
        }
        source = f'UCB_TEST_DIR scrape {test_dir}'
        print(f'  UCB fallback test PDBs: {len(test_pdbs):,}; '
              f'paired BMRBs: {len(excluded):,}')

    in_pool = excluded & {str(b) for b in bmrb_pool}
    print(f'  UCBShift-200 excluded BMRBs: {len(excluded):,} ({source})')
    print(f'    of which in the entity fold pool: {len(in_pool):,}')
    if not excluded and not allow_empty:
        raise RuntimeError(
            'UCBShift-200 exclusion set is EMPTY: neither the shipped exclusion '
            f'file ({json_path}) nor the UCB test-dir fallback ({test_dir}) '
            'yielded a BMRB. Refusing benchmark train-on-test; restore the '
            'exclusion authority or explicitly pass --allow_empty_ucb_exclusion.')
    expected = _packaged_ucb_excluded_bmrbs()
    missing_expected = expected - excluded
    if missing_expected and not allow_empty:
        raise RuntimeError(
            'UCBShift-200 exclusion is stale or wrong: it does not contain the '
            f'packaged {len(expected)}-BMRB benchmark authority '
            f'({len(missing_expected)} missing; examples '
            f'{sorted(missing_expected)[:8]}). Refusing benchmark train-on-test; '
            'restore the packaged exclusion or explicitly pass '
            '--allow_empty_ucb_exclusion.')
    if missing_expected:
        print('WARN: --allow_empty_ucb_exclusion explicitly overrides an '
              f'unauthenticated UCB boundary ({len(missing_expected)} packaged '
              'BMRBs missing).')
    return excluded


OUT_LABELSET_FOLD_MAP = os.path.join(DATA_DIR, 'fold_assignments_labelset_identity90.json')
OUT_LABELSET_BMRB_MAP = os.path.join(DATA_DIR, 'fold_assignments_labelset_derived_bmrb.json')
OUT_QUARANTINE = os.path.join(DATA_DIR, 'quarantine_short_entities.csv')
# Edge families that must not span two training folds.
LEAKAGE_FAMILIES = ('same_bmrb', 'uniprot', 'exact_token', 'alignment',
                    'mmseqs_cov0', 'mmseqs_containment', 'partner_chain',
                    'modeled_region')
# this stage's own edge tables (same_bmrb/uniprot/modeled_region are computed
# only here) plus their edge-group record and balance, so a cross-fold check can read them.
OUT_BALANCE_REPORT = os.path.join(DATA_DIR, 'fold_balance_report.json')
OUT_EDGE_PROVENANCE = os.path.join(DATA_DIR, 'labelset_edges_provenance.json')
# Fold-0 label sets that share an edge group with a training-fold label set. Fold 0 is
# the interval-calibration set, so a fold-0 protein with a training-fold relative would
# make the calibration residuals optimistic; the interval fit excludes this list.
OUT_FOLD0_RELATIVES = os.path.join(DATA_DIR, 'fold0_training_relatives.json')
# The case-study pin: every label set moved to fold 0, with its previous fold.
OUT_CASE_STUDY_PIN = os.path.join(DATA_DIR, 'fold_case_study_pin.json')
# What the interval fit must drop from fold 0: the label sets with a training-fold relative,
# and the case-study records. Fold 0 is the calibration set as well as the excluded fold, so a
# case-study protein left in it would calibrate the uncertainty of the very applications it is
# held out of.
OUT_CALIBRATION_EXCLUSIONS = os.path.join(DATA_DIR, 'fold0_calibration_exclusions.json')
# The external boundary: how each case-study seed record was caught, and every training
# record that still shares a full-graph edge group with one, by family, with the decision.
OUT_CASE_STUDY_BOUNDARY = os.path.join(DATA_DIR, 'fold_case_study_boundary.json')
TRAINING_FOLDS = tuple(range(1, N_FOLDS + 1))


def labelset_edge_table_path(family, data_dir=DATA_DIR):
    return os.path.join(data_dir, f'labelset_edges_{family}.csv')


def write_labelset_edge_tables(edge_families, ls_rec, ls_to_fold, *,
                               data_dir=DATA_DIR, extra_provenance=None):
    """Publish every label-set leakage edge family as labelset_edges_<family>.csv, plus
    labelset_edges_provenance.json with the edge-group counts and the fold stage's gate
    settings. Returns that record."""
    os.makedirs(data_dir, exist_ok=True)
    counts = {}
    for family in LEAKAGE_FAMILIES:
        groups = edge_families.get(family, [])
        path = labelset_edge_table_path(family, data_dir)
        with open(path, 'w', newline='') as fh:
            import csv as _csv
            w = _csv.writer(fh)
            w.writerow(['edge_group', 'label_set_id', 'bmrb_id', 'fold'])
            for gid, group in enumerate(groups):
                for ls in group:
                    rec = ls_rec.get(ls, {})
                    w.writerow([gid, ls, rec.get('bmrb_id', ''),
                                ls_to_fold.get(ls, '')])
        counts[family] = len(groups)
    prov = {'edge_group_counts': counts,
            'families': list(LEAKAGE_FAMILIES), 'grain': 'labelset'}
    if extra_provenance:
        prov.update(extra_provenance)
    prov_path = os.path.join(data_dir, os.path.basename(OUT_EDGE_PROVENANCE))
    with open(prov_path, 'w') as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)
    return prov


def read_labelset_edge_tables(data_dir=DATA_DIR):
    """The published label-set edge families and their record.

    Returns ``({family: [[label_set_id, ...], ...]}, record)`` for every family in
    ``LEAKAGE_FAMILIES``, so a reader outside this stage groups label sets by the same
    edges the fold assignment was atomic over. ``record`` is
    ``labelset_edges_provenance.json``: the edge-group counts and gate settings.
    """
    import csv as _csv

    prov_path = os.path.join(data_dir, os.path.basename(OUT_EDGE_PROVENANCE))
    if not os.path.isfile(prov_path):
        raise FileNotFoundError(
            f'{prov_path} is absent, so the label-set edge tables are unbound')
    with open(prov_path) as fh:
        prov = json.loads(fh.read())
    families = {}
    for family in LEAKAGE_FAMILIES:
        path = labelset_edge_table_path(family, data_dir)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{path} is absent')
        groups = defaultdict(list)
        with open(path, newline='') as fh:
            for row in _csv.DictReader(fh):
                groups[row['edge_group']].append(row['label_set_id'])
        families[family] = [sorted(set(members)) for members in groups.values()
                            if len(set(members)) > 1]
    return families, prov


def verify_published_entity_folds(data_dir, ls_to_fold, bmrb_map, quarantine,
                                  balance, edge_families, ls_rec,
                                  extra_provenance=None, fold0_relatives=None):
    """Check the published fold maps, quarantine, balance report, every edge
    table, and the edge-group record against a fresh recomputation, without mutating
    any artifact — --verify_only alone does not read published output."""
    problems = []

    def _load_json(basename):
        path = os.path.join(data_dir, basename)
        if not os.path.isfile(path):
            problems.append(f"missing published artifact: {path}")
            return None
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"unreadable published artifact {path}: {exc}")
            return None

    def _json_shape(value):
        # Match JSON's stringification of integer dict keys before semantic compare.
        return json.loads(json.dumps(value, sort_keys=True))

    fold_name = os.path.basename(OUT_LABELSET_FOLD_MAP)
    bmrb_name = os.path.basename(OUT_LABELSET_BMRB_MAP)
    balance_name = os.path.basename(OUT_BALANCE_REPORT)
    published_fold_map = _load_json(fold_name)
    published_bmrb_map = _load_json(bmrb_name)
    published_balance = _load_json(balance_name)
    if published_fold_map is not None and published_fold_map != _json_shape(ls_to_fold):
        problems.append(f"{fold_name} differs from fresh fold assignment")
    if published_bmrb_map is not None and published_bmrb_map != _json_shape(bmrb_map):
        problems.append(f"{bmrb_name} differs from fresh derived BMRB map")
    if published_balance is not None and published_balance != _json_shape(balance):
        problems.append(f"{balance_name} differs from fresh balance report")
    if fold0_relatives is not None:
        relatives_name = os.path.basename(OUT_FOLD0_RELATIVES)
        published_relatives = _load_json(relatives_name)
        if published_relatives is not None:
            counts, relatives = fold0_relatives
            expected = fold0_relatives_record(counts, relatives)
            if published_relatives != _json_shape(expected):
                problems.append(f"{relatives_name} differs from fresh fold-0 relatives")

    quarantine_name = os.path.basename(OUT_QUARANTINE)
    quarantine_path = os.path.join(data_dir, quarantine_name)
    expected_quarantine = sorted(
        (
            str(rec["logical_entity_id"]),
            str(rec["label_set_id"]),
            str(rec["bmrb_id"]),
            str(rec["entity_id"]),
            str(rec["modeled_length"]),
            str(rec["entity_length"]),
        )
        for rec in quarantine
    )
    if not os.path.isfile(quarantine_path):
        problems.append(f"missing published artifact: {quarantine_path}")
    else:
        import csv as _csv
        try:
            with open(quarantine_path, newline="") as fh:
                reader = _csv.DictReader(fh)
                required = (
                    "logical_entity_id", "label_set_id", "bmrb_id", "entity_id",
                    "modeled_length", "entity_length",
                )
                if tuple(reader.fieldnames or ()) != required:
                    problems.append(
                        f"{quarantine_name} has columns {reader.fieldnames}, "
                        f"expected {list(required)}"
                    )
                    published_quarantine = []
                else:
                    published_quarantine = sorted(
                        tuple(str(row[name]) for name in required) for row in reader
                    )
                if published_quarantine != expected_quarantine:
                    problems.append(
                        f"{quarantine_name} differs from fresh quarantine"
                    )
        except OSError as exc:
            problems.append(f"unreadable published artifact {quarantine_path}: {exc}")

    expected_counts = {}
    for family in LEAKAGE_FAMILIES:
        path = labelset_edge_table_path(family, data_dir)
        name = os.path.basename(path)
        expected_counts[family] = len(edge_families.get(family, []))
        expected_groups = {
            tuple(sorted(str(label_set) for label_set in group))
            for group in edge_families.get(family, [])
        }
        if not os.path.isfile(path):
            problems.append(f"missing published artifact: {path}")
            continue
        try:
            import csv as _csv
            grouped = defaultdict(list)
            with open(path, newline="") as fh:
                reader = _csv.DictReader(fh)
                required = ("edge_group", "label_set_id", "bmrb_id", "fold")
                if tuple(reader.fieldnames or ()) != required:
                    problems.append(
                        f"{name} has columns {reader.fieldnames}, expected {list(required)}"
                    )
                    continue
                for row in reader:
                    label_set = row["label_set_id"]
                    if label_set not in ls_rec:
                        problems.append(f"{name} contains unknown label set {label_set}")
                        continue
                    expected_bmrb = str(ls_rec[label_set].get("bmrb_id", ""))
                    expected_fold = str(ls_to_fold.get(label_set, ""))
                    if row["bmrb_id"] != expected_bmrb or row["fold"] != expected_fold:
                        problems.append(
                            f"{name} row for {label_set} has bmrb/fold "
                            f"{row['bmrb_id']}/{row['fold']}, expected "
                            f"{expected_bmrb}/{expected_fold}"
                        )
                    grouped[row["edge_group"]].append(label_set)
            published_groups = {
                tuple(sorted(members)) for members in grouped.values()
            }
            if (len(grouped) != len(expected_groups)
                    or published_groups != expected_groups):
                problems.append(f"{name} differs from fresh {family} edge groups")
        except OSError as exc:
            problems.append(f"unreadable published artifact {path}: {exc}")

    provenance_name = os.path.basename(OUT_EDGE_PROVENANCE)
    provenance = _load_json(provenance_name)
    if provenance is not None:
        expected_provenance = {
            "edge_group_counts": expected_counts,
            "families": list(LEAKAGE_FAMILIES),
            "grain": "labelset",
        }
        if extra_provenance:
            expected_provenance.update(extra_provenance)
        if provenance != _json_shape(expected_provenance):
            problems.append(
                f"{provenance_name} differs from the fresh edge-group record"
            )

    if problems:
        raise RuntimeError(
            "published fold verification failed:\n  - " + "\n  - ".join(problems)
        )
    return True




def modeled_region_pairs(seqs, *, mmseqs_binary=None):
    """Homolog pairs between modeled regions at the training-fold axis, unioned over
    both coverage modes so a region contained in a longer one is an edge."""
    from . import cluster as _cluster
    from . import sequence_clustering as _search
    pairs = set()
    for cov_mode in _cluster.MMSEQS_COVERAGE_MODES:
        pairs |= _search.run_all_vs_all(
            seqs, identity=_cluster.MMSEQS_IDENTITY, coverage=_cluster.MMSEQS_COVERAGE,
            cov_mode=cov_mode, binary=mmseqs_binary)
    return pairs


def source_specific_modeled_groups(universe, min_len=None, *,
                                   pairwise_fn=None, mmseqs_binary=None):
    """Modeled-region edge groups (logical-entity grain) from per-(entity, source)
    modeled sub-sequences >= ``min_len``, each its own node so one source's domain
    is not diluted by another's positions. Multi-entity connected components only.
    ``pairwise_fn`` replaces the MMseqs search."""
    from . import cluster as _cluster
    min_len = _cluster.MIN_ENTITY_LEN if min_len is None else min_len
    seqs, node_to_le = {}, {}
    for le, r in universe.items():
        for src, seq in (r.get('modeled_by_source') or {}).items():
            if len(seq) >= min_len:
                node = f'{le}||{src}'
                seqs[node] = seq
                node_to_le[node] = le
    if not seqs:
        return []
    from .sequence_clustering import pairwise_edges_to_components
    pairs = (modeled_region_pairs(seqs, mmseqs_binary=mmseqs_binary)
             if pairwise_fn is None else pairwise_fn(seqs))
    components = pairwise_edges_to_components(seqs, pairs)
    groups = []
    for members in components:
        les = sorted({node_to_le[m] for m in members if m in node_to_le})
        if len(les) > 1:
            groups.append(les)
    return groups


def _le_groups_to_ls(groups, le_to_ls):
    """Map logical-entity-id groups onto label-set-id groups (drop singletons)."""
    out = []
    for members in groups:
        ls = sorted({le_to_ls[m] for m in members if m in le_to_ls})
        if len(ls) > 1:
            out.append(ls)
    return out


def build_labelset_edges(universe, cluster_edges, ls_uniprots,
                         modeled_groups=None):
    """All hard leakage edges as label-set-id groups, keyed by family.
    ``universe``={logical_entity_id: record} (fold universe only); ``cluster_edges``
    holds exact_token/alignment/mmseqs_cov0/mmseqs_containment/partner_chain at
    entity grain; ``ls_uniprots``={label_set_id: uniprot accessions};
    ``modeled_groups``=entity-id groups sharing a >=90% modeled region (default none)."""
    le_to_ls = {le: rec['label_set_id'] for le, rec in universe.items()}
    ls_set = set(le_to_ls.values())

    by_bmrb = defaultdict(list)
    for rec in universe.values():
        by_bmrb[str(rec['bmrb_id'])].append(rec['label_set_id'])
    same_bmrb = [sorted(set(v)) for v in by_bmrb.values() if len(set(v)) > 1]

    by_up = defaultdict(set)
    for ls, ups in ls_uniprots.items():
        if ls not in ls_set:
            continue
        for up in ups:
            if up:
                by_up[up].add(ls)
    uniprot = [sorted(v) for v in by_up.values() if len(v) > 1]

    def _fam(name):
        g = cluster_edges.get(name, {})
        members = list(g.values()) if isinstance(g, dict) else list(g)
        return _le_groups_to_ls(members, le_to_ls)

    return {
        'same_bmrb': same_bmrb,
        'uniprot': uniprot,
        'exact_token': _fam('exact_token'),
        'alignment': _fam('alignment'),
        'mmseqs_cov0': _fam('mmseqs_cov0'),
        'mmseqs_containment': _fam('mmseqs_containment'),
        'partner_chain': _fam('partner_chain'),
        'modeled_region': _le_groups_to_ls(modeled_groups or [], le_to_ls),
    }


def labelset_components(universe, edge_families):
    """Union-find over label sets across every edge family. Returns
    (ls_to_canonical, canonical_to_members); canonical = min label-set id of each
    component. Every label set is a node, even with no edges (singleton)."""
    nodes = [rec['label_set_id'] for rec in universe.values()]
    uf = UnionFind(nodes)
    for family in edge_families.values():
        for group in family:
            anchor = group[0]
            for other in group[1:]:
                uf.union(anchor, other)
    comp = defaultdict(list)
    for ls in nodes:
        comp[uf.find(ls)].append(ls)
    ls_to_canon, canon_to_members = {}, {}
    for members in comp.values():
        canon = min(members)
        canon_to_members[canon] = sorted(members)
        for ls in members:
            ls_to_canon[ls] = canon
    return ls_to_canon, canon_to_members


def component_weights(members, ls_rec):
    """Balancing weights for one component: bmrb/label-set/residue and per-source
    counts. ``primary`` (shift-bearing residue mass) drives the bin-pack."""
    bmrbs, residues, labelsets = set(), 0, len(members)
    per_source = defaultdict(int)
    for ls in members:
        rec = ls_rec[ls]
        bmrbs.add(str(rec['bmrb_id']))
        residues += int(rec.get('n_shift_residues') or rec.get('modeled_length') or 0)
        for s in rec.get('sources', []):
            per_source[s] += 1
    return {'bmrb': len(bmrbs), 'labelset': labelsets,
            'residue': residues, 'per_source': dict(per_source),
            'primary': max(1, residues)}


# Balance dimensions the joint bin-pack equalizes across CV folds: residue mass
# plus each structure source's label-set count.
_BALANCE_SOURCES = ('af', 'nmr', 'xtal')


def _component_balance_vector(weight):
    """Per-dimension pack contribution for one component under the joint balance:
    residue mass (``primary``) plus each source's label-set count. Keys mirror
    ``check_labelset_balance`` so the pack objective matches the reported metric."""
    per_source = weight.get('per_source', {})
    vec = {'residue': weight['primary']}
    for s in _BALANCE_SOURCES:
        vec[f'source_{s}'] = int(per_source.get(s, 0))
    return vec


def assign_labelset_folds(canon_to_members, ls_rec, excluded_canons,
                          seed=SPLIT_SEED, n_folds=N_FOLDS,
                          holdout_frac=HOLDOUT_FRAC):
    """Seeded, deterministic greedy bin-pack of whole components.
    UCB-excluded components go to fold 0 before the holdout draw. A seeded
    shuffle of the sorted manifest orders eligibility by identity alone (never
    outcome); the prefix closest to ``holdout_frac`` of label sets becomes the
    fold-6 holdout, whole components so no twin splits. The remainder
    packs heaviest-first by residue mass (ties: shuffle, then canonical id),
    dealt to the fold minimizing worst relative load across a joint residue +
    per-source (af/nmr/xtal) objective. Returns ls_to_fold with every
    label set assigned once (0/1..n/6; no -1)."""
    ls_to_fold = {}
    cv = []
    for canon, members in canon_to_members.items():
        if canon in excluded_canons:
            for ls in members:
                ls_to_fold[ls] = EXCLUDED_FOLD
        else:
            cv.append(canon)

    rng = random.Random(seed)
    # order is a seeded permutation of component identity alone, not of outcome;
    # only the prefix length targets mass, since components are atomic and
    # heavy-tailed and one of them can carry several percent of the label sets on
    # its own. Taking a holdout_frac share of components scatters the label-set
    # share across the band's edges; taking the prefix of components whose
    # label-set count is closest to target holds the band.
    manifest = sorted(cv)
    rng.shuffle(manifest)
    target = holdout_frac * sum(len(canon_to_members[c]) for c in manifest)
    n_holdout, running, closest = 0, 0, target
    for taken, canon in enumerate(manifest, start=1):
        running += len(canon_to_members[canon])
        if abs(running - target) < closest:
            closest, n_holdout = abs(running - target), taken
        if running >= target:
            break          # accumulation is monotone, so the distance only grows now
    holdout = set(manifest[:n_holdout])
    for c in holdout:
        for ls in canon_to_members[c]:
            ls_to_fold[ls] = HOLDOUT_FOLD

    # Joint bin-pack of the remaining CV components (heaviest-first).
    cv_pack = [c for c in cv if c not in holdout]
    tiebreak = {c: rng.random() for c in sorted(cv_pack)}
    weights = {c: component_weights(canon_to_members[c], ls_rec) for c in cv_pack}
    vectors = {c: _component_balance_vector(weights[c]) for c in cv_pack}
    keys = ('residue', *tuple(f'source_{s}' for s in _BALANCE_SOURCES))
    # Scale each dimension by its whole-manifest total so magnitudes are comparable
    # (residue mass vs. small source counts); empty dimensions never divide by zero.
    totals = {k: sum(vectors[c][k] for c in cv_pack) or 1 for k in keys}
    order = sorted(cv_pack, key=lambda c: (-weights[c]['primary'], tiebreak[c], c))
    load = {f: dict.fromkeys(keys, 0) for f in range(1, n_folds + 1)}
    for c in order:
        vec = vectors[c]
        # Deal to the fold whose heaviest post-add relative dimension is smallest.
        fold = min(range(1, n_folds + 1),
                   key=lambda f: (max((load[f][k] + vec[k]) / totals[k]
                                      for k in keys), f))
        for ls in canon_to_members[c]:
            ls_to_fold[ls] = fold
        for k in keys:
            load[fold][k] += vec[k]
    return ls_to_fold


def check_labelset_balance(ls_to_fold, ls_rec, tol=FOLD_BALANCE_TOL,
                           n_folds=N_FOLDS):
    """Per-count 15% balance gate across CV folds (not bare BMRB only)."""
    folds = list(range(1, n_folds + 1))
    counts = {k: dict.fromkeys(folds, 0)
              for k in ('labelset', 'residue')}
    bmrbs = {f: set() for f in folds}
    per_source = defaultdict(lambda: dict.fromkeys(folds, 0))
    for ls, f in ls_to_fold.items():
        if not (1 <= f <= n_folds):
            continue
        rec = ls_rec[ls]
        counts['labelset'][f] += 1
        counts['residue'][f] += int(rec.get('n_shift_residues')
                                    or rec.get('modeled_length') or 0)
        bmrbs[f].add(str(rec['bmrb_id']))
        for s in rec.get('sources', []):
            per_source[s][f] += 1

    def _dev(per_fold):
        vals = [per_fold[f] for f in folds]
        mean = sum(vals) / len(vals) if vals else 0.0
        return (max(abs(v - mean) for v in vals) / mean) if mean else 0.0

    report = {k: {'per_fold': counts[k], 'max_dev_frac': _dev(counts[k])}
              for k in counts}
    report['bmrb'] = {'per_fold': {f: len(bmrbs[f]) for f in folds},
                      'max_dev_frac': _dev({f: len(bmrbs[f]) for f in folds})}
    for s, pf in per_source.items():
        report[f'source_{s}'] = {'per_fold': pf, 'max_dev_frac': _dev(pf)}
    worst = max((r['max_dev_frac'] for r in report.values()), default=0.0)
    return {'per_count': report, 'max_dev_frac': worst, 'tol': tol,
            'passed': worst <= tol}


def check_holdout_gate(ls_to_fold, ls_rec, tol=FOLD_BALANCE_TOL,
                       low=HOLDOUT_MIN_FRAC, high=HOLDOUT_MAX_FRAC,
                       holdout_fold=HOLDOUT_FOLD, n_folds=N_FOLDS):
    """Fold-6 holdout gate: is the deterministic drawn holdout in-band and
    balanced? (Not resampled to hit a target — proves the seed's draw is
    acceptable, fails closed otherwise.) ``passed`` iff fold-6 share of eligible
    label sets/BMRBs/residue mass each fall in ``[low, high]`` (9-11%), and every
    af/nmr/xtal source is held out within ``tol`` (relative) of the overall rate.
    Pure, like ``check_labelset_balance``; fold 0 is not eligible."""
    def _eligible(f):
        return (1 <= f <= n_folds) or f == holdout_fold

    tot_ls = tot_res = hold_ls = hold_res = 0
    tot_bmrb, hold_bmrb = set(), set()
    src_tot = defaultdict(int)
    src_hold = defaultdict(int)
    for ls, f in ls_to_fold.items():
        if not _eligible(f):
            continue
        rec = ls_rec[ls]
        res = int(rec.get('n_shift_residues') or rec.get('modeled_length') or 0)
        bmrb = str(rec['bmrb_id'])
        tot_ls += 1
        tot_res += res
        tot_bmrb.add(bmrb)
        sources = rec.get('sources', [])
        for s in sources:
            src_tot[s] += 1
        if f == holdout_fold:
            hold_ls += 1
            hold_res += res
            hold_bmrb.add(bmrb)
            for s in sources:
                src_hold[s] += 1

    fractions = {
        'labelset': (hold_ls / tot_ls) if tot_ls else 0.0,
        'residue': (hold_res / tot_res) if tot_res else 0.0,
        'bmrb': (len(hold_bmrb) / len(tot_bmrb)) if tot_bmrb else 0.0,
    }
    overall = fractions['labelset']
    source_frac, source_dev = {}, {}
    for s, n in src_tot.items():
        frac = (src_hold.get(s, 0) / n) if n else 0.0
        source_frac[s] = frac
        if overall:
            source_dev[s] = abs(frac - overall) / overall
        else:
            source_dev[s] = 0.0 if frac == 0.0 else float('inf')

    in_band = {k: (low <= v <= high) for k, v in fractions.items()}
    source_ok = all(d <= tol for d in source_dev.values())
    passed = all(in_band.values()) and source_ok

    reasons = [f'{k} holdout frac {fractions[k]:.4f} outside [{low:.2f},{high:.2f}]'
               for k, ok in in_band.items() if not ok]
    reasons += [f'source {s} holdout frac {source_frac[s]:.4f} deviates '
                f'{source_dev[s]:.4f} from overall {overall:.4f} (> tol {tol})'
                for s in sorted(source_dev) if source_dev[s] > tol]
    return {'fractions': fractions, 'band': [low, high], 'source_frac': source_frac,
            'source_dev': source_dev, 'tol': tol, 'in_band': in_band,
            'source_ok': source_ok, 'passed': passed,
            'reason': '; '.join(reasons) or 'ok'}


def holdout_gate_remedy(gate, seed=SPLIT_SEED, holdout_frac=HOLDOUT_FRAC):
    """Operator-facing message for a failed holdout gate: which dimension missed,
    and the remedies named in the returned message."""
    fractions = gate['fractions']
    return (
        f'fold-6 holdout gate FAILED: {gate["reason"]}. The draw is DETERMINISTIC, '
        f'so rerunning with SHIFT_SPLIT_SEED={seed} reproduces this exact miss — it is '
        f'not a flake. The prefix already targets {holdout_frac:.0%} of the label sets, '
        f'so a miss means either one atomic leakage component is too heavy for the band '
        f'(drawn fold 6 holds {fractions["labelset"]:.4f} of label sets, '
        f'{fractions["residue"]:.4f} of residue mass, {fractions["bmrb"]:.4f} of BMRBs) '
        f'or a source is held out unrepresentatively (deviations {gate["source_dev"]}, '
        f'tol {gate["tol"]}). Choose ONE and record it with the build: a '
        f'different SHIFT_SPLIT_SEED, a widened '
        f'SHIFT_HOLDOUT_MIN_FRAC/SHIFT_HOLDOUT_MAX_FRAC, or a wider --holdout_source_tol. '
        f'Do NOT search seeds until a gate passes — that selects the split on the '
        f'outcome the gate audits.')


def fold0_training_relatives(ls_to_fold, edge_families, training_folds=TRAINING_FOLDS):
    """Per family, the number of edge groups spanning fold 0 and a training fold, and
    the sorted fold-0 label sets in such groups."""
    training = set(training_folds)
    counts, relatives = {}, set()
    for family, groups in edge_families.items():
        spanning = 0
        for group in groups:
            folds = {ls_to_fold[ls] for ls in group if ls in ls_to_fold}
            if EXCLUDED_FOLD in folds and folds & training:
                spanning += 1
                relatives.update(ls for ls in group if ls_to_fold.get(ls) == EXCLUDED_FOLD)
        counts[family] = spanning
    return counts, sorted(relatives)


def calibration_exclusion_record(relatives, case_seed, ls_to_fold) -> dict:
    """The fold-0 label sets the interval fit must not calibrate on: those with a
    training-fold relative, plus the case-study records that landed in fold 0."""
    case = sorted(ls for ls in case_seed if ls_to_fold.get(ls) == EXCLUDED_FOLD)
    return {'what': 'fold-0 label sets excluded from the interval calibration',
            'reasons': {'training_fold_relative': len(relatives),
                        'case_study_holdout': len(case),
                        'both': len(set(relatives) & set(case))},
            'training_fold_relatives': sorted(relatives),
            'case_study': case,
            'label_sets': sorted(set(relatives) | set(case))}


def fold0_relatives_record(counts, relatives, training_folds=TRAINING_FOLDS):
    return {'training_folds': list(training_folds),
            'edge_groups_spanning_fold0_and_training_by_family': counts,
            'label_sets': list(relatives)}


def verify_no_labelset_leakage(ls_to_fold, edge_families):
    """Assert no edge family spans two distinct folds among 1..N.
    Fold 0 (excluded) is not a leak by itself — components are atomic, so a
    family group is always in one fold. Returns {family: n_spanning_groups},
    all zero when correct."""
    out = {}
    for family, groups in edge_families.items():
        spanning = 0
        for group in groups:
            folds = {ls_to_fold[ls] for ls in group if ls in ls_to_fold}
            train_folds = {f for f in folds if f != EXCLUDED_FOLD}
            if len(train_folds) > 1 or (len(folds) > 1 and train_folds):
                spanning += 1
        out[family] = spanning
    return out


def derived_bmrb_fold_map(ls_to_fold, ls_rec):
    """Project the label-set fold map to a BMRB->fold map and check agreement.
    Every label set of a BMRB shares a component (same-BMRB is a hard edge), so
    each BMRB must resolve to one fold. Returns (bmrb_to_fold, agree, conflicts)."""
    by_bmrb = defaultdict(set)
    for ls, f in ls_to_fold.items():
        by_bmrb[str(ls_rec[ls]['bmrb_id'])].add(f)
    bmrb_to_fold, conflicts = {}, {}
    for bmrb, folds in by_bmrb.items():
        bmrb_to_fold[bmrb] = min(folds)
        if len(folds) > 1:
            conflicts[bmrb] = sorted(folds)
    return bmrb_to_fold, (not conflicts), conflicts


def _load_bmrb_uniprot_map(data_dir):
    """Protected BMRB->UniProt mapping (merged curated + augmented) under
    ``<data_dir>/alphafold``: source of shared-UniProt edges for entities without
    a local AF model — annotation is a property of the deposition, not of whether AF
    coordinates exist. The mapping is required: it feeds the shared-UniProt hard edge
    and the same-UniProt arm of the UCBShift-200 rule, and an absent mapping would
    disable both while reports still printed ``"uniprot": 0``, which reads as
    'checked, clean'."""
    from .uniprot import load_merged_mapping
    mapping = load_merged_mapping([os.path.join(data_dir, 'alphafold')],
                                  required=True)
    if not mapping:
        raise RuntimeError(
            f'BMRB->UniProt mapping under {data_dir}/alphafold is EMPTY; the '
            f'shared-UniProt leakage edge and the same-UniProt arm of the '
            f'UCBShift-200 rule would both be silently inactive.')
    return mapping


def _load_ls_uniprots(ledger_path, ls_rec=None, uniprot_map=None):
    """{label_set_id: set(shared-UniProt accession)} spanning all realized
    sources. A shared accession is an unconditional hard edge: AF
    structure-fit identity/coverage decides usability, not ownership, so it
    never gates this. Over-connecting a sibling is the conservative choice vs.
    missing a true shared-accession leak."""
    out = defaultdict(set)
    if ledger_path and os.path.isfile(ledger_path):
        import csv as _csv
        with open(ledger_path, newline='') as fh:
            for row in _csv.DictReader(fh):
                if row.get('source') != 'af':
                    continue
                acc = row.get('accession', '')
                if not acc:
                    continue
                ls = row.get('label_set_id')
                if not ls:
                    continue
                out[ls].add(acc)
    # Protected deposition BMRB->UniProt mapping projected onto all realized label sets, including sources without a local AF coordinate model.
    if ls_rec and uniprot_map:
        for ls, rec in ls_rec.items():
            ups = uniprot_map.get(str(rec.get('bmrb_id')))
            if not ups:
                continue
            for up in (ups if isinstance(ups, (list, tuple, set)) else [ups]):
                up = str(up)
                if up:
                    out[ls].add(up)
    return {k: v for k, v in out.items() if v}


def load_ucb_test_target_sequences(path):
    """Load the 200 UCBShift test targets from their JSON file (test id -> sequence)."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"UCBShift-200 test target authority missing: {path}")
    if not str(path).lower().endswith(".json"):
        raise ValueError(
            f"{path}: the UCBShift-200 test targets must be a JSON file mapping test id "
            "to sequence; convert CSpred's test_200.pkl once and point "
            "SHIFT_UCB_TEST_TARGETS at the JSON"
        )
    with open(path) as fh:
        obj = json.loads(fh.read())
    if not isinstance(obj, dict):
        raise ValueError(f'{path}: expected a dict of test_id -> sequence')
    sequences = {str(k): str(v).upper() for k, v in obj.items() if v}
    if len(sequences) != 200:
        raise ValueError(
            f"{path}: expected exactly 200 UCBShift targets, got {len(sequences)}"
        )
    invalid = {
        key: sequence
        for key, sequence in sequences.items()
        if not sequence or any(code not in "ACDEFGHIKLMNPQRSTVWYX" for code in sequence)
    }
    if invalid:
        raise ValueError(
            f"{path}: invalid one-letter protein sequences for "
            f"{sorted(invalid)[:10]}"
        )
    return sequences


def ucb_homology_matched_labelsets(ls_rec, test_sequences, *, pairwise_fn=None,
                                   mmseqs_binary=None,
                                   identity=UCB_HOMOLOGY_IDENTITY,
                                   coverage=UCB_HOMOLOGY_COVERAGE):
    """Universe label sets homologous to a benchmark test protein: identity >=
    ``identity`` and coverage of the shorter side >= ``coverage`` (defaults: the fold-0
    arm's UCB_HOMOLOGY_IDENTITY/COVERAGE). ``pairwise_fn`` replaces the MMseqs search;
    production searches universe plus test sequences in UCB_HOMOLOGY_COVERAGE_MODES and
    keeps the cross pairs. Returns matched ``label_set_id``s."""
    if not ls_rec or not test_sequences:
        return set()
    ls_seq = {ls: (rec.get('entity_sequence') or rec.get('modeled_sequence') or '')
              for ls, rec in ls_rec.items()}
    ls_seq = {ls: s for ls, s in ls_seq.items() if s}
    if not ls_seq:
        return set()
    seqs = dict(ls_seq)
    # Namespace the test ids so they can never collide with a label-set id.
    test_nodes = {}
    for tid, seq in test_sequences.items():
        if not seq:
            continue
        node = f'__ucbtest__{tid}'
        seqs[node] = seq
        test_nodes[node] = tid
    if not test_nodes:
        return set()
    if pairwise_fn is not None:
        pairs = pairwise_fn(seqs)
    else:
        from . import sequence_clustering as _search
        pairs = set()
        for cov_mode in UCB_HOMOLOGY_COVERAGE_MODES:
            pairs |= _search.run_all_vs_all(
                seqs, identity=identity, coverage=coverage, cov_mode=cov_mode,
                binary=mmseqs_binary)
    matched = set()
    for a, b in pairs:
        a_test, b_test = a in test_nodes, b in test_nodes
        if a_test == b_test:
            continue               # both test or both universe -> not a test match
        ls = b if a_test else a
        if ls in ls_seq:
            matched.add(ls)
    return matched


def direct_ucb_excluded_labelsets(ls_rec, test_bmrbs, test_uniprots,
                                  ls_uniprots, homology_matched):
    """The direct, per-label-set UCBShift-200 exclusion seed: a label set is in the
    seed iff its BMRB is a test protein's (``test_bmrbs``), it shares a validated
    UniProt with one (``test_uniprots``), or it is a homolog (``homology_matched``:
    identity >= UCB_HOMOLOGY_IDENTITY, maxcov >= UCB_HOMOLOGY_COVERAGE). Same-BMRB
    closure then adds every label set sharing a BMRB with a seed member. A component
    with no seed member returns whole to CV rather than being force-excluded."""
    test_bmrbs = {str(b) for b in (test_bmrbs or ())}
    test_uniprots = {str(u) for u in (test_uniprots or ()) if u}
    excluded = set()
    for ls, rec in ls_rec.items():
        if str(rec.get('bmrb_id')) in test_bmrbs or (test_uniprots and (
                {str(u) for u in ls_uniprots.get(ls, ())} & test_uniprots)):
            excluded.add(ls)
    excluded |= {ls for ls in (homology_matched or ()) if ls in ls_rec}
    # same-BMRB ownership closure
    by_bmrb = defaultdict(set)
    for ls, rec in ls_rec.items():
        by_bmrb[str(rec.get('bmrb_id'))].add(ls)
    closed = set(excluded)
    for ls in excluded:
        closed |= by_bmrb.get(str(ls_rec[ls].get('bmrb_id')), set())
    return closed


def assign_folds_direct_exclusion(universe, cluster_edges, ls_uniprots,
                                  modeled_groups, excluded_seed, *,
                                  seed=SPLIT_SEED, n_folds=N_FOLDS,
                                  holdout_frac=HOLDOUT_FRAC):
    """Fold assignment without a component-transitive projection of the seed.
    ``excluded_seed`` (already same-BMRB closed) is the only thing forced to
    fold 0; every other member of its leakage components returns to CV — graph,
    edge families, and bin-pack are recomputed over the universe minus the seed,
    so a non-matching sibling folds normally instead of being force-excluded.
    Components
    stay atomic in the training pool; the leakage gate only sees
    training-eligible edges. Returns (ls_to_fold, cv_edge_families,
    cv_ls_to_canon, cv_canon_to_members)."""
    excluded_seed = set(excluded_seed)
    cv_universe = {le: rec for le, rec in universe.items()
                   if rec['label_set_id'] not in excluded_seed}
    cv_edge_families = build_labelset_edges(
        cv_universe, cluster_edges, ls_uniprots, modeled_groups=modeled_groups)
    cv_ls_to_canon, cv_canon_to_members = labelset_components(
        cv_universe, cv_edge_families)
    cv_ls_rec = {rec['label_set_id']: rec for rec in cv_universe.values()}
    ls_to_fold = assign_labelset_folds(
        cv_canon_to_members, cv_ls_rec, set(),
        seed=seed, n_folds=n_folds, holdout_frac=holdout_frac)
    for ls in excluded_seed:
        ls_to_fold[ls] = EXCLUDED_FOLD
    return ls_to_fold, cv_edge_families, cv_ls_to_canon, cv_canon_to_members


def entity_main(argv=None) -> int:
    """Entity-grained fold assignment driver."""
    from . import cluster as _cluster

    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=DATA_DIR)
    ap.add_argument('--edge_dir', default=None,
                    help='dir with cluster-stage entity_edges_*.csv (default data_dir)')
    ap.add_argument('--ledger', default=None,
                    help='discovery candidate_ledger.csv for validated UniProt edges')
    ap.add_argument('--excluded_bmrbs', default=None,
                    help=f'shipped UCBShift-200 excluded-BMRB list (default '
                         f'<data_dir>/{UCB_EXCLUDED_BASENAME}).')
    ap.add_argument('--test_targets',
                    default=os.environ.get('SHIFT_UCB_TEST_TARGETS'),
                    help='UCBShift-200 test targets as JSON (<PDB4><CHAIN> -> sequence), '
                         'the sequences of the fold-0 homology arm (default '
                         f'$SHIFT_UCB_TEST_TARGETS or <data_dir>/'
                         f'{UCB_TEST_TARGETS_BASENAME}).')
    ap.add_argument('--case_study_bmrbs',
                    default=os.path.join(_ROOT, CASE_STUDY_EXCLUDED_BASENAME),
                    help='packaged case-study hold-out (BMRBs and sequences), pinned to '
                         'fold 0 after the draw (default: the packaged authority).')
    ap.add_argument('--allow_empty_ucb_exclusion', action='store_true',
                    help='Explicitly permit an empty or unauthenticated UCBShift-200 '
                         'exclusion set (unsafe: benchmark entries may enter folds).')
    ap.add_argument('--mmseqs_bin', default=os.environ.get('MMSEQS_BIN'))
    ap.add_argument('--verify_only', action='store_true')
    ap.add_argument('--balance_tol', type=float, default=FOLD_BALANCE_TOL,
                    help='max per-fold count deviation. Leakage-safe '
                         'component bin-packing can make 15%% mathematically '
                         'impossible on a tiny/hard universe; relax the '
                         'balance gate only; the leakage gates are not relaxed.')
    ap.add_argument('--holdout_source_tol', type=float, default=FOLD_BALANCE_TOL,
                    help='max relative deviation of each af/nmr/xtal share of fold 6 '
                         'from the overall holdout share; the smallest source sits in '
                         'the clumpiest components, so its draw varies most.')
    args = ap.parse_args(argv)
    edge_dir = args.edge_dir or args.data_dir
    ledger = args.ledger or os.path.join(args.data_dir, 'candidate_ledger.csv')
    excluded_bmrbs_path = args.excluded_bmrbs or os.path.join(
        args.data_dir, UCB_EXCLUDED_BASENAME)
    test_targets_path = args.test_targets or os.path.join(
        args.data_dir, UCB_TEST_TARGETS_BASENAME)

    masters = _cluster._find_masters(args.data_dir)
    if not masters:
        print(f'ERROR: no entity-grained masters under {args.data_dir}', file=sys.stderr)
        return 2
    # The modeled region comes from the materialization build logs, not the sparse master rows, so those logs are required.
    entities = _cluster.collect_realized_entities(masters, require_build_logs=True)
    print(f'realized logical entities: {len(entities):,}')

    # Fold universe = entities with full polymer length >= 10, measured on
    # declared length not shift-bearing row count (else a long low-label protein drops).
    universe = {le: r for le, r in entities.items()
                if r['entity_length'] >= _cluster.MIN_ENTITY_LEN}
    quarantine = [r for le, r in entities.items()
                  if r['entity_length'] < _cluster.MIN_ENTITY_LEN]
    ls_rec = {rec['label_set_id']: rec for rec in universe.values()}
    print(f'  fold universe: {len(universe):,}  quarantined(<10 polymer): '
          f'{len(quarantine):,}')

    # Cluster-stage entity edges (recompute if the tables are absent).
    cluster_edges = _read_edge_tables(edge_dir, masters)
    if cluster_edges is None:
        cluster_edges = _cluster.build_entity_edges(
            universe, mmseqs_binary=args.mmseqs_bin)

    # Modeled-region edges at the training-fold axis, both coverage modes.
    modeled_groups = source_specific_modeled_groups(
        universe, mmseqs_binary=args.mmseqs_bin)

    # Shared-UniProt edges from all sources, not just local-AF entities (see _load_ls_uniprots/_load_bmrb_uniprot_map).
    uniprot_map = _load_bmrb_uniprot_map(args.data_dir)
    ls_uniprots = _load_ls_uniprots(ledger, ls_rec=ls_rec, uniprot_map=uniprot_map)

    # Resolve the UCBShift-200 boundary, refusing an empty one: the packaged BMRBs are the
    # same-BMRB base, mapped UniProts the same-UniProt base, the test-target JSON the
    # homology-arm sequences.
    excluded_bmrbs = resolve_ucb_excluded_bmrbs(
        excluded_bmrbs_path,
        {str(rec['bmrb_id']) for rec in universe.values()},
        pairs_csv=os.path.join(args.data_dir, 'pairs.csv'),
        test_dir=UCB_TEST_DIR,
        allow_empty=args.allow_empty_ucb_exclusion,
    )
    test_uniprots = _uniprots_of(excluded_bmrbs, uniprot_map)
    test_sequences = load_ucb_test_target_sequences(test_targets_path)
    homology_matched = ucb_homology_matched_labelsets(
        ls_rec, test_sequences, mmseqs_binary=args.mmseqs_bin)

    # The direct seed goes to fold 0 only; everything else returns to CV (see direct_ucb_excluded_labelsets/assign_folds_direct_exclusion).
    excluded_seed = direct_ucb_excluded_labelsets(
        ls_rec, excluded_bmrbs, test_uniprots, ls_uniprots, homology_matched)
    excluded_seed = {ls for ls in excluded_seed if ls in ls_rec}
    print(f'  UCB DIRECT rule seed: {len(excluded_seed):,} label sets -> fold 0 '
          f'(same-BMRB base {len(excluded_bmrbs):,}, homology '
          f'{len(homology_matched):,} at identity>={UCB_HOMOLOGY_IDENTITY} '
          f'maxcov>={UCB_HOMOLOGY_COVERAGE}); NO component projection -- '
          f'non-matching homologs return to CV')

    # Fold assignment recomputed over the universe minus the excluded seed -- see assign_folds_direct_exclusion.
    ls_to_fold, edge_families, _ls_to_canon, _canon_to_members = (
        assign_folds_direct_exclusion(
            universe, cluster_edges, ls_uniprots, modeled_groups, excluded_seed))

    # Case-study proteins leave training by the same direct rule, pinned after the draw
    # (see pin_after_draw); the gate then sees the universe without either seed.
    case_bmrbs, case_sequences = load_case_study_exclusion(args.case_study_bmrbs)
    case_uniprots = _uniprots_of(case_bmrbs, uniprot_map)
    case_homologs = ucb_homology_matched_labelsets(
        ls_rec, case_sequences, mmseqs_binary=args.mmseqs_bin)
    case_segments = case_study_segment_matched_labelsets(
        ls_rec, case_sequences, mmseqs_binary=args.mmseqs_bin)
    case_seed = direct_ucb_excluded_labelsets(
        ls_rec, case_bmrbs, case_uniprots, ls_uniprots,
        set(case_homologs) | set(case_segments))
    case_direct = {ls for ls, rec in ls_rec.items()
                   if str(rec.get('bmrb_id')) in case_bmrbs
                   or ({str(u) for u in ls_uniprots.get(ls, ())} & case_uniprots)
                   } | set(case_homologs) | set(case_segments)
    case_routes = {}
    for ls in case_seed:
        rec = ls_rec[ls]
        if str(rec.get('bmrb_id')) in case_bmrbs:
            case_routes.setdefault(ls, set()).add('same_bmrb')
        if {str(u) for u in ls_uniprots.get(ls, ())} & case_uniprots:
            case_routes.setdefault(ls, set()).add('same_uniprot')
        if ls in case_homologs:
            case_routes.setdefault(ls, set()).add('homology')
        if ls in case_segments:
            case_routes.setdefault(ls, set()).add(
                f"segment:{case_segments[ls]['case']}:{case_segments[ls]['aligned']}aa"
                f"@{case_segments[ls]['identity']:.2f}")
        if not case_routes.get(ls):
            case_routes[ls] = {'same_bmrb_closure'}
    moved = pin_after_draw(ls_to_fold, case_seed)
    if case_seed:
        edge_families = build_labelset_edges(
            {le: rec for le, rec in universe.items()
             if rec['label_set_id'] not in excluded_seed | case_seed},
            cluster_edges, ls_uniprots, modeled_groups=modeled_groups)
    print(f'  case-study hold-out: {len(case_bmrbs)} BMRBs, {len(case_sequences)} sequences '
          f'-> seed {len(case_seed):,} label sets (homology {len(case_homologs):,}, '
          f'segment {len(case_segments):,}); {len(moved):,} pinned to fold 0 from training folds')

    leaks = verify_no_labelset_leakage(ls_to_fold, edge_families)
    # The training-fold gate above sees the universe without the fold-0 seed. The edge
    # families over the whole universe show which fold-0 label sets have a relative in a
    # training fold at the same axis; the interval fit excludes them.
    full_edge_families = build_labelset_edges(
        universe, cluster_edges, ls_uniprots, modeled_groups=modeled_groups)
    relative_counts, relatives = fold0_training_relatives(ls_to_fold, full_edge_families)
    boundary = case_study_boundary(case_seed, case_routes, case_direct, ls_to_fold,
                                   ls_rec, full_edge_families)
    print('  ' + case_study_boundary_summary(boundary))
    balance = check_labelset_balance(ls_to_fold, ls_rec, tol=args.balance_tol)
    balance['fold0_training_relatives'] = {
        'edge_groups_by_family': relative_counts, 'label_sets': len(relatives)}
    holdout_gate = check_holdout_gate(ls_to_fold, ls_rec, tol=args.holdout_source_tol)
    bmrb_map, agree, conflicts = derived_bmrb_fold_map(ls_to_fold, ls_rec)

    print('leakage per family: ' + json.dumps(leaks))
    print(f'fold-0 label sets with a training-fold relative: {len(relatives)} '
          + json.dumps(relative_counts))
    print(f'balance max_dev={balance["max_dev_frac"]:.4f} tol={balance["tol"]} '
          f'passed={balance["passed"]}')
    print(f'fold-6 holdout gate: fractions={holdout_gate["fractions"]} '
          f'band={holdout_gate["band"]} passed={holdout_gate["passed"]} '
          f'({holdout_gate["reason"]})')
    print(f'derived BMRB map agrees: {agree} (conflicts={len(conflicts)})')

    # Every realized (universe) label set gets exactly one fold, no -1 drops.
    missing = [ls for ls in ls_rec if ls not in ls_to_fold]
    if missing:
        print(f'FATAL: {len(missing)} label sets unassigned (-1 drop)', file=sys.stderr)
        return 1
    if any(v for v in leaks.values()):
        print('FATAL: leakage edge spans folds', file=sys.stderr)
        return 1
    if boundary['prohibited']:
        print(f"FATAL: {len(boundary['prohibited'])} label sets match a case-study protein "
              f"under the rule and are still in a training fold; see "
              f"{os.path.basename(OUT_CASE_STUDY_BOUNDARY)}", file=sys.stderr)
        return 1
    if not agree:
        print('FATAL: derived BMRB map disagrees with label-set map', file=sys.stderr)
        return 1
    if not balance['passed']:
        print('FATAL: fold balance exceeds tolerance', file=sys.stderr)
        return 1
    # The fold-6 draw must be 9-11% and source-balanced; a miss is refused, not redrawn -- see holdout_gate_remedy.
    if not holdout_gate['passed']:
        remedy = holdout_gate_remedy(holdout_gate)
        print('FATAL: ' + remedy, file=sys.stderr)
        raise RuntimeError(remedy)
    # Outputs land in the data dir this run read from; the OUT_* constants are defaults, re-resolved here against --data_dir.
    data_out = args.data_dir
    os.makedirs(data_out, exist_ok=True)
    out_fold_map = os.path.join(data_out, os.path.basename(OUT_LABELSET_FOLD_MAP))
    out_bmrb_map = os.path.join(data_out, os.path.basename(OUT_LABELSET_BMRB_MAP))
    out_quarantine = os.path.join(data_out, os.path.basename(OUT_QUARANTINE))
    out_balance = os.path.join(data_out, os.path.basename(OUT_BALANCE_REPORT))
    out_relatives = os.path.join(data_out, os.path.basename(OUT_FOLD0_RELATIVES))
    fold_provenance_fields = {
        'leaks_per_family': leaks,
        'ucb_homology_gate': {
            'identity': UCB_HOMOLOGY_IDENTITY,
            'coverage': UCB_HOMOLOGY_COVERAGE,
            'coverage_modes': list(UCB_HOMOLOGY_COVERAGE_MODES),
            'matched_label_sets': len(homology_matched)},
        'modeled_region_gate': {
            'identity': _cluster.MMSEQS_IDENTITY,
            'coverage': _cluster.MMSEQS_COVERAGE,
            'coverage_modes': list(_cluster.MMSEQS_COVERAGE_MODES)},
        'case_study_exclusion': {
            'authority': os.path.basename(args.case_study_bmrbs),
            'bmrbs': sorted(case_bmrbs, key=int),
            'homology_matched': len(case_homologs),
            'segment_matched': len(case_segments),
            'segment_rule': {'identity': CASE_STUDY_SEGMENT_IDENTITY,
                             'min_aligned': CASE_STUDY_SEGMENT_MIN_ALIGNED},
            'seed_label_sets': len(case_seed),
            'pinned_from_training_folds': len(moved)},
    }
    case_study_pin = {'seed': sorted(case_seed), 'moved': moved}
    if args.verify_only:
        verify_published_entity_folds(
            data_out, ls_to_fold, bmrb_map, quarantine, balance,
            edge_families, ls_rec, extra_provenance=fold_provenance_fields,
            fold0_relatives=(relative_counts, relatives),
        )
        print(f"verified published fold artifacts under {data_out}")
        return 0

    with open(out_fold_map, 'w') as fh:
        json.dump(ls_to_fold, fh, sort_keys=True)
    with open(out_bmrb_map, 'w') as fh:
        json.dump(bmrb_map, fh, sort_keys=True)
    with open(out_quarantine, 'w', newline='') as fh:
        import csv as _csv
        w = _csv.writer(fh)
        w.writerow(['logical_entity_id', 'label_set_id', 'bmrb_id', 'entity_id',
                    'modeled_length', 'entity_length'])
        for r in sorted(quarantine, key=lambda r: r['label_set_id']):
            w.writerow([r['logical_entity_id'], r['label_set_id'], r['bmrb_id'],
                        r['entity_id'], r['modeled_length'], r['entity_length']])
    # publish this stage's own edge tables, their record and the balance report (they are
    # computed nowhere else); the record also holds the fold-0 gate thresholds, which are
    # not recoverable from the map.
    write_labelset_edge_tables(edge_families, ls_rec, ls_to_fold,
                               data_dir=data_out,
                               extra_provenance=fold_provenance_fields)
    with open(out_balance, 'w') as fh:
        json.dump(balance, fh, indent=2, sort_keys=True)
    with open(out_relatives, 'w') as fh:
        json.dump(fold0_relatives_record(relative_counts, relatives), fh,
                  indent=2, sort_keys=True)
    with open(os.path.join(data_out, os.path.basename(OUT_CASE_STUDY_PIN)), 'w') as fh:
        json.dump(case_study_pin, fh, indent=2, sort_keys=True)
    with open(os.path.join(data_out, os.path.basename(OUT_CASE_STUDY_BOUNDARY)), 'w') as fh:
        json.dump(boundary, fh, indent=2, sort_keys=True)
    calibration = calibration_exclusion_record(relatives, case_seed, ls_to_fold)
    with open(os.path.join(data_out, os.path.basename(OUT_CALIBRATION_EXCLUSIONS)), 'w') as fh:
        json.dump(calibration, fh, indent=2, sort_keys=True)
    print(f"  interval calibration excludes {len(calibration['label_sets']):,} fold-0 label sets "
          f"({calibration['reasons']})")
    print(f'wrote {out_fold_map}, {out_bmrb_map}, {out_quarantine}, '
          f'{os.path.join(data_out, os.path.basename(OUT_EDGE_PROVENANCE))}, '
          f'{out_balance}, {out_relatives}')
    return 0


def _read_edge_tables(edge_dir, masters=()):
    """Load cluster-stage entity_edges_{family}.csv as logical-entity-id groups.

    Returns a dict in build_entity_edges shape, or None if the tables are absent."""
    from . import cluster as _cluster
    fams = ('exact_token', 'alignment', 'mmseqs_cov0', 'mmseqs_containment',
            'partner_chain')
    paths = {f: os.path.join(edge_dir, f'entity_edges_{f}.csv') for f in fams}
    if not all(os.path.isfile(p) for p in paths.values()):
        return None
    # Reuse the cluster edge tables only when they were written after every master they
    # are built from and entity_edges_provenance.json records the homolog axis in force;
    # otherwise recompute.
    written = min(os.path.getmtime(p) for p in paths.values())
    if any(os.path.getmtime(m) > written for m in masters):
        print('WARNING: a master is newer than the cluster edge tables; recomputing '
              'cluster edges', file=sys.stderr)
        return None
    prov_path = os.path.join(edge_dir, 'entity_edges_provenance.json')
    if not os.path.isfile(prov_path):
        print(f'WARNING: {prov_path} absent; recomputing cluster edges', file=sys.stderr)
        return None
    with open(prov_path) as fh:
        prov = json.loads(fh.read())
    axis = (prov.get('identity'), prov.get('coverage'))
    if axis != (_cluster.MMSEQS_IDENTITY, _cluster.MMSEQS_COVERAGE):
        print(f'WARNING: {prov_path} records homolog axis {axis}, not the axis in '
              f'force ({_cluster.MMSEQS_IDENTITY}/{_cluster.MMSEQS_COVERAGE}); '
              f'recomputing cluster edges', file=sys.stderr)
        return None
    import csv as _csv
    out = {}
    for f, p in paths.items():
        groups = defaultdict(list)
        with open(p, newline='') as fh:
            for row in _csv.DictReader(fh):
                groups[row['edge_group']].append(row['logical_entity_id'])
        members = [sorted(set(v)) for v in groups.values() if len(set(v)) > 1]
        out[f] = members
    return out


def main() -> int:
    return entity_main(sys.argv[1:])


if __name__ == '__main__':
    sys.exit(main())
