"""Corpus census over one build's realization ledger, fold map and edge tables.

One definition for every figure and table that counts the corpus: a realization is one
successful ``(label set, structure source)`` row of ``realization_ledger.csv``; the
protein count is the number of components the fold assignment is atomic over, which is
the connected components of the fold stage's published label-set edge tables.

Every reader binds the fold map: the realized label sets and the fold map's keys must be
the same set, so a census cannot be computed from one build's ledger and another's split.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from ..domain import VALID_SOURCES
from . import folds as _folds

LEDGER_BASENAME = 'realization_ledger.csv'
FOLD_MAP_BASENAME = 'fold_assignments_labelset_identity90.json'
SUCCESS = 'success'
# The columns that identify a realization; a caller asking for more gets them appended.
IDENTITY_COLUMNS = ('label_set_id', 'logical_entity_id', 'bmrb_id', 'source', 'status')


def read_realizations(data_dir, *, extra_columns=()) -> pd.DataFrame:
    """Successful realizations of ``data_dir``'s ledger, one row per (label set, source).

    A repeated pair is refused: a per-source count would then be a structure count while
    the label-set counts beside it stay label sets, and the two are not the same unit.
    """
    path = os.path.join(data_dir, LEDGER_BASENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} is absent, so no census can be taken')
    columns = list(dict.fromkeys(IDENTITY_COLUMNS + tuple(extra_columns)))
    frame = pd.read_csv(path, dtype=str, usecols=columns)
    frame = frame[frame['status'] == SUCCESS].drop(columns=['status'])
    if frame.empty:
        raise ValueError(f'{path} carries no row with status {SUCCESS!r}')
    unknown = sorted(set(frame['source']) - set(VALID_SOURCES))
    if unknown:
        raise ValueError(f'{path} realizes unknown structure sources: {unknown}')
    if frame.duplicated(subset=['label_set_id', 'source']).any():
        raise ValueError(
            f'{path} realizes a (label set, source) pair more than once, so a per-source '
            'count is not a count of label sets')
    return frame.reset_index(drop=True)


def read_fold_map(data_dir) -> dict[str, int]:
    """``label set -> fold``."""
    path = os.path.join(data_dir, FOLD_MAP_BASENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} is absent, so the census is unbound to a split')
    with open(path) as fh:
        payload = json.loads(fh.read())
    return {str(k): int(v) for k, v in payload.items()}


def bind_fold_map(fold_of, label_sets) -> None:
    """Raise unless the fold map's label sets are exactly the realized ones."""
    realized, assigned = set(label_sets), set(fold_of)
    if realized != assigned:
        raise ValueError(
            f'the fold map and the ledger describe different builds: '
            f'{len(realized - assigned)} realized label sets carry no fold and '
            f'{len(assigned - realized)} assigned label sets are not realized')


def source_census(frame: pd.DataFrame) -> dict:
    """Counts a source panel needs: per-source label sets, sole sources, multiplicity."""
    combo = frame.groupby('label_set_id')['source'].agg(frozenset)
    n_sources = combo.map(len)
    return {
        'label_sets_realized': int(frame['label_set_id'].nunique()),
        'depositions': int(frame['bmrb_id'].nunique()),
        'realizations': len(frame),
        'per_source_label_sets': {src: int((frame['source'] == src).sum())
                                  for src in VALID_SOURCES},
        'per_source_sole': {src: int((combo == frozenset({src})).sum())
                            for src in VALID_SOURCES},
        'multiplicity': {str(k): int((n_sources == k).sum())
                         for k in range(1, len(VALID_SOURCES) + 1)},
        'multi_source': int((n_sources >= 2).sum()),
        'pairwise': {f'{a}+{b}': int(combo.map(lambda c, a=a, b=b: {a, b} <= c).sum())
                     for a, b in (('af', 'nmr'), ('af', 'xtal'), ('nmr', 'xtal'))},
    }


def edge_table_label_sets(fold_of) -> set[str]:
    """The label sets the fold stage's edge tables cover: those of folds 1-6. Fold 0 is
    filled by rule before the seeded draw and has no rows in them, so a component count
    over it would count each of its label sets as a protein of its own."""
    return {ls for ls, fold in fold_of.items() if fold != _folds.EXCLUDED_FOLD}


def protein_components(data_dir, label_sets) -> tuple[dict[str, list[str]], dict]:
    """``(canonical label set -> members, edge-table record)`` over ``label_sets``.

    The grouping is the fold stage's own: every published leakage edge family, closed by
    union-find, so a component is the unit a fold assignment moves as one and the number
    of components is the number of distinct proteins the corpus contains.
    """
    families, provenance = _folds.read_labelset_edge_tables(data_dir)
    universe = {ls: {'label_set_id': ls} for ls in label_sets}
    unknown = {ls for groups in families.values() for group in groups for ls in group
               if ls not in universe}
    if unknown:
        raise ValueError(
            f'{len(unknown)} label sets in the edge tables are not realized in this '
            'ledger, so the edges and the ledger are from different builds')
    _, canon_to_members = _folds.labelset_components(universe, families)
    return canon_to_members, provenance
