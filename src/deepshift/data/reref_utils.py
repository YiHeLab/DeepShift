"""Rereferencing helpers: nucleus vocabulary, channel expansion, numbering alignment."""

from collections import Counter

from ..domain import AA_3_TO_1, SHIFT_COLUMNS

NUCS = ['HA', 'CA', 'CB', 'CO', 'N', 'HN']
CUTOFF = {'HA': 0.7, 'CA': 5.0, 'CB': 5.0, 'CO': 5.0, 'N': 10.0, 'HN': 2.0}
HDR2SX = {'HA': 'HA', 'CA': 'CA', 'CB': 'CB', 'CO': 'C', 'N': 'N', 'HN': 'H'}
SXCOLS = ['HA', 'H', 'N', 'CA', 'CB', 'C']
THREE2ONE = dict(AA_3_TO_1)

# An offset corrects a whole reference channel: a 13C zero-point error moves every carbon
# together, a 15N error every nitrogen. The joint CA/CB aliphatic offset also corrects the
# side-chain carbons, and the 15N offset the side-chain amide nitrogens. C' is calibrated
# in its own experiment and keeps its own offset (RefDB, Zhang et al. 2003, p.179 groups
# by experiment, not element); the proton columns are never corrected, because 1H
# referencing errors are insignificant (RefDB p.183).
def _by_element():
    """Group shift columns by NMR-active element (first char of the atom name, exact for
    this vocabulary). An unclassifiable column raises."""
    carbons, nitrogens, protons = [], [], []
    for column in SHIFT_COLUMNS:
        atom = column[:-len('_shift')]
        bucket = {'c': carbons, 'n': nitrogens, 'h': protons}.get(atom[0])
        if bucket is None:
            raise ValueError(f'{column!r}: cannot classify by element')
        bucket.append(column)
    return tuple(carbons), tuple(nitrogens), tuple(protons)


_CARBONS, _NITROGENS, _PROTONS = _by_element()

# C' is calibrated in its own experiment and keeps its own offset (RefDB p.179 groups by
# experiment, not by element). Every other carbon takes the joint CA/CB aliphatic value.
CO_COL = 'c_shift'
NUC2COLS = {'CA': ('ca_shift',), 'CB': ('cb_shift',), 'CO': (CO_COL,),
            'N': _NITROGENS}

# Every carbon except C' and the two the offset was fitted on, including aromatic-ring
# carbons (usually a separate experiment) -- corrected with the aliphatic value since the
# dominant term is the shared 13C DSS zero point, not an aromatic-specific one. Measured:
# secondary shift vs CA offset gives slope -0.956 (r -0.906) aliphatic, -0.931 aromatic
# (expectation -1 if shared; already-corrected columns give ~0).
SIDECHAIN_C_COLS = tuple(c for c in _CARBONS
                         if c not in (CO_COL, 'ca_shift', 'cb_shift'))

def offset_columns(offsets):
    """{nucleus: offset} -> {shift column: offset}, added to the observed shift. CA/CB
    share one joint aliphatic offset (see derive_rulesearch_offsets); side-chain carbons
    take CA's value, or CB's if only CB was derived."""
    offsets = {nuc: value for nuc, value in (offsets or {}).items() if value}
    cols = {col: value for nuc, value in offsets.items()
            for col in NUC2COLS.get(nuc, ())}
    aliphatic = offsets.get('CA', offsets.get('CB'))
    if aliphatic is not None:
        cols.update(dict.fromkeys(SIDECHAIN_C_COLS, aliphatic))
    return cols


def best_shift(body, predictions):
    """Find the integer residue-number shift maximizing sequence identity."""
    candidates = Counter()
    for sequence_id, (aa, _) in body.items():
        for predicted_id, (predicted_aa, _) in predictions.items():
            if predicted_aa == aa:
                candidates[predicted_id - sequence_id] += 1
    if not candidates:
        return 0, 0.0
    best = (0, -1, 0)
    for shift, _ in candidates.most_common(20):
        matches = overlap = 0
        for sequence_id, (aa, _) in body.items():
            predicted = predictions.get(sequence_id + shift)
            if predicted:
                overlap += 1
                matches += predicted[0] == aa
        if overlap >= 5 and matches > best[1]:
            best = (shift, matches, overlap)
    shift, matches, overlap = best
    return shift, (matches / overlap if overlap else 0.0)
