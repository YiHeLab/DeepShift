#!/usr/bin/env python3
"""The fold-6 cells every competitor scorer reads, and how each method's cell is formed.

The rules, applied identically to our model and every competitor:
- Targets are the raw deposited shifts (npz raw `true`), not the DeepShift-resolved
  `true_ppm`; each method resolves prochiral pairs independently by the package rule in
  ``deepshift.evaluate.resolve``, the rule the trainer and the evaluator use.
- A cell is scorable only where the fold-6 ensemble declares it observed, so the run's
  exclusion fence -- the p100 spans and the structural cells composed into one authority
  -- reaches every arm through one application, on the ground truth.
- Residue-name normalization maps SHIFTX2 'B' -> CYS, recovering cystine cells.
- Self-exclusion: for every homology tool, inspect the hit log and drop any protein whose
  kept DB candidate is same-BMRB, >=90% identity, or same-UniProt (SIFTS) to the query.
  A leaky protein is dropped before scoring and never contributes a cell.
"""
import json
from collections import defaultdict

import numpy as np

from deepshift import evaluate
from deepshift.chemistry import AA_3_TO_1, STANDARD_RESIDUES
from deepshift.loss import build_loss_tensors
from deepshift.self_exclusion import (
    RETRIEVAL_ARMS as HOMOLOGY_TOOLS,
)
from deepshift.self_exclusion import (
    ExclusionLevel,
    arm_directory,
)
from deepshift.self_exclusion import (
    verify_self_exclusion as verify_retrieval_exclusion,
)

from . import cells as common

OUT_ROOT = common.OUT_ROOT
RESNAME_ALIAS = {'B': 'C', 'CYX': 'C', 'CSS': 'C', 'HID': 'H', 'HIE': 'H', 'HIP': 'H',
                 'MSE': 'M'}                 # SHIFTX2 'B' = cystine; common variants
NUC_ALIAS = {'HN': 'H', "C'": 'C', 'CO': 'C', 'HN1': 'H'}


def one_letter(res):
    res = (res or '').strip().upper()
    res = RESNAME_ALIAS.get(res, res)
    return res if len(res) == 1 else AA_3_TO_1.get(res, res[:1] if res else '')


def _normalize_cell(cell):
    out = {}
    for k, v in cell.items():
        out['resname' if k == 'resname' else NUC_ALIAS.get(k.upper(), k.upper())] = v
    return out


def load_comp(tool, sid):
    p = arm_directory(OUT_ROOT, tool) / (sid + '.json')
    if not p.exists():
        return None
    with open(p) as fh:
        d = json.load(fh)
    d['predictions'] = {rn: _normalize_cell(c) for rn, c in d.get('predictions', {}).items()}
    return d


# ---------- fold-6 rows for the requested structures (raw target + partner) ----------

def fold6_rows(structure_ids):
    """Per structure instance, the scorable cells of each of its residues.

    A cell is scorable where the fold-6 ensemble scored it: the exclusion fence was
    applied to those predictions when they were written.
    """
    npz = common.load_pred_npz()
    cols = npz['cols']
    atom_of_col = [c[:-6].upper() for c in cols]
    _go, rows = common.build_row_index()
    sid_of = [common.structure_id(r) for r in rows]
    want = set(structure_ids)
    aa = npz['aa']
    amb = npz['amb']
    partner = build_loss_tensors(tuple(cols), tuple(STANDARD_RESIDUES))[1].numpy()
    out = defaultdict(list)
    for i, r in enumerate(rows):
        sid = sid_of[i]
        if sid not in want:
            continue
        obs_cols = [int(j) for j in np.flatnonzero(npz['mask'][i])]
        out[sid].append({'struct_resnum': r['struct_resnum'], 'res1': one_letter(r['resname']),
                          'label_set_id': r['label_set_id'], 'aa': int(aa[i]),
                          'obs_cols': obs_cols, 'true_raw': npz['true_raw'][i],
                          'amb': amb[i], 'our_pred': npz['pred'][i]})
    return out, cols, atom_of_col, partner


def _resolved_cells(pred_vec, true_raw, obs_cols, aa, partner, amb):
    """Per-method ``{col: (prediction, target)}`` on the observed cells, in ppm.

    Prochiral pairs are resolved by ``deepshift.evaluate.resolve``, the one rule the
    trainer and the evaluator use, applied to this method's own prediction, so the
    target a method is scored against is the assignment its own prediction chose. A
    cell the method did not predict is absent.
    """
    pred = np.asarray(pred_vec, dtype=np.float64)[None, :]
    mask = np.zeros(pred.shape, dtype=bool)
    mask[0, list(obs_cols)] = True
    mask &= np.isfinite(pred)
    resolved, effective = evaluate.resolve(
        np.where(mask, pred, 0.0), np.asarray(true_raw, dtype=np.float64)[None, :], mask,
        np.asarray(amb)[None, :], np.asarray([aa]), None, partner=np.asarray(partner))
    return {int(j): (float(pred[0, j]), float(resolved[0, j]))
            for j in np.flatnonzero(effective[0])}


def verify_self_exclusion(tool, sid, task, exc):
    """Verify the benchmark's broad-homology sensitivity policy."""
    return verify_retrieval_exclusion(
        tool,
        sid,
        task,
        exc,
        output_root=OUT_ROOT,
        retrieval=tool in HOMOLOGY_TOOLS,
        level=ExclusionLevel.BROAD_HOMOLOGY,
    )

