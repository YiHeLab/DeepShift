"""The matched structure-source cohort that ``fig_plddt_source.py`` decomposes.

Pooling every held-out AlphaFold, NMR and crystal residue and scoring the three
piles is confounded, because the three piles are not the same proteins. The control is
exact pairing, never reweighting. Every row of the held-out CV dumps is one (structure
realization, residue), and ``label_residue_id`` names the observed shift that row is scored
against. A cell realized by all three sources gives three predictions of the *same
experimental number* from three structures of the same protein, so a difference between
the arms cannot be a difference in cohort difficulty. This is a strict refinement of
``evaluate.matched_rows`` (which matches whole label sets): matching the label residue also
removes the residue-composition difference that survives inside a matched protein, because
a crystal structure need not resolve every residue the NMR structure does. Multiplicity is
checked, never assumed -- each (label residue, source) occurs exactly once -- so no
averaging over duplicate structures is involved.
"""
from __future__ import annotations

from dataclasses import replace

import frozen
import numpy as np
import style

from deepshift import evaluate
from deepshift.domain import VALID_SOURCES

PREDS = frozen.PREDS_DIR

SOURCE_NAME = {"af": "AlphaFold", "nmr": "NMR", "xtal": "crystal"}
SOURCE_COLOR = {"af": style.PALETTE["slot1"], "nmr": style.PALETTE["slot2"],
                "xtal": style.PALETTE["slot3"]}

_CELLS = ("pred_ppm", "true_ppm", "resolved_ppm", "mask", "effective_mask", "ambiguity")
_ROWS = ("residue_code", "row_key", "label_set", "bmrb", "source")


# --------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------

def columns_only(preds, columns):
    """Keep six nuclei out of sixty-four before pooling five ~1 GB dumps."""
    take = [preds.shift_cols.index(c) for c in columns]
    return replace(preds, shift_cols=tuple(columns),
                   **{f: getattr(preds, f)[:, take] for f in _CELLS})


def rows_only(preds, rows):
    return replace(preds, **{f: getattr(preds, f)[rows] for f in _CELLS + _ROWS})


def label_residue_of(path) -> np.ndarray:
    """The label-residue identity, read straight from the dump.

    ``read_predictions`` folds it into ``row_key`` together with the materialization, and
    recovering it by splitting that string would depend on a private key format.
    """
    with np.load(path, allow_pickle=False) as dump:
        return np.asarray([str(x) for x in dump["label_residue_id"]])


def read(paths, columns):
    """Pool the held-out folds into one population, carrying label-residue identity."""
    parts, label_residue = [], []
    for path in paths:
        parts.append(columns_only(evaluate.read_predictions(path), columns))
        label_residue.append(label_residue_of(path))
    return evaluate.pool(parts), np.concatenate(label_residue)


def apply_fence(preds, fence: dict, label_residue):
    """Drop QC-flagged (label residue, nucleus) cells from every arm identically."""
    drop = np.zeros_like(preds.effective_mask)
    for i, key in enumerate(label_residue):
        for col in fence.get(key, ()):
            if col in preds.shift_cols:
                drop[i, preds.shift_cols.index(col)] = True
    kept = preds.effective_mask & ~drop
    return replace(preds, effective_mask=kept), int((preds.effective_mask & drop).sum())


# --------------------------------------------------------------------------------
# The matched cohort
# --------------------------------------------------------------------------------

def paired_arms(preds, cells, lr_id):
    """One Predictions per source over identical label residues, in one order.

    The arms are re-keyed on the label residue so ``evaluate.compare`` -- which refuses
    arms that do not describe the same rows -- can pair them. Its label agreement check
    then also verifies the pairing: three arms keyed to one label residue must be scored
    against the same deposited ppm.
    """
    keep_row = cells.any(axis=1)
    arms = {}
    for code, name in enumerate(VALID_SOURCES):
        rows = np.flatnonzero(keep_row & (preds.source == code))
        rows = rows[np.argsort(lr_id[rows], kind="stable")]
        arm = rows_only(replace(preds, effective_mask=preds.effective_mask & cells), rows)
        arms[name] = replace(arm, row_key=lr_id[rows].astype(str))
    order = [arms[n].row_key for n in VALID_SOURCES]
    if not all(np.array_equal(order[0], o) for o in order[1:]):
        raise SystemExit("the three arms do not carry the same label residues")
    return arms
