"""Named row- and cell-matching rules for prediction cohorts."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .domain import VALID_SOURCES
from .statistics import BootstrapSpec, score


def load_paired_cells(path: str | Path):
    """Load the shared per-cell benchmark format; residuals are raw, prediction minus
    the resolved observed value."""
    with np.load(path, allow_pickle=True) as dump:
        arms = [str(arm) for arm in dump["arms"]]
        atom = dump["atom"].astype(str)
        label = dump["label_set"].astype(str)
        residual = {
            arm: (dump[f"pred_{arm}"] - dump["true_ppm"]).astype(np.float64)
            for arm in arms
        }
        payload = {
            "arms": arms,
            "atom": atom,
            "residue": dump["residue"].astype(str),
            "label": label,
            "sid": dump["sid"].astype(str),
            "source": dump["source"].astype(str),
            "bmrb": dump["bmrb"].astype(str),
            "true": dump["true_ppm"],
        }
    _, label_code = np.unique(label, return_inverse=True)
    return {
        **payload,
        "resid": residual,
        "label_id": label_code,
        "n_labels": int(label_code.max()) + 1,
    }


def matched_label_sets(predictions) -> np.ndarray:
    """Select rows whose label set is represented by every structure source."""
    seen = defaultdict(set)
    for label_set, source in zip(
        predictions.label_set, predictions.source, strict=True
    ):
        seen[label_set].add(int(source))
    complete = [key for key, sources in seen.items() if len(sources) == len(VALID_SOURCES)]
    return np.isin(predictions.label_set, complete)


def matched_cells(predictions, label_residue) -> tuple[np.ndarray, np.ndarray]:
    """Return cells scorable for one label residue from every structure source."""
    _, label_residue_code = np.unique(label_residue, return_inverse=True)
    label_residue_code = np.asarray(label_residue_code).ravel()
    n_residues = int(label_residue_code.max()) + 1
    seen = []
    for source_code, _name in enumerate(VALID_SOURCES):
        rows = np.flatnonzero(predictions.source == source_code)
        residue_codes = label_residue_code[rows]
        if len(np.unique(residue_codes)) != len(residue_codes):
            raise ValueError("a label-residue/source pair occurs more than once")
        table = np.zeros((n_residues, len(predictions.shift_cols)), dtype=bool)
        table[residue_codes] = predictions.effective_mask[rows]
        seen.append(table)
    complete = np.logical_and.reduce(seen)
    return complete[label_residue_code], label_residue_code


def by_source(
    predictions,
    *,
    bootstrap: BootstrapSpec,
    matched: bool = False,
    columns=None,
) -> dict:
    """Return per-source scores on either marginal or label-set-matched rows."""
    keep = (
        matched_label_sets(predictions)
        if matched
        else np.ones(predictions.n_rows, dtype=bool)
    )
    return {
        name: score(
            predictions,
            rows=keep & (predictions.source == code),
            columns=columns,
            bootstrap=bootstrap,
        )
        for code, name in enumerate(VALID_SOURCES)
    }
