"""The one ppm-to-model-unit transform used by training and prediction."""
from __future__ import annotations

import copy

import numpy as np
import torch

from .data.norm import BMRB_FILT, get_center
from .domain import STANDARD_RESIDUES

BACKBONE_SHIFTS = ("h_shift", "ha_shift", "c_shift", "ca_shift", "cb_shift", "n_shift")
STD_FLOOR = 1e-6


def build_fixed_per_aa(stats, shift_cols):
    """Attach the fixed-center and fitted per-nucleus scale table to ``stats``."""
    new = copy.deepcopy(stats) if stats else {}
    fitted_snuc = (stats or {}).get("snuc") or {}
    if missing := [col for col in shift_cols if col not in fitted_snuc]:
        raise ValueError(f"train-fold S_NUC table has no scale for {missing}")
    per_aa = {}
    for aa3 in BMRB_FILT:
        values = {}
        for col in shift_cols:
            center, scale = get_center(aa3, col), fitted_snuc.get(col, 1.0)
            if center is None:
                global_stats = (stats or {}).get(col)
                center = global_stats["mean"] if global_stats else 0.0
            values[col] = {"mean": float(center), "std": float(max(scale, 0.1))}
        per_aa[aa3] = values
    new["per_aa"] = per_aa
    new["normalization"] = (stats or {}).get("normalization") or (
        "secondary_snuc_legacy_allfolds_2026-07-09"
    )
    return new


def build_aa_tables(stats, shift_cols):
    """Return the residue-by-nucleus center and scale tables."""
    per_aa_stats = stats.get("per_aa", {}) if isinstance(stats, dict) else {}
    n_aa = len(STANDARD_RESIDUES)
    means = np.zeros((n_aa + 1, len(shift_cols)), dtype=np.float64)
    stds = np.ones((n_aa + 1, len(shift_cols)), dtype=np.float64)
    for si, col in enumerate(shift_cols):
        has_global = isinstance(stats, dict) and col in stats
        global_mean = stats[col]["mean"] if has_global else 0.0
        global_std = max(stats[col]["std"], STD_FLOOR) if has_global else 1.0
        for ai, aa_name in enumerate(STANDARD_RESIDUES):
            aa_cell = per_aa_stats.get(aa_name, {}).get(col)
            if aa_cell:
                means[ai, si] = aa_cell["mean"]
                stds[ai, si] = max(aa_cell["std"], STD_FLOOR)
            else:
                means[ai, si] = global_mean
                stds[ai, si] = global_std
        means[n_aa, si] = global_mean
        stds[n_aa, si] = global_std
    return means, stds, n_aa


def resolve_residue_codes(residue_codes, n_aa):
    """Map invalid residue codes to the global-fallback table row."""
    codes = np.asarray(residue_codes).astype(np.int64).reshape(-1)
    in_range = (codes >= 0) & (codes < n_aa)
    return np.where(in_range, codes, n_aa)


def denormalize(pred, residue_codes, stats, shift_cols):
    """Convert model units to ppm."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    pred = np.asarray(pred, dtype=np.float64)
    means, stds, n_aa = build_aa_tables(stats, shift_cols)
    safe = resolve_residue_codes(residue_codes, n_aa)
    return pred * stds[safe] + means[safe]


def normalize(ppm, residue_codes, stats, shift_cols):
    """Convert ppm to model units."""
    if isinstance(ppm, torch.Tensor):
        ppm = ppm.detach().cpu().numpy()
    ppm = np.asarray(ppm, dtype=np.float64)
    means, stds, n_aa = build_aa_tables(stats, shift_cols)
    safe = resolve_residue_codes(residue_codes, n_aa)
    return (ppm - means[safe]) / stds[safe]

