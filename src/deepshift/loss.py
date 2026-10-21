"""Chemical equivalence, prochiral swaps, and the training-loss reduction."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .normalization import BACKBONE_SHIFTS, STD_FLOOR

EQUIV_GROUPS = {
    "ALA": [("HB1", "HB2", "HB3")],
    "VAL": [("HG11", "HG12", "HG13"), ("HG21", "HG22", "HG23")],
    "LEU": [("HD11", "HD12", "HD13"), ("HD21", "HD22", "HD23")],
    "ILE": [("HG21", "HG22", "HG23"), ("HD11", "HD12", "HD13")],
    "THR": [("HG21", "HG22", "HG23")],
    "MET": [("HE1", "HE2", "HE3")],
    "PHE": [("HD1", "HD2"), ("HE1", "HE2"), ("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("HD1", "HD2"), ("HE1", "HE2"), ("CD1", "CD2"), ("CE1", "CE2")],
}

PROCHIRAL_PAIRS = {
    "GLY": [("HA2", "HA3")],
    "SER": [("HB2", "HB3")],
    "CYS": [("HB2", "HB3")],
    "ASP": [("HB2", "HB3")],
    "ASN": [("HB2", "HB3"), ("HD21", "HD22")],
    "PHE": [("HB2", "HB3")],
    "TYR": [("HB2", "HB3")],
    "HIS": [("HB2", "HB3")],
    "TRP": [("HB2", "HB3")],
    "LEU": [
        ("HB2", "HB3"),
        ("CD1", "CD2"),
        ("HD11", "HD21"),
        ("HD12", "HD22"),
        ("HD13", "HD23"),
    ],
    "VAL": [
        ("CG1", "CG2"),
        ("HG11", "HG21"),
        ("HG12", "HG22"),
        ("HG13", "HG23"),
    ],
    "ILE": [("HG12", "HG13")],
    "ARG": [
        ("HB2", "HB3"),
        ("HG2", "HG3"),
        ("HD2", "HD3"),
        ("HH11", "HH21"),
        ("HH12", "HH22"),
    ],
    "LYS": [("HB2", "HB3"), ("HG2", "HG3"), ("HD2", "HD3"), ("HE2", "HE3")],
    "PRO": [("HB2", "HB3"), ("HG2", "HG3"), ("HD2", "HD3")],
    "GLN": [("HB2", "HB3"), ("HG2", "HG3"), ("HE21", "HE22")],
    "GLU": [("HB2", "HB3"), ("HG2", "HG3")],
    "MET": [("HB2", "HB3"), ("HG2", "HG3")],
}

AMBIGUITY_SET_CODES = (4, 5, 6, 9)


def _columns(atoms):
    return tuple(f"{atom.lower()}_shift" for atom in atoms)


def _loss_structure(shift_cols):
    index = {column: i for i, column in enumerate(shift_cols)}
    group_weights, pairs = {}, {}
    for residue in set(EQUIV_GROUPS) | set(PROCHIRAL_PAIRS):
        weights = np.ones(len(shift_cols), dtype=np.float32)
        for group in EQUIV_GROUPS.get(residue, []):
            present = [index[column] for column in _columns(group) if column in index]
            if len(present) >= 2:
                weights[present] = 1.0 / len(present)
        group_weights[residue] = weights
        pairs[residue] = [
            (index[first], index[second])
            for first, second in map(_columns, PROCHIRAL_PAIRS.get(residue, []))
            if first in index and second in index
        ]
    return group_weights, pairs


def build_loss_tensors(shift_cols, standard_residues):
    """Return equivalence weights and prochiral partner indices."""
    group_map, pair_map = _loss_structure(shift_cols)
    n_aa, n_shifts = len(standard_residues), len(shift_cols)
    group_weight = torch.ones(n_aa, n_shifts, dtype=torch.float32)
    partner = torch.full((n_aa, n_shifts), -1, dtype=torch.long)
    for aa_index, residue in enumerate(standard_residues):
        if residue in group_map:
            group_weight[aa_index] = torch.from_numpy(group_map[residue])
        for first, second in pair_map.get(residue, []):
            partner[aa_index, first] = second
            partner[aa_index, second] = first
    return group_weight, partner


def hierarchical_weight_recipe(effective_counts, shift_cols):
    """Fit the fixed 60/40 backbone/side-chain loss weighting recipe."""
    counts = np.asarray(effective_counts, dtype=np.float64)
    backbone = np.array([column in BACKBONE_SHIFTS for column in shift_cols])
    sidechain_indices = np.flatnonzero(~backbone)
    sidechain_counts = counts[sidechain_indices]
    positive = sidechain_counts[sidechain_counts > 0]
    median = float(np.median(positive)) if positive.size else 0.0
    weights = np.zeros(len(shift_cols), dtype=np.float64)
    for index in sidechain_indices:
        if counts[index] > 0 and median > 0:
            weights[index] = float(np.clip(np.sqrt(median / counts[index]), 0.5, 2.0))
        else:
            weights[index] = 2.0
    order = sidechain_indices[np.argsort(sidechain_counts, kind="stable")]
    count = max(1, round(len(sidechain_indices) * 0.25))
    rare = [shift_cols[index] for index in order[:count]]
    recipe = {
        "backbone_group": [column for column in shift_cols if column in BACKBONE_SHIFTS],
        "sidechain_group": [shift_cols[index] for index in sidechain_indices],
        "effective_counts": {
            shift_cols[index]: float(counts[index]) for index in range(len(shift_cols))
        },
        "sidechain_weights": {
            shift_cols[index]: float(weights[index]) for index in sidechain_indices
        },
        "median_positive_count": median,
        "rare_nucleus_quartile": rare,
    }
    return recipe, backbone.astype(np.float32), weights.astype(np.float32)


def swap_elements(
    pred,
    target,
    mask,
    aa_code,
    group_weight,
    partner,
    delta,
    center,
    scale,
    ambiguity,
):
    """Return per-cell Huber losses after allowed prochiral swaps."""
    aa = aa_code.clamp(0, group_weight.size(0) - 1)
    equivalence_weight = group_weight[aa]
    partner_index = partner[aa]
    has_partner, safe_partner = partner_index >= 0, partner_index.clamp(min=0)
    direct = F.huber_loss(pred, target, reduction="none", delta=delta)
    center_i = center[aa]
    scale_i = scale[aa].clamp(min=STD_FLOOR)
    center_j = torch.gather(center_i, 1, safe_partner)
    scale_j = torch.gather(scale_i, 1, safe_partner)
    target_j_ppm = torch.gather(target, 1, safe_partner) * scale_j + center_j
    partner_target = (target_j_ppm - center_i) / scale_i
    partner_mask = torch.gather(mask.float(), 1, safe_partner) > 0.5
    crossed = F.huber_loss(pred, partner_target, reduction="none", delta=delta)
    both = has_partner & mask & partner_mask
    pair_cost = torch.minimum(
        direct + torch.gather(direct, 1, safe_partner),
        crossed + torch.gather(crossed, 1, safe_partner),
    )
    swapped = torch.where(both, 0.5 * pair_cost, direct)
    effective_mask = mask
    if ambiguity is None:
        element = swapped
    else:
        partner_code = torch.gather(ambiguity, 1, safe_partner)
        set_codes = torch.tensor(AMBIGUITY_SET_CODES, device=ambiguity.device)
        self_set = torch.isin(ambiguity, set_codes)
        partner_set = torch.isin(partner_code, set_codes)
        swap_ok = both & ~self_set & ~partner_set & ~(
            (ambiguity == 1) & (partner_code == 1)
        )
        element = torch.where(swap_ok, swapped, direct)
        effective_mask = mask & ~self_set
    return element, equivalence_weight, effective_mask


def hierarchical_loss(element, equivalence_weight, effective_mask, backbone, sidechain_weight):
    """Reduce cell losses with fixed 60% backbone and 40% side-chain group weights."""
    active = effective_mask.to(element.dtype)
    backbone = backbone.to(element.dtype).unsqueeze(0)
    sidechain = sidechain_weight.to(element.dtype).unsqueeze(0)
    numerator_backbone = (element * equivalence_weight * active * backbone).sum()
    denominator_backbone = (equivalence_weight * active * backbone).sum()
    numerator_sidechain = (element * equivalence_weight * active * sidechain).sum()
    denominator_sidechain = (equivalence_weight * active * sidechain).sum()
    parts, weight_sum = [], 0.0
    if float(denominator_backbone) > 0:
        parts.append(0.60 * (numerator_backbone / denominator_backbone))
        weight_sum += 0.60
    if float(denominator_sidechain) > 0:
        parts.append(0.40 * (numerator_sidechain / denominator_sidechain))
        weight_sum += 0.40
    if not parts:
        return element.new_tensor(float("nan"))
    return sum(parts) / weight_sum

