#!/usr/bin/env python3
"""Fit a 64-nucleus p100 fence from genuine out-of-fold predictions.

For calibration fold k, ``refold_fold{k}.final.pt`` must declare k as its held-out
fold and ``refold_fold{k}.preds.npz`` must address fold k's cache rows exactly by
``(materialization_id, label_residue_id)``.  The per-nucleus error scale is the
held-out MAE on the prediction dump's effective evaluation mask.  That scale is
then used to test every originally observed, atom-specific label on every
model-valid row from the explicit unfiltered calibration cache.
Prochiral swaps are resolved exactly as in evaluation, then acceptance is mapped
back to the deposited atom name before fitting the deposited-name interval.

The original-observation mask comes from the immutable fold CSV.  Corrected ppm
values come from ``structural/shifts.npy``, because the cache has already applied
the build's bound rereferencing offsets there. This avoids reimplementing rereferencing.

After fitting on folds 1--5, the script applies the ranges to the originally
observed, structurally matchable cells in folds 0--6 and writes exact
``label_residue_id -> [shift_col, ...]`` flags.  It fails closed if any observed
residue/nucleus population lacks a calibrated range.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPOSITORY_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPOSITORY_SRC) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_SRC))
from deepshift import quality_control as flag_rules
from deepshift.atom_cloud import AtomCloudRecipe
from deepshift.chemistry import CHEMICALLY_IMPOSSIBLE_PAIRS
from deepshift.loss import AMBIGUITY_SET_CODES
from deepshift.quality_control import ARG_NE_OBSERVED_RANGE_PPM

CALIBRATION_FOLDS = (1, 2, 3, 4, 5)
APPLICATION_FOLDS = (0, 1, 2, 3, 4, 5, 6)
IMPOSSIBLE_PAIRS = frozenset(
    f"{residue}:{column}" for residue, column in CHEMICALLY_IMPOSSIBLE_PAIRS
)
IDENTITY_COLUMNS = (
    "materialization_id",
    "label_residue_id",
    "residue_code",
    "mismatch_type",
)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _validate_calibration_checkpoint_manifest(
    manifest: dict,
    *,
    run_root: Path,
    heldout_fold: int,
    checkpoint_epoch: int,
) -> None:
    """Require an unfiltered p100 member trained on the other four folds."""
    expected_folds = list(CALIBRATION_FOLDS)
    expected_train = [fold for fold in expected_folds if fold != heldout_fold]
    arguments = manifest.get("arguments") or {}
    protocol = manifest.get("protocol") or {}
    folds = manifest.get("folds") or {}
    inputs = manifest.get("inputs") or {}
    flags = inputs.get("flags") or {}
    if flags.get("policy") != "unfiltered_calibration" or int(
            flags.get("unique_cells", -1)) != 0:
        raise ValueError(
            f"fold {heldout_fold}: p100 calibration checkpoint is not stamped as "
            "unfiltered_calibration with zero excluded cells"
        )
    if (arguments.get("fold") != heldout_fold
            or folds.get("heldout") != [heldout_fold]
            or folds.get("train") != expected_train):
        raise ValueError(
            f"fold {heldout_fold}: checkpoint does not use the other four folds "
            "as its complete training set"
        )
    if (protocol.get("checkpoint_selection") != "final"
            or protocol.get("test_fold_used_for_selection") is not False
            or checkpoint_epoch != int(protocol.get("epochs", -1))):
        raise ValueError(
            f"fold {heldout_fold}: checkpoint is not the unselected final epoch"
        )

    cache = run_root / "work" / "unified_cache"
    recipes = []
    for fold in expected_folds:
        fold_dir = cache / f"fold_{fold}"
        atom_config = json.loads((fold_dir / "atom" / "config.json").read_text())
        recipes.append(atom_config.get("recipe", atom_config))
    if any(recipe != manifest.get("recipe") for recipe in recipes):
        raise ValueError(
            f"fold {heldout_fold}: checkpoint atom recipe does not match every cache fold"
        )

    heldout_config = json.loads(
        (cache / f"fold_{heldout_fold}" / "config.json").read_text()
    )
    stats = heldout_config.get("stats") or {}
    normalization = manifest.get("normalization") or {}
    if (normalization.get("marker") != stats.get("normalization")
            or normalization.get("fit") != stats.get("normalization_fit")):
        raise ValueError(
            f"fold {heldout_fold}: checkpoint normalization does not match its cache fold"
        )


def load_domain(source_root: Path):
    domain_path = source_root / "src" / "deepshift" / "domain.py"
    if not domain_path.is_file():
        raise FileNotFoundError(f"missing domain authority: {domain_path}")
    spec = importlib.util.spec_from_file_location("_p100_domain", domain_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {domain_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, domain_path


@dataclass
class FoldData:
    fold: int
    pred: np.ndarray | None
    dump_mask: np.ndarray | None
    dump_effective_mask: np.ndarray | None
    dump_resolved_ppm: np.ndarray | None
    pred_global_rows: np.ndarray | None
    raw_mask: np.ndarray
    structurally_matchable: np.ndarray
    shifts: np.ndarray
    ambiguity: np.ndarray
    residue_idx: np.ndarray
    label_residue_ids: np.ndarray
    config: dict
    csv_rows: int
    predicted_rows: int
    dump_mask_cells: int
    dump_effective_mask_cells: int
    true_ppm_max_abs_delta: float
    stored_resolved_vs_canonical_max_abs_delta: float
    model_valid_pred_rows: np.ndarray | None
    prediction_missing_cache_rows: int
    prediction_missing_in_scope_rows: int
    prediction_missing_model_valid_in_scope_rows: int


def _cache_identity(fold_dir: Path) -> tuple[pd.MultiIndex, np.ndarray, np.ndarray]:
    structural = fold_dir / "structural"
    materialization_ids = json.loads(
        (structural / "materialization_ids.json").read_text()
    )
    materialization_idx = np.load(
        structural / "materialization_idx.npy", mmap_mode="r"
    )
    label_map = json.loads(
        (structural / "global_to_label_residue.json").read_text()
    )
    n_rows = len(materialization_idx)
    if len(label_map) != n_rows:
        raise ValueError(
            f"{fold_dir}: label-residue map has {len(label_map)} rows, expected {n_rows}"
        )
    try:
        label_ids = np.asarray([label_map[str(i)] for i in range(n_rows)], dtype=object)
    except KeyError as exc:
        raise ValueError(f"{fold_dir}: non-contiguous label-residue map") from exc
    if np.any(materialization_idx < 0) or np.any(
        materialization_idx >= len(materialization_ids)
    ):
        raise ValueError(f"{fold_dir}: invalid materialization index")
    materialization = np.asarray(
        [materialization_ids[int(i)] for i in materialization_idx], dtype=object
    )
    identity = pd.MultiIndex.from_arrays(
        [materialization, label_ids],
        names=["materialization_id", "label_residue_id"],
    )
    if not identity.is_unique:
        raise ValueError(f"{fold_dir}: cache row identities are not unique")
    return identity, label_ids, np.asarray(materialization_idx)


def _csv_support(
    csv_path: Path,
    shift_cols: tuple[str, ...],
    cache_identity: pd.MultiIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    frame = pd.read_csv(
        csv_path,
        usecols=[*IDENTITY_COLUMNS, *shift_cols],
        low_memory=False,
    )
    csv_identity = pd.MultiIndex.from_frame(
        frame[["materialization_id", "label_residue_id"]]
    )
    if not csv_identity.is_unique:
        duplicates = int(csv_identity.duplicated().sum())
        raise ValueError(f"{csv_path}: {duplicates} duplicate cache identities")
    positions = csv_identity.get_indexer(cache_identity)
    if np.any(positions < 0):
        raise ValueError(
            f"{csv_path}: {int(np.sum(positions < 0))} cache rows missing from fold CSV"
        )
    if len(frame) != len(cache_identity):
        cache_set = set(cache_identity.tolist())
        extras = sum(key not in cache_set for key in csv_identity.tolist())
        raise ValueError(
            f"{csv_path}: CSV/cache row count differs "
            f"({len(frame)} vs {len(cache_identity)}; {extras} CSV-only identities)"
        )
    ordered = frame.iloc[positions]
    raw_mask = ordered[list(shift_cols)].notna().to_numpy(dtype=bool)
    matchable = ordered["mismatch_type"].fillna("").eq("match").to_numpy(dtype=bool)
    residue_names = ordered["residue_code"].fillna("UNK").astype(str).str.upper().to_numpy()
    return raw_mask, matchable, residue_names, len(frame)


def _model_valid_rows(atom_dir: Path, rows: np.ndarray) -> np.ndarray:
    """Replay the model's ``center.any()`` validity gate for selected cache rows."""
    config = json.loads((atom_dir / "config.json").read_text())
    center_offset = AtomCloudRecipe.from_mapping(config["recipe"]).center_offset_code
    atom_offset = np.load(atom_dir / "atom_offset.npy", mmap_mode="r")
    atom_count = np.load(atom_dir / "atom_count.npy", mmap_mode="r")
    if atom_offset.ndim != 2 or atom_count.shape != (len(atom_offset),):
        raise ValueError(f"{atom_dir}: atom offset/count shapes disagree")
    if np.any(rows < 0) or np.any(rows >= len(atom_offset)):
        raise ValueError(f"{atom_dir}: requested model-valid rows are out of range")
    valid = np.zeros(len(rows), dtype=bool)
    chunk_size = 8192
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        selected_rows = rows[start:stop]
        offsets = np.asarray(atom_offset[selected_rows])
        counts = np.asarray(atom_count[selected_rows], dtype=np.int64)
        if np.any(counts < 0) or np.any(counts > atom_offset.shape[1]):
            raise ValueError(f"{atom_dir}: atom_count is outside the atom array width")
        positions = np.arange(atom_offset.shape[1])[None, :]
        valid[start:stop] = np.any(
            (positions < counts[:, None]) & (offsets == center_offset),
            axis=1,
        )
    return valid


def load_fold(
    *,
    fold: int,
    run_root: Path,
    shift_cols: tuple[str, ...],
    residues: tuple[str, ...],
    with_predictions: bool,
    allow_unsafe_prediction_pickle: bool = False,
    prediction_resolver=None,
) -> FoldData:
    """Validate and open one fold's cache, CSV, and optionally its prediction dump.

    Each pass reloads the fold and never caches it: the arrays are large, which is
    why every pass ends by dropping them.
    """
    fold_dir = run_root / "work" / "unified_cache" / f"fold_{fold}"
    csv_path = (
        run_root
        / "work"
        / "unified_folds"
        / f"structure_data_unified_fold_{fold}.csv"
    )
    config = json.loads((fold_dir / "config.json").read_text())
    if tuple(config.get("shift_cols", ())) != shift_cols:
        raise ValueError(f"fold {fold}: cache shift columns differ from domain authority")
    cache_identity, label_ids, _ = _cache_identity(fold_dir)
    raw_mask, matchable, csv_residues, csv_rows = _csv_support(
        csv_path, shift_cols, cache_identity
    )
    structural = fold_dir / "structural"
    shifts = np.load(structural / "shifts.npy", mmap_mode="r")
    cache_mask = np.load(structural / "shift_mask.npy", mmap_mode="r")
    ambiguity = np.load(structural / "ambiguity.npy", mmap_mode="r")
    residue_idx = np.load(structural / "residue_idx.npy", mmap_mode="r")
    if (
        shifts.shape != raw_mask.shape
        or cache_mask.shape != raw_mask.shape
        or ambiguity.shape != raw_mask.shape
        or residue_idx.shape != (len(raw_mask),)
    ):
        raise ValueError(f"fold {fold}: cache/CSV shapes disagree")
    normalized_csv_residue = np.asarray(
        [name if name in residues else "UNK" for name in csv_residues], dtype=object
    )
    cache_residue_names = np.asarray(
        [residues[int(code)] if 0 <= int(code) < len(residues) else "<INVALID>"
         for code in residue_idx],
        dtype=object,
    )
    if not np.array_equal(normalized_csv_residue, cache_residue_names):
        raise ValueError(f"fold {fold}: CSV residue names disagree with cache residue codes")
    if np.any(np.asarray(cache_mask, dtype=bool) & ~raw_mask):
        raise ValueError(f"fold {fold}: cache observes cells absent from the fold CSV")
    if np.any(np.asarray(cache_mask, dtype=bool) & ~matchable[:, None]):
        raise ValueError(f"fold {fold}: cache observes structurally unmatchable rows")
    if np.any(raw_mask & ~np.isfinite(np.asarray(shifts))):
        raise ValueError(f"fold {fold}: raw observation mask selects non-finite cache shifts")

    pred = dump_mask = effective_mask = resolved_ppm = pred_rows = None
    predicted_rows = dump_cells = effective_cells = 0
    true_delta = 0.0
    resolved_delta_max = 0.0
    model_valid_pred_rows = None
    missing_cache_rows = 0
    missing_in_scope_rows = 0
    missing_model_valid_in_scope_rows = 0
    if with_predictions:
        checkpoint_path = run_root / f"refold_fold{fold}.final.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_cols = tuple(str(x) for x in checkpoint.get("shift_cols", ()))
        checkpoint_manifest = checkpoint.get("manifest") or {}
        if checkpoint_cols != shift_cols or int(checkpoint.get("n_shifts", -1)) != len(
            shift_cols
        ):
            raise ValueError(f"fold {fold}: checkpoint output schema is not 64-column authority")
        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        _validate_calibration_checkpoint_manifest(
            checkpoint_manifest,
            run_root=run_root,
            heldout_fold=fold,
            checkpoint_epoch=checkpoint_epoch,
        )
        del checkpoint

        pred_path = run_root / f"refold_fold{fold}.preds.npz"
        try:
            with np.load(
                pred_path, allow_pickle=allow_unsafe_prediction_pickle
            ) as dump:
                dump_cols = tuple(str(x) for x in dump["shift_cols"].tolist())
                if dump_cols != shift_cols:
                    raise ValueError(
                        f"fold {fold}: prediction columns differ from authority"
                    )
                pred = np.asarray(dump["pred_ppm"], dtype=np.float64)
                true_ppm = np.asarray(dump["true_ppm"], dtype=np.float64)
                resolved_ppm = np.asarray(
                    dump["resolved_true_ppm"], dtype=np.float64
                )
                dump_mask = np.asarray(dump["mask"], dtype=bool)
                effective_mask = np.asarray(dump["effective_mask"], dtype=bool)
                dump_ambiguity = np.asarray(dump["ambiguity"])
                dump_residue = np.asarray(dump["residue_code"], dtype=np.int64)
                pred_identity = pd.MultiIndex.from_arrays(
                    [
                        np.asarray(dump["materialization_id"], dtype=object),
                        np.asarray(dump["label_residue_id"], dtype=object),
                    ],
                    names=cache_identity.names,
                )
        except ValueError as exc:
            if (
                not allow_unsafe_prediction_pickle
                and "Object arrays cannot be loaded" in str(exc)
            ):
                raise ValueError(
                    f"{pred_path}: object arrays require explicit "
                    "--allow-unsafe-legacy-prediction-pickle; convert the "
                    "dump to Unicode arrays for a release artifact"
                ) from exc
            raise
        if not pred_identity.is_unique:
            raise ValueError(f"fold {fold}: prediction row identities are not unique")
        pred_rows = cache_identity.get_indexer(pred_identity)
        if np.any(pred_rows < 0):
            raise ValueError(
                f"fold {fold}: {int(np.sum(pred_rows < 0))} predictions miss the cache"
            )
        expected_shape = (len(pred_rows), len(shift_cols))
        if (
            pred.shape != expected_shape
            or true_ppm.shape != expected_shape
            or resolved_ppm.shape != expected_shape
            or dump_mask.shape != expected_shape
            or effective_mask.shape != expected_shape
            or dump_ambiguity.shape != expected_shape
        ):
            raise ValueError(f"fold {fold}: prediction array shapes disagree")
        if not np.all(np.isfinite(pred)):
            raise ValueError(f"fold {fold}: non-finite prediction")
        if not np.array_equal(dump_residue, np.asarray(residue_idx[pred_rows])):
            raise ValueError(f"fold {fold}: prediction/cache residue codes disagree")
        if not np.array_equal(dump_ambiguity, np.asarray(ambiguity[pred_rows])):
            raise ValueError(f"fold {fold}: prediction/cache ambiguity codes disagree")
        if np.any(dump_mask & ~np.asarray(cache_mask[pred_rows], dtype=bool)):
            raise ValueError(f"fold {fold}: dump mask contains cells cleared in its cache")
        if np.any(dump_mask & ~raw_mask[pred_rows]):
            raise ValueError(f"fold {fold}: dump mask contains originally absent cells")
        if np.any(effective_mask & ~dump_mask):
            raise ValueError(f"fold {fold}: effective mask contains cells absent from dump mask")
        expected_effective_mask = (
            dump_mask & ~np.isin(ambiguity[pred_rows], AMBIGUITY_SET_CODES)
        )
        if not np.array_equal(effective_mask, expected_effective_mask):
            raise ValueError(
                f"fold {fold}: effective mask is not the canonical atom-specific mask"
            )
        if prediction_resolver is None:
            raise ValueError(f"fold {fold}: prediction resolver is required")
        canonical_resolved, canonical_effective = prediction_resolver(
            pred,
            true_ppm,
            dump_mask,
            dump_ambiguity,
            dump_residue,
            shift_cols,
        )
        if not np.array_equal(effective_mask, canonical_effective):
            raise ValueError(
                f"fold {fold}: effective mask disagrees with canonical resolution"
            )
        resolved_delta = np.abs(resolved_ppm - canonical_resolved)
        resolved_disagreement = (
            canonical_effective
            & (~np.isfinite(resolved_delta) | (resolved_delta > 1e-3))
        )
        resolved_delta_max = float(
            np.max(resolved_delta[canonical_effective], initial=0.0)
        )
        if np.any(resolved_disagreement):
            if not allow_unsafe_prediction_pickle:
                raise ValueError(
                    f"fold {fold}: resolved targets disagree with canonical resolution "
                    f"by {resolved_delta_max:.6g} ppm"
                )
        resolved_ppm = canonical_resolved
        deltas = np.abs(true_ppm - np.asarray(shifts[pred_rows], dtype=np.float64))
        true_delta = float(deltas[dump_mask].max(initial=0.0))
        if true_delta > 1e-4:
            raise ValueError(
                f"fold {fold}: dump true_ppm differs from rereferenced cache by {true_delta}"
            )
        predicted_rows = len(pred_rows)
        dump_cells = int(dump_mask.sum())
        effective_cells = int(effective_mask.sum())
        model_valid_pred_rows = _model_valid_rows(fold_dir / "atom", pred_rows)
        if np.any(dump_mask & ~model_valid_pred_rows[:, None]):
            raise ValueError(
                f"fold {fold}: dump mask scores a model-invalid row"
            )
        prediction_present = np.zeros(len(raw_mask), dtype=bool)
        prediction_present[pred_rows] = True
        missing_cache_rows = int((~prediction_present).sum())
        atom_specific_in_scope = (
            raw_mask
            & matchable[:, None]
            & ~np.isin(ambiguity, AMBIGUITY_SET_CODES)
        ).any(axis=1)
        missing_scope = np.flatnonzero(
            ~prediction_present & atom_specific_in_scope
        )
        missing_in_scope_rows = len(missing_scope)
        if len(missing_scope):
            missing_scope_valid = _model_valid_rows(
                fold_dir / "atom", missing_scope)
            missing_model_valid_in_scope_rows = int(missing_scope_valid.sum())
            if missing_model_valid_in_scope_rows:
                raise ValueError(
                    f"fold {fold}: prediction dump omits "
                    f"{missing_model_valid_in_scope_rows} model-valid rows carrying "
                    "raw atom-specific labels"
                )

    return FoldData(
        fold=fold,
        pred=pred,
        dump_mask=dump_mask,
        dump_effective_mask=effective_mask,
        dump_resolved_ppm=resolved_ppm,
        pred_global_rows=pred_rows,
        raw_mask=raw_mask,
        structurally_matchable=matchable,
        shifts=shifts,
        ambiguity=ambiguity,
        residue_idx=residue_idx,
        label_residue_ids=label_ids,
        config=config,
        csv_rows=csv_rows,
        predicted_rows=predicted_rows,
        dump_mask_cells=dump_cells,
        dump_effective_mask_cells=effective_cells,
        true_ppm_max_abs_delta=true_delta,
        stored_resolved_vs_canonical_max_abs_delta=resolved_delta_max,
        model_valid_pred_rows=model_valid_pred_rows,
        prediction_missing_cache_rows=missing_cache_rows,
        prediction_missing_in_scope_rows=missing_in_scope_rows,
        prediction_missing_model_valid_in_scope_rows=(
            missing_model_valid_in_scope_rows
        ),
    )


def _accepted_deposited_cells(
    *,
    pred: np.ndarray,
    deposited: np.ndarray,
    resolved: np.ndarray,
    effective_mask: np.ndarray,
    residue_code: np.ndarray,
    ambiguity: np.ndarray,
    partner: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Map swap-resolved prediction acceptance back to deposited atom names.

    Evaluation assigns a deposited prochiral value to whichever model output gives
    the lower pair cost.  p100 intervals, however, are applied later by deposited
    atom name without a model prediction.  The acceptance decision therefore has
    to be inverse-permuted before its deposited value enters that name's interval.
    """
    pred = np.asarray(pred, dtype=np.float64)
    deposited = np.asarray(deposited, dtype=np.float64)
    resolved = np.asarray(resolved, dtype=np.float64)
    effective_mask = np.asarray(effective_mask, dtype=bool)
    ambiguity = np.asarray(ambiguity)
    residue_code = np.asarray(residue_code, dtype=np.int64)
    if (
        pred.shape != deposited.shape
        or resolved.shape != pred.shape
        or effective_mask.shape != pred.shape
        or ambiguity.shape != pred.shape
        or residue_code.shape != (len(pred),)
        or thresholds.shape != (pred.shape[1],)
    ):
        raise ValueError("accepted-span arrays have inconsistent shapes")
    if np.any(residue_code < 0) or np.any(residue_code >= len(partner)):
        raise ValueError("accepted-span residue code is outside partner authority")

    part = partner[residue_code]
    columns = np.arange(pred.shape[1])[None, :]
    safe = np.where(part >= 0, part, 0)
    partner_mask = np.take_along_axis(effective_mask, safe, axis=1)
    partner_ambiguity = np.take_along_axis(ambiguity, safe, axis=1)
    both = (part >= 0) & effective_mask & partner_mask
    swappable = (
        both
        & ~np.isin(ambiguity, AMBIGUITY_SET_CODES)
        & ~np.isin(partner_ambiguity, AMBIGUITY_SET_CODES)
        & ~((ambiguity == 1) & (partner_ambiguity == 1))
    )
    lead = swappable & (part > columns)
    partner_deposited = np.take_along_axis(deposited, safe, axis=1)
    partner_pred = np.take_along_axis(pred, safe, axis=1)
    direct_cost = np.abs(pred - deposited) + np.abs(
        partner_pred - partner_deposited
    )
    crossed_cost = np.abs(pred - partner_deposited) + np.abs(
        partner_pred - deposited
    )
    swap_rows, swap_columns = np.nonzero(lead & (crossed_cost < direct_cost))
    swap_partners = part[swap_rows, swap_columns]

    canonical = np.where(effective_mask, deposited, 0.0)
    canonical[swap_rows, swap_columns] = deposited[swap_rows, swap_partners]
    canonical[swap_rows, swap_partners] = deposited[swap_rows, swap_columns]
    disagreement = np.abs(canonical - resolved)
    if np.any(effective_mask & (disagreement > 1e-6)):
        maximum = float(np.max(disagreement[effective_mask], initial=0.0))
        raise ValueError(
            "accepted-span swap mapping disagrees with evaluator resolution "
            f"by {maximum:.6g} ppm"
        )

    accepted_prediction_channel = effective_mask & (
        np.abs(pred - resolved) <= thresholds[None, :]
    )
    accepted_deposited = accepted_prediction_channel.copy()
    accepted_deposited[swap_rows, swap_columns] = accepted_prediction_channel[
        swap_rows, swap_partners
    ]
    accepted_deposited[swap_rows, swap_partners] = accepted_prediction_channel[
        swap_rows, swap_columns
    ]
    return accepted_deposited, len(swap_rows)


def _accumulate_mae(
    total: np.ndarray,
    count: np.ndarray,
    errors: np.ndarray,
    mask: np.ndarray,
) -> None:
    total += np.where(mask, errors, 0.0).sum(axis=0, dtype=np.float64)
    count += mask.sum(axis=0, dtype=np.int64)


def _arg_ne_alias_seed_mask(
    data: FoldData,
    rows: np.ndarray,
    shift_cols: tuple[str, ...],
    residues: tuple[str, ...],
) -> np.ndarray:
    """Arg N-epsilon observations outside the paper's model-free interval."""
    mask = np.zeros((len(rows), len(shift_cols)), dtype=bool)
    arg_code = residues.index("ARG")
    ne_column = shift_cols.index("ne_shift")
    labels = np.asarray(data.shifts[rows, ne_column], dtype=np.float64)
    low, high = ARG_NE_OBSERVED_RANGE_PPM
    mask[:, ne_column] = (
        (np.asarray(data.residue_idx[rows], dtype=np.int64) == arg_code)
        & data.raw_mask[rows, ne_column]
        & ((labels < low) | (labels > high))
    )
    return mask


@dataclass
class Authorities:
    """The resolved roots plus the domain and evaluator modules every pass reads."""

    run_root: Path
    source_root: Path
    out_dir: Path
    shift_cols: tuple[str, ...]
    residues: tuple[str, ...]
    evaluation: Any
    partner: np.ndarray


@dataclass
class HeldoutErrors:
    """Pass 1 output: the per-nucleus error scale and the per-fold accounting."""

    baseline_mae: np.ndarray
    baseline_n: np.ndarray
    raw_mae: np.ndarray
    raw_n: np.ndarray
    fold_records: dict[str, dict]
    alias_seed_calibration_cells: int


@dataclass
class AcceptedSpans:
    """Pass 2 output: per residue/nucleus counts, the observed interval and the accepted
    deposited interval."""

    population_n: np.ndarray
    accepted_n: np.ndarray
    observed_lo: np.ndarray
    observed_hi: np.ndarray
    accepted_lo: np.ndarray
    accepted_hi: np.ndarray


@dataclass
class FoldHits:
    """One application fold's instance counters; the components must sum to ``hits``."""

    hits: int = 0
    range_hits: int = 0
    set_code_hits: int = 0
    impossible_hits: int = 0
    observed: int = 0


@dataclass
class Application:
    """Pass 3 output: p100 range flags plus out-of-scope schema accounting."""

    records: dict[str, dict] = field(default_factory=dict)
    required_pairs: set[str] = field(default_factory=set)
    flag_cells: set[tuple[str, str]] = field(default_factory=set)
    range_flag_cells: set[tuple[str, str]] = field(default_factory=set)
    by_nucleus: dict[str, int] = field(default_factory=dict)
    instance_hits: int = 0
    range_instance_hits: int = 0
    set_code_instance_hits: int = 0
    impossible_instance_hits: int = 0


def _resolve_authorities(args: argparse.Namespace) -> Authorities:
    """Open a fresh output directory and bind the domain and evaluator authorities."""
    source_root = args.source_root.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"--out-dir must be absent or empty so stale candidates cannot survive: "
            f"{out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    domain, _ = load_domain(source_root)
    shift_cols = tuple(domain.SHIFT_COLUMNS)
    residues = tuple(domain.STANDARD_RESIDUES)
    if len(shift_cols) != 64:
        raise ValueError(f"domain authority has {len(shift_cols)} shifts, expected 64")
    source_src = str(source_root / "src")
    if source_src not in sys.path:
        sys.path.insert(0, source_src)
    from deepshift import evaluate as evaluation
    from deepshift.loss import build_loss_tensors

    loaded_evaluator = Path(evaluation.__file__).resolve()
    if source_root not in loaded_evaluator.parents:
        raise RuntimeError(
            f"prediction resolver loaded from {loaded_evaluator}, "
            f"outside requested source root {source_root}"
        )
    _, partner_tensor = build_loss_tensors(shift_cols, residues)
    return Authorities(
        run_root=args.run_root.resolve(),
        source_root=source_root,
        out_dir=out_dir,
        shift_cols=shift_cols,
        residues=residues,
        evaluation=evaluation,
        partner=partner_tensor.numpy(),
    )


def _scored_mask(data: FoldData, rows: np.ndarray) -> np.ndarray:
    """Originally observed, matchable, model-valid, atom-specific cells of ``rows``."""
    return (
        data.raw_mask[rows]
        & data.structurally_matchable[rows, None]
        & data.model_valid_pred_rows[:, None]
        & ~np.isin(data.ambiguity[rows], AMBIGUITY_SET_CODES)
    )


def _calibration_fold_record(
    data: FoldData,
    run_root: Path,
    *,
    loaded_with_allow_pickle: bool,
    scored_raw_cells: int,
) -> dict:
    """Cell accounting for one calibration fold."""
    fold = data.fold
    csv_path = (
        run_root / "work" / "unified_folds" / f"structure_data_unified_fold_{fold}.csv"
    )
    pred_path = run_root / f"refold_fold{fold}.preds.npz"
    checkpoint_path = run_root / f"refold_fold{fold}.final.pt"
    raw_matchable_cells = int(
        (data.raw_mask & data.structurally_matchable[:, None]).sum()
    )
    raw_atom_specific_cells = int(
        (
            data.raw_mask
            & data.structurally_matchable[:, None]
            & ~np.isin(data.ambiguity, AMBIGUITY_SET_CODES)
        ).sum()
    )
    return {
        "csv": {"name": csv_path.name},
        "checkpoint": {"name": checkpoint_path.name},
        "predictions": {
            "name": pred_path.name,
            "loaded_with_allow_pickle": bool(loaded_with_allow_pickle),
        },
        "cache_rows": int(data.csv_rows),
        "predicted_rows": int(data.predicted_rows),
        "rows_without_predictions": int(data.csv_rows - data.predicted_rows),
        "raw_matchable_observed_cells": raw_matchable_cells,
        "raw_matchable_atom_specific_observed_cells": raw_atom_specific_cells,
        "raw_atom_specific_observed_cells_with_valid_predictions": scored_raw_cells,
        "heldout_atom_specific_cell_fraction_of_cache": (
            scored_raw_cells / raw_atom_specific_cells
            if raw_atom_specific_cells
            else None
        ),
        "dump_mask_cells": data.dump_mask_cells,
        "dump_effective_mask_cells": data.dump_effective_mask_cells,
        "model_valid_predicted_rows": int(data.model_valid_pred_rows.sum()),
        "prediction_missing_cache_rows": data.prediction_missing_cache_rows,
        "prediction_missing_in_scope_rows": data.prediction_missing_in_scope_rows,
        "prediction_missing_model_valid_in_scope_rows": (
            data.prediction_missing_model_valid_in_scope_rows
        ),
        "atom_specific_ambiguity_codes_excluded": list(AMBIGUITY_SET_CODES),
        "true_ppm_vs_cache_max_abs_delta": data.true_ppm_max_abs_delta,
        "stored_resolved_vs_canonical_max_abs_delta": (
            data.stored_resolved_vs_canonical_max_abs_delta
        ),
        "resolution_binding": (
            "canonical_recomputed_from_pred_ppm_raw_labels_and_ambiguity"
        ),
    }


def measure_heldout_mae(
    authorities: Authorities, args: argparse.Namespace
) -> HeldoutErrors:
    """Pass 1: the per-nucleus held-out MAE that sets every acceptance threshold."""
    shift_cols = authorities.shift_cols
    baseline_sum = np.zeros(len(shift_cols), dtype=np.float64)
    baseline_n = np.zeros(len(shift_cols), dtype=np.int64)
    raw_sum = np.zeros(len(shift_cols), dtype=np.float64)
    raw_n = np.zeros(len(shift_cols), dtype=np.int64)
    fold_records: dict[str, dict] = {}
    alias_seed_calibration_cells = 0

    for fold in CALIBRATION_FOLDS:
        print(f"[pass 1/3] fold {fold}: validating and measuring held-out errors", flush=True)
        data = load_fold(
            fold=fold,
            run_root=authorities.run_root,
            shift_cols=shift_cols,
            residues=authorities.residues,
            with_predictions=True,
            allow_unsafe_prediction_pickle=(
                args.allow_unsafe_legacy_prediction_pickle
            ),
            prediction_resolver=authorities.evaluation.resolve,
        )
        assert data.pred is not None
        assert data.dump_mask is not None
        assert data.dump_effective_mask is not None
        assert data.dump_resolved_ppm is not None
        assert data.pred_global_rows is not None
        assert data.model_valid_pred_rows is not None
        rows = data.pred_global_rows
        labels = np.asarray(data.shifts[rows], dtype=np.float64)
        errors = np.abs(data.pred - data.dump_resolved_ppm)
        raw_scored = _scored_mask(data, rows)
        alias_seed = _arg_ne_alias_seed_mask(
            data, rows, shift_cols, authorities.residues
        )
        alias_seed &= raw_scored
        alias_seed_calibration_cells += int(alias_seed.sum())
        _accumulate_mae(
            baseline_sum,
            baseline_n,
            errors,
            data.dump_effective_mask & ~alias_seed,
        )
        raw_scored &= ~alias_seed
        raw_resolved, raw_effective = authorities.evaluation.resolve(
            data.pred,
            labels,
            raw_scored,
            np.asarray(data.ambiguity[rows]),
            np.asarray(data.residue_idx[rows], dtype=np.int64),
            shift_cols,
        )
        if not np.array_equal(raw_effective, raw_scored):
            raise ValueError(
                f"fold {fold}: atom-specific raw support changed during resolution"
            )
        _accumulate_mae(
            raw_sum,
            raw_n,
            np.abs(data.pred - raw_resolved),
            raw_effective,
        )

        fold_records[str(fold)] = _calibration_fold_record(
            data,
            authorities.run_root,
            loaded_with_allow_pickle=args.allow_unsafe_legacy_prediction_pickle,
            scored_raw_cells=int(raw_scored.sum()),
        )
        del data, labels, errors, raw_scored, raw_resolved, raw_effective, alias_seed

    if np.any(baseline_n == 0):
        missing = [shift_cols[i] for i in np.flatnonzero(baseline_n == 0)]
        raise ValueError(f"no held-out evaluation support for nuclei: {missing}")
    return HeldoutErrors(
        baseline_mae=baseline_sum / baseline_n,
        baseline_n=baseline_n,
        raw_mae=np.divide(
            raw_sum, raw_n, out=np.full_like(raw_sum, np.nan), where=raw_n > 0
        ),
        raw_n=raw_n,
        fold_records=fold_records,
        alias_seed_calibration_cells=alias_seed_calibration_cells,
    )


def _accumulate_column_spans(
    spans: AcceptedSpans,
    column_index: int,
    *,
    aa: np.ndarray,
    labels: np.ndarray,
    observed: np.ndarray,
    accepted: np.ndarray,
) -> None:
    """Fold one nucleus column of one calibration fold into the running spans."""
    observed_rows = np.flatnonzero(observed)
    if len(observed_rows):
        np.add.at(spans.population_n[:, column_index], aa[observed_rows], 1)
        observed_codes = aa[observed_rows]
        observed_values = labels[observed_rows, column_index]
        for code in np.unique(observed_codes):
            values = observed_values[observed_codes == code]
            spans.observed_lo[code, column_index] = min(
                spans.observed_lo[code, column_index], float(values.min())
            )
            spans.observed_hi[code, column_index] = max(
                spans.observed_hi[code, column_index], float(values.max())
            )
    accepted_rows = np.flatnonzero(accepted)
    if not len(accepted_rows):
        return
    accepted_codes = aa[accepted_rows]
    accepted_values = labels[accepted_rows, column_index]
    np.add.at(spans.accepted_n[:, column_index], accepted_codes, 1)
    for code in np.unique(accepted_codes):
        values = accepted_values[accepted_codes == code]
        spans.accepted_lo[code, column_index] = min(
            spans.accepted_lo[code, column_index], float(values.min())
        )
        spans.accepted_hi[code, column_index] = max(
            spans.accepted_hi[code, column_index], float(values.max())
        )


def fit_accepted_spans(
    authorities: Authorities, args: argparse.Namespace, heldout: HeldoutErrors
) -> AcceptedSpans:
    """Pass 2: the span of deposited labels the model accepts within k * MAE."""
    shift_cols = authorities.shift_cols
    shape = (len(authorities.residues), len(shift_cols))
    spans = AcceptedSpans(
        population_n=np.zeros(shape, dtype=np.int64),
        accepted_n=np.zeros(shape, dtype=np.int64),
        observed_lo=np.full(shape, np.inf, dtype=np.float64),
        observed_hi=np.full(shape, -np.inf, dtype=np.float64),
        accepted_lo=np.full(shape, np.inf, dtype=np.float64),
        accepted_hi=np.full(shape, -np.inf, dtype=np.float64),
    )
    for fold in CALIBRATION_FOLDS:
        print(f"[pass 2/3] fold {fold}: fitting accepted spans", flush=True)
        data = load_fold(
            fold=fold,
            run_root=authorities.run_root,
            shift_cols=shift_cols,
            residues=authorities.residues,
            with_predictions=True,
            allow_unsafe_prediction_pickle=(
                args.allow_unsafe_legacy_prediction_pickle
            ),
            prediction_resolver=authorities.evaluation.resolve,
        )
        assert data.pred is not None
        assert data.dump_mask is not None
        assert data.dump_effective_mask is not None
        assert data.pred_global_rows is not None
        assert data.model_valid_pred_rows is not None
        rows = data.pred_global_rows
        aa = np.asarray(data.residue_idx[rows], dtype=np.int64)
        labels = np.asarray(data.shifts[rows], dtype=np.float64)
        raw_scored = _scored_mask(data, rows)
        raw_scored &= ~_arg_ne_alias_seed_mask(
            data, rows, shift_cols, authorities.residues
        )
        resolved, effective = authorities.evaluation.resolve(
            data.pred,
            labels,
            raw_scored,
            np.asarray(data.ambiguity[rows]),
            aa,
            shift_cols,
        )
        if not np.array_equal(effective, raw_scored):
            raise ValueError(
                f"fold {fold}: atom-specific raw support changed during resolution"
            )
        accepted, swapped_pairs = _accepted_deposited_cells(
            pred=data.pred,
            deposited=labels,
            resolved=resolved,
            effective_mask=effective,
            residue_code=aa,
            ambiguity=np.asarray(data.ambiguity[rows]),
            partner=authorities.partner,
            thresholds=args.k * heldout.baseline_mae,
        )
        heldout.fold_records[str(fold)]["accepted_span_swap_resolved_pairs"] = int(
            swapped_pairs
        )
        for column_index in range(len(shift_cols)):
            _accumulate_column_spans(
                spans,
                column_index,
                aa=aa,
                labels=labels,
                observed=raw_scored[:, column_index],
                accepted=accepted[:, column_index],
            )
        del data, aa, labels, raw_scored, resolved, effective, accepted
    return spans


def _nucleus_metric_records(
    shift_cols: tuple[str, ...], heldout: HeldoutErrors, k: float
) -> dict[str, dict]:
    """Per-nucleus held-out MAE and the ppm threshold it sets."""
    return {
        column: {
            "heldout_mae_on_effective_evaluation_mask": float(heldout.baseline_mae[i]),
            "heldout_effective_evaluation_n": int(heldout.baseline_n[i]),
            "heldout_mae_on_all_raw_atom_specific_scored_labels": (
                float(heldout.raw_mae[i]) if np.isfinite(heldout.raw_mae[i]) else None
            ),
            "heldout_all_raw_atom_specific_scored_n": int(heldout.raw_n[i]),
            "threshold_ppm": float(k * heldout.baseline_mae[i]),
        }
        for i, column in enumerate(shift_cols)
    }


def build_cutoffs(
    spans: AcceptedSpans,
    residues: tuple[str, ...],
    shift_cols: tuple[str, ...],
) -> tuple[dict[str, dict], list[str], dict[str, dict]]:
    """One interval per observed residue/nucleus: the span of the accepted labels, or the
    observed-value interval for Arg Nε. A population with no accepted label is returned
    as missing and no range is written. The third value lists each fitted span's width
    beside the width of its observed population."""
    cutoff_entries: dict[str, dict] = {}
    missing_accepted_pairs: list[str] = []
    span_widths: dict[str, dict] = {}
    for residue_index, residue in enumerate(residues):
        for column_index, column in enumerate(shift_cols):
            pair = f"{residue}:{column}"
            if pair in IMPOSSIBLE_PAIRS:
                continue
            n = int(spans.population_n[residue_index, column_index])
            if not n:
                continue
            accepted_count = int(spans.accepted_n[residue_index, column_index])
            if not accepted_count:
                missing_accepted_pairs.append(pair)
                continue
            fitted_lo = float(spans.accepted_lo[residue_index, column_index])
            fitted_hi = float(spans.accepted_hi[residue_index, column_index])
            observed_lo = float(spans.observed_lo[residue_index, column_index])
            observed_hi = float(spans.observed_hi[residue_index, column_index])
            if residue == "ARG" and column == "ne_shift":
                low, high = ARG_NE_OBSERVED_RANGE_PPM
            else:
                low, high = fitted_lo, fitted_hi
            cutoff_entries[pair] = {
                "lo": float(low),
                "hi": float(high),
                "n": n,
                "accepted_n": accepted_count,
            }
            span_widths[pair] = {
                "fitted_lo_ppm": fitted_lo,
                "fitted_hi_ppm": fitted_hi,
                "fitted_span_ppm": fitted_hi - fitted_lo,
                "observed_lo_ppm": observed_lo,
                "observed_hi_ppm": observed_hi,
                "observed_span_ppm": observed_hi - observed_lo,
                "n": n,
                "accepted_n": accepted_count,
            }
    return cutoff_entries, missing_accepted_pairs, span_widths


def _application_fold_record(
    data: FoldData,
    hits: FoldHits,
    unmatchable_observed: int,
) -> dict:
    """One fold's p100 hits and separately counted out-of-scope cells."""
    return {
        "rows": int(data.csv_rows),
        "matchable_observed_cells": int(hits.observed),
        "already_structurally_unmatchable_observed_cells": unmatchable_observed,
        "p100_instance_hits": int(hits.hits),
        "accepted_span_range_instance_hits": int(hits.range_hits),
        "unscorable_ambiguity_set_code_instance_hits": int(hits.set_code_hits),
        "chemically_impossible_instance_hits": int(hits.impossible_hits),
    }


def _flag_fold_column(
    application: Application,
    hits: FoldHits,
    *,
    data: FoldData,
    aa: np.ndarray,
    residues: tuple[str, ...],
    column_index: int,
    column: str,
    cutoff_entries: dict[str, dict],
) -> None:
    """Test one nucleus column of one fold against each residue's calibrated range."""
    observed = data.raw_mask[:, column_index] & data.structurally_matchable
    rows = np.flatnonzero(observed)
    hits.observed += len(rows)
    if not len(rows):
        return
    for code in np.unique(aa[rows]):
        pair = f"{residues[int(code)]}:{column}"
        application.required_pairs.add(pair)
        selected = rows[aa[rows] == code]
        if pair in IMPOSSIBLE_PAIRS:
            # Schema-invalid, not an extreme value of a valid p100 population.
            hits.impossible_hits += len(selected)
            continue
        bounds = cutoff_entries.get(pair)
        if bounds is None:
            continue
        values = np.asarray(data.shifts[selected, column_index], dtype=np.float64)
        set_code = np.isin(
            np.asarray(data.ambiguity[selected, column_index]),
            AMBIGUITY_SET_CODES,
        )
        range_bad = (
            ~set_code
            & (
                (values < bounds["lo"])
                | (values > bounds["hi"])
            )
        )
        bad_rows = selected[range_bad]
        hits.set_code_hits += int(set_code.sum())
        hits.range_hits += int(range_bad.sum())
        hits.hits += len(bad_rows)
        application.by_nucleus[column] = (
            application.by_nucleus.get(column, 0) + len(bad_rows)
        )
        application.flag_cells.update(
            (str(data.label_residue_ids[int(row)]), column) for row in bad_rows
        )
        application.range_flag_cells.update(
            (str(data.label_residue_ids[int(row)]), column)
            for row in selected[range_bad]
        )


def apply_flags(
    authorities: Authorities,
    cutoff_entries: dict[str, dict],
) -> Application:
    """Pass 3: apply the fitted ranges to every originally observed matchable cell."""
    application = Application()
    for fold in APPLICATION_FOLDS:
        print(f"[pass 3/3] fold {fold}: applying ranges to original observations", flush=True)
        data = load_fold(
            fold=fold,
            run_root=authorities.run_root,
            shift_cols=authorities.shift_cols,
            residues=authorities.residues,
            with_predictions=False,
        )
        hits = FoldHits()
        unmatchable_observed = int(
            (data.raw_mask & ~data.structurally_matchable[:, None]).sum()
        )
        aa = np.asarray(data.residue_idx, dtype=np.int64)
        for column_index, column in enumerate(authorities.shift_cols):
            _flag_fold_column(
                application,
                hits,
                data=data,
                aa=aa,
                residues=authorities.residues,
                column_index=column_index,
                column=column,
                cutoff_entries=cutoff_entries,
            )
        application.instance_hits += hits.hits
        application.range_instance_hits += hits.range_hits
        application.set_code_instance_hits += hits.set_code_hits
        application.impossible_instance_hits += hits.impossible_hits
        application.records[str(fold)] = _application_fold_record(
            data, hits, unmatchable_observed
        )
        del data
    return application


def _check_flag_partition(application: Application) -> None:
    """Only observed-value range failures may enter the p100 flag artifact."""
    if (
        application.instance_hits != application.range_instance_hits
        or application.flag_cells != application.range_flag_cells
    ):
        raise ValueError("non-p100 schema or ambiguity cells entered the p100 flags")


def scientific_sanity(
    cutoff_entries: dict[str, dict], nucleus_metrics: dict[str, dict]
) -> list[dict]:
    """Promotion gate: every fitted span is finite and non-empty, and Arg Nε equals the
    observed-value interval."""
    failures = []
    for pair, entry in sorted(cutoff_entries.items()):
        low, high = float(entry["lo"]), float(entry["hi"])
        if not (np.isfinite(low) and np.isfinite(high) and low <= high):
            failures.append({
                "gate": "fitted_span",
                "pair": pair,
                "status": "fail",
                "reason": "empty or non-finite accepted span",
                "fitted_lo_ppm": low,
                "fitted_hi_ppm": high,
            })
    arg_ne = cutoff_entries.get("ARG:ne_shift")
    if arg_ne is None:
        failures.append({
            "gate": "arg_ne_observed_value_exception",
            "status": "fail",
            "reason": "ARG:ne_shift has no calibrated range",
        })
    elif (float(arg_ne["lo"]), float(arg_ne["hi"])) != ARG_NE_OBSERVED_RANGE_PPM:
        failures.append({
            "gate": "arg_ne_observed_value_exception",
            "status": "fail",
            "reason": "Arg NE range differs from the paper's observed-value interval",
            "required_range_ppm": list(ARG_NE_OBSERVED_RANGE_PPM),
            "fitted_lo_ppm": float(arg_ne["lo"]),
            "fitted_hi_ppm": float(arg_ne["hi"]),
            "heldout_mae_ppm": float(
                nucleus_metrics["ne_shift"]["heldout_mae_on_effective_evaluation_mask"]
            ),
            "threshold_ppm": float(nucleus_metrics["ne_shift"]["threshold_ppm"]),
        })
    return failures


def derive_structural(
    authorities: Authorities, flags: dict[str, list[str]], flags_path: Path
) -> tuple[dict, dict]:
    """Structural flags over all seven folds, and their union with the p100 flags."""
    loaded_from = Path(flag_rules.__file__).resolve()
    if authorities.source_root not in loaded_from.parents:
        raise RuntimeError(
            f"structural flag implementation loaded from {loaded_from}, "
            f"outside requested source root {authorities.source_root}"
        )

    structural, structural_report = flag_rules.derive_structural_flags(
        authorities.run_root / "work" / "unified_cache"
    )
    expected_fold_names = [f"fold_{fold}" for fold in APPLICATION_FOLDS]
    if structural_report.get("folds_read") != expected_fold_names:
        raise ValueError(
            "structural flag derivation did not cover folds 0--6 exactly: "
            f"{structural_report.get('folds_read')}"
        )
    out_dir = authorities.out_dir
    structural_path = out_dir / "qc_flags_structural_all7.json"
    structural_report_path = (
        out_dir / "qc_flags_structural_all7.report.json"
    )
    flag_rules.write_flags(structural_path, structural)
    write_json_atomic(structural_report_path, structural_report)
    structural_record = {
        "path": str(structural_path),
        "report_path": str(structural_report_path),
        "label_residues": len(structural),
        "unique_cells": sum(len(columns) for columns in structural.values()),
        "observation_authority": (
            "unfiltered cache observation mask; schema-invalid and structurally "
            "unmatchable cells are absent"
        ),
    }

    composite = flag_rules.union_flags(flags, structural)
    # Written by the one flag-file writer, so the build's re-derivation of the same
    # cells produces the same bytes.
    composite_path = out_dir / "qc_flags_p100_structural_oof64_all7.json"
    flag_rules.write_flags(composite_path, composite)
    composite_record = {
        "path": str(composite_path),
        "label_residues": len(composite),
        "unique_cells": sum(len(columns) for columns in composite.values()),
        "promotable": True,
    }
    return structural_record, composite_record


def _fit_report_record(
    args: argparse.Namespace,
    *,
    promotable: bool,
    sanity_failures: list[dict],
    cutoff_path: Path,
    cutoff_entries: dict[str, dict],
    flags: dict[str, list[str]] | None,
    flags_path: Path,
    application: Application,
    heldout: HeldoutErrors,
    nucleus_metrics: dict[str, dict],
    missing_accepted_pairs: list[str],
    missing_required: list[str],
    span_widths: dict[str, dict],
    structural_record: dict | None,
    composite_record: dict | None,
) -> dict:
    """The fit report: status, artifacts, and the cell accounting."""
    return {
        "status": (
            "complete"
            if promotable
            else (
                "candidate_failed_scientific_sanity"
                if sanity_failures
                else "incomplete_missing_calibrated_population"
            )
        ),
        "cutoffs": {
            "path": str(cutoff_path),
            "range_count": len(cutoff_entries),
        },
        "flags": (
            {
                "path": str(flags_path),
                "label_residues": len(flags),
                "unique_cells": sum(len(columns) for columns in flags.values()),
                "instance_hits": int(application.instance_hits),
                "components": {
                    "accepted_span_range": {
                        "rule": f"outside deposited-name {args.k:g}*MAE accepted span",
                        "label_residues": len(
                            {owner for owner, _ in application.range_flag_cells}
                        ),
                        "unique_cells": len(application.range_flag_cells),
                        "instance_hits": int(application.range_instance_hits),
                    },
                },
                "out_of_scope_not_qc": {
                    "ambiguity_set_codes": list(AMBIGUITY_SET_CODES),
                    "ambiguity_instances": int(application.set_code_instance_hits),
                    "chemically_impossible_instances": int(
                        application.impossible_instance_hits
                    ),
                },
                "instance_hits_by_nucleus": dict(
                    sorted(application.by_nucleus.items())
                ),
                "promotable": promotable,
            }
            if flags is not None
            else None
        ),
        "structural_flags": structural_record,
        "composite_flags": composite_record,
        "calibration": {
            "folds": list(CALIBRATION_FOLDS),
            "fold_records": heldout.fold_records,
            "nucleus_metrics": nucleus_metrics,
            "range_population_count": len(cutoff_entries),
            "populations_with_observations_but_no_accepted_label": sorted(
                missing_accepted_pairs
            ),
            "span_widths_ppm": span_widths,
        },
        "application": {
            "folds": list(APPLICATION_FOLDS),
            "fold_records": application.records,
            "required_population_count": len(application.required_pairs),
            "required_populations_without_cutoff": missing_required,
        },
        "scientific_sanity": {
            "status": "pass" if not sanity_failures else "fail",
            "failures": sanity_failures,
        },
        "policy_notes": {
            "input_checkpoint_mask_does_not_define_original_observation_support": True,
            "calibration_checkpoints_are_required_to_be_unfiltered": True,
            "baseline_uses_effective_evaluation_mask": True,
            "baseline_uses_swap_resolved_targets": True,
            "accepted_spans_exclude_non_atom_specific_ambiguity_codes": list(
                AMBIGUITY_SET_CODES
            ),
            "accepted_spans_inverse_map_swap_acceptance_to_deposited_names": True,
            "application_excludes_ambiguity_sets_from_p100_without_flagging": list(
                AMBIGUITY_SET_CODES
            ),
            "chemically_impossible_pairs_are_schema_not_qc": True,
            "folds_0_and_6_are_not_used_to_fit_ranges": True,
            "structurally_unmatchable_rows_are_out_of_scope_for_p100": True,
            "legacy_prediction_npz_loaded_with_pickle": bool(
                args.allow_unsafe_legacy_prediction_pickle
            ),
            "legacy_stored_resolved_targets_are_recomputed_canonically": True,
            "arg_ne_observed_value_exception": list(ARG_NE_OBSERVED_RANGE_PPM),
        },
    }


def fit(args: argparse.Namespace) -> tuple[dict, dict, dict | None]:
    authorities = _resolve_authorities(args)
    out_dir = authorities.out_dir
    heldout = measure_heldout_mae(authorities, args)
    spans = fit_accepted_spans(authorities, args, heldout)
    nucleus_metrics = _nucleus_metric_records(
        authorities.shift_cols, heldout, args.k
    )
    cutoff_entries, missing_accepted_pairs, span_widths = build_cutoffs(
        spans, authorities.residues, authorities.shift_cols
    )

    cutoffs = dict(cutoff_entries)
    cutoff_path = out_dir / "p100_oof64_cutoffs.json"
    write_json_atomic(cutoff_path, cutoffs)

    application = apply_flags(authorities, cutoff_entries)
    _check_flag_partition(application)

    # An impossible pair has no cutoff by construction, so it must not read as a gap.
    missing_required = sorted(
        application.required_pairs - cutoff_entries.keys() - IMPOSSIBLE_PAIRS
    )
    sanity_failures = scientific_sanity(cutoff_entries, nucleus_metrics)
    promotable = (
        not missing_required
        and not missing_accepted_pairs
        and not sanity_failures
    )
    flags_path = out_dir / (
        "qc_flags_p100_oof64_all7.json"
        if promotable
        else "qc_flags_p100_oof64_all7.candidate-failed-sanity.json"
    )
    flags: dict[str, list[str]] | None = None
    if not missing_required and not missing_accepted_pairs:
        flags = flag_rules.flags_by_label_residue(application.flag_cells)
        flag_rules.write_flags(flags_path, flags)

    structural_record = None
    composite_record = None
    if promotable and flags is not None:
        structural_record, composite_record = derive_structural(
            authorities, flags, flags_path
        )

    report = _fit_report_record(
        args,
        promotable=promotable,
        sanity_failures=sanity_failures,
        cutoff_path=cutoff_path,
        cutoff_entries=cutoff_entries,
        flags=flags,
        flags_path=flags_path,
        application=application,
        heldout=heldout,
        nucleus_metrics=nucleus_metrics,
        missing_accepted_pairs=missing_accepted_pairs,
        missing_required=missing_required,
        span_widths=span_widths,
        structural_record=structural_record,
        composite_record=composite_record,
    )
    report_path = out_dir / "p100_oof64_fit_report.json"
    write_json_atomic(report_path, report)
    return cutoffs, report, flags if promotable else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="calibration run root containing checkpoints, predictions, and work caches",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="DeepShift source checkout (default: repository containing this script)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="fresh output directory for cutoffs, flags, and fit report",
    )
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument(
        "--allow-unsafe-legacy-prediction-pickle",
        action="store_true",
        help=(
            "explicitly trust prediction NPZ object arrays; release dumps "
            "should first be converted to Unicode and must not need this option"
        ),
    )
    args = parser.parse_args()
    if not np.isfinite(args.k) or args.k <= 0:
        parser.error("--k must be finite and positive")
    return args


def main() -> int:
    args = parse_args()
    _, report, flags = fit(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "range_count": report["cutoffs"]["range_count"],
                "flags": report["flags"],
                "composite_flags": report["composite_flags"],
                "missing_required": report["application"][
                    "required_populations_without_cutoff"
                ],
            },
            indent=2,
        )
    )
    complete = (report.get("status") == "complete" and flags is not None
                and report.get("composite_flags") is not None)
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
