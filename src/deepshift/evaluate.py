"""Canonical scoring of held-out chemical-shift predictions.

Every number the paper reports comes through here, under these rules:

  * Error is per nucleus, RMSE first with MAE beside it. Nothing averages ppm across
    nuclei -- HA and N differ by an order of magnitude, so their mean means nothing.
  * Targets are the prochiral-swap-resolved ones; the mask is the effective one.
  * r and amplitude are on the secondary shift: raw-ppm r within one nucleus is dominated
    by residue identity and reads ~1.0 for a model that learned nothing.
  * Uncertainty resamples label sets (or BMRB depositions) rather than residues, which
    are not independent within a protein.
  * Everything is ppm. Predictions arrive denormalized by their own checkpoint's stats,
    so no z reaches this module -- two CV members carry different per-nucleus scales.

Two entry points produce the same ``Predictions``: ``read_predictions`` from a saved
dump (no GPU, no cache) and ``score_checkpoints`` from a cache fold plus checkpoints.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .cohorts import by_source as score_by_source
from .cohorts import matched_label_sets
from .domain import STANDARD_RESIDUES
from .loss import build_loss_tensors
from .normalization import BACKBONE_SHIFTS
from .statistics import (
    DEFAULT_SEED,
    LABEL_TOLERANCE_PPM,
    MIN_SCORABLE_CELLS,
    BootstrapSpec,
    column_index,
)
from .statistics import (
    compare as compare_predictions,
)
from .statistics import (
    score as score_predictions,
)

# The swap resolver comes from the trainer, so scoring and training resolve alike.
from .training import _ambiguity_eval_targets

DEFAULT_BOOTSTRAP = 1000

_CELL_FIELDS = ("pred_ppm", "true_ppm", "resolved_ppm", "mask", "effective_mask", "ambiguity")
_ROW_FIELDS = ("residue_code", "row_key", "label_set", "bmrb", "source")


@dataclass(frozen=True)
class Predictions:
    """One scored population: ``rows x nuclei``, all in ppm.

    ``mask`` and ``effective_mask`` are bool by dtype, checked on construction: stored
    masks are often uint8, and numpy reads a uint8 array as row numbers, not as a
    selection, producing absurdly good metrics instead of an error.
    """
    shift_cols: tuple[str, ...]
    pred_ppm: np.ndarray
    true_ppm: np.ndarray            # deposited labels, prochiral swaps unresolved
    resolved_ppm: np.ndarray        # swap-resolved targets: the only truth score() reads
    mask: np.ndarray                # observed cells
    effective_mask: np.ndarray      # observed AND scorable (ambiguity set codes dropped)
    ambiguity: np.ndarray
    residue_code: np.ndarray
    row_key: np.ndarray             # identity of one (structure, label residue) row
    label_set: np.ndarray
    bmrb: np.ndarray
    source: np.ndarray

    def __post_init__(self):
        rows, cols = self.pred_ppm.shape
        if len(self.shift_cols) != cols:
            raise ValueError(f"{len(self.shift_cols)} column names for {cols} columns")
        # A repeated name resolves to its first column everywhere: the duplicate is
        # scored twice under one heading and its own column never at all.
        if len(set(self.shift_cols)) != cols:
            raise ValueError("the nucleus vocabulary repeats a name")
        for name in _CELL_FIELDS:
            if getattr(self, name).shape != (rows, cols):
                raise ValueError(f"{name} is not shaped like pred_ppm")
        for name in _ROW_FIELDS:
            if len(getattr(self, name)) != rows:
                raise ValueError(f"{name} does not carry one entry per row")
        if self.mask.dtype != bool or self.effective_mask.dtype != bool:
            raise TypeError("masks must be bool; cast with .astype(bool) at the boundary")
        if not np.all(np.isfinite(self.pred_ppm)):
            raise ValueError("pred_ppm contains non-finite values")
        if not np.all(np.isfinite(self.true_ppm[self.mask])):
            raise ValueError("observed true_ppm contains non-finite values")
        if not np.all(np.isfinite(self.resolved_ppm[self.effective_mask])):
            raise ValueError("scored resolved_ppm contains non-finite values")

    @property
    def n_rows(self):
        return self.pred_ppm.shape[0]


def _bool(array):
    """The single cast that keeps a uint8 mask from being read as an index array."""
    return np.asarray(array).astype(bool)


def _index(preds, name):
    return column_index(preds, name)


def _take(preds, rows, columns):
    """A Predictions restricted to ``rows`` (indices) and ``columns`` (names, in order)."""
    take = np.asarray([_index(preds, c) for c in columns], dtype=np.int64)
    rows = np.asarray(rows)
    cells = {name: getattr(preds, name)[np.ix_(rows, take)] for name in _CELL_FIELDS}
    per_row = {name: getattr(preds, name)[rows] for name in _ROW_FIELDS}
    return Predictions(shift_cols=tuple(columns), **cells, **per_row)


def resolve(pred_ppm, true_ppm, mask, ambiguity, residue_code, shift_cols, *, partner=None):
    """Swap-resolve prochiral pairs against the prediction actually scored; returns
    ``(resolved_ppm, effective_mask)``.

    Resolution relabels a both-observed pair, so it runs in ppm (center 0, scale 1), not
    in any model's z: an ensemble has no single z space, and per-member resolution would
    score the members under different targets. ``partner`` is the per-residue partner
    table from ``build_loss_tensors``; a caller scoring many rows one at a time builds
    it once and passes it in.
    """
    if partner is None:
        _, partner = build_loss_tensors(tuple(shift_cols), tuple(STANDARD_RESIDUES))
        partner = partner.numpy()
    flat = np.zeros((len(STANDARD_RESIDUES), partner.shape[1]))
    mask = _bool(mask)
    resolved, effective = _ambiguity_eval_targets(
        np.asarray(pred_ppm, dtype=np.float64),
        np.where(mask, np.asarray(true_ppm, dtype=np.float64), 0.0),
        mask, np.asarray(residue_code), np.asarray(ambiguity), partner, flat, flat + 1.0)
    return resolved, _bool(effective)


def _unique_row_keys(keys):
    """Row identities must be unique, or ``align`` pairs the wrong residues and ``pool``
    cannot tell a second fold from a second ensemble member."""
    if len(np.unique(keys)) != len(keys):
        raise ValueError("the scored rows are not uniquely identified")
    return keys


def _row_keys(materialization, label_residue, source):
    """Identity of one scored row. The same label residue is realized once per structure
    source, so the materialization (or the source) is part of the key; without it a
    pooled AF+NMR+crystal dump looks like it holds duplicate rows."""
    if (label_residue == "").any():
        raise ValueError("dump carries no label_residue_id, so its rows cannot be identified")
    stem = np.where(materialization != "", materialization, np.char.add("src", source.astype(str)))
    return _unique_row_keys(np.char.add(np.char.add(stem, "|"), label_residue))


def read_predictions(path, *, allow_unsafe_pickle=False) -> Predictions:
    """Load one saved ``*.preds.npz``. No GPU, no cache, no checkpoint.

    A dump missing the resolved targets or the effective mask is refused, never fallen
    back to ``true_ppm``/``mask``. A dump storing strings as pickled object arrays is
    rejected by default, since loading an untrusted pickle can execute code;
    ``allow_unsafe_pickle=True`` is only for converting a dump from a trusted source.
    """
    try:
        with np.load(path, allow_pickle=allow_unsafe_pickle) as dump:
            have = set(dump.files)
            required = {
                "pred_ppm", "true_ppm", "resolved_true_ppm", "mask",
                "effective_mask", "ambiguity", "residue_code", "shift_cols",
            }
            if missing := sorted(required - have):
                extra = (
                    "; it stores z only -- denormalize with the checkpoint's own stats first"
                    if "pred_z" in have and "pred_ppm" not in have else ""
                )
                raise ValueError(
                    f"{path} is not a scorable prediction dump: missing {missing}{extra}"
                )
            def text(name):
                return (
                    np.asarray([str(x) for x in dump[name]])
                    if name in have
                    else np.full(len(dump["residue_code"]), "", dtype="<U1")
                )
            cols = tuple(str(c) for c in dump["shift_cols"])
            pred_ppm = np.asarray(dump["pred_ppm"], dtype=np.float64)
            true_ppm = np.asarray(dump["true_ppm"], dtype=np.float64)
            stored_resolved = np.asarray(dump["resolved_true_ppm"], dtype=np.float64)
            mask, effective = _bool(dump["mask"]), _bool(dump["effective_mask"])
            ambiguity = np.asarray(dump["ambiguity"])
            residue_code = np.asarray(dump["residue_code"])
            canonical_resolved, canonical_effective = resolve(
                pred_ppm, true_ppm, mask, ambiguity, residue_code, cols
            )
            if not np.array_equal(effective, canonical_effective):
                raise ValueError(
                    f"{path}: effective_mask is not the canonical ambiguity-resolved mask"
                )
            delta = np.abs(stored_resolved - canonical_resolved)
            if np.any(
                canonical_effective
                & (~np.isfinite(delta) | (delta > LABEL_TOLERANCE_PPM))
            ):
                maximum = float(
                    np.max(delta[canonical_effective], initial=0.0)
                )
                raise ValueError(
                    f"{path}: resolved_true_ppm disagrees with canonical recomputation "
                    f"(max {maximum:.6g} ppm)"
                )
            row_key = (
                _unique_row_keys(text("row_key"))
                if "row_key" in have
                else _row_keys(
                    text("materialization_id"),
                    text("label_residue_id"),
                    np.asarray(dump["source"])
                    if "source" in have
                    else np.zeros(len(mask), dtype=int),
                )
            )
            return Predictions(
                shift_cols=cols,
                pred_ppm=pred_ppm,
                true_ppm=true_ppm,
                resolved_ppm=canonical_resolved,
                mask=mask,
                effective_mask=canonical_effective,
                ambiguity=ambiguity,
                residue_code=residue_code,
                row_key=row_key,
                label_set=text("label_set_id"),
                bmrb=text("bmrb"),
                source=(
                    np.asarray(dump["source"])
                    if "source" in have
                    else np.zeros(len(mask), dtype=np.int64)
                ),
            )
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" in str(exc):
            raise ValueError(
                f"{path}: pickled object arrays are unsafe by default; "
                "convert the dump or pass allow_unsafe_pickle=True explicitly"
            ) from exc
        raise


def write_predictions(preds: Predictions, path):
    """Save a dump ``read_predictions`` can read back, under the trainer's own key names."""
    path = _refuse_repo_root(path)
    if path.suffix != ".npz":       # numpy would append it, and the caller's path would miss
        path = path.with_name(path.name + ".npz")
    fields = {
        "pred_ppm": preds.pred_ppm, "true_ppm": preds.true_ppm,
        "resolved_true_ppm": preds.resolved_ppm, "mask": preds.mask,
        "effective_mask": preds.effective_mask, "ambiguity": preds.ambiguity,
        "residue_code": preds.residue_code,
        "shift_cols": np.asarray(preds.shift_cols, dtype=np.str_),
        "row_key": np.asarray(preds.row_key, dtype=np.str_),
        "label_set_id": np.asarray(preds.label_set, dtype=np.str_),
        "bmrb": np.asarray(preds.bmrb, dtype=np.str_),
        "source": preds.source}
    np.savez_compressed(path, **fields)
    return path


def read_label_residues(path) -> np.ndarray:
    """The label-residue identity of every row of a saved dump.

    ``read_predictions`` folds it into ``row_key`` beside the materialization, so a cohort
    that matches rows across structure sources -- one label residue realized by an
    AlphaFold model, a crystal structure and a conformer -- reads it here instead of
    splitting that key on the separator ``_row_keys`` happens to use.
    """
    with np.load(path, allow_pickle=False) as dump:
        if "label_residue_id" not in dump.files:
            raise ValueError(f"{path}: dump carries no label_residue_id")
        return np.asarray([str(value) for value in dump["label_residue_id"]])


def ensemble(members) -> Predictions:
    """Average member predictions in ppm over one shared set of rows.

    Members must be the same rows scored by different checkpoints. Averaging in z is not
    offered: each member's z units are its own checkpoint's per-nucleus scale. Targets
    are then re-resolved once against the averaged prediction, since a swap resolved
    against member A is not the target the ensemble is scored on.
    """
    members = list(members)
    if not members:
        raise ValueError("an ensemble needs at least one member")
    if len(members) == 1:
        return members[0]
    head = members[0]
    for other in members[1:]:
        if other.shift_cols != head.shift_cols:
            raise ValueError("ensemble members disagree on the nucleus vocabulary")
        # label_set, bmrb and source are checked too, not just the row identity:
        # row_key is materialization|label_residue and carries none of them, yet they
        # decide the bootstrap's clusters and the per-source split. Members that agreed
        # on rows and disagreed on those would average silently, and the first member's
        # metadata would decide how the result is grouped.
        for name in ("row_key", "mask", "effective_mask", "ambiguity", "residue_code",
                     "label_set", "bmrb", "source"):
            if not np.array_equal(getattr(head, name), getattr(other, name)):
                raise ValueError(f"ensemble members disagree on {name}; they do not score "
                                 "the same rows")
        both = head.mask & other.mask
        if both.any() and np.abs(head.true_ppm - other.true_ppm)[both].max() > LABEL_TOLERANCE_PPM:
            raise ValueError("ensemble members disagree on the ground truth")
    pred = np.mean([m.pred_ppm for m in members], axis=0)
    resolved, effective = resolve(pred, head.true_ppm, head.mask, head.ambiguity,
                                  head.residue_code, head.shift_cols)
    return replace(head, pred_ppm=pred, resolved_ppm=resolved, effective_mask=effective)


def pool(parts) -> Predictions:
    """Concatenate dumps that score different rows -- the five CV held-out folds.

    Shared rows are refused: pooling ensemble members would count every residue once per
    member and shrink every confidence interval.
    """
    parts = list(parts)
    if not parts:
        raise ValueError("nothing to pool")
    if len(parts) == 1:
        return parts[0]
    head = parts[0]
    for other in parts[1:]:
        if other.shift_cols != head.shift_cols:
            raise ValueError("pooled dumps disagree on the nucleus vocabulary")
    cells = {name: np.concatenate([getattr(p, name) for p in parts]) for name in _CELL_FIELDS}
    per_row = {name: np.concatenate([getattr(p, name) for p in parts]) for name in _ROW_FIELDS}
    if len(np.unique(per_row["row_key"])) != len(per_row["row_key"]):
        raise ValueError("these dumps share rows, so pooling would count them twice -- "
                         "several checkpoints over one fold are an ensemble(), not a pool()")
    return Predictions(shift_cols=head.shift_cols, **cells, **per_row)


def align(first: Predictions, second: Predictions):
    """Restrict two arms to the rows and nuclei they share, in one matching order.

    ``compare`` refuses mismatched arms and never aligns them silently, because
    "the same 200 proteins" often means two different row sets.
    """
    columns = [c for c in first.shift_cols if c in set(second.shift_cols)]
    if not columns:
        raise ValueError("the two arms share no nucleus")
    # intersect1d keeps one occurrence of a repeated key, so a duplicated row identity
    # would quietly shrink both arms.
    for arm in (first, second):
        _unique_row_keys(arm.row_key)
    shared, here, there = np.intersect1d(first.row_key, second.row_key, return_indices=True)
    if not shared.size:
        raise ValueError("the two arms share no row")
    return _take(first, here, columns), _take(second, there, columns)


def score(preds: Predictions, *, columns=None, rows=None, boots=DEFAULT_BOOTSTRAP,
          seed=DEFAULT_SEED, group="label_set") -> dict:
    """Score a prediction population with an explicit clustered-bootstrap contract."""
    return score_predictions(
        preds,
        columns=columns,
        rows=rows,
        bootstrap=BootstrapSpec(unit=group, resamples=boots, seed=seed),
    )


def matched_rows(preds: Predictions) -> np.ndarray:
    """Select label sets represented by every structure source."""
    return matched_label_sets(preds)


def by_source(preds: Predictions, *, matched=False, columns=None, boots=DEFAULT_BOOTSTRAP,
              seed=DEFAULT_SEED, group="label_set") -> dict:
    """Score structure sources on marginal or label-set-matched rows."""
    return score_by_source(
        preds,
        matched=matched,
        columns=columns,
        bootstrap=BootstrapSpec(unit=group, resamples=boots, seed=seed),
    )


def compare(baseline: Predictions, candidate: Predictions, *, columns=None,
            boots=DEFAULT_BOOTSTRAP, seed=DEFAULT_SEED, group="label_set") -> dict:
    """Compare two aligned arms using one paired clustered bootstrap."""
    return compare_predictions(
        baseline,
        candidate,
        columns=columns,
        bootstrap=BootstrapSpec(unit=group, resamples=boots, seed=seed),
    )


def score_checkpoints(cache, fold, checkpoints, *, flags, device="cuda", batch_size=256,
                      workers=2) -> Predictions:
    """Score checkpoints on one held-out cache fold and return their ppm ensemble.

    Each checkpoint reads the fold through its own stamped stats and nucleus order, and
    members combine only once each is in ppm. ``flags`` is the QC fence; an empty dict is
    the explicit way to ask for none.
    """
    import torch

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("scoring a cache fold requires a GPU; CPU evaluation is not a "
                           "supported path for reported numbers")
    fold_dir = Path(cache) / f"fold_{fold}"
    return ensemble([score_one_checkpoint(path, fold_dir, flags=flags, device=device,
                                          batch_size=batch_size, workers=workers)
                     for path in checkpoints])


def score_one_checkpoint(checkpoint, fold_dir, *, flags, device, batch_size=256,
                         workers=2) -> Predictions:
    """One member's ppm predictions on one cache fold, read through its own stamped
    stats and nucleus order and resolved by the trainer's evaluation."""
    import torch
    from torch.utils.data import DataLoader

    from .model import load_checkpoint
    from .training import LeanCacheDataset, dataset_identity, trim_collate
    from .training import evaluate as evaluate_fold

    device = torch.device(device)
    loaded = load_checkpoint(checkpoint, device=device)
    columns = list(loaded.shift_columns)
    dataset = LeanCacheDataset([Path(fold_dir)], loaded.stats, columns)
    dataset.apply_flags(flags)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda", collate_fn=trim_collate)
    _, partner = build_loss_tensors(tuple(columns), tuple(STANDARD_RESIDUES))
    block = evaluate_fold(loaded.model, loader, device, loaded.stats, columns,
                          return_predictions=True, partner=partner.numpy())["predictions"]
    # shuffle=False, so row i of the prediction block is dataset.index[i].
    return _from_evaluate(block, columns, dataset_identity(dataset))


def _from_evaluate(block, columns, identity) -> Predictions:
    def text(name):
        return np.asarray([str(row.get(name) or "") for row in identity])
    mask, effective = _bool(block["mask"]), _bool(block["effective_mask"])
    source = np.asarray([int(row.get("source", 0)) for row in identity])
    return Predictions(
        shift_cols=tuple(columns), pred_ppm=np.asarray(block["pred_ppm"], dtype=np.float64),
        true_ppm=np.asarray(block["true_ppm"], dtype=np.float64),
        resolved_ppm=np.asarray(block["resolved_true_ppm"], dtype=np.float64),
        mask=mask, effective_mask=effective, ambiguity=np.asarray(block["ambiguity"]),
        residue_code=np.asarray(block["residue_code"]),
        row_key=_row_keys(text("materialization_id"), text("label_residue_id"), source),
        label_set=text("label_set_id"), bmrb=text("bmrb"), source=source)


def _refuse_repo_root(path):
    """Outputs never land in a checkout's root; results belong beside the run."""
    path = Path(path)
    if (path.resolve().parent / ".git").exists():
        raise ValueError(f"{path} would write into a repository root; write results next "
                         "to the run or under the build root instead")
    return path


def _load_arm(specs, *, allow_unsafe_pickle=False):
    """One arm from the command line: separate arguments are pooled (different rows, e.g.
    the held-out folds), comma-separated paths inside one argument are ensembled."""
    return pool([ensemble([
        read_predictions(p, allow_unsafe_pickle=allow_unsafe_pickle)
        for p in spec.split(",") if p
    ])
                 for spec in specs])


def _ordered(names):
    """Backbone first -- it is what the field reads -- then the side chains in cache order."""
    return ([c for c in BACKBONE_SHIFTS if c in names]
            + [c for c in names if c not in BACKBONE_SHIFTS])


def _print_scores(result, title):
    print(f"\n{title}")
    print(f'{"nucleus":14s} {"n":>9s} {"medAE":>7s} {"MAE":>7s} {"RMSE":>7s} '
          f'{"RMSE 95% CI":>19s} {"r":>6s} {"amp":>6s}')
    for name in _ordered(result):
        row = result[name]
        ci = row.get("rmse_ci")
        span = f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else "--"
        print(f'{name:14s} {row["n"]:9,} {row["medae"]:7.4f} {row["mae"]:7.4f} '
              f'{row["rmse"]:7.4f} {span:>19s} {row["r"]:6.3f} {row["amplitude"]:6.3f}')


def _print_thin(names, result, cells="observed cells"):
    """Name every nucleus a table dropped for want of scorable cells."""
    if thin := [c for c in names if c not in result]:
        print(f"not scored (under {MIN_SCORABLE_CELLS} {cells}): {', '.join(thin)}")


def _print_comparison(result, title):
    print(f"\n{title}")
    print(f'{"nucleus":14s} {"cells":>9s} {"RMSE base":>9s} {"RMSE cand":>9s} {"delta":>8s} '
          f'{"95% CI on delta":>22s}  verdict')
    for name in _ordered(result):
        row = result[name]
        span = f'[{row["ci"][0]:+.4f}, {row["ci"][1]:+.4f}]'
        print(f'{name:14s} {row["n"]:9,} {row["rmse_baseline"]:9.4f} '
              f'{row["rmse_candidate"]:9.4f} {row["delta_rmse"]:+8.4f} {span:>22s}  '
              f'{row["verdict"]}')


def evaluate_command(args):
    if bool(args.preds) == bool(args.cache):
        raise ValueError("score either saved dumps (--preds) or a cache fold (--cache), not both")
    if args.cache:
        import torch
        # Cluster nodes give a job a small /dev/shm, which the default sharing strategy
        # passes every collated batch through; it surfaces as a DataLoader "Bus error".
        torch.multiprocessing.set_sharing_strategy("file_system")
        if args.fold is None or not args.ckpt or args.flags is None:
            raise ValueError("--cache scoring needs --fold, --ckpt and --flags")
        preds = score_checkpoints(args.cache, args.fold, args.ckpt,
                                  flags=json.loads(Path(args.flags).read_text()),
                                  device=args.device, batch_size=args.batch_size,
                                  workers=args.workers)
    else:
        preds = _load_arm(
            args.preds, allow_unsafe_pickle=args.allow_unsafe_legacy_pickle
        )
    print(f"{preds.n_rows:,} rows x {len(preds.shift_cols)} nuclei, "
          f"{int(preds.effective_mask.sum()):,} scorable cells, "
          f"{len(np.unique(preds.label_set)):,} label sets")
    if args.save_npz:
        print(f"-> {write_predictions(preds, args.save_npz)}")

    common = {"boots": args.boots, "seed": args.seed}
    report = {"per_nucleus": score(preds, **common)}
    print("\nr / amplitude are on the SECONDARY shift (per-residue-type mean removed);\n"
          "MAE and RMSE are ppm error, which that centring does not change.")
    _print_scores(report["per_nucleus"], "held-out error per nucleus")
    _print_thin(preds.shift_cols, report["per_nucleus"])
    if args.by_source:
        backbone = [c for c in BACKBONE_SHIFTS if c in preds.shift_cols]
        for matched in (False, True):
            key = "matched" if matched else "marginal"
            report[key] = by_source(preds, matched=matched, columns=backbone, **common)
            for source, table in report[key].items():
                _print_scores(table, f"{key.upper()} -- {source}")
    if args.baseline:
        baseline = _load_arm(
            args.baseline, allow_unsafe_pickle=args.allow_unsafe_legacy_pickle
        )
        report["comparison"] = compare(baseline, preds, **common)
        _print_comparison(report["comparison"], "paired difference (baseline - candidate)")
        # Pairing intersects two effective masks, so it drops more than score did; the
        # paired table carries the model-vs-model comparison, so the drop is reported too.
        _print_thin(preds.shift_cols, report["comparison"], "cells both arms scored")
    if args.out:
        Path(_refuse_repo_root(args.out)).write_text(json.dumps(report, indent=1) + "\n")
        print(f"\n-> {args.out}")
    return report


def add_evaluate_arguments(parser):
    parser.add_argument("--preds", nargs="+", metavar="NPZ[,NPZ...]",
                        help="saved prediction dumps. Separate arguments are POOLED (the "
                             "held-out folds); comma-separated paths in one argument are "
                             "ENSEMBLED (averaged in ppm)")
    parser.add_argument("--cache", help="cache root to score directly (requires a GPU)")
    parser.add_argument("--fold", type=int, help="held-out fold of --cache to score")
    parser.add_argument("--ckpt", nargs="+", help="checkpoints to score on --fold")
    parser.add_argument("--flags", help="QC fence JSON; an empty object asks for no fence")
    parser.add_argument("--baseline", nargs="+", metavar="NPZ[,NPZ...]",
                        help="a second arm; reported as a paired per-nucleus difference")
    parser.add_argument(
        "--allow-unsafe-legacy-pickle",
        action="store_true",
        help="explicitly trust NPZ object arrays; convert the dump and stop using this flag",
    )
    parser.add_argument("--by-source", action="store_true",
                        help="also break the backbone down by structure source, marginal "
                             "and matched on label sets every source realizes")
    parser.add_argument("--boots", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-npz", help="write the scored population as a dump")
    parser.add_argument("--out", help="write the report as JSON")
    parser.set_defaults(handler=evaluate_command)
