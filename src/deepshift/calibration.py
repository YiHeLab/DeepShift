"""Fit and apply DeepShift prediction intervals.

The final protocol has one direction. The five model members are evaluated on fold 0;
fold 0 alone fits both the variance mapping and the split-conformal multipliers. Fold 6
is reserved for one evaluation of the finished artifact. Runtime prediction therefore
requires an explicit fitted artifact: it never falls back to constants from another
model or to a fit that used fold 6.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CALIBRATION_SCHEMA = "deepshift.interval-calibration.v2"
EVALUATION_SCHEMA = "deepshift.interval-calibration-evaluation.v1"
CALIBRATION_FOLD = 0
EVALUATION_FOLD = 6
MEMBER_FOLDS = (1, 2, 3, 4, 5)
BACKBONE_NUCLEI = ("N", "H", "CA", "CB", "C", "HA")
DEFAULT_LEVELS = (0.68, 0.90, 0.95)
DEFAULT_N_BINS = 10


def _level_key(level: float | str) -> str:
    value = float(level)
    if not 0.0 < value < 1.0:
        raise ValueError(f"interval level must be between zero and one; got {level!r}")
    return f"{value:.2f}"


def _vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def member_spread(member_predictions) -> np.ndarray:
    """Per-cell standard deviation across ensemble members in ppm (``ddof=1``)."""
    members = np.asarray(member_predictions, dtype=np.float64)
    if members.ndim < 2 or members.shape[0] < 2:
        raise ValueError("member predictions need at least two members")
    if not np.isfinite(members).all():
        raise ValueError("member predictions contain a non-finite value")
    return members.std(axis=0, ddof=1)


def load_label_set_exclusion(path: str | Path) -> dict[str, Any]:
    """Read the fold-0 label sets the interval fit removes.

    The folds stage writes this file (``fold0_calibration_exclusions.json``): the label sets
    with a training-fold relative and the case-study records.
    """
    payload = json.loads(Path(path).read_text())
    label_sets = payload.get("label_sets")
    if not isinstance(label_sets, list) or any(not isinstance(x, str) for x in label_sets):
        raise ValueError(f"{path}: exclusion list needs a list of label-set identities")
    return {
        "edge_groups_by_family": dict(
            payload.get("edge_groups_spanning_fold0_and_training_by_family", {})
        ),
        "label_sets": sorted(set(label_sets)),
    }


def exclude_label_sets(predictions, exclusion: Mapping[str, Any] | None):
    """Rows to keep after removing the listed label sets, and the record of what went."""
    keep = np.ones(predictions.n_rows, dtype=bool)
    if exclusion is None:
        return keep, None
    keep = ~np.isin(predictions.label_set, exclusion["label_sets"])
    present = set(predictions.label_set)
    removed = sorted(present & set(exclusion["label_sets"]))
    record = {
        "edge_groups_by_family": exclusion["edge_groups_by_family"],
        "label_sets_listed": len(exclusion["label_sets"]),
        "label_sets_removed": len(removed),
        "label_sets_remaining": int(len(present) - len(removed)),
        "rows_removed": int((~keep).sum()),
        "rows_remaining": int(keep.sum()),
        "cells_removed": int((predictions.effective_mask & ~keep[:, None]).sum()),
    }
    return keep, record


def load_channel_exclusion(path: str | Path) -> dict[str, Any]:
    """Read the deposited channels the fold-6 benchmark drops.

    The file is ``benchmarks/reference/anticorrelated_channels.json``: one entry per
    (label set, backbone nucleus) whose deposited series every independent judge
    disagrees with in sign. The fold-6 evaluation drops the same cells every benchmark
    float drops.
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    entries = payload.get("channels")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: channel list needs a non-empty 'channels' list")
    channels = set()
    for entry in entries:
        label_set, nucleus = entry.get("label_set"), entry.get("nucleus")
        if not isinstance(label_set, str) or nucleus not in BACKBONE_NUCLEI:
            raise ValueError(f"{path}: each channel needs a label_set and a backbone nucleus")
        channels.add((label_set, nucleus))
    return {"channels": [list(pair) for pair in sorted(channels)]}


def exclude_channels(predictions, channels: Mapping[str, Any]):
    """Cells to keep after clearing the listed (label set, nucleus) channels, and the
    record of what went."""
    from .chemistry import atom_from_shift_column

    keep = np.ones(predictions.effective_mask.shape, dtype=bool)
    columns = list(predictions.shift_cols)
    removed = {}
    present = 0
    for label_set, nucleus in channels["channels"]:
        index = next((j for j, column in enumerate(columns)
                      if atom_from_shift_column(column) == nucleus), None)
        if index is None:
            continue
        rows = predictions.label_set == label_set
        cells = int((predictions.effective_mask[rows, index] & keep[rows, index]).sum())
        present += int(cells > 0)
        removed[nucleus] = removed.get(nucleus, 0) + cells
        keep[rows, index] = False
    record = {
        "channels_listed": len(channels["channels"]),
        "channels_present": present,
        "cells_removed": int(sum(removed.values())),
        "cells_removed_by_nucleus": {n: removed[n] for n in BACKBONE_NUCLEI if n in removed},
    }
    return keep, record


def calibration_population(paths, *, exclusion: Mapping[str, Any] | None = None,
                           channels: Mapping[str, Any] | None = None):
    """Extract per-nucleus errors and member spread from five canonical dumps.

    Returns ``(errors, spreads, record)``. The record describes the label sets
    removed (``exclusion``, the fold-0 fit) or the channels removed (``channels``, the
    fold-6 evaluation), and is ``None`` when neither was applied.
    """
    if exclusion is not None and channels is not None:
        raise ValueError("label sets are removed from the fold-0 fit and channels from the "
                         "fold-6 evaluation; one population takes one of them")
    from . import evaluate
    from .chemistry import atom_from_shift_column

    paths = [Path(path) for path in paths]
    if len(paths) != len(MEMBER_FOLDS) or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("calibration needs five distinct member prediction dumps")
    members = [evaluate.read_predictions(path) for path in paths]
    ensemble = evaluate.ensemble(members)
    spread_all = member_spread(np.stack([member.pred_ppm for member in members]))
    keep, record = exclude_label_sets(ensemble, exclusion)
    cells = np.ones(ensemble.effective_mask.shape, dtype=bool)
    if channels is not None:
        cells, record = exclude_channels(ensemble, channels)
    errors, spreads = {}, {}
    columns = list(ensemble.shift_cols)
    for column in columns:
        nucleus = atom_from_shift_column(column)
        if nucleus not in BACKBONE_NUCLEI:
            continue
        index = columns.index(column)
        rows = ensemble.effective_mask[:, index] & keep & cells[:, index]
        errors[nucleus] = ensemble.pred_ppm[rows, index] - ensemble.resolved_ppm[rows, index]
        spreads[nucleus] = spread_all[rows, index]
    if set(errors) != set(BACKBONE_NUCLEI):
        raise ValueError(f"prediction dumps must cover exactly the backbone set {BACKBONE_NUCLEI}")
    return errors, spreads, record


def equal_count_bins(spread, *, n_bins: int = DEFAULT_N_BINS) -> list[np.ndarray]:
    """Split cells into ``n_bins`` equal-count groups ordered by spread.

    Sorting and ``array_split`` keep the bins equal-count even when spread values tie,
    which quantile edges do not. Every producer that bins by spread -- the fit and the
    figure that plots measured error against it -- reads its bins from here, so the two
    are the same partition of the same cells.
    """
    sd = _vector(spread, "spread")
    if not isinstance(n_bins, int) or n_bins < 2 or sd.size < n_bins:
        raise ValueError("n_bins must be an integer >=2 and no larger than the cell count")
    return list(np.array_split(np.argsort(sd, kind="stable"), n_bins))


def variance_fit(errors, spread, *, n_bins: int = DEFAULT_N_BINS) -> tuple[dict, list[dict]]:
    """Fit ``RMSE^2 = floor^2 + gain * spread^2`` over equal-count spread bins."""
    error = _vector(errors, "errors")
    sd = _vector(spread, "spread")
    if error.shape != sd.shape:
        raise ValueError("errors and spread must have the same shape")

    bins = []
    for index, rows in enumerate(equal_count_bins(sd, n_bins=n_bins), start=1):
        bins.append({
            "bin": index,
            "n": int(rows.size),
            "spread_mean_ppm": float(sd[rows].mean()),
            "rmse_ppm": float(np.sqrt(np.mean(error[rows] ** 2))),
        })
    return variance_fit_from_bins(bins), bins


def variance_fit_from_bins(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    """Fit the variance mapping from measured bins: one straight line through
    ``(spread_mean_ppm^2, rmse_ppm^2)``, whose intercept is the residual floor."""
    if len(rows) < 2:
        raise ValueError("variance mapping needs at least two bins")
    x = np.asarray([float(row["spread_mean_ppm"]) for row in rows]) ** 2
    y = np.asarray([float(row["rmse_ppm"]) for row in rows]) ** 2
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("variance bins contain a non-finite value")
    if np.ptp(x) == 0.0:
        raise ValueError("variance mapping is unidentified because every spread is equal")
    gain, intercept = np.polyfit(x, y, 1)
    floor = float(np.sqrt(max(float(intercept), 0.0)))
    if not np.isfinite(gain) or gain <= 0.0 or floor <= 0.0:
        raise ValueError(
            f"non-physical variance mapping: floor={floor:.6g}, gain={float(gain):.6g}"
        )
    return {"floor_ppm": floor, "gain": float(gain)}


def error_scale(spread, *, floor_ppm: float, gain: float) -> np.ndarray:
    """Map ensemble spread to the fitted per-cell error scale in ppm."""
    sd = np.asarray(spread, dtype=np.float64)
    if not np.isfinite(sd).all() or (sd < 0.0).any():
        raise ValueError("spread must be finite and non-negative")
    if not np.isfinite(floor_ppm) or floor_ppm <= 0.0:
        raise ValueError("floor_ppm must be finite and positive")
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("gain must be finite and positive")
    return np.sqrt(float(floor_ppm) ** 2 + float(gain) * sd ** 2)


def conformal_quantile(errors, scale, level: float | str) -> float:
    """Finite-sample split-conformal quantile of ``abs(error) / scale``."""
    error = _vector(errors, "errors")
    sigma = _vector(scale, "scale")
    if error.shape != sigma.shape:
        raise ValueError("errors and scale must have the same shape")
    if (sigma <= 0.0).any():
        raise ValueError("scale must be positive")
    probability = float(_level_key(level))
    scores = np.sort(np.abs(error) / sigma)
    index = int(np.ceil((scores.size + 1) * probability))
    if index > scores.size:
        raise ValueError(
            f"{scores.size} calibration cells cannot support level {probability:.2f}"
        )
    return float(scores[index - 1])


@dataclass(frozen=True)
class IntervalCalibration:
    """Validated, fold-0-only interval calibration artifact."""

    constants: dict[str, dict[str, Any]]
    n_bins: int = DEFAULT_N_BINS
    levels: tuple[str, ...] = tuple(_level_key(x) for x in DEFAULT_LEVELS)
    calibration_fold: int = CALIBRATION_FOLD
    evaluation_fold: int = EVALUATION_FOLD
    member_folds: tuple[int, ...] = MEMBER_FOLDS
    exclusion: dict[str, Any] | None = None

    def __post_init__(self):
        if self.calibration_fold != CALIBRATION_FOLD or self.evaluation_fold != EVALUATION_FOLD:
            raise ValueError("the final interval protocol is fit on fold 0 and evaluated on fold 6")
        if tuple(self.member_folds) != MEMBER_FOLDS:
            raise ValueError(f"the final interval protocol requires model folds {MEMBER_FOLDS}")
        if set(self.constants) != set(BACKBONE_NUCLEI):
            raise ValueError(f"calibration constants must cover exactly {BACKBONE_NUCLEI}")
        expected_levels = tuple(_level_key(level) for level in self.levels)
        if len(set(expected_levels)) != len(expected_levels):
            raise ValueError("calibration levels must be unique")
        for nucleus, constants in self.constants.items():
            error_scale([0.0], floor_ppm=constants["floor_ppm"], gain=constants["gain"])
            if int(constants.get("n_calibration_cells", 0)) < self.n_bins:
                raise ValueError(f"{nucleus}: too few calibration cells")
            if set(constants["q"]) != set(expected_levels):
                raise ValueError(f"{nucleus}: conformal levels do not match the artifact")
            if any(not np.isfinite(float(q)) or float(q) <= 0.0
                   for q in constants["q"].values()):
                raise ValueError(f"{nucleus}: conformal multipliers must be finite and positive")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": CALIBRATION_SCHEMA,
            "method": {
                "spread": "standard deviation across the five member predictions in ppm; ddof=1",
                "variance_mapping": "RMSE^2 = floor_ppm^2 + gain * spread^2",
                "binning": f"{self.n_bins} equal-count spread bins per nucleus",
                "conformal_score": "abs(error) / mapped_error_scale",
                "conformal_index": "ceil((n + 1) * level)",
            },
            "calibration_fold": self.calibration_fold,
            "reserved_evaluation_fold": self.evaluation_fold,
            "member_folds": list(self.member_folds),
            "n_bins": self.n_bins,
            "levels": list(self.levels),
            "constants": self.constants,
            "fold0_exclusion": self.exclusion,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> IntervalCalibration:
        if payload.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError(
                f"expected calibration schema {CALIBRATION_SCHEMA!r}; got {payload.get('schema')!r}"
            )
        return cls(
            constants={str(k): dict(v) for k, v in payload["constants"].items()},
            n_bins=int(payload["n_bins"]),
            levels=tuple(str(x) for x in payload["levels"]),
            calibration_fold=int(payload["calibration_fold"]),
            evaluation_fold=int(payload["reserved_evaluation_fold"]),
            member_folds=tuple(int(x) for x in payload["member_folds"]),
            exclusion=payload.get("fold0_exclusion"),
        )


def fit_interval_calibration(
    errors: Mapping[str, Any],
    spreads: Mapping[str, Any],
    *,
    levels=DEFAULT_LEVELS,
    n_bins: int = DEFAULT_N_BINS,
    exclusion: Mapping[str, Any] | None = None,
) -> IntervalCalibration:
    """Fit the complete artifact from fold 0. Fold 6 data are not accepted here."""
    if set(errors) != set(BACKBONE_NUCLEI) or set(spreads) != set(BACKBONE_NUCLEI):
        raise ValueError(f"errors and spreads must cover exactly {BACKBONE_NUCLEI}")
    level_keys = tuple(_level_key(level) for level in levels)
    constants = {}
    for nucleus in BACKBONE_NUCLEI:
        error = _vector(errors[nucleus], f"{nucleus} errors")
        spread = _vector(spreads[nucleus], f"{nucleus} spread")
        params, bins = variance_fit(error, spread, n_bins=n_bins)
        sigma = error_scale(spread, **params)
        constants[nucleus] = {
            **params,
            "q": {key: conformal_quantile(error, sigma, key) for key in level_keys},
            "n_calibration_cells": int(error.size),
            "bins": bins,
        }
    return IntervalCalibration(
        constants=constants,
        n_bins=n_bins,
        levels=level_keys,
        exclusion=None if exclusion is None else dict(exclusion),
    )


def load_calibration(value: IntervalCalibration | Mapping[str, Any] | str | Path):
    if isinstance(value, IntervalCalibration):
        return value
    if isinstance(value, Mapping):
        return IntervalCalibration.from_mapping(value)
    return IntervalCalibration.from_mapping(json.loads(Path(value).read_text()))


def save_calibration(calibration: IntervalCalibration, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(calibration.to_mapping(), indent=2) + "\n")
    return destination


def halfwidth(nucleus, ensemble_sd_ppm, *, calibration, level="0.90"):
    """Prediction-interval half-width in ppm; NaN outside the six backbone nuclei."""
    fitted = load_calibration(calibration)
    key = _level_key(level)
    nuclei, sd = np.broadcast_arrays(
        np.asarray(nucleus, dtype=object), np.asarray(ensemble_sd_ppm, dtype=np.float64)
    )
    out = np.full(sd.shape, np.nan, dtype=np.float64)
    for name, constants in fitted.constants.items():
        selected = nuclei == name
        if selected.any():
            out[selected] = float(constants["q"][key]) * error_scale(
                sd[selected], floor_ppm=constants["floor_ppm"], gain=constants["gain"]
            )
    return float(out) if out.ndim == 0 else out


def evaluate_interval_calibration(
    calibration: IntervalCalibration,
    errors: Mapping[str, Any],
    spreads: Mapping[str, Any],
    *,
    channel_exclusion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen fold-0 artifact once on reserved fold 6.

    ``channel_exclusion`` is the record of the deposited channels removed from the
    population before the errors were taken; the report carries it.
    """
    fitted = load_calibration(calibration)
    if set(errors) != set(BACKBONE_NUCLEI) or set(spreads) != set(BACKBONE_NUCLEI):
        raise ValueError(f"errors and spreads must cover exactly {BACKBONE_NUCLEI}")
    results = {}
    for nucleus in BACKBONE_NUCLEI:
        error = _vector(errors[nucleus], f"{nucleus} errors")
        spread = _vector(spreads[nucleus], f"{nucleus} spread")
        if error.shape != spread.shape:
            raise ValueError(f"{nucleus}: errors and spread must have the same shape")
        constants = fitted.constants[nucleus]
        sigma = error_scale(spread, floor_ppm=constants["floor_ppm"], gain=constants["gain"])
        levels = {}
        for level in fitted.levels:
            widths = float(constants["q"][level]) * sigma
            levels[level] = {
                "realized_coverage": float(np.mean(np.abs(error) <= widths)),
                "median_half_width_ppm": float(np.median(widths)),
            }
        results[nucleus] = {
            "n_evaluation_cells": int(error.size),
            "rmse_ppm": float(np.sqrt(np.mean(error ** 2))),
            "levels": levels,
        }
    return {
        "schema": EVALUATION_SCHEMA,
        "calibration_fold": CALIBRATION_FOLD,
        "evaluation_fold": EVALUATION_FOLD,
        "member_folds": list(MEMBER_FOLDS),
        "results": results,
        **({"fold6_channel_exclusion": dict(channel_exclusion)}
           if channel_exclusion is not None else {}),
    }


def fit_command(args):
    exclusion = load_label_set_exclusion(args.exclude_label_sets)
    errors, spreads, record = calibration_population(args.member, exclusion=exclusion)
    fitted = fit_interval_calibration(
        errors, spreads, levels=args.level, n_bins=args.bins, exclusion=record,
    )
    path = save_calibration(fitted, args.out)
    cells = {nucleus: fitted.constants[nucleus]["n_calibration_cells"]
             for nucleus in BACKBONE_NUCLEI}
    print(f"wrote {path}: fold 0 minus {record['label_sets_removed']} label sets "
          f"({record['cells_removed']} cells), cells={cells}")
    return fitted


def evaluation_command(args):
    fitted = load_calibration(args.calibration)
    if fitted.exclusion is None:
        raise ValueError(
            f"{args.calibration} records no fold-0 exclusion; the final protocol fits on "
            "fold 0 minus its training-fold relatives and case-study records"
        )
    channels = load_channel_exclusion(args.exclude_channels)
    errors, spreads, record = calibration_population(args.member, channels=channels)
    report = evaluate_interval_calibration(
        fitted, errors, spreads, channel_exclusion=record
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    coverage = {nucleus: row["levels"].get("0.90", {}).get("realized_coverage")
                for nucleus, row in report["results"].items()}
    print(f"wrote {destination}: fold 6 evaluated once minus {record['cells_removed']} cells "
          f"of {record['channels_present']} excluded channels, 90% coverage={coverage}")
    return report


def add_interval_arguments(subparsers):
    fit = subparsers.add_parser(
        "fit", help="fit variance mapping and split conformal multipliers on fold 0"
    )
    fit.add_argument(
        "--member", nargs=5, required=True, metavar=("M1", "M2", "M3", "M4", "M5"),
        help="five single-checkpoint canonical prediction dumps for fold 0",
    )
    fit.add_argument(
        "--exclude-label-sets", required=True, metavar="JSON",
        help="the folds stage's fold0_calibration_exclusions.json: the fold-0 label sets "
             "with a training-fold relative and the case-study records, removed before "
             "the fit",
    )
    fit.add_argument("--bins", type=int, default=DEFAULT_N_BINS)
    fit.add_argument("--level", type=float, nargs="+", default=list(DEFAULT_LEVELS))
    fit.add_argument("--out", required=True)
    fit.set_defaults(handler=fit_command)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate a frozen fold-0 interval artifact once on fold 6"
    )
    evaluate.add_argument(
        "--member", nargs=5, required=True, metavar=("M1", "M2", "M3", "M4", "M5"),
        help="the same five checkpoints' single-member prediction dumps for fold 6",
    )
    evaluate.add_argument("--calibration", required=True)
    evaluate.add_argument(
        "--exclude-channels", required=True, metavar="JSON",
        help="benchmarks/reference/anticorrelated_channels.json: the deposited channels "
             "every fold-6 benchmark drops, removed before coverage is measured",
    )
    evaluate.add_argument("--out", required=True)
    evaluate.set_defaults(handler=evaluation_command)
