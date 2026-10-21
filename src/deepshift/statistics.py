"""Per-nucleus estimates and clustered bootstrap comparisons."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_SCORABLE_CELLS = 8
LABEL_TOLERANCE_PPM = 1e-3
# The one seed every reported interval is drawn from, so a table and the figure beside
# it resample identically.
DEFAULT_SEED = 20260725


@dataclass(frozen=True)
class BootstrapSpec:
    """The complete resampling contract for one reported interval."""

    unit: str
    resamples: int
    seed: int

    def __post_init__(self):
        if self.unit not in {"label_set", "bmrb"}:
            raise ValueError(f"unsupported bootstrap unit: {self.unit!r}")
        if self.resamples < 0:
            raise ValueError("bootstrap resamples must be non-negative")


def column_index(predictions, name: str) -> int:
    """Return the position of a named nucleus in a prediction population."""
    if name not in predictions.shift_cols:
        raise KeyError(f"{name} is not one of the scored nuclei")
    return predictions.shift_cols.index(name)


def unit_codes(keys) -> tuple[np.ndarray, int]:
    """Return a compact integer code for each resampling unit."""
    keys = np.asarray(keys)
    if (keys == "").any():
        raise ValueError("a clustered bootstrap requires a non-empty identity for every unit")
    _, codes = np.unique(keys, return_inverse=True)
    return np.asarray(codes).ravel(), int(codes.max()) + 1


def block_counts(codes, n_groups):
    """Return cell counts by resampling unit and the units represented."""
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    return counts, counts > 0


def bootstrap_multiplicities(n_units: int, spec: BootstrapSpec, *, stream: int = 0):
    """Draw unit multiplicities using a stream fixed by the reported variable."""
    rng = np.random.default_rng([spec.seed, stream])
    return rng.multinomial(
        n_units,
        np.full(n_units, 1.0 / n_units),
        size=spec.resamples,
    )


def bootstrap_indices(n_units: int, spec: BootstrapSpec, *, stream: int | None = None):
    """Draw resampling-unit indices for statistics that do not reduce to counts."""
    seed = spec.seed if stream is None else [spec.seed, stream]
    return np.random.default_rng(seed).integers(
        0, n_units, size=(spec.resamples, n_units)
    )


def rmse_replicates(weights, counts, squared_error_sums):
    """Return RMSE for each matrix row of resampling-unit multiplicities."""
    return np.sqrt(weights @ squared_error_sums / (weights @ counts))


def error_replicates(residual, unit, n_units, draws):
    """Return RMSE/MAE estimates and replicates for one arm."""
    squared = np.bincount(unit, weights=residual**2, minlength=n_units)
    absolute = np.bincount(unit, weights=np.abs(residual), minlength=n_units)
    counts = np.bincount(unit, minlength=n_units).astype(float)
    bootstrap_squared = squared[draws].sum(1)
    bootstrap_absolute = absolute[draws].sum(1)
    bootstrap_counts = counts[draws].sum(1)
    keep = bootstrap_counts > 0
    return (
        float(np.sqrt(squared.sum() / counts.sum())),
        np.sqrt(bootstrap_squared[keep] / bootstrap_counts[keep]),
        float(absolute.sum() / counts.sum()),
        bootstrap_absolute[keep] / bootstrap_counts[keep],
    )


def percentile_interval(replicates, low=2.5, high=97.5):
    """Return a two-sided percentile interval as JSON-native floats."""
    return [float(np.percentile(replicates, low)), float(np.percentile(replicates, high))]


def paired_verdict(interval) -> str:
    """Name the arm a paired interval on ``baseline - candidate`` error favours.

    An interval spanning zero is a tie: the error the two arms make on the same cells is
    not separated by the resampling.
    """
    low, high = interval
    return "candidate" if low > 0 else "baseline" if high < 0 else "tie"


def weighted_median_replicates(
    values,
    unit,
    multiplicities,
    *,
    even: str = "mean",
    block: int = 1024,
):
    """Return exact medians of cells reweighted by bootstrap-unit multiplicity."""
    if even not in {"mean", "lower"}:
        raise ValueError("even must be 'mean' or 'lower'")
    n_units = multiplicities.shape[1]
    order = np.argsort(values, kind="stable")
    padding = -len(values) % block
    sorted_values = np.concatenate([values[order], np.full(padding, np.inf)])
    sorted_units = np.concatenate(
        [unit[order], np.full(padding, n_units)]
    ).reshape(-1, block)
    weights = np.concatenate(
        [
            multiplicities,
            np.zeros((len(multiplicities), 1), multiplicities.dtype),
        ],
        axis=1,
    )
    per_block = np.stack(
        [np.bincount(row, minlength=n_units + 1) for row in sorted_units]
    ).astype(float)
    cumulative_blocks = np.rint(
        np.cumsum(weights @ per_block.T, axis=1)
    ).astype(np.int64)
    total = cumulative_blocks[:, -1]
    rows = np.arange(len(total))
    targets = ((total - 1) // 2, total // 2) if even == "mean" else (
        (total - 1) // 2,
        (total - 1) // 2,
    )
    middle = []
    for target in targets:
        target_block = np.clip(
            (cumulative_blocks < target[:, None] + 1).sum(1),
            0,
            cumulative_blocks.shape[1] - 1,
        )
        before = np.where(
            target_block > 0,
            cumulative_blocks[rows, np.maximum(target_block - 1, 0)],
            0,
        )
        cumulative_cells = np.cumsum(
            np.take_along_axis(weights, sorted_units[target_block], axis=1),
            axis=1,
            dtype=np.int64,
        )
        cell = np.clip(
            (
                cumulative_cells + before[:, None]
                < target[:, None] + 1
            ).sum(1),
            0,
            block - 1,
        )
        middle.append(sorted_values[target_block * block + cell])
    return np.where(total > 0, 0.5 * (middle[0] + middle[1]), np.nan)


def _centred_secondary(predicted, observed, residue_code):
    """Predicted and observed with each residue type's observed mean removed, plus the
    cells that had a residue type to centre on."""
    predicted = predicted.astype(np.float64).copy()
    observed = observed.astype(np.float64).copy()
    keep = np.zeros(predicted.size, dtype=bool)
    for code in np.unique(residue_code):
        rows = residue_code == code
        if rows.sum() < 2:
            continue
        center = observed[rows].mean()
        predicted[rows] -= center
        observed[rows] -= center
        keep |= rows
    return predicted, observed, keep


def secondary_shift_replicates(predicted, observed, residue_code, unit, draws):
    """Bootstrap replicates ``(correlation, amplitude)`` of the secondary-shift agreement.

    The residue-type centring is the cohort's, computed once, so a replicate differs
    only in which clusters it draws. Each drawn cluster contributes its sums of the
    centred values, and the correlation is formed from those sums without rebuilding
    the cells, which is what makes a cluster bootstrap of a correlation affordable. ``unit`` is one dense cluster code per cell and ``draws`` the cluster
    indices of each replicate.
    """
    predicted, observed, keep = _centred_secondary(predicted, observed, residue_code)
    predicted, observed = predicted[keep], observed[keep]
    unit = np.asarray(unit)[keep]
    draws = np.asarray(draws)
    # A cluster the draws can select but this nucleus has no cell in still needs a row,
    # or the draw would index past the sums.
    n_units = max(int(unit.max()) + 1 if unit.size else 0,
                  int(draws.max()) + 1 if draws.size else 0)
    weights = (predicted, observed, predicted * predicted,
               observed * observed, predicted * observed)
    sums = [np.bincount(unit, weights=w, minlength=n_units) for w in weights]
    counts = np.bincount(unit, minlength=n_units).astype(np.float64)
    n = counts[draws].sum(1)
    sx, sy, sxx, syy, sxy = (s[draws].sum(1) for s in sums)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread_x = sxx - sx * sx / n
        spread_y = syy - sy * sy / n
        correlation = (sxy - sx * sy / n) / np.sqrt(spread_x * spread_y)
        amplitude = np.sqrt(spread_x / spread_y)
    usable = (n > 1) & (spread_x > 0) & (spread_y > 0)
    return correlation[usable], amplitude[usable]


def secondary_shift_agreement(predicted, observed, residue_code):
    """Return correlation and amplitude after centring each observed residue type."""
    predicted, observed, keep = _centred_secondary(predicted, observed, residue_code)
    predicted, observed = predicted[keep], observed[keep]
    if predicted.size < 2 or observed.std() == 0 or predicted.std() == 0:
        return float("nan"), float("nan"), int(keep.sum())
    return (
        float(np.corrcoef(predicted, observed)[0, 1]),
        float(predicted.std() / observed.std()),
        int(keep.sum()),
    )


def prediction_unit_codes(predictions, unit: str):
    """Return bootstrap codes from a prediction population's declared identity."""
    keys = {"label_set": predictions.label_set, "bmrb": predictions.bmrb}[unit]
    try:
        return unit_codes(keys)
    except ValueError as exc:
        raise ValueError(
            f"the dump carries no {unit} identity, so a clustered bootstrap "
            "would silently degenerate to one block"
        ) from exc


def score(predictions, *, bootstrap: BootstrapSpec, columns=None, rows=None) -> dict:
    """Return per-nucleus ppm errors and clustered-bootstrap RMSE intervals."""
    cells = predictions.effective_mask
    if rows is not None:
        rows = np.asarray(rows)
        if rows.dtype != bool or rows.shape != (predictions.n_rows,):
            raise TypeError("rows selects with a bool mask over every row, not indices")
        cells = cells & rows[:, None]
    codes, n_groups = prediction_unit_codes(predictions, bootstrap.unit)
    result = {}
    for name in (columns or predictions.shift_cols):
        column = column_index(predictions, name)
        picked = np.flatnonzero(cells[:, column])
        if picked.size < MIN_SCORABLE_CELLS:
            continue
        error = (
            predictions.pred_ppm[picked, column]
            - predictions.resolved_ppm[picked, column]
        )
        absolute = np.abs(error)
        correlation, amplitude, n_secondary = secondary_shift_agreement(
            predictions.pred_ppm[picked, column],
            predictions.resolved_ppm[picked, column],
            predictions.residue_code[picked],
        )
        entry = {
            "n": int(error.size),
            "n_label_sets": len(np.unique(codes[picked])),
            "rmse": float(np.sqrt((error**2).mean())),
            "mae": float(absolute.mean()),
            "medae": float(np.median(absolute)),
            "r": correlation,
            "amplitude": amplitude,
            "n_secondary": n_secondary,
        }
        if bootstrap.resamples:
            counts, present = block_counts(codes[picked], n_groups)
            sums = np.bincount(codes[picked], weights=error**2, minlength=n_groups)
            draws = rmse_replicates(
                bootstrap_multiplicities(
                    int(present.sum()), bootstrap, stream=column
                ),
                counts[present],
                sums[present],
            )
            entry["rmse_ci"] = [
                float(value) for value in np.percentile(draws, [2.5, 97.5])
            ]
        result[name] = entry
    return result


def compare(
    baseline,
    candidate,
    *,
    bootstrap: BootstrapSpec,
    columns=None,
) -> dict:
    """Return paired per-nucleus RMSE differences; positive values favor candidate."""
    if bootstrap.resamples < 1:
        raise ValueError("a paired comparison must carry a confidence interval")
    if baseline.shift_cols != candidate.shift_cols:
        raise ValueError("the arms carry different nucleus vocabularies; align() them first")
    if not np.array_equal(baseline.row_key, candidate.row_key):
        raise ValueError(
            "the arms do not describe the same rows in the same order; "
            "align() them first; a paired comparison needs the same residues"
        )
    cells = baseline.effective_mask & candidate.effective_mask
    disagreement = np.abs(baseline.true_ppm - candidate.true_ppm)[cells]
    if disagreement.size and disagreement.max() > LABEL_TOLERANCE_PPM:
        raise ValueError(
            "the arms disagree on the deposited labels by up to "
            f"{disagreement.max():.4g} ppm, so they are not reading one cache"
        )
    codes, n_groups = prediction_unit_codes(baseline, bootstrap.unit)
    result = {}
    for name in (columns or baseline.shift_cols):
        column = column_index(baseline, name)
        picked = np.flatnonzero(cells[:, column])
        if picked.size < MIN_SCORABLE_CELLS:
            continue
        error_a = baseline.pred_ppm[picked, column] - baseline.resolved_ppm[picked, column]
        error_b = candidate.pred_ppm[picked, column] - candidate.resolved_ppm[picked, column]
        rmse_a = float(np.sqrt((error_a**2).mean()))
        rmse_b = float(np.sqrt((error_b**2).mean()))
        counts, present = block_counts(codes[picked], n_groups)
        sums_a = np.bincount(codes[picked], weights=error_a**2, minlength=n_groups)
        sums_b = np.bincount(codes[picked], weights=error_b**2, minlength=n_groups)
        weights = bootstrap_multiplicities(
            int(present.sum()), bootstrap, stream=column
        )
        deltas = rmse_replicates(
            weights, counts[present], sums_a[present]
        ) - rmse_replicates(weights, counts[present], sums_b[present])
        low, high = (float(value) for value in np.percentile(deltas, [2.5, 97.5]))
        result[name] = {
            "n": int(picked.size),
            "rmse_baseline": rmse_a,
            "rmse_candidate": rmse_b,
            "mae_baseline": float(np.abs(error_a).mean()),
            "mae_candidate": float(np.abs(error_b).mean()),
            "delta_rmse": rmse_a - rmse_b,
            "ci": [low, high],
            "verdict": paired_verdict((low, high)),
        }
    return result
