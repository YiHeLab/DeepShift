#!/usr/bin/env python3
"""Does the ensemble know when it is wrong? Spread against realised error.

The five cross-validation members are trained on different fold subsets, so on a fold
outside every member's training set the five predictions for one cell are five
independent opinions. Their standard deviation is the only uncertainty signal the
shipped model emits. This figure asks whether that signal is usable, in two halves that
a single number would blur:

  (a) **Realised error against the fitted mapping.** Cells are put into equal-count
      spread bins per nucleus -- the package's own partition, the one the mapping was
      fitted over -- and each bin's realised RMSE is scored with a clustered interval.
      Beside it runs the variance mapping fitted on fold 0 and read from the interval
      calibration artifact, so the panel shows how far the reserved fold's error sits
      from what the frozen mapping predicts, and how far both sit from the identity
      line where spread would be an error bar. Log-log axes carry the ranking skill in
      the slope and the miscalibration in the vertical distance; each nucleus keeps its
      own ppm and none is averaged with another.

  (b) **What the ranking skill is worth, against a baseline.** Keeping the cells whose
      spread is below the cohort median cuts RMSE by a ratio reported per nucleus. The
      comparison is against a ranking that costs nothing -- residue type alone, fitted
      in sample on this cohort, which favours it -- and against the oracle ranking on
      the true error, which is the ceiling. Ratios are within nucleus and therefore
      dimensionless; no ppm is ever pooled.

The mapping and every conformal multiplier are fitted on fold 0 by ``deepshift intervals
fit``. This figure fits nothing: it reads that artifact and plots the measured bins
against it.

Backbone only: prochiral ambiguity would let two members disagree merely about which of
HB2/HB3 they wrote in, which is a labelling artefact, not an uncertainty. Panel (b) uses
a fixed global threshold (the cohort median spread) so the bootstrap resamples proteins
instead of re-deriving a rank inside every draw.

Usage:
    DEEPSHIFT_RUN=<finished run> python figures/fig_uncertainty_calibration.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import frozen
import numpy as np
import style

from deepshift.calibration import (
    EVALUATION_FOLD,
    MEMBER_FOLDS,
    equal_count_bins,
    error_scale,
    load_calibration,
)
from deepshift.calibration import (
    member_spread as spread_across_members,
)
from deepshift.chemistry import atom_from_shift_column
from deepshift.statistics import DEFAULT_SEED

# One hue means one thing across both panels: blue is the ensemble-spread signal.
# Panel (a) is therefore monochrome -- its six curves are separated by position and
# carry their own labels, so a second colour scheme (by isotope) would only compete
# with panel (b), where the three hues have to carry the three ranking arms.
SPREAD = style.PALETTE["slot1"]

#: Panel (a): equal-count bins in ensemble spread, per nucleus.
N_SPREAD_BINS = 10
#: Panel (b): the share of cells each ranking retains.
RETAIN_FRACTION = 0.5


# --------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------

def ratio_ci(err, keep, keys, *, n_boot=2000, seed=DEFAULT_SEED):
    """RMSE(kept) / RMSE(all) with a 95% interval, resampling whole label sets.

    Numerator and denominator move together inside a resample, which is the point: the
    quantity claimed is a *ratio* within one cohort, and bootstrapping the two ends
    separately would widen it with variation that cancels. That coupling is why this is
    not a call to ``evaluate.score`` -- the blocks are the same label sets that module
    resamples, and a resample is a sum over per-block sums, so the interval is exact
    for the draw, never approximated.
    """
    err = np.asarray(err, float) ** 2
    keep = np.asarray(keep, bool)
    _, code = np.unique(np.asarray(keys), return_inverse=True)
    code = np.asarray(code).ravel()
    n_blocks = int(code.max()) + 1
    all_sq = np.bincount(code, weights=err, minlength=n_blocks)
    all_n = np.bincount(code, minlength=n_blocks).astype(float)
    kept_sq = np.bincount(code, weights=err * keep, minlength=n_blocks)
    kept_n = np.bincount(code, weights=keep.astype(float), minlength=n_blocks)

    def ratio(w):
        return np.sqrt(w @ kept_sq / max(w @ kept_n, 1.0)) / np.sqrt(w @ all_sq / (w @ all_n))

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=n_boot)
    draws = np.array([ratio(w) for w in weights.astype(float)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"ratio": float(ratio(np.ones(n_blocks))), "lo": float(lo), "hi": float(hi),
            "rmse_all": float(np.sqrt(all_sq.sum() / all_n.sum())),
            "rmse_kept": float(np.sqrt(kept_sq.sum() / kept_n.sum())),
            "n_kept": int(keep.sum()), "n": int(keep.size)}


def calibration_bins(preds, col, spread, mapping, *, n_bins=10, boots=1000,
                     seed=DEFAULT_SEED):
    """Realised RMSE per equal-count spread bin, with a clustered CI, beside the ppm the
    fitted mapping predicts for that bin.

    The bins are the package's own partition (``calibration.equal_count_bins``), the one
    the mapping was fitted over, and each bin is scored by the canonical evaluator
    (``evaluate.score`` restricted to those rows and that column). ``mapped_rmse`` is the
    fold-0 mapping applied to the bin's mean spread; nothing here is fitted.
    """
    from deepshift import evaluate

    j = preds.shift_cols.index(col)
    scorable = np.flatnonzero(preds.effective_mask[:, j])
    rows = []
    for index, cells in enumerate(equal_count_bins(spread[scorable], n_bins=n_bins)):
        selected = scorable[cells]
        here = np.zeros(preds.n_rows, dtype=bool)
        here[selected] = True
        entry = evaluate.score(preds, columns=[col], rows=here, boots=boots,
                               seed=seed + index)[col]
        mean_spread = float(spread[selected].mean())
        rows.append({"bin": index + 1, "sd_mean": mean_spread,
                     "sd_lo": float(spread[selected].min()),
                     "sd_hi": float(spread[selected].max()),
                     "rmse": entry["rmse"], "mae": entry["mae"],
                     "rmse_lo": entry["rmse_ci"][0], "rmse_hi": entry["rmse_ci"][1],
                     "mapped_rmse": float(error_scale([mean_spread], **mapping)[0]),
                     "n": entry["n"], "n_label_sets": entry["n_label_sets"]})
    return rows


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", type=Path, default=None,
                    help="directory holding the members' prediction dumps")
    ap.add_argument("--calibration", type=Path, default=frozen.INTERVAL_CALIBRATION,
                    help="the run's fold-0 interval calibration artifact")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    fitted = load_calibration(args.calibration)
    paths = frozen.member_dumps(EVALUATION_FOLD, MEMBER_FOLDS, directory=args.dumps)

    from deepshift import evaluate

    parts = [evaluate.read_predictions(path) for path in paths]
    sd = spread_across_members(np.stack([part.pred_ppm for part in parts]))
    preds = evaluate.ensemble(parts)
    # the same fence the benchmark tables use
    preds, _ = frozen.drop_anticorrelated(preds)
    cols, eff = list(preds.shift_cols), preds.effective_mask
    err_all = preds.pred_ppm - preds.resolved_ppm
    aa = np.asarray(preds.residue_code, dtype=int)

    scores = {}
    for col in style.BACKBONE:
        j = cols.index(col)
        constants = fitted.constants[atom_from_shift_column(col)]
        mapping = {"floor_ppm": constants["floor_ppm"], "gain": constants["gain"]}
        bins = calibration_bins(preds, col, sd[:, j], mapping, n_bins=N_SPREAD_BINS,
                                boots=args.boot // 2, seed=args.seed)

        rows = np.flatnonzero(eff[:, j])
        err, spread = err_all[rows, j], sd[rows, j]
        keys, types = preds.label_set[rows], aa[rows]
        # Baseline: rank by residue type alone, using this cohort's own per-type RMSE.
        # It is fitted in sample, which favours the baseline over the spread ranking.
        by_type = np.zeros(types.size)
        for t in np.unique(types):
            here = types == t
            by_type[here] = np.sqrt((err[here] ** 2).mean())

        keep = {}
        for name, rank in (("spread", spread), ("residue_type", by_type),
                           ("oracle", np.abs(err))):
            cut = np.quantile(rank, RETAIN_FRACTION)
            keep[name] = ratio_ci(err, rank <= cut, keys,
                                  n_boot=args.boot, seed=args.seed)
        scores[col] = {"bins": bins, "fit": mapping, "retain": keep,
                       "n": int(rows.size), "rmse": keep["spread"]["rmse_all"],
                       "spread_median": float(np.median(spread)),
                       "decile_ratio": bins[-1]["rmse"] / bins[0]["rmse"]}

    fig = draw(scores, retain=RETAIN_FRACTION)
    stem = f"fig_uncertainty_calibration_fold{EVALUATION_FOLD}"
    frozen.emit(fig, stem)
    print(json.dumps(scores, indent=1))


# --------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------

# Hand-placed: CA and CB track each other closely, so their curves interleave and only a
# leader line can say which label belongs to which.
# ``at`` picks the bin the label points at, ``to`` is a data-coordinate parking spot.
LABEL_SPEC = {
    "n_shift":  {"at": -1, "dx": 5, "dy": 0},
    "h_shift":  {"at": -1, "dx": 5, "dy": -1},
    "ha_shift": {"at": -1, "dx": 5, "dy": -3},
    "c_shift":  {"at": 0, "dx": -4, "dy": 4, "ha": "right"},
    "ca_shift": {"at": -1, "to": (0.98, 0.70)},
    "cb_shift": {"at": -1, "to": (1.15, 1.85)},
}


def draw(scores: dict, *, retain: float):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    style.use_house_style()
    fig = plt.figure(figsize=(style.WIDTH["wide"], 3.0))
    ax_a, ax_b = fig.subplots(1, 2, width_ratios=[1.06, 1.0])

    # ---- (a) calibration on log-log axes ----------------------------------------
    (x0, x1), (y0, y1) = (0.008, 2.6), (0.09, 4.2)
    ax_a.plot([x0, y1], [x0, y1], color=style.INK["axis"], linewidth=0.6, zorder=1)

    for col in style.BACKBONE:
        rows = scores[col]["bins"]
        x = np.array([r["sd_mean"] for r in rows])
        y = np.array([r["rmse"] for r in rows])
        ax_a.vlines(x, [r["rmse_lo"] for r in rows], [r["rmse_hi"] for r in rows],
                    color=SPREAD, linewidth=style.LINE, alpha=0.9, zorder=2)
        ax_a.plot(x, [r["mapped_rmse"] for r in rows], color=style.INK["faint"],
                  linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
        ax_a.plot(x, y, color=SPREAD, linewidth=0.9, marker="o",
                  markersize=style.MARKER - 1.4, markeredgecolor=style.INK["surface"],
                  markeredgewidth=style.RING, zorder=3)
        spec = dict(LABEL_SPEC[col])
        at = spec.pop("at")
        text, park = style.nucleus_label(col), spec.pop("to", None)
        if park is None:
            style.direct_label(ax_a, x[at], y[at], text, color=SPREAD, **spec)
        else:
            ax_a.annotate(text, (x[at], y[at]), xytext=park, textcoords="data",
                          fontsize=6.6, color=SPREAD, va="center", **spec,
                          arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                                      "color": style.INK["axis"], "shrinkA": 2,
                                      "shrinkB": 3})

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(x0, x1)
    ax_a.set_ylim(y0, y1)
    ax_a.set_xlabel("ensemble spread within decile (ppm)")
    ax_a.set_ylabel("realised RMSE within decile (ppm)")
    decade = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax_a.xaxis, ax_a.yaxis):
        axis.set_major_formatter(decade)
        axis.set_minor_locator(LogLocator(subs=(3,)))     # label the half-decades too
        axis.set_minor_formatter(decade)
    ax_a.tick_params(which="minor", length=1.5, labelsize=7.0)
    style.finish(ax_a, xgrid=False, spines=("bottom", "left"))
    ax_a.grid(True, which="major", color=style.INK["grid"], linewidth=0.5)
    ax_a.set_axisbelow(True)
    # The diagonal is labelled at its own on-screen angle and not a guessed one:
    # the panel is not square in decades, so 45 degrees would visibly miss.
    p0, p1 = ax_a.transData.transform([(0.3, 0.3), (1.0, 1.0)])
    ax_a.text(1.5, 1.42, "  spread = error", fontsize=6.2, color=style.INK["muted"],
              ha="left", va="top", rotation=np.degrees(np.arctan2(*(p1 - p0)[::-1])),
              rotation_mode="anchor")
    style.panel_label(ax_a, "(a)", x=-0.16)
    ax_a.set_title("realised error, and the\nfold-0 mapping (dashed)", loc="left", pad=3,
                   color=style.INK["primary"])

    # ---- (b) what the ranking buys ----------------------------------------------
    order = sorted(style.BACKBONE, key=lambda c: scores[c]["retain"]["spread"]["ratio"])
    arms = ("spread", "residue_type", "oracle")
    names = ("ensemble spread", "residue type (baseline)", "oracle (true error)")
    series = []
    for arm, name in zip(arms, names):
        vals = [scores[c]["retain"][arm] for c in order]
        series.append({"name": name,
                       "values": [v["ratio"] for v in vals],
                       "lo": [v["lo"] for v in vals], "hi": [v["hi"] for v in vals]})
    ax_b.axvline(1.0, color=style.INK["axis"], linewidth=0.6, zorder=1)
    style.forest(
        ax_b, [style.nucleus_label(c) for c in order], series,
        xlabel=f"RMSE of the retained {retain:.0%} $\\div$ RMSE of all cells",
        value_column=[f"{scores[c]['rmse']:.3f}  {scores[c]['n']:>7,}" for c in order],
        value_header="ppm      cells", xlim=(0, 1.12))
    ax_b.text(1.0, len(order) - 0.62, "no skill", ha="center", va="top", fontsize=6.2,
              color=style.INK["muted"])
    style.panel_label(ax_b, "(b)", x=-0.13)
    ax_b.set_title("keeping the confident half", loc="left", pad=3,
                   color=style.INK["primary"])

    handles, labels = ax_b.get_legend_handles_labels()
    style.legend(fig, handles=handles, labels=labels, loc="outside lower center",
                 ncol=3, columnspacing=1.4)
    return fig


if __name__ == "__main__":
    main()
