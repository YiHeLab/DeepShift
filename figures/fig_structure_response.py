#!/usr/bin/env python3
"""Four tests of whether the predictor reads structure, on one \\textwidth canvas.

(a) Cysteine C-beta. A free thiol sits near 28 ppm and a disulfide near 41, and which
    mode a cysteine is in is not a sequence property -- it is whether a sulfur is 2 A
    away. A structure-reading model separates the modes; a residue-prior model collapses
    into the gap.
(b) Decoy discrimination. Rank 3DRobot candidates by shift agreement, keep the top k,
    and plot the mean C-alpha RMSD retained.
(c) Ring-current dose. Mean observed secondary shift against the computed ring-current
    contribution, and each predictor's response to the same field.
(d) The same ring-current test per site, one panel per predictor.

Nothing is recomputed here. Panels (a), (c) and (d) read this run's own fold-6 records:
``benchmarks/score_cys_modes.py`` places every cysteine by its coordinates and scores the
arms inside each state, and ``benchmarks/dump_ring_cells.py`` computes the ring-current
field at every aromatic 1H site and pairs it with each arm's value there. Panel (b) is the
cumulative mean C-alpha RMSD retained over the cohort the per-target roster names, and the
script refuses to draw it unless every arm ranked the same candidates of the same targets.

Two stems are emitted, and both are the deployed names -- ``fig_04_structure`` for the
main text and ``fig_si_decoy_r_vs_rmsd`` for the SI. The module is named for its topic
because a figure number in a filename rots the moment a float moves.

Usage:
    DEEPSHIFT_CYS_CELLS=... DEEPSHIFT_RING_CELLS=... DEEPSHIFT_DECOY_POOLED=... \\
        python figures/fig_structure_response.py
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import frozen
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import style

from benchmarks.dump_ring_cells import RING_COLUMNS
from benchmarks.master_cells import arms_returning
from deepshift.data.norm import RC_BACKBONE

# This run's own fold-6 records for bands (a), (c) and (d).
CYS = frozen.env_path("DEEPSHIFT_CYS_CELLS")
RING_CELLS = frozen.env_path("DEEPSHIFT_RING_CELLS")
# The decoy pool, unchanged: its arms are a separate GPU batch.
DECOY_POOL = frozen.env_path("DEEPSHIFT_DECOY_POOLED")
# The pool's per-candidate score: the equal-weight mean over the nuclei scored (H, HA, N)
# of the Pearson r between predicted and observed shifts.
DECOY_SCORE = "mean_r"
# The exclusion tier the paper reports, which benchmarks.dump_ring_cells defaults to.
RING_TIER = 2

# Random-coil 1H reference, Wishart et al., J. Biomol. NMR 5, 67 (1995). Observed and
# predicted must be referenced the same way; leaving predictions in raw ppm puts the two
# sides ~7 ppm apart and every scatter falls off axis.
WISHART_1995 = {
    ("H", "HD2"): 7.14, ("H", "HE1"): 8.12,
    ("F", "HD1"): 7.30, ("F", "HD2"): 7.30, ("F", "HE1"): 7.39, ("F", "HE2"): 7.39,
    ("F", "HZ"): 7.34,
    ("W", "HD1"): 7.24, ("W", "HE1"): 10.22, ("W", "HE3"): 7.65,
    ("W", "HZ2"): 7.50, ("W", "HZ3"): 7.17, ("W", "HH2"): 7.24,
    ("Y", "HD1"): 7.15, ("Y", "HD2"): 7.15, ("Y", "HE1"): 6.86, ("Y", "HE2"): 6.86,
}

#: Three-letter to one-letter for the four aromatic residues, to index the reference.
THREE_TO_ONE = {"PHE": "F", "TYR": "Y", "TRP": "W", "HIS": "H"}

# Page point sizes for the marginal band labels; style.py owns everything on the axes.
LETTER_PT, QUESTION_PT, NAME_PT, SMALL_PT = 11.0, 9.5, 9.5, 8.0

# Every arm the ring roster can put in the dump: the name the SI competitor tables print,
# and its house hue. Four hues is the palette's ceiling, so SHIFTX+ is the SHIFTX2 hue
# dashed and NMRGNN takes the neutral, which no other arm on these panels uses.
RING_ARM_STYLE = {
    "deepshift": ("DeepShift", style.ACCENT, "-"),
    "ucbshift2_sameprot": ("UCBShift 2.0", style.AMBER, "-"),
    "shiftx2_sameprot": ("SHIFTX2", style.VIOLET, "-"),
    "shiftx2_noshifty": ("SHIFTX+", style.VIOLET, "--"),
    "nmrgnn": ("NMRGNN", style.NEUTRAL, "-"),
}

# Decoy arms. No SHIFTX2 is on this panel, so violet is free for the MLFF baseline.
DECOY_ARMS = (("DeepShift", "DeepShift", style.ACCENT),
              ("UCBShift-X", "UCBShift-X", style.AMBER),
              ("MLFF-MACE", "MLFF", style.VIOLET))


def tidy(ax):
    """House chrome, plus the y grid these paired-value panels need."""
    style.finish(ax, spines=("left", "bottom"))
    ax.yaxis.grid(True, color=style.INK["grid"], linewidth=0.5)


def band_question(fig, y, letter, question):
    """One plain-language line per band: what this row of panels tests."""
    fig.text(0.008, y, letter, fontsize=LETTER_PT, fontweight="bold",
             color=style.INK["primary"], va="center", ha="left")
    fig.text(0.030, y, question, fontsize=QUESTION_PT, color=style.INK["primary"],
             va="center", ha="left")


def corner_label(ax, text, fontsize):
    """Name a panel inside its own frame; these scatters leave the corner empty."""
    ax.text(0.96, 0.05, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=fontsize, color=style.INK["primary"], linespacing=1.3,
            bbox={"facecolor": style.INK["surface"], "edgecolor": "none", "alpha": 0.82,
                  "boxstyle": "round,pad=0.18"})


# ------------------------------------------------------------------ band a: cysteine
def band_cysteine(fig, rects, cys_path, values):
    z = np.load(cys_path, allow_pickle=True)
    obs_raw = z["true_ppm"]
    # Secondary shift, as in bands (c) and (d): against the random-coil reference, here the
    # reduced cysteine's C-beta, so a free thiol sits near 0 and a disulfide near +13.
    centre = RC_BACKBONE["CYS"]["CB"]
    obs = obs_raw - centre
    # The dump labels the oxidised state "ss" (SG-SG within 2.3 A), not "disulfide".
    disulfide = z["ss_state"].astype(str) == "ss"
    arms = (("pred_deepshift", "DeepShift"),
            ("pred_sparta", "SPARTA+"),
            ("pred_ucbshift_full_sameprot", "UCBShift"),
            ("pred_shiftx2_sameprot", "SHIFTX2"))
    values["cysteine"] = {"n_cells": len(obs), "n_disulfide": int(disulfide.sum()),
                          "n_free_thiol": int((~disulfide).sum()),
                          "centre_ppm": centre,
                          "observed_mean_ppm": float(obs_raw.mean()), "arms": {}}
    for k, (col, name) in enumerate(arms):
        ax = fig.add_axes(rects[k])
        pred = z[col] - centre
        ok = np.isfinite(pred)
        # Hue encodes the disulfide state here, not a predictor, so it
        # avoids the accent: cyan means DeepShift everywhere else in the paper.
        for mask, colour, lab in ((~disulfide & ok, style.VIOLET, "free thiol"),
                                  (disulfide & ok, style.AMBER, "disulfide")):
            ax.scatter(obs[mask], pred[mask], s=3.0, color=colour, alpha=0.45,
                       linewidths=0, label=lab, rasterized=True)
        lim = (-6, 26)
        ax.plot(lim, lim, color=style.INK["primary"], lw=0.8, alpha=0.45, zorder=0)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xticks([0, 10, 20])
        ax.set_yticks([0, 10, 20])
        # Amplitude within the disulfide cluster is what separates the arms: pooled over
        # both modes the 13 ppm gap is most of the variance and every arm scores ~1.
        d = disulfide & ok
        values["cysteine"]["arms"][name] = {
            "amplitude_within_disulfides": float(np.std(pred[d]) / np.std(obs[d])),
            "rmse_ppm": float(np.sqrt(np.mean((pred[ok] - obs[ok]) ** 2)))}
        corner_label(ax, name, NAME_PT)
        ax.set_xlabel("observed (ppm)")
        if k == 0:
            ax.set_ylabel("predicted (ppm)")
            ax.legend(loc="upper left", markerscale=3.4, handletextpad=0.3,
                      borderaxespad=0.25, labelspacing=0.25, fontsize=SMALL_PT)
        else:
            ax.set_yticklabels([])
        tidy(ax)


# ------------------------------------------------------------------ band b: decoys
def decoy_cohort(metrics, candidates):
    """The targets panel (b) ranks, and the property of its inputs that makes the arms
    comparable.

    The per-target roster names the cohort; the candidate table carries one score per arm
    per candidate. A top-k curve is a comparison of arms only when every arm ranked the
    same candidates of the same targets, so an arm that is missing a target of the
    roster, or that ranked a different number of some target's candidates, stops the
    figure rather than averaging over two different candidate sets.
    """
    roster = metrics[(metrics["metric"] == DECOY_SCORE)
                     & (metrics["method"] == "DeepShift")]
    cohort = sorted(roster["target"].unique())
    ranked = {}
    for key, name, _colour in DECOY_ARMS:
        sub = candidates[(candidates["method"] == key)
                         & (candidates["target"].isin(cohort))]
        ranked[name] = sub.groupby("target").size().to_dict()
    absent = {name: sorted(set(cohort) - set(seen)) for name, seen in ranked.items()}
    absent = {name: gaps for name, gaps in absent.items() if gaps}
    if absent:
        raise SystemExit(f"the decoy arms do not cover the roster's cohort: {absent}")
    disagree = sorted(target for target in cohort
                      if len({seen[target] for seen in ranked.values()}) != 1)
    if disagree:
        raise SystemExit(
            f"{len(disagree)} targets were ranked over a different candidate set by "
            f"different arms, so their top-k curves are not comparable: {disagree[:5]}")
    return cohort, {target: next(iter(ranked.values()))[target] for target in cohort}


def panel_topk(ax, decoy_pool, values):
    df = pd.read_parquet(decoy_pool / "relaxed_full_candidate_scores.parquet")
    metrics = pd.read_csv(decoy_pool / "relaxed_full_target_metrics.csv")
    cohort, candidates_per_target = decoy_cohort(metrics, df)
    ks = np.arange(1, 101)
    values["decoy_topk"] = {"n_targets": len(cohort), "arms": {},
                            "candidates_per_target":
                                sorted(set(candidates_per_target.values()))}
    for key, name, colour in DECOY_ARMS:
        sub = df[(df["method"] == key) & (df["target"].isin(cohort))]
        rows = {t: np.cumsum(g.sort_values(DECOY_SCORE, ascending=False)["rmsd"]
                             .to_numpy()[:100]) / ks
                for t, g in sub.groupby("target")}
        curve = np.stack([rows[t] for t in sorted(rows)]).mean(axis=0)
        ax.plot(ks, curve, color=colour, lw=1.6, label=name)
        values["decoy_topk"]["arms"][name] = {
            "retained_rmsd_k1": float(curve[0]),
            "retained_rmsd_k10": float(curve[9]),
            "retained_rmsd_k100": float(curve[-1])}
    # Best possible: rank by true RMSD instead of by shift agreement.
    best = {}
    native = df[(df["method"] == "DeepShift") & (df["target"].isin(cohort))]
    for t, g in native.groupby("target"):
        best[t] = np.cumsum(np.sort(g["rmsd"].to_numpy())[:100]) / ks
    ax.plot(ks, np.stack([best[t] for t in sorted(best)]).mean(axis=0),
            color=style.INK["muted"], lw=1.2, ls="--", label="best possible")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel(r"top $k$ candidates kept")
    ax.set_ylabel(r"mean C$\alpha$ RMSD kept ($\AA$)")
    style.legend(ax, loc="upper left", handletextpad=0.5, labelspacing=0.3)
    tidy(ax)


# ------------------------------------------------------------------ bands c, d: rings
def ring_arms(found, roster):
    """The dump's arms as ``(name, column, colour, dash)``, in the roster's order.

    ``roster`` is this run's own answer to which arms return an aromatic 1H -- the
    coverage census read through ``benchmarks.master_cells.arms_returning``, which is the
    one roster ``benchmarks.dump_ring_cells`` itself dumps against. A dump whose arm list
    is not that roster was written before an arm joined the benchmark or after one left
    it, so these panels would compare a different set of predictors from the tables; that
    stops the figure rather than being drawn.
    """
    found = [str(arm) for arm in found]
    expected = ["deepshift", *roster]
    if found != expected:
        raise SystemExit(
            f"the ring dump carries the arms {found}, while the arms this run's coverage "
            f"census says return an aromatic 1H are {expected}: re-run "
            "benchmarks.dump_ring_cells")
    unstyled = [arm for arm in found if arm not in RING_ARM_STYLE]
    if unstyled:
        raise SystemExit(f"no plot style is declared for the ring arms {unstyled}")
    return [(RING_ARM_STYLE[arm][0], f"pred_{arm}", *RING_ARM_STYLE[arm][1:])
            for arm in found]


def ring_data(ring_path, arms):
    """Observed and predicted secondary shifts against the computed ring-current field.

    One-letter residue codes index the random-coil reference; the dump names residues by
    their three-letter code, which is what the structure carries.
    """
    z = np.load(ring_path, allow_pickle=True)
    residue = np.array([THREE_TO_ONE[name] for name in z["residue"].astype(str)])
    rc = np.array([WISHART_1995[(r, a)] for r, a in zip(residue, z["atom"].astype(str))])
    series = tuple((name, z[column] - rc, colour, dash)
                   for name, column, colour, dash in arms)
    return z["true_ppm"] - rc, z["ring_ppm"], series


def panel_dose(ax, obs, field, arms, values):
    """Mean secondary shift against the computed ring-current contribution.

    Binning, edges, sign convention and the 30-cell floor are the deployed builder's:
    x is the ring-current SHIFT, i.e. -ring_ppm, so shielding is negative, and each arm
    is binned over its OWN finite cells, since one NaN must not blank a bin.
    """
    contrib = -field
    edges = [-1.65, -1.0, -0.6, -0.3, -0.15, -0.05, 0.05, 0.15, 0.25]

    def bin_means(vals, mask):
        xs, ys = [], []
        for lo, hi in itertools.pairwise(edges):
            m = mask & (contrib > lo) & (contrib <= hi)
            if m.sum() < 30:
                continue
            xs.append(contrib[m].mean())
            ys.append(vals[m].mean())
        return np.array(xs), np.array(ys)

    ax.plot([-1.7, 0.3], [-1.7, 0.3], color=style.INK["faint"], lw=0.8, ls=":", zorder=1)
    ax.axhline(0, color=style.INK["grid"], lw=0.7, zorder=0)
    x0, y0 = bin_means(obs, np.isfinite(obs))
    ax.plot(x0, y0, "o-", color=style.INK["primary"], ms=4.6, lw=1.9, label="observed",
            zorder=5)
    values["ring_dose"] = {"edges": edges, "arms": {},
                           "observed_most_shielded_bin_ppm": float(y0[0])}
    for name, pred, colour, dash in arms:
        xs, ys = bin_means(pred, np.isfinite(pred))
        ax.plot(xs, ys, marker="o", ls=dash, color=colour, ms=3.2, lw=1.3, label=name,
                zorder=3)
        values["ring_dose"]["arms"][name] = {"most_shielded_bin_ppm": float(ys[0])}
    ax.set_xlabel("computed ring-current shift (ppm)")
    ax.set_ylabel("mean secondary shift (ppm)")
    # The curves run lower-left to upper-right, so with every arm listed the taller key
    # clears them only in the upper-left corner.
    style.legend(ax, loc="upper left", handletextpad=0.5, borderaxespad=0.35,
                 labelspacing=0.22, fontsize=SMALL_PT)
    tidy(ax)


def band_ring_scatter(fig, rects, obs, field, arms, values):
    hot = np.abs(field) >= 0.05
    values["ring_scatter"] = {"n_cells": int(hot.sum()), "arms": {}}
    for k, (name, pred, _colour, _dash) in enumerate(arms):
        ax = fig.add_axes(rects[k])
        x, y = obs[hot], pred[hot]
        ok = np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=1.4, color=style.INK["muted"], alpha=0.20,
                   linewidths=0, rasterized=True)
        lim = (-2.6, 1.6)
        ax.plot(lim, lim, color=style.INK["primary"], lw=0.8, alpha=0.55, zorder=3)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xticks([-2, -1, 0, 1])
        ax.set_yticks([-2, -1, 0, 1])
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        slope = float(np.polyfit(x[ok], y[ok], 1)[0])
        values["ring_scatter"]["arms"][name] = {"R": r, "slope": slope}
        corner_label(ax, f"{name}\n$R$ {r:.2f} · slope {slope:.2f}", SMALL_PT)
        ax.set_xlabel("observed (ppm)")
        if k == 0:
            ax.set_ylabel("predicted (ppm)")
        else:
            ax.set_yticklabels([])
        tidy(ax)


# --------------------------------------------------------- the SI's own decoy panel
SLOPE_RESAMPLES = 200_000
SLOPE_SEED = 20260814


def paired_slope_difference(df, left, right):
    """The pooled least-squares slope of shift agreement on RMSD, left minus right, with
    an interval from resampling the targets with replacement (both arms take the same
    draw). Each target enters through its sums, so a draw refits in closed form."""
    targets = sorted(df["target"].unique())
    sums = {}
    for arm in (left, right):
        sub = df[df["method"] == arm]
        g = sub.assign(xy=sub["rmsd"] * sub[DECOY_SCORE], xx=sub["rmsd"] ** 2).groupby(
            "target")[["rmsd", DECOY_SCORE, "xy", "xx"]].agg(["sum", "count"])
        sums[arm] = np.stack(
            [g.loc[targets, (c, f)].to_numpy(float) for c, f in
             (("rmsd", "count"), ("rmsd", "sum"), (DECOY_SCORE, "sum"), ("xy", "sum"),
              ("xx", "sum"))], axis=1)

    def slope(weights, table):
        n, sx, sy, sxy, sxx = (weights @ table).T
        return (n * sxy - sx * sy) / (n * sxx - sx ** 2)

    rng = np.random.default_rng(SLOPE_SEED)
    draws = rng.integers(0, len(targets), size=(SLOPE_RESAMPLES, len(targets)))
    weights = np.stack([np.bincount(d, minlength=len(targets)) for d in draws]).astype(float)
    one = np.ones((1, len(targets)))
    point = float(slope(one, sums[left])[0] - slope(one, sums[right])[0])
    boot = slope(weights, sums[left]) - slope(weights, sums[right])
    lo, hi = (float(v) for v in np.quantile(boot, [0.025, 0.975]))
    return {"arms": f"{left} minus {right}", "difference_per_angstrom": point,
            "ci95": [lo, hi], "two_sided_p": float(min(1.0, 2 * min((boot >= 0).mean(),
                                                                       (boot <= 0).mean()))),
            "resamples": SLOPE_RESAMPLES, "seed": SLOPE_SEED, "unit": "target"}


def si_decoy_scatter(decoy_pool, values):
    """Shift agreement against RMSD-to-native, per candidate."""
    df = pd.read_parquet(decoy_pool / "relaxed_full_candidate_scores.parquet")
    # 0.72\linewidth in the SI is 4.68 in, so this canvas is built at its placed width
    # exactly as the main figure is at \textwidth.
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    fig.set_layout_engine("none")
    values["si_decoy_scatter"] = {"n_candidates_total": len(df), "arms": {}}
    fits = []
    for key, name, colour in DECOY_ARMS:
        sub = df[df["method"] == key]
        ax.scatter(sub["rmsd"], sub[DECOY_SCORE], s=1.4, color=colour, alpha=0.30,
                   linewidths=0, label=name, rasterized=True)
        slope, intercept = (float(v) for v in np.polyfit(sub["rmsd"], sub[DECOY_SCORE], 1))
        fits.append((colour, slope, intercept))
        values["si_decoy_scatter"]["arms"][name] = {
            "n_candidates": len(sub), "slope_r_vs_rmsd": slope,
            "intercept_r_at_rmsd_0": intercept}
    values["si_decoy_scatter"]["slope_difference"] = paired_slope_difference(
        df, "DeepShift", "UCBShift-X")
    span = np.array([df["rmsd"].min(), df["rmsd"].max()])
    for colour, slope, intercept in fits:
        # A surface-coloured underlay, not a contrasting border, where the fits overlap.
        ax.plot(span, slope * span + intercept, color=style.INK["surface"], lw=2.8,
                zorder=3)
        ax.plot(span, slope * span + intercept, color=colour, lw=1.5, zorder=4)
    ax.set_xlabel(r"C$\alpha$ RMSD to native structure ($\AA$)")
    ax.set_ylabel("shift-agreement correlation")
    leg = style.legend(ax, loc="upper right", markerscale=5, handletextpad=0.4,
                       borderaxespad=0.3)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    tidy(ax)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cys-cells", type=Path, default=CYS)
    ap.add_argument("--ring-cells", type=Path, default=RING_CELLS)
    ap.add_argument("--decoy-pool", type=Path, default=DECOY_POOL,
                    help="the decoy pool directory, read unchanged")
    args = ap.parse_args()
    style.use_house_style()
    values = {}

    # Explicit positions in figure fractions, so the layout engine must stand down.
    # Panel names sit inside the frames, so a band needs only its question and its axes.
    fig = plt.figure(figsize=(style.WIDTH["text"], 6.9))
    fig.set_layout_engine("none")
    left, width, gutter = 0.105, 0.878, 0.012
    def band(bottom, height, columns=4):
        # 4 panels are 1.37 in wide, the 5 of a band with every ring arm 1.08 in.
        cell = (width - (columns - 1) * gutter) / columns
        return [[left + k * (cell + gutter), bottom, cell, height]
                for k in range(columns)]

    band_question(fig, 0.978, "a",
                  "Does the model see the disulfide, or guess the residue average?")
    band_cysteine(fig, band(0.768, 0.192), args.cys_cells, values)

    band_question(fig, 0.698, "b",
                  "Can it pick the true structure out of 300 decoys?")
    panel_topk(fig.add_axes([left, 0.400, 0.355, 0.265]), args.decoy_pool, values)

    fig.text(0.520, 0.698, "c", fontsize=LETTER_PT, fontweight="bold",
             color=style.INK["primary"], va="center")
    fig.text(0.542, 0.698, "Does it respond like the experiment?",
             fontsize=QUESTION_PT, color=style.INK["primary"], va="center")
    with np.load(args.ring_cells) as dump:
        dumped_arms = dump["arms"].tolist()
    arms = ring_arms(dumped_arms, arms_returning(*RING_COLUMNS, tiers=(RING_TIER,)))
    obs, field, series = ring_data(args.ring_cells, arms)
    panel_dose(fig.add_axes([0.628, 0.400, 0.355, 0.265]), obs, field, series, values)

    band_question(fig, 0.322, "d",
                  "Per-site ring-current response, one panel per predictor")
    band_ring_scatter(fig, band(0.088, 0.198, len(series)), obs, field, series, values)

    frozen.emit(fig, "fig_04_structure")
    plt.close(fig)

    si = si_decoy_scatter(args.decoy_pool, values)
    frozen.emit(si, "fig_si_decoy_r_vs_rmsd")
    plt.close(si)

    print(json.dumps({"decoy_topk": values["decoy_topk"]}, indent=1))


if __name__ == "__main__":
    main()
