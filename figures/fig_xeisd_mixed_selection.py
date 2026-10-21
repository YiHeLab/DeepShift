#!/usr/bin/env python3
"""What shift-only X-EISD selection keeps from the MIXED pool, and what it buys.

X-EISD is run on the 50:50 MIXED pool of the unfolded drkN SH3 domain using CHEMICAL
SHIFTS ALONE, 1,000 independent restarts per forward model, changing only the shift
predictor.

(a) Selection. How much of the experimentally refined ENSEMBLE sub-pool each forward
    model recovers. SHIFTX2 recovers 53.8%, barely above the 50% it started from;
    DeepShift recovers 82.0%, past the 76% the source paper reached using all eight
    observables jointly.
(b) Validation. The paired change in the total X-EISD score of the SEVEN data types the
    selection never saw (SAXS, FRET, J-coupling, NOE, PRE, RDC, Rh) when the shift table
    is swapped SHIFTX2 -> DeepShift.
(c) The same paired change split by data type.

Every number is read from the shipped result JSONs and asserted against them; nothing is
recomputed. ``fig_xeisd_composition.py`` draws the composition alone from the
``optimize_cs`` indices; this figure draws the ``rescore_all`` indices, where the paired
held-out scores live, so the two panels of one comparison come from one run.

The stem is the DEPLOYED name, ``fig_05_xeisd_mixed``; the module is named for its topic
because a figure number in a filename rots the moment a float moves.

Usage:
    DEEPSHIFT_XEISD_PROTOCOL=... python figures/fig_xeisd_mixed_selection.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import frozen
import matplotlib.pyplot as plt
import numpy as np
import style

PROTOCOL = frozen.env_path("DEEPSHIFT_XEISD_PROTOCOL")
RESULTS = PROTOCOL / "results.json"
COMPOSITION = PROTOCOL / "mixed_composition.json"
RUNS = PROTOCOL / "runs"
MODELS = (("shiftx2", "SHIFTX2", style.VIOLET), ("deepshift", "DeepShift", style.ACCENT))

# results.csv has no header; columns from analyze_independent_restarts.py:26.
SCORE_COLUMNS = {"saxs": 2, "cs": 3, "fret": 4, "jc": 5, "noe": 6, "pre": 7,
                 "rdc": 8, "rh": 9}
NON_CS = ("saxs", "fret", "jc", "noe", "pre", "rdc", "rh")
TYPE_LABEL = {"saxs": "SAXS", "fret": "FRET", "jc": "J-coupling", "noe": "NOE",
              "pre": "PRE", "rdc": "RDC", "rh": r"$R_h$"}

# MIXED is 1,700 conformers from the optimized ENSEMBLE pool followed by 1,700 from the
# optimized RANDOM pool, so a selected index below 1,700 came from ENSEMBLE.
ENSEMBLE_BLOCK = 1700

BY_TYPE_SEED, BY_TYPE_BOOTS = 20260728, 2000
COMPOSITION_TOLERANCE, HELD_OUT_TOLERANCE = 1e-4, 0.05

# Page point sizes for the marginal panel labels; style.py owns everything on the axes.
LETTER_PT, QUESTION_PT, SMALL_PT = 11.0, 9.5, 8.0


def tidy(ax):
    """House chrome, plus the y grid these paired-value panels need."""
    style.finish(ax, spines=("left", "bottom"))
    ax.yaxis.grid(True, color=style.INK["grid"], linewidth=0.5)


def label(fig, x, y, letter, question):
    fig.text(x, y, letter, fontsize=LETTER_PT, fontweight="bold",
             color=style.INK["primary"], va="center", ha="left")
    fig.text(x + 0.022, y, question, fontsize=QUESTION_PT,
             color=style.INK["primary"], va="center", ha="left")


def results_csv(model: str) -> Path:
    return RUNS / "mixed" / model / "rescore_all/results.csv"


def indices_csv(model: str) -> Path:
    return RUNS / "mixed" / model / "rescore_all/indices.csv"


def load_pairs() -> dict:
    """Per-restart scores, one row per restart, both models. Paired by row index."""
    return {model: np.loadtxt(results_csv(model), delimiter=",", ndmin=2)
            for model, _, _ in MODELS}


def load_composition() -> dict:
    """Per-restart fraction of the selected 100 conformers drawn from ENSEMBLE."""
    return {model: (np.loadtxt(indices_csv(model), delimiter=",")
                    < ENSEMBLE_BLOCK).mean(axis=1)
            for model, _, _ in MODELS}


def panel_composition(ax, frames, shipped_composition, values):
    """Where the selected conformers came from, 1,000 restarts per model."""
    for model, name, colour in MODELS:
        fracs = frames[model]
        want = shipped_composition["models"][model]["mean_ensemble_fraction"]
        if abs(float(fracs.mean()) - want) > COMPOSITION_TOLERANCE:
            raise SystemExit(f"STOP: {model} composition {fracs.mean()} != shipped {want}")
        ax.hist(fracs, bins=np.arange(0.42, 0.92, 0.012), color=colour, alpha=0.9,
                label=name, linewidth=0)
        values.setdefault("composition", {})[name] = {
            "mean_ensemble_fraction": float(fracs.mean()),
            "shipped_mean_ensemble_fraction": want,
            "n_restarts": len(fracs)}
    ax.axvline(0.50, color=style.INK["muted"], lw=0.9, ls=":")
    # Below the bars, not across them: the SHIFTX2 histogram peaks right at this line.
    ax.text(0.497, ax.get_ylim()[1] * 0.40, "starting\npool 50% ", fontsize=SMALL_PT,
            color=style.INK["muted"], ha="right", va="center", linespacing=1.25)
    ax.axvline(0.76, color=style.INK["primary"], lw=1.1, ls="--", alpha=0.75)
    ax.text(0.757, ax.get_ylim()[1] * 0.96,
            "published, all eight\nobservables jointly 76% ", fontsize=SMALL_PT,
            color=style.INK["primary"], ha="right", va="top", linespacing=1.25)
    ax.set_xlabel("ENSEMBLE fraction of the optimized 100-conformer set")
    ax.set_ylabel("restarts")
    style.legend(ax, loc="upper left", handletextpad=0.5, labelspacing=0.25)
    tidy(ax)


def panel_heldout(ax, pairs, shipped, values):
    """Paired change in the seven held-out data types, DeepShift minus SHIFTX2."""
    cols = [SCORE_COLUMNS[t] for t in NON_CS]
    delta = pairs["deepshift"][:, cols].sum(1) - pairs["shiftx2"][:, cols].sum(1)
    rng = np.random.default_rng(shipped["definition"]["bootstrap_seed"])
    n = shipped["definition"]["bootstrap_samples"]
    boot = np.array([delta[rng.integers(0, len(delta), len(delta))].mean()
                     for _ in range(n)])
    redraw = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    ref = shipped["pools"]["MIXED"]["primary_non_cs_score"]
    # The interval PRINTED is the protocol's own (results.json), the one the text quotes; the
    # redraw here is a check on it, not a second interval for the reader.
    lo, hi = float(ref["ci95_low"]), float(ref["ci95_high"])
    if not (lo <= delta.mean() <= hi) or abs(redraw[0] - lo) > 1.0 or abs(redraw[1] - hi) > 1.0:
        raise SystemExit(f"STOP: shipped MIXED CI [{lo}, {hi}] is not reproduced by the "
                         f"redraw {redraw} about mean {delta.mean()}")
    values["held_out"] = {"mean": float(delta.mean()), "ci95": [lo, hi],
                          "ci95_source": "results.json pools.MIXED.primary_non_cs_score",
                          "redraw_ci95": redraw,
                          "fraction_improved": float((delta > 0).mean()),
                          "n_pairs": len(delta),
                          "shipped_mean": ref["mean_improvement"],
                          "shipped_ci95": [ref["ci95_low"], ref["ci95_high"]]}
    # The jitter draws from the same generator as the bootstrap, so the cloud is
    # reproducible only in that order.
    jitter = rng.normal(0, 0.085, len(delta))
    ax.scatter(jitter, delta, s=3.0, color=style.INK["faint"], alpha=0.45, linewidths=0,
               rasterized=True)
    ax.errorbar([0.42], [delta.mean()], yerr=[[delta.mean() - lo], [hi - delta.mean()]],
                fmt="o", color=style.ACCENT, ms=5.5, lw=1.8, capsize=3, zorder=5)
    ax.axhline(0, color=style.INK["primary"], lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(-0.42, 0.72)
    ax.set_xticks([])
    ax.set_ylabel("change in held-out score,\nall seven data types", linespacing=1.3)
    ax.text(0.02, 0.97, f"+{delta.mean():.1f}  [{lo:+.1f}, {hi:+.1f}]\n"
                        f"{(delta > 0).mean() * 100:.0f}% of restarts improved",
            transform=ax.transAxes, fontsize=SMALL_PT, va="top",
            color=style.INK["primary"], linespacing=1.35)
    tidy(ax)


def panel_by_type(ax, pairs, values):
    rng = np.random.default_rng(BY_TYPE_SEED)
    values["by_type"] = {}
    ys = np.arange(len(NON_CS))
    for y, t in zip(ys, NON_CS):
        d = pairs["deepshift"][:, SCORE_COLUMNS[t]] - pairs["shiftx2"][:, SCORE_COLUMNS[t]]
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean()
                         for _ in range(BY_TYPE_BOOTS)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # Direction, not identity: the accent marks data types the swap improves, amber
        # the ones it costs, slate the ones whose interval crosses zero.
        colour = style.ACCENT if d.mean() > 0 else style.AMBER
        if lo <= 0 <= hi:
            colour = style.INK["muted"]
        ax.errorbar([d.mean()], [y], xerr=[[d.mean() - lo], [hi - d.mean()]],
                    fmt="o", color=colour, ms=4.6, lw=1.5, capsize=2.4)
        values["by_type"][t] = {"mean": float(d.mean()), "ci95": [float(lo), float(hi)]}
    ax.axvline(0, color=style.INK["primary"], lw=0.9, ls="--", alpha=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([TYPE_LABEL[t] for t in NON_CS])
    ax.set_ylim(len(NON_CS) - 0.4, -0.6)
    ax.set_xlabel("change in X-EISD score")
    tidy(ax)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    style.use_house_style()
    shipped = json.loads(RESULTS.read_text())
    shipped_composition = json.loads(COMPOSITION.read_text())
    pairs, frames = load_pairs(), load_composition()
    values = {"pool": "MIXED (50:50), chemical shifts alone, 1,000 paired restarts"}

    # Explicit positions in figure fractions, so the layout engine must stand down.
    fig = plt.figure(figsize=(style.WIDTH["text"], 5.55))
    fig.set_layout_engine("none")
    label(fig, 0.008, 0.968, "a", "Which conformers does shift-only selection keep?")
    panel_composition(fig.add_axes([0.105, 0.605, 0.865, 0.300]), frames,
                      shipped_composition, values)

    label(fig, 0.008, 0.508, "b", "Does that agree better with data it never saw?")
    panel_heldout(fig.add_axes([0.105, 0.085, 0.300, 0.360]), pairs, shipped, values)

    label(fig, 0.470, 0.508, "c", "Where the gain comes from")
    panel_by_type(fig.add_axes([0.615, 0.085, 0.355, 0.360]), pairs, values)

    # ---- gate: the held-out mean must reproduce the shipped summary
    got, want = values["held_out"]["mean"], values["held_out"]["shipped_mean"]
    values["gate_vs_shipped"] = {"difference": abs(got - want),
                                 "tolerance": HELD_OUT_TOLERANCE,
                                 "reproduced": bool(abs(got - want) <= HELD_OUT_TOLERANCE)}
    if abs(got - want) > HELD_OUT_TOLERANCE:
        raise SystemExit(f"STOP: held-out mean {got} != shipped {want}")

    frozen.emit(fig, "fig_05_xeisd_mixed")
    print(json.dumps({k: values[k] for k in
                      ("gate_vs_shipped", "held_out", "composition")}, indent=1))


if __name__ == "__main__":
    main()
