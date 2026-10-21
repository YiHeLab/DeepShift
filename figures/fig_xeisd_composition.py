#!/usr/bin/env python3
"""X-EISD MIXED-pool composition selected by chemical shifts alone.

The published X-EISD analysis starts from a 50:50 MIXED pool of PED-derived (ENSEMBLE) and
RANDOM conformers and reports that joint optimization over ALL EIGHT observables moves it
to 76% ENSEMBLE (Lincoff et al., Commun Chem 3, 74 (2020), Results, "X-EISD applied to the
unfolded state of the drkN SH3 domain"). That 76% is a sentence in the text, not a figure,
and it is an all-data-selected reference, not a known population.

Our arm changes exactly one thing -- the chemical-shift forward model -- and optimizes on
chemical shifts ALONE, so the other seven observables stay out of the objective. The
question the panel answers is therefore: which CS predictor, on its own, moves the pool
where all eight observables jointly moved it?

Each restart is an independent 100-conformer optimization; there are 1,000 per model, and
the histogram is over restarts, not over conformers. Nothing is recomputed here beyond the
per-restart ENSEMBLE fraction, which is read straight off the saved index matrices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import frozen
import matplotlib.pyplot as plt
import numpy as np
import style

# The X-EISD protocol run, named by environment with no fallback; see REPRODUCTION.md.
RUNS = frozen.env_path("DEEPSHIFT_XEISD_PROTOCOL", "runs")
# The MIXED pool is 3,400 conformers: rows 0..1699 are ENSEMBLE, 1700..3399 are RANDOM.
# assemble_mixed_cs.py verified this against the paper's own supplied MIXED tables.
N_ENSEMBLE = 1_700
PAPER_FRACTION = 0.76
N_BOOT = 2000
SEED = 20260727
ARMS = (("shiftx2", "SHIFTX2", "slot2"), ("deepshift", "DeepShift", "slot1"))


def fractions(runs: Path, model: str) -> np.ndarray:
    idx = np.loadtxt(runs / "mixed" / model / "optimize_cs" / "indices.csv",
                     delimiter=",", dtype=np.int64)
    return (idx < N_ENSEMBLE).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=RUNS,
                    help="the X-EISD protocol run directory holding mixed/<model>/")
    runs = ap.parse_args().runs
    style.use_house_style()

    fig, ax = plt.subplots(figsize=(style.WIDTH["text"], 2.5))
    bins = np.arange(0.44, 0.92, 0.01)
    values, per_model = {}, {}
    for model, label, slot in ARMS:
        f = per_model[model] = fractions(runs, model)
        ax.hist(f, bins=bins, color=style.PALETTE[slot], alpha=0.85, label=label,
                edgecolor=style.INK["surface"], linewidth=0.4)
        values[model] = {"mean_ensemble_fraction": float(f.mean()),
                         "median_ensemble_fraction": float(np.median(f)),
                         "sd_across_restarts": float(f.std(ddof=1)),
                         "restart_quantile_2p5_97p5": [float(np.percentile(f, 2.5)),
                                                       float(np.percentile(f, 97.5))],
                         "absolute_distance_from_paper": float(abs(f.mean() - PAPER_FRACTION)),
                         "n_restarts": int(f.size)}

    # Headroom first, so the reference annotations sit in empty axes and not on the bars.
    ymax = max(p.get_height() for p in ax.patches)
    ax.set_ylim(0, ymax * 1.30)
    ax.axvline(0.50, color=style.INK["axis"], linewidth=0.9, linestyle=":", zorder=1)
    ax.axvline(PAPER_FRACTION, color=style.INK["muted"], linewidth=1.2, linestyle="--", zorder=4)
    # Both labels point leftward into the empty band between the two distributions.
    ax.text(PAPER_FRACTION - 0.006, ymax * 1.22,
            "published, all eight\nobservables jointly: 76%",
            fontsize=6.2, color=style.INK["muted"], va="top", ha="right")
    ax.text(0.50 - 0.006, ymax * 0.72, "starting\npool: 50%",
            fontsize=6.2, color=style.INK["muted"], va="top", ha="right")

    ax.set_xlabel("ENSEMBLE fraction of the optimized 100-conformer set\n"
                  "(chemical shifts alone; 1,000 independent restarts)")
    ax.set_ylabel("restarts")
    ax.set_xlim(0.415, bins[-1])
    style.finish(ax)
    style.legend(ax, loc="upper left", fontsize=6.6, frameon=False)
    fig.tight_layout()

    d = values["deepshift"]["mean_ensemble_fraction"] - values["shiftx2"]["mean_ensemble_fraction"]
    values["deepshift_minus_shiftx2"] = float(d)
    # Restarts are independent by construction, so the interval on the difference is a
    # plain bootstrap over them. Computed here and never quoted: a number a caption
    # states and no script produces cannot be checked.
    rng = np.random.default_rng(SEED)
    draws = np.array([
        100.0 * (rng.choice(per_model["deepshift"], per_model["deepshift"].size).mean()
                 - rng.choice(per_model["shiftx2"], per_model["shiftx2"].size).mean())
        for _ in range(N_BOOT)])
    values["deepshift_minus_shiftx2_points_ci95"] = [float(np.percentile(draws, 2.5)),
                                                     float(np.percentile(draws, 97.5))]
    values["bootstrap"] = {"unit": "independent optimizer restart", "resamples": N_BOOT,
                           "seed": SEED}
    values["paper_reference"] = {"ensemble_fraction": PAPER_FRACTION,
                                 "optimization": "joint over all eight observables",
                                 "source": "Lincoff et al., Commun Chem 3, 74 (2020)",
                                 "location": "Results text, not a figure or table"}

    frozen.emit(fig, "fig_xeisd_composition")
    print(json.dumps(values, indent=1))


if __name__ == "__main__":
    main()
