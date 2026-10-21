#!/usr/bin/env python3
"""Does training on AlphaFold models create the structure-source result?

The shipped ensemble is trained on AlphaFold, crystal and NMR rows together, and
it predicts a deposited shift better from an AlphaFold model than from either experimental
structure. The obvious objection is circularity: the model was shown AlphaFold geometry, so
of course it reads AlphaFold geometry best. The control removes exactly that -- five members
retrained on crystal + NMR rows only, every other hyperparameter the shipped members' own --
and scores both arms on the identical held-out fold-6 cells.

One panel per structure source, six backbone nuclei, both arms. Whiskers are each arm's own
label-set bootstrap interval from the canonical scorer. The paired per-nucleus differences
and the sentence about the experimental-only arm's matched cells are printed.

Both arms are scored here, from their fold-6 prediction dumps, with the fourteen
anticorrelated channels Methods excludes dropped first (``frozen.drop_anticorrelated``).
The by-source reports the evaluator wrote beside the dumps score every cell, those channels
included, so they are not read.

Usage:
    DEEPSHIFT_EXP_SOURCE_CONTROL=<directory holding the two ensemble dumps> \\
        python figures/fig_exp_source_control.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import frozen
import matplotlib.pyplot as plt
import numpy as np
import style

from deepshift import evaluate

EVAL_DIR = frozen.env_path("DEEPSHIFT_EXP_SOURCE_CONTROL")
# Each arm's five-member fold-6 ensemble, as `deepshift evaluate --save-npz` wrote it.
DUMPS = {"shipped": "shipped_ensemble_fold6.preds.npz",
         "exp": "exp_ensemble_fold6.preds.npz"}

SOURCES = (("af", "AlphaFold"), ("xtal", "crystal"), ("nmr", "NMR"))
# Both arms are DeepShift, so both take the DeepShift hue; the retrained arm is the
# open face, the same facet the competitor families use for a family member.
ARMS = (("shipped", "all-source (shipped)", True),
        ("exp", "experimental-only", False))


def plain_nucleus(column: str) -> str:
    """``'ca_shift'`` -> ``'CA'``: the nucleus name a caption fact carries as text."""
    return (column[:-6] if column.endswith("_shift") else column).upper()


def matched_control_fact(matched, nuclei) -> str:
    """The matched-cell sentence, counted from the matched RMSE table itself.

    ``matched`` is ``{source: {column: rmse}}`` over the cells all three sources share.
    The sentence says on how many backbone nuclei the experimental-only arm predicts
    AlphaFold structures best, and names every nucleus another source wins, with the two
    numbers that decide it.
    """
    best = {c: min(matched, key=lambda src: matched[src][c]) for c in nuclei}
    won = [c for c in nuclei if best[c] == "af"]
    lost = [(c, best[c]) for c in nuclei if best[c] != "af"]
    fact = ("on cells all three sources share, the experimental-only model still predicts "
            f"AlphaFold structures best on {len(won)} of {len(nuclei)} backbone nuclei")
    if lost:
        detail = "; ".join(
            f"{plain_nucleus(c)} goes to {src} ({matched[src][c]:.4g} against AF "
            f"{matched['af'][c]:.4g})" for c, src in lost)
        fact += f" ({detail})"
    return fact


def draw(reports, nuclei):
    style.use_house_style()
    # The figure legend sits in reserved headroom above the panel titles, so the layout
    # is set here and not solved.
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(style.WIDTH["text"], 2.65))
    x = np.arange(len(nuclei), dtype=float)
    for ax, (src, title) in zip(axes, SOURCES, strict=True):
        for offset, (key, label, filled) in zip((-0.16, 0.16), ARMS, strict=True):
            table = reports[key]["marginal"][src]
            rmse = np.array([table[c]["rmse"] for c in nuclei])
            lo = np.array([table[c]["rmse_ci"][0] for c in nuclei])
            hi = np.array([table[c]["rmse_ci"][1] for c in nuclei])
            ax.vlines(x + offset, lo, hi, color=style.ACCENT, linewidth=style.LINE, zorder=2)
            ax.plot(x + offset, rmse, marker="o", linestyle="none",
                    markersize=style.MARKER, color=style.ACCENT,
                    markerfacecolor=style.ACCENT if filled else style.INK["surface"],
                    markeredgecolor=style.ACCENT,
                    markeredgewidth=style.RING if filled else 0.9,
                    label=label, zorder=3)
        ax.set_title(title, loc="left", fontsize=8.2, color=style.INK["primary"], pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([style.nucleus_label(c) for c in nuclei], fontsize=7.0)
        ax.set_xlim(-0.55, len(nuclei) - 0.45)
        style.finish(ax, xgrid=False, spines=("bottom", "left"))
        ax.yaxis.grid(True, color=style.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.6)
    axes[0].set_ylabel("held-out RMSE (ppm)", fontsize=7.4)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.13, wspace=0.24)
    handles, labels = axes[0].get_legend_handles_labels()
    style.legend(fig, handles, labels, loc="upper center", ncol=2,
                 bbox_to_anchor=(0.5, 1.005), fontsize=7.0)
    return fig


def score_arms(paths):
    """Both arms scored by source, marginal and matched, and their paired difference,
    after the anticorrelated channels are dropped from each.

    The canonical evaluator's own functions and bootstrap contract, on the backbone columns
    in its own order, so a point here is the number ``deepshift evaluate --by-source``
    would print on the fenced dump.
    """
    arms, dropped = {}, {}
    for key, path in paths.items():
        arms[key], dropped[key] = frozen.drop_anticorrelated(evaluate.read_predictions(path))
    baseline, candidate = evaluate.align(arms["shipped"], arms["exp"])
    columns = [c for c in evaluate.BACKBONE_SHIFTS if c in baseline.shift_cols]
    reports = {key: {"marginal": evaluate.by_source(preds, columns=columns),
                     "matched": evaluate.by_source(preds, matched=True, columns=columns)}
               for key, preds in (("shipped", baseline), ("exp", candidate))}
    return reports, evaluate.compare(baseline, candidate, columns=columns), dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    args = ap.parse_args()

    nuclei = list(style.BACKBONE)
    paths = {k: args.eval_dir / name for k, name in DUMPS.items()}
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing frozen artifacts: {missing}")
    reports, compare, dropped = score_arms(paths)

    fig = draw(reports, nuclei)
    # baseline - candidate, so negative means the shipped arm has the lower RMSE.
    paired_delta = {c: {"delta": compare[c]["delta_rmse"], "ci95": compare[c]["ci"],
                        "n": compare[c]["n"]} for c in nuclei}
    matched = {src: {c: reports["exp"]["matched"][src][c]["rmse"] for c in nuclei}
               for src, _ in SOURCES}

    frozen.emit(fig, "fig_exp_source_control")
    print(json.dumps(paired_delta, indent=1))
    print(matched_control_fact(matched, nuclei))
    print(f"cells dropped by the anticorrelated fence: {dropped}")


if __name__ == "__main__":
    main()
