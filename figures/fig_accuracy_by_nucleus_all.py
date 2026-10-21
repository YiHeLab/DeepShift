#!/usr/bin/env python3
"""Held-out accuracy of the five-model ensemble, one row per scorable nucleus.

Every value is read from the run's five held-out member prediction dumps and recomputed
here.

Two decisions carry the science:

  * **Per nucleus, never pooled.** An amide proton and a backbone nitrogen differ by
    more than an order of magnitude in ppm, so a mean across nuclei is not a quantity.
    The panels facet by isotope and each carries its own ppm axis.
  * **RMSE leads, MAE follows.** RMSE is the quantity a spectroscopist feels; MAE sits
    beside it because RMSE alone cannot say whether a number is a broad spread or a
    thin tail. Intervals are 95% percentile bootstrap over BMRB clusters.

``--scores`` draws the rows from a per-nucleus scoring report
(``python -m benchmarks.score_by_nucleus``) instead of recomputing from the dumps. That is
how the figure reads the frozen master cell set: the scorer owns which cells each nucleus
is scored on, so this figure and the side-chain table beside it print one DeepShift
number per nucleus and not two.

Usage:
    DEEPSHIFT_RUN=<finished run> python figures/fig_accuracy_by_nucleus_all.py
    python figures/fig_accuracy_by_nucleus_all.py --scores <score_by_nucleus>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import frozen
import numpy as np
import style

from deepshift.quality_control import ARG_NE_OBSERVED_RANGE_PPM
from deepshift.statistics import DEFAULT_SEED

STEM = "fig_accuracy_by_nucleus_all"

# Arg NE carries a label artefact, not only a prediction error: a per-deposition
# spectrometer fold aliases part of its shifts by one sweep width, so its RMSE is mostly
# the satellite. Those cells are excluded from the scored row -- they are wrong labels,
# not wrong predictions -- and the footnote reports how many were dropped and what the
# RMSE would be with them.
#
# The fence is on the observed value, never on the residual. A cut on |prediction -
# label| would decide which labels are scored by how well the model did on them, keeping
# an aliased label the model happened to chase and dropping a sound one it missed. This
# reads the same model-free fence the supplementary tables use, so a figure and the table
# beside it stand on one definition of which cells exist.
ALIASED_NUCLEUS = "ne_shift"
ARG_NE_ATOM = "NE"                 # how a per-nucleus scoring report names that column
NE_LO, NE_HI = ARG_NE_OBSERVED_RANGE_PPM
# The fence file states the same interval in its predicate, so a copy that drifts from
# the authority fails here, since a drifted copy would score different cells in silence.
NE_FENCE_PATH = Path(__file__).resolve().parent / "reference" / "ne_alias_fence.json"
NE_FENCE = json.loads(NE_FENCE_PATH.read_text())
assert f"[{NE_LO}, {NE_HI}] ppm" in NE_FENCE["rule"]["predicate"], NE_FENCE["rule"]["predicate"]


def alias_report(preds, column=ALIASED_NUCLEUS, lo=NE_LO, hi=NE_HI):
    """Measure the aliased satellite for one nucleus. Returns None if it is absent.

    Aliased is a property of the label: an observed value outside the population fence.
    The prediction never enters the decision, so the reported error is the error on the
    cells the fence keeps, not the cells the model happened to fit.
    """
    if column not in preds.shift_cols:
        return None
    j = preds.shift_cols.index(column)
    seen = preds.effective_mask[:, j]
    if seen.sum() == 0:
        return None
    obs = preds.resolved_ppm[seen, j]
    err = preds.pred_ppm[seen, j] - obs
    far = (obs < lo) | (obs > hi)
    if not far.any():
        return None
    keep = ~far
    return {
        "column": column, "n": int(seen.sum()),
        "n_aliased": int(far.sum()), "frac_aliased": float(far.mean()),
        "median_offset_ppm": float(np.median(err[far])),
        "frac_negative": float((err[far] < 0).mean()),
        "rmse_all": float(np.sqrt((err ** 2).mean())),
        "rmse_dealiased": float(np.sqrt((err[keep] ** 2).mean())),
        "fence_ppm": [lo, hi],
        "fence_basis": NE_FENCE["rule"]["basis_note"],
    }


def report_scores(report: dict, min_cells: int) -> dict:
    """One row per nucleus, read from a per-nucleus scoring report's DeepShift arm.

    The report names the cells each nucleus was scored on and carries DeepShift's error
    with its cluster-bootstrap interval; nothing is recomputed here.
    """
    scores = {}
    for node in report["by_nucleus"].values():
        arm = node["arms"]["deepshift"]
        if arm.get("n_cells", node["n_cells"]) < min_cells:
            continue
        scores[node["column"]] = {
            "n": arm.get("n_cells", node["n_cells"]),
            "n_label_sets": node["n_label_sets"],
            "rmse": arm["rmse"], "rmse_ci": arm["rmse_ci95"], "mae": arm["mae"],
        }
    return dict(sorted(scores.items(), key=lambda item: item[1]["rmse"]))


def report_alias(report: dict) -> dict | None:
    """The aliased-label footnote a scoring report supports, or None when it removed none.

    The report fences the same labels this figure does, on the observed value alone, and
    for every arm. What it does not carry is the error those labels would have added, so
    the note drawn from it states the exclusion and not the counterfactual.
    """
    fence = report["exclusions_on_observed"]["arg_ne_sweep_width"]
    if not fence["n_cells_removed"]:
        return None
    column = fence["column"]
    kept = report["by_nucleus"].get(ARG_NE_ATOM, {}).get("n_cells", 0)
    return {"column": column, "n": kept + fence["n_cells_removed"],
            "n_aliased": fence["n_cells_removed"],
            "frac_aliased": fence["n_cells_removed"] / (kept + fence["n_cells_removed"]),
            "fence_ppm": fence["accepted_ppm"], "fence_basis": fence["rule"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, default=None,
                    help="a benchmarks.score_by_nucleus report to draw the rows from")
    ap.add_argument("--dumps", type=Path, default=None,
                    help="directory holding the members' prediction dumps")
    ap.add_argument("--fold", type=int, default=6, help="held-out fold scored")
    ap.add_argument("--members", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--min-cells", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if args.scores:
        report = json.loads(args.scores.read_text())
        scores, alias = report_scores(report, args.min_cells), report_alias(report)
    else:
        paths = frozen.member_dumps(args.fold, args.members, directory=args.dumps)

        from deepshift import evaluate

        preds = evaluate.ensemble([evaluate.read_predictions(path) for path in paths])
        # the same fence the benchmark tables use
        preds, _ = frozen.drop_anticorrelated(preds)
        # Measure the Arg NE aliased satellite, then drop those cells before scoring.
        alias = alias_report(preds)
        if alias is not None:
            j = preds.shift_cols.index(ALIASED_NUCLEUS)
            obs = preds.resolved_ppm[:, j]
            preds.effective_mask[:, j] &= (obs >= NE_LO) & (obs <= NE_HI)
        scored = evaluate.score(preds, columns=list(preds.shift_cols), boots=args.boot,
                                seed=args.seed, group="bmrb")
        # A nucleus seen a handful of times cannot carry a readable interval, so the
        # figure holds a higher floor than the library's.
        scores = {c: s for c, s in scored.items() if s["n"] >= args.min_cells}
        if ALIASED_NUCLEUS not in scores:
            alias = None

    fig = draw(scores, alias=alias)
    frozen.emit(fig, STEM)


def draw(scores: dict, *, alias: dict | None = None):
    """Facet by isotope; each panel keeps its own ppm axis."""
    import matplotlib.pyplot as plt

    style.use_house_style()
    groups = {}
    for col in scores:
        groups.setdefault(style.nucleus_element(col), []).append(col)
    order = [e for e in ("1H", "13C", "15N") if e in groups]
    for element in order:                     # best nucleus at the top of each panel
        groups[element].sort(key=lambda c: scores[c]["rmse"])

    counts = [len(groups[e]) for e in order]
    height = 0.75 + sum(counts) * style.ROW_HEIGHT + 0.45 * len(order)
    note = _alias_note(alias)
    band = 0.44 if note else 0.0   # footnote strip, kept out of the layout engine's reach
    fig = plt.figure(figsize=(style.WIDTH["text"], height + band))
    if band:
        # constrained_layout will otherwise expand the axes straight through a figure-
        # level text, so the strip is fenced off in the layout rect.
        fig.get_layout_engine().set(rect=(0, band / (height + band), 1,
                                          1 - band / (height + band)))
    axes = np.atleast_1d(fig.subplots(len(order), 1, height_ratios=counts))

    for ax, element in zip(axes, order, strict=True):
        cols = groups[element]
        s = [scores[c] for c in cols]
        marked = alias["column"] if alias else None
        style.forest(
            ax, [style.nucleus_label(c) + (r"$^{\dagger}$" if c == marked else "")
                 for c in cols],
            # Only RMSE carries an interval: it is the headline, and a second set of
            # whiskers on the same row competes with it for no extra information.
            # Both rows are DeepShift, so MAE takes the de-emphasis ink and not a
            # second hue -- an amber dot here would read as UCBShift.
            [{"name": "RMSE", "values": [x["rmse"] for x in s], "color": style.ACCENT,
              "lo": [x["rmse_ci"][0] for x in s], "hi": [x["rmse_ci"][1] for x in s]},
             {"name": "MAE", "values": [x["mae"] for x in s],
              "color": style.INK["faint"]}],
            # Each isotope keeps its own ppm scale, so the unit rides on the panel
            # title instead of repeating under three different axes.
            xlabel=None,
            value_column=[f"{x['rmse']:.3f}   {x['n']:>7,}" for x in s],
            value_header="RMSE     cells",
            xlim=(0, None),
        )
        ax.set_title(f"{style.ELEMENT_LABEL[element]}  (ppm)", loc="left", pad=3,
                     color=style.INK["primary"])
    handles, labels = axes[0].get_legend_handles_labels()
    style.legend(fig, handles=handles, labels=labels, loc="outside upper right", ncol=2)
    if note:
        fig.text(0.012, 0.006, note, ha="left", va="bottom", fontsize=6.2,
                 color=style.INK["secondary"], linespacing=1.45)
    return fig


def _alias_note(alias: dict | None) -> str:
    """The footnote that keeps a label artefact from reading as a prediction error."""
    if not alias:
        return ""
    satellite = (f", in a satellite at {alias['median_offset_ppm']:.1f} ppm"
                 if "median_offset_ppm" in alias else "")
    counterfactual = (f"; with them the RMSE would be {alias['rmse_all']:.2f} ppm rather "
                      f"than the {alias['rmse_dealiased']:.2f} shown"
                      if "rmse_all" in alias else "")
    return (
        f"$^{{\\dagger}}$ {alias['n_aliased']:,} of {alias['n']:,} Arg "
        f"{style.nucleus_label(alias['column'])} labels ({alias['frac_aliased']:.0%}) "
        f"fall outside {alias['fence_ppm'][0]}--{alias['fence_ppm'][1]} ppm{satellite}: "
        f"a per-deposition sweep-width fold in the deposited label, not a prediction "
        f"error. The fence is on the observed value alone. Those cells are excluded\n"
        f"   from the scored row{counterfactual}.")


if __name__ == "__main__":
    main()
