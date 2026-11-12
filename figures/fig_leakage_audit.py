#!/usr/bin/env python3
"""Extended Data Fig. 2: the homology modules of UCBShift and SHIFTX2 retrieving the test protein.

(a) How often the test protein's own deposition is retrieved: UCBShift-Y on UCBShift's 200
    test chains (``ucbshift_eval_per_protein_paper_exclusions.csv``) and SHIFTY+ on SHIFTX2's
    61 test proteins (``shiftx2_test61_selfretrieval.csv``: own BMRB entry, a duplicate BMRB
    entry of the same protein, or neither).
(b) The percentage of predictions within 0.005 ppm of the deposited shift. The pooled bars
    of the distributed UCBShift-Y run and of both SHIFTY+ runs are the Supplementary
    Information's numbers, read from ``si_build/emit_text_numbers.py``'s output; the
    own-record-blocked UCBShift-Y bar and the per-chain dots are counted here with the same
    function (``benchmarks.self_retrieval``), and the distributed run's chains must sum to
    the printed totals.
(c) Backbone RMSE on the UCBShift-200 set relative to DeepShift, on the cells every arm
    shares: UCBShift as distributed and with every deposition of the same protein and
    references at >= 90 % identity blocked, SHIFTX2 as distributed and with its own BMRB
    entry blocked, and the values UCBShift's paper prints.

UCBShift's test targets are third-party (``$SI_GATE_CSPRED_TM``), so (b) is tier C.

Usage:
    python figures/fig_leakage_audit.py --text-numbers si_build/out/text_numbers.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import frozen
import journal_style as J
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from benchmarks.self_retrieval import ucbshift_y_chains

STEM = "ed_fig_02_leakage_audit"
LEAKAGE = frozen.env_path("DEEPSHIFT_UCBSHIFT_LEAKAGE")
UCBY_RETRIEVAL = LEAKAGE / "ucbshift_eval_per_protein_paper_exclusions.csv"
SHIFTY_RETRIEVAL = LEAKAGE / "shiftx2_test61_selfretrieval.csv"
UCBY_BLOCKED_PREDS = LEAKAGE / "preds_own_record_blocked"
UCBY_PREDS = frozen.env_path("SI_SELF_RETRIEVAL_UCBY", "preds")
TARGETS = frozen.env_path("SI_GATE_CSPRED_TM", "test_targets")
UCB200_SCORES = frozen.env_path("SI_UCB200_ARTIFACTS_DIR", "score_allpred_paper_exclusions.json")
EXCLUSIONS = frozen.REPO / "benchmarks/reference/ucb200_paper_exclusions.json"

NUC = [("HA", r"H$\alpha$"), ("H", r"H$^{\rm N}$"), ("CA", r"C$\alpha$"), ("CB", r"C$\beta$"),
       ("C", r"C$^{\prime}$"), ("N", "N")]
DUPLICATE = "#9d7be0"          # a lighter violet: SHIFTY+ retrieving another entry of the same protein
MIN_CHAIN_CELLS = 20           # a chain is drawn as a dot only with this many scored cells
RMSE_ARMS = (  # key in the score report, label, colour, marker, filled
    ("UCBShift_H", "UCBShift, as distributed", J.AMBER, "o", False),
    ("ucbshift_strict", "UCBShift, self-retrieval blocked", J.AMBER, "o", True),
    ("shiftx2_full", "SHIFTX2, as distributed", J.VIOLET, "s", False),
    ("shiftx2_selfexcl", "SHIFTX2, own BMRB entry blocked", J.VIOLET, "s", True),
)
RMSE_OFFSET = {"UCBShift_H": -0.17, "ucbshift_strict": -0.06, "shiftx2_full": 0.06, "shiftx2_selfexcl": 0.17}
# The text-number groups panel (b) prints, and the third-party roots each one needs.
TEXT_NUMBER_GROUPS = {"self_retrieval_ucbshift_y": "SI_GATE_CSPRED_TM",
                      "self_retrieval_shifty61": "DEEPSHIFT_REFDB_SHIFTS and DEEPSHIFT_REFDB_LIBRARY"}


def panel_retrieval(ax, values):
    leak = pd.read_csv(UCBY_RETRIEVAL)
    used, n_ucb = int(leak.own_record_used_as_reference_full_db.sum()), len(leak)
    sx = pd.read_csv(SHIFTY_RETRIEVAL)
    via = sx.self_via.fillna("none").str.extract(r"^(\w+)")[0]
    n_own, n_sis, n_none = (int((via == k).sum()) for k in ("own", "sister", "none"))
    if n_own + n_sis + n_none != len(sx):
        raise SystemExit(f"{SHIFTY_RETRIEVAL}: a protein is neither own, sister nor none")
    values["a"] = {"ucbshift_own_record_retrieved": used, "ucbshift_chains": n_ucb, "shiftx2_61_own": n_own,
                   "shiftx2_61_duplicate_deposition": n_sis, "shiftx2_61_none": n_none}
    rows = [(f"UCBShift-Y\n{n_ucb} chains", [used, n_ucb - used], [J.AMBER, J.LIGHT_AMBER]),
            (f"SHIFTY+\n{len(sx)} proteins", [n_own, n_sis, n_none], [J.VIOLET, DUPLICATE, J.LIGHT_VIOLET])]
    for i, (_, parts, cols) in enumerate(rows):
        y, left, tot = 1 - i, 0, sum(parts)
        for p, c in zip(parts, cols):
            ax.barh(y, p / tot * 100, left=left, color=c, height=0.55, edgecolor="white", linewidth=0.4)
            if p / tot > 0.07:
                ax.text(left + p / tot * 50, y, f"{p}", ha="center", va="center", fontsize=J.FS["small"],
                        color="white" if c in (J.AMBER, J.VIOLET, DUPLICATE) else J.INK)
            left += p / tot * 100
    ax.set_yticks([1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=J.FS["small"], linespacing=1.2)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("% of test proteins")
    J.clean(ax)
    ax.text(0.5, 1.02, "Test protein's own deposition\nretrieved by the homology module", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=J.FS["small"], color=J.SECOND, linespacing=1.2)
    ax.legend(handles=[Patch(color=J.AMBER, label="own BMRB entry retrieved (UCBShift-Y, SHIFTY+)"),
                       Patch(color=DUPLICATE, label="duplicate BMRB entry of the same protein"),
                       Patch(color=J.LIGHT_AMBER, label="not retrieved (UCBShift-Y)"),
                       Patch(color=J.LIGHT_VIOLET, label="different protein only (SHIFTY+)")],
              loc="upper left", bbox_to_anchor=(-0.50, -0.18), ncol=1, fontsize=J.FS["note"],
              handlelength=0.9, handleheight=0.7, labelspacing=0.3, borderaxespad=0)
    return dict(zip(leak.pid, leak.own_record_used_as_reference_full_db))


def panel_copies(ax, printed, retrieved, values):
    exclusions = json.loads(EXCLUSIONS.read_text())
    dist = ucbshift_y_chains(UCBY_PREDS, TARGETS, exclusions)
    blk = ucbshift_y_chains(UCBY_BLOCKED_PREDS, TARGETS, exclusions)
    copied, cells = printed["self_retrieval_ucbshift_y_copied"], printed["self_retrieval_ucbshift_y_cells"]
    if (sum(c[0] for c in dist.values()), sum(c[1] for c in dist.values())) != (copied, cells):
        raise SystemExit("the distributed run's chains do not sum to the printed UCBShift-Y totals")
    bars = [("UCBShift-Y\ndistributed", J.AMBER, copied, cells),
            ("UCBShift-Y,\nown record\nremoved", J.LIGHT_AMBER, sum(c[0] for c in blk.values()), sum(c[1] for c in blk.values())),
            ("SHIFTY+\ndistributed", J.VIOLET, printed["self_retrieval_shifty61_copied"], printed["self_retrieval_shifty61_cells"]),
            ("SHIFTY+,\nexact-match\nreferences\nremoved", DUPLICATE, printed["self_retrieval_shifty61_noexact_copied"],
             printed["self_retrieval_shifty61_noexact_cells"])]
    values["b"] = {}
    rng = np.random.default_rng(1)                  # horizontal jitter of the dots only
    for i, (lab, col, c, t) in enumerate(bars):
        pct = 100 * c / t
        ax.bar(i, pct, color=col, width=0.66, edgecolor="none", zorder=2)
        ax.text(i, 120, f"{pct:.1f}%", ha="center", va="bottom", fontsize=J.FS["text"], color=J.INK, fontweight="bold")
        ax.text(i, 103.5, f"{c:,} of\n{t:,}", ha="center", va="bottom", fontsize=J.FS["note"], color=J.SECOND,
                linespacing=1.1)
        values["b"][lab.replace("\n", " ")] = {"copied": c, "cells": t, "percent": pct}
        if i < 2:
            for pid, (cc, tt, _) in (dist if i == 0 else blk).items():
                if tt < MIN_CHAIN_CELLS:
                    continue
                ax.plot(i + rng.uniform(-0.22, 0.22), 100 * cc / tt, "o", ms=1.5, mew=0,
                        color=J.INK if retrieved.get(pid, False) else J.FAINT, alpha=0.6, zorder=3)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=J.FS["note"], linespacing=1.15)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of predictions within 0.005 ppm\nof the deposited shift", linespacing=1.2)
    J.clean(ax, ygrid=True)
    ax.text(0.0, 1.03, "Predictions that equal the deposited shift", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=J.FS["small"], color=J.SECOND)
    ax.plot([], [], "o", ms=2.2, color=J.INK, alpha=0.8, label="chain, own record retrieved")
    ax.plot([], [], "o", ms=2.2, color=J.FAINT, alpha=0.9, label="chain, own record not retrieved")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.36), fontsize=J.FS["note"], handletextpad=0.2,
              borderaxespad=0, labelspacing=0.25)


def panel_rmse(ax, values):
    report = json.loads(UCB200_SCORES.read_text())
    cc, pub = report["common_cells"], report["published"]["UCBShift"]
    ds = {n: cc["deepshift_ensemble"][n]["rmse"] for n, _ in NUC}
    x = np.arange(len(NUC))
    ax.axhline(1.0, color=J.ACCENT, lw=1.1, zorder=1)
    values["c"] = {"deepshift_rmse": ds, "n_cells": {n: cc["deepshift_ensemble"][n]["n"] for n, _ in NUC},
                   "n_chains_common": report["funnel"]["common_chains"], "arms": {}}
    for key, lab, col, mk, filled in RMSE_ARMS:
        values["c"]["arms"][key] = {n: cc[key][n]["rmse"] for n, _ in NUC}
        ax.plot(x + RMSE_OFFSET[key], [cc[key][n]["rmse"] / ds[n] for n, _ in NUC], mk, ms=3.0, color=col,
                mfc=col if filled else "white", mew=0.8, linestyle="none", label=lab, zorder=3)
    values["c"]["ucbshift_paper_reported"] = {n: pub[n] for n, _ in NUC}
    ax.plot(x - 0.30, [pub[n] / ds[n] for n, _ in NUC], "D", ms=2.2, color=J.AMBER, mfc=J.AMBER, mew=0, alpha=0.55,
            linestyle="none", label="UCBShift, value in its paper", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([n[1] for n in NUC])
    ax.set_xlim(-0.55, len(NUC) - 0.45)
    ax.set_ylim(0.70, 1.40)
    ax.set_yticks([0.75, 1.0, 1.25])
    ax.set_ylabel("RMSE relative to DeepShift")
    J.clean(ax, ygrid=True)
    ax.text(0.0, 1.03, "Backbone RMSE, UCBShift-200 set", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=J.FS["small"], color=J.SECOND)
    ax.text(0.99, 0.60, "DeepShift = 1", color=J.ACCENT, fontsize=J.FS["small"], va="bottom", ha="right",
            fontweight="bold", transform=ax.transAxes)
    ax.text(-0.02, -0.215, "DeepShift\nRMSE, ppm", ha="right", va="center", fontsize=J.FS["note"], color=J.ACCENT,
            transform=ax.transAxes, linespacing=1.1)
    for i, (n, _) in enumerate(NUC):
        ax.text(i, -0.215, f"{ds[n]:.2f}", ha="center", va="center", fontsize=J.FS["note"], color=J.ACCENT,
                transform=ax.get_xaxis_transform())
    ax.legend(loc="upper left", bbox_to_anchor=(-0.18, -0.33), ncol=1, fontsize=J.FS["note"], handletextpad=0.2,
              borderaxespad=0, labelspacing=0.25)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--text-numbers", required=True, type=Path,
                    help="si_build/out/text_numbers.json, written by si_build/emit_text_numbers.py")
    args = ap.parse_args(argv)
    printed = {k: v["value"] for k, v in json.loads(args.text_numbers.read_text()).items()
               if k.startswith("self_retrieval_")}
    for group, variable in TEXT_NUMBER_GROUPS.items():
        if f"{group}_copied" not in printed:
            raise SystemExit(f"{args.text_numbers} lacks the {group} group (made with "
                             f"--allow-missing?): rerun si_build/emit_text_numbers.py with "
                             f"{variable} set")
    J.use_journal_style()
    fig = plt.figure(figsize=(J.DOUBLE, 66 * J.MM))
    ax_a = fig.add_axes([0.105, 0.36, 0.20, 0.50])
    ax_b = fig.add_axes([0.405, 0.37, 0.25, 0.49])
    ax_c = fig.add_axes([0.765, 0.36, 0.225, 0.50])
    values = {}
    retrieved = panel_retrieval(ax_a, values)
    panel_copies(ax_b, printed, retrieved, values)
    panel_rmse(ax_c, values)
    for x0, letter in ((0.004, "a"), (0.335, "b"), (0.685, "c")):
        fig.text(x0, 0.985, letter, fontsize=J.FS["panel"], fontweight="bold", va="top")
    frozen.emit(fig, STEM)
    print(json.dumps(values, indent=1))


if __name__ == "__main__":
    main()
