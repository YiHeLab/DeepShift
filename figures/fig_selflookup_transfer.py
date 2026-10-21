#!/usr/bin/env python3
"""What a homology-transfer module is worth once it cannot retrieve the query itself.

Two panels, two cohorts, one question.

  (a) UCBShift on its own published 200-protein test set, scored twice: with the target
      protein's own BMRB record present in the lookup database, and with that one record
      removed. The Y module retrieves the target's own deposition for 113 of 200 proteins.
      No DeepShift model is involved in this panel, which is why it stays a finished
      measurement of its own.

  (b) This run's held-out fold 6. Per identity stratum, RMSE(no-transfer baseline) -
      RMSE(arm) in ppm, for UCBShift and SHIFTX2 with self-retrieval allowed and with
      every deposition of the target's own protein blocked -- the exclusion the paper
      reports. The no-transfer baseline is UCBShift-X for the UCBShift arms and SHIFTX2
      with SHIFTY+ off for the SHIFTX2 arms.

The x axis of (b) is the identity of the closest reference the method actually retrieved
when self-retrieval was allowed, read from the leaky arm's own per-structure record. Both
arms of a method are binned by that one number, so the paired curves describe the same
structures.

Panel (b) is a comparison of predictors, so its cells are the frozen master set when
``--master-cells`` names one; without it the panel falls back to the cells its own arms
all predicted, and the console says which cohort was drawn.

The emitted stem is the deployed name, ``fig_04_selflookup``. The module is named for
its topic, because a figure number in a filename rots the moment a float moves.

Usage:
    DEEPSHIFT_RUN=... DEEPSHIFT_SELFLOOKUP_CELLS=... DEEPSHIFT_UCBSHIFT_LEAKAGE=... \\
        python figures/fig_selflookup_transfer.py [--master-cells <master>.npz]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import frozen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import style

from benchmarks import master_cells
from deepshift.self_exclusion import REPORTED_EXCLUSION_PHRASE, arm_directory

# The per-cell fold-6 dump `benchmarks.dump_paired_cells` writes for these six arms.
CELLS = frozen.env_path("DEEPSHIFT_SELFLOOKUP_CELLS")
# The competitor outputs themselves: panel (b)'s x axis is read from their hit logs.
ARM_ROOT = Path(os.environ.get("DEEPSHIFT_BENCH_OUT") or frozen.RUN_ROOT / "benchmark")
# Panel (a)'s per-protein errors, without the cells UCBShift's paper removes from its test set
# (benchmarks/reference/ucb200_paper_exclusions.json), as every table on that test set
# scores it; eval_per_protein_paper_exclusions.py writes it beside the unexcluded file.
LEAK_EVAL = frozen.env_path("DEEPSHIFT_UCBSHIFT_LEAKAGE",
                            "ucbshift_eval_per_protein_paper_exclusions.csv")

# (leaky arm, blocked arm, no-transfer baseline, printed name, colour, marker).
# SHIFTX2 first: one shared generator feeds every stratum's bootstrap, so the order the
# strata are computed in is part of the interval.
METHODS_REPORTED_EXCLUSION = [
    ("shiftx2_leaky", "shiftx2_sameprot", "shiftx2_noshifty", "SHIFTX2", style.VIOLET, "s"),
    ("ucbshift_full_leaky", "ucbshift_full_sameprot", "ucbshift_x", "UCBShift",
     style.AMBER, "o"),
]
BINS = [(">=0.90", 0.90, 1.001), ("0.70-0.90", 0.70, 0.90),
        ("0.50-0.70", 0.50, 0.70), ("<0.50", 0.0, 0.50)]
BIN_TICK = {">=0.90": "$\\geq$0.90", "0.70-0.90": "0.70–0.90",
            "0.50-0.70": "0.50–0.70", "<0.50": "$<$0.50"}
MIN_CELLS = 2000
N_BOOT = 2000
SEED = 20260728

A_X = ("mean $|\\delta_{\\mathrm{pred}}-\\delta_{\\mathrm{exp}}|$ over the protein's six\n"
       "backbone nuclei (ppm), with its own BMRB\nrecord in the lookup database")
A_Y = ("mean $|\\delta_{\\mathrm{pred}}-\\delta_{\\mathrm{exp}}|$ over the protein's six\n"
       "backbone nuclei (ppm), with every test protein's\nBMRB record removed from the lookup database")
B_X = ("sequence identity of the closest reference the method\n"
       "retrieves when self-retrieval is allowed\n(both arms binned by this number)")
B_Y = ("RMSE$_{\\mathrm{no\\,transfer}}$ $-$ RMSE$_{\\mathrm{arm}}$ (ppm)\n"
       "no transfer = UCBShift-X, or SHIFTX2 with SHIFTY+ off")


# --------------------------------------------------------------------------------
# Panel (b) statistics
# --------------------------------------------------------------------------------

def kept_identity(arm_root: Path, arm: str) -> dict:
    """structure instance -> identity of the best reference this arm kept (None if none).

    The key is the file's own name, which is the benchmark's structure instance and what
    the cell dump carries. An arm's record names the coordinate file it was run on
    instead, which is a different identifier for the same task.
    """
    out = {}
    for path in arm_directory(arm_root, arm).glob("*.json"):
        record = json.loads(path.read_text())
        out[path.stem] = (record.get("self_exclusion", {}).get("max_kept_ref_identity"))
    return out


def load_cells(path, master):
    """The dump's cells, restricted to the master set when one is named.

    Returns the arrays panel (b) needs and the record of which cohort was drawn, so a
    fallback is visible rather than implied.
    """
    with np.load(path, allow_pickle=False) as dump:
        arms = [str(arm) for arm in dump["arms"]]
        data = {"true": dump["true_ppm"], "sid": dump["sid"].astype(str),
                "atom": dump["atom"].astype(str), "resnum": dump["resnum"].astype(str),
                "pred": {arm: dump["pred_" + arm] for arm in arms}}
    if master is None:
        record = {"cohort": f"the cells all {len(arms)} arms of the dump predicted; no "
                            "master cell file was given", "master_cells": None}
    else:
        keys = master_cells.cell_keys(
            [f"{atom.lower()}_shift" for atom in data["atom"]], data["sid"], data["resnum"])
        keep = master_cells.restrict(keys, master_cells.load_master(master),
                                     what="fold-6 backbone")
        data = {"true": data["true"][keep], "sid": data["sid"][keep],
                "atom": data["atom"][keep], "resnum": data["resnum"][keep],
                "pred": {arm: value[keep] for arm, value in data["pred"].items()}}
        record = {"cohort": "the frozen master cell set", "master_cells": str(master)}
    record["n_cells"] = int(data["true"].size)
    record["n_structures"] = len(set(data["sid"].tolist()))
    print(f"panel (b): {record['n_cells']:,} cells over {record['n_structures']} "
          f"structures -- {record['cohort']}")
    return data, record


def _boot_units(sid):
    codes = {s: k for k, s in enumerate(sorted(set(sid)))}
    return np.array([codes[s] for s in sid]), len(codes)


def gain_with_ci(true, base, arm, sid, rng):
    """RMSE(baseline) - RMSE(arm) on one stratum, with a CI clustered on structures."""
    units, n_units = _boot_units(sid)
    sq_b = np.bincount(units, (base - true) ** 2, minlength=n_units)
    sq_a = np.bincount(units, (arm - true) ** 2, minlength=n_units)
    count = np.bincount(units, minlength=n_units).astype(float)
    point = np.sqrt(sq_b.sum() / count.sum()) - np.sqrt(sq_a.sum() / count.sum())
    draws = rng.integers(0, n_units, size=(N_BOOT, n_units))
    c = count[draws].sum(1)
    boot = np.sqrt(sq_b[draws].sum(1) / c) - np.sqrt(sq_a[draws].sum(1) / c)
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def dose_response(data, arm_root: Path, methods):
    """-> (rows per curve, one series per drawn curve). One rng feeds every stratum."""
    rng = np.random.default_rng(SEED)
    true, sid, pred = data["true"], data["sid"], data["pred"]

    values, series = {}, []
    for leaky_arm, blocked_arm, baseline, name, colour, marker in methods:
        ident = kept_identity(arm_root, leaky_arm)
        iv = np.array([ident.get(s) if ident.get(s) is not None else np.nan for s in sid],
                      dtype=float)
        for arm, is_leaky in ((leaky_arm, True), (blocked_arm, False)):
            pts, los, his, rows = [], [], [], []
            for label, lo, hi in BINS:
                m = np.isfinite(iv) & (iv >= lo) & (iv < hi)
                if m.sum() < MIN_CELLS:
                    pts.append(np.nan)
                    los.append(np.nan)
                    his.append(np.nan)
                    rows.append({"bin": label, "n_cells": int(m.sum()), "gain_ppm": None})
                    continue
                g, l, h = gain_with_ci(true[m], pred[baseline][m], pred[arm][m], sid[m], rng)
                pts.append(g)
                los.append(l)
                his.append(h)
                rows.append({"bin": label, "n_cells": int(m.sum()),
                             "n_structures": len(set(sid[m])),
                             "baseline_rmse": float(
                                 np.sqrt(np.mean((pred[baseline][m] - true[m]) ** 2))),
                             "arm_rmse": float(np.sqrt(np.mean((pred[arm][m] - true[m]) ** 2))),
                             "gain_ppm": g, "ci95": [l, h]})
            values[f"{arm}_vs_{baseline}__binned_by_{leaky_arm}_identity"] = rows
            series.append({"name": name, "arm": arm, "leaky": is_leaky, "colour": colour,
                           "marker": marker, "pts": pts, "lo": los, "hi": his})
    return values, series


# --------------------------------------------------------------------------------
# Draw
# --------------------------------------------------------------------------------

def panel_a(ax, table):
    """UCBShift's published per-protein error against its own-record-excluded rerun."""
    published = table["MAE_combined_full_db"].to_numpy()
    excluded = table["MAE_combined_test_excluded"].to_numpy()
    used = table["own_record_used_as_reference_full_db"].to_numpy().astype(bool)

    hi = 1.65                       # range set by the bulk; the off-scale protein is marked
    ax.plot([0, hi], [0, hi], color=style.INK["primary"], linewidth=0.8, zorder=1)
    ax.scatter(published[~used], excluded[~used], s=9, facecolor="none",
               edgecolor=style.INK["faint"], linewidths=0.6, zorder=2,
               label=f"own record not used ({int((~used).sum())})")
    ax.scatter(published[used], excluded[used], s=9, color=style.AMBER, zorder=3,
               label=f"own record used ({int(used.sum())})")
    off = excluded > hi
    for x in published[off]:
        ax.annotate("", xy=(x, hi), xytext=(x, hi - 0.13),
                    arrowprops={"arrowstyle": "-|>", "color": style.AMBER, "lw": 0.8})
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(A_X, fontsize=6.4)
    ax.set_ylabel(A_Y, fontsize=6.4)
    style.finish(ax, xgrid=False, spines=("bottom", "left"))
    style.legend(ax, loc="lower right", scatterpoints=1, fontsize=6.4)
    return {"n_proteins_scatter": int(np.isfinite(published).sum()),
            "n_own_record_used": int(used.sum()),
            "n_own_record_not_used": int((~used).sum()),
            "median_published_mae": float(np.nanmedian(published)),
            "median_excluded_mae": float(np.nanmedian(excluded)),
            "n_above_plotted_range": int(off.sum())}


def panel_b(ax, series):
    """What each transfer module buys, by the identity it actually retrieved."""
    x = np.arange(len(BINS), dtype=float)
    # Marker shape is the method; the blocked arm -- the one the manuscript asserts -- is
    # solid and filled, and the leaky arm is the open dashed facet of the same hue.
    for spec in series:
        pts, lo, hi = (np.array(spec[k], dtype=float) for k in ("pts", "lo", "hi"))
        rule = "self-retrieval allowed" if spec["leaky"] else REPORTED_EXCLUSION_PHRASE
        ax.errorbar(x, pts, yerr=[pts - lo, hi - pts], color=spec["colour"],
                    linestyle="--" if spec["leaky"] else "-", marker=spec["marker"],
                    markersize=style.MARKER - 0.8,
                    linewidth=style.LINE, elinewidth=style.LINE * 0.8, capsize=2,
                    markerfacecolor=style.INK["surface"] if spec["leaky"] else spec["colour"],
                    markeredgecolor=spec["colour"], markeredgewidth=0.8,
                    label=f"{spec['name']}, {rule}")
    ax.axhline(0, color=style.INK["secondary"], linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([BIN_TICK[b[0]] for b in BINS])
    ax.set_xlim(-0.45, len(BINS) - 0.55)
    ax.set_xlabel(B_X, fontsize=6.4)
    ax.set_ylabel(B_Y, fontsize=6.4)
    style.finish(ax, xgrid=False, spines=("bottom", "left"))
    ax.yaxis.grid(True, color=style.INK["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    style.legend(ax, loc="upper right", fontsize=6.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=CELLS)
    ap.add_argument("--arm-root", type=Path, default=ARM_ROOT)
    ap.add_argument("--leak-eval", type=Path, default=LEAK_EVAL)
    ap.add_argument("--master-cells", type=Path, default=None,
                    help="the frozen master cell file panel (b) is scored on; without "
                         "it the panel falls back to its own arms' intersection")
    args = ap.parse_args()

    missing = [p for p in (args.cells, args.arm_root, args.leak_eval) if not p.exists()]
    if missing:
        raise SystemExit(f"missing frozen artifacts: {missing}")

    data, cohort = load_cells(args.cells, args.master_cells)
    style.use_house_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(style.WIDTH["text"], 3.3))
    a_values = panel_a(ax_a, pd.read_csv(args.leak_eval))
    b_values, series = dose_response(data, args.arm_root, METHODS_REPORTED_EXCLUSION)
    panel_b(ax_b, series)
    style.panel_label(ax_a, "(a)", x=-0.20, y=1.02)
    style.panel_label(ax_b, "(b)", x=-0.16, y=1.02)
    fig.tight_layout(pad=0.5, w_pad=1.8)

    frozen.emit(fig, "fig_04_selflookup")
    print(json.dumps(a_values, indent=1))
    print(json.dumps({"panel_b": b_values, "panel_b_cohort": cohort}, indent=1))


if __name__ == "__main__":
    main()
