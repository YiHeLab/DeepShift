#!/usr/bin/env python3
"""Held-out error against the best sequence identity to anything in training.

The body claims model error is only weakly related to homology to the training set, and
that whatever relation exists is reproduced by ProCS15 -- a quantum-chemistry predictor
with no training set and therefore nothing to leak. The ProCS15 arm is the whole
argument: if a physics model shows the same dependence, the dependence is target
difficulty and not memorization.

Three decisions worth stating.

Cohort. ProCS15 crashes permanently on a source-skewed subset of structures (an
out-of-bounds assertion in its 6-D H-bond lookup; idealised AlphaFold geometry never
leaves the table, real experimental geometry does), so scoring the two arms on their own
coverage would compare different proteins. Both arms are therefore scored on exactly the
structures ProCS15 completed and our model scored -- the intersection, printed with its
per-source composition.

Axis. Best MMseqs2 identity from each held-out structure to the training entities,
produced by si_build/emit_homology_axis.py with the instrument Table S9 (tab:train_distance)
uses, so the figure and the table carry one homology axis.

Units. ProCS15's absolute RMSE is roughly twice ours, so on a shared ppm axis its
dependence would be invisible. Panel (a) plots each structure's error as a multiple of
its own method's average, and panel (b) plots the change in that method's own error
between its least-related and most-related held-out proteins. The claim is about the
shape of the dependence, so both panels are within-method quantities; the absolute ppm
each one is a fraction of is printed beside panel (b) and, for every cell, on the
console.

Cells. The dump is restricted to the frozen master cell set every predictor comparison on
fold 6 reads (``--master-cells``, by default the fenced set beside the benchmark), through
``benchmarks.master_cells.load_master``, which refuses a set that still carries the
anticorrelated channels Methods excludes.

Residuals are raw, prediction minus the resolved observed value.

Drawn at the size it is placed. The supplement sets this figure at 0.74\\linewidth, so
the canvas is 3.55 in and the two panels are stacked and not set side by side, which
gives each of them the full width at the house font sizes.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import frozen
import numpy as np
import style
from scipy.stats import spearmanr

from benchmarks import master_cells
from deepshift import cohorts
from deepshift.statistics import bonferroni, tail_count

# Run products, named by environment with no fallback; see REPRODUCTION.md.
# The cell dump is two-arm: an eleven-arm dump would additionally impose
# UCBShift's and SHIFTX2's coverage, shrinking the cohort for reasons unrelated to this
# claim.
CELLS = frozen.env_path("DEEPSHIFT_HOMOLOGY_CELLS", "ds_procs15_cells.npz")
TRAIN_IDENT = frozen.env_path("DEEPSHIFT_LEAKAGE_DOSE_ROOT", "sid_to_training_identity.json")
TASKS = frozen.env_path("DEEPSHIFT_HOMOLOGY_BENCHMARK_ROOT", "fold6_tasks.tsv")
RECEIPTS = frozen.env_path("DEEPSHIFT_HOMOLOGY_BENCHMARK_ROOT", "procs15")
# The fenced master cell set the competitor tables are scored on.
MASTER = frozen.env_path("DEEPSHIFT_HOMOLOGY_BENCHMARK_ROOT", "master_cells",
                         "master_cells_pf14.npz")

ARMS = [("deepshift", "DeepShift", "slot1"), ("procs15", "ProCS15", "slot2")]
NUCLEI = ["h_shift", "ha_shift", "ca_shift", "cb_shift", "c_shift", "n_shift"]
ATOM_OF = {"h_shift": "H", "ha_shift": "HA", "ca_shift": "CA",
           "cb_shift": "CB", "c_shift": "C", "n_shift": "N"}
BACKBONE = list(ATOM_OF.values())

# Same four strata as the homology-dose panel, so the two figures bin identically.
BINS = [(0.00, 0.30, "<30"), (0.30, 0.50, "30-50"), (0.50, 0.70, "50-70"), (0.70, 1.001, r"$\geq$70")]
N_BOOT = 2000
SEED = 20260731

# SI_v3 sets this figure at 0.74\linewidth of a 345 pt text block, so it is drawn at
# exactly that: 0.74 * 4.79 in. Nothing is rescaled on the page and 8 pt stays 8 pt.
PLACED_WIDTH_IN = round(0.74 * style.WIDTH["text"], 2)
Y_CAP = 3.0                       # panel a's linear ceiling; set from the data in main()


# --------------------------------------------------------------------------------
# Quantities
# --------------------------------------------------------------------------------

def on_master(d, master):
    """The loaded dump restricted to the master cells, and how many cells that removed.

    A cell is named by (nucleus column, structure instance, residue number), the key the
    master file is frozen on. Every per-cell array is cut by the same mask.
    """
    with np.load(d["path"], allow_pickle=False) as dump:
        resnum = dump["resnum"].astype(str)
    keys = master_cells.cell_keys([f"{a.lower()}_shift" for a in d["atom"]], d["sid"], resnum)
    keep = master_cells.restrict(keys, master_cells.load_master(master), what="fold-6")
    cut = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
           for k, v in d.items() if k != "resid"}
    cut["resid"] = {arm: r[keep] for arm, r in d["resid"].items()}
    return cut, int((~keep).sum())


def load_cells(path, master):
    """The two-arm dump on the master cells."""
    d = cohorts.load_paired_cells(path)
    d["path"] = path
    return on_master(d, master)


def scored_cells(d):
    """The cells this figure reports: the six backbone nuclei.

    The dump carries cells outside them, which have no cohort RMS of their own to be
    scaled by and are no part of any quantity here.
    """
    return np.isin(d["atom"], BACKBONE)


def relative_error(d, arm, struct_code, n_struct):
    """Per-structure error in units of that arm's own cohort RMS.

    Each residual is divided by its arm's cohort RMSE for that nucleus before pooling, so
    a structure rich in nitrogen is not automatically the worst one. 1.0 is the cohort
    average for that arm; the quantity is dimensionless and comparable across arms.

    Unscored cells are masked out of both the sum and the count, so a nucleus this figure
    does not report cannot enter a structure's error.
    """
    r = d["resid"][arm]
    scored = scored_cells(d)
    z = np.zeros_like(r)
    for atom in BACKBONE:
        m = d["atom"] == atom
        z[m] = r[m] / np.sqrt(np.mean(r[m] ** 2))
    num = np.bincount(struct_code[scored], weights=z[scored] ** 2, minlength=n_struct)
    cnt = np.bincount(struct_code[scored], minlength=n_struct).astype(float)
    return np.sqrt(num / np.maximum(cnt, 1.0))


def spearman_ci(x, y, rng, boots=1000):
    res = spearmanr(x, y)
    n = x.size
    draws = rng.integers(0, n, size=(boots, n))
    b = np.array([spearmanr(x[k], y[k]).statistic for k in draws])
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n_structures": int(n),
            "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def median_ci(v, rng, boots=N_BOOT):
    if v.size == 0:
        return None
    b = np.median(v[rng.integers(0, v.size, size=(boots, v.size))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def ratio_to_low_bin(d, arm, atom, ident_cell, struct_code, lo, hi, rng, boots=N_BOOT):
    """RMSE(stratum) / RMSE(<30% stratum) for one nucleus, bootstrapped on structures.

    The two strata are resampled from the same draw over the pooled structures, so the
    interval is on the ratio and not on two independent estimates.
    """
    is_atom = d["atom"] == atom
    hi_m = is_atom & (ident_cell >= lo) & (ident_cell < hi)
    lo_m = is_atom & (ident_cell < 0.30)
    pooled = hi_m | lo_m
    units, idx = np.unique(struct_code[pooled], return_inverse=True)
    sq = d["resid"][arm][pooled] ** 2
    in_hi, in_lo = hi_m[pooled].astype(float), lo_m[pooled].astype(float)
    sh = np.bincount(idx, sq * in_hi, units.size)
    ch = np.bincount(idx, in_hi, units.size)
    sl = np.bincount(idx, sq * in_lo, units.size)
    cl = np.bincount(idx, in_lo, units.size)
    point = float(np.sqrt(sh.sum() / ch.sum()) / np.sqrt(sl.sum() / cl.sum()))
    draws = rng.integers(0, units.size, size=(boots, units.size))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (np.sqrt(sh[draws].sum(1) / ch[draws].sum(1))
             / np.sqrt(sl[draws].sum(1) / cl[draws].sum(1)))
    b = b[np.isfinite(b)]
    return {"ratio": point, "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "rmse_ppm": float(np.sqrt(sh.sum() / ch.sum())),
            "n_cells": int(hi_m.sum()), "n_structures": len(set(struct_code[hi_m])),
            "k_tail": tail_count(b - 1.0, point - 1.0), "resamples": int(b.size)}


def coverage_ledger():
    """Which fold-6 structures ProCS15 finished, by source -- the confound this figure
    has to neutralise and must not inherit."""
    tasks = {t["structure_id"]: t for t in csv.DictReader(TASKS.open(), delimiter="\t")}
    failed = set()
    for path in RECEIPTS.glob("*.json"):
        rec = json.loads(path.read_text())
        if not rec.get("predictions"):
            failed.add(rec["structure_id"])
    failed |= {s for s in tasks if not (RECEIPTS / f"{s}.json").exists()}
    by_source = Counter(t["source"] for t in tasks.values())
    fail_source = Counter(tasks[s]["source"] for s in failed if s in tasks)
    return tasks, {
        "n_fold6_structure_instances": len(tasks),
        "procs15_permanent_failures": len(failed),
        "procs15_failures_by_source": {k: f"{fail_source[k]}/{by_source[k]} "
                                          f"({100 * fail_source[k] / by_source[k]:.1f}%)"
                                       for k in sorted(by_source)},
        "mechanism": "out-of-bounds assertion in ProCS15's 6-D tabulated H-bond/geometry "
                     "lookup; AlphaFold geometry is idealised and never leaves the table, "
                     "so the failures are absent from AF and concentrated in experimental "
                     "structures",
    }


# --------------------------------------------------------------------------------
# Draw
# --------------------------------------------------------------------------------

def draw(ident_s, rel, rho, trend, ratios, absolute):
    import matplotlib.pyplot as plt

    style.use_house_style()
    # Stacked, and drawn at the width SI_v3 actually sets this figure at, so nothing is
    # scaled down after the fact and the house font sizes survive to the page.
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(PLACED_WIDTH_IN, 4.65),
                                  gridspec_kw={"height_ratios": [1.15, 1]})

    # Bin markers sit at the median identity INSIDE the bin, not at the bin's midpoint:
    # 36% of the cohort has no training relative at all, so a midpoint would draw the
    # lowest stratum at 15% identity when almost all of it is at zero.
    centres = np.array([100 * np.median(ident_s[(ident_s >= lo) & (ident_s < hi)])
                        for lo, hi, _ in BINS])
    for arm, _name, slot in ARMS:
        colour = style.PALETTE[slot]
        ax.plot(100 * ident_s, rel[arm], linestyle="none", marker="o", markersize=1.4,
                color=colour, alpha=0.16, markeredgewidth=0, zorder=2)
        pts = np.array([trend[f"{arm}|{lab}"][0] for _, _, lab in BINS])
        lo = np.array([trend[f"{arm}|{lab}"][1] for _, _, lab in BINS])
        hi = np.array([trend[f"{arm}|{lab}"][2] for _, _, lab in BINS])
        ax.errorbar(centres, pts, yerr=[pts - lo, hi - pts], color=colour,
                    marker="o", markersize=style.MARKER, linewidth=style.LINE,
                    elinewidth=style.LINE * 0.9, capsize=2,
                    markeredgecolor=style.INK["surface"], markeredgewidth=style.RING,
                    zorder=4)
    ax.axhline(1.0, color=style.INK["axis"], linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xlabel("best sequence identity to any training protein (%)")
    ax.set_ylabel("error, as a multiple of\nthis method's own average")
    # Linear, because a reader should not have to decode a log axis to see a flat line.
    # The cap is set from the data and the console reports what sits above it.
    ax.set_ylim(0.0, Y_CAP)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0][: int(Y_CAP / 0.5) + 1])
    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    style.finish(ax, xgrid=False)
    ax.yaxis.grid(True, color=style.INK["grid"], linewidth=0.5)
    ax.set_axisbelow(True)

    # The correlation IS the sentence, so it is printed and not left to the eye. The
    # arm name wears its own hue and doubles as the key -- style.py's rule for a slot
    # that cannot carry identity on contrast alone.
    for k, (arm, name, slot) in enumerate(ARMS):
        sig = "n.s." if rho[arm]["p"] >= 0.05 else f"p = {rho[arm]['p']:.2g}"
        ax.text(0.035, 0.965 - 0.085 * k,
                rf"{name}  $\rho$ = {rho[arm]['rho']:+.4f}  ({sig})",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                color=style.PALETTE[slot])
    style.panel_label(ax, "a")

    # Panel b draws a percentage change against a zero line meaning "no change": a bare
    # ratio of two RMSEs on an axis running 0.82 to 1.52 states no quantity a reader can
    # name. The RMSE each percentage is a percentage OF is printed beside the rows, so
    # the absolute size of the error is on the page and not only on the console.
    def pct(v):
        return 100.0 * (v - 1.0)

    labels = [style.nucleus_label(n) for n in NUCLEI]
    series = [{"name": name,
               "values": [pct(ratios[f"{arm}|{n}"]["ratio"]) for n in NUCLEI],
               "lo": [pct(ratios[f"{arm}|{n}"]["ci95"][0]) for n in NUCLEI],
               "hi": [pct(ratios[f"{arm}|{n}"]["ci95"][1]) for n in NUCLEI],
               "color": style.PALETTE[slot]} for arm, name, slot in ARMS]
    column = [" / ".join(f"{absolute[f'{arm}|{n}|<30']['rmse_ppm']:.2f}"
                         for arm, _, _ in ARMS) for n in NUCLEI]
    style.forest(ax2, labels, series,
                 xlabel="change in error for proteins that have a close relative\n"
                        "in training, against those that have none (%)",
                 value_column=column,
                 value_header="RMSE (ppm) with no\nclose relative,\n"
                              + " / ".join(name for _, name, _ in ARMS))
    ax2.axvline(0.0, color=style.INK["axis"], linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
    ax2.set_xlim(-20, 55)
    ax2.set_xticks([-20, -10, 0, 10, 20, 30, 40, 50])
    ax2.tick_params(axis="y", length=0)
    style.panel_label(ax2, "b")
    return fig


# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=str(CELLS))
    ap.add_argument("--master-cells", type=Path, default=MASTER,
                    help="the frozen, fenced master cell file the cells are restricted to")
    ns = ap.parse_args()

    global Y_CAP
    rng = np.random.default_rng(SEED)
    d, off_master = load_cells(Path(ns.cells), ns.master_cells)
    ident = json.loads(TRAIN_IDENT.read_text())
    tasks, ledger = coverage_ledger()

    sids = np.unique(d["sid"])
    missing = [s for s in sids if s not in ident]
    if missing:
        raise SystemExit(f"{len(missing)} scored structures have no training-identity "
                         f"value (e.g. {missing[:3]}); the axis does not cover the cohort")
    code = {s: i for i, s in enumerate(sids)}
    struct_code = np.array([code[s] for s in d["sid"]])
    ident_s = np.array([ident[s] for s in sids])
    ident_cell = ident_s[struct_code]

    rel = {arm: relative_error(d, arm, struct_code, sids.size) for arm, _, _ in ARMS}
    rho = {arm: spearman_ci(ident_s, rel[arm], rng) for arm, _, _ in ARMS}

    trend, ratios, absolute = {}, {}, {}
    for arm, _, _ in ARMS:
        for lo, hi, lab in BINS:
            sel = (ident_s >= lo) & (ident_s < hi)
            trend[f"{arm}|{lab}"] = median_ci(rel[arm][sel], rng)
        for nuc in NUCLEI:
            ratios[f"{arm}|{nuc}"] = ratio_to_low_bin(
                d, arm, ATOM_OF[nuc], ident_cell, struct_code, 0.70, 1.001, rng)
            for lo, hi, lab in BINS:
                absolute[f"{arm}|{nuc}|{lab}"] = ratio_to_low_bin(
                    d, arm, ATOM_OF[nuc], ident_cell, struct_code, lo, hi, rng)

    # Linear ceiling for panel a, chosen so the visible range covers all but the very
    # top of the distribution; whatever sits above it is counted and printed.
    pooled = np.concatenate([rel[arm] for arm, _, _ in ARMS])
    Y_CAP = float(min(3.0, 0.5 * np.ceil(np.percentile(pooled, 99.5) / 0.5)))
    off_scale = {arm: int((rel[arm] > Y_CAP).sum()) for arm, _, _ in ARMS}

    fig = draw(ident_s, rel, rho, trend, ratios, absolute)

    cohort_source = Counter(tasks[s]["source"] for s in sids)
    n_struct, n_cells = int(sids.size), int(scored_cells(d).sum())
    frozen.emit(fig, "fig_homology_training_identity")
    print(json.dumps({
        "cohort": {"structure_instances": n_struct, "by_source": dict(cohort_source),
                   "common_backbone_cells": n_cells,
                   "label_sets": int(np.unique(d["label"]).size),
                   "cells_of_the_dump_not_in_the_master_set": off_master,
                   "procs15_coverage": ledger},
        "spearman_by_arm": rho,
        "median_relative_error_by_bin": trend,
        "ratio_ge70_over_lt30_by_nucleus": ratios,
        "percent_change_ge70_vs_lt30_by_nucleus": {
            k: {"pct": 100.0 * (v["ratio"] - 1.0),
                "pct_ci95": [100.0 * (v["ci95"][0] - 1.0), 100.0 * (v["ci95"][1] - 1.0)]}
            for k, v in ratios.items()},
        "rmse_ppm_by_nucleus_and_bin": absolute,
        # Panel b's claim: the nuclei whose RMSE changes significantly between DeepShift's
        # least- and most-related proteins, Bonferroni over its six nuclei.
        "deepshift_changed_bonferroni": [
            nuc for nuc in NUCLEI
            if bonferroni(ratios[f"deepshift|{nuc}"]["k_tail"],
                          ratios[f"deepshift|{nuc}"]["resamples"], len(NUCLEI))["significant"]],
        "panel_a_axis": {"linear_ceiling": Y_CAP, "points_above_ceiling": off_scale,
                         "points_per_method": n_struct},
        "median_best_identity": float(np.median(ident_s)),
        "fraction_with_no_relative": float((ident_s <= 1e-9).mean())}, indent=1))


if __name__ == "__main__":
    main()
