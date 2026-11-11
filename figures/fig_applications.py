#!/usr/bin/env python3
"""Figure 5: applications.

(a) Resonance assignment on the fold-6 benchmark: the percentage of residues assigned to
    their own position, residue and spin-system matching, instance mean with its 95%
    interval (``benchmarks/assignment_accuracy.py``, primary set, common sigma).
(b) X-EISD on the MIXED pool of the unfolded drkN SH3 domain, chemical shifts alone,
    1,000 restarts per forward model: the fraction of the selected 100 conformers drawn
    from the experimentally refined ENSEMBLE pool.
(c) The paired change in the X-EISD score of the seven data types the selection never saw
    when the shift table is swapped SHIFTX2 -> DeepShift, in total and by data type.
(d) IDP8: DeepShift's per-ensemble RMSE against the published SPARTA+ RMSE.
(e) Tau: per-site conformational sensitivity S on the sites DeepShift, UCBShift 2.0, MLFF
    and SHIFTX2 share.

(b) and (c) read the ``rescore_all`` indices and scores of one protocol run, where the
paired held-out scores live, and assert them against ``mixed_composition.json`` and
``results.json``; ``fig_xeisd_composition.py`` draws the composition from ``optimize_cs``.

Usage:
    DEEPSHIFT_ASSIGNMENT_ROOT=... DEEPSHIFT_XEISD_PROTOCOL=... DEEPSHIFT_IDP8_ROOT=... \\
        DEEPSHIFT_TAU_BENCHMARK_ROOT=... python figures/fig_applications.py
"""
from __future__ import annotations

import json
from pathlib import Path

import frozen
import journal_style as J
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STEM = "fig_05_applications"

ASSIGNMENT = frozen.env_path("DEEPSHIFT_ASSIGNMENT_ROOT", "assignment_core5.csv")
PROTOCOL = frozen.env_path("DEEPSHIFT_XEISD_PROTOCOL")
RESULTS = PROTOCOL / "results.json"
COMPOSITION = PROTOCOL / "mixed_composition.json"
RUNS = PROTOCOL / "runs"
IDP8_SCORES = frozen.env_path("DEEPSHIFT_IDP8_ROOT", "scores_audited.json")
TAU_PER_ATOM = frozen.env_path("DEEPSHIFT_TAU_BENCHMARK_ROOT", "common_cells.csv")

ASSIGNMENT_ARMS = (("deepshift", "DeepShift", J.ACCENT), ("ucbshift_full_sameprot", "UCBShift", J.AMBER),
                   ("ucbshift_x", "UCBShift-X", J.AMBER), ("sparta", "SPARTA+", J.NEUTRAL),
                   ("shiftx2_sameprot", "SHIFTX2", J.VIOLET), ("randomcoil", "random coil", J.FAINT))
XEISD_MODELS = (("shiftx2", "SHIFTX2", J.VIOLET), ("deepshift", "DeepShift", J.ACCENT))
# X-EISD writes results.csv without a header; the score columns by position.
SCORE_COLUMNS = {"saxs": 2, "cs": 3, "fret": 4, "jc": 5, "noe": 6, "pre": 7, "rdc": 8, "rh": 9}
NON_CS = ("saxs", "fret", "jc", "noe", "pre", "rdc", "rh")
TYPE_LABEL = {"saxs": "SAXS", "fret": "FRET", "jc": "J-coupling", "noe": "NOE", "pre": "PRE", "rdc": "RDC",
              "rh": r"$R_h$"}
# MIXED is 1,700 conformers from the optimized ENSEMBLE pool followed by 1,700 from the
# optimized RANDOM pool, so a selected index below 1,700 came from ENSEMBLE.
ENSEMBLE_BLOCK = 1700
BY_TYPE_SEED, BY_TYPE_BOOTS = 20260728, 2000
IDP8_NUCLEI = (("CB", r"C$\beta$", "o"), ("C", r"C$^{\prime}$", "s"), ("CA", r"C$\alpha$", "^"))
TAU_METHODS = (("DeepShift_original", "DeepShift", J.ACCENT, "o"), ("UCBShift2_combined_selfexcl", "UCBShift 2.0", J.AMBER, "s"),
               ("SHIFTX2_release", "SHIFTX2", J.VIOLET, "D"), ("MLFF_MACE_relaxed", "MLFF", J.NEUTRAL, "^"))
# The sensitivity criterion the tau benchmark defines a state-discriminating site by.
THRESHOLD = 3.29


def results_csv(model: str) -> Path:
    return RUNS / "mixed" / model / "rescore_all/results.csv"


def indices_csv(model: str) -> Path:
    return RUNS / "mixed" / model / "rescore_all/indices.csv"


def panel_assignment(axes, values):
    table = pd.read_csv(ASSIGNMENT)
    table = table[table.sigma.eq("common") & table.source.eq("pooled")]
    values["assignment"] = {}
    for ax, variant, title in zip(axes, ("residue", "spin_system"), ("Residue assignment", "Spin-system assignment")):
        b = table[table.variant.eq(variant)].set_index("arm")
        y = np.arange(len(ASSIGNMENT_ARMS))
        for yi, (arm, _, col) in zip(y, ASSIGNMENT_ARMS):
            r = b.loc[arm]
            ax.errorbar(100 * r.mean_frac, yi, xerr=[[100 * (r.mean_frac - r.ci_lo)], [100 * (r.ci_hi - r.mean_frac)]],
                        fmt="o", ms=3.2, color=col, mfc=col, mec="white", mew=0.4, elinewidth=0.8, capsize=0, zorder=3)
            if arm in ("deepshift", "ucbshift_full_sameprot"):
                ax.text(100 * r.ci_hi + 2.5, yi, f"{100 * r.mean_frac:.1f}", va="center", ha="left",
                        fontsize=J.FS["note"], color=J.SECOND)
        d = b.loc["ucbshift_x"]
        values["assignment"][variant] = {name: float(b.loc[arm].mean_frac) for arm, name, _ in ASSIGNMENT_ARMS}
        values["assignment"][variant]["deepshift_minus_ucbshift_x"] = [float(d.diff_vs_ref), float(d.diff_lo),
                                                                       float(d.diff_hi)]
        ax.set_yticks(y)
        ax.set_yticklabels([a[1] for a in ASSIGNMENT_ARMS])
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.set_ylim(len(ASSIGNMENT_ARMS) - 0.4, -0.6)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.grid(True, color=J.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_xlabel("residues assigned correctly (%)")
        ax.text(0.5, 1.04, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=J.FS["small"], color=J.SECOND)
        J.clean(ax, spines=("bottom",))
        ax.tick_params(axis="y", length=0)


def panel_composition(ax, values):
    shipped = json.loads(COMPOSITION.read_text())
    frac = {}
    for model, name, col in XEISD_MODELS:
        frac[model] = (np.loadtxt(indices_csv(model), delimiter=",") < ENSEMBLE_BLOCK).mean(axis=1)
        want = shipped["models"][model]["mean_ensemble_fraction"]
        if abs(frac[model].mean() - want) > 1e-4:
            raise SystemExit(f"{model}: ENSEMBLE fraction {frac[model].mean():.5f} is not the shipped {want}")
        ax.hist(frac[model], bins=np.arange(0.42, 0.92, 0.012), color=col, alpha=0.9, label=name, linewidth=0)
    values["composition"] = {m: {"mean_ensemble_fraction": float(f.mean()), "n_restarts": len(f)} for m, f in frac.items()}
    ax.axvline(0.50, color=J.MUTED, lw=0.7, ls=":")
    ax.axvline(shipped["published_reference"]["ensemble_fraction"], color=J.INK, lw=0.8, ls="--", alpha=0.75)
    ymax = ax.get_ylim()[1]
    ax.text(0.497, ymax * 0.62, "start\n50%", fontsize=J.FS["note"], color=J.MUTED, ha="right", va="center", linespacing=1.15)
    ax.text(0.768, ymax * 0.97, "published, all eight\nobservables jointly: 76%", fontsize=J.FS["note"], color=J.INK,
            ha="left", va="top", linespacing=1.15)
    ax.set_xlabel("ENSEMBLE fraction of the selected 100 conformers")
    ax.set_ylabel("optimizer restarts")
    ax.legend(loc="center", bbox_to_anchor=(0.45, 0.42), handlelength=0.9, handleheight=0.8, labelspacing=0.25,
              borderaxespad=0.1)
    ax.text(0.5, 1.04, "drkN SH3: shift-only ensemble selection", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=J.FS["small"], color=J.SECOND)
    J.clean(ax)


def panel_heldout(ax, values):
    pairs = {m: np.loadtxt(results_csv(m), delimiter=",", ndmin=2) for m, _, _ in XEISD_MODELS}
    cols = [SCORE_COLUMNS[t] for t in NON_CS]
    delta = pairs["deepshift"][:, cols].sum(1) - pairs["shiftx2"][:, cols].sum(1)
    ref = json.loads(RESULTS.read_text())["pools"]["MIXED"]["primary_non_cs_score"]
    if abs(delta.mean() - ref["mean_improvement"]) > 0.05:
        raise SystemExit(f"held-out change {delta.mean():.2f} is not the shipped {ref['mean_improvement']}")
    rng = np.random.default_rng(BY_TYPE_SEED)
    rows = [("All seven", float(delta.mean()), float(ref["ci95_low"]), float(ref["ci95_high"]))]
    for t in NON_CS:
        d = pairs["deepshift"][:, SCORE_COLUMNS[t]] - pairs["shiftx2"][:, SCORE_COLUMNS[t]]
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BY_TYPE_BOOTS)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append((TYPE_LABEL[t], float(d.mean()), float(lo), float(hi)))
    values["heldout"] = {r[0]: {"mean": r[1], "ci95": [r[2], r[3]]} for r in rows}
    values["heldout_fraction_restarts_improved"] = float((delta > 0).mean())
    y = np.arange(len(rows)).astype(float)
    y[1:] += 0.45
    for yi, (_, m, lo, hi) in zip(y, rows):
        sig = not (lo <= 0 <= hi)
        ax.errorbar([m], [yi], xerr=[[m - lo], [hi - m]], fmt="o", ms=3.4 if yi else 4.2, lw=0.9, capsize=1.6,
                    capthick=0.7, color=J.ACCENT if sig else J.NEUTRAL, mfc=J.ACCENT if (sig and m > 0) else "white",
                    mew=0.9, zorder=3)
    ax.axvline(0, color=J.INK, lw=0.7, ls="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_ylim(y[-1] + 0.6, -0.7)
    ax.set_xscale("symlog", linthresh=10, linscale=0.6)
    ax.set_xticks([-10, 0, 10, 100])
    ax.set_xticklabels(["-10", "0", "10", "100"])
    ax.set_xlim(-12, 160)
    ax.set_xlabel("change in X-EISD score\n(DeepShift minus SHIFTX2; symlog axis)", linespacing=1.15)
    ax.axhline(0.72, color=J.GRID, lw=0.6)
    ax.text(0.5, 1.04, "Fit to the seven data types not used in selection", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=J.FS["small"], color=J.SECOND)
    ax.text(0.99, 0.12, f"{(delta > 0).mean() * 100:.0f}% of restarts improved", transform=ax.transAxes, ha="right",
            va="center", fontsize=J.FS["note"], color=J.SECOND)
    J.clean(ax, spines=("bottom",))
    ax.tick_params(axis="y", length=0)


def panel_idp8(ax, values):
    frame = pd.read_json(IDP8_SCORES)
    allv, values["idp8"] = [], {}
    for atom, lab, mk in IDP8_NUCLEI:
        b = frame.dropna(subset=[f"{atom}_rmsd", f"{atom}_published"])
        xx, yy = b[f"{atom}_published"].to_numpy(float), b[f"{atom}_rmsd"].to_numpy(float)
        wins, n = int((yy < xx).sum()), len(b)
        values["idp8"][atom] = {"n_ensembles": n, "deepshift_wins": wins, "mean_published": float(xx.mean()),
                                "mean_deepshift": float(yy.mean())}
        allv += list(xx) + list(yy)
        ax.plot(xx, yy, mk, ms=3.2, mfc=J.ACCENT, mec="white", mew=0.4, linestyle="none", label=f"{lab}  {wins}/{n}",
                zorder=3)
    lo, hi = min(allv), max(allv)
    pad = 0.07 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=J.AXIS, lw=0.7, zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("published SPARTA+ RMSE (ppm)")
    ax.set_ylabel("DeepShift RMSE (ppm)")
    ax.grid(True, color=J.GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", handletextpad=0.2, labelspacing=0.2, borderaxespad=0.1)
    ax.text(0.5, 1.02, "IDP8 ensembles: per-ensemble RMSE", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=J.FS["small"], color=J.SECOND)
    J.clean(ax)


def panel_tau(ax, values):
    tau = pd.read_csv(TAU_PER_ATOM)
    values["tau"], keys0 = {}, None
    for key, lab, col, mk in TAU_METHODS:
        b = tau[tau.method.eq(key)]
        keys = set(zip(b.position.astype(int), b.atom.astype(str)))
        keys0 = keys0 or keys
        if keys != keys0 or len(keys) != len(b):
            raise SystemExit(f"{key}: its sites are not the sites every method shares")
        s = np.sort(b.sensitivity.to_numpy(float))
        n, k = len(s), int((s >= THRESHOLD).sum())
        values["tau"][lab] = {"n_sites": n, "sensitive": k, "percent": 100 * k / n}
        ax.step(s, np.arange(n, 0, -1) / n, where="post", color=col, lw=0.9, label=f"{lab}  {100 * k / n:.1f}%", zorder=3)
        ax.plot(THRESHOLD, k / n, mk, ms=3.0, mfc=col, mec="white", mew=0.4, zorder=4)
    ax.axvline(THRESHOLD, color=J.MUTED, lw=0.6, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlim(0.02, 60)
    ax.set_xlabel(r"conformational sensitivity $S$ (dashed: $S$ = 3.29)")
    ax.set_ylabel("fraction of sites with $S$ or more")
    ax.grid(True, color=J.GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", handlelength=1.2, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.1)
    ax.text(0.5, 1.02, f"Tau: conformational-state separation ({len(keys0)} sites)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=J.FS["small"], color=J.SECOND)
    J.clean(ax)


def main() -> None:
    J.use_journal_style()
    fig = plt.figure(figsize=(J.DOUBLE, 98 * J.MM))
    ax_a = fig.add_axes([0.118, 0.605, 0.152, 0.325])
    ax_a2 = fig.add_axes([0.118, 0.085, 0.152, 0.325])
    ax_b = fig.add_axes([0.370, 0.605, 0.255, 0.325])
    ax_c = fig.add_axes([0.745, 0.605, 0.240, 0.325])
    ax_d = fig.add_axes([0.370, 0.085, 0.255, 0.355])
    ax_e = fig.add_axes([0.745, 0.085, 0.240, 0.355])
    values = {}
    panel_assignment((ax_a, ax_a2), values)
    panel_composition(ax_b, values)
    panel_heldout(ax_c, values)
    panel_idp8(ax_d, values)
    panel_tau(ax_e, values)
    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"), (ax_e, "e")):
        x0 = 0.005 if letter == "a" else (0.29 if letter in "bd" else 0.665)
        fig.text(x0, ax.get_position().y1 + 0.045, letter, fontsize=J.FS["panel"], fontweight="bold", va="top")
    frozen.emit(fig, STEM)
    print(json.dumps(values, indent=1))


if __name__ == "__main__":
    main()
