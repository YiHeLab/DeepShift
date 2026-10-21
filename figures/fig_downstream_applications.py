#!/usr/bin/env python3
"""Two downstream uses of the shipped model: ensemble accuracy, and state discrimination.

Panel (a): per-ensemble, per-nucleus paired accuracy on the public IDP8 ensembles --
DeepShift mean ensemble RMSD against the published SPARTA+ RMSD, both ppm.
Panel (b): survival curves of the per-atom torsion-sensitivity statistic S on the tau
benchmark, over the residue--atom sites shared by DeepShift, UCBShift 2.0, MLFF and
SHIFTX2.

Every count drawn is measured from the rows read here.

Usage:
    DEEPSHIFT_IDP8_ROOT=... DEEPSHIFT_TAU_BENCHMARK_ROOT=... \\
        python figures/fig_downstream_applications.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import frozen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import style

STEM = "fig_downstream_applications"

IDP8_SCORES = frozen.env_path("DEEPSHIFT_IDP8_ROOT", "scores_audited.json")
TAU_PER_ATOM = frozen.env_path("DEEPSHIFT_TAU_BENCHMARK_ROOT", "common_cells.csv")

# The sensitivity criterion the tau benchmark defines a state-discriminating site by.
THRESHOLD = 3.29

# Every panel-(a) mark is DeepShift, so the three nuclei are separated by marker shape
# and not by hue: lending amber or violet to a nucleus would read as a competitor.
IDP8_NUCLEI = (("CB", r"C$\beta$", "o"), ("C", r"C$^{\prime}$", "s"),
               ("CA", r"C$\alpha^{*}$", "^"))

# Panel (b) is four methods, so here the hues are the role table. MLFF has no family of
# its own and takes the neutral; no SPARTA+ series is on this panel to collide with it.
TAU_METHODS = (
    ("DeepShift_original", "DeepShift", style.ACCENT, "o"),
    ("UCBShift2_combined_selfexcl", "UCBShift 2.0", style.AMBER, "s"),
    ("SHIFTX2_release", "SHIFTX2", style.VIOLET, "D"),
    ("MLFF_MACE_relaxed", "MLFF", style.NEUTRAL, "^"),
)


def build_panel_a(ax, frame: pd.DataFrame) -> dict:
    """Per-ensemble, per-nucleus scatter: published SPARTA+ against DeepShift RMSD."""
    all_vals: list[float] = []
    idp8_values: dict[str, dict] = {}
    for atom, label, marker in IDP8_NUCLEI:
        block = frame.dropna(subset=[f"{atom}_rmsd", f"{atom}_published"])
        x = block[f"{atom}_published"].to_numpy(float)
        y = block[f"{atom}_rmsd"].to_numpy(float)
        wins, n = int((y < x).sum()), len(block)
        if n == 0:
            raise SystemExit(f"no scored ensemble carries both {atom} columns")
        idp8_values[atom] = {
            "label": label, "n_ensembles": n, "deepshift_wins": wins,
            "per_ensemble": {
                ens: {"published_spartaplus_rmsd_ppm": float(pub),
                      "deepshift_rmsd_ppm": float(ds)}
                for ens, pub, ds in zip(block["label"], x, y, strict=True)},
        }
        all_vals.extend(x.tolist())
        all_vals.extend(y.tolist())
        ax.plot(x, y, linestyle="none", marker=marker, markersize=style.MARKER + 0.6,
                markerfacecolor=style.ACCENT, markeredgecolor=style.INK["surface"],
                markeredgewidth=style.RING, label=f"{label}  {wins}/{n}", zorder=3)

    lo, hi = min(all_vals), max(all_vals)
    pad = 0.07 * (hi - lo)
    lo_lim, hi_lim = lo - pad, hi + pad
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color=style.INK["axis"], linewidth=0.9,
            zorder=1)
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_ylim(lo_lim, hi_lim)
    ax.set_aspect("equal")
    ax.set_xlabel("published SPARTA+ RMSE (ppm)")
    ax.set_ylabel("DeepShift RMSE (ppm)")
    style.finish(ax, xgrid=False, spines=("bottom", "left"))
    ax.grid(True, color=style.INK["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    style.legend(ax, loc="upper left", fontsize=6.4)
    ax.text(0.98, 0.03, r"$^{*}$C$\alpha$ partly entered Groups B/C refinement",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0,
            color=style.INK["muted"])
    style.panel_label(ax, "(a)", x=-0.1)
    return idp8_values


def build_panel_b(ax, df: pd.DataFrame) -> tuple[dict, int]:
    """Survival curves of S on the four-method common support.

    The support is the join-key set itself: every method carries one row for each of the
    same sites, or the four curves describe different populations and are not drawn.
    """
    all_vals: list[float] = []
    tau_values: dict[str, dict] = {}
    shared_keys: set[tuple[int, str]] | None = None
    for method_key, label, color, marker in TAU_METHODS:
        block = df[df.method.eq(method_key)].copy()
        keys = set(zip(block["position"].astype(int), block["atom"].astype(str),
                       strict=True))
        n = len(block)
        if n == 0:
            raise SystemExit(f"the tau table carries no rows for {method_key}")
        if len(keys) != n:
            raise SystemExit(f"tau rows for {method_key}: {n} rows over {len(keys)} "
                             "distinct sites, so a site is scored more than once")
        if shared_keys is None:
            shared_keys = keys
        elif keys != shared_keys:
            raise SystemExit(f"tau site keys differ for {method_key}: the four methods "
                             "are not scored on one common support")

        sensitivity = np.sort(block["sensitivity"].to_numpy(float))
        sensitive = int((sensitivity >= THRESHOLD).sum())
        percent = 100.0 * sensitive / n

        tau_values[method_key] = {
            "label": label, "n_common_sites": n, "sensitive": sensitive,
            "sensitive_percent": percent, "median_s": float(np.median(sensitivity)),
        }
        all_vals.extend(sensitivity.tolist())
        survival = np.arange(n, 0, -1, dtype=float) / n
        ax.step(sensitivity, survival, where="post", color=color, linewidth=1.35,
                label=f"{label}  {percent:.1f}%", zorder=3)
        ax.plot(THRESHOLD, sensitive / n, linestyle="none", marker=marker,
                markersize=style.MARKER + 0.5, markerfacecolor=color,
                markeredgecolor=style.INK["surface"], markeredgewidth=style.RING,
                zorder=4)

    support = len(shared_keys)
    lo, hi = min(all_vals), max(all_vals)
    ax.axvline(THRESHOLD, color=style.INK["muted"], linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(lo / 1.35, hi * 1.25)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"sensitivity threshold $S$")
    ax.set_ylabel(r"fraction of common sites with $S$ or more")
    style.finish(ax, xgrid=False, spines=("bottom", "left"))
    ax.grid(True, which="major", color=style.INK["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    style.legend(ax, loc="upper right", fontsize=6.2)
    ax.text(0.02, 0.03,
            rf"{support} shared residue–atom sites; line: $S={THRESHOLD}$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.9,
            color=style.INK["muted"])
    style.panel_label(ax, "(b)", x=-0.1)
    return tau_values, support


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idp8-scores", type=Path, default=IDP8_SCORES)
    ap.add_argument("--tau-cells", type=Path, default=TAU_PER_ATOM)
    args = ap.parse_args()

    frame = pd.read_json(args.idp8_scores)
    tau_df = pd.read_csv(args.tau_cells)

    style.use_house_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(style.WIDTH["text"], 3.15))
    build_panel_a(ax_a, frame)
    build_panel_b(ax_b, tau_df)

    frozen.emit(fig, STEM)


if __name__ == "__main__":
    main()
