#!/usr/bin/env python3
"""The same shift predicted from three kinds of coordinates -- deployed as fig_06_source.

Cohort: cross-validation folds 1-5, every protein scored by the one ensemble member that
held its fold out. That is the design of the five-member cross-validation, not a second
test set.

Matching: a site enters only when the identical (label residue, nucleus) was predicted
from an AlphaFold model, a crystal structure and an NMR conformer, so the observed
value is literally the same number in all three columns and the coordinates are the only
thing that differs.

Residuals are raw -- prediction minus the re-referenced observed label, no centring. The
physical-range fence is applied to the observed value. Intervals are a cluster bootstrap
over label sets.

``benchmarks/score_source_matched.py`` computes all of that and writes the report drawn
here; this script reads it and defines no statistic:

    python -m benchmarks.score_source_matched --out <out>/source_matched.json
    cd figures && python fig_source_matched.py --scores <out>/source_matched.json

The letter is a because the deployed float stacks fig_05_sidechain below.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import frozen
import matplotlib.pyplot as plt
import numpy as np
import style

# The manuscript packet's rcParams; see fig_backbone_benchmark for why this is shared.
from build_paper_figures import house_style, panel_letter

ORDER = [("n_shift", r"$^{15}$N"), ("ca_shift", r"$^{13}$C$\alpha$"),
         ("cb_shift", r"$^{13}$C$\beta$"), ("c_shift", r"$^{13}$C$^{\prime}$"),
         ("h_shift", r"$^{1}$H$^{N}$"), ("ha_shift", r"$^{1}$H$\alpha$")]
# Structure sources, not predictors, but they take the same three validated hues so the
# paper reads as one document; ACCENT is AlphaFold wherever the split is by source. The
# report keys these by the package's source names, and the display names live here.
SRC = [("af", "AlphaFold", style.ACCENT), ("xtal", "crystal", style.VIOLET),
       ("nmr", "NMR", style.AMBER)]


def draw(nuclei: dict):
    house_style()
    fig, ax = plt.subplots(figsize=(style.WIDTH["text"], 2.75))
    x = np.arange(len(ORDER))
    width = 0.66 / len(SRC)
    for i, (source, name, color) in enumerate(SRC):
        off = (i - (len(SRC) - 1) / 2) * width
        v = np.array([nuclei[a]["per_source"][source]["rmse"] for a, _l in ORDER])
        ci = np.array([nuclei[a]["per_source"][source]["rmse_ci95"] for a, _l in ORDER]).T
        ax.bar(x + off, v, width, label=name, **style.bar_style(color))
        ax.errorbar(x + off, v, yerr=np.abs(ci - v), fmt="none",
                    ecolor=style.INK["primary"], elinewidth=0.5, capsize=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _a, lab in ORDER])
    ax.set_ylabel("RMSE (ppm)")
    ax.set_ylim(0, max(nuclei[a]["per_source"][source]["rmse_ci95"][1]
                       for a, _l in ORDER for source, _n, _c in SRC) * 1.3)
    leg = ax.legend(ncol=3, loc="upper center", columnspacing=1.0, frameon=True,
                    framealpha=1.0, edgecolor=style.INK["grid"], fancybox=False,
                    handlelength=1.2)
    leg.get_frame().set_linewidth(0.6)
    ax.grid(axis="y", lw=0.5, color=style.INK["grid"])
    ax.set_axisbelow(True)
    panel_letter(ax, "a", dx=-0.055)
    fig.tight_layout(pad=0.5)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True,
                    help="the report benchmarks/score_source_matched.py writes")
    report = ap.parse_args().scores
    rep = json.loads(report.read_text())
    nuclei = rep["nuclei"]
    frozen.emit(draw(nuclei), "fig_06_source")

    print(f"{'nucleus':10s}{'sites':>8s}{'labelsets':>10s}"
          + "".join(f"{name:>16s}" for _s, name, _c in SRC))
    for a, _l in ORDER:
        node = nuclei[a]
        print(f"{a:10s}{node['n_matched_sites']:8d}{node['n_label_sets']:10d}"
              + "".join(f"{node['per_source'][source]['rmse']:16.4f}"
                        for source, _n, _c in SRC))


if __name__ == "__main__":
    main()
