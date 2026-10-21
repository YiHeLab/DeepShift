#!/usr/bin/env python3
"""Side-chain accuracy against two competitors, uncentred -- deployed as fig_05_sidechain.

Both panels come from one report:

    python -m benchmarks.score_by_nucleus --tier 2 \\
        --tools ucbshift2_sameprot shiftx2_sameprot --out <scores>.json

That scorer owns every number drawn here -- the cells each nucleus is scored on, each
arm's RMSE, and the paired verdict against every competitor that emits the nucleus --
and this script recomputes none of them.

Grouped bars by nucleus, heavy nuclei (carbon and nitrogen) in the upper panel and
protons in the lower, after UCBShift 2.0 Fig. 3. Every bar is one side of a paired
comparison: a competitor's bar and the DeepShift bar beside it are scored on the same
cells, the cells both predict, which is what the report's ``vs_deepshift`` entry holds.
A competitor that covers less than the master set of a nucleus is scored on fewer cells
than DeepShift's own master-set RMSE, so where the two competitors' cells differ the
nucleus carries one DeepShift bar per competitor; where they coincide, one DeepShift bar
serves both. DeepShift's master-set RMSE is drawn only for a nucleus no competitor emits. The scope is every side-chain nucleus
DeepShift emits, not only the ones both incumbents also emit, so a nucleus with one
comparator carries two bars and a nucleus with none carries one; a caret under the tick
marks the nuclei no released competitor predicts. Residuals are raw -- prediction minus
the re-referenced observed label, no per-method constant removed.

The letters are b and c because the deployed float stacks fig_06_source above this one.

Exclusion tier 2, the one the paper reports: every retrieval arm is blocked from all
depositions of the query's own protein -- own BMRB, same-UniProt family, own-PDB-keyed
references. A report scored at another tier is refused.
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

# (arm base, legend label, colour)
ARMS = (("deepshift", "DeepShift", style.ACCENT),
        ("ucbshift2", "UCBShift 2.0", style.AMBER),
        ("shiftx2", "SHIFTX2", style.VIOLET))
TIERED = {"ucbshift2", "shiftx2"}     # the two arms whose key carries the tier
TIER = 2

_GREEK = {"A": r"\alpha", "B": r"\beta", "G": r"\gamma", "D": r"\delta",
          "E": r"\epsilon", "Z": r"\zeta", "H": r"\eta"}


def arm_key(base: str) -> str:
    return f"{base}_sameprot" if base in TIERED else base


def greek(name: str) -> str:
    """``'CD1'`` -> ``'C$^{\\delta}$1'``."""
    head, rest = name[0], name[1:]
    if rest and rest[0] in _GREEK:
        return f"{head}$^{{{_GREEK[rest[0]]}}}${rest[1:]}"
    return name


def read(path: Path) -> dict:
    """The scoring report, refused when it scored another exclusion tier."""
    rep = json.loads(path.read_text())
    if rep["exclusion_tier"] != TIER:
        raise SystemExit(f"{path} is tier {rep['exclusion_tier']}, not {TIER}")
    return rep


def paired_bars(node: dict) -> list:
    """``[(arm base, rmse), ...]`` in drawing order, every value from a paired comparison.

    Each competitor's RMSE and the DeepShift RMSE it is set beside come from the same
    ``vs_deepshift`` entry, so the pair is scored on one set of cells. DeepShift's bar is
    shared by the competitors whose paired DeepShift value is the same number (the same
    cells); a competitor scored on other cells gets its own DeepShift bar in front of it.
    A nucleus no competitor emits draws DeepShift's master-set RMSE alone.
    """
    paired = [(base, node["vs_deepshift"][arm_key(base)]["rmse"])
              for base, *_ in ARMS[1:] if arm_key(base) in node["vs_deepshift"]]
    if not paired:
        return [("deepshift", node["arms"]["deepshift"]["rmse"])]
    bars, shown = [], None
    for base, pair in paired:
        if shown is None or abs(pair["deepshift"] - shown) > 1e-12:
            bars.append(("deepshift", pair["deepshift"]))
            shown = pair["deepshift"]
        bars.append((base, pair["competitor"]))
    return bars


def draw(nodes: dict, heavy: list, protons: list):
    house_style()
    # The proton row carries the most nuclei, so it takes the smaller tick size; the
    # 45-degree rotation is what buys the room.
    fig, (ax_c, ax_h) = plt.subplots(2, 1, figsize=(style.WIDTH["text"], 4.35))
    for ax, nuclei, fontsize in ((ax_c, heavy, 8.0), (ax_h, protons, 7.2)):
        x = np.arange(len(nuclei))
        # The widest group sets the bar width, so every bar of the panel is one width.
        groups = [paired_bars(nodes[n]) for n in nuclei]
        width = 0.8 / max(len(g) for g in groups)
        drawn = {base: [] for base, *_ in ARMS}
        for xi, group in zip(x, groups):
            for i, (base, value) in enumerate(group):
                drawn[base].append((xi + (i - (len(group) - 1) / 2) * width, value))
        for base, label, color in ARMS:
            ax.bar([p for p, _v in drawn[base]], [v for _p, v in drawn[base]], width,
                   label=label, **style.bar_style(color))
        ax.set_xticks(x)
        ax.set_xticklabels([greek(n) for n in nuclei], rotation=45, ha="right",
                           fontsize=fontsize)
        for xi, n in zip(x, nuclei):
            if not nodes[n]["emitting_competitors"]:
                ax.plot(xi, 0, marker="^", markersize=3.0, color=style.INK["primary"],
                        clip_on=False, zorder=5)
        ax.set_ylabel("RMSE (ppm)")
        ax.set_xlim(-0.7, len(nuclei) - 0.3)
        ax.set_ylim(0, max(v for g in groups for _b, v in g) * 1.32)
        ax.grid(axis="y", lw=0.5, color=style.INK["grid"])
        ax.set_axisbelow(True)
        leg = ax.legend(ncol=3, loc="upper left", columnspacing=1.0, frameon=True,
                        framealpha=1.0, edgecolor=style.INK["grid"], fancybox=False,
                        handlelength=1.2)
        leg.get_frame().set_linewidth(0.6)
    for ax, letter in ((ax_c, "b"), (ax_h, "c")):
        panel_letter(ax, letter, dx=-0.055)
    fig.tight_layout(pad=0.5, h_pad=1.2)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True,
                    help="the side-chain scoring report this figure draws")
    args = ap.parse_args()
    path = args.scores
    rep = read(path)

    nodes = {n: v for n, v in rep["by_nucleus"].items() if not v["backbone"]}
    heavy = sorted(n for n in nodes if n[0] in "CN")
    protons = sorted(n for n in nodes if n[0] == "H")
    frozen.emit(draw(nodes, heavy, protons), "fig_05_sidechain")
    print(f"heavy panel {len(heavy)} nuclei, proton panel {len(protons)} nuclei")


if __name__ == "__main__":
    main()
