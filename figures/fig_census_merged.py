#!/usr/bin/env python3
"""What the dataset is, in one panel: who supplies the coordinates, and how many
genuinely different proteins are underneath.

Two census panels on one canvas. They belong together because the top half is the reason
the bottom half is not obvious: a label set may be realized by up to three sources, so
the per-source counts sum well past the number of label sets, and a reader who only sees
the source bars will read the sum as the dataset size. It is not. Neither is the
label-set count, because label sets are related to each other -- closing the split's
leakage edges is what leaves the number of distinct proteins.

Every count is re-derived from one run's realization ledger (``status == "success"``) and
from the fold stage's published label-set edge tables, so the panel is drawn with the
grouping the split itself is atomic over and a rebuild invalidates it.

Usage:
    DEEPSHIFT_RUN=/path/to/a/run python figures/fig_census_merged.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import frozen
import numpy as np
import style

from deepshift.data import census as corpus

SOURCES = (("af", "AlphaFold"), ("nmr", "NMR"), ("xtal", "crystal"))

# Ink tints for the count rows. Hueless: the three categorical slots are spent on the
# three structure sources, and a fourth hue down here would read as a fourth source
# instead of as a different kind of quantity.
TINTS = ("#cfcec7", "#a5a39b", "#6e6c65")


# --------------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------------

def census(build: Path) -> dict:
    """Re-derive every census quantity from one build."""
    frame = corpus.read_realizations(build)
    label_sets = set(frame["label_set_id"])
    fold_of = corpus.read_fold_map(build)
    corpus.bind_fold_map(fold_of, label_sets)
    # Components over folds 1-6 only: fold 0 has no rows in the edge tables.
    in_edge_tables = corpus.edge_table_label_sets(fold_of)
    components, edges = corpus.protein_components(build, in_edge_tables)

    sizes = [len(members) for members in components.values()]
    counts = {
        "build": str(build),
        **corpus.source_census(frame),
        "distinct_proteins": len(components),
        "label_sets_in_components": len(in_edge_tables),
        "components_of_one": int(sum(1 for n in sizes if n == 1)),
        "largest_component": int(max(sizes)),
        "grouping": {"families": edges["families"],
                     "edge_group_counts": edges["edge_group_counts"]},
    }
    return counts



# --------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------

def draw(c: dict):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    style.use_house_style()
    fig = plt.figure(figsize=(style.WIDTH["column"], 3.35))
    top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=[3.0, 3.2])

    total = c["label_sets_realized"]
    xmax = total * 1.24
    slots = (style.PALETTE["slot1"], style.PALETTE["slot2"], style.PALETTE["slot3"])

    # (a) who supplies the coordinates. The solid head of each bar is the part of that
    # source's contribution nothing else covers -- the only reason a source is load
    # bearing and not redundant.
    y = np.arange(len(SOURCES), dtype=float)
    for row, (src, _name), color in zip(y, SOURCES, slots):
        whole, only = c["per_source_label_sets"][src], c["per_source_sole"][src]
        top.barh(row, whole, height=0.46, color=color, alpha=0.34, zorder=2)
        top.barh(row, only, height=0.46, color=color, zorder=3)
        # The value wears the series hue: for the aqua slot the label is the identity
        # channel, since the fill itself sits under 3:1 against the page.
        top.text(whole + xmax * 0.015, row, f"{whole:,}", va="center", ha="left",
                 fontsize=7.0, color=color)
        # The sole-source count goes inside its own segment, and outside when the
        # segment is too narrow to hold text.
        if only > xmax * 0.12:
            top.text(only / 2, row, f"{only:,}", va="center", ha="center",
                     fontsize=6.2, color=style.INK["surface"])
        else:
            top.text(only + xmax * 0.008, row, f"{only:,}", va="center", ha="left",
                     fontsize=6.2, color=style.INK["secondary"])
    top.set_yticks(y)
    top.set_yticklabels([name for _, name in SOURCES], color=style.INK["primary"])
    top.set_ylim(len(SOURCES) - 0.35, -0.75)
    style.finish(top)
    style.panel_label(top, "a", x=-0.30, y=1.02)
    style.legend(top, handles=[Patch(facecolor=style.INK["muted"],
                                     label="sole source for that label set"),
                               Patch(facecolor=style.INK["muted"], alpha=0.34,
                                     label="also realized by another source")],
                 loc="lower right", ncol=1, labelspacing=0.25, handlelength=1.1,
                 fontsize=6.0)

    # (b) what those bars do not say: how many label sets there are, and how many
    # different proteins that is. Same axis, so the protein count can be read against
    # the label-set count it is drawn from and against every source bar above it.
    rows = [
        ("realized\nlabel sets", total, None),
        ("BMRB\ndepositions", c["depositions"], TINTS[1]),
        ("distinct proteins\n(split components)\nin folds 1–6", c["distinct_proteins"], None),
    ]
    yb = np.arange(len(rows), dtype=float)

    # Row 0 is segmented by how many sources realize the label set: the multiplicity is
    # the whole reason the bars in (a) oversum, so it is shown where the oversum lands.
    left = 0.0
    for k, tint in zip(("1", "2", "3"), TINTS):
        width = c["multiplicity"][k]
        bottom.barh(0, width, left=left, height=0.46, color=tint,
                    edgecolor=style.INK["surface"], linewidth=0.8, zorder=2)
        bottom.text(left + width / 2, 0.35, f"{width:,}", ha="center", va="top",
                    fontsize=5.9, color=style.INK["muted"])
        left += width
    bottom.text(total + xmax * 0.015, 0, f"{total:,}", va="center", ha="left",
                fontsize=7.0, color=style.INK["primary"])

    # The bracket names the quantity the segments imply but do not print.
    x0, x1 = c["multiplicity"]["1"], total
    bottom.plot([x0, x0, x1, x1], [-0.30, -0.38, -0.38, -0.30], lw=0.6,
                color=style.INK["axis"], zorder=3, solid_capstyle="butt")
    bottom.text((x0 + x1) / 2, -0.45, f"$\\geq$2 sources: {c['multi_source']:,}",
                ha="center", va="bottom", fontsize=5.9, color=style.INK["secondary"])

    bottom.barh(1, c["depositions"], height=0.46, color=TINTS[1], zorder=2)
    bottom.text(c["depositions"] + xmax * 0.015, 1, f"{c['depositions']:,}",
                va="center", ha="left", fontsize=7.0, color=style.INK["primary"])

    # The headline row is the darkest thing on the page because it is the number a
    # reader should leave with: the dataset's size in proteins, not in rows.
    bottom.barh(2, c["distinct_proteins"], height=0.46, color=style.INK["primary"],
                zorder=2)
    bottom.text(c["distinct_proteins"] + xmax * 0.015, 2, f"{c['distinct_proteins']:,}",
                va="center", ha="left", fontsize=7.6, color=style.INK["primary"],
                fontweight="bold")

    bottom.set_yticks(yb)
    bottom.set_yticklabels([label for label, _, _ in rows], color=style.INK["primary"])
    bottom.set_ylim(len(rows) - 0.4, -0.85)
    bottom.set_xlim(0, xmax)
    bottom.set_xlabel("label sets")
    bottom.xaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    style.finish(bottom)
    style.panel_label(bottom, "b", x=-0.30, y=1.02)
    return fig


# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, default=frozen.env_path("DEEPSHIFT_RUN", "data"),
                    help="run's data directory the figure is drawn from")
    args = ap.parse_args()

    counts = census(args.build)
    fig = draw(counts)
    frozen.emit(fig, "fig_census_merged")


if __name__ == "__main__":
    main()
