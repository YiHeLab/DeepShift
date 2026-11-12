#!/usr/bin/env python3
"""Figure 1: the pipeline, from BMRB shifts to predictions and their uses.

(a) Seven steps, as a schematic in the manner of Extended Data Fig. 1, whose ray-traced
    protein and atom cloud two of the cards reuse. Its numbers are counted from the released
    tables in ``data/``: BMRB label sets, label sets and distinct structures with a
    structure, label sets per fold. The other pictures carry no data; the Predictions card
    is labelled "schematic data".
(b) Proteins with matched shifts and structure: RefDB (the files of UCBShift's RefDB tree,
    ``$DEEPSHIFT_REFDB_SHIFTS``, third-party) against the DeepShift dataset.
(c) Structure realizations (label set x source) by source.
(d) Label sets per fold.

Usage:
    DEEPSHIFT_REFDB_SHIFTS=... python figures/fig_pipeline.py
"""
from __future__ import annotations

import json

import frozen
import journal_style as J
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

STEM = "fig_01_pipeline"
DATA = frozen.REPO / "data"
REFDB = frozen.env_path("DEEPSHIFT_REFDB_SHIFTS")

# The census the paper prints (Supplementary Tables S1 and S2); the counts here must match it.
PRINTED = {"label_sets": 12710, "structures": 12896, "realizations": 21346, "fold0": 1615, "fold6": 1100}
SRC = {"xtal": "#475569", "nmr": "#94a3b8", "af": "#cbd5e1"}
SOURCES = (("xtal", "X-ray"), ("nmr", "NMR"), ("af", "AlphaFold"))
W, H = 183.0, 77.5                                   # canvas, mm
BW, GAP, TOP, BH = 22.6, 4.1, 74.0, 27.0             # step cards: width, gap, top edge, height
XS = [0.2 + i * (BW + GAP) for i in range(7)]
ICON = TOP - BH / 2                                  # centre line of a card
CARD, HIGHLIGHT = "#fbfcfd", "#e0f2f7"               # the card face of Extended Data Fig. 1, and its accent tint
RENDERS = frozen.FIGURES_DIR / "reference" / "renders"   # the ray-traced panels of Extended Data Fig. 1


def counts() -> dict:
    st = pd.read_parquet(DATA / "deepshift_structures.parquet")
    sh = pd.read_parquet(DATA / "deepshift_shifts.parquet", columns=["label_set_id", "fold"]).drop_duplicates("label_set_id")
    seq = pd.read_parquet(DATA / "deepshift_sequences.parquet", columns=["logical_entity_id"])
    folds = sh[sh.label_set_id.isin(st.label_set_id)].fold.value_counts().sort_index()
    c = {"label_sets": st.label_set_id.nunique(),
         "structures": st[["source", "accession"]].drop_duplicates().shape[0],
         "realizations": len(st), "by_source": st.source.value_counts().to_dict(),
         "multi_source_label_sets": int((st.groupby("label_set_id").source.nunique() >= 2).sum()),
         "folds": {int(k): int(v) for k, v in folds.items()}, "bmrb_label_sets": len(seq),
         "refdb_proteins": len(list(REFDB.iterdir()))}
    c["multi_source_pct"] = 100 * c["multi_source_label_sets"] / c["label_sets"]
    c["ratio_vs_refdb"] = c["label_sets"] / c["refdb_proteins"]
    got = {"label_sets": c["label_sets"], "structures": c["structures"], "realizations": c["realizations"],
           "fold0": c["folds"][0], "fold6": c["folds"][6]}
    if got != PRINTED or sum(c["folds"].values()) != c["label_sets"]:
        raise SystemExit(f"released tables {got} do not match the printed census {PRINTED}")
    return c


def main() -> None:
    c = counts()
    print(json.dumps(c, indent=1))
    by_src, folds = c["by_source"], c["folds"]
    J.use_journal_style()
    fig = plt.figure(figsize=(W * J.MM, H * J.MM))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    bg.axis("off")
    S_ = J.FS["note"]
    rng = np.random.default_rng(3)                  # schematic icons only

    def card(i, title, spec, focus=False):
        """One step in the manner of Extended Data Fig. 1: a card that holds a picture, its
        label and its figures below it, chevrons to the next card. Returns the centre x."""
        x = XS[i]
        bg.add_patch(FancyBboxPatch((x, TOP - BH), BW, BH, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=HIGHLIGHT if focus else CARD, ec=J.ACCENT if focus else J.AXIS,
                                    lw=0.9 if focus else 0.7))
        bg.text(x + BW / 2, TOP - BH - 2.2, title.upper(), fontsize=J.FS["small"] + 0.2, fontweight="bold",
                color=J.ACCENT if focus else J.SECOND, ha="center", va="top")
        bg.text(x + BW / 2, TOP - BH - 5.6, spec, fontsize=S_ + 0.4, color=J.MUTED, ha="center", va="top",
                linespacing=1.25)
        if i < 6:
            for k in (0, 1):
                xa = x + BW + GAP / 2 - 1.15 + k * 1.3
                bg.plot([xa, xa + 1.0, xa], [ICON + 1.35, ICON, ICON - 1.35], lw=1.1, color=J.AXIS,
                        solid_capstyle="round", solid_joinstyle="round")
        return x + BW / 2

    def render(i, name, pad=1.2):
        ax = fig.add_axes([(XS[i] + pad) / W, (TOP - BH + pad) / H, (BW - 2 * pad) / W, (BH - 2 * pad) / H])
        ax.imshow(plt.imread(RENDERS / f"{name}.png"), interpolation="antialiased")
        ax.set_axis_off()

    # 1. Deposited shifts, drawn as the peaks of a 1H-15N spectrum.
    cx = card(0, "BMRB shifts", f"{c['bmrb_label_sets']:,} label sets\nassigned 1H, 13C, 15N")
    t = rng.uniform(0, 1, 34)
    px = cx - 7 + 14 * (0.15 + 0.7 * t + rng.normal(0, 0.16, 34)).clip(0.05, 0.97)
    py = ICON - 8.5 + 17 * (0.2 + 0.6 * t + rng.normal(0, 0.2, 34)).clip(0.05, 0.97)
    bg.plot([cx - 8, cx - 8, cx + 8], [ICON + 9.5, ICON - 9.5, ICON - 9.5], color=J.AXIS, lw=0.6)
    bg.scatter(px, py, s=3.6, color=J.ACCENT, lw=0)

    # 2. The structures matched to them (the protein of Extended Data Fig. 1).
    card(1, "Structures", f"{c['label_sets']:,} label sets\n{c['structures']:,} structures")
    render(1, "structure")

    # 3. Re-referencing: the deposited peaks, moved by one offset.
    cx = card(2, "Re-referencing", "one offset per label\nset and nucleus")
    peaks, heights, shift = np.array([-7.0, -4.2, -2.7, 0.4, 2.3]), np.array([4.2, 6.6, 3.2, 5.7, 4.0]), 3.4
    for y0, dx, col in ((ICON + 2.6, 0.0, J.FAINT), (ICON - 9.5, shift, J.ACCENT)):
        bg.plot([cx - 8, cx + 8], [y0, y0], color=J.AXIS, lw=0.6)
        for p_, h_ in zip(peaks, heights):
            bg.plot([cx + p_ + dx, cx + p_ + dx], [y0, y0 + h_], color=col, lw=1.0, solid_capstyle="butt")
    bg.add_patch(FancyArrowPatch((cx + peaks[1] - 0.4, ICON + 0.6), (cx + peaks[1] + shift + 0.4, ICON + 0.6),
                                 arrowstyle="-|>", mutation_scale=4, lw=0.6, color=J.SECOND, shrinkA=0, shrinkB=0))

    # 4. Clusters of related proteins, and the folds they are dealt to (widths = label sets).
    cx = card(3, "Homology splits", "clusters to folds;\nfold 6 held out")
    for (dx, dy, r), col in zip(((-5.6, 6.6, 2.6), (0.6, 8.0, 2.0), (5.8, 5.6, 2.4), (-2.4, 1.2, 1.9), (3.2, 0.4, 2.3)),
                                (J.FAINT, J.MUTED, J.FAINT, J.MUTED, J.FAINT)):
        bg.add_patch(Circle((cx + dx, ICON + dy), r, fc="white", ec=col, lw=0.6))
        bg.scatter(cx + dx + rng.normal(0, r / 3.2, 4), ICON + dy + rng.normal(0, r / 3.2, 4), s=1.4, color=col, lw=0)
    x0, wtot, total = cx - 8.8, 17.6, sum(folds.values())
    for f in range(7):
        w = wtot * folds[f] / total
        bg.add_patch(Rectangle((x0, ICON - 9.2), w, 3.8, fc={0: J.FAINT, 6: J.INK}.get(f, J.LIGHT_ACCENT), ec=CARD, lw=0.5))
        bg.text(x0 + w / 2, ICON - 7.35, str(f), fontsize=S_ - 0.6, ha="center", va="center",
                color="white" if f == 6 else J.INK)
        x0 += w

    # 5. The model's input: the atoms within 8 A of one residue (the cloud of Extended Data Fig. 1).
    card(4, "DeepShift", "atom types and\ndistances within 8 Å;\n5-model ensemble", focus=True)
    render(4, "atom_cloud")

    # 6. A predicted shift and its interval along a chain (schematic).
    cx = card(5, "Predictions", "every atom, 64 types;\n90% intervals\n(schematic data)")
    xx = np.linspace(-8, 8, 60)
    mu = 3.4 * np.sin(xx / 2.1) + 1.4 * np.sin(xx / 0.9 + 1)
    bg.plot([cx - 8, cx - 8, cx + 8], [ICON + 9.5, ICON - 9.5, ICON - 9.5], color=J.AXIS, lw=0.6)
    bg.fill_between(cx + xx, ICON + mu - 2.2, ICON + mu + 2.2, color=J.LIGHT_ACCENT, lw=0)
    bg.plot(cx + xx, ICON + mu, color=J.ACCENT, lw=0.9)

    # 7. What the predictions are used for.
    cx = card(6, "Applications", "")
    for k, lab in enumerate(("Resonance\nassignment", "Structure\nvalidation", "Ensemble\nreweighting")):
        y = TOP - 9.0 - k * 8.3
        bg.add_patch(FancyBboxPatch((cx - 9.2, y), 18.4, 6.9, boxstyle="round,pad=0,rounding_size=0.9", fc="white",
                                    ec=J.AXIS, lw=0.6))
        bg.text(cx, y + 3.45, lab, fontsize=J.FS["small"], ha="center", va="center", color=J.SECOND, linespacing=1.15)

    n_refdb, n_ls, n_real = c["refdb_proteins"], c["label_sets"], c["realizations"]
    Y0, AH = 8.5 / H, 15.0 / H

    def thousands(ax, ticks):
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}" for t in ticks])

    a1 = fig.add_axes([0.068, Y0, 0.205, AH])
    a1.barh([1, 0], [n_refdb, n_ls], color=[J.NEUTRAL, J.ACCENT], height=0.64)
    a1.set_yticks([1, 0])
    a1.set_yticklabels(["RefDB", "DeepShift"], color=J.INK)
    a1.tick_params(axis="y", length=0)
    a1.text(n_refdb + 300, 1, f"{n_refdb:,}", va="center", fontsize=J.FS["small"], color=J.SECOND)
    a1.text(n_ls + 300, 0, f"{n_ls:,}", va="center", fontsize=J.FS["small"], color=J.SECOND)
    a1.text(n_ls + 300, 0.62, f"{c['ratio_vs_refdb']:.1f}×", va="center", fontsize=J.FS["panel"] - 0.5, color=J.INK,
            fontweight="bold")
    a1.set_xlim(0, 16500)
    a1.set_ylim(-0.5, 1.5)
    thousands(a1, [0, 5000, 10000])
    a1.set_xlabel("proteins with matched shifts and structure")
    J.clean(a1)

    a2 = fig.add_axes([0.345, Y0, 0.265, AH])
    left = 0
    for src, lab in SOURCES:
        v = by_src[src]
        a2.barh(0, v, left=left, color=SRC[src], height=0.74, ec="white", lw=0.8)
        a2.text(left + v / 2, 0, f"{lab}\n{v:,}", ha="center", va="center", fontsize=J.FS["small"],
                color="white" if src == "xtal" else J.INK, linespacing=1.2)
        left += v
    a2.set_xlim(0, n_real)
    a2.set_ylim(-0.5, 1.0)
    a2.set_yticks([])
    thousands(a2, [0, 5000, 10000, 15000, 20000])
    a2.set_xlabel(f"structure realizations (label set × source), {n_real:,} in total")
    J.clean(a2, spines=("bottom",))

    a3 = fig.add_axes([0.680, Y0, 0.308, AH])
    left, edges = 0, [0]
    for f in range(7):
        v = folds[f]
        a3.barh(0, v, left=left, color={0: J.FAINT, 6: J.INK}.get(f, J.LIGHT_ACCENT), height=0.74, ec="white", lw=0.8)
        a3.text(left + v / 2, 0, f"{f}\n{v:,}", ha="center", va="center", fontsize=S_ + 0.3 if f == 6 else J.FS["small"],
                color="white" if f == 6 else J.INK, linespacing=1.2)
        left += v
        edges.append(left)
    for lo, hi, lab in ((edges[0], edges[1], "excluded"), (edges[1], edges[6], "training and cross-validation"),
                        (edges[6], edges[7], "test")):
        a3.plot([lo + 60, hi - 60], [0.52, 0.52], color=J.FAINT, lw=0.5, clip_on=False)
        a3.text((lo + hi) / 2, 0.60, lab, ha="center", va="bottom", fontsize=J.FS["small"], color=J.SECOND)
    a3.set_xlim(0, n_ls)
    a3.set_ylim(-0.5, 1.0)
    a3.set_yticks([])
    thousands(a3, [0, 4000, 8000, 12000])
    a3.set_xlabel("label sets per fold")
    J.clean(a3, spines=("bottom",))

    for x, y, letter in ((0.002, 0.995, "a"), (0.002, 0.375, "b"), (0.325, 0.375, "c"), (0.660, 0.375, "d")):
        fig.text(x, y, letter, fontsize=J.FS["panel"], fontweight="bold", va="top")
    frozen.emit(fig, STEM)


if __name__ == "__main__":
    main()
