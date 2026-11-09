#!/usr/bin/env python3
"""Figure 1: the pipeline, from BMRB shifts to predictions and their uses.

(a) Seven steps, as a schematic. Its numbers are counted from the released tables in
    ``data/``: BMRB label sets, label sets and distinct structures with a structure,
    realizations by source, label sets with two or three sources, label sets per fold.
    The icons carry no data; panel 6 is labelled "schematic data".
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
H = 98.0                                            # canvas height, mm
BOX_Y, BOX_H, BW, GAP = 44.0, 51.0, 22.0, 4.83      # step boxes, mm
XS = [i * (BW + GAP) for i in range(7)]


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
    fig = plt.figure(figsize=(J.DOUBLE, H * J.MM))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 183)
    bg.set_ylim(0, H)
    bg.axis("off")
    S_ = J.FS["note"]

    def box(i, title, fill="#f1f5f9", edge=J.AXIS, tc=J.INK):
        x = XS[i]
        bg.add_patch(FancyBboxPatch((x, BOX_Y), BW, BOX_H, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=edge, lw=0.6))
        bg.text(x + 1.4, BOX_Y + BOX_H - 1.0, str(i + 1), fontsize=J.FS["panel"] - 1, fontweight="bold",
                color=J.MUTED if fill != J.ACCENT else "white", va="top", ha="left")
        bg.text(x + BW / 2, BOX_Y + BOX_H - 8.2, title, fontsize=J.FS["text"] + 0.3, fontweight="bold", color=tc,
                ha="center", va="center", linespacing=1.1)
        if i < 6:
            bg.add_patch(FancyArrowPatch((x + BW + 0.4, BOX_Y + BOX_H / 2), (XS[i + 1] - 0.4, BOX_Y + BOX_H / 2),
                                         arrowstyle="-|>", mutation_scale=5, lw=0.8, color=J.MUTED))

    def txt(i, s, y, color=J.SECOND, weight="normal"):
        bg.text(XS[i] + BW / 2, y, s, fontsize=S_ + 0.3, color=color, ha="center", va="top", linespacing=1.2,
                fontweight=weight)

    rng = np.random.default_rng(3)                  # schematic icons only

    box(0, "BMRB\ndeposited shifts")
    gx, gy = rng.uniform(XS[0] + 4, XS[0] + BW - 4, 26), rng.uniform(BOX_Y + 25, BOX_Y + 36, 26)
    bg.scatter(gx, gy, s=3, color=J.MUTED, lw=0)
    txt(0, f"{c['bmrb_label_sets']:,}\nBMRB label sets", BOX_Y + 22.5, color=J.INK, weight="bold")
    txt(0, "assigned 1H, 13C, 15N\nshifts for one protein", BOX_Y + 14.5)

    box(1, "Matched\nstructures")
    for k, (key, lab) in enumerate((("xtal", "X-ray"), ("nmr", "NMR"), ("af", "AlphaFold"))):
        y = BOX_Y + 33.5 - k * 5.2
        bg.add_patch(FancyBboxPatch((XS[1] + 2.5, y), BW - 5, 4.0, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=SRC[key], ec="none"))
        bg.text(XS[1] + BW / 2, y + 2.0, f"{lab}  {by_src[key]:,}", fontsize=S_, ha="center", va="center",
                color="white" if key == "xtal" else J.INK)
    txt(1, f"{c['label_sets']:,} label sets\n{c['structures']:,} structures", BOX_Y + 17.5, color=J.INK, weight="bold")
    txt(1, f"{c['multi_source_pct']:.0f}% matched to\n2 or 3 sources", BOX_Y + 9.5)

    box(2, "Re-\nreferencing")
    base = BOX_Y + 25
    for p in (4.5, 8.5, 12.5, 17.0):
        bg.plot([XS[2] + p, XS[2] + p], [base + 6.5, base + 11.5 + (p % 3)], color=J.FAINT, lw=0.9)
        bg.plot([XS[2] + p + 1.6, XS[2] + p + 1.6], [base, base + 5 + (p % 3)], color=J.ACCENT, lw=0.9)
    txt(2, "one offset per label\nset and nucleus", BOX_Y + 21.5, color=J.INK, weight="bold")
    txt(2, "RefDB method,\nextended to all usable\nBMRB entries", BOX_Y + 13.5)

    box(3, "Homology-\naudited splits")
    clusters = [(5.5, 33.5, 2.6), (11.5, 35.5, 2.0), (17.0, 32.5, 2.4), (8.5, 28.5, 1.8), (14.5, 28.0, 2.5)]
    for (cx, cy, r), col in zip(clusters, (J.FAINT, J.MUTED, J.FAINT, J.MUTED, J.FAINT)):
        bg.add_patch(Circle((XS[3] + cx, BOX_Y + cy), r, fc="none", ec=col, lw=0.7))
        bg.scatter(XS[3] + cx + rng.normal(0, r / 3, 4), BOX_Y + cy + rng.normal(0, r / 3, 4), s=1.2, color=col, lw=0)
    x0, wtot, total = XS[3] + 2.5, BW - 5, sum(folds.values())
    for f in range(7):                              # the fold strip, widths proportional to label sets
        w = wtot * folds[f] / total
        bg.add_patch(Rectangle((x0, BOX_Y + 20.5), w, 3.2, fc={0: J.FAINT, 6: J.INK}.get(f, J.LIGHT_ACCENT), ec="white", lw=0.4))
        bg.text(x0 + w / 2, BOX_Y + 22.1, str(f), fontsize=S_ - 0.3, ha="center", va="center",
                color="white" if f == 6 else J.INK)
        x0 += w
    txt(3, "clusters to folds;\nfold 6 held out;\nfold 0 excludes the\nUCBShift test set", BOX_Y + 18.0)

    box(4, "DeepShift", fill=J.ACCENT, edge=J.ACCENT, tc="white")
    cx0, cy0 = XS[4] + BW / 2, BOX_Y + 31.0
    ang, rad = rng.uniform(0, 2 * np.pi, 16), 6.5 * np.sqrt(rng.uniform(0.05, 1, 16))
    px, py = cx0 + rad * np.cos(ang) * 1.3, cy0 + rad * np.sin(ang) * 0.9
    for i1 in range(0, 14, 2):
        bg.plot([px[i1], px[i1 + 1]], [py[i1], py[i1 + 1]], color="white", lw=0.3, alpha=0.6)
    bg.scatter(px, py, s=3.5, color="white", lw=0)
    txt(4, "atom types and\ninteratomic distances\nwithin 8 Å", BOX_Y + 21.5, color="white", weight="bold")
    txt(4, "transformer;\n5-model ensemble\n(CV folds 1–5)", BOX_Y + 11.5, color="white")

    box(5, "Per-atom\npredictions")
    axp = fig.add_axes([(XS[5] + 3) / 183, (BOX_Y + 22.5) / H, (BW - 6) / 183, 11.5 / H])
    xx = np.arange(14)
    mu = 2.0 * np.sin(xx / 2.3) + np.random.default_rng(5).normal(0, 0.15, 14)
    axp.fill_between(xx, mu - 1.0, mu + 1.0, color=J.LIGHT_ACCENT, lw=0)
    axp.plot(xx, mu, color=J.ACCENT, lw=0.8)
    axp.set_xticks([])
    axp.set_yticks([])
    J.clean(axp, spines=("left", "bottom"))
    axp.patch.set_alpha(0)
    for s_ in axp.spines.values():
        s_.set_color(J.FAINT)
    txt(5, "shift of every atom\n(64 nucleus types)\nwith 90% prediction\nintervals", BOX_Y + 19.0, color=J.INK)
    bg.text(XS[5] + BW / 2, BOX_Y + 3.0, "schematic data", fontsize=S_ - 0.5, color=J.FAINT, ha="center", va="center",
            style="italic")

    box(6, "Applications")
    for k, lab in enumerate(("Resonance\nassignment", "Structure\nvalidation", "Ensemble\nreweighting")):
        y = BOX_Y + 28.5 - k * 11.5
        bg.add_patch(FancyBboxPatch((XS[6] + 2.2, y), BW - 4.4, 9.2, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc="white", ec=J.ACCENT, lw=0.7))
        bg.text(XS[6] + BW / 2, y + 4.6, lab, fontsize=S_ + 0.3, ha="center", va="center", color=J.INK, linespacing=1.1)

    AX_Y, AX_H = 0.07, 0.31
    n_refdb, n_ls, n_real = c["refdb_proteins"], c["label_sets"], c["realizations"]
    a1 = fig.add_axes([0.075, AX_Y + 0.02, 0.17, AX_H - 0.07])
    a1.barh([1, 0], [n_refdb, n_ls], color=[J.NEUTRAL, J.ACCENT], height=0.62)
    a1.set_yticks([1, 0])
    a1.set_yticklabels(["RefDB", "DeepShift"])
    for y_, v in ((1, n_refdb), (0, n_ls)):
        a1.text(v + 250, y_, f"{v:,}", va="center", fontsize=S_ + 0.3, color=J.INK)
    a1.set_xlim(0, 17500)
    a1.set_xticks([0, 5000, 10000])
    a1.set_xticklabels(["0", "5,000", "10,000"])
    a1.text(15000, 0.55, f"{c['ratio_vs_refdb']:.1f}×", fontsize=J.FS["panel"] - 1, color=J.ACCENT, fontweight="bold",
            ha="center", va="center")
    a1.set_xlabel("proteins with matched shifts and structure")
    J.clean(a1)

    a2 = fig.add_axes([0.385, AX_Y + 0.02, 0.215, AX_H - 0.07])
    left = 0
    for key, lab in (("xtal", "X-ray"), ("nmr", "NMR"), ("af", "AlphaFold")):
        v = by_src[key]
        a2.barh(0, v, left=left, color=SRC[key], height=0.55, ec="white", lw=0.4)
        a2.text(left + v / 2, 0, f"{lab}\n{v:,}", ha="center", va="center", fontsize=S_,
                color="white" if key == "xtal" else J.INK, linespacing=1.1)
        left += v
    a2.set_xlim(0, n_real)
    a2.set_ylim(-0.6, 0.6)
    a2.set_yticks([])
    a2.set_xticks([0, 5000, 10000, 15000, 20000])
    a2.set_xticklabels(["0", "5,000", "10,000", "15,000", "20,000"])
    a2.set_xlabel(f"structure realizations (label set × source), {n_real:,} in total")
    J.clean(a2, spines=("bottom",))
    a2.text(0.5, 1.02, "structure sources", transform=a2.transAxes, ha="center", va="bottom", fontsize=S_ + 0.3,
            color=J.SECOND)

    a3 = fig.add_axes([0.70, AX_Y + 0.02, 0.285, AX_H - 0.07])
    left = 0
    for f in range(7):
        v = folds[f]
        a3.barh(0, v, left=left, color={0: J.FAINT, 6: J.INK}.get(f, J.LIGHT_ACCENT), height=0.55, ec="white", lw=0.5)
        a3.text(left + v / 2, 0, f"{f} · {v:,}" if f == 6 else f"{f}\n{v:,}", ha="center", va="center", fontsize=S_,
                color="white" if f == 6 else J.INK, linespacing=1.1, rotation=90 if f == 6 else 0)
        left += v
    a3.set_xlim(0, n_ls)
    a3.set_ylim(-0.6, 0.6)
    a3.set_yticks([])
    a3.set_xticks([0, 4000, 8000, 12000])
    a3.set_xticklabels(["0", "4,000", "8,000", "12,000"])
    a3.set_xlabel("label sets per fold")
    J.clean(a3, spines=("bottom",))
    a3.text(folds[0] / 2, 0.68, "excluded", ha="center", va="bottom", fontsize=S_, color=J.MUTED, clip_on=False)
    a3.text(folds[0] + sum(folds[f] for f in range(1, 6)) / 2, 0.68, "training and cross-validation", ha="center",
            va="bottom", fontsize=S_, color=J.SECOND, clip_on=False)
    a3.text(n_ls - folds[6] / 2, 0.68, "test", ha="center", va="bottom", fontsize=S_, color=J.INK, clip_on=False)

    for x, y, letter in ((0.004, 0.999, "a"), (0.004, 0.405, "b"), (0.325, 0.405, "c"), (0.645, 0.405, "d")):
        fig.text(x, y, letter, fontsize=J.FS["panel"], fontweight="bold", va="top")
    frozen.emit(fig, STEM)


if __name__ == "__main__":
    main()
