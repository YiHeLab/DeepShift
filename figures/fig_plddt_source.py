#!/usr/bin/env python3
"""Does AlphaFold's own confidence explain the structure-source result?

``fig_source_matched_absolute`` says the model predicts a deposited shift better from an
AlphaFold model than from the crystal or NMR structure of the same protein, on
the same cells. The obvious objection is that AlphaFold wins on the easy, well-ordered
residues it is confident about, and that the comparison would reverse where its confidence
is low. This script decomposes that exact cohort by AlphaFold's own per-residue pLDDT and
answers the objection with the data, not with an argument.

``fig_plddt_source_advantage`` draws, per pLDDT band, the paired RMSE difference of each
experimental arm from the AlphaFold arm as a per cent of it: the source claim, conditioned
on confidence. The band is not "how wrong the AlphaFold coordinates are": the two
experimental arms carry none of AlphaFold's error and are scored on the identical cells,
so whatever they do with the band is a property of the residues, not of AlphaFold.

The confidence axis is per residue, not per chain. pLDDT is the AlphaFold model's
temperature-factor column, and the build's own ``structure_data_af.csv`` carries the
(label residue -> structure residue) mapping the cache was built from, so the join is the
build's, not a re-derived alignment. It is checked, never trusted: the residue name
recorded on the label side must equal the residue name in the coordinate file, and the run
aborts if they disagree anywhere.

Usage:
    python figures/fig_plddt_source.py
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import fig_structure_source as base
import frozen
import numpy as np
import style

from deepshift import cohorts, evaluate, structure

ARMS = ("af", "nmr", "xtal")

# AlphaFold's published confidence bands, with its "very high" half split: 95% of this
# cohort's matched residues sit above 70, so the standard four bands would put almost
# everything in one bin.
BANDS = ((0.0, 50.0, "<50"), (50.0, 70.0, "50-70"), (70.0, 80.0, "70-80"),
         (80.0, 90.0, "80-90"), (90.0, 95.0, "90-95"), (95.0, 100.01, "$\\geq$95"))


# --------------------------------------------------------------------------------
# The confidence column
# --------------------------------------------------------------------------------

def af_residue_map(structure_data: Path, wanted: set) -> dict:
    """label_residue_id -> (accession, chain, structure residue number, residue name).

    Straight from the build's own structure table, restricted to rows whose label residue
    actually mapped onto a modelled position (``mismatch_type == 'match'``). A row the
    build itself called a mismatch or a gap carries no trustworthy residue number, and
    guessing one is how a confidence axis silently becomes a shuffle.
    """
    out = {}
    with open(structure_data, newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["label_residue_id"]
            if key not in wanted or row["mismatch_type"] != "match":
                continue
            if not row["struct_res_id"] or not row["accession"]:
                continue
            out[key] = (row["accession"], row["chain"] or "A",
                        int(float(row["struct_res_id"])), row["comp_id"])
    return out


def plddt_per_label_residue(label_residue, keep, build: Path):
    """pLDDT for each label residue in ``keep``; NaN where the model is not on disk."""
    wanted = set(np.asarray(label_residue)[keep].tolist())
    mapping = af_residue_map(build / "structure_data_af.csv", wanted)
    store = build / "alphafold"
    cache, resolved, checked, disagreed = {}, {}, 0, 0
    for key, (accession, chain, resid, comp_id) in mapping.items():
        # Keyed on (accession, chain), not accession: AlphaFold writes one chain today, and
        # a cache that assumed so would return another chain's numbering the day it does not.
        if (accession, chain) not in cache:
            hits = sorted(store.glob(f"AF-{accession}-F1-model_v*.pdb"))
            cache[(accession, chain)] = (
                structure.ca_temperature_factors(hits[-1], chain) if hits else {})
        entry = cache[(accession, chain)].get(resid)
        if entry is None:
            continue
        checked += 1
        disagreed += int(entry[1] != comp_id)
        resolved[key] = entry[0]
    # The join is only as good as its keys. A silent mis-key would draw a perfectly
    # plausible confidence axis out of a shuffle, so it fails loudly instead.
    if checked and disagreed:
        raise SystemExit(f"the label side and the coordinate file disagree on the residue "
                         f"name for {disagreed:,}/{checked:,} joined residues")
    values = np.full(len(label_residue), np.nan)
    for i, key in enumerate(label_residue):
        if keep[i]:
            values[i] = resolved.get(key, np.nan)
    return values, {"n_label_residues_wanted": len(wanted),
                    "n_mapped_by_build": len(mapping), "n_resolved": len(resolved),
                    "n_alphafold_models_read": len(cache),
                    "residue_name_check": f"{checked:,} joined residues, 0 name disagreements"}


# --------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------

def by_band(arms, plddt, nuclei, *, boots, seed):
    """Per band: each arm's matched RMSE, and the paired difference from the AlphaFold arm."""
    bands = []
    for low, high, label in BANDS:
        inside = {name: (plddt[name] >= low) & (plddt[name] < high) for name in arms}
        entry = {"label": label, "low": low, "high": high, "arms": {}, "paired": {}}
        scored = {}
        for name, arm in arms.items():
            keep = replace(arm, effective_mask=arm.effective_mask & inside[name][:, None])
            scored[name] = keep
            entry["arms"][name] = evaluate.score(keep, columns=nuclei, boots=boots, seed=seed,
                                                 group="label_set")
        for name in ("nmr", "xtal"):
            paired = evaluate.compare(scored[name], scored["af"], columns=nuclei,
                                      boots=boots, seed=seed, group="label_set")
            entry["paired"][name] = {
                col: dict(d, pct=100.0 * d["delta_rmse"] / d["rmse_candidate"],
                          pct_lo=100.0 * d["ci"][0] / d["rmse_candidate"],
                          pct_hi=100.0 * d["ci"][1] / d["rmse_candidate"])
                for col, d in paired.items()}
        bands.append(entry)
    return bands


# --------------------------------------------------------------------------------
# Draw
# --------------------------------------------------------------------------------

def _grid(nuclei, height):
    import matplotlib.pyplot as plt

    style.use_house_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(style.WIDTH["wide"], height))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.845, bottom=0.145,
                          wspace=0.30, hspace=0.62)
    return fig, [fig.add_subplot(gs[i // 3, i % 3]) for i in range(len(nuclei))]


def visible_bands(bands, nuclei):
    """Band positions with something to draw; both figures use the same set of columns."""
    return [j for j, b in enumerate(bands)
            if any(col in b["arms"][s] for s in ARMS for col in nuclei)]


def _band_axis(ax, bands, drawn):
    ax.set_xticks(range(len(drawn)))
    ax.set_xticklabels([bands[j]["label"] for j in drawn], fontsize=5.9)
    ax.set_xlim(-0.55, len(drawn) - 0.45)
    style.finish(ax, xgrid=False, spines=("bottom", "left"))
    ax.tick_params(labelsize=6.2)


def draw_advantage(bands, nuclei):
    """The paired difference from the AlphaFold arm, per band, as a per cent of that arm."""
    from matplotlib.lines import Line2D

    fig, axes = _grid(nuclei, 3.85)
    drawn = visible_bands(bands, nuclei)
    for ax, col in zip(axes, nuclei):
        for k, src in enumerate(("nmr", "xtal")):
            c = base.SOURCE_COLOR[src]
            xs, ys, los, his = [], [], [], []
            for pos, j in enumerate(drawn):
                entry = bands[j]["paired"][src].get(col)
                if entry is None:
                    continue
                xs.append(pos + (k - 0.5) * 0.26)
                ys.append(entry["pct"])
                los.append(entry["pct_lo"])
                his.append(entry["pct_hi"])
            ax.vlines(xs, los, his, color=c, linewidth=style.LINE, zorder=3)
            ax.plot(xs, ys, marker="o", markersize=style.MARKER - 0.9, color=c,
                    markeredgecolor=style.INK["surface"], markeredgewidth=style.RING,
                    linestyle="none", zorder=4)
        ax.axhline(0, color=style.INK["secondary"], linewidth=0.8, zorder=1)
        ax.set_title(style.nucleus_label(col), loc="left", fontsize=8.2,
                     color=style.INK["primary"], pad=4)
        _band_axis(ax, bands, drawn)
    for ax in axes[::3]:
        ax.set_ylabel("RMSE vs the AlphaFold arm (%)", fontsize=7.0)
    for ax in axes[3:]:
        ax.set_xlabel("AlphaFold pLDDT of the residue", fontsize=7.0)
    handles = [Line2D([], [], marker="o", linestyle="none", color=base.SOURCE_COLOR[s],
                      markersize=style.MARKER - 0.9, markeredgecolor=style.INK["surface"],
                      label=base.SOURCE_NAME[s]) for s in ("nmr", "xtal")]
    style.legend(fig, handles=handles, labels=[h.get_label() for h in handles],
                 loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    return fig


# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, default=base.PREDS)
    ap.add_argument("--fence", type=Path, required=True,
                    help="authoritative p100/p100_structural QC flags JSON")
    ap.add_argument("--build", type=Path, default=frozen.BUILD_DIR)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=evaluate.DEFAULT_SEED)
    args = ap.parse_args()

    nuclei = list(style.BACKBONE)
    paths = [args.preds / f"refold_fold{f}.preds.npz" for f in args.folds]
    structure_data = args.build / "structure_data_af.csv"
    missing = [p for p in [*paths, args.fence, structure_data] if not p.exists()]
    if missing:
        raise SystemExit(f"missing frozen artifacts: {missing}")

    # Exactly the cohort of the matched source figures: same read, same fence, same match.
    preds, label_residue = base.read(paths, nuclei)
    preds, _fenced = base.apply_fence(preds, json.loads(args.fence.read_text()), label_residue)
    cells, lr_id = cohorts.matched_cells(preds, label_residue)
    arms = base.paired_arms(preds, cells, lr_id)

    # pLDDT is a property of the RESIDUE, so all three arms are binned by the same value:
    # the AlphaFold model's confidence in that residue. The AlphaFold arm's rows carry the
    # mapping, and the arms are keyed to one label residue in one order.
    af_rows = np.flatnonzero(cells.any(axis=1) & (preds.source == 0))
    keep = np.zeros(preds.n_rows, dtype=bool)
    keep[af_rows] = True
    values, _join = plddt_per_label_residue(label_residue, keep, args.build)
    per_label_residue = {}
    for row in af_rows:
        if np.isfinite(values[row]):
            per_label_residue[int(lr_id[row])] = float(values[row])
    plddt = {name: np.asarray([per_label_residue.get(int(k), np.nan)
                               for k in arm.row_key.astype(int)])
             for name, arm in arms.items()}
    for name in ARMS[1:]:
        if not np.array_equal(np.isnan(plddt["af"]), np.isnan(plddt[name])):
            raise SystemExit("the arms carry different pLDDT coverage; they are not paired")
    covered = int(np.isfinite(plddt["af"]).sum())
    print(f"pLDDT resolved for {covered:,}/{len(plddt['af']):,} matched label residues")
    if covered < len(plddt["af"]):
        # pLDDT is read from each AlphaFold model file; without them (a reproduction
        # bundle carries no coordinates) the bands would be drawn on whatever resolved.
        raise SystemExit(f"REFUSING: pLDDT resolved for only {covered:,} of "
                         f"{len(plddt['af']):,} matched label residues; the AlphaFold model "
                         f"files under {args.build} are missing")

    bands = by_band(arms, plddt, nuclei, boots=args.boots, seed=args.seed)

    frozen.emit(draw_advantage(bands, nuclei), "fig_plddt_source_advantage")


if __name__ == "__main__":
    main()
