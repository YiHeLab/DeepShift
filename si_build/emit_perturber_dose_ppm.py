#!/usr/bin/env python3
"""Build tab:perturber_dose: prediction error in ppm near a perturber.

The table prints the RMSE in ppm for every (category, distance band, nucleus) cell and
repeats the baseline -- residues with no nonstandard entity of any kind within 12
Angstrom -- as the last line of every category block, so the comparison sits next to the
numbers being compared.

The cohort, the distance definition, the bins and the quality fences are
analyze_perturber.py's. Residuals are raw: no per-method constant is removed
anywhere.

Bold values have Benjamini-Hochberg q < 0.05 in the test carried from
analyze_perturber.py: a Welch t-test of per-label-set means of
log1p(|error| / baseline median), in the band against the baseline, corrected over every
eligible cell.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import si_common

# The shadow-fold measurement of THIS run's five final members (perturber exclusion off
# in the atom stage, so the rows the shipped cache drops are scored). No default.
REBUILD = si_common.root("SI_PERTURBER_REBUILD")

# analyze_perturber.py sits alongside this script and is imported locally, so no path is
# inserted into an experiment tree to reach it.
import analyze_perturber as ap

OUT = si_common.OUT

BACKBONE = ["h_shift", "ha_shift", "ca_shift", "cb_shift", "c_shift", "n_shift"]
HEAD = {"h_shift": r"H$^{\mathrm{N}}$", "ha_shift": r"H$\alpha$", "ca_shift": r"C$\alpha$",
        "cb_shift": r"C$\beta$", "c_shift": r"C$'$", "n_shift": "N"}
PRETTY = {
    "heme_porphyrin": "Heme or porphyrin",
    "paramagnetic_transition_metal": "Paramagnetic transition metal",
    "zinc": "Zinc",
    "large_aromatic_ligand": "Large aromatic ligand",
    "phosphate_nucleotide": "Phosphate or nucleotide",
    "other_metal": "Other metal",
    "other": "Other ligand",
    "water": "Water",
    "organic_solvent_buffer_detergent": "Solvent, buffer or detergent",
    "sugar": "Sugar",
    "mse": "Selenomethionine",
}
ORDER = ["heme_porphyrin", "paramagnetic_transition_metal", "zinc",
         "large_aromatic_ligand", "phosphate_nucleotide", "other_metal", "other",
         "water", "organic_solvent_buffer_detergent", "sugar", "mse"]
BASELINE_STUB = "no perturber within 12"


def rmse(values: np.ndarray) -> float | None:
    """RMSE in ppm, or None for an empty cell (recorded as null, printed as ---)."""
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else None


def measure(preds: Path, dists: Path, folds=(0, 6)):
    """Re-join predictions to perturber distances and measure RMSE per cell.

    The selection is byte-for-byte the selection analyze_perturber.analyze makes for
    the 'experimental' build: NMR and crystal rows, every category distance resolved,
    the shipped quality fence already baked into effective_mask, and the published
    p100 range rule applied on top.
    """
    parts = [ap.load_fold(f, preds, dists) for f in folds]
    cols = parts[0]["cols"]
    if any(p["cols"] != cols for p in parts):
        raise SystemExit("shift columns differ between folds")
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "cols"}

    keep_sources = ("nmr", "xtal")
    keep_codes = {ap.SOURCE_NAMES.index(s) for s in keep_sources}
    keep = np.isin(d["source"], list(keep_codes)) & ~np.isnan(d["dist"]).any(1)
    sel = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
           for k, v in d.items()}

    inrange = ap.p100_mask(sel["comp_id"], sel["target"], cols)
    mask = sel["mask"] & inrange

    err = sel["pred"] - sel["target"]                       # raw residual, ppm
    within12 = np.min(sel["dist"], axis=1) < 12.0
    baseline_rows = ~within12

    jj = {n: cols.index(n) for n in BACKBONE}
    base = {}
    for n, j in jj.items():
        q = baseline_rows & mask[:, j]
        base[n] = {"rmse_ppm": rmse(err[q, j]), "mae_ppm": float(np.abs(err[q, j]).mean()),
                   "n_cells": int(q.sum())}
    baseline_row_count = int(baseline_rows.sum())

    cells = {}
    for cj, cat in enumerate(ap.CATEGORIES):
        distance = sel["dist"][:, cj]
        for lo, hi, label in ap.BINS:
            in_bin = (distance >= lo) & (distance < hi)
            entry = {"n_rows": int(in_bin.sum()), "nuclei": {}}
            for n, j in jj.items():
                q = in_bin & mask[:, j]
                entry["nuclei"][n] = {"rmse_ppm": rmse(err[q, j]),
                                      "mae_ppm": (float(np.abs(err[q, j]).mean())
                                                  if q.sum() else None),
                                      "n_cells": int(q.sum())}
            cells[f"{cat}|{label}"] = entry

    return cells, base, baseline_row_count


def build_table(cells, base, baseline_row_count, significance) -> str:
    body = [r"\begin{tabular}{ll r r r r r r r}", r"\toprule",
            (r"Perturber & Distance (\AA{}) & Residues & "
             + " & ".join(HEAD[n] for n in BACKBONE) + r" \\"),
            r"\midrule"]
    for i, cat in enumerate(ORDER):
        if i:
            body.append(r"\addlinespace")
        first = True
        for _lo, _hi, label in ap.BINS:
            e = cells.get(f"{cat}|{label}")
            if e is None:
                continue
            vals = []
            for n in BACKBONE:
                v = e["nuclei"][n]["rmse_ppm"]
                value = "---" if v is None else f"{v:.2f}"
                q = significance.get(f"{cat}|{label}|{n}", {}).get("q_bh")
                if value != "---" and q is not None and q < 0.05:
                    value = rf"\textbf{{{value}}}"
                vals.append(value)
            body.append(" & ".join([PRETTY[cat] if first else "", label,
                                    f"{e['n_rows']:,}", *vals]) + r" \\")
            first = False
        body.append(" & ".join(["", BASELINE_STUB, f"{baseline_row_count:,}"]
                               + [f"{base[n]['rmse_ppm']:.2f}" for n in BACKBONE])
                    + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    # The float carries a caption, so the facts it must assert go in the %-comment
    # beneath it, where every other table in this directory keeps them.
    facts = (
        "% FACTS the caption needs:\n"
        "%   Each entry is the root-mean-square error in ppm of the released five-model\n"
        "%   ensemble on the held-out folds 0 and 6, for the residues whose nearest atom\n"
        "%   of that perturber falls in that distance band.\n"
        f"%   The last line of every block is the same reference set of {baseline_row_count:,}\n"
        "%   residues, those with no nonstandard entity of any kind within 12 Angstrom,\n"
        "%   and it is repeated so the comparison sits beside the numbers.\n"
        "%   Residues are counted once per band; the number of scored shifts behind a\n"
        "%   column is smaller. Residuals are raw.\n")
    # Eleven five-row blocks fill a page: at the default row height the float ran 58 pt
    # past the text block and the page number printed inside it (SI page 12).
    return ("\\begin{table}[H]\\centering\\small\\renewcommand{\\arraystretch}{0.88}\n"
            "\\caption{Prediction error in ppm near a perturber}\n"
            "\\label{tab:perturber_dose}\n"
            "\\resizebox{0.98\\textwidth}{!}{%\n" + "\n".join(body) + "}\n"
            "\\end{table}\n\n" + facts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preds = REBUILD / "work/preds"
    dists = REBUILD / "work/distances"
    cells, base, baseline_row_count = measure(preds, dists)

    # The significance testing is carried from the stratified-error table that
    # `analyze_perturber.py --out <SI_PERTURBER_REBUILD>/out` writes from the same dumps;
    # its q_bh field sets the bolding.
    ratio = pd.read_csv(REBUILD / "out/stratified_errors_experimental.csv")
    ratio = ratio[ratio.nucleus.isin(BACKBONE)]
    carried = {}
    for r in ratio.itertuples():
        carried[f"{r.category}|{r.distance_bin}|{r.nucleus}"] = {
            "q_bh": float(r.cluster_q_bh) if r.cluster_q_bh == r.cluster_q_bh else None}

    tex = build_table(cells, base, baseline_row_count, carried)
    banner = "\n".join([
        "% The printed quantity is RMSE in ppm, and each category block ends with the",
        "% baseline the ppm numbers compare to.",
        "% Lineage: the released five-member ensemble (dim 384, 6 layers, 64 nuclei) on",
        "% folds 0 and 6, which are held out from all five members; each member",
        "% denormalized with its own stats before the members are averaged in ppm.",
    ])
    si_common.emit("si_perturber_dose", tex, banner=banner,
                   script="emit_perturber_dose_ppm.py")
    for n in BACKBONE:
        print(f"  baseline {n}: RMSE {base[n]['rmse_ppm']:.3f} ppm "
              f"(MAE {base[n]['mae_ppm']:.3f}, {base[n]['n_cells']:,} shifts)")


if __name__ == "__main__":
    main()
