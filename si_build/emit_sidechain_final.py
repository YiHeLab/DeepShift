#!/usr/bin/env python3
"""Table S15, the side-chain three-way comparison (DeepShift, UCBShift 2.0, SHIFTX2), from
this run's per-nucleus scoring report (``benchmarks.score_by_nucleus --tier 2``).

Rows are the nuclei both comparators emit, on the cell set the report scored -- the frozen
master cells for that nucleus when the scorer was given them -- most cells first; backbone
nuclei form a first block. The spread column is the standard deviation of the measured
shifts on those same cells when the report carries it (``observed_sd``), otherwise a dash.
Bold marks the lowest RMSE in the row. The caption is the author's (kept by the deploy step).

An arm covering less of a nucleus than the master floor asks is scored on the master cells
it produced. Its RMSE then carries a marker and the row beneath prints how many cells each
arm was scored on, and the advantage column is the paired difference the report carries --
DeepShift on that comparator's own cells -- so the percentage is never a ratio of two
numbers measured on different cells.

Input: ``$SI_FINAL_SCORES_DIR/score_by_nucleus_tier2.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import si_common

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from deepshift import self_exclusion

REPORT = si_common.root("SI_FINAL_SCORES_DIR") / "score_by_nucleus_tier2.json"
UCB2 = self_exclusion.reported_arm("ucbshift2")
SX2 = self_exclusion.reported_arm("shiftx2")
ARMS = [("deepshift", r"\textbf{DeepShift}"), (UCB2, "UCBShift 2.0"), (SX2, "SHIFTX2")]
PARTIAL = r"$^{\ast}$"      # this arm covers less of the nucleus than the master floor asks


def row(nuc, e):
    """One nucleus, and the row of per-arm cell counts a partial arm makes necessary."""
    a = e["arms"]
    rmse = {k: a[k]["rmse"] for k, _ in ARMS}
    best = min(rmse, key=rmse.get)
    cells = [si_common.tex_nucleus(nuc), si_common.thousands(e["n_cells"]),
             f"{e['observed_sd']:.3f}" if e.get("observed_sd") is not None else "---"]
    for k, _ in ARMS:
        r = f"{rmse[k]:.3f}"
        mark = PARTIAL if a[k].get("below_master_coverage") else ""
        cells += [(rf"\textbf{{{r}}}" if k == best else r) + mark, f"{a[k]['mae']:.3f}"]
    for k in (UCB2, SX2):
        # the paired difference: DeepShift on that comparator's own cells
        ours = e["vs_deepshift"][k]["rmse"]["deepshift"]
        cells.append(f"{100.0 * (rmse[k] - ours) / rmse[k]:+.1f}")
    lines = [" & ".join(cells) + r" \\"]
    if any(a[k].get("below_master_coverage") for k, _ in ARMS):
        counts = [r"\quad \emph{cells scored}", "", ""]
        for k, _ in ARMS:
            counts += [si_common.thousands(a[k]["n_cells"]), ""]
        counts += ["", ""]
        lines.append(" & ".join(counts) + r" \\")
    return lines


def main() -> None:
    rep = json.loads(REPORT.read_text())
    both = {n: e for n, e in rep["by_nucleus"].items()
            if UCB2 in e["emitting_competitors"] and SX2 in e["emitting_competitors"]}
    backbone = sorted((n for n, e in both.items() if e["backbone"]), key=lambda n: -both[n]["n_cells"])
    side = sorted((n for n, e in both.items() if not e["backbone"]), key=lambda n: -both[n]["n_cells"])
    L = [r"\begin{table}[H]\centering\small", rf"\caption{{{si_common.PLACEHOLDER_CAPTION}}}", r"\label{tab:sidechain}",
         r"\resizebox{0.92\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r"& & & \multicolumn{2}{c}{\textbf{DeepShift}} & \multicolumn{2}{c}{UCBShift 2.0} & "
         r"\multicolumn{2}{c}{SHIFTX2} & \multicolumn{2}{c}{DeepShift RMSE advantage} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
         r"Nucleus & Cells & Spread of measured shifts & RMSE & MAE & RMSE & MAE & RMSE & MAE & "
         r"vs UCBShift 2.0 & vs SHIFTX2 \\",
         r" & & \multicolumn{7}{c}{ppm} & \multicolumn{2}{c}{\%} \\", r"\midrule"]
    if backbone:
        L.append(r"\multicolumn{11}{l}{\emph{Backbone}} \\")
        L += [line for n in backbone for line in row(n, both[n])]
        L.append(r"\midrule")
    L.append(r"\multicolumn{11}{l}{\emph{Side chain}} \\")
    L += [line for n in side for line in row(n, both[n])]
    L += [r"\bottomrule", r"\end{tabular}}"]
    if any(e["arms"][k].get("below_master_coverage") for e in both.values() for k, _ in ARMS):
        L += [r"\vspace{0.4em}",
              r"{\footnotesize $^{\ast}$\,This predictor covers fewer than 90\% of the "
              r"nucleus's cells, so its RMSE and MAE are on the master cells it produced, "
              r"counted beneath the row. DeepShift's RMSE and MAE in that row are on all of "
              r"its cells; the RMSE advantage compares DeepShift with the predictor on the "
              r"predictor's cells.}"]
    L.append(r"\end{table}")
    si_common.emit("si_sidechain", "\n".join(L) + "\n", script=Path(__file__).name)
    print(f"wrote si_sidechain: {len(backbone)} backbone + {len(side)} side-chain rows")


if __name__ == "__main__":
    main()
