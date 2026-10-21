#!/usr/bin/env python3
"""Tables S12--S14 (backbone RMSE, MAE, MedAE for every predictor) from this run's own
tiered scoring report on the frozen master cells.

Input: ``$SI_FINAL_SCORES_DIR/score_tier2_master.json``, written by

    python -m benchmarks.score_by_source_tier --tier 2 --metrics rmse,mae,medae \\
        --master-cells <run>/benchmark/master_cells/master_cells.npz --tools <every arm>

Every arm is scored on the same cells: Master(nucleus), the cells of DeepShift's scorable
set that every arm covering at least 90 % of it produced. An arm below that floor on a
nucleus does not shrink the set; it is scored on the master cells it did produce, and its
row carries a marker and the cell count it was scored on, printed beneath it, so a number
on fewer cells is never read as a number on the whole set.

Every retrieval arm carries the reported exclusion (self_exclusion.REPORTED_EXCLUSION), the
residual is raw, intervals are label-set bootstrap. The captions are the author's: the
deploy step keeps the manuscript's caption over the placeholder written here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import si_common

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from deepshift import self_exclusion

SCORES = si_common.root("SI_FINAL_SCORES_DIR")
REPORT = SCORES / "score_tier2_master.json"
NUCLEI = [("HA", r"H$\alpha$"), ("H", r"H$^{\mathrm{N}}$"), ("CA", r"C$\alpha$"),
          ("CB", r"C$\beta$"), ("C", r"C$'$"), ("N", "N")]
SOURCES = [("xtal", "X-ray crystal"), ("af", "AlphaFold"), ("nmr", "NMR")]
_BLOCKED = self_exclusion.REPORTED_EXCLUSION_PHRASE
PARTIAL = r"$^{\ast}$"       # this arm is outside a nucleus's master set (below the floor,
                              # or kept out by design: benchmarks.config.MASTER_MARKED_ARMS)
# The dagger marks every arm with a residual over 100 ppm on the master cells it produced.
ROWS = [
    ("deepshift", r"\textbf{DeepShift}"),
    ("ucbshift_full_leaky", "UCBShift, as distributed"),
    ("ucbshift_full_sameprot", f"UCBShift, {_BLOCKED}"),
    ("ucbshift_x", "UCBShift-X (structure only)"),
    ("ucbshift2_sameprot", f"UCBShift 2.0, {_BLOCKED}"),
    ("shiftx2_leaky", "SHIFTX2, as distributed"),
    ("shiftx2_sameprot", f"SHIFTX2, {_BLOCKED}"),
    ("shiftx2_noshifty", "SHIFTX2, SHIFTY+ disabled"),
    ("sparta", "SPARTA+"),
    ("napshift", "NapShift"),
    ("legolas", "LEGOLAS"),
    ("procs15", "PROCS15"),
    ("nmrgnn", r"NMRGNN$^{\dagger}$"),
    ("camshift", r"CAMSHIFT$^{\dagger}$"),
    ("mlff", r"MLFF\cite{MLFF}"),
    ("plmcs", r"PLM-CS\cite{PLM-CS}$^{\dagger}$"),
]
TABLES = [("rmse", "si_tables_rmse", "tab:competitor_rmse", "RMSE"),
          ("mae", "si_tables_mae", "tab:competitor_mae", "MAE"),
          ("medae", "si_tables_medae", "tab:competitor_medae_refold", "MedAE")]


def cell(entry, metric):
    v, ci = entry[metric], entry[f"{metric}_ci95"]
    return rf"{v:.3f} \tiny[{ci[0]:.3f}, {ci[1]:.3f}]"


def counts_row(label, values):
    return rf"\quad\quad \emph{{{label}}} & " + " & ".join(values) + r" \\"


def main() -> None:
    report = json.loads(REPORT.read_text())
    partial_seen = False
    for metric, stem, label, _word in TABLES:
        lines = [r"\begin{table}[H]\centering\small",
                 rf"\caption{{{si_common.PLACEHOLDER_CAPTION}}}", rf"\label{{{label}}}",
                 r"\resizebox{0.96\textwidth}{!}{%",
                 r"\begin{tabular}{lcccccc}", r"\toprule",
                 "Method & " + " & ".join(t for _, t in NUCLEI) + r" \\", r"\midrule"]
        for src, pretty in SOURCES:
            panel = report["by_source"][src]
            n_cells = sum(panel["nuclei"][n]["n_cells"] for n, _ in NUCLEI)
            lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{pretty}}}, $n={panel['n_structures']}$ "
                         rf"structures, {si_common.thousands(n_cells)} master cells}} \\")
            lines.append(counts_row("master cells",
                                    [si_common.thousands(panel["nuclei"][n]["n_cells"])
                                     for n, _ in NUCLEI]))
            for key, name in ROWS:
                if key not in panel["nuclei"][NUCLEI[0][0]]:
                    continue                 # an arm this run did not score has no row
                arms = [panel["nuclei"][n][key] for n, _ in NUCLEI]
                partial = [a for a in arms if a["below_master_coverage"]]
                lines.append(rf"\quad {name}{PARTIAL if partial else ''} & "
                             + " & ".join(cell(a, metric) for a in arms) + r" \\")
                if partial:
                    lines.append(counts_row("cells scored",
                                            [si_common.thousands(a["n_cells"]) for a in arms]))
                    partial_seen = True
            if src != SOURCES[-1][0]:
                lines.append(r"\midrule")
        marker_note = (r"; $^{\ast}$\,this predictor is outside a nucleus's master cell set, "
                       r"so it is scored on the master cells it produced, counted in "
                       r"the row beneath it" if partial_seen else "")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\vspace{0.4em}",
                  r"{\footnotesize $^{\dagger}$rare outliers of more than a hundred ppm"
                  + marker_note + "}", r"\end{table}"]
        si_common.emit(stem, "\n".join(lines) + "\n", script=Path(__file__).name)
        print(f"wrote {stem}")


if __name__ == "__main__":
    main()
