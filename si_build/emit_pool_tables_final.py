#!/usr/bin/env python3
"""Tables S17 (gross-error rates) and S21 (exclusion ladder) from this run's crystal +
AlphaFold pool (``benchmarks.score_pool_af_xtal``: one common cell set over every listed arm,
NMR conformers excluded, each retrieval arm verified at its own exclusion level).

Input: ``$SI_FINAL_SCORES_DIR/pool_af_xtal.json``. The captions are the author's (the deploy
step keeps them); the ladder has three exclusion levels, no exclusion, the reported
same-protein exclusion and the stricter same-protein-plus-identity exclusion.

On the frozen master cells an arm covering less of a nucleus than the floor asks is scored
on the master cells it produced. Its entry then carries a marker and the row beneath prints
the cells it was scored on, so a rate over a smaller set is never read as one over the whole.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import si_common

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from deepshift import self_exclusion

POOL = si_common.root("SI_FINAL_SCORES_DIR") / "pool_af_xtal.json"
NUCLEI = [("HA", r"H$\alpha$"), ("H", r"H$^{\mathrm{N}}$"), ("CA", r"C$\alpha$"),
          ("CB", r"C$\beta$"), ("C", r"C$'$"), ("N", "N")]
_BLOCKED = self_exclusion.REPORTED_EXCLUSION_PHRASE
PARTIAL = r"$^{\ast}$"      # this arm covers less of the nucleus than the master floor asks
PARTIAL_NOTE = (r"{\footnotesize $^{\ast}$\,This predictor covers fewer than 90\% of the "
                r"nucleus's cells, so it is scored on the master cells it produced, counted "
                r"in the row beneath it}")


def partial(per, key):
    """The nuclei this arm was scored on fewer than every master cell of."""
    return [n for n, _ in NUCLEI if per[n]["arms"][key].get("below_master_coverage")]

S16_MAIN = [("deepshift", r"\textbf{DeepShift}"), ("ucbshift_full_sameprot", f"UCBShift, {_BLOCKED}"),
            ("ucbshift_x", "UCBShift-X (structure only)"), ("shiftx2_sameprot", f"SHIFTX2, {_BLOCKED}"),
            ("shiftx2_noshifty", "SHIFTX2, SHIFTY+ disabled"), ("sparta", "SPARTA+"), ("mlff", "MLFF"),
            ("napshift", "NapShift"), ("legolas", "LEGOLAS"), ("procs15", "PROCS15"),
            ("nmrgnn", r"NMRGNN$^{\dagger}$")]
S16_DIST = [("ucbshift_full_leaky", "UCBShift, as distributed"), ("shiftx2_leaky", "SHIFTX2, as distributed")]
S19_BLOCKS = [
    ("No transfer database", [("deepshift", r"\textbf{DeepShift}"), ("ucbshift_x", "UCBShift-X"),
                              ("shiftx2_noshifty", "SHIFTX2 (SHIFTY+ off)"), ("sparta", "SPARTA+"),
                              ("mlff", "MLFF"), ("napshift", "NapShift"), ("legolas", "LEGOLAS"),
                              ("procs15", "PROCS15"), ("nmrgnn", r"NMRGNN$^{\ddagger}$")]),
    ("No exclusion", [("ucbshift_full_leaky", "UCBShift"), ("shiftx2_leaky", "SHIFTX2")]),
    (f"{_BLOCKED[0].upper() + _BLOCKED[1:]}", [("ucbshift_full_sameprot", "UCBShift"),
                                               ("ucbshift2_sameprot", "UCBShift 2.0"),
                                               ("shiftx2_sameprot", "SHIFTX2")]),
    (r"Same-protein depositions and $\geq$90\% identity blocked",
     [("ucbshift_full_selfexcl", "UCBShift"), ("ucbshift2_selfexcl", "UCBShift 2.0"),
      ("shiftx2_selfexcl", "SHIFTX2")]),
]


def main() -> None:
    pool = json.loads(POOL.read_text())
    per = pool["per_nucleus"]

    # ---- S16 ----
    L = [r"\begin{table}[H]\centering\small", rf"\caption{{{si_common.PLACEHOLDER_CAPTION}}}", r"\label{tab:gross_error}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lrrrrrrrrrrrr}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{t}}}" for _, t in NUCLEI) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(6)),
         "Predictor & " + " & ".join(rf"$>${t1:g} & $>${t2:g}" for t1, t2 in
                                     (per[n]["thresholds_ppm"] for n, _ in NUCLEI)) + r" \\",
         r"\midrule",
         r"\quad \emph{cells scored} & " + " & ".join(
             rf"\multicolumn{{2}}{{c}}{{{si_common.thousands(per[n]['n_cells'])}}}" for n, _ in NUCLEI) + r" \\",
         r"\addlinespace"]
    any_partial = False

    def ge_row(key, name):
        nonlocal any_partial
        cells = []
        for n, _ in NUCLEI:
            g = per[n]["arms"][key]["gross_error_pct"]
            t1, t2 = per[n]["thresholds_ppm"]
            cells += [f"{g[str(t1)]:.2f}", f"{g[str(t2)]:.2f}"]
        short = partial(per, key)
        rows = [rf"\quad {name}{PARTIAL if short else ''} & " + " & ".join(cells) + r" \\"]
        if short:
            counts = [rf"\multicolumn{{2}}{{c}}{{{si_common.thousands(per[n]['arms'][key]['n_cells'])}}}"
                      for n, _ in NUCLEI]
            rows.append(r"\quad\quad \emph{cells scored} & " + " & ".join(counts) + r" \\")
            any_partial = True
        return rows
    L += [line for k, nm in S16_MAIN for line in ge_row(k, nm)]
    L += [r"\midrule", r"\multicolumn{13}{l}{\emph{As distributed}} \\"]
    L += [line for k, nm in S16_DIST for line in ge_row(k, nm)]
    L += [r"\bottomrule", r"\end{tabular}}", r"\vspace{0.5em}",
          r"{\footnotesize $^{\dagger}$rare outliers of more than a hundred ppm}",
          *([PARTIAL_NOTE] if any_partial else []), r"\end{table}"]
    si_common.emit("si_gross_error", "\n".join(L) + "\n", script=Path(__file__).name)
    print("wrote si_gross_error")

    # ---- S19 ----
    L = [r"\begin{table}[H]\centering\small", rf"\caption{{{si_common.PLACEHOLDER_CAPTION}}}", r"\label{tab:homology}", r"\label{tab:transfer}",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}", r"\toprule",
         "Predictor & " + " & ".join(t for _, t in NUCLEI) + r" \\", r"\midrule"]
    marked = False
    for title, rows in S19_BLOCKS:
        L.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for key, name in rows:
            vals = [per[n]["arms"][key]["rmse"] for n, _ in NUCLEI]
            txt = [f"{v:.3f}" for v in vals]
            if key == "deepshift":
                txt = [rf"\textbf{{{t}}}" for t in txt]
            short = partial(per, key)
            marked = marked or bool(short)
            L.append(rf"\quad {name}{PARTIAL if short else ''} & " + " & ".join(txt) + r" \\")
            if short:
                L.append(r"\quad\quad \emph{cells scored} & " + " & ".join(
                    si_common.thousands(per[n]["arms"][key]["n_cells"]) for n, _ in NUCLEI)
                    + r" \\")
        if title != S19_BLOCKS[-1][0]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular*}", r"\vspace{0.4em}",
          r"\begin{minipage}{\textwidth}\footnotesize\raggedright",
          r"$^{\ddagger}$tail-dominated RMSE\par",
          *([r"$^{\ast}$\,This predictor covers fewer than 90\% of the nucleus's cells, so it "
             r"is scored on the master cells it produced, counted in the row beneath it\par"]
            if marked else []),
          r"\end{minipage}", r"\end{table}"]
    si_common.emit("si_exclusion_ladder", "\n".join(L) + "\n", script=Path(__file__).name)
    print("wrote si_exclusion_ladder")


if __name__ == "__main__":
    main()
