#!/usr/bin/env python3
"""Agreement between our rereferenced labels and RefDB's.

The body claims a per-nucleus agreement rate with RefDB. This is that table. It reads the
census ``si_build/build_refdb_census.py`` measures on this run: our offsets against
RefDB's own published ones, over the label sets both databases hold.

Two cohorts, and the split is the whole point: RefDB fitted its offset on a structure
it chose, we fit ours on a structure we chose, and the selectors do not always agree.
Where the structure is the same the two procedures agree closely; where it differs the
disagreement is mostly the structure, not the arithmetic.

Weighting is per label (a label set contributes its own number of shifts), because the
sentence in the body is about labels.
"""
from __future__ import annotations

import json

import si_common as C

# The census this run measured; SI_REFDB_CENSUS names it, as every root here is named.
CENSUS = C.root("SI_REFDB_CENSUS")

NUCLEI = ["CA", "CB", "CO", "N"]
HEAD = {"CA": r"C$\alpha$", "CB": r"C$\beta$", "CO": r"C$'$", "N": "N"}
COHORTS = [("all", "Every comparable label set"),
           ("same_structure", "RefDB and we referenced the same structure")]


def rows(census, cohort):
    out = []
    for nuc in NUCLEI:
        v = census["per_label"][cohort][nuc]
        out.append(" & ".join([
            HEAD[nuc],
            C.thousands(v["n_cells"]),
            C.thousands(v["n"]),
            C.num(v["median"], 2),
            f"{100 * v['within_0p05']:.1f}",
            f"{100 * v['within_0p10']:.1f}",
            C.num(v["p95"], 2),
            C.num(v["max"], 2),
        ]) + r" \\")
    return out


def main():
    census = json.loads(CENSUS.read_text())

    body = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
            r" & Label sets & Labels & Median & \multicolumn{2}{c}{\% of labels within}"
            r" & p95 & Max \\",
            r"\cmidrule(lr){5-6}",
            r" & & & $|\Delta|$ & 0.05~ppm & 0.10~ppm & $|\Delta|$ & $|\Delta|$ \\"]
    for key, title in COHORTS:
        body += [r"\midrule", r"\multicolumn{8}{l}{\emph{" + title + r"}} \\"]
        body += rows(census, key)
    body += [r"\bottomrule", r"\end{tabular}"]

    tex = ("\\begin{table}[H]\\centering\\small\n"
           "\\caption{Agreement with RefDB.}\n"
           "\\label{tab:refdb_agreement}\n"
           + "\n".join(body) + "\n\\end{table}\n")

    C.emit("si_refdb_agreement", tex, script="si_refdb_agreement.py")

if __name__ == "__main__":
    main()
