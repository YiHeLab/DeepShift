"""Render si_error_covariates.tex from the covariate report this run measured."""
import json

import si_common

# The report benchmarks/score_error_covariates.py writes on this run's fold 6.
REPORT = si_common.root("SI_ERROR_COVARIATES_REPORT")
# Where the table lands. SI_OUT redirects it so a regeneration can be checked against
# what is deployed without overwriting it first.
OUT = si_common.OUT
NUC = [("HA", r"H$\alpha$"), ("H", r"H$^{\mathrm{N}}$"), ("CA", r"C$\alpha$"),
       ("CB", r"C$\beta$"), ("C", r"C$'$"), ("N", "N")]
ORDER = ["plddt", "resolution", "ring", "rsa", "source", "n_sources"]
DISPLAY = {"solution NMR": "NMR",
           "Structure sources available for the deposition":
               "Structure sources available for the label set"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(REPORT.read_text())
    inv = d["checks"]["plddt_inversion"]
    join = d["join_coverage"]
    bands = list(d["blocks"]["plddt"]["strata"])
    best_is_low = [n for n, _p in NUC if inv["per_nucleus"][n]["best_band_by_rmse"] == bands[0]]
    other = [n for n, _p in NUC if n not in best_is_low]

    banner = []
    a = banner.append
    a("% What this is. DeepShift's held-out error cut six ways by the properties usually")
    a("%   taken to measure how good a structure is.")
    a("% The pLDDT result. AlphaFold confidence runs the wrong way. The")
    a("%   residues AlphaFold is least sure of are the ones predicted best: the band below")
    a(f"%   50 has the lowest RMSE on {len(best_is_low)} of the six backbone nuclei"
      + (f" (the exception is {', '.join(other)}," if other else ",")
      + " and the")
    a("%   same holds on mean absolute error. The reading is about which residues carry a")
    a("%   measurable shift, not about AlphaFold: a low-confidence residue is a flexible")
    a("%   one, and a flexible residue's observed shift sits near its random-coil value,")
    a("%   which is the easy prediction. pLDDT is therefore not a structure-quality gate")
    a("%   on this task, and this block must not be pooled with the paired between-source")
    a("%   comparison, which asks a different question on the same bands.")
    a("% Residuals are raw: prediction minus the re-referenced observed value, with no")
    a("%   per-method constant removed.")
    a(f"% Cells. Every backbone cell this run's fold-6 five-member ensemble scored: "
      f"{d['n_cells']:,}")
    a(f"%   residue-nucleus cells over {d['n_structure_instances']} structure instances. "
      "This is not a")
    a("%   predictor comparison, so the cohort is DeepShift's own rather than a common")
    a("%   set. The composite exclusion fence is on the observed value, and the "
      f"{d.get('anticorrelated_cells_dropped', 0):,} cells")
    a("%   of the anticorrelated channels are out, as in every float.")
    a(f"% Verified. Every cell carries a geometry covariate ({join['cells_without']} do not); "
      f"all {join['alphafold_cells_total']:,}")
    a(f"%   AlphaFold cells carry a pLDDT and all {join['crystal_cells_total']:,} crystal "
      "cells a resolution, and the")
    a("%   ring, accessibility, source and source-count strata each sum to the cohort")
    a("%   exactly; a residue whose structure has no aromatic ring of another residue is")
    a("%   in the farthest ring band.")
    a("% Blocks are not comparable to each other: the confidence block is AlphaFold")
    a("%   structures only and the resolution block is X-ray structures only.")

    L = []
    a = L.append
    a(r"\begin{table}[H]\centering\small")
    a(r"\caption{Held-out error stratified by AlphaFold per-residue confidence, "
      r"crystallographic resolution, distance to an aromatic ring, solvent "
      r"accessibility, structure source, and number of structure sources present "
      r"for a given label set.}")
    a(r"\label{tab:error_covariates}")
    # Layout only: the target width is 0.80\textwidth, not \textwidth, because
    # \resizebox scales height with width and this table is tall; at \textwidth the
    # scaled height overflows the text block. No number or row changes with it.
    a(r"\resizebox{0.80\textwidth}{!}{%")
    a(r"\begin{tabular}{lrrrrrrr}")
    a(r"\toprule")
    a("Stratum & Cells & " + " & ".join(p for _n, p in NUC) + r" \\")
    a(r" & & \multicolumn{6}{c}{RMSE, ppm} \\")
    a(r"\midrule")

    for i, key in enumerate(ORDER):
        node = d["blocks"][key]
        if i:
            a(r"\addlinespace")
        # The scorer counts sources per label set (sources_of_label); its title says
        # deposition, so the table prints what was counted.
        a(rf"\multicolumn{{8}}{{l}}{{\emph{{{DISPLAY.get(node['title'], node['title'])}}}}} \\")
        for name, ent in node["strata"].items():
            # The report names the nmr lane "solution NMR"; the lane holds every NMR
            # method (crystal_census.is_nmr), so the table prints "NMR".
            cs = [r"\quad " + DISPLAY.get(name, name), f"{ent['n_cells']:,}"]
            for n, _p in NUC:
                v = ent["nuclei"][n]["rmse"]
                cs.append("---" if v is None else f"{v:.3f}")
            a(" & ".join(cs) + r" \\")
    a(r"\bottomrule")
    a(r"\end{tabular}}")
    a(r"\vspace{0.5em}")
    a(r"\end{table}")

    si_common.emit(
        "si_error_covariates", "\n".join(L), script="emit_error_covariates.py",
        banner="\n".join(banner), placeholder_caption=False)


if __name__ == "__main__":
    main()
