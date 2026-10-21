"""Render si_cys_modes.tex from the cysteine-mode report this run scored."""
import json

import numpy as np
import si_common
from emit_competitor_tables_final import ROWS

# The report benchmarks/score_cys_modes.py writes on this run's fold 6.
REPORT = si_common.root("SI_CYS_MODES_REPORT")
OUT = si_common.OUT
# Printed names, in the order every other competitor table prints them.
LABELS = dict(ROWS)
PANELS = [("all", "Every structure source"), ("af", "AlphaFold structures only")]
# The dagger marks an arm with a residual of more than OUTLIER_PPM on this table's own
# cells, read from the cell file beside the report -- not carried over from the backbone
# tables' labels, whose cells differ.
DAGGER = r"$^{\dagger}$"
OUTLIER_PPM = 100.0


def outlier_arms(arms):
    """The arms with any |prediction - observed| above OUTLIER_PPM on these cells."""
    cells = np.load(REPORT.parent / "cys_cb_cells.npz", allow_pickle=True)
    observed = cells["true_ppm"]
    marked = set()
    for arm in arms:
        pred = cells[f"pred_{arm}"]
        ok = np.isfinite(pred) & np.isfinite(observed)
        if (np.abs(pred[ok] - observed[ok]) > OUTLIER_PPM).any():
            marked.add(arm)
    return marked


def label(arm, marked=frozenset()):
    """The printed name, with the dagger only where this table's cells call for it."""
    return LABELS[arm].replace(DAGGER, "") + (DAGGER if arm in marked else "")


def partial_cohort(d, arm):
    """The footnote of a starred arm: why its cohort is smaller, and that cohort per panel."""
    withheld = (d.get("withheld_as_training_data") or {}).get(arm)
    why = ("is not scored on the structures of proteins it was trained on"
           if withheld else "produced fewer of these cells than the other predictors")
    panels = []
    for src, _title in PANELS:
        e = d["by_source"][src]["arms"][arm]
        dis, free = e["disulfide"], e["free_thiol"]
        split = dis["mean_observed_ppm"] - free["mean_observed_ppm"]
        where = {"all": "every structure source", "af": "AlphaFold structures only"}[src]
        panels.append(f"{where}, {e['n_cells']:,} cells "
                      f"({dis['n_cells']:,} disulfide, {free['n_cells']:,} free thiol; "
                      f"observed split {split:.2f}~ppm)")
    return (f"{label(arm).split(chr(92) + 'cite')[0]} {why}, so its rows are scored on its "
            "own cells: " + "; ".join(panels) + ".")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(REPORT.read_text())
    cells = d["cell_set"]
    arms = [arm for arm in LABELS if arm in d["by_source"]["all"]["arms"]]
    marked = outlier_arms(arms)
    missing = sorted(set(LABELS) - set(arms))
    # An arm scored on fewer cells than the panel's cohort -- PLM-CS, withheld on the
    # proteins it trained on -- is starred, and its own cohort is printed beneath the table.
    partial = [arm for arm in arms if arm in (d.get("partial_coverage_arms") or [])]

    banner = []
    a = banner.append
    a("% What this is. The numeric companion to the cysteine C-beta scatter. The two")
    a("%   chemical states of a cysteine sit about 12 ppm apart on C-beta, and a predictor")
    a("%   can look accurate simply by placing each cysteine in the right state while")
    a("%   getting nothing right inside a state. The within-mode columns take that free")
    a("%   variance away.")
    a(f"% Mode rule. {d['mode_rule']['predicate'][0].upper()}{d['mode_rule']['predicate'][1:]}")
    a(f"%   means oxidised, reduced otherwise, through {d['mode_rule']['code']}. The state")
    a("%   is read from the coordinates, never from the shift.")
    a("% Amplitude is the standard deviation of the predictions divided by the standard")
    a("%   deviation of the observations inside one mode. The pooled figure is a")
    a("%   do-not-quote: pooled over both modes the gap between the modes dominates the")
    a("%   variance and every predictor scores near 1.00 whatever it does inside a mode.")
    a("% Residuals are raw: prediction minus the re-referenced observed value, with no")
    a("%   per-(label set, nucleus) mean removed for any arm.")
    a(f"% Cells. Held-out fold 6 of this run, five-member ensemble: "
      f"{d['cohort']['cells_scored']:,} cysteine C-beta cells over")
    a(f"%   {d['cohort']['n_structure_instances']} structure instances and "
      f"{d['cohort']['n_label_sets']} label sets. The composite")
    a("%   exclusion fence is on the observed value.")
    a(f"% Cell set. {cells['source']}.")
    coverage = cells.get("coverage", {})
    if missing:
        a("% Not in the table, for want of cells: " + ", ".join(
            f"{label(arm)}" + (f" (covers {coverage[arm]:.0%} of the "
                                f"{cells['n_universe']:,} cysteine C-beta cells)"
                                if arm in coverage else "")
            for arm in missing) + ".")
    check = d["checks"]["geometry_state_vs_deposited_shift_state"]
    a(f"% Consistency, not evidence: on the {check['n_cells_with_both']:,} cells whose "
      "deposited C-beta value also places")
    a(f"%   the residue, the coordinate rule and the shift agree on {check['n_agree']:,} and "
      f"disagree on {check['n_disagree']:,}, as they must:")
    a("%   the structural QC fence (deepshift.quality_control) already drops a cysteine "
      "whose geometry and C-beta shift disagree.")

    L = []
    a = L.append
    a(r"\begin{table}[H]\centering\small")
    a(r"\caption{Cysteine C$\beta$ accuracy within each disulfide state, "
      r"increased detail}")
    a(r"\label{tab:cys_modes}")
    a(r"\resizebox{\textwidth}{!}{%")
    a(r"\begin{tabular}{lrrrrrrrr}")
    a(r"\toprule")
    a(r" & All cells & \multicolumn{3}{c}{Disulfide} & \multicolumn{3}{c}{Free thiol} "
      r"& Recovered \\")
    a(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
    a(r"Predictor & RMSE & RMSE & amplitude & $r$ & RMSE & amplitude & $r$ & split \\")
    a(r" & ppm & ppm & & & ppm & & & ppm \\")
    a(r"\midrule")

    for i, (src, title) in enumerate(PANELS):
        n = d["by_source"][src]
        if i:
            a(r"\addlinespace")
        a(rf"\multicolumn{{9}}{{l}}{{\emph{{{title}}}}} \\")
        a(rf"\quad \emph{{cells}} & {n['n_cells']:,} & "
          rf"\multicolumn{{3}}{{c}}{{{n['n_disulfide']:,}}} & "
          rf"\multicolumn{{3}}{{c}}{{{n['n_free_thiol']:,}}} & \\")
        a(rf"\quad \emph{{observed}} & & \multicolumn{{3}}{{c}}"
          rf"{{{n['observed_mean_disulfide_ppm']:.2f}~ppm mean}} & "
          rf"\multicolumn{{3}}{{c}}{{{n['observed_mean_free_thiol_ppm']:.2f}~ppm mean}} & "
          rf"{n['observed_split_ppm']:.2f} \\")
        for arm in arms:
            e = n["arms"][arm]
            star = r"$^{\ast}$" if arm in partial else ""
            cs = [r"\quad " + label(arm, marked) + star, f"{e['pooled']['rmse']:.3f}"]
            for mode in ("disulfide", "free_thiol"):
                cs += [f"{e[mode]['rmse']:.3f}", f"{e[mode]['amplitude']:.3f}",
                       f"{e[mode]['r']:.3f}"]
            cs.append(f"{e['recovered_split_ppm']:.2f}")
            a(" & ".join(cs) + r" \\")
    a(r"\bottomrule")
    a(r"\end{tabular}}")
    a(r"\vspace{0.5em}")
    # The dagger some arm labels carry is defined once, ahead of the cohort footnotes.
    notes = [r"$^{\dagger}$\,rare outliers of more than a hundred ppm."] if marked else []
    notes += [r"$^{\ast}$\," + partial_cohort(d, arm) for arm in partial]
    if notes:
        a(r"{\footnotesize " + " ".join(notes) + "}")
    a(r"\end{table}")

    si_common.emit(
        "si_cys_modes", "\n".join(L), script="emit_cys_modes.py",
        banner="\n".join(banner), placeholder_caption=False)
    print("mode check:", json.dumps(check))


if __name__ == "__main__":
    main()
