#!/usr/bin/env python3
"""Build the reproduce-published-results table: each predictor on its own published
test set, before it is run on ours. Both UCBShift blocks carry all six backbone nuclei.

The published column. Every anchor in it is read from the predictor's own
paper or supporting information, never from this project's rescoring of that predictor's
released files: comparing a number with a rounded copy of itself passes by construction.
On the 200-protein uncurated test set the UCBShift paper publishes

  combined output          RMSE, Table 2, and MAE, supporting-information Table S4
  structure module alone   RMSE, Table 3

so the combined block is reported as a mean absolute error and the structure module as a
root mean square error, each against the statistic its own paper prints.

The "this work" column is our invocation of their released code, run over their released
test structures with hydrogens rebuilt. A cache of predictions that ships alongside a source
checkout is not the authors' release and cannot stand in for one: prediction files made
from structures whose hydrogens are absent understate every proton nucleus and shift the
heavy ones, so the run that this column reports is the one whose inputs are complete.

Both UCBShift blocks are scored without the cells UCBShift's paper removes from its own
test set (benchmarks/reference/ucb200_paper_exclusions.json: every C' of three chains whose
C' is anticorrelated with prediction, and nine ESI cells), the cells Tables S19 and S20
remove too; the published values those blocks are set against were computed without them.

The SHIFTX2 block is SHIFTX2 as distributed (SHIFTY+ with every reference) on its own test
set, read from the 61-set pool Table S20 reads ($SI_UCB200_POOL61), whose residues are matched
to each deposition by sequence alignment; the transfer-module row is read from the
benchmark's frozen gate report.
"""
import json

import si_common

# Every input lives outside the release repo: the frozen gate report is written by the
# competitor run, and the rerun on rebuilt structures is a finished measurement kept
# beside it.
REPORT = si_common.COMPETITOR_ROOT / "reproduction" / "reproduction_report.json"
RERUN_ROOT = si_common.root("SI_GATE_UCB200_RERUN")
POOL61 = si_common.root("SI_UCB200_POOL61")
DDAGGER = r"$^{\ddagger}$"
RERUN = RERUN_ROOT / "score_ucb200.json"
# The rerun scored without the excluded cells (probe/paper_exclusions.py beside it).
PAPER = RERUN_ROOT / "probe" / "paper_exclusions_result.json"
# The authors' own test set, in their checkout: it is the denominator of the cohort, the
# proteins their released code was invoked on.
CSPRED_TM = si_common.root("SI_GATE_CSPRED_TM")
OUT = si_common.OUT

ATOMS = ["N", "CA", "CB", "C", "H", "HA"]        # print order, six nuclei
PLACES = 4                                       # printed precision of every value cell

# ---------------------------------------------------------------- published anchors
# UCBShift: Li, Bhattacharya and Head-Gordon, Chem. Sci. 2020, 11, 3180-3191.
#   Table 3 (RMSE of the structure module) of the paper;
#   Table S4 (RMSE and MAE, uncurated Test set, 200 proteins) of its supporting information.
PUB_UCB_COMBINED_MAE = {"N": "1.06", "CA": "0.43", "CB": "0.47",
                        "C": "0.48", "H": "0.18", "HA": "0.11"}
PUB_UCB_X_RMSE = {"N": "2.49", "CA": "1.08", "CB": "1.28",
                  "C": "1.17", "H": "0.44", "HA": "0.25"}
# SHIFTX2: Han, Liu, Ginzinger and Wishart, J. Biomol. NMR 2011, 50, 43-57.
#   supporting information Table S9, SHIFTX2 RMSD column.
PUB_SHIFTX2_RMSE = {"N": "1.1169", "CA": "0.4412", "CB": "0.5163",
                    "C": "0.5330", "H": "0.1711", "HA": "0.1231"}

NUC_TEX = {"H": r"H$^{\mathrm{N}}$", "HA": r"H$\alpha$", "C": r"C$'$",
           "CA": r"C$\alpha$", "CB": r"C$\beta$", "N": "N"}
DAGGER = r"$^{\dagger}$"


# ---------------------------------------------------------------- table
def fmt(x, places=PLACES):
    return f"{x:.{places}f}"


def signed(x, places=PLACES):
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.{places}f}"


def row(nucleus, cells, published, ours):
    """One line. The difference is between the two values the row prints, so a reader
    checks the column by subtracting what is in front of them."""
    difference = round(ours, PLACES) - float(published)
    return (f"{nucleus} & {cells:,} & {published} & {fmt(ours)} & "
            f"{signed(difference)} \\\\")


def block(title, statistic, cells, published, ours):
    """One block: a spanning title, then the six nuclei."""
    lines = [rf"\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{title}}} --- {statistic}, ppm}} \\"]
    for atom in ATOMS:
        lines.append(row(NUC_TEX[atom], cells[atom], published[atom], ours[atom]))
    return lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text())
    rerun = json.loads(RERUN.read_text())
    paper = json.loads(PAPER.read_text())
    pool61 = json.loads(POOL61.read_text())
    shiftx2 = pool61["own_cells"]["shiftx2_asdistributed"]
    scorable = pool61["n_scorable_proteins"]
    test_proteins = scorable + len(pool61["protein_exclusions"])
    combined = paper["arms"]["UCBShift_H"]
    structure = paper["arms"]["X_H"]

    checks = {}
    checks["shiftx2_scored_on_its_own_test_set"] = {
        "what": "SHIFTX2 as distributed in the 61-set pool: every nucleus carries cells",
        "proteins_scored": scorable,
        "agrees": scorable > 0 and all(shiftx2[a]["n"] > 0 for a in ATOMS),
    }

    # The excluded cells are removed by name, so per nucleus they must be exactly the
    # cells the rows lose against the unexcluded rerun.
    lost = {label: {a: rerun["per_arm_own_cells"][label][a]["n"] - paper["arms"][label][a]["n"]
                    for a in ATOMS} for label in ("UCBShift_H", "X_H")}
    named = {label: {a: sum(n for key, n in paper["removed"][label].items()
                            if key.split(":")[1] == a) for a in ATOMS}
             for label in ("UCBShift_H", "X_H")}
    checks["paper_exclusions_account_for_the_cells_removed"] = {
        "what": "per nucleus, the cells the excluded list names against the cells the two "
                "UCBShift blocks lose",
        "removed_by_chain_and_atom": paper["removed"],
        "agrees": lost == named,
    }

    # The structure module's published anchors are also recorded by the rerun that
    # produced the column beside them; a disagreement means one of the two read the
    # paper differently.
    recorded = rerun["published"]["UCBShift-X"]
    checks["structure_module_anchors_agree_with_the_rerun_record"] = {
        "what": "the Table 3 values typed here, against the same values the rerun stored",
        "agrees": all(float(PUB_UCB_X_RMSE[a]) == recorded[a] for a in ATOMS),
    }

    # The cohort is a subset of the authors' own test set: every protein scored here is
    # one of theirs, and the ones missing are those their code returned nothing for.
    test_set = len(list(CSPRED_TM.glob("test_targets/*.csv")))
    scored_proteins = rerun["funnel"]["chains_with_new_preds"]
    checks["the_cohort_is_within_the_released_test_set"] = {
        "what": "proteins with a prediction, against the size of the released test set",
        "test_set_proteins": test_set,
        "proteins_with_predictions": scored_proteins,
        "agrees": 0 < scored_proteins <= test_set,
    }
    for name, check in checks.items():
        if not check["agrees"]:
            raise ValueError(f"{name}: {json.dumps(check)}")

    y_cells = sum(v["_Y"]["n"] for v in report["ucb_rerun_agreement"].values()
                  if isinstance(v, dict) and v.get("_Y"))
    y_max = max(v["_Y"]["max_abs_diff"] for v in report["ucb_rerun_agreement"].values()
                if isinstance(v, dict) and v.get("_Y"))
    # UCBShift released Y-module outputs for a few test proteins only; the row says how many.
    y_proteins = sum(1 for v in report["ucb_rerun_agreement"].values()
                     if isinstance(v, dict) and v.get("_Y"))

    banner = "\n".join([
        "% Component only: no caption prose. Title line is a short title, not a caption.",
        "%",
        "% The published column carries only a value the predictor's own paper prints:",
        "%   the combined output's mean absolute error from the UCBShift supporting",
        "%   information, the structure module's root mean square error from its Table 3,",
        "%   and SHIFTX2's from its own supporting information. Each block is reported in",
        "%   the statistic its paper publishes, which is why the two differ.",
        "% This work is our invocation of their released code on their released test",
        "%   structures with hydrogens rebuilt, scored against their released measured",
        "%   values with their own filter.",
        "% The daggered UCBShift blocks drop the cells UCBShift's paper removes from its",
        "%   test set (benchmarks/reference/ucb200_paper_exclusions.json), as Tables S19",
        "%   and S20 do.",
        "% The difference column is this work minus published, so positive is worse.",
    ])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Each predictor reproduced on its own published test set, before it was "
        r"run on ours}\label{tab:reproduction_gate}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Nucleus & Cells & Published & This work & Difference \\",
        r"\midrule",
    ]
    lines += block(
        "UCBShift" + DAGGER, "MAE",
        {a: combined[a]["n"] for a in ATOMS}, PUB_UCB_COMBINED_MAE,
        {a: combined[a]["mae"] for a in ATOMS})
    lines.append(r"\addlinespace")
    lines += block(
        "UCBShift-X" + DAGGER, "RMSE",
        {a: structure[a]["n"] for a in ATOMS}, PUB_UCB_X_RMSE,
        {a: structure[a]["rmse"] for a in ATOMS})
    lines += [
        r"\addlinespace",
        r"\multicolumn{5}{@{}l}{\textbf{UCBShift-Y}\cite{UCBShift} --- ppm} \\",
        rf"Max $|\Delta|$ from released values, {y_proteins} proteins & {y_cells:,} & & & "
        rf"{fmt(y_max)} \\",
        r"\addlinespace",
        r"\multicolumn{5}{@{}l}{\textbf{SHIFTX2" + DDAGGER + r"} --- RMSE, ppm} \\",
    ]
    for atom in ATOMS:
        lines.append(row(NUC_TEX[atom], shiftx2[atom]["n"],
                         PUB_SHIFTX2_RMSE[atom], shiftx2[atom]["rmse"]))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"{\footnotesize " + DAGGER + r"excludes the cells UCBShift's paper excludes from "
        r"its test set\quad " + DDAGGER + r"as distributed, every reference available, on "
        + f"{scorable}" + r" of its " + f"{test_proteins}" + r" test proteins}",
        r"\end{table}", ""]

    si_common.emit(
        "si_reproduction_gate", "\n".join(lines), banner=banner,
        script="build_reproduction_gate.py", placeholder_caption=False)
    for name, c in checks.items():
        print(f"  check {name}: agrees={c['agrees']}")


if __name__ == "__main__":
    main()
