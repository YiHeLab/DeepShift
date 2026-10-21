#!/usr/bin/env python3
"""Backbone accuracy against four competitors, uncentred -- deployed as fig_03_benchmark.

Both panels and Table 1's body come from two reports of one scoring run:

    python -m benchmarks.score_by_source_tier --tier 2 --tools ... --out <marginal>.json
    python -m benchmarks.score_paired_tier    --tier 2 --tools ... --out <paired>.json \\
        --check_marginal <marginal>.json

The marginal report carries each arm's RMSE, MAE and secondary-shift correlation with
their cluster-bootstrap intervals; the paired report carries the interval on the
difference, which is what an asterisk may claim. This script refuses the pair unless
they are one run: the same cohort and the same resampling unit.

Panel (b) is the secondary-shift correlation: each residue type's observed mean is
removed from the observation and from every prediction, so a per-residue-type offset
cannot flatter an arm. Panel (c) is the RMSE of the residual, prediction minus the
re-referenced observed label, under whichever residual convention the marginal report
records. The letters are b and c because the deployed float stacks fig_02_accuracy above
this panel.

Exclusion tier. Tier 2 is the deployed policy, and the only one drawn: every retrieval arm
is blocked from all depositions of the query's own protein -- own BMRB, same-UniProt
family, own-PDB-keyed references -- with homologs of other proteins left alone. The
reports state the tier; `--arms` names the arms that were scored under it.

Cells. When the scoring run was given the frozen master cells, every arm of the run --
not only the five drawn here -- was scored on Master(nucleus), so the table is comparable
with the supplementary tables built from the same reports. An arm covering less of a
nucleus than the master floor asks is scored on the master cells it produced; its name
then carries a marker and the row beneath prints what it was scored on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import frozen
import matplotlib.pyplot as plt
import numpy as np
import style

# The manuscript packet's rcParams: the house style with constrained_layout off (these
# panels call tight_layout) and the packet's hairline weights. One definition, shared
# with the seven deployed floats build_paper_figures.py draws.
from build_paper_figures import house_style, panel_letter

from benchmarks.runner_up_leads import lead_tests

NUCLEI = ["CA", "CB", "C", "HA", "H", "N"]
TEX_HEAD = [r"C$\alpha$", r"C$\beta$", r"C$'$", r"H$\alpha$", "H", "N"]
FIG_ORDER = ["N", "CA", "CB", "C", "H", "HA"]
FIG_LABEL = [r"$^{15}$N", r"$^{13}$C$\alpha$", r"$^{13}$C$\beta$",
             r"$^{13}$C$^{\prime}$", r"$^{1}$H$^{N}$", r"$^{1}$H$\alpha$"]
SOURCES = [("xtal", "Crystal"), ("af", "AlphaFold"), ("nmr", "NMR")]

# How each arm is drawn, by the base its scored name starts with. The `ucbshift_full_*`
# and `ucbshift_x` arms invoke UCBShift, backbone and C-beta only -- not UCBShift 2.0,
# which is the side-chain predictor in fig_sidechain_benchmark. UCBShift-X is a FACET of
# the UCBShift amber, drawn hollow, not a fifth hue.
ARM_STYLE = {
    "deepshift": (r"\textbf{DeepShift}", "DeepShift", style.ACCENT, False),
    "ucbshift_full": ("UCBShift", "UCBShift", style.AMBER, False),
    "ucbshift_x": ("UCBShift-X", "UCBShift-X", style.AMBER, True),
    "shiftx2": ("SHIFTX2", "SHIFTX2", style.VIOLET, False),
    "sparta": ("SPARTA+", "SPARTA+", style.NEUTRAL, False),
}
DEFAULT_ARMS = ("deepshift", "ucbshift_full_sameprot", "ucbshift_x", "shiftx2_sameprot",
                "sparta")
PARTIAL = r"$^{\ast}$"              # this arm covers less of a nucleus than the master asks
# What the paired scorer's resampling unit is called on the page. Clustering on the label
# set is the coarser of the two -- one deposition's shifts are realized on up to three
# structures, so the instances within it are not independent draws.
UNIT_NAMED = {"structure": "structure instances", "labelset": "label sets"}
DEPLOYED_TIER = 2
#: Table 1 rows typed into main.tex from Table S12's record of the same marginal report.
TABLE1_TYPED_ARMS = ("mlff",)


def arm_style(key: str):
    """The drawing style of one scored arm, from the competitor its name begins with."""
    for base, entry in ARM_STYLE.items():
        if key == base or key.startswith(base + "_"):
            return entry
    raise SystemExit(f"no drawing style for the arm {key!r}; it names no known competitor")


def read(marginal: Path, paired: Path, arms) -> tuple[dict, dict]:
    """The two reports, checked to be one scoring run."""
    reports = {name: json.loads(path.read_text())
               for name, path in (("marginal", marginal), ("paired", paired))}
    for field in ("n_common_structures", "n_common_cells"):
        if reports["marginal"][field] != reports["paired"][field]:
            raise SystemExit(f"the two reports disagree on {field}: "
                             f"{reports['marginal'][field]} against {reports['paired'][field]}")
    for name, path in (("paired", paired), ("marginal", marginal)):
        if reports[name].get("unit") not in UNIT_NAMED:
            raise SystemExit(f"{path.name} names the resampling unit "
                             f"{reports[name].get('unit')!r}, which this float cannot "
                             "state; an interval is a claim about one clustering")
    if missing := [a for a in arms if a not in reports["marginal"]["methods"]]:
        raise SystemExit(f"{marginal.name} scored {reports['marginal']['methods']}, "
                         f"which does not include {missing}")
    return reports["marginal"], reports["paired"]


def verdicts(marginal: dict, paired: dict, source: str, nucleus: str, arms):
    """(winning arm, whether its lead is significant) for one table cell.

    The winner is the lowest RMSE among the arms the table prints, since a cell can only
    claim to win the comparison it shows. The asterisk is the paired interval on the
    winner's lead over the runner-up, which exists only for differences the paired report
    carries: those against its reference arm.
    """
    block = marginal["by_source"][source]["nuclei"][nucleus]
    ranked = sorted(arms, key=lambda arm: block[arm]["rmse"])
    winner, runner_up = ranked[0], ranked[1]
    if winner != paired["reference_arm"]:
        return winner, False
    differences = paired[source][nucleus]
    if runner_up not in differences:
        raise SystemExit(f"the paired report carries no {source}/{nucleus} difference "
                         f"against {runner_up}, so its lead cannot be called")
    return winner, differences[runner_up]["ci"][1] < 0


def table(marginal: dict, paired: dict, arms) -> str:
    """Table 1's body: bold on the lowest value, asterisk when the lead over the
    runner-up has a 95 % paired interval excluding zero, and, for an arm scored on fewer
    than every master cell of a nucleus, the count it was scored on beneath its row."""
    rows = []
    for src, pretty in SOURCES:
        block = marginal["by_source"][src]
        rows.append(r"\midrule")
        lines = []
        for arm in arms:
            nodes = [block["nuclei"][nuc][arm] for nuc in NUCLEI]
            cells = []
            for nuc, node in zip(NUCLEI, nodes, strict=True):
                txt = f"{node['rmse']:.4f}"
                winner, significant = verdicts(marginal, paired, src, nuc, arms)
                if winner == arm:
                    txt = rf"\textbf{{{txt}}}" + ("*" if significant else "")
                cells.append(txt)
            short = any(node.get("below_master_coverage") for node in nodes)
            lines.append(" & " + arm_style(arm)[0] + (PARTIAL if short else "") + " & "
                         + " & ".join(cells) + r" \\")
            if short:
                lines.append(r" & \emph{cells scored} & "
                             + " & ".join(f"{node['n_cells']:,}".replace(",", "{,}")
                                          for node in nodes) + r" \\")
        # the source label spans every line the block prints, the cell counts included
        n_structures = block["n_structures"]
        rows.append(rf"\multirow{{{len(lines)}}}{{*}}{{{pretty} ($n={n_structures}$)}}")
        rows += lines
    head = (r"\begin{tabular}{llcccccc}" + "\n" + r"\toprule" + "\n"
            + "Source & Method & " + " & ".join(TEX_HEAD) + r" \\" + "\n")
    return head + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"


def draw(block: dict, arms):
    house_style()
    fig, (ax_r, ax_e) = plt.subplots(2, 1, figsize=(style.WIDTH["text"], 3.80),
                                     sharex=True)
    x = np.arange(len(FIG_ORDER))
    width = 0.62 / len(arms)
    for i, arm in enumerate(arms):
        _tex, name, color, open_face = arm_style(arm)
        off = (i - (len(arms) - 1) / 2) * width
        bar = style.bar_style(color, open_face)
        for ax, field in ((ax_r, "r"), (ax_e, "rmse")):
            v = np.array([block[a][arm][field] for a in FIG_ORDER])
            ci = np.array([block[a][arm][f"{field}_ci95"] for a in FIG_ORDER]).T
            ax.bar(x + off, v, width, label=name if ax is ax_r else "_nolegend_", **bar)
            ax.errorbar(x + off, v, yerr=np.abs(ci - v), fmt="none",
                        ecolor=style.INK["primary"], elinewidth=0.5, capsize=1.2)

    ax_r.set_ylabel("Secondary-shift $R$")
    ax_r.set_ylim(0, 1.32)                      # headroom for the framed legend
    ax_r.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    leg = ax_r.legend(ncol=5, loc="upper center", columnspacing=1.0, borderpad=0.4,
                      frameon=True, framealpha=1.0, edgecolor=style.INK["grid"],
                      fancybox=False, handlelength=1.2)
    leg.get_frame().set_linewidth(0.6)
    for ax in (ax_r, ax_e):
        ax.grid(axis="y", lw=0.5, color=style.INK["grid"])
        ax.set_axisbelow(True)
    ax_e.set_ylabel("RMSE (ppm)")
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(FIG_LABEL)
    for ax, letter in ((ax_r, "b"), (ax_e, "c")):
        panel_letter(ax, letter, dx=-0.055)
    fig.tight_layout(pad=0.5, h_pad=0.7)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marginal", type=Path, required=True,
                    help="the report benchmarks.score_by_source_tier wrote")
    ap.add_argument("--paired", type=Path, required=True,
                    help="the report benchmarks.score_paired_tier wrote for the same run")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS),
                    help="the scored arm names to draw and tabulate, DeepShift first")
    ap.add_argument("--marginal-without-exclusions", type=Path, default=None,
                    help="the same scoring on the master set before the anticorrelated "
                         "channels are removed, for Methods' without-exclusions sentence")
    ap.add_argument("--paired-without-exclusions", type=Path, default=None)
    args = ap.parse_args()
    if (args.marginal_without_exclusions is None) != (args.paired_without_exclusions is None):
        raise SystemExit("--marginal-without-exclusions and --paired-without-exclusions "
                         "come as a pair")
    arms = list(args.arms)
    marginal, paired = read(args.marginal, args.paired, arms)
    tier = paired["tier"]
    if tier != DEPLOYED_TIER:
        raise SystemExit(f"the reports are tier {tier}; fig_03_benchmark and Table 1 are "
                         f"scored at tier {DEPLOYED_TIER}")

    block = marginal["by_source"]["all"]["nuclei"]
    frozen.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = frozen.OUT_DIR / "tab_accuracy_common_rmse.tex"
    out.write_text(table(marginal, paired, arms))
    print(f"wrote {out}")

    # main.tex's significance count: the runner-up among Table 1's printed rows
    printed = [a for a in (*arms, *TABLE1_TYPED_ARMS) if a != paired["reference_arm"]]
    print(json.dumps({"lead_over_runner_up": lead_tests(marginal, paired, printed)}))
    if args.marginal_without_exclusions:
        # Methods: without the anticorrelated-channel exclusion, which leads keep their
        # significance. The same test on the same rows, on the set before the exclusion.
        m2, p2 = read(args.marginal_without_exclusions, args.paired_without_exclusions, arms)
        if p2["tier"] != tier:
            raise SystemExit("the without-exclusions reports are not the same tier")
        print(json.dumps({"lead_over_runner_up_without_exclusions": {
            "master_cells": m2.get("master_cells"),
            "n_cells": m2["n_common_cells"],
            "table1_rows": lead_tests(m2, p2, printed)}}))
    frozen.emit(draw(block, arms), "fig_03_benchmark")


if __name__ == "__main__":
    main()
