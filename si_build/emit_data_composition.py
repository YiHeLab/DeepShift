#!/usr/bin/env python3
"""The training-composition table, with every baseline stated.

Two questions sit in one table: whether more proteins beat more repeats at equal rows,
and how much repeats reduce error. They are different questions, so the table prints
absolute RMSE in ppm on both sides of every comparison and names the reference set in the
row stub itself, where a bare column of reductions would leave the direction and the size
of the effect unreadable.

Both blocks hold out fold 1 and train on folds 2-5, but they score different cohorts, so
the two blocks are not comparable line by line and the table says so.

Numbers come from the experiment records the environment names, never from the deployed
table:
  block 1  the data-diversity arms report and arms comparison
  block 2  the source-transfer one-per-protein and source-matched reports
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import si_common

# Two finished measurements held in the archive; the SI_*_DIR
# variables name another copy of either.
DIV = Path(os.environ.get("SI_DATA_DIVERSITY_DIR",
                          str(si_common.archive("data_diversity"))))
REP = Path(os.environ.get("SI_SOURCE_TRANSFER_DIR",
                          str(si_common.archive("source_transfer"))))
OUT = si_common.OUT

NUC = ["h_shift", "ha_shift", "ca_shift", "cb_shift", "c_shift", "n_shift"]
HEAD = {"h_shift": r"H$^{\mathrm{N}}$", "ha_shift": r"H$\alpha$", "ca_shift": r"C$\alpha$",
        "cb_shift": r"C$\beta$", "c_shift": r"C$'$", "n_shift": "N"}
SRC = {"af": "AlphaFold", "nmr": "NMR", "xtal": "X-ray crystal"}


def load():
    arms = json.loads((DIV / "arms_report.json").read_text())
    cmp_ = json.loads((DIV / "arms_comparison.json").read_text())
    pool = json.loads((REP / "one_per_protein_report.json").read_text())
    matched = json.loads((REP / "source_matched_report.json").read_text())

    # Block 1. The comparison file stores each contrast as baseline -> candidate, and
    # both growth paths start from the same floor set, so the floor's RMSE is
    # read back from either contrast and the two must agree.
    c_red = cmp_["contrasts"]["floor__to__redundant"]
    c_div = cmp_["contrasts"]["floor__to__diverse"]
    for n in NUC:
        if abs(c_red[n]["rmse_baseline"] - c_div[n]["rmse_baseline"]) > 1e-12:
            raise SystemExit(f"floor RMSE disagrees between contrasts at {n}")
    block1 = {
        "floor": {n: c_red[n]["rmse_baseline"] for n in NUC},
        "redundant": {n: c_red[n]["rmse_candidate"] for n in NUC},
        "diverse": {n: c_div[n]["rmse_candidate"] for n in NUC},
        "cells": {n: int(c_red[n]["n"]) for n in NUC},
        "rows": {k: arms["arms"][k]["n_rows"] for k in ("floor", "redundant", "diverse")},
        "label_sets": {k: arms["arms"][k]["n_label_sets"]
                       for k in ("floor", "redundant", "diverse")},
        "heldout_fold": arms["heldout_fold"], "train_folds": arms["train_folds"],
    }

    # Block 2. Both absolute sides of the one_per_protein vs shipped_all comparison on the
    # source-matched cohort.
    block2 = {"pool": pool["pool"], "one_per": pool["arm"], "per_source": {}}
    for s in SRC:
        x = matched["table_a"]["matched"][s]
        block2["per_source"][s] = {
            "one_per_protein": {n: x["one_per_protein"][n]["rmse"] for n in NUC},
            "every_structure": {n: x["shipped_all"][n]["rmse"] for n in NUC},
            "cells": sum(x["shipped_all"][n]["n"] for n in NUC)}
    return block1, block2


def caption_facts_comment(b1, b2) -> str:
    """The facts the caption needs, as the comment beneath the float.

    They are facts about the cohort rather than about the rows, so they belong to the
    caption and not to the page; printing them beside the table would put a second set of
    counts on it.
    """
    folds = b1["train_folds"]
    facts = (
        "Facts to assert: RMSE in ppm, raw residuals, on structures held out of "
        f"training. Both halves hold out fold {b1['heldout_fold']} and train on folds "
        f"{min(folds)}--{max(folds)}, and the top half scores "
        f"{sum(b1['cells'].values()):,} shifts, but the two score different sets of "
        "shifts, so a line in the top half is not comparable to a line in the bottom "
        "half. Top half: the second and third lines carry the same number of training "
        f"rows ({b1['rows']['redundant']:,} and {b1['rows']['diverse']:,}), and differ "
        "only in whether those rows are three structures of "
        f"{b1['label_sets']['redundant']:,} proteins or one structure of "
        f"{b1['label_sets']['diverse']:,} proteins; the first line is the one-third-size "
        f"starting point both grew from. Bottom half: both sides train on all "
        f"{b2['pool']['label_sets']:,} proteins, and differ only in whether every "
        "deposited structure is kept or one is drawn at random per protein. Read "
        "together: repeat structures do help, but a row of a new protein is worth more "
        "than a row of a protein already in the set. Protein here means one deposited "
        "label set."
    )
    return textwrap.fill(facts, width=86, initial_indent="% ", subsequent_indent="% ")


def build_table(b1, b2) -> str:
    def cells(d):
        return " & ".join(f"{d[n]:.3f}" for n in NUC)

    L = [r"\begin{table}[H]\centering\small",
         r"\caption{What extra training data buys, by what kind of extra it is}",
         r"\label{tab:data_diversity}",
         r"\label{tab:data_repeats}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l r r r r r r r}",
         r"\toprule",
         r"Training set & Rows & " + " & ".join(HEAD[n] for n in NUC) + r" \\",
         r"\midrule",
         r"\multicolumn{8}{l}{\textit{Adding proteins vs adding structures}} \\",
         (f"\\quad {b1['label_sets']['floor']:,} proteins, one structure each "
          r"--- the starting point & "
          f"{b1['rows']['floor']:,} & " + cells(b1["floor"]) + r" \\"),
         (f"\\quad the same {b1['label_sets']['redundant']:,} proteins, all three "
          r"structures each & "
          f"{b1['rows']['redundant']:,} & " + cells(b1["redundant"]) + r" \\"),
         (f"\\quad {b1['label_sets']['diverse']:,} different proteins, one structure "
          r"each & "
          f"{b1['rows']['diverse']:,} & " + cells(b1["diverse"]) + r" \\"),
         r"\addlinespace",
         r"\multicolumn{8}{l}{\textit{Repeat structures, protein set fixed}} \\"]

    for s in ("af", "nmr", "xtal"):
        d = b2["per_source"][s]
        L += [(r"\multicolumn{8}{l}{\quad tested on " + SRC[s] + " structures, "
               f"{d['cells']:,}" r" shifts} \\"),
              (r"\qquad one randomly chosen structure per protein --- the reference & "
               f"{b2['one_per']['rows']:,} & " + cells(d["one_per_protein"]) + r" \\"),
              (r"\qquad every structure kept, against one per protein & "
               f"{b2['pool']['rows']:,} & " + cells(d["every_structure"]) + r" \\")]
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", "",
          caption_facts_comment(b1, b2)]
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b1, b2 = load()
    banner = "\n".join([
        "% Both tab:data_diversity and tab:data_repeats label this one table, because",
        "% main.tex cites both.",
        "% Numbers are read from the experiment records, not from the deployed table:",
        "%   the data-diversity arms report and arms comparison,",
        "%   the source-transfer one-per-protein and source-matched reports.",
    ])

    si_common.emit(
        "si_data_composition", build_table(b1, b2), banner=banner,
        script="emit_data_composition.py")
    print(f"block 1 rows: floor {b1['rows']['floor']:,}  "
          f"repeats {b1['rows']['redundant']:,}  proteins {b1['rows']['diverse']:,}")
    print(f"block 2 rows: one-per {b2['one_per']['rows']:,}  "
          f"all {b2['pool']['rows']:,}  ({b2['pool']['label_sets']:,} proteins both sides)")


if __name__ == "__main__":
    main()
