#!/usr/bin/env python3
"""The fold-composition census of one run.

Every number is counted here, from the run named by ``$DEEPSHIFT_RUN``: the realization
ledger, the per-source materialization masters, the fold map and the exclusion authority
the run trains under. The realized label sets and the fold map must be the same set, or
nothing is written. Table S1's per-source counts are printed after the table is written.

Usage:
    DEEPSHIFT_RUN=/path/to/a/run python gen_census_tables.py \\
        --flags /path/to/a/run/data/qc_flags_p100_structural.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import si_common
from si_common import tex_thousands as n

# Run directly (`python si_build/gen_census_tables.py`), so make the package importable
# without asking the caller to set PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deepshift.data import census as corpus
from deepshift.data import folds as fold_stage
from deepshift.domain import SHIFT_COLUMNS, VALID_SOURCES
from deepshift.quality_control import FINAL_QC_POLICY

MASTER = "structure_data_{source}.csv"
MATCH = "match"
CHUNK_ROWS = 200_000


# ---------------------------------------------------------------- counting

def fold_role(fold: int) -> str:
    if fold == fold_stage.EXCLUDED_FOLD:
        return "excluded from training"
    if fold == fold_stage.HOLDOUT_FOLD:
        return "held out"
    return "cross-validation"


def fold_counts(data_dir: Path, fold_of: dict[str, int]) -> tuple[dict, dict]:
    """Residue rows and observed cells at realization grain, by fold.

    A row counts when its residue matches the coordinates; a cell counts when the shift
    is present.
    """
    columns = ["label_set_id", "mismatch_type", *SHIFT_COLUMNS]
    by_fold = {fold: {"residue_rows": 0, "observed_cells": 0}
               for fold in sorted(set(fold_of.values()))}
    # Table S1's per-source rows (typed into SI.tex) are counted here too: every residue
    # row of the source's master, the rows matched to a resolved coordinate, and their cells.
    by_source = {s: {"residue_rows": 0, "mapped_rows": 0, "observed_cells": 0}
                 for s in VALID_SOURCES}
    for source in VALID_SOURCES:
        path = data_dir / MASTER.format(source=source)
        if not path.is_file():
            raise SystemExit(f"missing materialization master: {path}")
        for chunk in pd.read_csv(path, usecols=columns, chunksize=CHUNK_ROWS,
                                 dtype={"label_set_id": str, "mismatch_type": str},
                                 low_memory=False):
            per_row = chunk[list(SHIFT_COLUMNS)].notna().to_numpy().sum(axis=1)
            matched = (chunk["mismatch_type"] == MATCH).to_numpy()
            assigned = chunk["label_set_id"].map(fold_of)
            unassigned = int((matched & assigned.isna().to_numpy()).sum())
            if unassigned:
                raise SystemExit(
                    f"{path.name} matches {unassigned} residue rows whose label set "
                    "carries no fold, so the master and the fold map are from "
                    "different builds")
            folds = assigned.to_numpy()
            by_source[source]["residue_rows"] += len(chunk)
            by_source[source]["mapped_rows"] += int(matched.sum())
            by_source[source]["observed_cells"] += int(per_row[matched].sum())
            for fold in by_fold:
                keep = matched & (folds == fold)
                by_fold[fold]["residue_rows"] += int(keep.sum())
                by_fold[fold]["observed_cells"] += int(per_row[keep].sum())
    return by_fold, by_source


def check_exclusion_authority(path: Path) -> None:
    """Refuse an exclusion authority whose report does not declare the fence this release
    applies."""
    report_path = path.with_suffix(".report.json")
    if not report_path.is_file():
        raise SystemExit(f"{path} carries no {report_path.name}, so its policy is unstated")
    report = json.loads(report_path.read_text())
    if report.get("policy") != FINAL_QC_POLICY:
        raise SystemExit(
            f"{report_path.name} declares policy {report.get('policy')!r}; the census "
            f"counts the corpus under {FINAL_QC_POLICY!r}")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=si_common.root("DEEPSHIFT_RUN"),
                    help="run root; the census is taken from its data directory")
    ap.add_argument("--flags", type=Path, required=True,
                    help="the exclusion authority the run trains under")
    args = ap.parse_args()

    data_dir = args.run / "data"
    out = si_common.OUT
    out.mkdir(parents=True, exist_ok=True)

    ledger = corpus.read_realizations(data_dir)
    fold_of = corpus.read_fold_map(data_dir)
    corpus.bind_fold_map(fold_of, set(ledger["label_set_id"]))
    check_exclusion_authority(args.flags)
    by_fold, by_source = fold_counts(data_dir, fold_of)

    banner = "\n".join([
        "% The caption is a short title; the facts it must assert are listed in the",
        "% \\note{} beside the float.",
    ])

    # -------------------------------------------------- fold composition
    fold_rows, totals = [], dict.fromkeys(
        ("label_sets", "bmrb_entries", "af", "nmr", "xtal", "structures_total",
         "rows", "cells"), 0)
    composition = {}
    for fold in sorted(set(fold_of.values())):
        members = {ls for ls, f in fold_of.items() if f == fold}
        block = ledger[ledger["label_set_id"].isin(members)]
        structures = {src: int((block["source"] == src).sum()) for src in VALID_SOURCES}
        record = {
            "role": fold_role(fold), "label_sets": len(members),
            "bmrb_entries": int(block["bmrb_id"].nunique()), "structures": structures,
            "structures_total": len(block),
            "residue_rows": by_fold[fold]["residue_rows"],
            "observed_cells": by_fold[fold]["observed_cells"],
        }
        composition[str(fold)] = record
        fold_rows.append(
            f"{fold} & {record['role']} & {n(record['label_sets'])} & "
            f"{n(record['bmrb_entries'])} & {n(structures['af'])} & "
            f"{n(structures['nmr'])} & {n(structures['xtal'])} & "
            f"{n(record['structures_total'])} & {n(record['residue_rows'])} & "
            f"{n(record['observed_cells'])} \\\\")
        totals["label_sets"] += record["label_sets"]
        totals["bmrb_entries"] += record["bmrb_entries"]
        for src in VALID_SOURCES:
            totals[src] += structures[src]
        totals["structures_total"] += record["structures_total"]
        totals["rows"] += record["residue_rows"]
        totals["cells"] += record["observed_cells"]
    fold_rows.append(
        f"\\midrule \\textbf{{all}} & & \\textbf{{{n(totals['label_sets'])}}} & "
        f"{n(totals['bmrb_entries'])} & {n(totals['af'])} & {n(totals['nmr'])} & "
        f"{n(totals['xtal'])} & \\textbf{{{n(totals['structures_total'])}}} & "
        f"{n(totals['rows'])} & {n(totals['cells'])} \\\\")

    holdout = composition[str(fold_stage.HOLDOUT_FOLD)]
    excluded = composition[str(fold_stage.EXCLUDED_FOLD)]
    folds_tex = r"""\begin{table}[H]\centering\small
\caption{Fold composition of the training build}
\label{tab:census_folds}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrr}
\toprule
Fold & Role & Label sets & BMRB\cite{BMRB} & AF & NMR & X-ray & Structure realizations & Mapped residue rows & Observed cells \\
\midrule
""" + "\n".join(fold_rows) + r"""
\midrule
\multicolumn{10}{@{}p{1.3\linewidth}@{}}{\footnotesize \emph{Cohort}: the whole training
build. \emph{Units}: counts; residue rows and observed cells are at the realization grain
and count matched residues only. \emph{Tier}: not applicable. Fold identity is
build-specific and does not carry across rebuilds. Fold~""" + str(fold_stage.HOLDOUT_FOLD) \
        + " is the held-out cohort (" + n(holdout["label_sets"]) + " label sets / " \
        + n(holdout["structures_total"]) + " structure realizations); fold~" \
        + str(fold_stage.EXCLUDED_FOLD) + " holds the " + n(excluded["label_sets"]) \
        + r""" label sets excluded from training.} \\
\bottomrule
\end{tabular}}
\end{table}
"""
    si_common.emit("si_census_folds", folds_tex, script="gen_census_tables.py",
                   banner=banner, placeholder_caption=False)

    print("wrote si_census_folds to", out)
    print("Table S1 per-source counts:", json.dumps(by_source, indent=1))


if __name__ == "__main__":
    main()
