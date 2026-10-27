#!/usr/bin/env python3
"""Who actually produced a prediction, and on which cells.

The table the competitor comparison needs before any of its numbers can be read, built
from this run: the fold-6 task table, the per-task outputs each predictor
wrote under the run's benchmark root, the coverage census measured on them, and the
scoring report the results tables print.

  si_coverage_ledger    Every predictor crashes or declines on a different subset, and
                        on the subsets it answers it labels a different set of nuclei.
                        Until that is stated, "all methods are scored on the same cells"
                        is an assertion, not a demonstration. This is the ledger: tasks
                        dispatched, tasks answered, tasks lost, how many of the scored
                        nuclei the tool labels, and the share of DeepShift's scorable
                        cells it produced a value for -- the coverage the master cell set
                        was frozen on. `Nuclei returned' is counted over the whole run,
                        so a tool is not judged on whichever residues its first answer
                        happened to contain, and a tool that also labels heme, ligand or
                        solvent atoms is credited with its protein prediction only.

Usage:
    python si_competitor_ledger.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import si_common as C
from emit_sidechain_final import ARMS as SIDECHAIN_ARMS

from benchmarks.config import MASTER_MARKED_ARMS
from deepshift.chemistry import SHIFT_ATOM_NAMES, atom_from_shift_column
from deepshift.self_exclusion import arm_directory

BENCH = Path(os.environ.get("DEEPSHIFT_BENCH_OUT") or C.BUILD_ROOT / "benchmark")
TASKS = BENCH / "fold6_tasks.tsv"
# DEEPSHIFT_COVERAGE_CENSUS names a census re-run beside the frozen one (benchmarks.master_cells)
CENSUS = Path(os.environ.get("DEEPSHIFT_COVERAGE_CENSUS")
              or BENCH / "master_cells" / "coverage_census.json")
SCORES = C.root("SI_FINAL_SCORES_DIR") / "score_tier2_master.json"
SIDECHAIN_SCORES = C.root("SI_FINAL_SCORES_DIR") / "score_by_nucleus_tier2.json"
FOLD_MAP = C.BUILD_ROOT / "data" / "fold_assignments_labelset_identity90.json"
BACKBONE_NUCLEI = {"N", "H", "CA", "CB", "C", "HA"}
DEEPSHIFT = "deepshift"

# How each staged arm is named on the page. The roster itself is the census's -- adding a
# predictor is a line in benchmarks/config.py and a re-run of the census -- so an arm with
# no label here prints its staged name, visibly.
LABELS = {
    DEEPSHIFT: r"\textbf{DeepShift}",
    "ucbshift_full_leaky": "UCBShift, as distributed",
    "ucbshift_full_sameprot": "UCBShift, same-protein depositions blocked",
    "ucbshift_full_selfexcl": r"UCBShift, same-protein and $\geq$90\% identity blocked",
    "ucbshift_x": r"UCBShift-X\cite{UCBShift} (structure only)",
    "ucbshift2_sameprot": r"UCBShift 2.0\cite{UCBShift2}, same-protein depositions blocked",
    "ucbshift2_selfexcl": r"UCBShift 2.0, same-protein and $\geq$90\% identity blocked",
    "shiftx2_leaky": r"SHIFTX2\cite{shiftx2_2011}, as distributed",
    "shiftx2_sameprot": "SHIFTX2, same-protein depositions blocked",
    "shiftx2_selfexcl": r"SHIFTX2, same-protein and $\geq$90\% identity blocked",
    "shiftx2_noshifty": r"SHIFTX2, SHIFTY+\cite{shiftx2_2011} disabled",
    "sparta": r"SPARTA+\cite{spartaplus2010}",
    "napshift": r"NapShift\cite{napshift2022}",
    "legolas": r"LEGOLAS\cite{LEGOLAS}",
    "procs15": r"PROCS15\cite{procs15_2015}",
    "nmrgnn": r"NMRGNN\cite{nmrgnn2021}",
    "camshift": r"CAMSHIFT\cite{camshift2009}",
    "mlff": r"MLFF\cite{MLFF}",
    "plmcs": r"PLM-CS\cite{PLM-CS}",
}
AMIDE_ALIAS = {"HN": "H"}
# Running-text names, where the row label carries a citation or an exclusion level.
SHORT = {"ucbshift2_sameprot": "UCBShift 2.0", "shiftx2_sameprot": "SHIFTX2",
         "plmcs": "PLM-CS", "camshift": "CAMSHIFT"}


def scan_arm(directory: Path) -> dict:
    """Tasks a predictor answered and the nuclei it labels over the whole run."""
    produced = 0
    labelled: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except ValueError:
            continue
        if record.get("status") != "ok":
            continue
        produced += 1
        for cell in (record.get("predictions") or {}).values():
            if isinstance(cell, dict):
                labelled |= set(cell)
    return {"produced": produced, "nuclei": sorted(labelled & SHIFT_ATOM_NAMES)}


def scan_all(arms) -> dict:
    """Scan every staged arm."""
    out = {}
    for key in arms:
        directory = arm_directory(BENCH, key)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{key}: no per-task output at {directory}. A predictor the ledger "
                "cannot find is a row of zeros, so the roster fails closed instead.")
        print(f"  scanning {directory.name} ...", flush=True)
        out[key] = scan_arm(directory)
    return out


def oxford(names) -> str:
    """A, B and C."""
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def short_name(key: str) -> str:
    """A predictor's name as running text uses it: no citation, no exclusion level."""
    return SHORT.get(key) or re.sub(r"\\cite\{[^}]*\}", "", LABELS.get(key, key)).split(",")[0]


def marked_rows(scores: dict, sidechain: dict) -> dict:
    """The rows the accuracy tables mark, read from the two reports those tables print.

    The backbone tables mark an arm the backbone report scored below the master floor on
    a backbone nucleus; the side-chain table marks one of the arms it prints wherever the
    side-chain report scored it below the floor. The coverage census alone is not the
    answer: it counts arms no accuracy table prints, and it cannot see an arm that is
    marked because it never joins the master set.
    """
    backbone = [arm for arm in scores["methods"]
                if set(scores.get("arms_below_master_coverage", {}).get(arm, ()))
                & BACKBONE_NUCLEI]
    printed = [key for key, _ in SIDECHAIN_ARMS]
    side = [arm for arm in printed
            if any(node["arms"].get(arm, {}).get("below_master_coverage")
                   for node in sidechain["by_nucleus"].values() if not node["backbone"])]
    return {"backbone_tables": backbone, "side_chain_table": side}


def below_note(marked: dict) -> str:
    """What a marker in the accuracy tables means, and who carries one."""
    parts = []
    for table, keys in (("the backbone tables", marked["backbone_tables"]),
                        ("the side-chain table", marked["side_chain_table"])):
        if keys:
            below = [k for k in keys if k not in MASTER_MARKED_ARMS]
            by_design = [k for k in keys if k in MASTER_MARKED_ARMS]
            words = []
            if below:
                verb = "covers" if len(below) == 1 else "cover"
                own = "it returns" if len(below) == 1 else "they return"
                words.append(f"{oxford(short_name(k) for k in below)} {verb} less than "
                             f"90\\% of a nucleus {own}")
            if by_design:
                verb = "is" if len(by_design) == 1 else "are"
                words.append(f"{oxford(short_name(k) for k in by_design)} {verb} kept out "
                             "of the master set by design")
            parts.append(f"in {table} " + ", and ".join(words))
    if not parts:
        return ""
    sentence = "; ".join(parts)
    return (f" {sentence[0].upper()}{sentence[1:]}. There each is scored on the master "
            "cells it produced, and every such row is marked and prints that count.")


def scored_note(sidechain: dict, emitted: int) -> str:
    """How many nuclei the accuracy tables score, read from the per-nucleus report they
    print: DeepShift emits every nucleus of its vocabulary, but a nucleus with too few
    cells is reported below the floor and receives no accuracy result."""
    below = sidechain["nuclei_below_the_floor"]
    sentence = (f"The accuracy tables score {len(sidechain['by_nucleus'])} of the "
                f"{emitted} nuclei DeepShift emits")
    if not below:
        return sentence + "."
    most = max(counts["deepshift"] for counts in below.values())
    verb = "has" if len(below) == 1 else "have"
    return (f"{sentence}; {oxford(sorted(below))} {verb} too few cells (at most "
            f"{C.thousands(most)}) to score.")


def fold6_realizations() -> int:
    """Fold-6 structure realizations (label set x source), the count Table S2 prints."""
    fold = json.loads(FOLD_MAP.read_text())
    realized = set()
    for source in ("xtal", "nmr", "af"):
        with open(C.BUILD_ROOT / "data" / f"build_log_{source}.csv") as fh:
            realized |= {(r["label_set_id"], source) for r in csv.DictReader(fh)
                         if r["status"] == "success" and fold.get(r["label_set_id"]) == 6}
    return len(realized)


def master_cell_set(scores: dict) -> tuple[Path, int]:
    """The master cell set the backbone report scored, and the structure instances it spans
    over all nuclei. Its backbone part must be that report's own cohort."""
    path = BENCH / "master_cells" / Path(scores["master_cells"]).name
    cells = np.load(path)
    backbone = np.isin(cells["atom"], [f"{n.lower()}_shift" for n in BACKBONE_NUCLEI])
    own = (len(set(cells["sid"][backbone].tolist())), int(backbone.sum()))
    if own != (scores["n_common_structures"], scores["n_common_cells"]):
        raise SystemExit(f"REFUSING to emit -- {path} holds {own[1]} backbone cells over "
                         f"{own[0]} structures; the backbone report scored "
                         f"{scores['n_common_cells']} over {scores['n_common_structures']}")
    return path, len(set(cells["sid"].tolist()))


def scorable_structures(census_path: Path, census: dict) -> tuple[Path, int]:
    """The census's scorable cells, and how many structure instances hold at least one."""
    path = census_path.parent / Path(census["coverage_npz"]).name
    sid = np.load(path)["sid"]
    if len(sid) != census["n_universe_cells"]:
        raise SystemExit(f"REFUSING to emit -- {path} holds {len(sid)} cells; the census "
                         f"counts {census['n_universe_cells']}")
    return path, len(set(sid.tolist()))


def cohort_note(outside, census, scores, scored, marked, sidechain) -> str:
    """The row under the ledger: what each predictor that loses tasks lost them to, the
    cell set every accuracy table is computed on, how many nuclei those tables score,
    and which returned nuclei fall outside DeepShift's vocabulary (so `Nuclei returned'
    can exceed the emitted count honestly)."""
    extras = "".join(
        f" {LABELS.get(key, key)} also returns {oxford(names)}, which no accuracy table scores."
        for key, names in outside.items())
    below = below_note(marked)
    realized, tasks = fold6_realizations(), census["n_structure_instances"]
    return (
        r"\multicolumn{6}{@{}p{0.97\linewidth}@{}}{"
        "UCBShift and UCBShift 2.0 are separate predictors that predict separate values "
        "for backbone nuclei. "
        f"Of the {C.thousands(realized)} fold-6 structure realizations "
        r"(Supplementary Table~\ref{tab:census_folds}), "
        f"{C.thousands(realized - tasks)} keep no residue after exclusions; the master "
        f"cell set spans {C.thousands(scores['n_master_structures'])} of the remaining "
        f"{C.thousands(tasks)}. "
        f"{C.thousands(census['n_universe_cells'])} residue--nucleus cells of "
        f"{C.thousands(census['n_scorable_structures'])} structure instances are scorable. "
        f"The results tables report the master cell set: per nucleus, the cells every "
        f"predictor covering at least {100 * census['coverage_floor']:.0f}\\% of them "
        "produced, "
        f"{C.thousands(scores['n_common_cells'])} of them on the six backbone nuclei, over "
        f"{C.thousands(scores['n_common_structures'])} structure instances.{below} "
        f"{scored_note(sidechain, len(scored))}{extras}}} \\\\")


def ledger_table(scan, census, n_tasks, scores, scored, marked, sidechain):
    rows, outside = [], {}
    universe = census["n_universe_cells"]
    for key, record in scan.items():
        # HN is the amide-proton name several tools write (domain.SHIFT_ATOM_NAMES); it
        # is the scored H under another name, so it is one nucleus, not an extra one.
        names = {AMIDE_ALIAS.get(n, n) for n in record["nuclei"]}
        covered = census["arms"][key]["n_cells"]
        if names - set(scored):
            outside[key] = sorted(names - set(scored))
        rows.append(f"{LABELS.get(key, key)} & {C.thousands(n_tasks)} & "
                    f"{C.thousands(record['produced'])} & "
                    f"{C.thousands(n_tasks - record['produced'])} & {len(names)} & "
                    f"{100 * covered / universe:.1f} \\\\")
    body = ["\\begin{tabular}{lrrrrr}", "\\toprule",
            (r"Method & Tasks & Produced & Failed / missing & "
             r"Nuclei returned & Cells covered (\%) \\"),
            "\\midrule", *rows, "\\midrule",
            cohort_note(outside, census, scores, scored, marked, sidechain),
            "\\bottomrule", "\\end{tabular}"]
    note = (
        "% FACTS the caption needs:\n"
        f"%   {C.thousands(n_tasks)} fold-6 structure tasks were dispatched to every "
        "predictor.\n"
        "%   `Produced' counts tasks that returned a usable prediction file; the rest\n"
        "%   either crashed, declined the structure, or were never written.\n"
        "%   `Cells covered' is the share of the "
        f"{C.thousands(census['n_universe_cells'])} residue--nucleus cells\n"
        "%   DeepShift can score on this fold that the predictor produced a finite value\n"
        "%   for. A structure it lost to self-exclusion counts as uncovered, because the\n"
        "%   comparison cannot use it.\n"
        "%   The master cell set follows from those coverages: per nucleus, the cells\n"
        f"%   every predictor covering at least {census['coverage_floor']:.0%} of that "
        "nucleus produced. A predictor\n"
        "%   below the floor does not shrink the set; it is scored on the master cells it\n"
        "%   produced, and every table row of that predictor prints the count and is marked.\n"
        f"%   The backbone tables therefore score "
        f"{C.thousands(scores['n_common_cells'])} cells over "
        f"{C.thousands(scores['n_common_structures'])} structure\n"
        "%   instances, the same cells for every predictor.\n"
        "%   `Nuclei returned' is what the tool actually returns over the whole run, not\n"
        "%   what it is scored on, and not what its first answer happened to contain:\n"
        "%   it is the number of distinct standard amino-acid atom names the predictor\n"
        "%   labels anywhere. UCBShift 1.0 returns only the six backbone nuclei, which is\n"
        "%   why the six-nucleus tables and the side-chain table have different lists.\n")
    return C.table(note, "tab:coverage_ledger", "\n".join(body) + "\n", resize=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    census = json.loads(CENSUS.read_text())
    scores = json.loads(SCORES.read_text())
    with open(TASKS) as fh:
        n_tasks = sum(1 for _ in csv.DictReader(fh, delimiter="\t"))
    arms = [DEEPSHIFT, *(a for a in census["arms_scanned"] if a != DEEPSHIFT)]
    scan = scan_all([a for a in arms if a != DEEPSHIFT])
    scored = sorted(atom_from_shift_column(c) for c in census["universe_by_nucleus"])
    # DeepShift is not a per-task external run: its predictions come from the cache, so
    # every task is covered by construction and it returns every scored nucleus.
    scan = {DEEPSHIFT: {"produced": n_tasks, "nuclei": scored}, **scan}

    scores["n_master_structures"] = master_cell_set(scores)[1]
    census["n_scorable_structures"] = scorable_structures(CENSUS, census)[1]
    sidechain = json.loads(SIDECHAIN_SCORES.read_text())
    marked = marked_rows(scores, sidechain)
    tex = ledger_table(scan, census, n_tasks, scores, scored, marked, sidechain)
    C.emit("si_coverage_ledger", tex, script="si_competitor_ledger.py")

if __name__ == "__main__":
    main()
