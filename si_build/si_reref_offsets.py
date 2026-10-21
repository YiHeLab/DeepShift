#!/usr/bin/env python3
"""Rereferencing offsets: magnitude, sign, coverage, and which structure fitted them.

Methods promises "Median, 95th%ile, and maximum rereferencing offsets are provided in
Table SX". This is that table, computed from the build's own offsets file
(``data/reref_offsets.json`` under the run root), which is the file the paper's labels
carry.

Three decisions, each of which sets a number in the paper:

  * Denominator. The offsets file covers every selected label set; the fold map covers
    the ones realized into the build; the body quotes the released subset. All three
    appear in the coverage table, since the same quantity otherwise reads against two
    different denominators in two places.

  * Sign. Both are given. |offset| is what Methods promised; the signed distribution is
    what carries the physics -- carbons skew positive and nitrogen negative, which a
    magnitude-only table hides.

  * Four nuclei, not six. Only CA, CB, CO and N carry offsets. Protons are left
    uncorrected (RefDB computes 1H offsets and never applies them), so a six-row table
    would be wrong.

Usage:
    python si_reref_offsets.py
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import si_common as C

OFFSETS = C.DATA / "reref_offsets.json"
FOLDMAP = C.DATA / "fold_assignments_labelset_identity90.json"
# Why each label set has or lacks an offset, written by scripts/reref_diagnostics.py;
# SI_REREF_DIAGNOSTICS names another file it wrote (its --out).
DIAGNOSTICS = (Path(os.environ["SI_REREF_DIAGNOSTICS"]).expanduser().resolve()
               if os.environ.get("SI_REREF_DIAGNOSTICS") else C.DATA / "reref_diagnostics.json")
OUTCOMES = [("offset", "with a non-zero offset"),
            ("no_structure", "no structure to compare with"),
            ("no_correctable_shift", r"no C$\alpha$, C$\beta$, C$'$ or N shift to correct"),
            ("rounds_to_zero", "offset rounds to zero"),
            ("too_few_residues", "too few residues, or no usable alignment"),
            ("shiftx_failed", "SHIFTX predicted nothing")]
NUC_ORDER = ["CA", "CB", "CO", "N"]
NUC_LABEL = {"CA": r"C$\alpha$", "CB": r"C$\beta$", "CO": r"C$'$", "N": "N"}


def load():
    offsets = json.loads(OFFSETS.read_text())
    realized = set(json.loads(FOLDMAP.read_text()))
    by_nuc = defaultdict(list)
    struct_with_offset = Counter()
    n_any = 0
    for rec in offsets.values():
        offs = rec.get("offsets") or {}
        if offs:
            n_any += 1
            struct_with_offset[rec.get("struct_type") or "none"] += 1
        for nuc, val in offs.items():
            by_nuc[nuc].append(float(val))
    return {"n_total": len(offsets), "n_realized": len(realized),
             "by_nuc": by_nuc,
             "struct_with_offset": struct_with_offset, "n_any": n_any,
             "diagnostics": json.loads(DIAGNOSTICS.read_text())}


def merged_offset_table(d):
    """Magnitude and sign in one float.

    A reader needs the sign (carbons skew positive, nitrogen negative -- that asymmetry is
    the empirical case for one 13C constant) and the magnitude (how big a correction the
    worst depositions need). Both fit in one row.
    """
    rows = []
    for nuc in NUC_ORDER:
        v = np.array(d["by_nuc"][nuc])
        a = np.abs(v)
        sp = np.percentile(v, [5, 50, 95])
        ap = np.percentile(a, [50, 95])
        rows.append(f"{NUC_LABEL[nuc]} & {C.thousands(v.size)} & {v.mean():+.3f} & "
                    + " & ".join(f"{x:+.2f}" for x in sp) + " & "
                    + " & ".join(f"{x:.2f}" for x in ap)
                    + f" & {a.max():.2f} \\\\")
    body = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
            r"Nucleus & Label sets & mean & \multicolumn{3}{c}{signed offset (ppm)}"
            r" & \multicolumn{3}{c}{$|$offset$|$ (ppm)} \\",
            r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
            r" & & & 5\% & 50\% & 95\% & 50\% & 95\% & max \\",
            "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]
    tex = ("\\begin{table}[H]\\centering\\small\n"
           "\\caption{Magnitude and sign of rereferencing offsets applied to deposited "
           "shifts}\n"
           "\\label{tab:reref_offsets}\n" + "\n".join(body)
           + "\n\\end{table}\n")
    return tex


def outcome_rows(counts: dict, total: int) -> list:
    """One row per outcome a label set can have, in a fixed order; they sum to the total."""
    if sum(counts.values()) != total or set(counts) - {k for k, _ in OUTCOMES}:
        raise SystemExit(f"{DIAGNOSTICS}: outcomes {counts} do not account for {total}")
    return [(rf"\quad {label}", C.thousands(counts[key]), "")
            for key, label in OUTCOMES if counts.get(key)]


def coverage_table(d):
    stw = d["struct_with_offset"]
    diag = d["diagnostics"]
    if diag["reproduces_stored_offsets"] != d["n_total"]:
        raise SystemExit(f"{DIAGNOSTICS} does not cover the offsets file")
    rows = [
        ("Label sets in the offsets file", C.thousands(d["n_total"]), "all selected"),
        *outcome_rows(diag["all"]["outcome"], d["n_total"]),
        (r"\midrule Label sets realized into the build", C.thousands(d["n_realized"]),
         "fold map"),
        *outcome_rows(diag["realized"]["outcome"], d["n_realized"]),
        (r"\midrule \emph{Structure the offset was fitted on}", "", ""),
        (r"\quad AlphaFold\cite{AlphaFold}", C.thousands(stw.get("af", 0)),
         f"{100 * stw.get('af', 0) / max(d['n_any'], 1):.1f}\\%"),
        (r"\quad NMR", C.thousands(stw.get("nmr", 0)),
         f"{100 * stw.get('nmr', 0) / max(d['n_any'], 1):.1f}\\%"),
        (r"\quad X-ray crystal", C.thousands(stw.get("xtal", 0)),
         f"{100 * stw.get('xtal', 0) / max(d['n_any'], 1):.1f}\\%"),
    ]
    body = ["\\begin{tabular}{lrl}", "\\toprule", r"& Label sets & \\", "\\midrule"]
    for a, b, c in rows:
        body.append(f"{a} & {b} & {c} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    note = (
        "% FACTS the caption needs:\n"
        "%   A nucleus is corrected only where at least three residues carry both an\n"
        "%   in-window SHIFTX prediction and an observed shift, so a label set with too\n"
        "%   little overlap keeps its deposited values untouched.\n"
        "%   No shift to correct: no Calpha, Cbeta, C' or N shift on an aligned residue.\n"
        "%   Too few residues: such a shift exists, but fewer than five residues align to\n"
        "%   any structure, the alignment identity is below 0.8, or fewer than three per\n"
        "%   nucleus fall in the window.\n"
        "%   Rounds to zero: the offset, or the joint Calpha/Cbeta mean, is under\n"
        "%   0.025 ppm and rounds to zero on the 0.05 ppm grid.\n"
        "% The structure selector takes whichever candidate structure has the best\n"
        "%   SHIFTX agreement, with no source priority, so the realized mix is\n"
        "%   AlphaFold-first rather than crystal-first.\n")
    return C.table(note, "tab:reref_coverage", "\n".join(body) + "\n",
                   caption="Coverage of the rereferencing procedure")


def main():
    d = load()
    coverage = coverage_table(d)
    C.emit("si_reref_offsets", merged_offset_table(d), script="si_reref_offsets.py",
           placeholder_caption=False)
    C.emit("si_reref_coverage", coverage, script="si_reref_offsets.py",
           placeholder_caption=False)


if __name__ == "__main__":
    main()
