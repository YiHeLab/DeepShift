#!/usr/bin/env python3
"""The 64 predicted nuclei, with how each one is scored.

A nucleus is only meaningful together with the rule used to score it, so the table also
carries the two things that rule depends on:

  Equivalence group. Magnetically equivalent protons (a methyl's three) are one
  observation, not three; the loss weights them 1/N so a methyl cannot outvote a
  backbone amide.

  Prochiral partner. Where two positions are stereochemically ambiguous in the
  deposition, the pair is scored in whichever assignment minimises the loss, so an arm is
  never charged for a labelling convention.

The partner table is incomplete: prochiral groups are scored swap-invariantly, and for
the columns listed below with no partner that does not hold.

Usage:
    python si_nuclei.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# si_build sits inside the repo, so the package is one directory up -- no install
# needed and no workstation path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import si_common as C

from deepshift.domain import STANDARD_RESIDUES
from deepshift.loss import build_loss_tensors

BACKBONE = {"c_shift", "ca_shift", "cb_shift", "n_shift", "h_shift", "ha_shift"}
# Ring positions that exchange under a flip the partner table does not encode. Checked
# against the partner tensor below: none of these columns has a partner on these residues.
KNOWN_GAPS = {
    "PHE": "ring CD1/CD2, CE1/CE2, HD1/HD2, HE1/HE2 exchange under a ring flip",
    "TYR": "ring CD1/CD2, CE1/CE2, HD1/HD2, HE1/HE2 exchange under a ring flip",
}
RING_FLIP = ("cd1_shift", "cd2_shift", "ce1_shift", "ce2_shift",
             "hd1_shift", "hd2_shift", "he1_shift", "he2_shift")


def main():
    cfg = json.loads((C.CACHE / "fold_1" / "config.json").read_text())
    columns = cfg["shift_cols"]
    group_weight, partner = build_loss_tensors(tuple(columns), tuple(STANDARD_RESIDUES))
    gw, pa = group_weight.numpy(), partner.numpy()

    observed = np.zeros(len(columns), dtype=np.int64)
    per_res = defaultdict(set)
    for i in range(7):
        sd = C.CACHE / f"fold_{i}" / "structural"
        if not sd.is_dir():
            continue
        mask = np.asarray(np.load(sd / "shift_mask.npy", mmap_mode="r")).astype(bool)
        res = np.asarray(np.load(sd / "residue_idx.npy", mmap_mode="r"))
        if res.ndim > 1:
            res = res[:, res.shape[1] // 2]
        observed += mask.sum(axis=0)
        for j in range(len(columns)):
            codes = np.unique(res[mask[:, j]])
            for c in codes:
                if 0 <= c < len(STANDARD_RESIDUES):
                    per_res[j].add(STANDARD_RESIDUES[int(c)])

    for res in KNOWN_GAPS:
        a = STANDARD_RESIDUES.index(res)
        paired = [c for c in RING_FLIP if c in columns and pa[a, columns.index(c)] >= 0]
        if paired:
            raise SystemExit(f"KNOWN_GAPS is stale: {res} {paired} have a partner")

    rows, kinds, gaps = [], [], []
    for j, col in enumerate(columns):
        residues = sorted(per_res[j])
        grouped = sorted({STANDARD_RESIDUES[a] for a in range(len(STANDARD_RESIDUES))
                          if gw[a, j] < 1.0})
        partner_cols = sorted({columns[pa[a, j]] for a in range(len(STANDARD_RESIDUES))
                               if pa[a, j] >= 0})
        kind = "backbone" if col in BACKBONE else "side chain"
        rows.append(
            f"{C.tex_nucleus(col)} & {kind} & {C.thousands(int(observed[j]))} & "
            f"{len(residues)} & "
            f"{', '.join(C.tex_nucleus(p) for p in partner_cols) if partner_cols else '---'} & "
            f"{len(grouped) or '---'} \\\\")
        kinds.append(kind)
        if not partner_cols and any(r in KNOWN_GAPS for r in residues):
            gaps.append(col)

    n_back = kinds.count("backbone")
    n_side = len(kinds) - n_back
    total = int(observed.sum())
    head = (r"Nucleus & Class & Observed cells & Residues & Prochiral partner"
            r" & Equiv.\ groups \\")
    foot = "\n".join(["\\midrule",
                      r"\multicolumn{6}{r}{\emph{continued on the next page}}\\",
                      "\\endfoot", "\\bottomrule", "\\endlastfoot"])

    note = (
        "% FACTS the caption needs:\n"
        f"%   All {len(kinds)} predicted nuclei: {n_back} backbone and {n_side} side chain.\n"
        f"%   `Observed cells' counts deposited values across all seven folds "
        f"({C.thousands(total)} in total);\n"
        "%   `Residues' is how many of the twenty standard amino acids carry that atom in\n"
        "%   the corpus. Both are properties of the data, not of the model -- the model\n"
        "%   emits all 64 for every residue and the mask decides what is scored.\n"
        "%   `Prochiral partner' is the column this one is swapped against when the\n"
        "%   deposition does not resolve the stereochemistry; the pair is scored in\n"
        "%   whichever assignment minimises the loss, identically for every arm in every\n"
        "%   comparison in this paper.\n"
        "%   `Equiv. groups' counts residue types where this column is part of a\n"
        "%   magnetically equivalent set (a methyl's three protons) and is therefore\n"
        "%   down-weighted to 1/N so it counts once, not N times.\n"
        "% The gap the caption states:\n"
        "%   the partner table is incomplete, so `scored swap-invariantly' is true of the\n"
        "%   pairs listed and not of these:\n"
        + "".join(f"%     {k} -- {v}\n" for k, v in KNOWN_GAPS.items()) +
        "% Two columns should not be read as results, and the reason is label\n"
        "%   quality rather than model error:\n"
        "%     Arg N-epsilon: roughly one sixth of deposited values are spectrometer\n"
        "%       aliased by one 15N sweep width, a whole deposition at a time, and the\n"
        "%       observed-value interval is what removes them.\n"
        "%     Pro N: no competitor predicts it at all, so the common-cell rule removes\n"
        "%       every proline nitrogen from every multi-arm comparison; and a sixth of\n"
        "%       its deposited values fall outside any plausible range.\n"
        "% SI.tex must \\usepackage{longtable} for this to compile.\n")

    tex = C.table(note, "tab:nuclei", "\n".join(rows), longtable="llrrlr",
                  head=head, foot=foot, size="\\footnotesize")
    C.emit("si_nuclei", tex, script="si_nuclei.py")
    print(f"    {len(kinds)} nuclei ({n_back} backbone, {n_side} side chain), "
          f"{C.thousands(total)} observed cells")
    print(f"    columns with no prochiral partner on a residue known to need one: "
          f"{len(gaps)}")


if __name__ == "__main__":
    main()
