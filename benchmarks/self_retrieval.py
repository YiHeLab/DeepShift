"""UCBShift-Y copying the test protein's own deposited shifts, counted per test chain.

One function for the two places that count it: ``si_build/emit_text_numbers.py`` (the
Supplementary Information's "27,305 of 48,617") and ``figures/fig_leakage_audit.py``
(Extended Data Fig. 2b, which draws the per-chain fractions of the distributed run and of
UCBShift's own-record-blocked rerun), so the printed totals and the figure's dots cannot
come from two implementations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SELF_NUCLEI = ("H", "HA", "C", "CA", "CB", "N")
SELF_EXACT_PPM = 0.005
#: Shifts are stored to a few decimals, so a prediction can sit exactly 0.005 ppm from its
#: target (5.39 against 5.395); binary subtraction puts such a tie on either side of the
#: threshold. "Within 0.005 ppm" counts the tie, with a tolerance far below any stored digit.
#: The tolerance, not exact decimals: some stored values carry float noise in the file itself
#: (UCBShift's target 3.9450000000000003 against Y's 3.94, 1GWYA residue 153 Ha), a tie read
#: literally as 0.0050000000000003. Exact decimals would count 27,304 of 48,617 there, not 27,305.
SELF_TIE_TOLERANCE_PPM = 1e-9


def within_copy_threshold(prediction, target):
    """|prediction - target| <= 0.005 ppm, ties at the stored decimals included."""
    return np.abs(prediction - target) <= SELF_EXACT_PPM + SELF_TIE_TOLERANCE_PPM


def ucbshift_y_chains(preds: Path, targets: Path, exclusions: dict) -> dict[str, tuple[int, int, int]]:
    """``{chain: (copied, scored, removed)}`` for UCBShift-Y in test mode.

    Scored cells have both a Y prediction and the target (residue names agreeing; UCBShift's
    evaluate.py placeholders H < 2, CB > 80, N < 60 blanked, and the cells UCBShift's paper
    excludes, ``exclusions`` as in ``benchmarks/reference/ucb200_paper_exclusions.json``);
    copied cells are within 0.005 ppm of the target; removed counts the target cells the
    paper's exclusions blank. A prediction file without a target file is skipped.
    """
    cprime_chains = set(exclusions["cprime_chains"]["chains"])
    excluded_cells = {(str(c), int(r), str(a)) for c, r, a in exclusions["cells"]["cells"]}
    pred_files = sorted(Path(preds).glob("*.csv"))
    if not pred_files:
        raise FileNotFoundError(preds)
    chains = {}
    for pf in pred_files:
        tf = Path(targets) / pf.name
        if not tf.is_file():
            continue
        m = pd.merge(pd.read_csv(pf), pd.read_csv(tf).rename(columns={"RES_NUM": "RESNUM"}),
                     on="RESNUM", suffixes=("_p", "_t"))
        m.loc[m["RESNAME_t"] != m["RESNAME_p"], list(SELF_NUCLEI)] = np.nan
        m.loc[m["H"] < 2, "H"] = np.nan
        m.loc[m["CB"] > 80, "CB"] = np.nan
        m.loc[m["N"] < 60, "N"] = np.nan
        before = int(m[list(SELF_NUCLEI)].notna().sum().sum())
        if pf.stem in cprime_chains:
            m["C"] = np.nan
        for chain, resnum, atom in excluded_cells:
            if chain == pf.stem:
                m.loc[m["RESNUM"] == resnum, atom] = np.nan
        removed = before - int(m[list(SELF_NUCLEI)].notna().sum().sum())
        copied = scored = 0
        for a in SELF_NUCLEI:
            t = pd.to_numeric(m[a], errors="coerce").to_numpy(float)
            y = pd.to_numeric(m[a + "_Y"], errors="coerce").to_numpy(float)
            ok = ~np.isnan(t) & ~np.isnan(y)
            copied += int(within_copy_threshold(y[ok], t[ok]).sum())
            scored += int(ok.sum())
        chains[pf.stem] = (copied, scored, removed)
    return chains
