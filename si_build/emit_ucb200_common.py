"""Emit tab:ucb200_common: backbone RMSE on UCBShift's 200-protein crystal test set,
DeepShift against UCBShift, UCBShift-X and SHIFTX2, on the cells common to all four.

Every retrieval arm is blocked from the target's own protein: UCBShift through its
same-protein patch, SHIFTX2 through a SHIFTY+ run that excludes, per query, the library
entries of the query's own PDB entry and UniProt accession and every entry
``benchmarks/reference/ucb200_competitor_same_protein.json`` lists (same BMRB, a shared
UniProt accession or UniRef100 cluster, or >=95 % identity over >=90 % of the shorter
sequence). The cells UCBShift's paper
removes from its own test set are removed here too (its ESI Fig. S1 and Tables S1-S2).

The asterisk marks a lead over the runner-up whose 95 % interval excludes zero, from a
paired bootstrap over chains (2,000 resamples, seed 20260806, the draw the main-text
UCBShift-200 block uses).

Reads four cell stores (npz: chain, resnum, atom, pred, target):
  $SI_UCB200_ARTIFACTS_DIR/cells_deepshift_final_ensemble.npz   this run's ensemble
  $SI_UCB200_ARTIFACTS_DIR/cells_shiftx2_sameprot.npz           SHIFTX2, same-protein
  $SI_UCB200_COMPETITOR_CELLS/cells_ucbshift_sameprot.npz       UCBShift, same-protein
  $SI_UCB200_COMPETITOR_CELLS/cells_X_H.npz                     UCBShift-X
"""
import json
from pathlib import Path

import numpy as np
import si_common

from deepshift.statistics import bonferroni, tail_count

ARTIFACTS_DIR = si_common.root("SI_UCB200_ARTIFACTS_DIR")
COMPETITOR_CELLS = si_common.root("SI_UCB200_COMPETITOR_CELLS")
ARMS = {
    "DeepShift": ARTIFACTS_DIR / "cells_deepshift_final_ensemble.npz",
    "UCBShift": COMPETITOR_CELLS / "cells_ucbshift_sameprot.npz",
    "UCBShift-X": COMPETITOR_CELLS / "cells_X_H.npz",
    "SHIFTX2": ARTIFACTS_DIR / "cells_shiftx2_sameprot.npz",
}
ATOMS = ["CA", "CB", "C", "HA", "H", "N"]
BOOT_N, SEED = 2000, 20260806

# Cells UCBShift's paper removes from its own test set (Li et al., Chem. Sci. 2020, ESI):
# every C' of three anticorrelated chains (Fig. S1), the one Table S1 row this funnel
# keeps, and the eight cysteine Cbeta of Table S2. One record, read by every producer
# that scores this test set.
EXCLUSIONS = (Path(__file__).resolve().parent.parent / "benchmarks" / "reference"
              / "ucb200_paper_exclusions.json")
_EXCLUDED = json.loads(EXCLUSIONS.read_text())
ANTICORRELATED_CPRIME = set(_EXCLUDED["cprime_chains"]["chains"])
ESI_CELLS = {(str(c), int(r), str(a)) for c, r, a in _EXCLUDED["cells"]["cells"]}


def paper_excluded(chain: str, resnum: int, atom: str) -> bool:
    """A cell UCBShift's paper removes from its test set."""
    return (atom == "C" and chain in ANTICORRELATED_CPRIME) or (chain, resnum, atom) in ESI_CELLS

CAPTION = (
    r"Backbone RMSE (ppm) on UCBShift's 200-protein crystal test set against UCBShift's "
    r"deposited labels, on the {cells} cells common to all four methods ({chains} chains; "
    r"per-method preprocessing failures shrink the intersection). Some cells, as well as the "
    r"C$'$ entries of 3 anticorrelated proteins, are excluded from the test set as in the "
    r"UCBShift paper\cite{{UCBShift}}. Homology modules are patched so that no predictor can "
    r"retrieve the target protein's own BMRB entry, any entry sharing its UniProt accession, "
    r"any reference keyed to its own PDB entry, or any reference at $\geq$95\% sequence "
    r"identity over $\geq$90\% of the shorter sequence, and any reference in its UniRef100 "
    r"cluster. Bold marks the lowest RMSE for each "
    r"nucleus; an asterisk marks a lead over the runner-up whose 95\,\% paired interval "
    r"excludes zero, and a dagger such a lead of DeepShift over UCBShift-X, from 2,000 "
    r"bootstrap resamples of chains, seed 20260806.")


def load(path):
    z = np.load(path)
    return {(str(c), int(r), str(a)): (float(p), float(t))
            for c, r, a, p, t in zip(z["chain"], z["resnum"], z["atom"], z["pred"], z["target"])
            if not paper_excluded(str(c), int(r), str(a))}


#: The pair main.tex quotes beside the table: DeepShift against UCBShift-X on every nucleus.
VERSUS_X = ("DeepShift", "UCBShift-X")


def block(arms):
    """RMSE per arm and nucleus on the common cells, the winner-vs-runner-up interval,
    and the DeepShift-vs-UCBShift-X interval on the same draws."""
    common = set.intersection(*(set(cells) for cells in arms.values()))
    chains = sorted({k[0] for k in common})
    index = {c: i for i, c in enumerate(chains)}
    draws = np.random.default_rng(SEED).integers(0, len(chains), size=(BOOT_N, len(chains)))
    rmse, significance, versus_x = {a: {} for a in arms}, {}, {}
    for atom in ATOMS:
        keys = sorted(k for k in common if k[2] == atom)
        chain = np.array([index[k[0]] for k in keys])
        err = {a: np.array([arms[a][k][0] - arms[a][k][1] for k in keys]) for a in arms}
        for a in arms:
            rmse[a][atom] = float(np.sqrt(np.mean(err[a] ** 2)))
        winner, runner = sorted(arms, key=lambda a: rmse[a][atom])[:2]
        count = np.bincount(chain, minlength=len(chains)).astype(float)[draws].sum(1)
        boot = {a: np.sqrt(np.bincount(chain, weights=err[a] ** 2, minlength=len(chains))
                           [draws].sum(1) / count) for a in {winner, runner, *VERSUS_X}}
        lo, hi = np.percentile(boot[runner] - boot[winner], [2.5, 97.5])
        significance[atom] = {"winner": winner, "runner": runner,
                              "ci": [round(float(lo), 4), round(float(hi), 4)],
                              "sig": bool(lo > 0)}
        ours, theirs = VERSUS_X
        lo, hi = np.percentile(boot[ours] - boot[theirs], [2.5, 97.5])
        diff = rmse[ours][atom] - rmse[theirs][atom]
        # main.tex's "significantly lower RMSE than UCBShift-X on four of six": Bonferroni
        # over the six nuclei.
        corrected = bonferroni(tail_count(boot[ours] - boot[theirs], diff), BOOT_N,
                               len(ATOMS))
        versus_x[atom] = {"diff": round(diff, 4),
                          "ci": [round(float(lo), 4), round(float(hi), 4)],
                          "deepshift_better": bool(hi < 0),
                          "deepshift_better_bonferroni": bool(diff < 0
                                                              and corrected["significant"])}
    return {"n_cells": len(common), "n_chains": len(chains), "rmse": rmse,
            "significance": significance,
            "deepshift_vs_ucbshift_x": {"by_nucleus": versus_x}}


def table(b):
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{" + CAPTION.format(cells=f"{b['n_cells']:,}", chains=b["n_chains"]) + "}",
             r"\label{tab:ucb200_common}",
             r"\resizebox{\textwidth}{!}{\begin{tabular}{lcccccc}", r"\toprule",
             r"Method & C$\alpha$ & C$\beta$ & C$'$ & H$\alpha$ & H & N \\", r"\midrule"]
    versus_x = b["deepshift_vs_ucbshift_x"]["by_nucleus"]
    for arm in b["rmse"]:
        cells = []
        for atom in ATOMS:
            value, s = f"{b['rmse'][arm][atom]:.4f}", b["significance"][atom]
            if s["winner"] == arm:
                value = rf"\textbf{{{value}}}" + ("*" if s["sig"] else "")
            # main.tex quotes DeepShift against UCBShift-X beside the table; the dagger is
            # that comparison, on the same draws, so the count it quotes can be read here.
            if arm == VERSUS_X[0] and versus_x[atom]["deepshift_better"]:
                value += r"$^{\dagger}$"
            cells.append(value)
        name = rf"\textbf{{{arm}}}" if arm == "DeepShift" else arm
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def main():
    b = block({name: load(path) for name, path in ARMS.items()})
    # The authors' caption types the cell and chain counts; they must be this run's.
    caption = (si_common.CAPTIONS / "si_ucb200_common.tex").read_text()
    for count in (f"{b['n_cells']:,}", f"{b['n_chains']} chains"):
        if count not in caption:
            raise SystemExit(f"REFUSING to emit -- the caption does not print {count}")
    si_common.emit("si_ucb200_common", table(b), script="emit_ucb200_common.py",
                   placeholder_caption=False)
    print(f"wrote si_ucb200_common: {b['n_cells']:,} cells, {b['n_chains']} chains")
    versus_x = b["deepshift_vs_ucbshift_x"]["by_nucleus"].values()
    print(f"DeepShift below UCBShift-X after Bonferroni over the {len(ATOMS)} nuclei: "
          f"{sum(v['deepshift_better_bonferroni'] for v in versus_x)} of {len(ATOMS)}")


if __name__ == "__main__":
    main()
