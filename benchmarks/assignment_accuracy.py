"""Resonance-assignment accuracy of the fold-6 predictors (Figure 5a; Methods, Assignment accuracy).

Reads a paired-cell dump of ``benchmarks.dump_paired_cells`` (tier 2, the fenced master cells;
no predictor is re-run). Each structure instance is one assignment problem. Its candidate
residues are those with at least three of the six backbone shifts observed; an instance with
fewer than two candidates is not scored. The cost of assigning observed residue i to position
j is the mean, over the nuclei observed at i and predicted at j, of ((obs - pred) / sigma)^2,
or ``PENALTY`` when they share none; each instance is solved as a one-to-one minimum-cost
matching (Hungarian algorithm) and scored as the fraction of residues assigned to their own
position. Two feature sets:

  residue      CA CB C HA H N of residue i
  spin system  H N CA CB C of residue i and CA CB C of residue i-1

sigma is one value per nucleus shared by every predictor: the mean RMSE, on the dump's cells,
of the predictors named by ``--sigma-arms`` (default: all in the dump). The rows with
``sigma == own`` repeat the analysis with each predictor's own RMSE. Random coil (each residue
at its type's random-coil value, on the observed cells) and the observed shifts themselves are
the floor and the ceiling. Intervals: 2,000 bootstrap resamples of label sets, the unit the
paper resamples; differences from DeepShift on the same resamples.

Random coil predicts every residue of a type alike, so its matchings have exact ties, broken by
the last bits of sigma; sigma is a floating-point mean over the dump's arms in their order. The
published outputs used the ``--tools`` order below: another order moves random coil by up to 0.5
points in the crystal subset and a predictor by one or two residues of 124,328.

    python -m benchmarks.dump_paired_cells --tier 2 --master-cells $MASTER \\
        --tools ucbshift_x ucbshift_full_sameprot shiftx2_sameprot sparta deepshift --out <dump.npz>
    python -m benchmarks.assignment_accuracy --npz <dump.npz> --out <dir>
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from deepshift.data.norm import RC_BACKBONE
from deepshift.domain import STANDARD_RESIDUES

NUC = ["CA", "CB", "C", "HA", "H", "N"]
SPIN_FEATS = [("H", 0), ("N", 0), ("CA", 0), ("CB", 0), ("C", 0), ("CA", 1), ("CB", 1), ("C", 1)]
MIN_NUC = 3
PENALTY = 4.0          # mean squared z when an (i, j) pair shares no feature (two sigma)
N_BOOT, SEED = 2000, 20260725
SRC = {"xtal": "crystal", "af": "AlphaFold", "nmr": "NMR"}
REF = "deepshift"


def tables(z, arms):
    """Per instance: observed and predicted [n_res, 6] arrays (NaN = no cell), previous-residue index."""
    d = pd.DataFrame({k: z[k] for k in ("sid", "resnum", "atom", "residue", "label_set", "source")})
    keep = d.resnum.str.fullmatch(r"-?\d+").to_numpy()
    n_drop = int((~keep).sum())
    d = d[keep].assign(resnum=lambda x: x.resnum.astype(int))
    cols = {"obs": z["true_ppm"][keep]}
    for a in arms:
        cols[a] = z[f"pred_{a}"][keep]
    aidx = d.atom.map({n: i for i, n in enumerate(NUC)}).to_numpy()
    out = []
    order = np.argsort(d.sid.to_numpy(), kind="stable")
    sid = d.sid.to_numpy()[order]
    cuts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1], True])
    for s, e in itertools.pairwise(cuts):
        ix = order[s:e]
        rn = d.resnum.to_numpy()[ix]
        u, inv = np.unique(rn, return_inverse=True)
        res = np.full(len(u), -1)
        res[inv] = d.residue.to_numpy()[ix]
        T = {}
        for k, v in cols.items():
            m = np.full((len(u), 6), np.nan)
            m[inv, aidx[ix]] = v[ix]
            T[k] = m
        pos = {r: i for i, r in enumerate(u)}
        prev = np.array([pos.get(r - 1, -1) for r in u])
        row = d.iloc[ix[0]]
        out.append({"sid": row.sid, "label_set": row.label_set, "source": row.source, "res": res, "prev": prev,
                    "T": T})
    return out, n_drop


def random_coil(res):
    m = np.full((len(res), 6), np.nan)
    for i, r in enumerate(res):
        rc = RC_BACKBONE.get(STANDARD_RESIDUES[r], {})
        for j, n in enumerate(NUC):
            if n in rc:
                m[i, j] = rc[n]
    return m


def random_coil_rmse(z):
    """RMSE of random coil against the observed shifts on the cells it covers (own-sigma rows)."""
    out = []
    for n in NUC:
        m = z["atom"] == n
        rc = np.array([RC_BACKBONE.get(STANDARD_RESIDUES[r], {}).get(n, np.nan) for r in z["residue"][m]])
        ok = np.isfinite(rc)
        out.append(np.sqrt(np.mean((rc[ok] - z["true_ppm"][m][ok]) ** 2)))
    return np.array(out)


def features(M, prev, spin):
    if not spin:
        return M
    cols = []
    for n, lag in SPIN_FEATS:
        c = M[:, NUC.index(n)]
        if lag:
            c = np.where(prev >= 0, c[np.maximum(prev, 0)], np.nan)
        cols.append(c)
    return np.stack(cols, 1)


def cost(O, P, sig):
    """Mean over shared features of ((obs_i - pred_j)/sigma)^2; PENALTY when none are shared."""
    mo, mp = np.isfinite(O), np.isfinite(P)
    with warnings.catch_warnings():                 # a nucleus observed at no residue has no mean;
        warnings.simplefilter("ignore", RuntimeWarning)   # its features are masked out below
        mu = np.nanmean(O, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    o = np.where(mo, (O - mu) / sig, 0.0)
    p = np.where(mp, (P - mu) / sig, 0.0)
    K = mo.astype(float) @ mp.astype(float).T
    C = (o ** 2 * mo) @ mp.T + mo @ (p ** 2 * mp).T - 2 * o @ p.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(K > 0, np.maximum(C, 0) / K, PENALTY)


def correct(O, P, sig):
    """Residues the minimum-cost one-to-one matching assigns to their own position."""
    r, c = linear_sum_assignment(cost(O, P, sig))
    return int((r == c).sum())


def run(inst, arms, sig, spin):
    """sig: arm -> per-nucleus sigma. Returns correct[n_inst, n_arms] and candidates[n_inst]."""
    cor = np.zeros((len(inst), len(arms)), int)
    nres = np.zeros(len(inst), int)
    for k, it in enumerate(inst):
        cand = np.flatnonzero(np.isfinite(it["T"]["obs"]).sum(1) >= MIN_NUC)
        nres[k] = len(cand)
        if len(cand) < 2:
            continue
        # A candidate's i-1 is its structural predecessor, which may lie outside the candidates.
        O = features(it["T"]["obs"], it["prev"], spin)[cand]
        for a_i, a in enumerate(arms):
            if a == "oracle":
                M = it["T"]["obs"]
            elif a == "randomcoil":
                # Random coil on the observed cells only, so it is compared on the same cells.
                M = np.where(np.isfinite(it["T"]["obs"]), random_coil(it["res"]), np.nan)
            else:
                M = it["T"][a]
            s = sig[a]
            if spin:
                s = np.array([s[NUC.index(n)] for n, _ in SPIN_FEATS])
            cor[k, a_i] = correct(O, features(M, it["prev"], spin)[cand], s)
    return cor, nres


def summarise(inst, arms, cor, nres):
    """Instance-mean and residue-weighted fractions per source, label-set bootstrap intervals."""
    ls = np.array([it["label_set"] for it in inst])
    src = np.array([it["source"] for it in inst])
    ok = nres >= 2
    frac = np.where(ok[:, None], cor / np.maximum(nres, 1)[:, None], np.nan)
    rows = []
    r = arms.index(REF)
    for name, mask in [("pooled", np.ones(len(inst), bool))] + [(SRC[s], src == s) for s in SRC]:
        m = mask & ok
        sets = np.unique(ls[m])
        sidx = {s: i for i, s in enumerate(sets)}
        gi = np.array([sidx[s] for s in ls[m]])
        n_inst = np.bincount(gi, minlength=len(sets)).astype(float)
        S = np.zeros((len(sets), len(arms)))
        np.add.at(S, gi, frac[m])
        Cc = np.zeros((len(sets), len(arms)))
        np.add.at(Cc, gi, cor[m])
        Nr = np.bincount(gi, weights=nres[m], minlength=len(sets))
        ib = np.random.default_rng(SEED).integers(0, len(sets), size=(N_BOOT, len(sets)))
        bm = S[ib].sum(1) / n_inst[ib].sum(1)[:, None]            # instance-mean fraction
        bw = Cc[ib].sum(1) / Nr[ib].sum(1)[:, None]               # residue-weighted fraction
        pm = S.sum(0) / n_inst.sum()
        pw = Cc.sum(0) / Nr.sum()
        for a_i, a in enumerate(arms):
            lo, hi = np.percentile(bm[:, a_i], [2.5, 97.5])
            wlo, whi = np.percentile(bw[:, a_i], [2.5, 97.5])
            dm, dw = bm[:, r] - bm[:, a_i], bw[:, r] - bw[:, a_i]
            rows.append({
                "source": name, "arm": a, "n_instances": int(m.sum()), "n_label_sets": len(sets),
                "n_residues": int(nres[m].sum()), "mean_frac": pm[a_i], "ci_lo": lo, "ci_hi": hi,
                "resweighted_frac": pw[a_i], "rw_lo": wlo, "rw_hi": whi,
                "diff_vs_ref": pm[r] - pm[a_i], "diff_lo": np.percentile(dm, 2.5), "diff_hi": np.percentile(dm, 97.5),
                "rw_diff_vs_ref": pw[r] - pw[a_i], "rw_diff_lo": np.percentile(dw, 2.5),
                "rw_diff_hi": np.percentile(dw, 97.5),
            })
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--npz", required=True, help="paired-cell dump of benchmarks.dump_paired_cells")
    p.add_argument("--out", required=True, type=Path, help="output directory")
    p.add_argument("--tag", default="core5", help="suffix of the output file names")
    p.add_argument("--sigma-arms", nargs="*", default=None,
                   help="arms whose mean RMSE sets the common sigma (default: every arm in the dump)")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.npz)
    arms = z["arms"].tolist()
    rmse = pd.DataFrame({x: [np.sqrt(np.mean((z[f"pred_{x}"][z["atom"] == n] - z["true_ppm"][z["atom"] == n]) ** 2))
                             for n in NUC] for x in arms}, index=NUC)
    sig_common = rmse[a.sigma_arms or arms].mean(1).to_numpy()
    rmse.to_csv(a.out / f"rmse_{a.tag}.csv")
    print("per-arm RMSE on these cells:\n", rmse.round(3), "\ncommon sigma:", dict(zip(NUC, sig_common.round(3))), flush=True)
    inst, n_drop = tables(z, arms)
    print(f"{len(inst)} instances; {n_drop} cells with insertion-code residue numbers dropped", flush=True)
    allarms = [*arms, "randomcoil", "oracle"]
    res = []
    for sigma_mode in ("common", "own"):
        sig = {x: (sig_common if sigma_mode == "common" else rmse[x].to_numpy()) for x in arms}
        sig["randomcoil"] = sig_common if sigma_mode == "common" else random_coil_rmse(z)
        sig["oracle"] = sig_common
        for spin in (False, True):
            variant = "spin" if spin else "residue"
            cor, nres = run(inst, allarms, sig, spin)
            t = summarise(inst, allarms, cor, nres)
            t.insert(0, "variant", "spin_system" if spin else "residue")
            t.insert(0, "sigma", sigma_mode)
            res.append(t)
            pd.DataFrame(cor, columns=allarms).assign(
                sid=[i["sid"] for i in inst], label_set=[i["label_set"] for i in inst],
                source=[i["source"] for i in inst], n_res=nres,
            ).to_csv(a.out / f"per_instance_{a.tag}_{sigma_mode}_{variant}.csv.gz", index=False)
            if sigma_mode == "common":
                bins = np.asarray(pd.cut(nres, [1, 29, 99, 299, 10**6],
                                         labels=["2-29", "30-99", "100-299", "300+"]).astype(str))
                f = pd.DataFrame(np.where(nres[:, None] >= 2, cor / np.maximum(nres, 1)[:, None], np.nan), columns=allarms)
                g = f.groupby(bins).mean()
                g.insert(0, "n_instances", pd.Series(bins).value_counts())
                g.to_csv(a.out / f"by_size_{a.tag}_{variant}.csv")
            print(sigma_mode, variant, "done", flush=True)
    out = pd.concat(res)
    out.to_csv(a.out / f"assignment_{a.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[out.sigma == "common"][["variant", "source", "arm", "n_instances", "n_residues", "mean_frac",
                                      "ci_lo", "ci_hi", "diff_vs_ref", "diff_lo", "diff_hi"]].round(4).to_string())


if __name__ == "__main__":
    main()
