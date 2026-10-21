#!/usr/bin/env python3
"""Per-target decoy ordering ledger.

One row per 3DRobot target: how many candidate structures were scored, which
out-of-fold checkpoint scored them, where the native structure ranked under each
method, and how well each method's score orders the candidates by accuracy.

The cells are the decoy pool's: the observed residue-nucleus cells DeepShift,
UCBShift-X and the MLFF predictor all predicted within each candidate, which is the
cohort the main text's decoy comparison and its figure are scored on. So the paired
Kendall difference printed under the table is the one the main text quotes, and it is
recomputed here and checked against the pool's own comparison report, interval and all.
The other paired differences the main text quotes (each pair of the three methods, by
Kendall tau, Spearman rho and Kendall tau within 2 Angstroms) are checked the same way
and printed.

Each per-target prediction file the pool read must carry the one member
``cohort_single.json`` names for that target.

Reads only cached scores. No model is run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import si_common
from scipy.stats import kendalltau

# The pooled decoy run and its cohort file are experiment records, not release code, so
# they stay outside the repo and are resolved by environment.
POOL = si_common.root("DEEPSHIFT_DECOY_POOLED")
SCORES = POOL / "relaxed_full_candidate_scores.parquet"
CHECK = POOL / "relaxed_full_target_metrics.csv"
COMPARISON = POOL / "relaxed_full_comparison.json"
COHORT = si_common.root("SI_DECOY_COHORT")      # the cohort_single.json that drove it

OUT = si_common.OUT
OUT_SNIPPET = OUT / "SNIPPET_decoy_ledger.tex"

# the score the deployed figure ranks by: the equal-weight mean of the available H, HA, N r
SCORE = "mean_r"
DEEPSHIFT, UCBX, MLFF = "DeepShift", "UCBShift-X", "MLFF-MACE"
# The pool's own paired bootstrap (pool_relaxed_decoys.bootstrap_difference): the table
# reproduces it exactly, so the interval it prints is the main text's.
N_BOOT = 200_000
SEED = 20260814


def paired_bootstrap(delta: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    boot = delta[rng.integers(0, len(delta), size=(N_BOOT, len(delta)))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def check_scoring_members(comparison: dict, cohort: dict) -> None:
    """Each per-target prediction file the pool read carries one member, the one the
    cohort names for that target."""
    files = comparison.get("deepshift_prediction_files")
    if not files:
        raise SystemExit(f"{COMPARISON} records no DeepShift prediction files")
    single = {t["target"]: str(t["single_member"]) for t in cohort["targets"]}
    for target, record in files.items():
        path = Path(os.path.expandvars(record["path"]))
        frame = pd.read_parquet(path, columns=["members", "n_members"])
        members = sorted(set(frame.members.astype(str)))
        if members != [single[target]] or set(frame.n_members) != {1}:
            raise SystemExit(f"{target}: the pool read members {members}, the cohort "
                             f"names member {single[target]} alone")


# The paired differences the main text quotes: each pair of the pool's three methods, by
# Kendall tau and Spearman rho over all 300 decoys and by Kendall tau over the decoys
# within 2 Angstroms of the experimental structure.
PAIRS = ((DEEPSHIFT, UCBX), (DEEPSHIFT, MLFF), (UCBX, MLFF))
ORDERINGS = {"kendall_tau": "kendall_score_vs_neg_rmsd",
             "spearman_rho": "spearman_score_vs_neg_rmsd",
             "kendall_tau_within_2A": "kendall_le_2A"}


def main_text_paired(check: pd.DataFrame, comparison: dict) -> dict:
    """Mean per-target difference and its interval for every pair and ordering statistic,
    recomputed from the pool's per-target metrics and checked against its report."""
    report = comparison["all_targets"]["paired"]
    out = {}
    for left, right in PAIRS:
        pair = f"{left} - {right}"
        out[pair] = {}
        for name, column in ORDERINGS.items():
            delta = (check.loc[left][column] - check.loc[right][column]).to_numpy()
            mean = float(delta.mean())
            lo, hi = paired_bootstrap(delta)
            quoted = report[f"{SCORE}/{pair}"][column]
            if not (np.isclose(quoted["mean_left_minus_right"], mean, rtol=0, atol=1e-12)
                    and np.allclose(quoted["ci95"], [lo, hi], rtol=0, atol=1e-12)):
                raise SystemExit(f"{pair} {column}: the pool's report gives {quoted}, "
                                 f"this ledger {mean} [{lo}, {hi}]")
            out[pair][name] = {"mean_difference": mean, "ci95": [lo, hi],
                               "n_targets": len(delta)}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(SCORES)
    methods = [DEEPSHIFT, UCBX]
    assert set(methods) <= set(scores.method.unique()), scores.method.unique()
    scores = scores[scores.method.isin(methods)]
    comparison = json.loads(COMPARISON.read_text())

    with open(COHORT) as fh:
        cohort = json.load(fh)
    check_scoring_members(comparison, cohort)
    # The scoring member per target, from the files the pool read (checked above to be
    # the one member the cohort names).
    member_from_preds = {target: int(record["members"]) for target, record in
                         comparison["deepshift_prediction_files"].items()}
    all_eligible = {
        t["target"]: bool(t["clean_for_all_members"]) for t in cohort["targets"]
    }

    rows = []
    for target, group in scores.groupby("target"):
        record = {"target": target, "member": member_from_preds[target]}
        record["all_five_eligible"] = all_eligible[target]
        per_method = {}
        for method in methods:
            g = group[group.method == method]
            n = len(g)
            native = g[g.candidate == "native.pdb"]
            assert len(native) == 1, (target, method)
            native_score = float(native[SCORE].iloc[0])
            # rank 1 = highest score; a decoy tying the native counts against it, as the
            # pool's own analyzer ranks
            rank = int((g[g.candidate != "native.pdb"][SCORE] >= native_score).sum()) + 1
            best = g.sort_values(SCORE, ascending=False).iloc[0]
            # Ordering quality is measured over the decoys only. The experimental
            # structure sits at exactly zero RMSD, so keeping it would add the same
            # extreme point to every target and inflate every correlation. This is
            # the convention the run's own scorer used.
            decoys = g[g.candidate != "native.pdb"]
            assert len(decoys) == n - 1, (target, method)
            tau = float(
                kendalltau(decoys[SCORE].to_numpy(), -decoys.rmsd.to_numpy()).statistic
            )
            per_method[method] = {
                "n_candidates": n,
                "n_decoys": len(decoys),
                "native_rank": rank,
                "kendall_tau": tau,
                "top1_candidate": str(best.candidate),
                "top1_rmsd_angstrom": float(best.rmsd),
            }
        record["n_candidates"] = per_method[methods[0]]["n_candidates"]
        assert per_method[methods[1]]["n_candidates"] == record["n_candidates"]
        record["deepshift"] = per_method[DEEPSHIFT]
        record["ucbshift_x"] = per_method[UCBX]
        record["tau_difference"] = (
            per_method[DEEPSHIFT]["kendall_tau"] - per_method[UCBX]["kendall_tau"]
        )
        rows.append(record)

    rows.sort(key=lambda r: r["target"])
    assert len(rows) == 52, len(rows)

    # Independent cross-check: the pool's own per-target metrics, written by the n52
    # analyzer it imports, recomputed native rank and Kendall tau. Every value must agree.
    check = pd.read_csv(CHECK)
    check = check[check.metric == SCORE].set_index(["method", "target"])
    disagreements = []
    for r in rows:
        for method, key in ((DEEPSHIFT, "deepshift"), (UCBX, "ucbshift_x")):
            ref = check.loc[(method, r["target"])]
            if int(ref.native_rank) != r[key]["native_rank"]:
                disagreements.append((r["target"], method, "native_rank"))
            if abs(float(ref.kendall_score_vs_neg_rmsd) - r[key]["kendall_tau"]) > 1e-9:
                disagreements.append((r["target"], method, "kendall_tau"))
    assert not disagreements, disagreements

    top1 = {
        m: sum(1 for r in rows if r[k]["native_rank"] == 1)
        for m, k in ((DEEPSHIFT, "deepshift"), (UCBX, "ucbshift_x"))
    }
    missed = {
        m: [r["target"] for r in rows if r[k]["native_rank"] != 1]
        for m, k in ((DEEPSHIFT, "deepshift"), (UCBX, "ucbshift_x"))
    }

    diffs = np.array([r["tau_difference"] for r in rows])
    mean_diff = float(diffs.mean())
    lo, hi = paired_bootstrap(diffs)
    # The number the main text quotes is the pool's own; this table must print it.
    quoted = comparison["all_targets"]["paired"][f"{SCORE}/{DEEPSHIFT} - {UCBX}"][
        "kendall_score_vs_neg_rmsd"]
    if not (np.isclose(quoted["mean_left_minus_right"], mean_diff, rtol=0, atol=1e-12)
            and np.allclose(quoted["ci95"], [lo, hi], rtol=0, atol=1e-12)):
        raise SystemExit(f"the pool's comparison report gives {quoted}, this ledger "
                         f"{mean_diff} [{lo}, {hi}]")
    paired = main_text_paired(check, comparison)
    mean_tau = {
        DEEPSHIFT: float(np.mean([r["deepshift"]["kendall_tau"] for r in rows])),
        UCBX: float(np.mean([r["ucbshift_x"]["kendall_tau"] for r in rows])),
    }

    # ---- table -------------------------------------------------------------
    def signed(value: float, places: int) -> str:
        return f"{value:+.{places}f}"

    head = "\n".join([
        " & & & \\multicolumn{2}{c}{Experimental rank} & "
        "\\multicolumn{2}{c}{Kendall $\\tau$} & \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Target & Candidates & Model & DeepShift & UCBShift-X & DeepShift & "
        "UCBShift-X & Difference \\\\",
    ])
    lines = []
    for r in rows:
        member = f"{r['member']}$^{{\\ast}}$" if r["all_five_eligible"] else str(r["member"])
        lines.append(
            f"{r['target']} & {r['n_candidates']} & {member} & "
            f"{r['deepshift']['native_rank']} & {r['ucbshift_x']['native_rank']} & "
            f"{r['deepshift']['kendall_tau']:.3f} & {r['ucbshift_x']['kendall_tau']:.3f} & "
            f"{signed(r['tau_difference'], 3)} \\\\"
        )
    lines.append("\\midrule")
    lines.append(
        "All 52 targets & 15{,}652 & --- & "
        f"{top1[DEEPSHIFT]}/52 first & {top1[UCBX]}/52 first & "
        f"{mean_tau[DEEPSHIFT]:.3f} & {mean_tau[UCBX]:.3f} & "
        f"{signed(mean_diff, 3)} \\\\"
    )
    lines.append(
        "\\multicolumn{8}{l}{\\footnotesize Mean difference "
        f"{signed(mean_diff, 4)}, 95\\% interval [{signed(lo, 4)}, {signed(hi, 4)}] "
        "from 200{,}000 resamples of the 52 targets.} \\\\"
    )
    lines.append(
        "\\multicolumn{8}{l}{\\footnotesize $^{\\ast}$ All five models were eligible for "
        "this target; one was chosen by a fixed rule.} \\\\"
    )
    lines.append("\\bottomrule")
    tex = si_common.table("", "tab:decoy_ledger", "\n".join(lines),
                          longtable="lrcrrrrr", head=head, caption="",
                          size="\\small")

    si_common.emit("si_decoy_ledger", tex, script="build_decoy_ledger.py")

    # ---- splice snippet ----------------------------------------------------
    snippet = f"""% Splice block for SI_v3.tex -- paste where the decoy ledger belongs.
% The \\capneeded argument is a bare fact list, not caption prose. The caption in
% si_decoy_ledger.tex is empty and needs a short title from the author.

\\input{{si/tables/si_decoy_ledger}}
\\cohort{{3DRobot decoy set, 52 targets x 301 candidates (the experimental structure
plus 300 decoys); scored by one out-of-fold cross-validation model per target;
Kendall $\\tau$ and ranks, not ppm; raw residuals, nothing mean-centred.
Read from \\texttt{{ucbshift\\_\\allowbreak common\\_\\allowbreak cell\\_\\allowbreak scores\\_\\_oof.\\allowbreak parquet}}.}}
\\capneeded{{Facts to assert: one row per 3DRobot target. Each target has 301 candidate
structures, the experimental one and 300 decoys, and is scored by the single
cross-validation model that never saw it. The score is the equal-weight mean of the
available H, HA and N correlations between predicted and observed shifts, the same score the
decoy figure ranks by. The rank column runs over all 301 candidates; the Kendall
$\\tau$ columns run over the 300 decoys only, because the experimental structure sits
at zero RMSD and would add the same extreme point to every target. Both methods place
the experimental structure first for
{top1[DEEPSHIFT]} of 52 targets; the one exception for both is
{missed[DEEPSHIFT][0]}, where DeepShift's top choice is a decoy
{rows[[r['target'] for r in rows].index(missed[DEEPSHIFT][0])]['deepshift']['top1_rmsd_angstrom']:.2f}
angstroms from the experimental structure and UCBShift-X's is one
{rows[[r['target'] for r in rows].index(missed[UCBX][0])]['ucbshift_x']['top1_rmsd_angstrom']:.2f}
angstroms away. Mean Kendall $\\tau$ is {mean_tau[DEEPSHIFT]:.3f} for DeepShift and
{mean_tau[UCBX]:.3f} for UCBShift-X; the mean paired difference is
{signed(mean_diff, 4)} with a 95\\% interval of [{signed(lo, 4)}, {signed(hi, 4)}] from
200,000 resamples of the 52 targets. State the claim boundary: a small
ordering advantage over the whole RMSD range under this one score, not an advantage
in picking the experimental structure and not shown for the other two scores.}}
"""
    OUT_SNIPPET.write_text(snippet)

    print(f"rows {len(rows)}  top1 {top1}  missed {missed}")
    print(f"mean tau {mean_tau}")
    print(f"mean diff {mean_diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("main-text paired differences:", json.dumps(paired, indent=1))
    print(f"wrote {OUT_SNIPPET}")


if __name__ == "__main__":
    main()
