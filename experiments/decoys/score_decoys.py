"""Score DeepShift, UCBShift-X and MLFF-MACE on the 3DRobot decoys and pool the three arms.

Per candidate (the native structure or a decoy), each arm is scored on the observed
residue-nucleus cells all three arms predicted: the Pearson r between predicted and
observed shift for H, Hα and N (a nucleus with fewer than five common cells, or a constant
series, is absent), and ``mean_r``, the equal-weight mean of the available r. Per target
and arm: the native structure's rank, and the ordering of the 300 decoys against their
C-alpha RMSD (Pearson, Spearman, Kendall). Predicted and observed residues are paired by
an affine-gap global alignment of their sequences. Bootstraps resample targets.

Writes the three files ``figures/fig_structure_response.py`` and
``si_build/build_decoy_ledger.py`` read: ``relaxed_full_candidate_scores.parquet``,
``relaxed_full_target_metrics.csv`` and ``relaxed_full_comparison.json``.

    python score_decoys.py --cohort cohort.json --deepshift <dir> --ucbshift <dir> \\
        --mlff <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from scipy.stats import kendalltau, pearsonr, spearmanr

from deepshift.domain import AA_3_TO_1

HERE = Path(__file__).resolve().parent
SHIFTS = HERE.parents[1] / "data" / "deepshift_shifts.parquet"

NUCLEI = ("h_shift", "ha_shift", "n_shift")
NUCLEUS_SHORT = {"h_shift": "H", "ha_shift": "HA", "n_shift": "N"}
METHODS = ("DeepShift", "UCBShift-X", "MLFF-MACE")
UCB_COLUMNS = {"h_shift": "H_X", "ha_shift": "HA_X", "n_shift": "N_X"}
MLFF_KEYS = {"H": "h_shift", "HA": "ha_shift", "N": "n_shift"}
ONE_TO_THREE = {one: three for three, one in AA_3_TO_1.items()}
SCORE = "mean_r"
PAIRED_COLUMNS = [
    "native_score", "native_minus_mean_decoy", "native_minus_best_decoy",
    "native_z", "selected_rmsd", "pearson_score_vs_neg_rmsd",
    "spearman_score_vs_neg_rmsd", "kendall_score_vs_neg_rmsd",
    "kendall_le_2A", "kendall_le_4A",
]


def load_labels(cohort: dict, shifts: Path) -> dict[str, pd.DataFrame]:
    """Observed H, Hα, N of each target's scored label set (fenced cells are null)."""
    wanted = {t["scored_label_set_id"] for t in cohort["targets"]}
    frame = pd.read_parquet(shifts, filters=[("label_set_id", "in", sorted(wanted))],
                            columns=["label_set_id", "residue_id", "residue_code", *NUCLEI])
    return {t["target"]: frame[frame.label_set_id == t["scored_label_set_id"]]
            .sort_values("residue_id").reset_index(drop=True) for t in cohort["targets"]}


def matched_pairs(predicted_names, observed_names) -> list[tuple[int, int]]:
    """Index pairs of identical residues under a global alignment (construct indels exist)."""
    predicted = "".join(AA_3_TO_1.get(str(n).upper(), "X") for n in predicted_names)
    observed = "".join(AA_3_TO_1.get(str(n).upper(), "X") for n in observed_names)
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    try:
        alignment = next(iter(aligner.align(predicted, observed)))
    except StopIteration:
        return []
    first, second = str(alignment[0]), str(alignment[1])
    pairs, i, j = [], 0, 0
    for left, right in zip(first, second):
        if left != "-" and right != "-" and left == right:
            pairs.append((i, j))
        if left != "-":
            i += 1
        if right != "-":
            j += 1
    return pairs


def load_mlff(directory: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(directory.glob("*.json")):
        rows = []
        for resnum, values in json.loads(path.read_text())["predictions"].items():
            row = {"residue_id": int(resnum),
                   "residue_name": ONE_TO_THREE.get(values["resname"], "UNK")}
            for key, column in MLFF_KEYS.items():
                if key in values:
                    row[column] = float(values[key])
            rows.append(row)
        frames[path.stem + ".pdb"] = pd.DataFrame(rows).sort_values(
            "residue_id").reset_index(drop=True)
    return frames


def value_map(prediction, columns, pairs) -> dict[str, dict[int, float]]:
    """{nucleus: {observed row index: predicted value}} over the aligned residues."""
    result = {}
    for nucleus, column in columns.items():
        values = {}
        if column in prediction.columns:
            series = prediction[column]
            for predicted_index, observed_index in pairs:
                value = series.iloc[predicted_index]
                if pd.notna(value):
                    values[observed_index] = float(value)
        result[nucleus] = values
    return result


def score_common_cells(maps: dict, observed: pd.DataFrame) -> dict[str, dict]:
    """Per-arm r on the cells every arm predicted, and their equal-weight mean."""
    records = {m: {} for m in maps}
    nucleus_r = {m: [] for m in maps}
    for nucleus in NUCLEI:
        indices = sorted(set.intersection(*(set(maps[m][nucleus]) for m in maps)))
        indices = [i for i in indices if pd.notna(observed[nucleus].iloc[i])]
        short = NUCLEUS_SHORT[nucleus]
        for m in maps:
            records[m][f"n_{short}"] = len(indices)
        if len(indices) < 5:
            continue
        obs = observed[nucleus].iloc[indices].to_numpy(float)
        for m in maps:
            pred = np.asarray([maps[m][nucleus][i] for i in indices], float)
            if pred.std() > 1e-8 and obs.std() > 1e-8:
                records[m][f"r_{short}"] = float(np.corrcoef(pred, obs)[0, 1])
                nucleus_r[m].append(records[m][f"r_{short}"])
    for m in maps:
        records[m]["n_nuclei_scored"] = len(nucleus_r[m])
        if nucleus_r[m]:
            records[m][SCORE] = float(np.mean(nucleus_r[m]))
    return records


def target_metrics(scores: pd.DataFrame) -> dict:
    """Native rank and decoy ordering for one target and arm."""
    group = scores.dropna(subset=[SCORE]).copy()
    native = group[group.candidate == "native.pdb"]
    decoys = group[group.candidate != "native.pdb"]
    if len(native) != 1 or len(decoys) < 5:
        raise SystemExit("invalid target support")
    native_score = float(native[SCORE].iloc[0])
    values = decoys[SCORE].to_numpy(float)
    selected = group.loc[group[SCORE].idxmax()]
    best_decoy = decoys.loc[decoys[SCORE].idxmax()]
    decoy_sd = float(values.std(ddof=0))
    result = {
        "metric": SCORE,
        "native_rank": 1 + int(np.sum(values >= native_score)),
        "native_score": native_score,
        "native_minus_mean_decoy": native_score - float(values.mean()),
        "native_minus_best_decoy": native_score - float(values.max()),
        "native_z": (native_score - float(values.mean())) / decoy_sd if decoy_sd > 0 else np.nan,
        "selected_rmsd": float(selected.rmsd),
        "selected_rmsd_regret": float(selected.rmsd - group.rmsd.min()),
        "best_decoy_rmsd": float(best_decoy.rmsd),
        # Ordering is over the decoys only: the native structure sits at RMSD 0 and would
        # add the same extreme point to every target.
        "pearson_score_vs_neg_rmsd": float(pearsonr(decoys[SCORE], -decoys.rmsd).statistic),
        "spearman_score_vs_neg_rmsd": float(spearmanr(decoys[SCORE], -decoys.rmsd).statistic),
        "kendall_score_vs_neg_rmsd": float(kendalltau(decoys[SCORE], -decoys.rmsd).statistic),
    }
    for cutoff in (2.0, 4.0):
        near = decoys[decoys.rmsd <= cutoff].dropna(subset=[SCORE])
        result[f"n_decoys_le_{cutoff:g}A"] = len(near)
        result[f"kendall_le_{cutoff:g}A"] = (
            float(kendalltau(near[SCORE], -near.rmsd).statistic) if len(near) >= 5 else np.nan)
    return result


def bootstrap_means(per_target: pd.DataFrame, reps: int = 200_000) -> dict:
    rng = np.random.default_rng(20260801)
    result = {}
    for column in PAIRED_COLUMNS:
        values = per_target[column].dropna().to_numpy(float)
        if len(values) < 2:
            continue
        means = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
        result[column] = {"mean": float(values.mean()),
                          "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
                          "n_targets": len(values)}
    return result


def bootstrap_difference(left, right, column, reps: int = 200_000) -> dict:
    paired = (left.set_index("target")[[column]]
              .join(right.set_index("target")[[column]], lsuffix="_l", rsuffix="_r").dropna())
    delta = (paired[f"{column}_l"] - paired[f"{column}_r"]).to_numpy(float)
    if len(delta) < 2:
        return {"n_targets": len(delta)}
    rng = np.random.default_rng(20260814)
    boot = delta[rng.integers(0, len(delta), size=(reps, len(delta)))].mean(axis=1)
    return {"n_targets": len(delta), "mean_left_minus_right": float(delta.mean()),
            "ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975])]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--deepshift", type=Path, required=True,
                        help="run_inference.py output: <target>.parquet")
    parser.add_argument("--ucbshift", type=Path, required=True,
                        help="ucbshift_x.py output: <target>/<candidate>.csv")
    parser.add_argument("--mlff", type=Path, required=True,
                        help="mlff_decoys.py output: <target>/<candidate>.json")
    parser.add_argument("--shifts", type=Path, default=SHIFTS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cohort = json.loads(args.cohort.read_text())
    entries = {t["target"]: t for t in cohort["targets"]}
    labels = load_labels(cohort, args.shifts)

    rows, prediction_files = [], {}
    for target in sorted(entries):
        member = entries[target]["single_member"]
        path = args.deepshift / f"{target}.parquet"
        deepshift = pd.read_parquet(path)
        if set(deepshift.members.astype(str)) != {str(member)}:
            raise SystemExit(f"{path} does not hold member {member} alone")
        prediction_files[target] = {"path": str(path.resolve()), "members": str(member)}
        mlff = load_mlff(args.mlff / target)
        observed = labels[target]
        observed_codes = observed.residue_code.tolist()
        pairs_cache: dict[tuple, list] = {}

        def pairs_for(names, cache=pairs_cache, codes=observed_codes):
            signature = tuple(names)
            if signature not in cache:
                cache[signature] = matched_pairs(list(signature), codes)
            return cache[signature]

        for (candidate, rmsd), deep in deepshift.groupby(["candidate", "rmsd"], sort=False):
            ucb_path = args.ucbshift / target / candidate.replace(".pdb", ".csv")
            if candidate not in mlff or not ucb_path.is_file():
                raise SystemExit(f"{target}/{candidate}: missing a UCBShift-X or MLFF output")
            deep = deep.sort_values("residue_id").reset_index(drop=True)
            ucb = pd.read_csv(ucb_path).rename(columns={"RESNAME": "residue_name"})
            mlff_frame = mlff[candidate]
            maps = {
                "DeepShift": value_map(deep, {n: n for n in NUCLEI},
                                       pairs_for(deep.residue_name.tolist())),
                "UCBShift-X": value_map(ucb, UCB_COLUMNS,
                                        pairs_for(ucb.residue_name.tolist())),
                "MLFF-MACE": value_map(mlff_frame, {n: n for n in NUCLEI},
                                       pairs_for(mlff_frame.residue_name.tolist())),
            }
            for method, values in score_common_cells(maps, observed).items():
                rows.append({"method": method, "target": target, "candidate": candidate,
                             "rmsd": float(rmsd), **values})
        print(f"{target}: scored", flush=True)

    candidates = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(args.out / "relaxed_full_candidate_scores.parquet", index=False)
    targets = pd.DataFrame(
        [{"method": method, "target": target, **target_metrics(group)}
         for (method, target), group in candidates.groupby(["method", "target"])])
    targets.to_csv(args.out / "relaxed_full_target_metrics.csv", index=False)

    summaries = {f"{method}/{SCORE}": {
        "native_top1": int((group.native_rank == 1).sum()), "n_targets": len(group),
        "bootstrap": bootstrap_means(group)} for method, group in targets.groupby("method")}
    pairs = {}
    for i, left in enumerate(METHODS):
        for right in METHODS[i + 1:]:
            pairs[f"{SCORE}/{left} - {right}"] = {
                column: bootstrap_difference(targets[targets.method == left],
                                             targets[targets.method == right], column)
                for column in PAIRED_COLUMNS}
    report = {"score": SCORE, "n_targets": int(candidates.target.nunique()),
              "deepshift_prediction_files": prediction_files,
              "all_targets": {"summaries": summaries, "paired": pairs}}
    (args.out / "relaxed_full_comparison.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
