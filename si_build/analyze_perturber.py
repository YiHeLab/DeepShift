#!/usr/bin/env python3
"""Dose-response of prediction error against distance to a nonstandard entity.

Statistic, bins, eligibility and multiple-testing correction:

  cell            = (category, distance bin, nucleus)
  bins            = [0,3), [3,5), [5,8), [8,12) A
  statistic       = mean |pred - observed| in the cell / mean |pred - observed| in
                    the baseline (rows with no nonstandard entity of any category
                    within 12 A)
  clustered test  = Welch two-sample t-test on per-cluster means of
                    log1p(|error| / baseline median); not a bootstrap
  eligibility     = n >= 30 targets and >= 3 clusters
  correction      = Benjamini-Hochberg over the eligible cells; bold = q < 0.05
                    (emit_perturber_dose_ppm.py), whichever way the ratio points

The cluster is the shipped grouping unit ``label_set_id``.
The build is entity-grained, one BMRB deposition can realize several entities, and the
AF/NMR/crystal realizations of one label set share their residues -- clustering on the
label set keeps those correlated rows in one cluster, which a per-structure key would
split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import si_common
from scipy import stats

CATEGORIES = ["water", "paramagnetic_transition_metal", "zinc", "other_metal",
              "heme_porphyrin", "phosphate_nucleotide", "sugar",
              "organic_solvent_buffer_detergent", "large_aromatic_ligand", "mse", "other"]
BINS = ((0.0, 3.0, "0-3"), (3.0, 5.0, "3-5"), (5.0, 8.0, "5-8"), (8.0, 12.0, "8-12"))
BACKBONE = ["h_shift", "ha_shift", "ca_shift", "cb_shift", "c_shift", "n_shift"]
SOURCE_NAMES = ["af", "nmr", "xtal"]
# The run's own cache: the prediction dumps' rows are its globals. The shadow folds the
# dumps were evaluated on share its structural arrays and differ only in the atom stage.
CACHE = si_common.CACHE
# The nonstandard-entity exclusion rule ships with the package, so it is read out of this
# repo, not from a copy outside it.
HETATM_RULE = (Path(__file__).resolve().parents[1]
               / "src/deepshift/data/hetatm_exclusion_rule.json")


def bh_adjust(pvalues) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    out = np.full(len(p), np.nan)
    good = np.isfinite(p)
    if not good.any():
        return out
    idx = np.where(good)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out


def cluster_means(values: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    if not len(values):
        return np.empty(0)
    order = np.argsort(clusters, kind="stable")
    c, v = clusters[order], values[order]
    starts = np.r_[0, np.where(c[1:] != c[:-1])[0] + 1]
    return np.add.reduceat(v, starts) / np.diff(np.r_[starts, len(v)])


def summary(values: np.ndarray) -> dict:
    if not len(values):
        return {"n": 0, "mean": np.nan, "median": np.nan, "p90": np.nan, "p95": np.nan}
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95))}


def load_fold(fold: int, preds_dir: Path, dist_dir: Path):
    """Join one fold's paper-model predictions to its per-row perturber distances."""
    sd = CACHE / f"fold_{fold}" / "structural"
    idm = json.loads((sd / "id_maps.json").read_text())
    n_cache = len(np.load(sd / "residue_idx.npy", mmap_mode="r"))
    key_to_g = {}
    mat, lres = idm["materialization_id"], idm["label_residue_id"]
    for g in range(n_cache):
        key_to_g[f"{mat[str(g)]}|{lres[str(g)]}"] = g

    dump = np.load(preds_dir / f"paper_model_fold{fold}.preds.npz", allow_pickle=True)
    cols = [str(c) for c in dump["shift_cols"]]
    rows = np.asarray([key_to_g.get(str(k), -1) for k in dump["row_key"]])
    if (rows < 0).any():
        raise SystemExit(f"fold {fold}: {int((rows < 0).sum())} scored rows have no cache global")

    dist_all = np.load(dist_dir / f"distances_fold{fold}.npy")
    own = np.load(dist_dir / f"distance_from_own_structure_fold{fold}.npy")
    af_only = np.load(CACHE / f"fold_{fold}" / "atom" / "af_only_entity.npy")
    comp = idm["comp_id"]
    return {
        "comp_id": np.asarray([str(comp[str(int(g))]) for g in rows]),
        "fold": np.full(len(rows), fold, dtype=np.int8),
        "global": rows,
        "pred": np.asarray(dump["pred_ppm"], dtype=np.float64),
        "target": np.asarray(dump["resolved_true_ppm"], dtype=np.float64),
        "mask": np.asarray(dump["effective_mask"], dtype=bool),
        "source": np.asarray(dump["source"], dtype=np.int64),
        "label_set": np.asarray([str(x) for x in dump["label_set_id"]]),
        "bmrb": np.asarray([str(x) for x in dump["bmrb"]]),
        "dist": dist_all[rows],
        "own_structure": own[rows],
        "af_only": np.asarray(af_only)[rows],
        "cols": cols,
    }


def p100_mask(comp_id: np.ndarray, target: np.ndarray, cols) -> np.ndarray:
    """The p100 range rule as a cell mask, applied per (residue, nucleus) from the fitted
    cutoff file named by SI_P100_CUTOFFS."""
    policy = si_common.p100_policy()
    keep = np.ones((len(comp_id), len(cols)), dtype=bool)
    for j, col in enumerate(cols):
        lo = np.full(len(comp_id), -np.inf)
        hi = np.full(len(comp_id), np.inf)
        for res in np.unique(comp_id):
            accepted = policy.accepted_range(res, col)
            if accepted is None:
                continue
            sel = comp_id == res
            lo[sel], hi[sel] = accepted
        keep[:, j] = (target[:, j] >= lo) & (target[:, j] <= hi)
    return keep


def analyze(d, cols, out: Path, tag: str, keep_sources, *, apply_p100: bool) -> pd.DataFrame:
    keep_codes = {SOURCE_NAMES.index(s) for s in keep_sources}
    # A row is usable when its source is in scope and every category distance is known.
    # inf means "resolved, no such entity in the structure"; NaN means "never resolved"
    # (structure unparseable, or an AF-only entity with no experimental realization).
    keep = np.isin(d["source"], list(keep_codes)) & ~np.isnan(d["dist"]).any(1)
    sel = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
           for k, v in d.items() if k != "cols"}
    n_p100_dropped = 0
    if apply_p100:
        inrange = p100_mask(sel["comp_id"], sel["target"], cols)
        n_p100_dropped = int((sel["mask"] & ~inrange).sum())
        sel["mask"] = sel["mask"] & inrange
    err = np.abs(sel["pred"] - sel["target"])
    within12 = np.min(sel["dist"], axis=1) < 12.0
    baseline_rows = ~within12
    _, cluster = np.unique(sel["label_set"], return_inverse=True)

    baselines = []
    for j, nuc in enumerate(cols):
        for src in ("pooled", *tuple(keep_sources)):
            q = baseline_rows & sel["mask"][:, j]
            if src != "pooled":
                q = q & (sel["source"] == SOURCE_NAMES.index(src))
            baselines.append({"nucleus": nuc, "source": src, **summary(err[q, j]),
                              "n_rows": int(q.sum()),
                              "n_clusters": int(np.unique(cluster[q]).size)})
    pd.DataFrame(baselines).to_csv(out / f"baseline_errors_{tag}.csv", index=False)

    cells = []
    for cj, cat in enumerate(CATEGORIES):
        distance = sel["dist"][:, cj]
        for lo, hi, label in BINS:
            in_bin = (distance >= lo) & (distance < hi)
            for nj, nuc in enumerate(cols):
                q = in_bin & sel["mask"][:, nj]
                vals = err[q, nj]
                bq = baseline_rows & sel["mask"][:, nj]
                bvals = err[bq, nj]
                cell = {"category": cat, "distance_bin": label, "distance_lo": lo,
                        "distance_hi": hi, "nucleus": nuc, **summary(vals),
                        "n_rows": int(in_bin.sum()),
                        "n_clusters": int(np.unique(cluster[q]).size)}
                for s in SOURCE_NAMES:
                    cell[f"n_{s}"] = int((q & (sel["source"] == SOURCE_NAMES.index(s))).sum())
                bs = summary(bvals)
                cell.update({f"baseline_{k}": v for k, v in bs.items()})
                cell["delta_mean"] = cell["mean"] - bs["mean"] if len(vals) and len(bvals) else np.nan
                cell["delta_median"] = (cell["median"] - bs["median"]
                                        if len(vals) and len(bvals) else np.nan)
                cell["mean_ratio"] = cell["mean"] / bs["mean"] if len(vals) and bs["mean"] > 0 else np.nan
                cell["median_ratio"] = (cell["median"] / bs["median"]
                                        if len(vals) and bs["median"] > 0 else np.nan)
                if len(vals) and len(bvals):
                    sb = np.sort(bvals)
                    less = np.searchsorted(sb, vals, side="left")
                    leq = np.searchsorted(sb, vals, side="right")
                    cell["cliffs_delta"] = float(2 * np.mean((less + leq) / (2.0 * len(sb))) - 1)
                else:
                    cell["cliffs_delta"] = np.nan
                scale = bs["median"] if np.isfinite(bs["median"]) and bs["median"] > 0 else 1.0
                cv = cluster_means(np.log1p(vals / scale), cluster[q])
                cb = cluster_means(np.log1p(bvals / scale), cluster[bq])
                if len(cv) >= 3 and len(cb) >= 3:
                    test = stats.ttest_ind(cv, cb, equal_var=False)
                    diff = float(cv.mean() - cb.mean())
                    se = float(np.sqrt(cv.var(ddof=1) / len(cv) + cb.var(ddof=1) / len(cb)))
                    cell.update(cluster_log_effect=diff, cluster_effect_ci_low=diff - 1.96 * se,
                                cluster_effect_ci_high=diff + 1.96 * se,
                                cluster_p=float(test.pvalue))
                else:
                    cell.update(cluster_log_effect=np.nan, cluster_effect_ci_low=np.nan,
                                cluster_effect_ci_high=np.nan, cluster_p=np.nan)
                cells.append(cell)
    frame = pd.DataFrame(cells)
    eligible = (frame.n >= 30) & (frame.n_clusters >= 3)
    frame["cluster_q_bh"] = np.nan
    frame.loc[eligible, "cluster_q_bh"] = bh_adjust(frame.loc[eligible, "cluster_p"])
    # Sensitivity: the BH family above spans all 64 nuclei the paper model carries. This
    # column corrects over the six printed nuclei only, so a reader can see whether any
    # bold cell depends on the family size.
    bb = eligible & frame.nucleus.isin(BACKBONE)
    frame["cluster_q_bh_backbone_family"] = np.nan
    frame.loc[bb, "cluster_q_bh_backbone_family"] = bh_adjust(frame.loc[bb, "cluster_p"])
    frame["ratio_vs_8_12_mean"] = np.nan
    frame["delta_vs_8_12_mean"] = np.nan
    for (_cat, _nuc), idx in frame.groupby(["category", "nucleus"]).groups.items():
        far = frame.loc[idx][frame.loc[idx].distance_bin == "8-12"]
        if len(far) and np.isfinite(far.iloc[0]["mean"]) and far.iloc[0]["mean"] > 0:
            fmean = float(far.iloc[0]["mean"])
            frame.loc[idx, "ratio_vs_8_12_mean"] = frame.loc[idx, "mean"] / fmean
            frame.loc[idx, "delta_vs_8_12_mean"] = frame.loc[idx, "mean"] - fmean
    frame.to_csv(out / f"stratified_errors_{tag}.csv", index=False)

    # ---- aggregate: inside the SHIPPED exclusion shell vs baseline ----
    # "Residues near a ligand degrade in RMSE" is a pooled statement, not a cell of the
    # table, so it gets its own test: every row inside the shipped cutoff of any
    # excluded category, against baseline.
    cutoffs = json.loads(HETATM_RULE.read_text())
    inside = np.zeros(len(sel["fold"]), dtype=bool)
    for cat, spec in cutoffs["exclude"].items():
        inside |= sel["dist"][:, CATEGORIES.index(cat)] < float(spec["cutoff_A"])
    aggregate = {}
    for j, nuc in enumerate(cols):
        q, bq = inside & sel["mask"][:, j], baseline_rows & sel["mask"][:, j]
        if not q.sum() or not bq.sum():
            continue
        a, b = err[q, j], err[bq, j]
        aggregate[nuc] = {
            "n_inside": int(q.sum()), "n_baseline": int(bq.sum()),
            "mae_inside": float(a.mean()), "mae_baseline": float(b.mean()),
            "mae_ratio": float(a.mean() / b.mean()),
            "rmse_inside": float(np.sqrt((a ** 2).mean())),
            "rmse_baseline": float(np.sqrt((b ** 2).mean())),
            "rmse_ratio": float(np.sqrt((a ** 2).mean()) / np.sqrt((b ** 2).mean()))}
    (out / f"aggregate_shell_vs_baseline_{tag}.json").write_text(
        json.dumps({"rule": cutoffs["exclude"], "per_nucleus": aggregate}, indent=1) + "\n")

    counts = {
        "tag": tag, "sources": list(keep_sources),
        "p100_range_rule_applied": bool(apply_p100),
        "p100_cells_dropped": n_p100_dropped,
        "rows_inside_shipped_shell": int(inside.sum()),
        "n_rows": len(sel["fold"]), "n_clusters": len(np.unique(cluster)),
        "n_label_sets": len(np.unique(sel["label_set"])),
        "n_bmrb": len(np.unique(sel["bmrb"])),
        "n_targets": int(sel["mask"].sum()),
        "baseline_rows": int(baseline_rows.sum()),
        "baseline_targets": int(sel["mask"][baseline_rows].sum()),
        "within12_rows": int(within12.sum()),
        "source_rows": {s: int((sel["source"] == SOURCE_NAMES.index(s)).sum())
                        for s in SOURCE_NAMES},
        "rows_with_own_structure_distance": int(sel["own_structure"].sum()),
        "rows_with_transferred_distance": int((~sel["own_structure"]).sum()),
        "fold_rows": {int(f): int((sel["fold"] == f).sum()) for f in np.unique(sel["fold"])},
        "dropped_no_distance": int((~keep & np.isin(d["source"], list(keep_codes))).sum()),
        "dropped_af_only_entity": int((d["af_only"] & np.isin(d["source"], list(keep_codes))
                                       & np.isnan(d["dist"]).any(1)).sum()),
    }
    (out / f"analysis_counts_{tag}.json").write_text(json.dumps(counts, indent=1) + "\n")
    print(json.dumps(counts, indent=1), flush=True)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--distances", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 6])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parts = [load_fold(f, args.preds, args.distances) for f in args.folds]
    cols = parts[0]["cols"]
    if any(p["cols"] != cols for p in parts):
        raise SystemExit("shift columns differ between folds")
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "cols"}
    print(f"pooled rows: {len(merged['fold']):,}", flush=True)
    for sources, name in ((("nmr", "xtal"), "experimental"), (("af", "nmr", "xtal"), "allsource")):
        for p100 in (True, False):
            analyze(merged, cols, args.out, name if p100 else f"{name}_nop100",
                    sources, apply_p100=p100)


if __name__ == "__main__":
    main()
