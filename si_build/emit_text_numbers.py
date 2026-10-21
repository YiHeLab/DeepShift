#!/usr/bin/env python3
"""Numbers the Methods and the supplement print in their text, with no float behind them.

Each number is recomputed here from the files of the run that measured it (or from the
released Parquet tables in ``data/``) and written to ``out/text_numbers.json`` as
``name -> {"value", "description"}``, the description naming the file key or computation it
comes from. REPRODUCTION.md maps each printed number to its name.

Roots: ``DEEPSHIFT_RUN`` (the build, its rereferencing diagnostics -- ``SI_REREF_DIAGNOSTICS``
names another file of ``scripts/reref_diagnostics.py`` -- and its member records),
``DEEPSHIFT_XEISD_PROTOCOL`` (the X-EISD restarts and their report), and for the
self-retrieval rates of the supplement ``SI_SELF_RETRIEVAL_UCBY`` and
``SI_SELF_RETRIEVAL_SHIFTY61`` (the two prediction runs), ``SI_GATE_CSPRED_TM`` and
``DEEPSHIFT_REFDB_SHIFTS`` (UCBShift's test targets and RefDB mirror, third-party),
``DEEPSHIFT_REFDB_LIBRARY`` and ``DEEPSHIFT_UCBSHIFT_LEAKAGE`` (SHIFTX2's RefDB copy and the
61-protein roster). ``--allow-missing`` records a group whose files are absent instead of
stopping.

    cd si_build && python emit_text_numbers.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import si_common as C

from benchmarks.runner_up_leads import lead_tests, table1_competitors
from deepshift.data import folds as fold_stage

# Two values of one cell differ when further apart than this (the fetch stage's rule).
from deepshift.data.fetch import DUP_VALUE_EPS
from deepshift.data.reref import REREF_APPLY_NUCS

# The released dataset: the three Parquet tables the code ships in data/.
RELEASED = C.REPO / "data"
# The cells UCBShift's paper removes from its own test set; every score on it removes them.
UCB200_EXCLUSIONS = C.REPO / "benchmarks" / "reference" / "ucb200_paper_exclusions.json"


def entry(value, description: str) -> dict:
    """One printed number and the file key or computation it comes from."""
    return {"value": value, "description": description}


class Group:
    """The values one group of numbers yields."""

    def __init__(self):
        self.values = {}

    def read(self, path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


# --------------------------------------------------------------------------------------
# Shift-list selection (fetch stage)
# --------------------------------------------------------------------------------------

_SELECTION_COLUMNS = ["logical_entity_id", "label_set_id", "bmrb_id", "shift_list_id",
                      "selected", "reason", "n_conflict_cells",
                      "alternate_label_set_disagrees", "candidate_payload_json"]


def _cells(payload_json: str) -> dict:
    """{(polymer position, atom): ppm} of one candidate, keyed as fetch compares them."""
    return {(int(c[0]), c[2]): float(c[3])
            for c in json.loads(payload_json)["observed_cells"]}


def shift_lists(g: Group, data: Path) -> None:
    """Depositions, selected lists, and how far the alternatives of a list disagree.

    The disagreement is the one ``alternate_label_set_disagrees`` flags: the selected list
    against every other candidate of its entity, on the cells both carry (same polymer
    position and atom), counting a cell where the two differ by more than DUP_VALUE_EPS;
    one |difference| per (alternate, cell). The flag is re-derived and must match.
    """
    m = g.read(data / "fetch_manifest.json")
    t = g.read(data / "label_set_selection.csv")
    manifest = json.loads(m.read_text())
    s = pd.read_csv(t, usecols=_SELECTION_COLUMNS, dtype=str, keep_default_na=False)
    for col in ("selected", "alternate_label_set_disagrees"):
        s[col] = s[col].map({"True": True, "False": False})
        if s[col].isna().any():
            raise SystemExit(f"{t}: unparsed boolean in {col}")
    s["n_conflict_cells"] = s["n_conflict_cells"].astype(int)
    sel = s[s["selected"]]
    if sel["logical_entity_id"].duplicated().any():
        raise SystemExit(f"{t}: more than one selected list for an entity")

    differences, flagged, mismatched = [], set(), 0
    multi = s[s.groupby("logical_entity_id")["label_set_id"].transform("size") > 1]
    for _entity, group in multi.groupby("logical_entity_id", sort=True):
        chosen = group[group["selected"]]
        if chosen.empty:
            continue
        chosen_cells = _cells(chosen["candidate_payload_json"].iloc[0])
        for alt in group[~group["selected"]].itertuples(index=False):
            disagrees = False
            for key, value in _cells(alt.candidate_payload_json).items():
                if key in chosen_cells and abs(value - chosen_cells[key]) > DUP_VALUE_EPS:
                    disagrees = True
                    differences.append(abs(value - chosen_cells[key]))
            mismatched += disagrees != alt.alternate_label_set_disagrees
            if disagrees:
                flagged.add(chosen["label_set_id"].iloc[0])
    stored = set(sel.loc[sel["alternate_label_set_disagrees"], "label_set_id"])
    if flagged != stored or mismatched:
        raise SystemExit(f"{t}: the recomputed disagreement does not reproduce "
                         "alternate_label_set_disagrees")

    conflicted = sel[sel["n_conflict_cells"] > 0]
    n_dep = int(manifest["n_api_ids"])
    n_with_candidate = int(s["bmrb_id"].nunique())
    diff = np.asarray(differences)
    g.values.update({
        "depositions": entry(n_dep, "n_api_ids"),
        "depositions_with_a_candidate_label_set": entry(
            n_with_candidate, "distinct bmrb_id over all rows"),
        "depositions_without_usable_shifts": entry(
            n_dep - n_with_candidate,
            "n_api_ids minus distinct bmrb_id of label_set_selection.csv"),
        "selected_label_sets": entry(len(sel), "rows with selected == True"),
        "selected_label_sets_in_manifest": entry(int(manifest["n_logical_entities"]),
                                                 "n_logical_entities"),
        "selected_distinct_shift_lists": entry(
            len(sel[["bmrb_id", "shift_list_id"]].drop_duplicates()),
            "distinct (bmrb_id, shift_list_id) over selected rows"),
        "selected_depositions": entry(int(sel["bmrb_id"].nunique()),
                                      "distinct bmrb_id over selected rows"),
        "selected_with_a_disagreeing_alternate": entry(
            len(stored), "selected rows with alternate_label_set_disagrees == True"),
        "selected_lists_with_conflicting_cells": entry(
            len(conflicted), "selected rows with n_conflict_cells > 0"),
        "conflicting_cells_in_selected_lists": entry(
            int(conflicted["n_conflict_cells"].sum()),
            "sum of n_conflict_cells over selected rows"),
        "alternate_disagreement_median_ppm": entry(
            round(float(np.median(diff)), 6),
            "median |selected - alternate| over the shared cells that disagree "
            "(candidate_payload_json observed_cells), one per (alternate, cell)"),
        "alternate_disagreement_p95_ppm": entry(
            round(float(np.percentile(diff, 95)), 6),
            "95th percentile (numpy linear) of the same differences"),
        "alternate_disagreement_cells": entry(
            int(diff.size), "(alternate, cell) pairs the two numbers above pool"),
    })


# --------------------------------------------------------------------------------------
# Atom stage and fold map
# --------------------------------------------------------------------------------------

def atom_cloud(g: Group, data: Path) -> None:
    path = g.read(data / "atom_census.json")
    census = json.loads(path.read_text())
    for name, key in (("target_residues", "n_centers"),
                      ("largest_precap_cloud_atoms", "precap_max"),
                      ("structures", "n_structures_ok"),
                      ("structures_missing", "n_structures_missing"),
                      ("clouds_over_cap", "n_centers_over_cap"),
                      ("atom_cap", "chosen_m_atoms")):
        g.values[name] = entry(census[key], key)


def fold0_pin(g: Group, data: Path) -> None:
    path = g.read(data / "labelset_edges_provenance.json")
    key = "case_study_exclusion.pinned_from_training_folds"
    g.values["label_sets_moved_to_fold0"] = entry(
        json.loads(path.read_text())["case_study_exclusion"]["pinned_from_training_folds"], key)



def fence_effect(g: Group) -> None:
    """Methods: lifting the p100 fence raises backbone MAE by x-y % and multiplies backbone
    RMSE by x to y across the evaluated predictors (benchmarks.score_fence_effect)."""
    path = g.read(C.root("SI_FINAL_SCORES_DIR") / "fence_effect.json")
    pooled = json.loads(path.read_text())["summary"]["backbone_pooled"]
    for name, key in (("fence_lift_backbone_mae_change_pct", "mae_change_pct_range"),
                      ("fence_lift_backbone_rmse_ratio", "rmse_ratio_range")):
        lo, hi = pooled[key]
        g.values[name + "_min"] = entry(lo, f"summary.backbone_pooled.{key}[0]")
        g.values[name + "_max"] = entry(hi, f"summary.backbone_pooled.{key}[1]")

def table1_significance(g: Group) -> None:
    """Results: of Table 1's 18 (source, nucleus) cells, how many DeepShift leads over the
    runner-up significantly at 5 %, after Bonferroni with the bootstrap tail k/B, and with
    (k + 1)/(B + 1) in its place (benchmarks.runner_up_leads, as Figure 2's record)."""
    scored = C.root("SI_FINAL_SCORES_DIR")
    marginal_path = g.read(scored / "score_tier2_master.json")
    paired_path = g.read(scored / "score_tier2_master_paired.json")
    paired = json.loads(paired_path.read_text())
    tests = lead_tests(json.loads(marginal_path.read_text()), paired,
                       table1_competitors(paired))
    for name, key in (("alpha", "n_significant_alpha"),
                      ("bonferroni", "n_significant_bonferroni"),
                      ("bonferroni_plus_one", "n_significant_bonferroni_plus_one")):
        g.values[f"table1_leads_significant_{name}"] = entry(
            tests[key], f"benchmarks.runner_up_leads.lead_tests: {key} of "
            f"{tests['n_tests']} Table 1 cells")

# --------------------------------------------------------------------------------------
# Prediction intervals: the fold-0 exclusion and the fold-6 coverage range (Methods)
# --------------------------------------------------------------------------------------

BACKBONE_NUCLEI = ("N", "H", "CA", "CB", "C", "HA")


def intervals(g: Group) -> None:
    """Methods: label sets the fold-0 exclusion lists and removes from the calibration
    fit, recounted from the exclusion file and the fold-0 member dumps; the range of
    fold-6 nominal-90% coverage over the six backbone nuclei."""
    final = C.BUILD_ROOT / "final"
    cal_path = g.read(final / "interval_calibration.json")
    record = json.loads(cal_path.read_text())["fold0_exclusion"]
    listing = g.read(C.BUILD_ROOT / "data" / "fold0_calibration_exclusions.json")
    payload = json.loads(listing.read_text())
    listed = set(payload["label_sets"])
    for reason, n in payload["reasons"].items():
        g.values[f"interval_fold0_exclusion_reason_{reason}"] = entry(n, f"reasons.{reason}")
    dumps = [g.read(final / "intervals" / f"fold0_member{k}.npz") for k in range(1, 6)]
    predicted = set()
    for dump in dumps:
        with np.load(dump, allow_pickle=False) as z:
            predicted |= set(z["label_set_id"].astype(str).tolist())
    removed = len(listed & predicted)
    for name, mine, key in (("listed", len(listed), "label_sets_listed"),
                            ("removed", removed, "label_sets_removed")):
        if mine != record[key]:
            raise SystemExit(f"fold-0 exclusion {name}: recounted {mine}, "
                             f"{cal_path.name} records {record[key]}")
        g.values[f"interval_fold0_exclusion_label_sets_{name}"] = entry(
            mine,
            f"label sets in the list{' with a row in a fold-0 member dump' if name == 'removed' else ''}"
            f" (= {cal_path.name} fold0_exclusion.{key})")
    ev_path = g.read(final / "interval_evaluation_fold6_pf14.json")
    results = json.loads(ev_path.read_text())["results"]
    coverage = [results[n]["levels"]["0.90"]["realized_coverage"] for n in BACKBONE_NUCLEI]
    g.values["interval_fold6_nominal90_coverage_min"] = entry(
        min(coverage), "min over backbone nuclei of results.<n>.levels.0.90.realized_coverage")
    g.values["interval_fold6_nominal90_coverage_max"] = entry(
        max(coverage), "max over backbone nuclei of results.<n>.levels.0.90.realized_coverage")


# --------------------------------------------------------------------------------------
# Structure source on the cross-validation set (results) and on fold 6 (74 depositions)
# --------------------------------------------------------------------------------------

def source_matched(g: Group) -> None:
    """Results: how far ahead of crystal and of NMR structures AlphaFold
    structures put DeepShift on matched cross-validation cells, as a per cent of the
    experimental structure's RMSE, and whether every paired interval excludes zero."""
    path = g.read(C.BUILD_ROOT / "benchmark" / "scored" / "master_pf14" / "source_matched.json")
    nuclei = json.loads(path.read_text())["nuclei"]
    for exp in ("xtal", "nmr"):
        pct = [100 * (v["per_source"][exp]["rmse"] - v["per_source"]["af"]["rmse"])
               / v["per_source"][exp]["rmse"] for v in nuclei.values()]
        key = f"100 * (rmse[{exp}] - rmse[af]) / rmse[{exp}] over the six nuclei"
        g.values[f"source_matched_af_ahead_of_{exp}_pct_min"] = entry(min(pct), "min " + key)
        g.values[f"source_matched_af_ahead_of_{exp}_pct_max"] = entry(max(pct), "max " + key)
        g.values[f"source_matched_af_vs_{exp}_af_significant_nuclei"] = entry(
            sum(v["paired"][f"af_vs_{exp}"]["verdict"] == "af" for v in nuclei.values()),
            f"nuclei whose paired af_vs_{exp} verdict is af (interval excludes zero)")


def three_source(g: Group) -> None:
    """Results and Discussion: the test depositions with a crystal, an AlphaFold and a
    NMR structure, and per predictor on how many backbone nuclei the NMR
    conformer is least accurate (benchmarks.score_three_source)."""
    path = g.read(C.root("SI_FINAL_SCORES_DIR") / "three_source.json")
    report = json.loads(path.read_text())
    cohort = report["cohort"]
    g.values["three_source_label_sets"] = entry(
        cohort["n_label_sets_with_all_three_sources"],
        "cohort.n_label_sets_with_all_three_sources")
    g.values["three_source_bmrb_depositions"] = entry(
        cohort["n_bmrb_depositions_with_all_three_sources"],
        "cohort.n_bmrb_depositions_with_all_three_sources")
    for arm, s in report["summary"].items():
        g.values[f"three_source_{arm}_nmr_least_accurate_nuclei"] = entry(
            s["n_nuclei_nmr_least_accurate"],
            f"summary.{arm}.n_nuclei_nmr_least_accurate (of {s['n_nuclei_scored']})")
    sx = report["arms"]["shiftx2_sameprot"]["nuclei"]
    g.values["three_source_shiftx2_crystal_significantly_better_than_af_nuclei"] = entry(
        sum(v["paired"]["xtal_minus_af"]["verdict"] == "xtal" for v in sx.values()),
        "arms.shiftx2_sameprot: nuclei whose xtal_minus_af verdict is xtal")
    g.values["three_source_shiftx2_nmr_significantly_worse_than_af_nuclei"] = entry(
        sum(v["paired"]["af_minus_nmr"]["verdict"] == "af" for v in sx.values()),
        "arms.shiftx2_sameprot: nuclei whose af_minus_nmr verdict is af")


# --------------------------------------------------------------------------------------
# The anticorrelated channels, re-derived (Methods)
# --------------------------------------------------------------------------------------

def anticorrelated(g: Group) -> None:
    """Methods: the excluded channels, re-derived from the judges' outputs on the master
    set before the exclusion (benchmarks.derive_anticorrelated_channels)."""
    path = g.read(C.BUILD_ROOT / "benchmark" / "anticorrelated_rederived"
                  / "anticorrelated_channels_derived.json")
    d = json.loads(path.read_text())
    if not d["matches_record"]["reproduced"]:
        raise SystemExit(f"{path} does not reproduce the frozen channel list")
    g.values["anticorrelated_channels"] = entry(d["n_channels"], "n_channels")
    g.values["anticorrelated_cells"] = entry(d["n_cells"], "n_cells")
    for nucleus, n in d["channels_by_nucleus"].items():
        g.values[f"anticorrelated_channels_{nucleus}"] = entry(
            n, f"channels_by_nucleus.{nucleus}")
    g.values["anticorrelated_minimum_cells"] = entry(
        d["minimum_cells_per_channel"], "minimum_cells_per_channel")
    g.values["anticorrelated_judges"] = entry(len(d["judges"]), "len(judges)")


# --------------------------------------------------------------------------------------
# The nucleus vocabulary, the fold balance, the unfiltered extremes (introduction, Methods)
# --------------------------------------------------------------------------------------

def dataset_shares(g: Group, released: Path) -> None:
    """Results and Methods: how many BMRB label sets the dataset realizes, and the held-out
    fold's share of the label sets, residues and depositions outside fold 0, from the
    released Parquet tables."""
    seq = g.read(released / "deepshift_sequences.parquet")
    st = g.read(released / "deepshift_structures.parquet")
    sh = g.read(released / "deepshift_shifts.parquet")
    n = len(pd.read_parquet(seq, columns=["logical_entity_id"]))
    structures = pd.read_parquet(st, columns=["label_set_id", "source"])
    realized = structures["label_set_id"].nunique()
    experimental = structures.loc[structures["source"].isin(["nmr", "xtal"]),
                                  "label_set_id"].nunique()
    g.values["bmrb_label_sets"] = entry(n, "rows")
    g.values["label_sets_realized"] = entry(realized, "distinct label_set_id")
    g.values["label_sets_realized_pct"] = entry(
        100 * realized / n, "100 * label_sets_realized / bmrb_label_sets")
    g.values["label_sets_with_experimental_structure"] = entry(
        experimental, "distinct label_set_id with source nmr or xtal")
    g.values["label_sets_with_experimental_structure_pct"] = entry(
        100 * experimental / n,
        "100 * label_sets_with_experimental_structure / bmrb_label_sets")
    g.values["label_sets_realized_by_alphafold_only"] = entry(
        realized - experimental,
        "label_sets_realized - label_sets_with_experimental_structure")
    rows = pd.read_parquet(sh, columns=["label_set_id", "bmrb_id", "fold"]).dropna(
        subset=["fold"])
    held = rows[rows["fold"] == fold_stage.HOLDOUT_FOLD]
    pool = rows[rows["fold"] != fold_stage.EXCLUDED_FOLD]
    for name, count, how in (
            ("label_sets", lambda r: r["label_set_id"].nunique(), "distinct label_set_id"),
            ("residues", len, "rows"),
            ("depositions", lambda r: r["bmrb_id"].nunique(), "distinct bmrb_id")):
        h, p = int(count(held)), int(count(pool))
        g.values[f"holdout_{name}"] = entry(h, f"{how} with fold {fold_stage.HOLDOUT_FOLD}")
        g.values[f"holdout_pool_{name}"] = entry(
            p, f"{how} with a fold other than {fold_stage.EXCLUDED_FOLD}")
        g.values[f"holdout_{name}_pct"] = entry(
            100 * h / p, f"100 * holdout_{name} / holdout_pool_{name}")


def nucleus_vocabulary(g: Group) -> None:
    """Introduction: the nuclei predicted, of every standard backbone and side-chain shift
    name a predictor can label (deepshift.domain; HN is the amide-proton alias of H)."""
    from deepshift import domain
    standard = domain.SHIFT_ATOM_NAMES - {"HN"}
    g.values["nuclei_predicted"] = entry(len(domain.NUCLEI), "len(NUCLEI)")
    g.values["standard_shift_types"] = entry(
        len(standard), "len(SHIFT_ATOM_NAMES - {'HN'}): the 64 plus HZ1, ND1, NH1, NH2, NZ")


def fold_balance(g: Group, data: Path) -> None:
    """Methods: the worst fold imbalance over the dimensions the text names (shift-bearing
    residues and each source's protein count), from the fold stage's balance report."""
    path = g.read(data / "fold_balance_report.json")
    counts = json.loads(path.read_text())["per_count"]
    named = {k: counts[k]["max_dev_frac"] for k in ("residue", "source_af", "source_nmr",
                                                     "source_xtal")}
    g.values["fold_balance_worst_named_dimension"] = entry(
        max(named.values()), "max of per_count.{residue,source_af,source_nmr,"
        "source_xtal}.max_dev_frac")
    for k, v in named.items():
        g.values[f"fold_balance_{k}"] = entry(v, f"per_count.{k}.max_dev_frac")
    g.values["fold_balance_tolerance"] = entry(json.loads(path.read_text())["tol"], "tol")


def unfiltered_n_extremes(g: Group, data: Path) -> None:
    """Methods: the most extreme backbone N shifts the unfiltered matched data hold."""
    lo, hi = np.inf, -np.inf
    for source in ("af", "nmr", "xtal"):
        path = g.read(data / f"structure_data_{source}.csv")
        for chunk in pd.read_csv(path, usecols=["n_shift"], chunksize=500_000):
            col = pd.to_numeric(chunk["n_shift"], errors="coerce")
            lo, hi = min(lo, col.min()), max(hi, col.max())
    g.values["unfiltered_n_shift_min"] = entry(float(lo), "min n_shift")
    g.values["unfiltered_n_shift_max"] = entry(float(hi), "max n_shift")


def ucbshift_y_example(g: Group) -> None:
    """Supplement: UCBShift-Y's sequence search for 1AE3 (BMRB 2039), the self-retrieval
    example: hits returned, the top hit and its identity."""
    path = g.read(C.BUILD_ROOT / "campaign" / "ucby_example_2039" / "blast_refdb_1AE3_.json")
    rec = json.loads(path.read_text())
    top = rec["top_hit"]
    g.values["ucbshift_y_example_blast_hits"] = entry(rec["n_hits"], "n_hits")
    g.values["ucbshift_y_example_top_hit"] = entry(top["subject"], "top_hit.subject")
    g.values["ucbshift_y_example_identities"] = entry(top["identities"], "top_hit.identities")
    g.values["ucbshift_y_example_aligned_length"] = entry(
        top["aligned_length"], "top_hit.aligned_length")
    g.values["ucbshift_y_example_identity"] = entry(
        round(top["identities"] / top["aligned_length"], 3),
        "top_hit.identities / top_hit.aligned_length")


# --------------------------------------------------------------------------------------
# X-EISD pool sizes (results)
# --------------------------------------------------------------------------------------

def xeisd_pools(g: Group) -> None:
    """Results: the conformers in each drkN SH3 pool, counted as rows of DeepShift's
    shift table for that pool."""
    tables = C.root("DEEPSHIFT_XEISD_PROTOCOL").parent / "tables"
    for pool in ("RANDOM", "ENSEMBLE", "MIXED"):
        path = g.read(tables / f"drksh3_{pool}_CS_deepshift.txt")
        with path.open() as fh:
            n = sum(1 for line in fh if line.strip())
        g.values[f"xeisd_{pool.lower()}_pool_conformers"] = entry(n, "rows")


# --------------------------------------------------------------------------------------
# Rereferencing: residues per offset, and the source-paired refit
# --------------------------------------------------------------------------------------

def diagnostics_path(data: Path) -> Path:
    value = os.environ.get("SI_REREF_DIAGNOSTICS")
    return Path(value).expanduser().resolve() if value else data / "reref_diagnostics.json"


def rereferencing(g: Group, data: Path) -> None:
    """Both from the per-source fits ``scripts/reref_diagnostics.py`` records, which it
    writes only after every stored offset is reproduced from them.

    Residues per offset: for each label set with an offset, the in-window residues each
    applied nucleus was fitted over on the selected structure. Source-paired refit: per
    nucleus, over label sets where both sources are candidates and fit it, Pearson r of
    the two unrounded offsets and the median |difference|."""
    offsets_path = g.read(data / "reref_offsets.json")
    diag_path = g.read(diagnostics_path(data))
    stored = json.loads(offsets_path.read_text())
    per_ls = json.loads(diag_path.read_text())["per_label_set"]
    if set(per_ls) != set(stored):
        raise SystemExit(f"{diag_path} does not cover {offsets_path}")
    if not all("fits" in row for row in per_ls.values()):
        raise SystemExit(f"{diag_path} records no per-source fits; rerun "
                         "scripts/reref_diagnostics.py")

    residues = []
    for lsid, row in per_ls.items():
        if row["outcome"] != "offset":
            continue
        fitted = row["fits"][row["struct_used"].split(":", 1)[0]]
        residues += [fitted[n]["residues"] for n in stored[lsid]["offsets"] if n in fitted]
    key = ("per_label_set[outcome == offset].fits[source of struct_used][nucleus].residues, "
           "over the nuclei reref_offsets.json applies")
    g.values["offset_fits"] = entry(len(residues), f"count of {key}")
    g.values["offset_fit_median_residues"] = entry(float(np.median(residues)), f"median of {key}")

    for left, right in (("af", "xtal"), ("af", "nmr")):
        pairs = defaultdict(list)
        for row in per_ls.values():
            a, b = row["fits"].get(left), row["fits"].get(right)
            for n in REREF_APPLY_NUCS:
                if a and b and n in a and n in b:
                    pairs[n].append((a[n]["offset_ppm"], b[n]["offset_ppm"]))
        for n in REREF_APPLY_NUCS:
            x, y = np.asarray(pairs[n], dtype=float).reshape(-1, 2).T
            enough = len(x) > 1
            stem = f"refit_{left}_vs_{right}_{n}"
            where = f"per_label_set[*].fits.{{{left},{right}}}.{n}.offset_ppm"
            g.values[f"{stem}_label_sets"] = entry(len(x), f"label sets with {where}")
            g.values[f"{stem}_pearson_r"] = entry(
                float(np.corrcoef(x, y)[0, 1]) if enough else None,
                f"Pearson r of {where}")
            g.values[f"{stem}_median_abs_difference_ppm"] = entry(
                float(np.median(np.abs(x - y))) if enough else None,
                f"median |{left} - {right}| of {where}")


# --------------------------------------------------------------------------------------
# X-EISD: the primary endpoint per pool
# --------------------------------------------------------------------------------------

# analyze_pools.py beside the runs: columns 2 and 4-9 of a rescore are the seven non-CS
# log scores; per pool one primary, eight score and eight RMSE bootstraps, all from one
# generator, pools in the order the report was made.
XEISD_PRIMARY_COLUMNS = (2, 4, 5, 6, 7, 8, 9)
XEISD_POOL_ORDER = ("ENSEMBLE", "MIXED", "RANDOM")
XEISD_BOOTSTRAPS_PER_POOL = 17
XEISD_SEED, XEISD_SAMPLES, XEISD_RESTARTS = 20260728, 10000, 1000


def _bootstrap_ci(diff, rng) -> tuple[float, float]:
    est = np.empty(XEISD_SAMPLES)
    for start in range(0, XEISD_SAMPLES, 500):
        stop = min(start + 500, XEISD_SAMPLES)
        est[start:stop] = diff[rng.integers(0, len(diff), size=(stop - start, len(diff)))].mean(1)
    low, high = np.percentile(est, [2.5, 97.5])
    return float(low), float(high)


def xeisd(g: Group) -> None:
    """DeepShift minus SHIFTX2, summed over the seven non-CS log scores, mean over the
    1,000 restarts with its percentile bootstrap interval; must equal the run's report."""
    protocol = C.root("DEEPSHIFT_XEISD_PROTOCOL")
    report_path = g.read(protocol / "results.json")
    report = json.loads(report_path.read_text())["pools"]
    rng = np.random.default_rng(XEISD_SEED)
    for pool in XEISD_POOL_ORDER:
        arms = {}
        for arm in ("deepshift", "shiftx2"):
            path = g.read(protocol / "runs" / pool.lower() / arm / "rescore_all" / "results.csv")
            values = pd.read_csv(path, header=None).to_numpy(dtype=float)
            if values.shape != (XEISD_RESTARTS, 18):
                raise SystemExit(f"{path}: not {XEISD_RESTARTS} restarts")
            arms[arm] = values[:, XEISD_PRIMARY_COLUMNS].sum(axis=1)
        diff = arms["deepshift"] - arms["shiftx2"]
        low, high = _bootstrap_ci(diff, rng)
        for _ in range(XEISD_BOOTSTRAPS_PER_POOL - 1):
            _bootstrap_ci(np.zeros(XEISD_RESTARTS), rng)   # draws depend on n only
        printed = report[pool]["primary_non_cs_score"]
        mine = {"mean_improvement": float(diff.mean()), "ci95_low": low, "ci95_high": high}
        if any(not np.isclose(mine[k], printed[k], rtol=0, atol=1e-9) for k in mine):
            raise SystemExit(f"X-EISD {pool}: recomputed {mine} != {report_path} {printed}")
        for k, v in mine.items():
            g.values[f"xeisd_{pool}_{k}"] = entry(v, f"pools.{pool}.primary_non_cs_score.{k}")


# --------------------------------------------------------------------------------------
# Self-retrieval of UCBShift-Y and SHIFTY+ (supplement)
# --------------------------------------------------------------------------------------

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
THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
             "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
             "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
             "TYR": "Y", "VAL": "V"}


def ucbshift_y(g: Group) -> None:
    """UCBShift-Y in test mode on UCBShift's 200 test proteins: cells where both a Y
    prediction and the target exist (residue names agreeing; evaluate.py's placeholders
    H < 2, CB > 80, N < 60 blanked, and the cells UCBShift's paper excludes, as every table
    scoring this test set excludes them); copied = within 0.005 ppm of the target, ties included."""
    preds = C.root("SI_SELF_RETRIEVAL_UCBY") / "preds"
    targets = C.root("SI_GATE_CSPRED_TM") / "test_targets"
    record = g.read(UCB200_EXCLUSIONS)
    excluded = json.loads(record.read_text())
    cprime_chains = set(excluded["cprime_chains"]["chains"])
    excluded_cells = {(str(c), int(r), str(a)) for c, r, a in excluded["cells"]["cells"]}
    copied = total = proteins = removed = 0
    pred_files = sorted(preds.glob("*.csv"))
    if not pred_files:
        raise FileNotFoundError(preds)
    for pf in pred_files:
        tf = targets / pf.name
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
        removed += before - int(m[list(SELF_NUCLEI)].notna().sum().sum())
        proteins += 1
        for a in SELF_NUCLEI:
            t = pd.to_numeric(m[a], errors="coerce").to_numpy(float)
            y = pd.to_numeric(m[a + "_Y"], errors="coerce").to_numpy(float)
            ok = ~np.isnan(t) & ~np.isnan(y)
            copied += int(within_copy_threshold(y[ok], t[ok]).sum())
            total += int(ok.sum())
    if not total:
        # The test targets are UCBShift's own (third-party): absent, nothing is scored.
        raise FileNotFoundError(targets)
    key = "cells with |prediction_Y - target| <= 0.005 ppm (stored-decimal ties included)"
    g.values["self_retrieval_ucbshift_y_proteins"] = entry(proteins, "proteins scored")
    g.values["self_retrieval_ucbshift_y_copied"] = entry(copied, key)
    g.values["self_retrieval_ucbshift_y_cells"] = entry(total, "cells scored")
    g.values["self_retrieval_ucbshift_y_percent"] = entry(
        round(100 * copied / total, 4), f"100 * ({key}) / cells scored")
    g.values["self_retrieval_ucbshift_y_targets_removed_as_in_the_paper"] = entry(
        removed, "target cells the paper's exclusions blank")


def _shifty_target(g: Group, bmrb: int, mirror: Path, library: Path) -> pd.DataFrame:
    """One protein's deposited shifts: UCBShift's RefDB mirror, else SHIFTX2's copy."""
    files = sorted(f for f in mirror.iterdir() if re.match(rf"{bmrb}\.", f.name))
    if files:
        d = pd.read_csv(files[0], usecols=lambda c: c in ["RES_NUM", "RESNAME", *SELF_NUCLEI])
        d["aa1"] = d["RESNAME"].map(THREE2ONE)
        long = d.melt(id_vars=["RES_NUM", "aa1"], value_vars=[a for a in SELF_NUCLEI if a in d],
                      var_name="atom", value_name="tgt").dropna()
        long["tgt"] = pd.to_numeric(long["tgt"], errors="coerce")
        return long.dropna()
    cs = g.read(library / f"bmr{bmrb}.cs")
    rows = [(int(p[0]), p[1], p[2], float(p[3]))
            for p in (line.strip().split(",") for line in cs.read_text().splitlines()
                      if not line.startswith("#"))
            if len(p) == 4 and p[2] in SELF_NUCLEI]
    return pd.DataFrame(rows, columns=["RES_NUM", "aa1", "atom", "tgt"])


def _residue_map(query: pd.DataFrame, target: pd.DataFrame) -> dict[int, int]:
    """SHIFTY+ residue number -> the deposition's residue number, by a global alignment of
    the two residue sequences in residue order; only aligned pairs of the same residue type
    are kept. The two numberings can differ by any offset, and by insertions."""
    from Bio import Align
    q = query.drop_duplicates("rn").sort_values("rn")
    t = target.drop_duplicates("RES_NUM").sort_values("RES_NUM")
    qs = "".join(str(a) for a in q["aa"])
    ts = "".join(a if isinstance(a, str) and len(a) == 1 else "X" for a in t["aa1"])
    if not qs or not ts:
        return {}
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score, aligner.mismatch_score = 2.0, -1.0
    aligner.open_gap_score, aligner.extend_gap_score = -4.0, -0.5
    alignment = next(iter(aligner.align(qs, ts)))
    qn, tn = q["rn"].to_numpy(), t["RES_NUM"].to_numpy()
    mapping = {}
    for (q0, q1), (t0, t1) in zip(*alignment.aligned):
        for i, j in zip(range(q0, q1), range(t0, t1)):
            if qs[i] == ts[j]:
                mapping[int(qn[i])] = int(tn[j])
    return mapping


def shifty61(g: Group) -> None:
    """SHIFTY+ on SHIFTX2's 61 test proteins against each protein's deposition, with and
    without references matching the query exactly: SHIFTY+'s residues are matched to the
    deposition's by aligning the two residue sequences (_residue_map); an empty output is
    a protein with no cells."""
    roster_path = g.read(C.root("DEEPSHIFT_UCBSHIFT_LEAKAGE") / "shiftx2_test61.csv")
    with roster_path.open() as fh:
        roster = {r["pdb"]: int(r["bmrb"]) for r in csv.DictReader(fh)}
    mirror, library = C.root("DEEPSHIFT_REFDB_SHIFTS"), C.root("DEEPSHIFT_REFDB_LIBRARY")
    targets = {pid: _shifty_target(g, bmrb, mirror, library) for pid, bmrb in roster.items()}
    run = C.root("SI_SELF_RETRIEVAL_SHIFTY61")
    for name, directory in (("shifty61", "out"), ("shifty61_noexact", "out_noexact")):
        copied = total = with_cells = 0
        for pid in sorted(roster):
            sf = g.read(run / directory / f"{pid}.shifty")
            if not sf.stat().st_size:
                continue
            sh = pd.read_csv(sf, header=None, names=["rn", "aa", "atom", "val", "iden"])
            sh = sh[sh["atom"].isin(SELF_NUCLEI)].copy()
            sh["val"] = pd.to_numeric(sh["val"], errors="coerce")
            mapping = _residue_map(sh, targets[pid])
            best = pd.merge(sh.assign(RES_NUM=sh["rn"].map(mapping)).dropna(subset=["RES_NUM"])
                            .astype({"RES_NUM": int}), targets[pid], on=["RES_NUM", "atom"])
            best = best[best["aa"] == best["aa1"]]
            copied += int(within_copy_threshold(best["val"], best["tgt"]).sum())
            total += len(best)
            with_cells += bool(len(best))
        if not total:
            raise FileNotFoundError(run / directory)
        key = "cells with |SHIFTY+ - deposited| <= 0.005 ppm (stored-decimal ties included)"
        g.values[f"self_retrieval_{name}_proteins"] = entry(
            len(roster), "proteins in SHIFTX2's test set, each run")
        g.values[f"self_retrieval_{name}_proteins_with_cells"] = entry(
            with_cells, "proteins contributing at least one scored cell")
        g.values[f"self_retrieval_{name}_copied"] = entry(copied, key)
        g.values[f"self_retrieval_{name}_cells"] = entry(total, "cells scored")
        g.values[f"self_retrieval_{name}_percent"] = entry(
            round(100 * copied / total, 4), f"100 * ({key}) / cells scored")


# --------------------------------------------------------------------------------------
# Training environments of the five members
# --------------------------------------------------------------------------------------

# The cluster record of the members trained on Hopper: their Slurm jobs, node GPUs and
# the conda environment's versions, measured on the cluster.
CLUSTER_ENVIRONMENT = Path("final") / "members45_environment_20261007.json"


def _cluster_member(g: Group, run: Path, fold: int):
    """(record path, record, member block) for a member the cluster record covers; None
    when it covers none."""
    path = run / CLUSTER_ENVIRONMENT
    if not path.is_file():
        return None
    record = json.loads(g.read(path).read_text())
    member = record["members"].get(str(fold))
    if member is None:
        return None
    return path, record, member


def environments(g: Group, run: Path) -> None:
    """Python from each member's training record where it names one; torch, CUDA, numpy
    and GPU from the preflight record of the node it was assigned to
    (logs/assigned_fold<k> and logs/assigned_host_<node> carry the same assignment time),
    or for a member trained on the cluster from the cluster record."""
    stamps = {}
    for path in sorted((run / "logs").glob("assigned_host_*")):
        stamps[g.read(path).read_text().strip()] = path.name.removeprefix("assigned_host_")
    preflight = {p.parent.name.lower(): p for p in sorted((run / "preflight").glob("*/environment.json"))}
    for fold in range(1, 6):
        record_path = g.read(run / "final" / f"fold{fold}.json")
        manifest = json.loads(record_path.read_text())["manifest"]
        prefix = f"member{fold}_"
        if "python" in manifest:
            g.values[prefix + "python"] = entry(manifest["python"].split()[0],
                                                "manifest.python (first token)")
        if "command_argv" in manifest:
            g.values[prefix + "interpreter"] = entry(manifest["command_argv"][0],
                                                     "manifest.command_argv[0]")
        assigned = run / "logs" / f"assigned_fold{fold}"
        host = None
        if assigned.is_file():
            host = stamps.get(g.read(assigned).read_text().strip())
        env_path = preflight.get(host or "")
        if env_path is None:
            cluster = _cluster_member(g, run, fold)
            if cluster is None:
                g.values[prefix + "environment"] = entry(
                    None, "no preflight or cluster record: torch, CUDA, numpy "
                          "and GPU are not recorded in any run file")
                continue
            _path, record, member = cluster
            env = record["environment"]
            g.values[prefix + "node"] = entry(member["node"], f"members.{fold}.node")
            g.values[prefix + "slurm_job"] = entry(member["slurm_job"],
                                                   f"members.{fold}.slurm_job")
            for key, field in (("torch", "torch"), ("cuda", "torch_cuda"), ("numpy", "numpy")):
                g.values[prefix + key] = entry(env[field], f"environment.{field}")
            g.values[prefix + "device"] = entry(record["node_gres"][member["node"]],
                                                f"node_gres.{member['node']}")
            continue
        env = json.loads(g.read(env_path).read_text())
        g.values[prefix + "node"] = entry(env["host"], "host")
        for key in ("python", "torch", "cuda", "numpy", "device"):
            g.values[prefix + key + ("_preflight" if key == "python" else "")] = entry(
                env[key], key)


# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-missing", action="store_true",
                    help="record a group whose files are absent instead of stopping")
    args = ap.parse_args(argv)

    data = C.DATA
    groups = {
        "shift_lists": lambda g: shift_lists(g, data),
        "atom_cloud": lambda g: atom_cloud(g, data),
        "fold0_pin": lambda g: fold0_pin(g, data),
        "fence_effect": fence_effect,
        "table1_significance": table1_significance,
        "rereferencing": lambda g: rereferencing(g, data),
        "xeisd": xeisd,
        "xeisd_pools": xeisd_pools,
        "nucleus_vocabulary": nucleus_vocabulary,
        "dataset_shares": lambda g: dataset_shares(g, RELEASED),
        "fold_balance": lambda g: fold_balance(g, data),
        "unfiltered_n_extremes": lambda g: unfiltered_n_extremes(g, data),
        "ucbshift_y_example": ucbshift_y_example,
        "intervals": intervals,
        "source_matched": source_matched,
        "three_source": three_source,
        "anticorrelated": anticorrelated,
        "self_retrieval_ucbshift_y": ucbshift_y,
        "self_retrieval_shifty61": shifty61,
        "environments": lambda g: environments(g, C.BUILD_ROOT),
    }
    values, missing = {}, {}
    for name, compute in groups.items():
        g = Group()
        try:
            compute(g)
        except FileNotFoundError as exc:
            if not args.allow_missing:
                raise
            missing[name] = f"not recomputed: {exc.filename or exc} is absent"
            print(f"  {name}: {missing[name]}")
            continue
        values.update(g.values)
    C.OUT.mkdir(parents=True, exist_ok=True)
    out = C.OUT / "text_numbers.json"
    out.write_text(json.dumps({**values, **({"not_recomputed": missing} if missing else {})},
                              indent=1, sort_keys=True, default=str) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
