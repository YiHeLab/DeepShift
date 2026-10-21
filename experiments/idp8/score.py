#!/usr/bin/env python3
"""Score DeepShift on the Nagy IDP8 ensembles against their published Table III.

Nagy et al. (Appl Spectrosc 2024, doi:10.1177/00037028241239977) ensemble-average
SPARTA+ v2.6 predictions over each deposited ensemble and report the RMSD against
BMRB shifts. The identical statistic is recomputed with DeepShift in place of SPARTA+,
changing nothing else: same ensembles, same BMRB references, no re-referencing, uniform
averaging over deposited models.

Residues are matched by SEQUENCE, not by residue number. PED numbering does not
always agree with BMRB numbering (PED00160/Sic1 starts at -1), and a silent
off-by-one would corrupt every RMSD without failing anything. The alignment is
required to agree on residue identity at every matched position.

Reads ``<data>/manifest.json`` and the BMRB files it names (``fetch.py``) and the
``<predictions>/<label>.parquet`` files ``run_inference.py`` wrote, and writes one JSON
list with a row per ensemble. The producers read that file as ``scores_audited.json``,
beside this script, whose ``PUBLISHED`` table (Nagy et al., Table III, CA / CB / C' RMSD
in ppm, None where the paper reports none) ``si_build/build_idp8_roster.py`` cross-checks.

    python score.py --data <idp8 data directory> --predictions <predictions directory> \\
        --out <scores_audited.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pynmrstar
from Bio import Align
from Bio.SeqUtils import seq1

# Nagy Table III, CA / CB / C' RMSD in ppm; None where the paper reports none.
PUBLISHED = {
    "asyn-A": (0.36, 0.66, 0.66),
    "mevn-A": (0.28, 0.37, 0.40),
    "sic1-A": (1.30, 0.49, None),
    "tk18-A": (0.56, 1.01, 0.52),
    "mevn-B": (0.34, 0.38, 0.52),
    "cbpn-B": (0.69, 0.38, 0.62),
    "p53t-B": (0.36, 0.34, 0.53),
    "actr-B": (0.35, 0.35, 0.46),
    "rsp8-B": (0.41, None, 0.60),
    "mevn-C": (0.34, 0.32, 0.37),
    "cbpn-C": (0.67, 0.40, 0.62),
    "p53t-C": (0.28, 0.32, 0.47),
    "actr-C": (0.28, 0.32, 0.39),
    "rsp8-C": (0.57, None, 0.64),
}

# BMRB atom name -> our column
NUCLEI = {"CA": "ca_shift", "CB": "cb_shift", "C": "c_shift"}


def read_bmrb(path: Path) -> pd.DataFrame:
    """Backbone CA/CB/C' shifts from an NMR-STAR file, one row per (seq, atom)."""
    entry = pynmrstar.Entry.from_file(str(path))
    rows = []
    for loop in entry.get_loops_by_category("Atom_chem_shift"):
        tags = ["Comp_index_ID", "Comp_ID", "Atom_ID", "Val"]
        for seq, comp, atom, value in loop.get_tag(tags):
            if atom not in NUCLEI or value in (".", "?", None):
                continue
            rows.append({"seq": int(seq), "resname": comp,
                         "atom": atom, "obs": float(value)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"{path}: no CA/CB/C' shifts")
    # A BMRB entry can carry several assignment sets; keep the first value per cell.
    return frame.drop_duplicates(subset=["seq", "atom"], keep="first")


def sequence(frame: pd.DataFrame, seq_col: str, name_col: str) -> tuple[str, list[int]]:
    """One-letter sequence and the residue ids behind it, in ascending order."""
    unique = frame[[seq_col, name_col]].drop_duplicates().sort_values(seq_col)
    letters, ids = [], []
    for _, row in unique.iterrows():
        try:
            letters.append(seq1(str(row[name_col]).upper().strip()))
        except Exception:
            letters.append("X")
        ids.append(int(row[seq_col]))
    return "".join(letters), ids


def align(pred: pd.DataFrame, obs: pd.DataFrame) -> dict[int, int]:
    """Map predicted residue id -> observed residue id via global alignment."""
    pred_seq, pred_ids = sequence(pred, "residue_id", "residue_name")
    obs_seq, obs_ids = sequence(obs, "seq", "resname")
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score, aligner.extend_gap_score = -10, -0.5
    aligner.match_score, aligner.mismatch_score = 2, -5
    alignment = aligner.align(pred_seq, obs_seq)[0]
    mapping = {}
    for (p0, p1), (o0, _) in zip(*alignment.aligned):
        for offset in range(p1 - p0):
            if pred_seq[p0 + offset] == obs_seq[o0 + offset]:
                mapping[pred_ids[p0 + offset]] = obs_ids[o0 + offset]
    if len(mapping) < 0.5 * min(len(pred_ids), len(obs_ids)):
        raise ValueError(
            f"alignment matched only {len(mapping)} of "
            f"{min(len(pred_ids), len(obs_ids))} residues"
        )
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True,
                        help="directory holding manifest.json (written by fetch.py)")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.data / "manifest.json").read_text())
    shifts = {s["protein"]: read_bmrb(args.data / s["path"]) for s in manifest["shifts"]}

    results = []
    for entry in manifest["ensembles"]:
        label, protein = entry["label"], entry["protein"]
        path = args.predictions / f"{label}.parquet"
        if not path.exists():
            print(f"{label}: MISSING predictions, skipping")
            continue
        pred = pd.read_parquet(path)
        obs = shifts[protein]
        mapping = align(pred, obs)

        # Uniform ensemble average over deposited models, exactly as Nagy averages.
        averaged = pred.groupby(["residue_id", "residue_name"], as_index=False).mean(
            numeric_only=True
        )
        averaged["obs_seq"] = averaged.residue_id.map(mapping)

        row = {
            "label": label,
            "protein": protein,
            "group": entry["group"],
            "members": pred.members.iloc[0],
            "n_models": int(pred.model_index.nunique()),
            "n_aligned_residues": len(mapping),
        }
        for atom, column in NUCLEI.items():
            merged = averaged.merge(
                obs[obs.atom == atom][["seq", "obs"]],
                left_on="obs_seq", right_on="seq", how="inner",
            ).dropna(subset=[column, "obs"])
            if merged.empty:
                row[f"{atom}_n"], row[f"{atom}_rmsd"] = 0, None
                continue
            error = merged[column].to_numpy() - merged["obs"].to_numpy()
            row[f"{atom}_n"] = len(error)
            row[f"{atom}_rmsd"] = float(np.sqrt(np.mean(error**2)))
            row[f"{atom}_mae"] = float(np.mean(np.abs(error)))
            row[f"{atom}_bias"] = float(np.mean(error))
        published = PUBLISHED.get(label)
        if published:
            for atom, value in zip(NUCLEI, published):
                row[f"{atom}_published"] = value
                if value is not None and row.get(f"{atom}_rmsd") is not None:
                    row[f"{atom}_delta"] = row[f"{atom}_rmsd"] - value
        results.append(row)

    frame = pd.DataFrame(results)
    columns = ["label", "group", "members", "n_models", "n_aligned_residues"]
    for atom in NUCLEI:
        columns += [f"{atom}_n", f"{atom}_rmsd", f"{atom}_published", f"{atom}_delta"]
    print(frame[[c for c in columns if c in frame]].to_string(index=False))

    print("\n--- means over scored ensembles (DeepShift vs Nagy SPARTA+) ---")
    for atom in NUCLEI:
        block = frame.dropna(subset=[f"{atom}_rmsd", f"{atom}_published"])
        if block.empty:
            continue
        print(
            f"  {atom:2s}  DeepShift {block[f'{atom}_rmsd'].mean():.3f}  "
            f"SPARTA+ {block[f'{atom}_published'].mean():.3f}  "
            f"delta {block[f'{atom}_delta'].mean():+.3f} ppm  "
            f"(wins {int((block[f'{atom}_delta'] < 0).sum())}/{len(block)})"
        )
    print(
        f"\n  coverage: {len(frame)}/{len(PUBLISHED)} IDP8 ensemble rows "
        "(mevn-A/B/C and rsp8-B/C require privately supplied shifts)"
    )

    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
