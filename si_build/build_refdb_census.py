#!/usr/bin/env python3
"""Agreement between this run's rereferencing offsets and RefDB's published ones.

RefDB publishes, per BMRB deposition, the offset it added to each nucleus and the PDB
structure it fitted that offset against. This census compares those two quantities with
ours over the depositions both databases hold.

UNIT. One comparable cell per (our label set, nucleus), kept only where RefDB applied a
value. Where we declined to rereference, our value is 0.0 -- the offset actually baked into
the labels -- so the difference is the real divergence between the two label sets rather
than a missing entry.

SPLIT, which is the point. RefDB fitted its offset on a structure it chose and we fit ours
on a structure we chose, and the two selectors do not always agree. Where the structure is
the same the two procedures agree closely; where it differs the disagreement is mostly the
structure, not the arithmetic. An AlphaFold pick can never be RefDB's structure, so the
comparison is gated on the source, not on how an accession is spelled.

GRAINS. Per offset -- one observation per (label set, nucleus) -- and per label, the same
cells weighted by the number of observed shifts of that nucleus in that label set, which is
what "for all rereferenced labels" means. A BMRB-grain restatement is carried too.

    python si_build/build_refdb_census.py --out <dir>/refdb_census.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import si_common

from deepshift.data.reref_utils import HDR2SX, NUCS

#: The four nuclei RefDB publishes an applied offset for and we rereference.
NUCLEI = ("CA", "CB", "CO", "N")
#: The column of the shift table each nucleus's labels are counted in.
NUC_COLUMN = {nucleus: f"{HDR2SX[nucleus].lower()}_shift" for nucleus in NUCLEI}
#: Where RefDB's own correction reports live: a tar of them, or a directory.
CORRECTIONS = si_common.root("DEEPSHIFT_REFDB_CORRECTIONS")
# Our offsets sit on a 0.05 ppm grid and RefDB's on a 0.01 grid, so many differences are
# exactly 0.05 or 0.10 ppm, and float subtraction leaves them a few 1e-17 either side. A
# difference counts as within a threshold up to this much above it, far below the grid.
WITHIN_TOLERANCE = 1e-9
STRUCTURE_RECORD = re.compile(r"Corrected using PDB structure:\s*(\S+)")
REPORT_NAME = re.compile(r"bmr(\d+)\.str\.corr")


def parse_block(text: str, marker: str) -> dict:
    """The six-nucleus row under one heading of a RefDB correction report."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if marker not in line:
            continue
        for candidate in lines[index + 1:index + 6]:
            tokens = candidate.lstrip("#").split()
            if len(tokens) != len(NUCS) or any(token in NUCS for token in tokens):
                continue
            out = {}
            for nucleus, token in zip(NUCS, tokens):
                found = re.search(r"-?\d+\.\d+", token)
                out[nucleus] = float(found.group()) if found else None
            return out
    return {}


def parse_report(text: str) -> dict:
    """The structure RefDB corrected against, and the offsets it applied."""
    found = STRUCTURE_RECORD.search(text)
    return {"struct": found.group(1) if found else None,
            "applied": parse_block(text, "have been added to the original")}


def load_refdb(path: Path) -> dict:
    """``BMRB -> parsed report`` from a tar of RefDB's reports, or a directory of them."""
    reports = {}
    if path.is_dir():
        for report in sorted(path.rglob("bmr*.str.corr")):
            name = REPORT_NAME.search(report.name)
            if name:
                reports[name.group(1)] = parse_report(report.read_text(errors="ignore"))
        return reports
    with tarfile.open(path) as archive:
        for member in archive:
            name = REPORT_NAME.search(member.name)
            if not name:
                continue
            reports[name.group(1)] = parse_report(
                archive.extractfile(member).read().decode("utf-8", "ignore"))
    return reports


def load_structures(data: Path):
    """``label set -> {source: accession, chain}`` and ``label set -> BMRB``.

    The label-set-to-BMRB map comes from the entity catalog, not the build logs: a label
    set with no materializable structure has no build-log row at all, and dropping those
    would delete exactly the label sets we could not rereference -- the ones that most
    need to be in the census.
    """
    per_label = defaultdict(dict)
    for source in ("af", "nmr", "xtal"):
        with open(data / f"structure_build_log_{source}.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") != "success":
                    continue
                per_label[row["label_set_id"]][source] = {
                    "accession": (row.get("accession") or "").strip(),
                    "chain": (row.get("chain_id") or "").strip()}
    bmrb_of_label = {}
    with open(data / "entity_catalog.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            bmrb_of_label[row["label_set_id"]] = (row.get("bmrb_id") or "").strip()
    return per_label, bmrb_of_label


def label_counts(shifts: Path, wanted: set) -> Counter:
    """``(label set, nucleus) -> observed shifts``, from the table the offsets apply to."""
    import pandas as pd

    counts: Counter = Counter()
    columns = ["label_set_id", *NUC_COLUMN.values()]
    for chunk in pd.read_csv(shifts, usecols=columns, chunksize=400_000,
                             dtype={"label_set_id": str}):
        chunk = chunk[chunk["label_set_id"].isin(wanted)]
        if chunk.empty:
            continue
        for nucleus, column in NUC_COLUMN.items():
            present = chunk.loc[chunk[column].notna(), "label_set_id"]
            for label_set, count in present.value_counts().items():
                counts[(label_set, nucleus)] += int(count)
    return counts


def pdb_id(label) -> str | None:
    """The 4-character PDB id in our accession or a RefDB header token ('1VEXA')."""
    if not label:
        return None
    text = Path(str(label).strip().upper()).name
    if text.endswith(".PDB"):
        text = text[:-4]
    return text[:4] if len(text) >= 4 else None


def describe(deltas, weights=None) -> dict:
    """The distribution of |our offset - RefDB's|, weighted by labels when asked."""
    values = np.asarray(deltas, dtype=float)
    if values.size == 0:
        return {"n": 0}
    w = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(values)
    sorted_values, cumulative = values[order], np.cumsum(w[order])

    def quantile(fraction):
        return float(sorted_values[np.searchsorted(cumulative, fraction * cumulative[-1])])

    return {"n": round(float(w.sum())) if weights is not None else values.size,
            "n_cells": values.size, "median": quantile(0.5),
            "mean": float(np.average(values, weights=w)),
            "within_0p01": float(w[values <= 0.01 + WITHIN_TOLERANCE].sum() / w.sum()),
            "within_0p05": float(w[values <= 0.05 + WITHIN_TOLERANCE].sum() / w.sum()),
            "within_0p10": float(w[values <= 0.10 + WITHIN_TOLERANCE].sum() / w.sum()),
            "p90": quantile(0.90), "p95": quantile(0.95), "p99": quantile(0.99),
            "max": float(values.max())}


def comparable_cells(offsets, reports, structures, bmrb_of_label) -> list[dict]:
    """One row per (label set, nucleus) RefDB applied a value for."""
    rows = []
    for label_set, record in offsets.items():
        bmrb = bmrb_of_label.get(label_set)
        if bmrb is None or bmrb not in reports:
            continue
        report = reports[bmrb]
        used = record.get("struct_used")                  # 'source:chain', or None
        our_source = our_chain = our_accession = None
        if used and ":" in used:
            our_source, our_chain = used.split(":", 1)
            found = structures.get(label_set, {}).get(our_source)
            if found:
                our_accession = found["accession"]
        # An AlphaFold pick is a model keyed by UniProt accession, never a deposition, so
        # it can never be the structure RefDB used.
        theirs = pdb_id(report["struct"])
        ours = None if our_source == "af" else pdb_id(our_accession)
        same = bool(theirs and ours and theirs == ours)
        their_chain = (str(report["struct"]).strip().upper()[4:5]
                       if report["struct"] and len(str(report["struct"]).strip()) >= 5
                       else None)
        applied = record.get("offsets") or {}
        for nucleus in NUCLEI:
            published = report["applied"].get(nucleus)
            if published is None:
                continue                                  # RefDB declined: not like-for-like
            our_value = float(applied.get(nucleus, 0.0))
            rows.append({
                "label_set_id": label_set, "bmrb": bmrb, "nuc": nucleus,
                "our": our_value, "refdb": float(published),
                "delta": abs(our_value - float(published)),
                "same_struct": same,
                "same_struct_strict": bool(same and their_chain and our_chain
                                           and their_chain == our_chain.upper()),
                "no_structure": our_accession is None,
                "our_struct": our_accession, "our_source": our_source,
                "refdb_struct": report["struct"], "method": record.get("method")})
    return rows


def blocks(rows, weighted: bool) -> dict:
    out = {}
    for nucleus in NUCLEI:
        selected = [row for row in rows if row["nuc"] == nucleus]
        deltas = [row["delta"] for row in selected]
        weights = [row["n_labels"] for row in selected] if weighted else None
        if weighted:
            kept = [(d, w) for d, w in zip(deltas, weights) if w > 0]
            deltas = [d for d, _w in kept]
            weights = [w for _d, w in kept]
        out[nucleus] = describe(deltas, weights)
    return out


def rounding_grid(rows) -> dict:
    """How much of the residual is the two grids rather than disagreement.

    Our offsets sit on a 0.05 ppm grid and RefDB publishes on 0.01, so a quantization
    floor is baked in. Snapping RefDB to our grid separates the two.
    """
    ours = np.asarray([row["our"] for row in rows])
    theirs = np.asarray([row["refdb"] for row in rows])
    nucleus = np.asarray([row["nuc"] for row in rows])
    same = np.asarray([row["same_struct"] for row in rows])
    snapped = np.round(theirs / 0.05) * 0.05
    grid = {"our_offsets_all_on_0p05_grid":
            bool(np.all(np.abs(ours / 0.05 - np.round(ours / 0.05)) < 1e-9)),
            "refdb_offsets_all_on_0p01_grid":
            bool(np.all(np.abs(theirs / 0.01 - np.round(theirs / 0.01)) < 1e-9))}
    for name in NUCLEI:
        inside = (nucleus == name) & same
        raw = np.abs(ours[inside] - theirs[inside])
        matched = np.abs(ours[inside] - snapped[inside])
        grid[name] = {"same_struct_n": int(inside.sum()),
                      "median_raw": float(np.median(raw)) if raw.size else None,
                      "median_grid_matched": float(np.median(matched)) if matched.size else None,
                      "within_0p10_raw":
                          float((raw <= 0.10 + WITHIN_TOLERANCE).mean()) if raw.size else None,
                      "within_0p10_grid_matched":
                          float((matched <= 0.10 + WITHIN_TOLERANCE).mean())
                          if matched.size else None,
                      "exactly_equal_on_0p05_grid":
                          float((matched < 1e-9).mean()) if matched.size else None}
    return grid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corrections", type=Path, default=CORRECTIONS,
                    help="RefDB's own .str.corr reports: a tar of them, or a directory")
    ap.add_argument("--out", required=True)
    ns = ap.parse_args(argv)

    data = si_common.DATA
    reports = load_refdb(ns.corrections)
    offsets = json.loads((data / "reref_offsets.json").read_text())
    structures, bmrb_of_label = load_structures(data)
    print(f"RefDB reports: {len(reports):,}; our label sets: {len(offsets):,}")

    rows = comparable_cells(offsets, reports, structures, bmrb_of_label)
    overlap = {row["label_set_id"] for row in rows}
    counts = label_counts(data / "chemical_shifts.csv", overlap)
    for row in rows:
        row["n_labels"] = int(counts.get((row["label_set_id"], row["nuc"]), 0))
    print(f"comparable cells: {len(rows):,} over {len(overlap):,} label sets")

    same = [row for row in rows if row["same_struct"]]
    different = [row for row in rows if not row["same_struct"]]
    without = [row for row in rows if row["no_structure"]]
    census = {}
    for grain, weighted in (("per_offset", False), ("per_label", True)):
        census[grain] = {
            "all": blocks(rows, weighted),
            "same_structure": blocks(same, weighted),
            "different_structure": blocks(different, weighted),
            "no_structure_chosen_subset_of_different": blocks(without, weighted),
            "same_structure_strict_pdb_and_chain":
                blocks([row for row in rows if row["same_struct_strict"]], weighted)}

    # One cell per (BMRB, nucleus): where a BMRB has several label sets, the one carrying
    # the most observed labels for that nucleus wins.
    dominant: dict[tuple, dict] = {}
    for row in rows:
        key = (row["bmrb"], row["nuc"])
        current = dominant.get(key)
        if current is None or ((row["n_labels"], row["label_set_id"])
                               > (current["n_labels"], current["label_set_id"])):
            dominant[key] = row
    by_bmrb = list(dominant.values())
    census["per_offset_bmrb_grain"] = {
        "rule": "one cell per (BMRB, nucleus); the label set with the most observed "
                "labels wins",
        "all": blocks(by_bmrb, False),
        "same_structure": blocks([r for r in by_bmrb if r["same_struct"]], False),
        "different_structure": blocks([r for r in by_bmrb if not r["same_struct"]], False)}

    label_sets_same = {row["label_set_id"] for row in same}
    census["population"] = {
        "refdb_entries_read": len(reports),
        "refdb_entries_with_a_structure_header":
            sum(1 for report in reports.values() if report["struct"]),
        "our_label_sets_total": len(offsets),
        "overlap_label_sets": len({row["label_set_id"] for row in rows}),
        "overlap_bmrbs": len({row["bmrb"] for row in rows}),
        "label_sets_with_a_comparable_cell": len(overlap),
        "label_sets_same_structure_as_refdb": len(label_sets_same),
        "label_sets_different_structure": len(overlap - label_sets_same),
        "label_sets_no_structure_chosen": len({row["label_set_id"] for row in without}),
        "comparable_cells_total": len(rows),
        "comparable_cells_by_nucleus": dict(Counter(row["nuc"] for row in rows)),
        "comparable_cells_same": len(same),
        "comparable_cells_different": len(different),
        "comparable_cells_no_structure": len(without),
        "comparable_cells_same_strict_pdb_and_chain":
            sum(1 for row in rows if row["same_struct_strict"]),
        "labels_total": sum(row["n_labels"] for row in rows),
        "labels_by_nucleus": {n: sum(row["n_labels"] for row in rows if row["nuc"] == n)
                              for n in NUCLEI},
        "our_source_mix_over_overlap_label_sets": dict(Counter(
            (offsets[ls].get("struct_used") or "none").split(":")[0] for ls in overlap)),
        "method_mix_over_overlap_label_sets": dict(Counter(
            offsets[ls].get("method") for ls in overlap))}
    census["diagnostics"] = {"rounding_grid": rounding_grid(rows)}

    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(census, indent=1) + "\n")
    cells = out.with_name(out.stem + "_cells.npz")
    np.savez_compressed(
        cells,
        nuc=np.asarray([row["nuc"] for row in rows]),
        delta=np.asarray([row["delta"] for row in rows], dtype=np.float64),
        n_labels=np.asarray([row["n_labels"] for row in rows], dtype=np.int64),
        same_struct=np.asarray([row["same_struct"] for row in rows], dtype=bool),
        same_struct_strict=np.asarray([row["same_struct_strict"] for row in rows], dtype=bool),
        no_structure=np.asarray([row["no_structure"] for row in rows], dtype=bool),
        bmrb=np.asarray([row["bmrb"] for row in rows]),
        label_set_id=np.asarray([row["label_set_id"] for row in rows]),
        our=np.asarray([row["our"] for row in rows], dtype=np.float64),
        refdb=np.asarray([row["refdb"] for row in rows], dtype=np.float64))

    for nucleus in NUCLEI:
        entry = census["per_label"]["all"][nucleus]
        print(f"  {nucleus:3s} labels={entry['n']:>9,} median={entry['median']:.2f} "
              f"within0.10={100 * entry['within_0p10']:.1f}%")
    print("wrote", out)
    return census


if __name__ == "__main__":
    main()
