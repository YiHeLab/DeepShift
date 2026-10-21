#!/usr/bin/env python
"""Export the shippable, queryable DeepShift dataset as parquet.

Reads one run's build artifacts through ``$DEEPSHIFT_RUN``, applies that build's
rereference offsets and its p100-plus-structural exclusion authority to the observed
chemical shifts, joins the fold assignments, and writes three parquet files and an export
report. The flag report must declare ``p100_structural``; other policies are refused.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# Allow running straight from the repo without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepshift.data.reref_utils import offset_columns
from deepshift.domain import SHIFT_COLUMNS
from deepshift.quality_control import FINAL_QC_POLICY

IDENT_COLS = [
    "label_set_id", "logical_entity_id", "label_residue_id", "bmrb_id",
    "entity_id", "comp_index_id", "comp_id", "residue_id", "residue_code",
]
#: Which nuclei the QC fence removed from this row, comma separated, empty
#: when it removed none. Without it a fenced cell is indistinguishable from
#: one the deposition never reported. It also names the amide proton deposited on a
#: proline, which has none (:func:`apply_proline_amide`).
QC_COLUMN = "qc_excluded_nuclei"

#: The same, for the hand-curated catalogue in ``data/deposition_notes.tsv``. Kept in its
#: own column because the two fences answer different questions: the p100 fence is a
#: statistical range rule fitted from out-of-fold predictions, the catalogue is a record of
#: depositions read back from the BMRB archive and judged wrong by hand.
CURATED_COLUMN = "curated_excluded_nuclei"

#: Catalogue statuses whose cells this export blanks. ``anticorrelated`` is
#: not among them: those channels are excluded from every benchmark comparison but kept in
#: the dataset, because the deposited values are what the archive contains and a reader
#: reproducing the archive must find them. data/README.md says so.
CURATED_BLANKED = ("defect", "recoverable", "suspect")

RUN_ROOT_VARIABLE = "DEEPSHIFT_RUN"
FLAGS_BASENAME = f"qc_flags_{FINAL_QC_POLICY}.json"
RELEASE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

#: The curated catalogue, tracked beside the release rather than in the run: it is a
#: hand-written record of archive reads, not a product of any one build.
NOTES_PATH = RELEASE_DATA_DIR / "deposition_notes.tsv"


class Inputs(NamedTuple):
    source: Path
    out: Path
    qc_path: Path
    qc_report_path: Path
    reref_path: Path
    fold_map_path: Path


class RerefSample(NamedTuple):
    """One label set's 15N column as deposited, held back to prove the offset landed."""
    label_set_id: object
    offset: object
    raw_nshift: object


class Exclusions(NamedTuple):
    qc: dict
    pos_of: dict
    cells: int
    nonnull: int


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_data_dir():
    """The build's data directory, named by the one run variable."""
    root = os.environ.get(RUN_ROOT_VARIABLE)
    if not root:
        raise ValueError(
            f"{RUN_ROOT_VARIABLE} names the run whose build the export reads; "
            "set it or pass --source and --qc-flags"
        )
    return Path(root).resolve() / "data"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help=f"build data directory holding chemical_shifts.csv and the authorities "
             f"(default: the data directory of ${RUN_ROOT_VARIABLE})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="output directory for the three release parquet tables and report "
             "(default: the data directory of this repository)",
    )
    parser.add_argument(
        "--qc-flags",
        type=Path,
        help=f"the label-residue flags JSON to apply "
             f"(default: the build's own {FLAGS_BASENAME})",
    )
    parser.add_argument("--reref-offsets", type=Path)
    parser.add_argument("--fold-map", type=Path)
    return parser.parse_args(argv)


def resolve_inputs(args):
    """Resolve every input path, refuse a missing one, and open the out dir."""
    source = (args.source.resolve() if args.source else run_data_dir())
    out = (args.out.resolve() if args.out else RELEASE_DATA_DIR)
    qc_path = (args.qc_flags.resolve() if args.qc_flags else source / FLAGS_BASENAME)
    qc_report_path = qc_path.with_suffix(".report.json")
    reref_path = (
        args.reref_offsets.resolve()
        if args.reref_offsets
        else source / "reref_offsets.json"
    )
    fold_map_path = (
        args.fold_map.resolve()
        if args.fold_map
        else source / "fold_assignments_labelset_identity90.json"
    )
    required = [
        source / "chemical_shifts.csv",
        source / "entity_catalog.csv",
        reref_path,
        fold_map_path,
        qc_path,
        qc_report_path,
        *[
            source / f"structure_build_log_{lane}.csv"
            for lane in ("af", "nmr", "xtal")
        ],
        source / "realization_ledger.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("release export inputs missing: " + ", ".join(missing))
    qc_report = json.loads(qc_report_path.read_text())
    if qc_report.get("policy") != FINAL_QC_POLICY:
        raise ValueError(
            f"final export requires policy {FINAL_QC_POLICY!r}; "
            f"got {qc_report.get('policy')!r}"
        )
    out.mkdir(parents=True, exist_ok=True)
    return Inputs(source, out, qc_path, qc_report_path, reref_path, fold_map_path)


def discover_columns(shifts_path):
    """Read the shift/ambiguity vocabulary off the header and hold it to the frozen one."""
    header = pd.read_csv(shifts_path, nrows=0).columns.tolist()
    shift_cols = sorted(c for c in header if c.endswith("_shift"))
    ambi_cols = sorted(c for c in header if c.endswith("_ambiguity_code"))
    # The schema is the package's nucleus vocabulary, not a count.
    if tuple(shift_cols) != tuple(SHIFT_COLUMNS):
        missing_columns = sorted(set(SHIFT_COLUMNS) - set(shift_cols))
        extra_columns = sorted(set(shift_cols) - set(SHIFT_COLUMNS))
        raise ValueError(
            "source does not carry the release nucleus schema: "
            f"missing={missing_columns}, extra={extra_columns}"
        )
    expected_ambiguity = sorted(
        column.removesuffix("_shift") + "_ambiguity_code"
        for column in shift_cols
    )
    if ambi_cols != expected_ambiguity:
        raise ValueError("shift and ambiguity schemas do not correspond exactly")
    return shift_cols, ambi_cols, IDENT_COLS + shift_cols + ambi_cols


def load_shifts(shifts_path, usecols):
    print(f"loading chemical_shifts.csv ({len(usecols)} cols)...", flush=True)
    df = pd.read_csv(shifts_path, usecols=usecols, dtype={
        "label_set_id": "string", "logical_entity_id": "string",
        "label_residue_id": "string", "bmrb_id": "string",
        "comp_id": "string", "residue_code": "string",
        # Without this, pandas infers entity_id as int64 here but write_sequences casts
        # it to string for deepshift_sequences.parquet: a naive join on entity_id between
        # the two release tables would then silently return nothing.
        "entity_id": "string",
    }, low_memory=False)
    # reorder to canonical layout
    df = df[usecols]
    print(f"  {len(df):,} rows", flush=True)
    return df


def capture_reref_sample(df, reref):
    """Copy one label set's raw 15N column before the offsets mutate it."""
    for label_set_id, record in reref.items():
        offsets = record.get("offsets", {})
        if "N" in offsets and (df["label_set_id"] == label_set_id).any():
            raw = df.loc[df["label_set_id"] == label_set_id, "n_shift"].copy()
            return RerefSample(label_set_id, offsets["N"], raw)
    return RerefSample(None, None, None)


def apply_reref(df, reref):
    """Add each label set's offset to the shift columns its nucleus corrects."""
    print("applying reref offsets...", flush=True)
    # build per-column offset series indexed by label_set_id
    col_offsets = {}
    for label_set_id, record in reref.items():
        for col, off in offset_columns(record.get("offsets", {})).items():
            col_offsets.setdefault(col, {})[label_set_id] = off
    for col, off_map in col_offsets.items():
        add = df["label_set_id"].map(off_map)  # NaN where no offset
        # add only where an offset exists; NaN shift stays NaN naturally
        df[col] = df[col] + add.fillna(0.0)


def curated_defects(notes_path, shift_cols) -> dict:
    """``{bmrb_id: [shift column, ...]}`` for every catalogued channel this export blanks.

    The catalogue is evidence and an authority at once: each row was read back from the
    BMRB archive by hand, so a cell it names is known wrong rather than merely unusual.
    A status it does not list in :data:`CURATED_BLANKED` is recorded and kept.
    """
    out: dict = {}
    with notes_path.open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or row[0].startswith("#") or row[0] == "bmrb_id":
                continue
            bmrb, nuclei, status = row[0], row[1], row[2]
            if status not in CURATED_BLANKED:
                continue
            for nucleus in nuclei.split(","):
                column = f"{nucleus.strip().lower()}_shift"
                if column not in shift_cols:
                    raise ValueError(
                        f"{notes_path} names nucleus {nucleus!r} for BMRB {bmrb}, which is "
                        f"not a shift column of this release")
                out.setdefault(bmrb, []).append(column)
    return out


def apply_curated(df, shift_cols, notes_path):
    """Blank every cell the curated catalogue names, and record which ones per row.

    The p100 fence cannot reach a cell it has no prediction for, so a deposition read back
    from the archive and judged wrong can still survive it. This closes that by name.
    """
    print(f"applying curated defects from {notes_path}...", flush=True)
    catalogue = curated_defects(notes_path, set(shift_cols))
    bmrb = df["bmrb_id"].astype(str).to_numpy()
    removed = [[] for _ in range(len(df))]
    blanked = 0
    for entry, columns in catalogue.items():
        rows = np.flatnonzero(bmrb == str(entry))
        for column in columns:
            values = df[column].to_numpy()
            # Only a cell that still held a value is this fence's: one the p100 fence
            # already removed is named in QC_COLUMN, and naming it twice would count it twice.
            for r in rows[~np.isnan(values[rows])]:
                blanked += 1
                values[r] = np.nan
                removed[r].append(column)
            df[column] = values
    df[CURATED_COLUMN] = [",".join(sorted(set(names))) for names in removed]
    print(f"  {blanked} live cells blanked across {len(catalogue)} catalogued depositions")
    return blanked


def apply_exclusions(df, shift_cols, qc_path, shifts_path):
    """Blank every QC-flagged cell; a flag naming an absent row or column is fatal."""
    print(f"applying exclusions from {qc_path}...", flush=True)
    qc = json.loads(qc_path.read_text())
    if not isinstance(qc, dict):
        raise ValueError("QC flags must be a JSON object")
    pos_of = {}
    # build label_residue_id -> row positions map once
    for pos, lr in enumerate(df["label_residue_id"].values):
        pos_of.setdefault(lr, []).append(pos)
    excluded_cells = 0
    excluded_nonnull = 0
    arr = {c: df[c].to_numpy() for c in shift_cols}
    # A blanked cell and a never-measured cell are both null, so the blanking alone
    # loses the distinction. QC_COLUMN records, per row, which nuclei this fence
    # removed, so a reader can tell "the archive gave a value we judged wrong" from
    # "the archive gave nothing".
    removed = [[] for _ in range(len(df))]
    unmatched_flag_ids = []
    for lr, cols in qc.items():
        rows = pos_of.get(lr)
        if not rows:
            unmatched_flag_ids.append(lr)
            continue
        for c in cols:
            if c not in arr:
                raise ValueError(f"QC flag names unknown shift column {c!r}")
            a = arr[c]
            for r in rows:
                excluded_cells += 1
                if not (isinstance(a[r], float) and np.isnan(a[r])):
                    excluded_nonnull += 1
                a[r] = np.nan
                removed[r].append(c)
    for c in shift_cols:
        df[c] = arr[c]
    df[QC_COLUMN] = [",".join(sorted(names)) for names in removed]
    if unmatched_flag_ids:
        raise ValueError(
            f"{len(unmatched_flag_ids)} QC label-residue ids are absent from "
            f"{shifts_path}; examples={unmatched_flag_ids[:10]}"
        )
    return Exclusions(qc, pos_of, excluded_cells, excluded_nonnull)


def apply_proline_amide(df):
    """Blank the amide proton deposited on a proline, which has none, and name it in QC_COLUMN."""
    rows = ((df["comp_id"] == "PRO") & df["h_shift"].notna()).fillna(False).to_numpy()
    df.loc[rows, "h_shift"] = np.nan
    df.loc[rows, QC_COLUMN] = [",".join(sorted([*filter(None, names.split(",")), "h_shift"]))
                               for names in df.loc[rows, QC_COLUMN]]
    print(f"  {rows.sum()} amide protons on prolines blanked", flush=True)
    return int(rows.sum())


def join_folds(df, fold_map_path, shift_cols, ambi_cols):
    """Attach each label set's fold, then put the frame in release column order."""
    print("joining folds...", flush=True)
    folds = json.loads(fold_map_path.read_text())
    df["fold"] = df["label_set_id"].map(folds).astype("Int64")
    fold_cov = df["fold"].notna().mean()
    df = df[[*IDENT_COLS, "fold", QC_COLUMN, CURATED_COLUMN, *shift_cols, *ambi_cols]]
    return df, fold_cov


def validate_application(df, reref, sample, exclusions, shift_cols):
    """Reread one rereferenced value and one flagged cell; a silent no-op is fatal."""
    reref_ok = False
    if sample.label_set_id is not None:
        new_nshift = df.loc[df["label_set_id"] == sample.label_set_id, "n_shift"]
        diff = (new_nshift - sample.raw_nshift).dropna()
        reref_ok = bool(len(diff) > 0 and (diff - sample.offset).abs().max() < 1e-6)

    excl_ok = False
    for lr, cols in exclusions.qc.items():
        rows = exclusions.pos_of.get(lr)
        if rows and cols:
            c = cols[0]
            if c in shift_cols:
                row = df.iloc[rows[0]]
                # blanked AND named: a fence that nulls a cell without recording it
                # would leave the release unable to distinguish it from a gap
                excl_ok = bool(row[c] != row[c]                      # NaN check
                               and c in str(row[QC_COLUMN]).split(","))
                break
    if exclusions.qc and not excl_ok:
        raise RuntimeError("QC application validation failed")
    if reref and not reref_ok:
        raise RuntimeError("rereferencing application validation failed")
    return reref_ok, excl_ok


def write_shifts(df, out):
    print("writing deepshift_shifts.parquet...", flush=True)
    shifts_out = out / "deepshift_shifts.parquet"
    df.to_parquet(shifts_out, engine="pyarrow", compression="snappy", index=False)
    return shifts_out


# How well the coordinate matches the deposited sequence, and how good the crystal is.
# These are the rows the paper's dataset census reports, so the released table carries them:
# without them the census cannot be reproduced from this dataset alone.
LEDGER_QUALITY = ["identity", "query_coverage", "resolution"]


def write_structures(source, out):
    """One row per successfully built structure, over the three source lanes."""
    print("writing deepshift_structures.parquet...", flush=True)
    scols = ["bmrb_id", "label_set_id", "source", "accession", "chain_id"]
    parts = []
    for src in ["af", "nmr", "xtal"]:
        s = pd.read_csv(source / f"structure_build_log_{src}.csv",
                        usecols=["bmrb_id", "label_set_id", "source", "accession",
                                 "chain_id", "status"], dtype="string")
        s = s[s["status"] == "success"][scols]
        parts.append(s)
    structures = pd.concat(parts, ignore_index=True).drop_duplicates()

    # The ledger holds one row per (label set, source); the build logs do not carry these
    # three columns at all. A left join keeps every built structure whether or not the
    # ledger scored it, and refuses silently duplicating a row.
    ledger = pd.read_csv(source / "realization_ledger.csv", dtype="string",
                         usecols=["label_set_id", "source", "status", *LEDGER_QUALITY])
    ledger = ledger[ledger["status"] == "success"].drop(columns=["status"])
    if ledger.duplicated(subset=["label_set_id", "source"]).any():
        raise RuntimeError("realization_ledger.csv repeats a (label set, source) pair, so "
                           "joining it would multiply structure rows")
    before = len(structures)
    structures = structures.merge(ledger, on=["label_set_id", "source"], how="left")
    if len(structures) != before:
        raise RuntimeError(f"the ledger join changed the row count {before} -> "
                           f"{len(structures)}")
    for column in LEDGER_QUALITY:
        structures[column] = pd.to_numeric(structures[column], errors="coerce")
    structures_out = out / "deepshift_structures.parquet"
    structures.to_parquet(
        structures_out, engine="pyarrow", compression="snappy", index=False
    )
    return structures, structures_out


def write_sequences(source, out):
    print("writing deepshift_sequences.parquet...", flush=True)
    catalog_path = source / "entity_catalog.csv"
    ec = pd.read_csv(catalog_path,
                     usecols=["logical_entity_id", "bmrb_id", "entity_id",
                              "polymer_length", "alignment_sequence"],
                     dtype={"logical_entity_id": "string", "bmrb_id": "string",
                            "entity_id": "string", "alignment_sequence": "string"})
    seqs = ec.drop_duplicates(subset=["logical_entity_id"]).reset_index(drop=True)
    sequences_out = out / "deepshift_sequences.parquet"
    seqs.to_parquet(
        sequences_out, engine="pyarrow", compression="snappy", index=False
    )
    return seqs, sequences_out


def write_report(out, outputs, *, qc_path, df, structures, seqs, exclusions,
                 sample, reref_ok, excl_ok, fold_cov, proline_amide, curated_blanked):
    report = {
        # The policy the shipped tables were fenced under, so the release document and
        # the artifact state the same authority.
        "qc_policy": FINAL_QC_POLICY,
        "qc_flags": qc_path.name,
        "shifts_rows": len(df),
        "structures_rows": len(structures),
        "sequences_rows": len(seqs),
        "excluded_cells": int(exclusions.cells),
        "proline_amide_blanked_cells": proline_amide,
        "qc_flag_column": QC_COLUMN,
        "curated_flag_column": CURATED_COLUMN,
        "curated_notes": NOTES_PATH.name,
        "curated_blanked_cells": int(curated_blanked),
        "excluded_cells_nonnull": int(exclusions.nonnull),
        "reref_applied_ok": reref_ok,
        "reref_val_label_set_id": sample.label_set_id,
        "reref_val_offset_N": sample.offset,
        "exclusions_applied_ok": excl_ok,
        "fold_coverage": float(fold_cov),
        "outputs": {
            path.name: sha256_file(path)
            for path in outputs
        },
    }
    print("REPORT " + json.dumps(report))
    report_path = out / "_export_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    return report


def main(argv=None):
    source, out, qc_path, _qc_report_path, reref_path, fold_map_path = resolve_inputs(
        parse_args(argv)
    )
    shifts_path = source / "chemical_shifts.csv"
    shift_cols, ambi_cols, usecols = discover_columns(shifts_path)
    df = load_shifts(shifts_path, usecols)

    reref = json.loads(reref_path.read_text())
    sample = capture_reref_sample(df, reref)
    apply_reref(df, reref)
    exclusions = apply_exclusions(df, shift_cols, qc_path, shifts_path)
    proline_amide = apply_proline_amide(df)
    curated_blanked = apply_curated(df, shift_cols, NOTES_PATH)
    df, fold_cov = join_folds(df, fold_map_path, shift_cols, ambi_cols)
    reref_ok, excl_ok = validate_application(df, reref, sample, exclusions, shift_cols)

    shifts_out = write_shifts(df, out)
    structures, structures_out = write_structures(source, out)
    seqs, sequences_out = write_sequences(source, out)
    write_report(
        out, (shifts_out, structures_out, sequences_out), qc_path=qc_path,
        df=df, structures=structures, seqs=seqs, exclusions=exclusions,
        sample=sample, reref_ok=reref_ok, excl_ok=excl_ok, fold_cov=fold_cov,
        proline_amide=proline_amide, curated_blanked=curated_blanked,
    )


if __name__ == "__main__":
    main()
