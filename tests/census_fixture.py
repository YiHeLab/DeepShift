"""A miniature build data directory: a realization ledger, a fold map, edge tables.

Six realized label sets over five depositions, joined into four components by two edge
families, plus one candidate that was never realized. Small enough to count by hand, so a
producer's census can be compared with the fixture's own arithmetic.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

LEDGER_HEADER = ("label_set_id", "logical_entity_id", "bmrb_id", "source", "status",
                 "accession", "chain", "identity", "query_coverage", "resolution")

LEDGER_ROWS = [
    ("ls_0", "le_0", "100", "xtal", "success", "0aaa", "A", "0.98", "0.92", "2.10"),
    ("ls_1", "le_1", "101", "af", "success", "P00001", "A", "1.0", "1.0", ""),
    ("ls_1", "le_1", "101", "nmr", "success", "1aaa", "A", "1.0", "1.0", ""),
    ("ls_2", "le_2", "101", "xtal", "success", "2bbb", "B", "0.99", "0.95", "1.60"),
    ("ls_3", "le_3", "102", "af", "success", "P00003", "A", "1.0", "1.0", ""),
    ("ls_4", "le_4", "103", "nmr", "success", "3ccc", "A", "1.0", "0.90", ""),
    ("ls_5", "le_5", "104", "af", "success", "P00005", "A", "1.0", "1.0", ""),
    ("ls_6", "le_6", "105", "af", "excluded", "P00006", "A", "0.30", "1.0", ""),
]

# The families the fold stage publishes; only two carry a group here.
EDGE_FAMILIES = ("same_bmrb", "uniprot", "exact_token", "alignment", "mmseqs_cov0",
                 "mmseqs_containment", "partner_chain", "modeled_region")
EDGE_GROUPS = {"same_bmrb": [["ls_1", "ls_2"]], "alignment": [["ls_3", "ls_4"]]}

FOLD_OF = {"ls_0": 0, "ls_1": 1, "ls_2": 1, "ls_3": 2, "ls_4": 2, "ls_5": 6}

REALIZED_LABEL_SETS = 6
REALIZATIONS = 7
DEPOSITIONS = 5
COMPONENTS = 4
# ls_0 sits in fold 0, which the edge tables do not cover.
LABEL_SETS_FOLDS_1_6 = 5
COMPONENTS_FOLDS_1_6 = 3


def write_build(data_dir: Path, *, ledger_rows=None, fold_of=None,
                edge_groups=None) -> Path:
    """Write the fixture build into ``data_dir`` and return it."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "realization_ledger.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(LEDGER_HEADER)
        writer.writerows(LEDGER_ROWS if ledger_rows is None else ledger_rows)

    fold_of = FOLD_OF if fold_of is None else fold_of
    (data_dir / "fold_assignments_labelset_identity90.json").write_text(
        json.dumps(fold_of, indent=1, sort_keys=True) + "\n")

    edge_groups = EDGE_GROUPS if edge_groups is None else edge_groups
    counts = {}
    for family in EDGE_FAMILIES:
        path = data_dir / f"labelset_edges_{family}.csv"
        groups = edge_groups.get(family, [])
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["edge_group", "label_set_id", "bmrb_id", "fold"])
            for group_id, members in enumerate(groups):
                for label_set in members:
                    writer.writerow([group_id, label_set, "",
                                     fold_of.get(label_set, "")])
        counts[family] = len(groups)
    (data_dir / "labelset_edges_provenance.json").write_text(json.dumps(
        {"edge_group_counts": counts,
         "families": list(EDGE_FAMILIES), "grain": "labelset"},
        indent=1, sort_keys=True) + "\n")
    return data_dir


# The per-source masters, as (label set, label residue, mismatch type, {nucleus: ppm})
# rows. One materialized row does not match its coordinates.
MASTER_ROWS = {
    "af": [("ls_1", "lr_1", "match", {"ca_shift": 55.0, "cb_shift": 30.0}),
           ("ls_1", "lr_2", "match", {"ca_shift": 56.0, "cb_shift": 31.0}),
           ("ls_3", "lr_4", "match", {"ca_shift": 58.0}),
           ("ls_5", "lr_6", "match", {"ca_shift": 60.0})],
    "nmr": [("ls_1", "lr_1", "match", {"ca_shift": 55.0, "cb_shift": 30.0}),
            ("ls_1", "lr_2", "match", {"ca_shift": 56.0, "cb_shift": 31.0}),
            ("ls_4", "lr_5", "mismatch", {"ca_shift": 59.0})],
    "xtal": [("ls_0", "lr_0", "match", {"ca_shift": 54.0}),
             ("ls_2", "lr_3", "match", {"ca_shift": 57.0})],
}
FLAGGED = {"lr_1": ["ca_shift"]}


def write_masters(data_dir: Path, *, shift_columns) -> None:
    """Write the three materialization masters."""
    data_dir = Path(data_dir)
    columns = list(shift_columns)

    def cells(values):
        return [values.get(column, "") for column in columns]

    for source, rows in MASTER_ROWS.items():
        with open(data_dir / f"structure_data_{source}.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["label_set_id", "label_residue_id", "mismatch_type",
                             *columns])
            for label_set, residue, mismatch, values in rows:
                writer.writerow([label_set, residue, mismatch, *cells(values)])


ENTITY_SEQUENCES = {
    "le_0": "ACDEFGHIKLMNPQRSTVWY",
    "le_1": "ACDEFGHIKLMNPQRSTVWYACDEFG",
    "le_2": "MNPQRSTVWYACDEFGHIKL",
    "le_3": "GGGGSSSSGGGGSSSSGGGG",
    "le_4": "KKKKEEEEKKKKEEEEKKKK",
    "le_5": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL",
}


def write_entity_fasta(data_dir: Path, sequences=None) -> Path:
    """The entity sequences a homology measurement searches with."""
    path = Path(data_dir) / "entity_query.fasta"
    sequences = ENTITY_SEQUENCES if sequences is None else sequences
    path.write_text("".join(f">{name}\n{seq}\n"
                            for name, seq in sorted(sequences.items())))
    return path


def write_exclusion_authority(path: Path, flags=None, policy="p100_structural") -> Path:
    """An exclusion authority plus the report that names its fence."""
    path = Path(path)
    path.write_text(json.dumps(FLAGGED if flags is None else flags,
                               indent=1, sort_keys=True) + "\n")
    path.with_suffix(".report.json").write_text(json.dumps(
        {"policy": policy},
        indent=1, sort_keys=True) + "\n")
    return path
