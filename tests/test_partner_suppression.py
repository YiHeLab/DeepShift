import csv
import json

import pytest

from deepshift.data.partner_suppression import build_suppression_map, main


def _write_log(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["label_set_id", "pdb_path", "chain_id", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_only_cross_fold_dataset_partners_are_suppressed(tmp_path):
    structure = tmp_path / "complex.pdb"
    log = tmp_path / "structure_build_log_af.csv"
    _write_log(log, [
        {"label_set_id": "ls_a", "pdb_path": structure, "chain_id": "A",
         "status": "success"},
        {"label_set_id": "ls_b", "pdb_path": structure, "chain_id": "B",
         "status": "success"},
        {"label_set_id": "ls_c", "pdb_path": structure, "chain_id": "C",
         "status": "success"},
    ])
    suppress, report = build_suppression_map([log], {"ls_a": 1, "ls_b": 2, "ls_c": 1})
    assert suppress == {
        "complex.pdb\tA": ["B"],
        "complex.pdb\tB": ["A", "C"],
        "complex.pdb\tC": ["B"],
    }
    assert report["cross_fold_pairs"] == 4
    assert report["suppression_keys"] == 3


def test_cli_writes_map_and_report(tmp_path):
    log = tmp_path / "structure_build_log_af.csv"
    _write_log(log, [
        {"label_set_id": "ls_a", "pdb_path": tmp_path / "one.pdb", "chain_id": "A",
         "status": "success"},
    ])
    folds = tmp_path / "folds.json"
    folds.write_text('{"ls_a": 0}\n')
    out = tmp_path / "partner_chain_suppression.json"
    report = tmp_path / "partner_chain_suppression.report.json"
    main(["--build-logs", str(log), "--fold-map", str(folds),
          "--out", str(out), "--report", str(report)])
    assert json.loads(out.read_text()) == {}
    payload = json.loads(report.read_text())
    assert payload["cross_fold_pairs"] == 0


def test_duplicate_structure_basenames_fail_closed(tmp_path):
    first = tmp_path / "source_a" / "complex.pdb"
    second = tmp_path / "source_b" / "complex.pdb"
    log = tmp_path / "structure_build_log.csv"
    _write_log(log, [
        {"label_set_id": "ls_a", "pdb_path": first, "chain_id": "A",
         "status": "success"},
        {"label_set_id": "ls_b", "pdb_path": second, "chain_id": "B",
         "status": "success"},
    ])
    with pytest.raises(ValueError, match="distinct structure paths share 1 basename"):
        build_suppression_map([log], {"ls_a": 1, "ls_b": 2})
