"""The release dataset export reads one run's build and declares the fence it applied."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepshift.quality_control import FINAL_QC_POLICY

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_dataset.py"
SPEC = importlib.util.spec_from_file_location("export_dataset", SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = EXPORT
SPEC.loader.exec_module(EXPORT)


def build_run(root, *, policy=FINAL_QC_POLICY):
    """A run root carrying the data directory the build's flags stage leaves behind."""
    data = root / "data"
    data.mkdir(parents=True)
    (data / "chemical_shifts.csv").write_text("label_set_id\nls_1\n")
    (data / "entity_catalog.csv").write_text("logical_entity_id\nle_1\n")
    (data / "reref_offsets.json").write_text(json.dumps({"ls_1": {"offsets": {"N": -0.2}}}))
    (data / "fold_assignments_labelset_identity90.json").write_text(json.dumps({"ls_1": 3}))
    for lane in ("af", "nmr", "xtal"):
        (data / f"structure_build_log_{lane}.csv").write_text("bmrb_id\n1\n")
    # the ledger carries identity/coverage/resolution, which the build logs do not
    (data / "realization_ledger.csv").write_text(
        "label_set_id,source,status,identity,query_coverage,resolution\n"
        "ls_1,xtal,success,0.98,0.99,1.80\n")
    flags = data / f"qc_flags_{FINAL_QC_POLICY}.json"
    flags.write_text(json.dumps({"lr_1": ["ca_shift"]}))
    flags.with_suffix(".report.json").write_text(json.dumps({"policy": policy}))
    return data, flags


def resolve(argv, run_root, monkeypatch):
    monkeypatch.setenv(EXPORT.RUN_ROOT_VARIABLE, str(run_root))
    return EXPORT.resolve_inputs(EXPORT.parse_args(argv))


def test_the_export_reads_the_builds_post_cache_flag_pair_from_the_run_variable(
    tmp_path, monkeypatch
):
    data, flags = build_run(tmp_path / "run")
    inputs = resolve(["--out", str(tmp_path / "out")], tmp_path / "run", monkeypatch)
    assert inputs.source == data
    assert inputs.qc_path == flags
    assert inputs.qc_report_path == flags.with_suffix(".report.json")


def test_the_export_refuses_a_flag_report_declaring_another_policy(tmp_path, monkeypatch):
    build_run(tmp_path / "run", policy="structural")
    with pytest.raises(ValueError, match=FINAL_QC_POLICY):
        resolve(["--out", str(tmp_path / "out")], tmp_path / "run", monkeypatch)


def test_the_export_names_the_run_variable_when_no_source_is_given(tmp_path, monkeypatch):
    monkeypatch.delenv(EXPORT.RUN_ROOT_VARIABLE, raising=False)
    with pytest.raises(ValueError, match=EXPORT.RUN_ROOT_VARIABLE):
        EXPORT.resolve_inputs(EXPORT.parse_args(["--out", str(tmp_path / "out")]))


def test_the_export_report_declares_the_fence_it_applied(tmp_path, monkeypatch):
    _data, flags = build_run(tmp_path / "run")
    inputs = resolve(["--out", str(tmp_path / "out")], tmp_path / "run", monkeypatch)
    frame = pd.DataFrame({"label_set_id": ["ls_1"]})
    table = inputs.out / "table.parquet"
    table.write_bytes(b"rows")
    report = EXPORT.write_report(
        inputs.out, (table,),
        qc_path=inputs.qc_path,
        df=frame, structures=frame, seqs=frame,
        exclusions=EXPORT.Exclusions({}, {}, 0, 0),
        sample=EXPORT.RerefSample(None, None, None),
        reref_ok=True, excl_ok=True, fold_cov=1.0,
        proline_amide=2, curated_blanked=7,
    )
    written = json.loads((inputs.out / "_export_report.json").read_text())
    assert report == written
    assert written["qc_policy"] == FINAL_QC_POLICY
    assert written["qc_flags"] == flags.name
    assert written["outputs"] == {table.name: hashlib.sha256(b"rows").hexdigest()}
    assert not {"inputs", "qc_flags_sha256", "curated_notes_sha256"} & written.keys()
    # the curated catalogue is a second fence, and the report names it the same way:
    # which column carries it, which file it came from, and how many cells it removed
    assert written["curated_flag_column"] == EXPORT.CURATED_COLUMN
    assert written["curated_notes"] == EXPORT.NOTES_PATH.name
    assert written["curated_blanked_cells"] == 7
    assert written["proline_amide_blanked_cells"] == 2


def test_the_curated_column_names_only_the_cells_it_removed(tmp_path):
    """A cell the p100 fence already removed is named there, not a second time here, so
    the column and the report count the same cells."""
    notes = tmp_path / "notes.tsv"
    notes.write_text("bmrb_id\tnuclei\tstatus\n7016\tH,N\trecoverable\n")
    df = pd.DataFrame({"bmrb_id": pd.array(["7016", "7016", "1"], dtype="string"),
                       "h_shift": [0.02, np.nan, 8.1], "n_shift": [0.2, 0.2, 120.0]})
    assert EXPORT.apply_curated(df, ["h_shift", "n_shift"], notes) == 3
    assert df[EXPORT.CURATED_COLUMN].tolist() == ["h_shift,n_shift", "n_shift", ""]
    assert df["h_shift"].isna().tolist() == [True, True, False]


def test_a_proline_amide_proton_is_blanked_and_named():
    df = pd.DataFrame({"comp_id": pd.array(["PRO", "PRO", "ALA"], dtype="string"),
                       "h_shift": [8.2, np.nan, 8.1],
                       EXPORT.QC_COLUMN: ["n_shift", "", ""]})
    assert EXPORT.apply_proline_amide(df) == 1
    assert df["h_shift"].isna().tolist() == [True, True, False]
    assert df[EXPORT.QC_COLUMN].tolist() == ["h_shift,n_shift", "", ""]


def test_entity_id_loads_as_string_like_write_sequences_casts_it(tmp_path):
    """``write_sequences`` casts ``entity_id`` to string for
    ``deepshift_sequences.parquet``. Left to pandas' own inference here, it would load as
    int64 in ``deepshift_shifts.parquet`` instead, and a join between the two release
    tables on ``entity_id`` would then silently return nothing."""
    shifts = tmp_path / "chemical_shifts.csv"
    shifts.write_text("label_set_id,entity_id\nls_1,1\n")
    df = EXPORT.load_shifts(shifts, ["label_set_id", "entity_id"])
    assert df["entity_id"].dtype == "string"
