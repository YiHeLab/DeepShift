"""The rereferencing stage declares as inputs exactly the files it reads."""
import argparse
import builtins
import csv
from pathlib import Path

from deepshift import domain
from deepshift.data import build, reref

TOKENS = ["ALA", "GLY", "SER", "THR", "VAL", "LEU", "ILE", "ASP", "GLU", "LYS"]


def _write_csv(path: Path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_reref_inputs(data: Path) -> None:
    """One label set realized once, in the tables the rereferencing stage reads."""
    exact = domain.exact_polymer_sequence_sha256(TOKENS)
    alignment = domain.alignment_sequence_from_tokens(TOKENS)
    label_set = domain.label_set_id("7", "1", "1", "1", exact)
    entity = domain.logical_entity_id("7", "1", exact)
    _write_csv(data / "entity_catalog.csv", [
        "logical_entity_id", "label_set_id", "bmrb_id", "entity_id", "shift_list_id",
        "entity_assembly_id", "sample_condition_ids", "exact_polymer_sequence_sha256",
        "alignment_sequence_sha256", "sequence_scope", "polymer_length",
        "polymer_seq_nums", "polymer_token_sequence", "alignment_sequence",
        "n_supported_residues", "n_supported_cells",
    ], [{
        "logical_entity_id": entity, "label_set_id": label_set, "bmrb_id": "7",
        "entity_id": "1", "shift_list_id": "1", "entity_assembly_id": "1",
        "sample_condition_ids": "", "exact_polymer_sequence_sha256": exact,
        "alignment_sequence_sha256": domain.alignment_sequence_sha256(alignment),
        "sequence_scope": "full", "polymer_length": len(TOKENS),
        "polymer_seq_nums": ",".join(str(i) for i in range(1, len(TOKENS) + 1)),
        "polymer_token_sequence": " ".join(TOKENS), "alignment_sequence": alignment,
        "n_supported_residues": len(TOKENS), "n_supported_cells": len(TOKENS),
    }])
    shift_rows = [{"label_set_id": label_set, "bmrb_id": "7", "shift_list_id": "1",
                   "comp_index_id": index, "comp_id": token, "ca_shift": 50.0 + index}
                  for index, token in enumerate(TOKENS, start=1)]
    _write_csv(data / "chemical_shifts.csv",
               ["label_set_id", "bmrb_id", "shift_list_id", "comp_index_id", "comp_id",
                "ca_shift"], shift_rows)
    materialization = "mz_" + "0" * 64
    for source in ("af", "nmr", "xtal"):
        rows = [{**row, "materialization_id": materialization, "struct_res_id": row["comp_index_id"]}
                for row in shift_rows] if source == "af" else []
        _write_csv(data / f"structure_data_{source}.csv",
                   ["label_set_id", "bmrb_id", "shift_list_id", "materialization_id",
                    "comp_index_id", "comp_id", "struct_res_id", "ca_shift"], rows)
        _write_csv(data / f"structure_build_log_{source}.csv",
                   ["materialization_id", "pdb_path", "chain_id", "status"],
                   [{"materialization_id": materialization, "pdb_path": str(data / "7.pdb"),
                     "chain_id": "A", "status": "success"}] if source == "af" else [])


def test_reref_declares_exactly_the_files_it_opens(tmp_path, monkeypatch):
    root = tmp_path / "build"
    data = root / "data"
    data.mkdir(parents=True)
    write_reref_inputs(data)
    shiftx = tmp_path / "shiftx"
    builder = build.Builder(argparse.Namespace(
        root=str(root), workers=1, folds=[1], resume=False, shiftx_bin=str(shiftx),
        shiftx_source=None, mmseqs_bin=None, data_archive=None, split_seed=None,
        holdout_frac=None))
    captured = {}

    def record(stage, module, *arguments, **keywords):
        captured.update(stage=stage, module=module, arguments=[str(a) for a in arguments],
                        inputs=[Path(p).resolve() for p in keywords["inputs"]])

    monkeypatch.setattr(builder, "command", record)
    builder.reref()
    assert captured["module"] == "deepshift.data.reref"
    declared = set(captured["inputs"])
    assert shiftx.resolve() in declared
    assert not any("refdb" in path.name.lower() for path in declared)

    opened = set()
    real_open = builtins.open

    def tracing_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            opened.add(Path(file).resolve())
        return real_open(file, *args, **kwargs)

    # The structure the label set names is absent, so SHIFTX is never invoked; it has
    # only to exist for the stage not to fetch one.
    shiftx.touch()
    monkeypatch.setattr(reref, "SHIFTX_BIN", str(shiftx))
    monkeypatch.setattr(builtins, "open", tracing_open)
    reref.entity_reref_main(captured["arguments"])
    monkeypatch.undo()
    unopened = {path for path in declared - opened if path != shiftx.resolve()}
    assert not unopened, sorted(str(path) for path in unopened)
