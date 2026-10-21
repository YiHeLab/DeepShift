"""Table S22 prints its rows; the cohort counts belong to the caption, not the page."""
import importlib.util
import re
import sys
from pathlib import Path

SI_BUILD = Path(__file__).resolve().parents[1] / "si_build"
NUC = ["h_shift", "ha_shift", "ca_shift", "cb_shift", "c_shift", "n_shift"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SI_BUILD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _emit_data_composition(monkeypatch, out_dir):
    monkeypatch.setenv("SI_OUT", str(out_dir))
    sys.path.insert(0, str(SI_BUILD))
    try:
        _load("si_common")
        return _load("emit_data_composition")
    finally:
        sys.path.remove(str(SI_BUILD))


def _blocks(seed):
    rmse = {n: seed + 0.001 * index for index, n in enumerate(NUC)}
    block1 = {
        "floor": rmse, "redundant": rmse, "diverse": rmse,
        "cells": {n: 100_000 + index for index, n in enumerate(NUC)},
        "rows": {"floor": 80_949, "redundant": 243_031, "diverse": 243_046},
        "label_sets": {"floor": 923, "redundant": 923, "diverse": 2_658},
        "structures": {"floor": 923, "redundant": 2_769, "diverse": 2_658},
        "heldout_fold": 1, "train_folds": [2, 3, 4, 5],
    }
    block2 = {
        "pool": {"label_sets": 7_994, "rows": 1_150_629, "structures": 12_000},
        "one_per": {"rows": 715_052, "structures": 7_994},
        "per_source": {s: {"one_per_protein": rmse, "every_structure": rmse,
                           "cells": 97_548} for s in ("af", "nmr", "xtal")},
    }
    return block1, block2


def test_the_float_prints_row_numbers_and_the_cohort_counts_sit_in_the_comment(
        monkeypatch, tmp_path):
    """The section headers name the contrast; the shifts scored and the protein counts
    that describe the cohort are caption facts, and a float that printed them too would
    carry a second set of counts the caption already states."""
    module = _emit_data_composition(monkeypatch, tmp_path / "out")
    block1, block2 = _blocks(0.4)
    table = module.build_table(block1, block2)

    float_body = table.split(r"\end{table}")[0]
    headers = re.findall(r"\\multicolumn\{8\}\{l\}\{\\textit\{(.*?)\}\}", float_body)
    assert headers and not [h for h in headers if any(c.isdigit() for c in h)]
    assert r"\capneeded" not in table

    comment = [line for line in table.split(r"\end{table}")[1].splitlines() if line.strip()]
    assert comment and all(line.startswith("%") for line in comment)
    comment = "\n".join(comment)
    for count in (f"{sum(block1['cells'].values()):,}", f"{block2['pool']['label_sets']:,}",
                  f"{block1['rows']['redundant']:,}", f"{block1['rows']['diverse']:,}",
                  f"{block1['label_sets']['diverse']:,}"):
        assert count in comment


def test_every_number_the_float_prints_comes_from_the_reports(monkeypatch, tmp_path):
    """Nothing in the table is typed: the row stubs and the row counts move when the
    reports move."""
    module = _emit_data_composition(monkeypatch, tmp_path / "out")
    block1, block2 = _blocks(0.4)
    block1["label_sets"] = {"floor": 11, "redundant": 11, "diverse": 22}
    block2["pool"]["label_sets"] = 33
    table = module.build_table(block1, block2)

    assert r"\quad 11 proteins, one structure each" in table
    assert r"\quad the same 11 proteins, all three structures each" in table
    assert r"\quad 22 different proteins, one structure each" in table
    assert "923" not in table and "2,658" not in table and "7,994" not in table
