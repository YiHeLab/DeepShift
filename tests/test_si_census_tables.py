"""The SI fold census counts one run under the fold map and fence it reads.

The fixture build is small enough to count by hand: six realized label sets and one
materialized row that does not match its coordinates.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "si_build"))

import census_fixture as fx

from deepshift.domain import SHIFT_COLUMNS


@pytest.fixture
def tables(tmp_path, monkeypatch):
    monkeypatch.setenv("SI_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("DEEPSHIFT_RUN", str(tmp_path / "run"))
    sys.modules.pop("si_common", None)
    sys.modules.pop("gen_census_tables", None)
    import gen_census_tables as module

    data_dir = fx.write_build(tmp_path / "run" / "data")
    fx.write_masters(data_dir, shift_columns=SHIFT_COLUMNS)
    flags = fx.write_exclusion_authority(data_dir / "qc_flags_p100_structural.json")
    return module, tmp_path / "out", data_dir, flags


def run(module, flags, extra=()):
    sys.argv = ["gen_census_tables.py", "--flags", str(flags), *extra]
    module.main()


def test_the_fold_table_counts_rows_and_cells_per_fold(tables):
    module, out, data_dir, flags = tables
    by_fold, by_source = module.fold_counts(data_dir, fx.FOLD_OF)
    assert by_fold[1] == {"residue_rows": 5, "observed_cells": 9}
    assert by_fold[2]["residue_rows"] == 1
    assert set(by_source) == {"af", "nmr", "xtal"}
    run(module, flags)
    assert "6 & held out &" in (out / "si_census_folds.tex").read_text()


def test_a_fold_map_from_another_build_is_refused(tables):
    module, _, data_dir, flags = tables
    fold_of = {k: v for k, v in fx.FOLD_OF.items() if k != "ls_5"}
    fx.write_build(data_dir, fold_of=fold_of)
    with pytest.raises(ValueError, match="different builds"):
        run(module, flags)


def test_the_census_refuses_an_authority_fitted_under_another_fence(tables):
    """The census counts the corpus this release publishes, so it reads only the fence
    this release applies: a report declaring another policy counts a different corpus."""
    module, _out, data_dir, _flags = tables
    other = fx.write_exclusion_authority(data_dir / "qc_flags_p100.json", policy="p100")
    with pytest.raises(SystemExit, match="declares policy 'p100'"):
        run(module, other)
