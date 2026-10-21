"""Figure S1 draws the split's own protein count.

The distinct-protein row is the number of components the fold assignment is atomic over,
so the figure moves with the split and never with a clustering rule of its own.
"""
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))

import census_fixture as fx
import fig_census_merged as fig


@pytest.fixture
def build(tmp_path):
    return fx.write_build(tmp_path / "data")


def test_the_distinct_protein_count_is_over_the_folds_the_edge_tables_cover(build):
    counts = fig.census(build)
    assert counts["distinct_proteins"] == fx.COMPONENTS_FOLDS_1_6
    assert counts["label_sets_in_components"] == fx.LABEL_SETS_FOLDS_1_6
    assert counts["label_sets_realized"] == fx.REALIZED_LABEL_SETS
    assert counts["largest_component"] == 2
    assert counts["components_of_one"] == 1


def test_the_figure_is_written(build, tmp_path, monkeypatch):
    monkeypatch.setattr(fig.frozen, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["fig_census_merged.py", "--build", str(build)])
    fig.main()
    assert (tmp_path / "out" / "fig_census_merged.pdf").is_file()
    assert (tmp_path / "out" / "fig_census_merged.png").is_file()


def test_a_build_whose_fold_map_misses_a_realized_label_set_is_not_drawn(build):
    fold_of = {k: v for k, v in fx.FOLD_OF.items() if k != "ls_5"}
    fx.write_build(build, fold_of=fold_of)
    with pytest.raises(ValueError, match="different builds"):
        fig.census(build)
