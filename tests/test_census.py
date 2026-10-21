"""The corpus census counts one build and refuses inputs from two.

The census is what every dataset figure and table counts with, so it is pinned here:
successful realizations only, the fold map bound to the ledger, and a distinct-protein
count that is the number of components the fold stage's own edge tables close.
"""
import json

import census_fixture as fx
import pytest

from deepshift.data import census, folds


@pytest.fixture
def build(tmp_path):
    return fx.write_build(tmp_path / "data")


def test_the_census_counts_only_successful_realizations(build):
    frame = census.read_realizations(build)
    counts = census.source_census(frame)
    assert counts["label_sets_realized"] == fx.REALIZED_LABEL_SETS
    assert counts["realizations"] == fx.REALIZATIONS
    assert counts["depositions"] == fx.DEPOSITIONS
    assert counts["per_source_label_sets"] == {"af": 3, "nmr": 2, "xtal": 2}
    assert counts["per_source_sole"] == {"af": 2, "nmr": 1, "xtal": 2}
    assert counts["multiplicity"] == {"1": 5, "2": 1, "3": 0}


def test_the_census_refuses_a_label_set_realized_twice_by_one_source(build):
    repeated = [*list(fx.LEDGER_ROWS), fx.LEDGER_ROWS[0]]
    fx.write_build(build, ledger_rows=repeated)
    with pytest.raises(ValueError, match="more than once"):
        census.read_realizations(build)


def test_the_distinct_protein_count_is_the_fold_stages_component_count(build):
    frame = census.read_realizations(build)
    label_sets = set(frame["label_set_id"])
    components, _ = census.protein_components(build, label_sets)
    assert len(components) == fx.COMPONENTS
    assert sorted(components.values()) == [["ls_0"], ["ls_1", "ls_2"], ["ls_3", "ls_4"],
                                           ["ls_5"]]


def test_the_edge_tables_cover_folds_1_to_6(build):
    fold_of = census.read_fold_map(build)
    covered = census.edge_table_label_sets(fold_of)
    assert covered == set(fx.FOLD_OF) - {"ls_0"}
    components, _ = census.protein_components(build, covered)
    assert len(components) == fx.COMPONENTS_FOLDS_1_6


def test_a_fold_map_from_another_build_is_refused(build):
    frame = census.read_realizations(build)
    fold_of = census.read_fold_map(build)
    fold_of.pop("ls_5")
    with pytest.raises(ValueError, match="different builds"):
        census.bind_fold_map(fold_of, set(frame["label_set_id"]))


def test_edges_naming_an_unrealized_label_set_are_refused(build):
    groups = dict(fx.EDGE_GROUPS, uniprot=[["ls_5", "ls_9"]])
    fx.write_build(build, edge_groups=groups)
    with pytest.raises(ValueError, match="different builds"):
        census.protein_components(build, set(fx.FOLD_OF))


def test_every_published_edge_family_is_read_back(build):
    families, record = folds.read_labelset_edge_tables(str(build))
    assert set(families) == set(fx.EDGE_FAMILIES)
    assert families["alignment"] == [["ls_3", "ls_4"]]
    assert record["edge_group_counts"] == json.loads(
        (build / "labelset_edges_provenance.json").read_text())["edge_group_counts"]
