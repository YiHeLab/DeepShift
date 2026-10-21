"""Figure 4 compares arms only over one candidate set, and one roster.

The decoy panel ranks the cohort its roster names; the ring panels draw the arms
this run's coverage census says return an aromatic 1H, and no others.
"""
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"


@pytest.fixture(scope="module")
def figure():
    if str(FIGURES) not in sys.path:
        sys.path.insert(0, str(FIGURES))
    return importlib.import_module("fig_structure_response")


def roster(targets):
    """The per-target metrics table, in the rows the cohort is read from."""
    return pd.DataFrame([{"target": target, "metric": "mean_r", "method": "DeepShift"}
                         for target in targets])


def candidates(arms, per_target):
    """One row per (arm, target, candidate), with the score the ranking reads."""
    rows = []
    for arm in arms:
        for target, count in per_target.items():
            for index in range(count):
                rows.append({"method": arm, "target": target,
                             "mean_r": 0.9 - 0.01 * index, "rmsd": 1.0 + index})
    return pd.DataFrame(rows)


def test_the_decoy_panel_ranks_the_cohort_the_roster_names(figure):
    arms = [key for key, _name, _colour in figure.DECOY_ARMS]
    per_target = {"T1": 4, "T2": 4}
    cohort, counts = figure.decoy_cohort(roster(per_target), candidates(arms, per_target))
    assert cohort == ["T1", "T2"]
    assert counts == per_target


def test_the_decoy_panel_refuses_an_arm_that_is_missing_a_target(figure):
    arms = [key for key, _name, _colour in figure.DECOY_ARMS]
    per_target = {"T1": 4, "T2": 4}
    table = candidates(arms, per_target)
    thinned = table[~((table["method"] == arms[-1]) & (table["target"] == "T2"))]
    with pytest.raises(SystemExit, match="do not cover the roster's cohort"):
        figure.decoy_cohort(roster(per_target), thinned)


def test_the_decoy_panel_refuses_arms_that_ranked_different_candidate_sets(figure):
    arms = [key for key, _name, _colour in figure.DECOY_ARMS]
    per_target = {"T1": 4, "T2": 4}
    table = candidates(arms, per_target)
    dropped = table[~((table["method"] == arms[-1]) & (table["target"] == "T2")
                      & (table["mean_r"] < 0.88))]
    with pytest.raises(SystemExit, match="different candidate set"):
        figure.decoy_cohort(roster(per_target), dropped)


# The arms this run's census names on the aromatic protons, in config.py's order.
RING_ROSTER = ["ucbshift2_sameprot", "shiftx2_sameprot", "shiftx2_noshifty", "nmrgnn"]


def test_the_ring_panels_draw_deepshift_and_every_arm_of_the_roster(figure):
    arms = figure.ring_arms(["deepshift", *RING_ROSTER], RING_ROSTER)
    assert [name for name, _column, _colour, _dash in arms] == [
        "DeepShift", "UCBShift 2.0", "SHIFTX2", "SHIFTX+", "NMRGNN"]
    assert [column for _name, column, _colour, _dash in arms] == [
        f"pred_{arm}" for arm in ("deepshift", *RING_ROSTER)]


def test_the_ring_panels_refuse_a_dump_missing_an_arm_of_the_roster(figure):
    """The dump written before NMRGNN joined: it would draw four predictors where the
    tables print five."""
    stale = ["deepshift", *RING_ROSTER[:-1]]
    with pytest.raises(SystemExit, match=r"re-run benchmarks\.dump_ring_cells"):
        figure.ring_arms(stale, RING_ROSTER)


def test_the_ring_panels_refuse_a_dump_carrying_an_arm_the_roster_dropped(figure):
    extra = ["deepshift", *RING_ROSTER, "ucbshift2_selfexcl"]
    with pytest.raises(SystemExit, match=r"re-run benchmarks\.dump_ring_cells"):
        figure.ring_arms(extra, RING_ROSTER)


def test_the_ring_panels_refuse_an_arm_they_have_no_hue_for(figure):
    """Adding a predictor to the roster is one line in benchmarks/config.py, so the
    figure says which arm it cannot draw instead of drawing four of five."""
    roster = [*RING_ROSTER, "sparta"]
    with pytest.raises(SystemExit, match="no plot style is declared"):
        figure.ring_arms(["deepshift", *roster], roster)
