"""The downstream-applications panels measure their counts from the rows they are given.

Both panels are drawn on synthetic rows whose win counts, common support and sensitive
fractions are none of the values the deployed figure carries, so a panel that compares
its measurement with a stored number cannot pass.
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))

import fig_downstream_applications as fig


def idp8_frame():
    """Three ensembles; DeepShift wins CB twice, C' once, CA never."""
    return pd.DataFrame({
        "label": ["ens_a", "ens_b", "ens_c"],
        "CB_published": [1.0, 1.0, 1.0], "CB_rmsd": [0.5, 0.5, 2.0],
        "C_published": [1.0, 1.0, 1.0], "C_rmsd": [0.5, 2.0, 2.0],
        "CA_published": [1.0, 1.0, 1.0], "CA_rmsd": [2.0, 2.0, 2.0],
    })


def tau_frame(sensitivities=None):
    """Four methods over four shared sites; DeepShift is sensitive at three of them."""
    sites = [(210, "CA"), (211, "CA"), (212, "CB"), (213, "N")]
    sensitivities = sensitivities or {
        "DeepShift_original": [10.0, 5.0, 4.0, 1.0],
        "UCBShift2_combined_selfexcl": [10.0, 1.0, 1.0, 1.0],
        "SHIFTX2_release": [1.0, 1.0, 1.0, 1.0],
        "MLFF_MACE_relaxed": [10.0, 10.0, 1.0, 1.0],
    }
    rows = []
    for method, values in sensitivities.items():
        for (position, atom), value in zip(sites, values, strict=True):
            rows.append({"method": method, "position": position, "atom": atom,
                         "sensitivity": value})
    return pd.DataFrame(rows)


def test_panel_a_reports_the_wins_the_rows_contain():
    fig.style.use_house_style()
    _, ax = plt.subplots()
    values = fig.build_panel_a(ax, idp8_frame())
    plt.close("all")
    assert {atom: (v["deepshift_wins"], v["n_ensembles"])
            for atom, v in values.items()} == {"CB": (2, 3), "C": (1, 3), "CA": (0, 3)}


def test_panel_b_reports_the_support_and_sensitive_counts_the_rows_contain():
    fig.style.use_house_style()
    _, ax = plt.subplots()
    values, support = fig.build_panel_b(ax, tau_frame())
    plt.close("all")
    assert support == 4
    assert values["DeepShift_original"]["sensitive"] == 3
    assert values["SHIFTX2_release"]["sensitive"] == 0
    assert values["DeepShift_original"]["sensitive_percent"] == pytest.approx(75.0)


def test_panel_b_refuses_methods_scored_on_different_sites():
    frame = tau_frame()
    frame.loc[frame.method.eq("SHIFTX2_release") & frame.position.eq(213),
              "position"] = 999
    fig.style.use_house_style()
    _, ax = plt.subplots()
    with pytest.raises(SystemExit, match="common support"):
        fig.build_panel_b(ax, frame)
    plt.close("all")
