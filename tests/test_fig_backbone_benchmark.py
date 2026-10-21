"""Figure 3 and Table 2 draw from the benchmark scorers' reports, or not at all.

The producer takes the marginal and the paired report of one scoring run. These tests
pin what it accepts as one run, and how a table cell earns its bold and its asterisk.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))

import fig_backbone_benchmark as producer

from benchmarks.runner_up_leads import lead_tests

ARMS = ["deepshift", "ucbshift_full_sameprot", "ucbshift_x", "shiftx2_sameprot", "sparta"]
NUCLEI = ("CA", "CB", "C", "HA", "H", "N")


def _marginal(rmse_by_arm):
    """A marginal report where every source and nucleus carries the same arm values."""
    nuclei = {nucleus: {"n_cells": 100,
                        **{arm: {"rmse": rmse, "mae": rmse * 0.8,
                                 "rmse_ci95": [rmse - 0.05, rmse + 0.05],
                                 "r": 0.9, "r_ci95": [0.88, 0.92]}
                           for arm, rmse in rmse_by_arm.items()}}
              for nucleus in NUCLEI}
    return {"n_common_structures": 12, "n_common_cells": 600, "methods": list(rmse_by_arm),
            "unit": "labelset",
            "residual_convention": "raw: prediction minus the re-referenced observed value",
            "secondary_shift_convention": "observed residue-type mean removed",
            "metrics": ["rmse", "mae"],
            "by_source": {source: {"n_structures": 4, "n_units": 3,
                                   "nuclei": json.loads(json.dumps(nuclei))}
                          for source in ("xtal", "af", "nmr", "all")}}


def _paired(rmse_by_arm, upper, unit="structure"):
    """A paired report of the same cohort, with one shared upper interval bound."""
    reference = "deepshift"
    differences = {arm: {"deepshift": rmse_by_arm[reference], "competitor": rmse,
                         "diff": rmse_by_arm[reference] - rmse,
                         "ci": [rmse_by_arm[reference] - rmse - 0.1, upper],
                         "p_deepshift_better": 0.99}
                   for arm, rmse in rmse_by_arm.items() if arm != reference}
    report = {"tier": 2, "reference_arm": reference, "methods": list(rmse_by_arm),
              "unit": unit, "boot": 2000, "n_common_structures": 12, "n_common_cells": 600}
    for source in ("xtal", "af", "nmr", "all"):
        report[source] = {nucleus: json.loads(json.dumps(differences)) for nucleus in NUCLEI}
    return report


def _write(tmp_path, marginal_report, paired_report):
    marginal = tmp_path / "score_tier2.json"
    paired = tmp_path / "score_tier2_paired.json"
    marginal.write_text(json.dumps(marginal_report))
    paired.write_text(json.dumps(paired_report))
    return marginal, paired


@pytest.fixture
def reports(tmp_path):
    rmse = {"deepshift": 0.30, "ucbshift_full_sameprot": 0.42, "ucbshift_x": 0.50,
            "shiftx2_sameprot": 0.55, "sparta": 0.60}
    return rmse, tmp_path


def test_the_figure_draws_from_the_named_scorers_reports(reports):
    rmse, tmp_path = reports
    marginal, paired = _write(tmp_path, _marginal(rmse), _paired(rmse, -0.02))
    report, paired_report = producer.read(marginal, paired, ARMS)
    assert paired_report["tier"] == 2
    assert report["n_common_cells"] == paired_report["n_common_cells"] == 600
    block = report["by_source"]["all"]["nuclei"]
    assert block["CA"]["deepshift"]["r"] == 0.9
    assert block["CA"]["deepshift"]["rmse_ci95"] == [0.25, 0.35]


def test_the_figure_refuses_two_reports_of_different_scoring_runs(reports):
    rmse, tmp_path = reports
    other = dict(_paired(rmse, -0.02), n_common_cells=599)
    marginal, paired = _write(tmp_path, _marginal(rmse), other)
    with pytest.raises(SystemExit, match="n_common_cells"):
        producer.read(marginal, paired, ARMS)


def test_a_table_cell_is_starred_only_when_the_paired_interval_excludes_zero(reports):
    rmse, tmp_path = reports
    for upper, starred in ((-0.02, True), (0.03, False)):
        marginal, paired = _write(tmp_path, _marginal(rmse), _paired(rmse, upper))
        body = producer.table(*producer.read(marginal, paired, ARMS)[:2], ARMS)
        assert (r"\textbf{0.3000}*" in body) is starred
        assert r"\textbf{0.3000}" in body


def test_a_table_cell_a_competitor_wins_makes_no_paired_claim(reports):
    rmse, tmp_path = reports
    winning_competitor = {**rmse, "deepshift": 0.90}
    marginal, paired = _write(tmp_path, _marginal(winning_competitor),
                              _paired(winning_competitor, -0.02))
    body = producer.table(*producer.read(marginal, paired, ARMS)[:2], ARMS)
    assert r"\textbf{0.4200}" in body and r"\textbf{0.4200}*" not in body


def test_every_drawn_arm_is_a_table_row_and_competes_for_its_bold(reports):
    """Table 1 prints the five arms the figure draws, SPARTA+ included, so a cell SPARTA+
    wins is bolded on SPARTA+'s row."""
    rmse, tmp_path = reports
    fastest = {**rmse, "sparta": 0.10}
    marginal, paired = _write(tmp_path, _marginal(fastest), _paired(fastest, -0.02))
    body = producer.table(*producer.read(marginal, paired, ARMS)[:2], ARMS)
    assert r"\textbf{0.1000}" in body and r"\textbf{0.3000}" not in body


@pytest.mark.parametrize("unit,named", [("structure", "structure instances"),
                                        ("labelset", "label sets")])
def test_the_asterisk_names_the_clustering_the_paired_report_declares(reports, unit, named):
    """The paired scorer resamples structure instances or whole label sets, and the two
    do not license the same claim, so the float states the one it was handed."""
    rmse, tmp_path = reports
    marginal, paired = _write(tmp_path, _marginal(rmse), _paired(rmse, -0.02, unit))
    _report, paired_report = producer.read(marginal, paired, ARMS)
    assert producer.UNIT_NAMED[paired_report["unit"]] == named


def test_a_paired_report_that_names_no_clustering_is_refused(reports):
    rmse, tmp_path = reports
    without = {k: v for k, v in _paired(rmse, -0.02).items() if k != "unit"}
    marginal, paired = _write(tmp_path, _marginal(rmse), without)
    with pytest.raises(SystemExit, match="resampling unit"):
        producer.read(marginal, paired, ARMS)


def test_a_marginal_report_that_names_no_clustering_is_refused(reports):
    """The bars' intervals are the marginal report's, so it must name its unit too."""
    rmse, tmp_path = reports
    without = {k: v for k, v in _marginal(rmse).items() if k != "unit"}
    marginal, paired = _write(tmp_path, without, _paired(rmse, -0.02))
    with pytest.raises(SystemExit, match="resampling unit"):
        producer.read(marginal, paired, ARMS)


@pytest.mark.parametrize("k,alpha,bonferroni,plus_one", [
    (0, True, True, True), (1, True, True, True), (2, True, True, False),
    (3, True, False, False), (40, True, False, False)])
def test_the_bonferroni_count_reads_the_paired_bootstrap(reports, k, alpha, bonferroni,
                                                         plus_one):
    """18 tests at a family-wise 5 %: a lead counts when fewer than 0.05/36 of the 2,000
    replicates do not favour DeepShift (k/B), or, beside it, when (k+1)/(B+1) is."""
    rmse, tmp_path = reports
    paired_report = _paired(rmse, -0.02)
    for source in ("xtal", "af", "nmr"):
        for nucleus in NUCLEI:
            paired_report[source][nucleus]["ucbshift_full_sameprot"][
                "p_deepshift_better"] = 1 - k / 2000
    marginal, paired = _write(tmp_path, _marginal(rmse), paired_report)
    report, paired_report = producer.read(marginal, paired, ARMS)
    tests = lead_tests(report, paired_report, ARMS[1:])
    assert tests["n_tests"] == 18 and tests["n_deepshift_leads"] == 18
    assert {t["runner_up"] for t in tests["tests"]} == {"ucbshift_full_sameprot"}
    assert tests["n_significant_alpha"] == 18 * alpha
    assert tests["n_significant_bonferroni"] == 18 * bonferroni
    assert tests["n_significant_bonferroni_plus_one"] == 18 * plus_one
