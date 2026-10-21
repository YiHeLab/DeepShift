"""The three-source matched scorer: which cells enter an arm, and what a verdict says.

The population below is built so that every answer is known by construction: each arm's
residual is one constant, so a nucleus' RMSE is that constant and a cell that should have
been excluded shows up as a number that is not it.
"""
import numpy as np
import pytest
from test_evaluate import _dump

from benchmarks import score_source_matched
from deepshift import cohorts, evaluate, statistics
from deepshift.domain import STANDARD_RESIDUES, VALID_SOURCES
from deepshift.statistics import DEFAULT_SEED, BootstrapSpec

COLUMNS = ("ca_shift", "cb_shift")
SITES = 12                       # label residues, above the package's scorable floor
SITES_PER_LABEL_SET = 3
OFFSET = {"af": 0.1, "nmr": 0.2, "xtal": 0.3}          # ppm, one residual per source
UNMATCHED = 0                    # the label residue whose cb the crystal arm cannot score
OUTLIER_PPM = 10.0               # what that cell would contribute to the AlphaFold arm


def _matched_population(sites=SITES):
    """Every label residue realized by all three sources, and one cell that is not."""
    residue = STANDARD_RESIDUES.index("ALA")
    rows = [(code, name, site) for code, name in enumerate(VALID_SOURCES)
            for site in range(sites)]
    true = np.tile([55.0, 30.0], (len(rows), 1))
    pred, mask = true.copy(), np.ones_like(true, dtype=bool)
    for index, (_code, name, site) in enumerate(rows):
        pred[index] += OFFSET[name]
        if site == UNMATCHED:
            mask[index, 1] = name != "xtal"
            pred[index, 1] = true[index, 1] + (OUTLIER_PPM if name == "af"
                                               else OFFSET[name])
    ambiguity = np.zeros_like(true, dtype=np.int64)
    residue_code = np.full(len(rows), residue)
    resolved, effective = evaluate.resolve(pred, true, mask, ambiguity, residue_code,
                                           COLUMNS)
    label_residue = np.asarray([f"lr_{site}" for _code, _name, site in rows])
    population = evaluate.Predictions(
        shift_cols=COLUMNS, pred_ppm=pred, true_ppm=true, resolved_ppm=resolved,
        mask=mask, effective_mask=effective, ambiguity=ambiguity,
        residue_code=residue_code,
        row_key=np.asarray([f"mz_{name}|lr_{site}" for _code, name, site in rows]),
        label_set=np.asarray([f"ls_{site // SITES_PER_LABEL_SET}"
                              for _code, _name, site in rows]),
        bmrb=np.asarray([str(site // SITES_PER_LABEL_SET) for _code, _name, site in rows]),
        source=np.asarray([code for code, _name, _site in rows]))
    return population, label_residue


def _bootstrap(resamples=64):
    return BootstrapSpec(unit="label_set", resamples=resamples, seed=DEFAULT_SEED)


def _report(population, label_residue):
    return score_source_matched.report(population, label_residue, columns=list(COLUMNS),
                                       bootstrap=_bootstrap())


def test_a_cell_only_two_sources_realize_is_scored_in_none_of_them():
    """The crystal arm cannot score one cb cell, so no arm scores it: the AlphaFold arm's
    cb error is its constant offset and not the outlier that cell carries."""
    per_source = _report(*_matched_population())["nuclei"]["cb_shift"]["per_source"]
    assert [per_source[name]["n_cells"] for name in VALID_SOURCES] == [SITES - 1] * 3
    assert per_source["af"]["rmse"] == pytest.approx(OFFSET["af"])
    assert per_source["af"]["rmse_ci95"] == pytest.approx([OFFSET["af"], OFFSET["af"]])


def test_a_cell_every_source_realizes_enters_each_arm_once():
    node = _report(*_matched_population())["nuclei"]["ca_shift"]
    assert [node["per_source"][name]["n_cells"] for name in VALID_SOURCES] == [SITES] * 3
    assert node["n_matched_sites"] == SITES
    for name in VALID_SOURCES:
        assert node["per_source"][name]["rmse"] == pytest.approx(OFFSET[name])


def test_n_matched_sites_counts_label_residues_not_cells():
    report = _report(*_matched_population())
    scored_cells = sum(node["per_source"]["af"]["n_cells"]
                       for node in report["nuclei"].values())
    assert scored_cells == len(COLUMNS) * SITES - 1
    assert report["n_matched_sites"] == SITES
    assert report["nuclei"]["cb_shift"]["n_matched_sites"] == SITES - 1
    assert report["nuclei"]["ca_shift"]["n_label_sets"] == SITES // SITES_PER_LABEL_SET


def test_the_paired_difference_flips_sign_with_the_arms_and_names_one_source():
    """A verdict is a statement about coordinates, not about argument order: reversing
    the arms reverses the difference and leaves the same source named."""
    population, label_residue = _matched_population()
    pair = _report(population, label_residue)["nuclei"]["ca_shift"]["paired"]["af_vs_xtal"]
    assert pair["delta_rmse"] == pytest.approx(OFFSET["af"] - OFFSET["xtal"])
    assert pair["verdict"] == "af"

    cells, _ = cohorts.matched_cells(population, label_residue)
    arms = score_source_matched.source_arms(population, cells, label_residue)
    reversed_pair = score_source_matched.paired_verdict("xtal", "af", statistics.compare(
        arms["xtal"], arms["af"], bootstrap=_bootstrap(), columns=["ca_shift"])["ca_shift"])
    assert reversed_pair["delta_rmse"] == pytest.approx(-pair["delta_rmse"])
    assert reversed_pair["verdict"] == "af"
    assert reversed_pair["rmse"] == pair["rmse"]


def test_the_command_line_refuses_four_dumps(tmp_path):
    population, _label_residue = _matched_population()
    dumps = [str(evaluate.write_predictions(population, tmp_path / f"fold{fold}.preds.npz"))
             for fold in (1, 2, 3, 4)]
    with pytest.raises(SystemExit):
        score_source_matched.main(["--preds", *dumps, "--out", str(tmp_path / "out.json")])


def test_a_dump_carries_the_label_residue_of_every_row(tmp_path):
    """Matching rows across structure sources needs the label residue each row was scored
    against, which the dump stores beside the materialization."""
    population, label_residue = _matched_population()
    path = _dump(tmp_path / "fold1.preds.npz", population, label_residue_id=label_residue)
    assert np.array_equal(evaluate.read_label_residues(path), label_residue)
