"""Fold-0-only variance mapping and split-conformal prediction intervals."""
import json

import numpy as np
import pytest

from deepshift.calibration import (
    BACKBONE_NUCLEI,
    conformal_quantile,
    evaluate_interval_calibration,
    fit_interval_calibration,
    halfwidth,
    load_calibration,
    member_spread,
)
from deepshift.prediction import interval_table


@pytest.fixture
def fitted():
    spread = np.linspace(0.05, 1.0, 200)
    sign = np.where(np.arange(spread.size) % 2, -1.0, 1.0)
    errors, spreads = {}, {}
    for index, nucleus in enumerate(BACKBONE_NUCLEI, start=1):
        spreads[nucleus] = spread * (1.0 + index / 20.0)
        errors[nucleus] = sign * np.sqrt((0.15 * index) ** 2 + 1.7 * spreads[nucleus] ** 2)
    return fit_interval_calibration(errors, spreads)


def test_member_spread_is_ddof_one():
    members = np.asarray([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]])
    np.testing.assert_allclose(member_spread(members), members.std(axis=0, ddof=1))


def test_conformal_quantile_uses_the_finite_sample_index():
    # n=10, level=.8 -> ceil(11*.8)=9, so this is the ninth ordered score, not an
    # interpolated percentile.
    assert conformal_quantile(np.arange(1.0, 11.0), np.ones(10), 0.80) == 9.0


def test_fit_is_fold_zero_only_and_round_trips(fitted, tmp_path):
    payload = fitted.to_mapping()
    assert payload["calibration_fold"] == 0
    assert payload["reserved_evaluation_fold"] == 6
    assert payload["member_folds"] == [1, 2, 3, 4, 5]
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))
    assert load_calibration(path).to_mapping() == payload


def test_old_pooled_constants_are_refused():
    with pytest.raises(ValueError, match="schema"):
        load_calibration({"constants": {}})


def test_halfwidth_requires_and_uses_the_explicit_artifact(fitted):
    constants = fitted.constants["CA"]
    sd = 0.4
    expected = constants["q"]["0.90"] * np.sqrt(
        constants["floor_ppm"] ** 2 + constants["gain"] * sd ** 2
    )
    assert halfwidth("CA", sd, calibration=fitted) == pytest.approx(expected)
    assert np.isnan(halfwidth("CG", 0.3, calibration=fitted))
    with pytest.raises(TypeError, match="calibration"):
        halfwidth("CA", sd)


def test_fold_six_evaluation_is_separate(fitted):
    spread = {nucleus: np.linspace(0.1, 0.8, 40) for nucleus in BACKBONE_NUCLEI}
    errors = {nucleus: np.linspace(-0.5, 0.5, 40) for nucleus in BACKBONE_NUCLEI}
    report = evaluate_interval_calibration(fitted, errors, spread)
    assert report["evaluation_fold"] == 6
    assert set(report["results"]) == set(BACKBONE_NUCLEI)


def test_interval_table_cells_and_spread(fitted):
    columns = ["ca_shift", "cb_shift", "cg_shift"]
    stack = np.random.default_rng(0).normal(size=(5, 2, 3))
    table = interval_table(
        stack, [1, 2], ["ALA", "GLY"], columns, calibration=fitted
    )
    # GLY has no CB and cg_shift carries no interval: exactly ALA CA, ALA CB, GLY CA.
    assert sorted(zip(table.residue_name, table.nucleus, strict=True)) == [
        ("ALA", "CA"), ("ALA", "CB"), ("GLY", "CA")]
    row = table[(table.residue_name == "ALA") & (table.nucleus == "CA")].iloc[0]
    sd = stack[:, 0, 0].std(ddof=1)
    assert row.ensemble_sd_ppm == pytest.approx(sd)
    assert row.ci90_halfwidth_ppm == pytest.approx(
        float(halfwidth("CA", sd, calibration=fitted))
    )


def _population(label_sets):
    from test_evaluate import _population as population

    rows = len(label_sets)
    true = np.full((rows, 2), 45.0)
    return population(true + 0.1, true, residue=np.zeros(rows, dtype=int),
                      label_set=list(label_sets))


def test_listed_label_sets_are_removed(tmp_path):
    from deepshift.calibration import exclude_label_sets, load_label_set_exclusion

    listing = tmp_path / "fold0_training_relatives.json"
    listing.write_text(json.dumps({
        "training_folds": [1, 2, 3, 4, 5],
        "edge_groups_spanning_fold0_and_training_by_family": {"exact_token": 1},
        "label_sets": ["ls1", "ls9"]}))
    exclusion = load_label_set_exclusion(listing)
    preds = _population(["ls0", "ls1", "ls1", "ls2"])
    keep, record = exclude_label_sets(preds, exclusion)
    assert keep.tolist() == [True, False, False, True]
    assert (record["label_sets_listed"], record["label_sets_removed"],
            record["label_sets_remaining"], record["rows_removed"], record["cells_removed"]) == (
        2, 1, 2, 2, 4)
    assert exclude_label_sets(preds, None) == (pytest.approx([True] * 4), None)


def test_the_artifact_carries_the_exclusion_record_and_evaluation_requires_it(fitted, tmp_path):
    from types import SimpleNamespace

    from deepshift.calibration import evaluation_command

    record = {"edge_groups_by_family": {}, "label_sets_listed": 3, "label_sets_removed": 2,
              "label_sets_remaining": 40, "rows_removed": 30, "rows_remaining": 700,
              "cells_removed": 120}
    with_record = fit_interval_calibration(
        {n: np.linspace(-0.5, 0.5, 200) for n in BACKBONE_NUCLEI},
        {n: np.linspace(0.05, 1.0, 200) for n in BACKBONE_NUCLEI},
        exclusion=record)
    assert load_calibration(with_record.to_mapping()).exclusion == record
    path = tmp_path / "no_exclusion.json"
    path.write_text(json.dumps(fitted.to_mapping()))
    with pytest.raises(ValueError, match="records no fold-0 exclusion"):
        evaluation_command(SimpleNamespace(calibration=str(path), member=[],
                                           exclude_channels="x", out="x"))


def test_equal_count_bins_stay_equal_count_when_spread_values_tie():
    """Spread bins are the sorted split the mapping is fitted over: every bin holds the
    same number of cells even where many cells share one spread, which quantile edges
    cannot guarantee."""
    from deepshift.calibration import equal_count_bins, variance_fit

    spread = np.concatenate([np.full(40, 0.2), np.linspace(0.3, 1.0, 60)])
    bins = equal_count_bins(spread, n_bins=5)
    assert [rows.size for rows in bins] == [20, 20, 20, 20, 20]
    assert sorted(int(row) for rows in bins for row in rows) == list(range(spread.size))
    means = [float(spread[rows].mean()) for rows in bins]
    assert means == sorted(means)

    errors = np.linspace(-1.0, 1.0, spread.size)
    _, fitted_bins = variance_fit(errors, spread, n_bins=5)
    assert [row["n"] for row in fitted_bins] == [20, 20, 20, 20, 20]
    assert [row["spread_mean_ppm"] for row in fitted_bins] == means


# ---------------------------------------------------------------------------
# From five saved prediction dumps to a fitted artifact
#
# Everything above starts from errors and spreads handed in directly. The paper's
# numbers -- Table S26 and the fold-6 coverage figure -- start from five npz dumps on
# disk, and the step that turns those into a population is where the protocol is
# actually enforced: five distinct dumps, one member per dump. Each guard below is checked both ways round: that it fires on the thing it
# names, and that the legitimate five-dump case passes it.
# ---------------------------------------------------------------------------

BACKBONE_COLUMNS = ("n_shift", "h_shift", "ca_shift", "cb_shift", "c_shift", "ha_shift")
#: Deviations of the five members from their own mean. They sum to zero, so the ensemble
#: mean is the intended prediction exactly, and their spread is a known multiple of the
#: per-cell spread the test asks for.
MEMBER_OFFSETS = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
FLOOR_PPM, GAIN = 0.2, 1.5


def _spread_and_error(rows, columns):
    """Per-cell spread, and the error the variance mapping implies for it."""
    spread = np.stack([
        np.linspace(0.05, 1.0, rows) * (1.0 + index / 20.0)
        for index in range(len(columns))], axis=1)
    sign = np.where(np.arange(rows) % 2, -1.0, 1.0)[:, None]
    return spread, sign * np.sqrt(FLOOR_PPM ** 2 + GAIN * spread ** 2)


def write_member_dumps(directory, *, fold, rows=60, columns=BACKBONE_COLUMNS,
                       member_folds=(1, 2, 3, 4, 5), mask=None):
    """Five single-member dumps over one shared population, with the spread across
    members and the ensemble error both set by construction."""
    from test_evaluate import _population as population

    from deepshift import evaluate

    directory.mkdir(parents=True, exist_ok=True)
    spread, error = _spread_and_error(rows, columns)
    truth = np.stack([np.full(rows, 45.0 + 10.0 * index) for index in range(len(columns))],
                     axis=1) + np.linspace(-2.0, 2.0, rows)[:, None]
    mean_prediction = truth + error
    mask = np.ones((rows, len(columns)), dtype=bool) if mask is None else mask
    paths = []
    for index, member in enumerate(member_folds):
        member_prediction = mean_prediction + MEMBER_OFFSETS[index] * spread / np.std(
            MEMBER_OFFSETS, ddof=1)
        preds = population(
            member_prediction, np.where(mask, truth, 0.0),
            residue=np.zeros(rows, dtype=int), mask=mask, columns=columns,
            label_set=[f"ls{i // 10}" for i in range(rows)],
            keys=[f"row{i}" for i in range(rows)])
        paths.append(evaluate.write_predictions(
            preds, directory / f"fold{fold}_member{member}.preds.npz"))
    return paths, spread, error


@pytest.fixture
def fold0_dumps(tmp_path):
    paths, spread, error = write_member_dumps(tmp_path / "fold0", fold=0)
    return paths, spread, error


def test_the_population_is_the_ensemble_error_and_the_spread_across_the_five_members(
        fold0_dumps):
    """The two quantities every interval is built from. The error is the *ensemble*
    residual, not a member's, and the spread is the standard deviation across members --
    read one cell at a time, per nucleus, never pooled across nuclei."""
    from deepshift.calibration import calibration_population

    paths, spread, error = fold0_dumps
    errors, spreads, record = calibration_population(paths)

    assert set(errors) == set(spreads) == set(BACKBONE_NUCLEI)
    assert record is None
    for index, column in enumerate(BACKBONE_COLUMNS):
        nucleus = column[: -len("_shift")].upper()
        np.testing.assert_allclose(errors[nucleus], error[:, index])
        np.testing.assert_allclose(spreads[nucleus], spread[:, index])


def test_only_observed_cells_enter_the_calibration_population(tmp_path):
    """An interval fitted on cells the deposition never reported would be fitted on a
    fill value. The mask decides the population, per nucleus and not per row."""
    from deepshift.calibration import calibration_population

    mask = np.ones((60, len(BACKBONE_COLUMNS)), dtype=bool)
    mask[:7, BACKBONE_COLUMNS.index("cb_shift")] = False
    paths, _spread, error = write_member_dumps(tmp_path / "masked", fold=0, mask=mask)
    errors, spreads, _record = calibration_population(paths)
    assert errors["CB"].size == spreads["CB"].size == 53
    assert errors["CA"].size == 60
    np.testing.assert_allclose(errors["CB"], error[7:, BACKBONE_COLUMNS.index("cb_shift")])


def test_a_nucleus_outside_the_backbone_set_is_not_given_an_interval(tmp_path):
    """Intervals are published for six nuclei. A seventh column in the dump must be
    dropped, not fitted, and a sixth that is missing must stop the fit."""
    from deepshift.calibration import calibration_population

    with_extra, _spread, _error = write_member_dumps(
        tmp_path / "extra", fold=0, columns=(*BACKBONE_COLUMNS, "cg_shift"))
    errors, _spreads, _record = calibration_population(with_extra)
    assert set(errors) == set(BACKBONE_NUCLEI)

    short, _spread, _error = write_member_dumps(
        tmp_path / "short", fold=0, columns=tuple(c for c in BACKBONE_COLUMNS
                                                  if c != "cb_shift"))
    with pytest.raises(ValueError, match="exactly the backbone set"):
        calibration_population(short)


def test_the_fit_needs_five_distinct_dumps_and_not_one_dump_five_times(fold0_dumps):
    """Five references to one member would report a spread of zero on every cell and an
    interval that was the residual floor everywhere."""
    from deepshift.calibration import calibration_population

    paths, _spread, _error = fold0_dumps
    for wrong in (paths[:4], [*paths, paths[0]], [paths[0]] * 5):
        with pytest.raises(ValueError, match="five distinct member prediction dumps"):
            calibration_population(wrong)


def _channel_file(path, channels):
    path.write_text(json.dumps({"channels": [
        {"label_set": label_set, "nucleus": nucleus} for label_set, nucleus in channels]}))
    return path


def _exclusion_file(path, label_sets):
    path.write_text(json.dumps({
        "training_folds": [1, 2, 3, 4, 5],
        "edge_groups_spanning_fold0_and_training_by_family": {"exact_token": 1},
        "label_sets": list(label_sets)}))
    return path


def test_the_fold_zero_relatives_are_removed_from_the_population_they_were_listed_for(
        tmp_path, fold0_dumps):
    """Fold 0 fits the intervals, and a fold-0 label set with a training-fold relative is
    not held out in the sense the fit assumes."""
    from deepshift.calibration import calibration_population, load_label_set_exclusion

    paths, _spread, error = fold0_dumps
    exclusion = load_label_set_exclusion(_exclusion_file(tmp_path / "rel.json", ["ls0"]))
    errors, _spreads, record = calibration_population(paths, exclusion=exclusion)
    assert errors["CA"].size == 50
    np.testing.assert_allclose(errors["CA"], error[10:, BACKBONE_COLUMNS.index("ca_shift")])
    assert (record["label_sets_removed"], record["rows_removed"]) == (1, 10)


def test_the_fit_command_writes_a_fold_zero_artifact(
        tmp_path, fold0_dumps, capsys):
    """The whole fitting entry point, end to end: five dumps and the relatives list in,
    one artifact out."""
    from types import SimpleNamespace

    from deepshift.calibration import (
        BACKBONE_NUCLEI as NUCLEI,
    )
    from deepshift.calibration import (
        fit_command,
        load_calibration,
    )

    paths, _spread, _error = fold0_dumps
    out = tmp_path / "interval_calibration.json"
    fitted = fit_command(SimpleNamespace(
        member=[str(p) for p in paths],
        exclude_label_sets=str(_exclusion_file(tmp_path / "rel.json", ["ls0"])),
        bins=10, level=[0.68, 0.90, 0.95], out=str(out)))

    assert load_calibration(out).to_mapping() == fitted.to_mapping()
    assert set(fitted.constants) == set(NUCLEI)
    assert fitted.exclusion["label_sets_removed"] == 1
    for nucleus in NUCLEI:
        constants = fitted.constants[nucleus]
        assert constants["n_calibration_cells"] == 50
        assert set(constants["q"]) == {"0.68", "0.90", "0.95"}
        assert constants["floor_ppm"] > 0.0 and constants["gain"] > 0.0
    assert "fold 0 minus 1 label sets" in capsys.readouterr().out


def test_the_evaluation_command_reads_fold_six_once_against_the_frozen_artifact(
        tmp_path, fold0_dumps, capsys):
    """The one direction the protocol allows: fit on fold 0, then read fold 6 -- the same
    five checkpoints, the same build -- against the frozen constants."""
    from types import SimpleNamespace

    from deepshift.calibration import evaluation_command, fit_command

    paths, _spread, _error = fold0_dumps
    artifact = tmp_path / "interval_calibration.json"
    fit_command(SimpleNamespace(
        member=[str(p) for p in paths],
        exclude_label_sets=str(_exclusion_file(tmp_path / "rel.json", ["ls0"])),
        bins=10, level=[0.68, 0.90, 0.95], out=str(artifact)))
    capsys.readouterr()

    fold6, _spread, _error = write_member_dumps(tmp_path / "fold6", fold=6)
    channels = _channel_file(tmp_path / "channels.json", [("ls1", "C"), ("ls2", "HA")])
    report = evaluation_command(SimpleNamespace(
        calibration=str(artifact), member=[str(p) for p in fold6],
        exclude_channels=str(channels), out=str(tmp_path / "evaluation.json")))

    assert report["evaluation_fold"] == 6 and report["calibration_fold"] == 0
    assert set(report["results"]) == set(BACKBONE_NUCLEI)
    fence = report["fold6_channel_exclusion"]
    assert (fence["channels_listed"], fence["channels_present"], fence["cells_removed"]) == (
        2, 2, 20)
    assert fence["cells_removed_by_nucleus"] == {"C": 10, "HA": 10}
    for nucleus, row in report["results"].items():
        assert row["n_evaluation_cells"] == (50 if nucleus in ("C", "HA") else 60)
        assert 0.0 <= row["levels"]["0.90"]["realized_coverage"] <= 1.0
        assert row["levels"]["0.95"]["median_half_width_ppm"] > (
            row["levels"]["0.68"]["median_half_width_ppm"])
    assert json.loads((tmp_path / "evaluation.json").read_text()) == report
    assert "fold 6 evaluated once" in capsys.readouterr().out


def test_the_fold_six_channel_fence_clears_only_the_listed_cells(tmp_path):
    """The evaluation drops the deposited channels every fold-6 benchmark drops: the
    listed nucleus of the listed label set, and nothing else of that label set."""
    from deepshift.calibration import (
        calibration_population,
        load_channel_exclusion,
        load_label_set_exclusion,
    )

    paths, _spread, error = write_member_dumps(tmp_path / "fold6", fold=6)
    channels = load_channel_exclusion(
        _channel_file(tmp_path / "channels.json", [("ls0", "C"), ("absent", "H")]))
    errors, _spreads, record = calibration_population(paths, channels=channels)
    column = BACKBONE_COLUMNS.index("c_shift")
    np.testing.assert_allclose(errors["C"], error[10:, column])
    assert errors["CA"].size == 60
    assert (record["channels_listed"], record["channels_present"],
            record["cells_removed"]) == (2, 1, 10)
    exclusion = load_label_set_exclusion(_exclusion_file(tmp_path / "rel.json", ["ls0"]))
    with pytest.raises(ValueError, match="one population takes one of them"):
        calibration_population(paths, exclusion=exclusion, channels=channels)
    with pytest.raises(ValueError, match="backbone nucleus"):
        load_channel_exclusion(_channel_file(tmp_path / "bad.json", [("ls0", "CG")]))
