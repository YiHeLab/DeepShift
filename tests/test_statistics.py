import numpy as np
import pytest

from deepshift.statistics import (
    BootstrapSpec,
    bootstrap_indices,
    error_replicates,
    paired_verdict,
    secondary_shift_agreement,
    secondary_shift_replicates,
    weighted_median_replicates,
)


def test_a_paired_interval_favours_the_arm_it_lies_on_the_side_of():
    """The interval is on baseline minus candidate error, so an interval below zero
    favours the baseline and one spanning zero separates nothing."""
    assert paired_verdict((-0.4, -0.1)) == "baseline"
    assert paired_verdict((0.1, 0.4)) == "candidate"
    assert paired_verdict((-0.1, 0.4)) == "tie"
    assert paired_verdict((0.0, 0.4)) == "tie"


def test_index_draws_preserve_the_declared_seed_stream():
    spec = BootstrapSpec(unit="label_set", resamples=12, seed=20260806)
    expected = np.random.default_rng(20260806).integers(0, 5, size=(12, 5))
    assert np.array_equal(bootstrap_indices(5, spec), expected)


def test_error_replicates_equal_direct_resampled_cells():
    residual = np.array([1.0, -2.0, 3.0, 4.0, -1.0])
    unit = np.array([0, 0, 1, 2, 2])
    draws = np.array([[0, 1, 2], [2, 2, 0], [1, 1, 1]])
    point_rmse, rmse, point_mae, mae = error_replicates(
        residual, unit, 3, draws
    )
    expanded = [
        np.concatenate([residual[unit == selected] for selected in draw])
        for draw in draws
    ]
    assert point_rmse == np.sqrt(np.mean(residual**2))
    assert point_mae == np.mean(np.abs(residual))
    assert np.allclose(rmse, [np.sqrt(np.mean(values**2)) for values in expanded])
    assert np.allclose(mae, [np.mean(np.abs(values)) for values in expanded])


def test_weighted_median_replicates_equal_expanded_multisets():
    values = np.array([5.0, 1.0, 3.0, 4.0, 9.0])
    unit = np.array([0, 0, 1, 2, 2])
    multiplicities = np.array([[1, 1, 1], [2, 0, 1], [0, 3, 0]])
    expanded = [
        np.concatenate(
            [np.tile(values[unit == index], count) for index, count in enumerate(row)]
        )
        for row in multiplicities
    ]
    expected_mean = np.array([np.median(row) for row in expanded])
    expected_lower = np.array(
        [np.sort(row)[(len(row) - 1) // 2] for row in expanded]
    )
    assert np.array_equal(
        weighted_median_replicates(values, unit, multiplicities, even="mean"),
        expected_mean,
    )
    assert np.array_equal(
        weighted_median_replicates(values, unit, multiplicities, even="lower"),
        expected_lower,
    )


def test_secondary_shift_replicates_equal_the_agreement_on_the_resampled_cells():
    """The cluster bootstrap of the correlation is the correlation of the drawn cells:
    the sums a replicate adds up are the centred cells it selected."""
    rng = np.random.default_rng(20260904)
    residue_code = rng.integers(0, 3, 40)
    observed = rng.normal(size=40) * 2 + residue_code
    predicted = observed * 0.8 + rng.normal(size=40) * 0.4
    unit = rng.integers(0, 8, 40)
    draws = np.array([np.arange(8), [0, 0, 1, 2, 3, 4, 5, 6], [1, 1, 1, 2, 2, 3, 4, 5]])

    correlation, amplitude = secondary_shift_replicates(
        predicted, observed, residue_code, unit, draws
    )
    for index, draw in enumerate(draws):
        cells = np.concatenate([np.flatnonzero(unit == selected) for selected in draw])
        centred_pred, centred_obs = predicted.copy(), observed.copy()
        for code in np.unique(residue_code):          # the cohort's centring, once
            rows = residue_code == code
            centred_obs[rows] -= observed[rows].mean()
            centred_pred[rows] -= observed[rows].mean()
        x, y = centred_pred[cells], centred_obs[cells]
        assert correlation[index] == pytest.approx(np.corrcoef(x, y)[0, 1])
        assert amplitude[index] == pytest.approx(x.std() / y.std())

    # The full draw is the point estimate the package reports.
    point, point_amplitude, _ = secondary_shift_agreement(predicted, observed, residue_code)
    assert correlation[0] == pytest.approx(point)
    assert amplitude[0] == pytest.approx(point_amplitude)
