"""The assignment-accuracy matching (Figure 5a) on small hand-made problems."""
import numpy as np

from benchmarks import assignment_accuracy as aa

SIGMA = np.array([1.0, 1.0, 1.0, 0.25, 0.4, 2.5])     # CA CB C HA H N


def _observed():
    """Four residues with distinct backbone shifts, one nucleus missing on the last."""
    obs = np.array([[56.0, 30.0, 176.0, 4.3, 8.2, 120.0],
                    [62.0, 70.0, 174.0, 4.6, 8.0, 115.0],
                    [45.0, np.nan, 174.5, 3.9, 8.4, 109.0],
                    [58.0, 40.0, 177.5, 4.1, 7.9, np.nan]])
    return obs


def test_exact_predictions_assign_every_residue():
    obs = _observed()
    assert aa.correct(obs, obs.copy(), SIGMA) == len(obs)


def test_swapped_predictions_cost_exactly_the_two_swapped_residues():
    obs = _observed()
    pred = obs.copy()
    pred[[0, 1]] = pred[[1, 0]]
    assert aa.correct(obs, pred, SIGMA) == len(obs) - 2


def test_a_pair_sharing_no_nucleus_costs_the_penalty():
    obs = np.array([[56.0, np.nan, np.nan, np.nan, np.nan, np.nan]])
    pred = np.array([[np.nan, 30.0, np.nan, np.nan, np.nan, np.nan]])
    assert aa.cost(obs, pred, SIGMA)[0, 0] == aa.PENALTY


def test_spin_system_features_carry_the_preceding_residue():
    obs = _observed()
    prev = np.array([-1, 0, 1, 2])
    spin = aa.features(obs, prev, spin=True)
    ca, cb, c = (aa.NUC.index(n) for n in ("CA", "CB", "C"))
    assert np.isnan(spin[0, 5:]).all()                     # no predecessor for the first residue
    assert spin[2, 5:].tolist() == [obs[1, ca], obs[1, cb], obs[1, c]]
