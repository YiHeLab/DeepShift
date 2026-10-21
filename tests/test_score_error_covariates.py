"""Table S24's strata: every block but the two source-restricted ones partitions the cells."""
import numpy as np

from benchmarks.score_error_covariates import strata


def test_a_residue_with_no_aromatic_ring_in_its_structure_is_in_the_farthest_band():
    """A structure with no aromatic ring of another residue gives an infinite distance.
    That residue is farther from a ring than any band edge, so it belongs to the
    farthest band; dropping it left the ring block 2,979 cells short of the cohort."""
    ring = np.array([3.0, 5.5, 7.0, 10.0, 15.0, np.inf])
    n = ring.size
    blocks = {key: rows for key, _title, rows in strata(
        np.array(["af"] * n), np.full(n, 90.0), np.full(n, np.nan), ring,
        np.full(n, 0.3), np.ones(n, int))}
    masks = [mask for _name, mask in blocks["ring"]]
    assert np.array_equal(np.sum(masks, axis=0), np.ones(n, int))
    assert masks[-1][-1] and masks[-1][-2]


def test_a_residue_with_no_geometry_is_in_no_ring_band():
    """A missing covariate is not a distance, and stays out rather than counting as far."""
    ring = np.array([np.nan])
    blocks = {key: rows for key, _title, rows in strata(
        np.array(["nmr"]), np.full(1, np.nan), np.full(1, np.nan), ring,
        np.full(1, 0.3), np.ones(1, int))}
    assert not any(mask[0] for _name, mask in blocks["ring"])
