"""Boundaries and composition of the final exclusion policy."""
import pandas as pd
import pytest

from deepshift.chemistry import CHEMICALLY_IMPOSSIBLE_PAIRS
from deepshift.data import cache
from deepshift.geometry_context import (
    CYS_OXIDIZED_MIN_PPM,
    CYS_REDUCED_MAX_PPM,
    OMEGA_CIS_MAX_DEGREES,
    OMEGA_TRANS_MIN_DEGREES,
    PRO_CIS_MIN_PPM,
    PRO_TRANS_MAX_PPM,
    cysteine_shift_state,
    proline_geometry_state,
    proline_shift_state,
)
from deepshift.quality_control import (
    ARG_NE_OBSERVED_RANGE_PPM,
    P100Policy,
    load_p100_cutoffs,
)


def test_p100_applies_the_fitted_span_to_every_nucleus_and_the_interval_to_arg_ne():
    """Hα, glycine's two Hα included, takes the table's fitted span like every column;
    only Arg Nε is an observed-value interval fixed outside the fit."""
    policy = P100Policy({
        "ALA:ha_shift": {"lo": 3.1, "hi": 5.9},
        "GLY:ha2_shift": {"lo": 2.8, "hi": 4.9},
        "ARG:ne_shift": {"lo": -100.0, "hi": 100.0},
        "ALA:ca_shift": {"lo": 45.0, "hi": 65.0},
    })
    assert policy.accepted_range("ALA", "ha_shift") == (3.1, 5.9)
    assert policy.accepted_range("GLY", "ha2_shift") == (2.8, 4.9)
    assert policy.excludes("ALA", "ha_shift", 6.0) and not policy.excludes("ALA", "ha_shift", 5.9)
    assert policy.accepted_range("ARG", "ne_shift") == ARG_NE_OBSERVED_RANGE_PPM
    assert not policy.excludes("ARG", "ne_shift", 78.976)
    assert not policy.excludes("ARG", "ne_shift", 90.212)
    assert policy.excludes("ARG", "ne_shift", 90.213)


def test_chemically_impossible_pair_is_schema_masked_not_p100():
    assert ("PRO", "h_shift") in CHEMICALLY_IMPOSSIBLE_PAIRS
    arrays = cache._allocate_fold_arrays(1, 1)
    frame = pd.DataFrame({"residue_code": ["PRO"], "h_shift": [8.5]})
    cache._bake_shift_columns(
        arrays, frame, 0, ("mz", "bmrb", "nmr", "label-set"),
        ["h_shift"], {}, {},
    )
    assert not arrays.shift_mask[0, 0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (CYS_REDUCED_MAX_PPM - 1e-6, "reduced"),
        (CYS_REDUCED_MAX_PPM, None),
        (CYS_OXIDIZED_MIN_PPM, None),
        (CYS_OXIDIZED_MIN_PPM + 1e-6, "oxidized"),
    ],
)
def test_cysteine_shift_boundaries(value, expected):
    assert cysteine_shift_state(value) == expected


@pytest.mark.parametrize(
    ("difference", "expected"),
    [
        (PRO_TRANS_MAX_PPM, "trans"),
        (PRO_TRANS_MAX_PPM + 1e-6, None),
        (PRO_CIS_MIN_PPM - 1e-6, None),
        (PRO_CIS_MIN_PPM, "cis"),
    ],
)
def test_proline_shift_boundaries(difference, expected):
    assert proline_shift_state(difference, 0.0) == expected


@pytest.mark.parametrize(
    ("omega", "expected"),
    [
        (OMEGA_CIS_MAX_DEGREES, "cis"),
        (OMEGA_CIS_MAX_DEGREES + 1e-6, None),
        (OMEGA_TRANS_MIN_DEGREES - 1e-6, None),
        (OMEGA_TRANS_MIN_DEGREES, "trans"),
    ],
)
def test_proline_geometry_boundaries(omega, expected):
    assert proline_geometry_state(omega) == expected


def test_the_p100_authority_is_a_run_output_with_no_packaged_default():
    with pytest.raises(TypeError):
        load_p100_cutoffs()
