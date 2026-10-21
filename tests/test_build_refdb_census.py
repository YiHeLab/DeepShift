"""Table S5's agreement shares count a difference that sits exactly on a threshold."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "si_build"))


def test_a_difference_on_the_grid_counts_however_the_subtraction_rounds():
    census = pytest.importorskip("build_refdb_census")
    # 0.30 - 0.20 is 0.10000000000000003 in floating point, 0.05 - (-0.05) exactly 0.1;
    # both offsets are on their grids, so both differences are 0.10 ppm.
    deltas = [abs(0.30000000000000004 - 0.2), abs(0.05 - (-0.05)), 0.15]
    summary = census.describe(deltas)
    assert summary["within_0p10"] == pytest.approx(2 / 3)
    assert summary["within_0p05"] == 0.0
