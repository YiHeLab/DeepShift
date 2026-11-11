"""The perturber-dose float carries one caption and no placeholder, so a float with a
caption does not also typeset a second note beside it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "si_build"))

import analyze_perturber as ap
import emit_perturber_dose_ppm as emit

BASELINE_ROWS = 141972


@pytest.fixture
def table():
    base = {n: {"rmse_ppm": 0.42} for n in emit.BACKBONE}
    cells = {f"{cat}|{label}": {"n_rows": 7,
                                "nuclei": {n: {"rmse_ppm": 0.5} for n in emit.BACKBONE}}
             for cat in ap.CATEGORIES for _lo, _hi, label in ap.BINS}
    return emit.build_table(cells, base, BASELINE_ROWS, {})


def printed(tex):
    return "\n".join(line for line in tex.splitlines() if not line.startswith("%"))


def test_the_float_prints_one_caption_and_no_placeholder(table):
    page = printed(table)
    assert page.count(r"\caption{") == 1
    assert r"\capneeded" not in page


def test_every_measured_perturber_category_reaches_the_table(table):
    for category in ap.CATEGORIES:
        assert emit.PRETTY[category] in table
