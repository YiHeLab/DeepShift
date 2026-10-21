"""Table S23 marks an arm scored on its own, smaller cohort and prints that cohort.

PLM-CS is withheld on the proteins it trained on, so its rows are scored on fewer cells
than the cohort line above them counts. Unmarked, a reader takes the panel's cohort and
observed split for its rows too.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "si_build"))

import emit_cys_modes as emit


def _arm(n_dis, n_free, mean_dis, mean_free):
    return {"n_cells": n_dis + n_free,
            "disulfide": {"n_cells": n_dis, "mean_observed_ppm": mean_dis},
            "free_thiol": {"n_cells": n_free, "mean_observed_ppm": mean_free}}


def test_a_withheld_arm_is_footnoted_with_its_own_cells_and_split():
    report = {"withheld_as_training_data": {"plmcs": {"depositions": 320}},
              "by_source": {"all": {"arms": {"plmcs": _arm(453, 651, 40.9077, 28.8399)}},
                            "af": {"arms": {"plmcs": _arm(157, 282, 40.9591, 28.9361)}}}}
    note = emit.partial_cohort(report, "plmcs")
    assert note.startswith("PLM-CS is not scored on the structures of proteins it was trained on")
    assert "1,104 cells (453 disulfide, 651 free thiol; observed split 12.07~ppm)" in note
    assert "AlphaFold structures only, 439 cells (157 disulfide, 282 free thiol" in note
    assert r"\cite" not in note
