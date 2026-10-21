"""Figure S2 scores the six nuclei it reports and nothing else.

The dump the figure reads carries cells outside those six. They have no cohort RMS of
their own, so a structure's error must be the error over its scored cells alone: a cell
the figure never scaled cannot be pooled into it, and cannot depend on what happened to
be in the array it was written to.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))

import fig_homology_training_identity as fig


def dump(unscored_residual):
    """Two structures of six scored cells each, plus one cell of an unreported nucleus.

    The unreported cell is given a residual far larger than any scored one, so pooling it
    would be visible in the second structure's error.
    """
    atom = np.array(fig.BACKBONE * 2 + ["HA2"])
    resid = np.concatenate([np.full(len(fig.BACKBONE), 1.0),
                            np.full(len(fig.BACKBONE), 3.0),
                            [unscored_residual]])
    return {"atom": atom, "resid": {"deepshift": resid}}


def test_a_nucleus_the_figure_does_not_report_is_not_pooled_into_a_structures_error():
    """The scaled residual of an unreported cell is not defined, so the structure it sits
    in must score the same as one that has no such cell."""
    struct_code = np.array([0] * len(fig.BACKBONE) + [1] * len(fig.BACKBONE) + [1])
    scored_only = fig.relative_error(
        {"atom": np.array(fig.BACKBONE * 2), "resid": {"deepshift": np.concatenate(
            [np.full(len(fig.BACKBONE), 1.0), np.full(len(fig.BACKBONE), 3.0)])}},
        "deepshift", struct_code[:-1], 2)

    for unscored_residual in (0.0, 1e6, -1e6):
        with_unscored = fig.relative_error(dump(unscored_residual), "deepshift",
                                           struct_code, 2)
        assert np.allclose(with_unscored, scored_only)


def test_the_scored_cells_are_the_backbone_cells():
    """The cell count the figure prints is of the six backbone nuclei."""
    assert int(fig.scored_cells(dump(1e6)).sum()) == 2 * len(fig.BACKBONE)
