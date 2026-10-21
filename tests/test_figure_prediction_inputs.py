"""The supplementary figures read the prediction dumps the evaluator writes."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from deepshift import evaluate
from deepshift.chemistry import RESIDUE_TO_IDX

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"

COLUMNS = ("n_shift", "h_shift", "ca_shift", "cb_shift", "c_shift", "ha_shift", "ne_shift")
CENTRE = {"n_shift": 120.0, "h_shift": 8.3, "ca_shift": 55.0, "cb_shift": 33.0,
          "c_shift": 176.0, "ha_shift": 4.4, "ne_shift": 84.0}
MEMBERS = (1, 2, 3, 4, 5)
FOLD = 6


@pytest.fixture(scope="module")
def figures_on_path():
    if str(FIGURES) not in sys.path:
        sys.path.insert(0, str(FIGURES))
    return FIGURES


def _predictions(seed: int, n_rows: int = 240):
    """One member's held-out predictions over a small cohort of ARG and ALA rows."""
    rng = np.random.default_rng(seed)
    residues = np.array([RESIDUE_TO_IDX["ARG"], RESIDUE_TO_IDX["ALA"]] * (n_rows // 2))
    true = np.stack([np.full(n_rows, CENTRE[column]) + rng.normal(0.0, 0.3, n_rows)
                     for column in COLUMNS], axis=1)
    pred = true + rng.normal(0.0, 0.25, true.shape)
    mask = np.ones(true.shape, dtype=bool)
    mask[:, COLUMNS.index("ne_shift")] = residues == RESIDUE_TO_IDX["ARG"]
    ambiguity = np.zeros(true.shape, dtype=np.int64)
    resolved, effective = evaluate.resolve(pred, true, mask, ambiguity, residues, COLUMNS)
    return evaluate.Predictions(
        shift_cols=COLUMNS, pred_ppm=pred, true_ppm=true, resolved_ppm=resolved,
        mask=mask, effective_mask=effective, ambiguity=ambiguity, residue_code=residues,
        row_key=np.array([f"mz{i}|lr_{i}" for i in range(n_rows)]),
        label_set=np.array([f"ls{i % 12}" for i in range(n_rows)]),
        bmrb=np.array([f"{10000 + i % 12}" for i in range(n_rows)]),
        source=np.zeros(n_rows, dtype=np.int64))


@pytest.fixture
def member_dumps(tmp_path):
    """Five members' dumps, written by the evaluator under the run's own file names."""
    head = _predictions(seed=0)
    for member in MEMBERS:
        shifted = evaluate.replace(
            head, pred_ppm=head.pred_ppm + 0.02 * member * np.abs(head.pred_ppm - head.true_ppm))
        evaluate.write_predictions(shifted, tmp_path / f"fold{FOLD}_member{member}.npz")
    return tmp_path


def test_the_accuracy_figure_reads_dumps_the_evaluator_wrote(figures_on_path, member_dumps):
    """Figure S3's inputs are canonical prediction dumps: it resolves them by name, reads
    them through ``read_predictions`` and scores their ensemble."""
    frozen = importlib.import_module("frozen")
    fig_accuracy = importlib.import_module("fig_accuracy_by_nucleus_all")

    paths = frozen.member_dumps(FOLD, MEMBERS, directory=member_dumps)
    assert [p.name for p in paths] == [f"fold{FOLD}_member{m}.npz" for m in MEMBERS]
    preds = evaluate.ensemble([evaluate.read_predictions(path) for path in paths])

    assert fig_accuracy.alias_report(preds) is None      # every Arg Nε label is in range
    scored = evaluate.score(preds, columns=["ca_shift"], boots=8, seed=1, group="bmrb")
    assert scored["ca_shift"]["n"] == preds.effective_mask[:, COLUMNS.index("ca_shift")].sum()


def test_the_uncertainty_figure_reads_dumps_the_evaluator_wrote(figures_on_path, member_dumps):
    """Figure S10's inputs are canonical prediction dumps: member spread and the binned
    error come from what ``read_predictions`` returns."""
    frozen = importlib.import_module("frozen")
    fig_uncertainty = importlib.import_module("fig_uncertainty_calibration")
    from deepshift.calibration import member_spread

    paths = frozen.member_dumps(FOLD, MEMBERS, directory=member_dumps)
    parts = [evaluate.read_predictions(path) for path in paths]
    spread = member_spread(np.stack([part.pred_ppm for part in parts]))
    preds = evaluate.ensemble(parts)

    column = "ca_shift"
    mapping = {"floor_ppm": 0.2, "gain": 2.0}
    rows = fig_uncertainty.calibration_bins(
        preds, column, spread[:, COLUMNS.index(column)], mapping,
        n_bins=4, boots=8, seed=1)
    assert [row["bin"] for row in rows] == [1, 2, 3, 4]
    assert sum(row["n"] for row in rows) == preds.effective_mask[:, COLUMNS.index(column)].sum()


def test_the_accuracy_grid_reads_dumps_the_evaluator_wrote(figures_on_path, member_dumps,
                                                           monkeypatch):
    """Figure 2's panels are the run's five member dumps, ensembled in ppm by the
    package: the columns, the mask and the targets are the ones ``read_predictions``
    returns, and no loader in this repository opens a dump by any other name."""
    monkeypatch.setenv("DEEPSHIFT_RUN", str(member_dumps.parent))
    frozen = importlib.import_module("frozen")
    monkeypatch.setattr(frozen, "MEMBER_DUMPS_DIR", member_dumps)
    build = importlib.import_module("build_paper_figures")

    assert not hasattr(build, "load_ensemble")
    assert not hasattr(build, "SOURCES")

    paths = frozen.member_dumps(FOLD, MEMBERS)
    ensemble = evaluate.ensemble([evaluate.read_predictions(path) for path in paths])
    assert list(ensemble.shift_cols) == list(COLUMNS)
    column = ensemble.shift_cols.index("ca_shift")
    drawn = build.stats(ensemble.pred_ppm[ensemble.effective_mask[:, column], column],
                        ensemble.resolved_ppm[ensemble.effective_mask[:, column], column])
    assert drawn["n"] == int(ensemble.effective_mask[:, column].sum())


def test_the_figure_inputs_resolve_under_the_run(figures_on_path, monkeypatch):
    """Figures 1, 2, S7 and S8 resolve under the roots the environment names."""
    monkeypatch.setenv("DEEPSHIFT_RUN", "/run")
    monkeypatch.setenv("DEEPSHIFT_SCALING_LADDER", "/run/campaign/ladder")
    monkeypatch.setenv("DEEPSHIFT_RAMACHANDRAN_OOF", "/run/campaign/ramachandran")
    from deepshift.data import paths
    importlib.reload(paths)
    for module in ("frozen", "build_paper_figures"):
        sys.modules.pop(module, None)
    build = importlib.import_module("build_paper_figures")

    # /run holds no checkpoint, so Figure 1 reads the shipped fold-1 member, which carries
    # the same tensors; with the run's own file present it reads that.
    assert build.frozen.REPO / "models" / "deepshift_fold1.pt" == build.FOLD1_CHECKPOINT
    assert Path("/run/campaign/ladder") == build.SCALING_LADDER
    assert Path("/run/campaign/ramachandran/cv_cells.npz") == build.RAMACHANDRAN_CELLS
    sys.modules.pop("frozen", None)
    sys.modules.pop("build_paper_figures", None)
    monkeypatch.undo()
    importlib.reload(paths)
