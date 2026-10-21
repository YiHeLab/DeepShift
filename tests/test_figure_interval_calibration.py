"""Figure S10 reads the fold-0 interval artifact and fits nothing on fold 6."""
import importlib
import sys
from pathlib import Path

import matplotlib
import matplotlib.figure
import numpy as np
import pytest

matplotlib.use("Agg")

from deepshift import calibration, evaluate
from deepshift.chemistry import RESIDUE_TO_IDX

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"

COLUMNS = ("n_shift", "h_shift", "ca_shift", "cb_shift", "c_shift", "ha_shift")
CENTRE = {"n_shift": 120.0, "h_shift": 8.3, "ca_shift": 55.0, "cb_shift": 33.0,
          "c_shift": 176.0, "ha_shift": 4.4}


@pytest.fixture(scope="module")
def figures_on_path():
    if str(FIGURES) not in sys.path:
        sys.path.insert(0, str(FIGURES))
    return FIGURES


def _member(seed: int, offset: float, n_rows: int = 300):
    """One member's fold-6 predictions; ``offset`` scales that member's disagreement."""
    rng = np.random.default_rng(seed)
    residues = np.array([RESIDUE_TO_IDX["ALA"], RESIDUE_TO_IDX["ARG"]] * (n_rows // 2))
    true = np.stack([np.full(n_rows, CENTRE[column]) + rng.normal(0.0, 0.4, n_rows)
                     for column in COLUMNS], axis=1)
    pred = true + rng.normal(0.0, 0.3, true.shape) + offset
    mask = np.ones(true.shape, dtype=bool)
    ambiguity = np.zeros(true.shape, dtype=np.int64)
    resolved, effective = evaluate.resolve(pred, true, mask, ambiguity, residues, COLUMNS)
    return evaluate.Predictions(
        shift_cols=COLUMNS, pred_ppm=pred, true_ppm=true, resolved_ppm=resolved,
        mask=mask, effective_mask=effective, ambiguity=ambiguity, residue_code=residues,
        row_key=np.array([f"mz{i}|lr_{i}" for i in range(n_rows)]),
        label_set=np.array([f"ls{i % 15}" for i in range(n_rows)]),
        bmrb=np.array([f"{20000 + i % 15}" for i in range(n_rows)]),
        source=np.zeros(n_rows, dtype=np.int64))


def _write_fold6_dumps(directory: Path) -> Path:
    """Five single-member fold-6 dumps."""
    directory.mkdir(parents=True, exist_ok=True)
    head = _member(seed=7, offset=0.0)
    for member in calibration.MEMBER_FOLDS:
        shifted = evaluate.replace(head, pred_ppm=head.pred_ppm + 0.03 * member)
        evaluate.write_predictions(
            shifted, directory / f"fold{calibration.EVALUATION_FOLD}_member{member}.npz")
    return directory


def _fold0_artifact(path: Path) -> calibration.IntervalCalibration:
    """A calibration artifact fitted on a fold-0 population unlike the fold-6 one, so a
    figure that refitted on fold 6 could not reproduce its constants."""
    rng = np.random.default_rng(3)
    errors, spreads = {}, {}
    for nucleus in calibration.BACKBONE_NUCLEI:
        spread = np.abs(rng.normal(0.0, 0.5, 400)) + 0.05
        errors[nucleus] = rng.normal(0.0, 1.0, 400) * np.sqrt(0.09 + 4.0 * spread ** 2)
        spreads[nucleus] = spread
    fitted = calibration.fit_interval_calibration(errors, spreads, n_bins=4)
    calibration.save_calibration(fitted, path)
    return fitted


def _run_figure(monkeypatch, tmp_path, dumps, artifact_path):
    fig_uncertainty = importlib.import_module("fig_uncertainty_calibration")
    captured = {}

    def capture(scores, **kwargs):
        captured["scores"] = scores
        return matplotlib.figure.Figure()

    monkeypatch.setattr(fig_uncertainty, "draw", capture)
    monkeypatch.setattr(fig_uncertainty.frozen, "emit", lambda fig, stem: None)
    monkeypatch.setattr(fig_uncertainty, "N_SPREAD_BINS", 4)
    monkeypatch.setattr(sys, "argv", [
        "fig_uncertainty_calibration.py", "--dumps", str(dumps),
        "--calibration", str(artifact_path), "--boot", "8"])
    fig_uncertainty.main()
    return captured


def test_the_uncertainty_figure_reports_the_artifacts_mapping_not_a_fold_six_fit(
        figures_on_path, monkeypatch, tmp_path):
    """Every backbone nucleus' mapping is the fold-0 artifact's own floor and gain, and
    each bin carries the ppm that mapping predicts for it."""
    artifact_path = tmp_path / "interval_calibration.json"
    fitted = _fold0_artifact(artifact_path)
    dumps = _write_fold6_dumps(tmp_path)

    captured = _run_figure(monkeypatch, tmp_path, dumps, artifact_path)

    for column, entry in captured["scores"].items():
        constants = fitted.constants[column.replace("_shift", "").upper()]
        assert entry["fit"] == {"floor_ppm": constants["floor_ppm"],
                                "gain": constants["gain"]}
        for row in entry["bins"]:
            expected = calibration.error_scale(
                [row["sd_mean"]], floor_ppm=constants["floor_ppm"],
                gain=constants["gain"])[0]
            assert row["mapped_rmse"] == pytest.approx(float(expected))
