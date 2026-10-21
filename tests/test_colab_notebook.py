"""The Colab notebook's code cells, run in order, against the command-line predictor."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from test_training import build_cache, run_train

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "colab" / "DeepShift_predict.ipynb"
STRUCTURE = REPO / "figures" / "reference" / "1POH.pdb"
CHAIN = "A"
# The batch size the notebook uses without a GPU; the command line is told the same one, so
# the two runs block their matrix multiplications identically.
BATCH_SIZE = 4


def code_cells() -> str:
    """The notebook's code cells concatenated in order, the way a run executes them."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return "\n\n".join("".join(cell["source"]) for cell in cells
                       if cell["cell_type"] == "code")


def shift_columns(frame) -> list[str]:
    return [column for column in frame.columns if column.endswith("_shift")]


def as_long_form(frame) -> pd.DataFrame:
    """A predictor's wide frame as (residue, nucleus, ppm) rows, blank cells dropped."""
    table = frame.melt(id_vars=["residue_id", "residue_name"],
                       value_vars=shift_columns(frame),
                       var_name="nucleus", value_name="pred_ppm").dropna(subset=["pred_ppm"])
    table["nucleus"] = table["nucleus"].str[:-len("_shift")].str.upper()
    return table


def environment(**named) -> dict:
    return {**os.environ, "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(REPO / "src"), **named}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """A checkpoint of this package's own making: one short structure-only run over the
    synthetic five-fold cache, on the CPU. Its weights are untrained, and every stamp the
    loader and the predictor read — nuclei, atom vocabulary, atom-cloud recipe,
    normalization — is the one the trainer writes."""
    cache = build_cache(tmp_path_factory.mktemp("cache"))
    run = run_train(cache, tmp_path_factory.mktemp("train") / "run.json")
    return Path(run["ckpt"])


@pytest.fixture(scope="module")
def notebook_csv(tmp_path_factory, checkpoint) -> pd.DataFrame:
    """Long-form CSV from a headless run of the cells: one chain, one model, CPU."""
    work = tmp_path_factory.mktemp("colab")
    checkpoints = work / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / checkpoint.name).symlink_to(checkpoint)
    subprocess.run([sys.executable, "-c", code_cells()], check=True, cwd=REPO,
                   env=environment(DEEPSHIFT_CKPT_DIR=str(checkpoints),
                                   DEEPSHIFT_CKPT_URL="",
                                   DEEPSHIFT_PDB=str(STRUCTURE),
                                   DEEPSHIFT_CHAIN=CHAIN,
                                   DEEPSHIFT_OUT_DIR=str(work)))
    return pd.read_csv(work / f"{STRUCTURE.stem}_{CHAIN}_deepshift.csv")


@pytest.fixture(scope="module")
def command_line_csv(tmp_path_factory, checkpoint) -> pd.DataFrame:
    """The same structure, chain and checkpoint through ``deepshift predict``."""
    out = tmp_path_factory.mktemp("command_line") / "shifts.csv"
    subprocess.run([sys.executable, "-m", "deepshift.pipeline", "predict",
                    "--pdb", str(STRUCTURE), "--chain", CHAIN,
                    "--checkpoint", str(checkpoint), "--cpu",
                    "--batch-size", str(BATCH_SIZE), "--out", str(out)],
                   check=True, cwd=REPO, env=environment())
    return pd.read_csv(out)


def test_the_notebook_predicts_the_same_ppm_as_the_command_line(
        notebook_csv, command_line_csv):
    merged = notebook_csv.merge(as_long_form(command_line_csv),
                                on=["residue_id", "residue_name", "nucleus"],
                                how="outer", suffixes=("_notebook", "_command_line"),
                                indicator=True)
    assert (merged["_merge"] == "both").all(), "the notebook and the predictor differ in cells"
    # The notebook's ppm went through decimal text on the way to the CSV, so the two
    # agree to every digit that round trip keeps.
    apart = (merged["pred_ppm_notebook"] - merged["pred_ppm_command_line"]).abs()
    assert apart.max() < 1e-9, f"largest disagreement {apart.max()} ppm"


def test_a_single_model_run_reports_no_ensemble_spread(notebook_csv, command_line_csv):
    assert list(notebook_csv.columns) == ["residue_id", "residue_name", "nucleus", "pred_ppm"]
    assert len(notebook_csv) == int(command_line_csv[shift_columns(command_line_csv)]
                                    .notna().to_numpy().sum())


def five_member_calibration(path: Path) -> Path:
    """An interval artifact of the shape the release stages: five members, fold 0."""
    import numpy as np

    from deepshift import calibration

    spread = np.linspace(0.05, 1.0, 200)
    sign = np.where(np.arange(spread.size) % 2, -1.0, 1.0)
    errors, spreads = {}, {}
    for index, nucleus in enumerate(calibration.BACKBONE_NUCLEI, start=1):
        spreads[nucleus] = spread * (1.0 + index / 20.0)
        errors[nucleus] = sign * np.sqrt((0.15 * index) ** 2 + 1.7 * spreads[nucleus] ** 2)
    calibration.save_calibration(
        calibration.fit_interval_calibration(errors, spreads, n_bins=4),
        path)
    return path


def test_the_notebook_hands_the_interval_calibration_to_the_predictor(
        tmp_path, checkpoint):
    """The artifact beside the checkpoints reaches ``predict_pdb``: given one member and a
    five-member artifact, the run stops on the package's own refusal instead of printing a
    prediction with no interval."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / checkpoint.name).symlink_to(checkpoint)
    five_member_calibration(checkpoints / "interval_calibration.json")

    finished = subprocess.run([sys.executable, "-c", code_cells()], cwd=REPO,
                              capture_output=True, text=True,
                              env=environment(DEEPSHIFT_CKPT_DIR=str(checkpoints),
                                              DEEPSHIFT_CKPT_URL="",
                                              DEEPSHIFT_PDB=str(STRUCTURE),
                                              DEEPSHIFT_CHAIN=CHAIN,
                                              DEEPSHIFT_OUT_DIR=str(tmp_path)))
    assert finished.returncode != 0
    assert "interval calibration requires 5 members" in finished.stderr


def test_the_notebook_says_so_when_no_interval_calibration_is_beside_the_checkpoints(
        notebook_csv):
    """Without the artifact the run finishes and the CSV carries no interval column."""
    assert "ci90_halfwidth_ppm" not in notebook_csv.columns
    assert "ensemble_sd_ppm" not in notebook_csv.columns
