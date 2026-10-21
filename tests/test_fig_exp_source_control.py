"""The structure-source control figure scores its two arms on the fenced cells."""
import importlib
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"
SOURCES = ("af", "xtal", "nmr")
COLUMNS = ("n_shift", "h_shift", "ca_shift", "cb_shift", "c_shift", "ha_shift")


@pytest.fixture(scope="module")
def figures_on_path():
    if str(FIGURES) not in sys.path:
        sys.path.insert(0, str(FIGURES))
    return FIGURES


def _by_source(offset: float) -> dict:
    """One arm's scores: every backbone nucleus, every source, marginal and matched."""
    table = {source: {column: {"rmse": 1.0 + offset + index / 10,
                               "rmse_ci": [0.9 + offset, 1.1 + offset],
                               "n": 1000 + index}
                      for index, column in enumerate(COLUMNS)}
             for source in SOURCES}
    return {"marginal": table, "matched": table}


def _scored(monkeypatch, figure):
    """Replace the scoring of the two dumps with fixed reports; the dumps themselves only
    have to exist."""
    compare = {column: {"delta_rmse": 0.05, "ci": [0.01, 0.09], "n": 900}
               for column in COLUMNS}
    monkeypatch.setattr(figure, "score_arms", lambda paths: (
        {"shipped": _by_source(0.0), "exp": _by_source(0.2)}, compare,
        {"shipped": 2396, "exp": 2396}))


def _dumps(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("shipped_ensemble_fold6.preds.npz", "exp_ensemble_fold6.preds.npz"):
        (directory / name).write_bytes(b"dump")
    return directory


def _run_figure(monkeypatch, eval_dir: Path) -> dict:
    figure = importlib.import_module("fig_exp_source_control")
    captured = {}

    def capture(fig, stem):
        captured["stem"] = stem

    _scored(monkeypatch, figure)
    monkeypatch.setattr(figure.frozen, "emit", capture)
    monkeypatch.setattr(sys, "argv", ["fig_exp_source_control.py",
                                      "--eval-dir", str(eval_dir)])
    figure.main()
    return captured


def test_the_source_control_figure_is_drawn_from_the_scored_arms(
        figures_on_path, monkeypatch, tmp_path, capsys):
    """The figure scores the two ensemble dumps itself, after the anticorrelated channels
    are dropped, and prints the cells the fence removed."""
    captured = _run_figure(monkeypatch, _dumps(tmp_path / "fold6_eval"))

    assert captured["stem"] == "fig_exp_source_control"
    assert "{'shipped': 2396, 'exp': 2396}" in capsys.readouterr().out


def test_both_arms_are_fenced_before_they_are_scored(figures_on_path, monkeypatch):
    """Every report the panels draw is computed on the fenced arms."""
    figure = importlib.import_module("fig_exp_source_control")
    seen = []
    monkeypatch.setattr(figure.evaluate, "read_predictions", lambda path: ("raw", path.name))
    monkeypatch.setattr(figure.frozen, "drop_anticorrelated",
                        lambda preds: (("fenced", preds[1]), 7))

    class Aligned:
        def __init__(self, preds):
            self.preds = preds
            self.shift_cols = list(COLUMNS)

    monkeypatch.setattr(figure.evaluate, "align",
                        lambda a, b: (Aligned(a), Aligned(b)))
    monkeypatch.setattr(figure.evaluate, "by_source",
                        lambda preds, **kw: seen.append(preds.preds) or {})
    monkeypatch.setattr(figure.evaluate, "compare",
                        lambda a, b, **kw: seen.append((a.preds, b.preds)) or {})
    _reports, _compare, dropped = figure.score_arms(
        {"shipped": Path("s.npz"), "exp": Path("e.npz")})
    assert dropped == {"shipped": 7, "exp": 7}
    assert all(item[0] == "fenced" if isinstance(item[0], str) else
               all(part[0] == "fenced" for part in item) for item in seen), seen
    assert len(seen) == 5


def test_the_matched_control_fact_is_counted_from_the_matched_rmse_table(figures_on_path):
    """The matched-cell sentence states what the values block computes: the count of
    nuclei the AlphaFold column wins, and the two numbers behind every nucleus it loses."""
    figure = importlib.import_module("fig_exp_source_control")
    nuclei = list(COLUMNS)
    matched = {source: dict.fromkeys(nuclei, 1.0) for source in SOURCES}
    for source in SOURCES:
        matched[source]["ca_shift"] = 0.9003 if source == "af" else 2.0
    matched["xtal"]["ca_shift"] = 0.8943

    fact = figure.matched_control_fact(matched, nuclei)
    assert "best on 5 of 6 backbone nuclei" in fact, fact
    assert "CA goes to xtal (0.8943 against AF 0.9003)" in fact, fact

    matched["xtal"]["ca_shift"] = 2.0
    assert figure.matched_control_fact(matched, nuclei).endswith(
        "best on 6 of 6 backbone nuclei")
