"""Both X-EISD figures read one protocol run from one root."""
import importlib.util
import sys
from pathlib import Path

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, FIGURES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_two_xeisd_figures_resolve_the_same_protocol_run(monkeypatch, tmp_path):
    """One root names the protocol run, so the selection figure and the composition
    figure read the same run directory and neither reaches a level below it."""
    protocol = tmp_path / "protocol"
    (protocol / "runs").mkdir(parents=True)
    monkeypatch.setenv("DEEPSHIFT_XEISD_PROTOCOL", str(protocol))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    sys.path.insert(0, str(FIGURES))
    try:
        selection = _load("fig_xeisd_mixed_selection")
        composition = _load("fig_xeisd_composition")
    finally:
        sys.path.remove(str(FIGURES))

    assert protocol / "runs" == selection.RUNS
    assert composition.RUNS == selection.RUNS
