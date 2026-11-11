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


def test_the_xeisd_panels_resolve_the_protocol_run(monkeypatch, tmp_path):
    """One root names the protocol run, so the applications figure (Figure 5b,c) reads the
    run directory and reaches no level below it."""
    protocol = tmp_path / "protocol"
    (protocol / "runs").mkdir(parents=True)
    monkeypatch.setenv("DEEPSHIFT_XEISD_PROTOCOL", str(protocol))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    sys.path.insert(0, str(FIGURES))
    try:
        applications = _load("fig_applications")
    finally:
        sys.path.remove(str(FIGURES))

    assert protocol / "runs" == applications.RUNS
