"""The perturber analysis joins its prediction dumps to the run's own cache."""
import importlib.util
import sys
from pathlib import Path

SI_BUILD = Path(__file__).resolve().parents[1] / "si_build"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SI_BUILD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_perturber_analysis_reads_the_cache_of_the_run_it_scores(monkeypatch, tmp_path):
    """Its prediction dumps are this run's five members on the shadow folds, which address
    rows by the run's own cache globals, so the join is to the run's cache and never to a
    previous model's copy under the archive."""
    run, archive = tmp_path / "run", tmp_path / "archive"
    monkeypatch.setenv("DEEPSHIFT_RUN", str(run))
    monkeypatch.setenv("DEEPSHIFT_ARCHIVE", str(archive))
    run.mkdir()
    archive.mkdir()
    sys.path.insert(0, str(SI_BUILD))
    try:
        _load("si_common")
        analyze = _load("analyze_perturber")
    finally:
        sys.path.remove(str(SI_BUILD))

    assert analyze.CACHE.relative_to(run) == Path("work/unified_cache")
    assert not analyze.CACHE.is_relative_to(archive)
