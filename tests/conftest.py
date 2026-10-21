"""Fixtures shared by more than one test module."""
import pytest
from benchmark_run import bind, build_run, side_chain_rows


def _bound(tmp_path, monkeypatch, **rows):
    built = bind(monkeypatch, build_run(tmp_path, **rows))
    yield built
    built["cells"].load_pred_npz.cache_clear()
    built["cells"].build_row_index.cache_clear()


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    """The benchmark modules, bound to a freshly built run root."""
    yield from _bound(tmp_path, monkeypatch)


@pytest.fixture
def sidechain_benchmark(tmp_path, monkeypatch):
    """The benchmark modules, bound to a run root that carries side-chain nuclei."""
    yield from _bound(tmp_path, monkeypatch, rows=side_chain_rows())
