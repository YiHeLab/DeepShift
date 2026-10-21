"""The supplementary tables read the p100 authority of the build they describe."""
import importlib.util
import sys
from pathlib import Path

import pytest

SI_BUILD = Path(__file__).resolve().parents[1] / "si_build"


def _load_si_common():
    spec = importlib.util.spec_from_file_location("si_common", SI_BUILD / "si_common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_si_common_has_no_default_p100_authority(monkeypatch, tmp_path):
    monkeypatch.delenv("SI_P100_CUTOFFS", raising=False)
    si_common = _load_si_common()
    assert not hasattr(si_common, "P100_CUTOFFS")
    with pytest.raises(EnvironmentError, match="SI_P100_CUTOFFS"):
        si_common.p100_cutoffs_path()
    with pytest.raises(EnvironmentError, match="SI_P100_CUTOFFS"):
        si_common.p100_policy()

    cutoffs = tmp_path / "p100_cutoffs.json"
    cutoffs.write_text('{"ALA:ca_shift": {"lo": 48.0, "hi": 58.0}}')
    monkeypatch.setenv("SI_P100_CUTOFFS", str(cutoffs))
    assert si_common.p100_cutoffs_path() == cutoffs.resolve()
    assert si_common.p100_policy().accepted_range("ALA", "ca_shift") == (48.0, 58.0)
