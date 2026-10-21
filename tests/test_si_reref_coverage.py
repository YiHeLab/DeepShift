"""The rereferencing coverage table says why each label set does or does not carry an offset.

A label set can be realized into the build and still be left unchanged: no backbone heavy-
atom shift to correct, an offset that rounds to zero, or too few residues (or no usable
alignment) to fit one. Those counts come from scripts/reref_diagnostics.py, which
reproduces the stored offsets, and they must account for every label set under both the
file-wide and the realized line.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "si_build"))

OFFSETS = {
    "ls_1": {"method": "computed", "struct_type": "af",
             "offsets": {"CA": -0.4, "CB": -0.4, "CO": 0.1, "N": -0.2}},
    "ls_2": {"method": "computed", "struct_type": "nmr", "offsets": {}},
    "ls_3": {"method": "none", "struct_type": None, "offsets": {}},
    # selected and never realized: it is not part of the realized denominator
    "ls_4": {"method": "none", "struct_type": None, "offsets": {}},
}
REALIZED = {"ls_1": 1, "ls_2": 2, "ls_3": 6}
DIAGNOSTICS = {
    "reproduces_stored_offsets": 4,
    "all": {"outcome": {"offset": 1, "rounds_to_zero": 1, "too_few_residues": 1,
                        "no_structure": 1}},
    "realized": {"outcome": {"offset": 1, "rounds_to_zero": 1, "too_few_residues": 1}},
}


@pytest.fixture
def producer(tmp_path, monkeypatch):
    monkeypatch.setenv("SI_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("DEEPSHIFT_RUN", str(tmp_path / "run"))
    monkeypatch.delenv("SI_REREF_DIAGNOSTICS", raising=False)   # a sourced release env must not leak in
    data = tmp_path / "run" / "data"
    data.mkdir(parents=True)
    (data / "reref_offsets.json").write_text(json.dumps(OFFSETS))
    (data / "fold_assignments_labelset_identity90.json").write_text(json.dumps(REALIZED))
    (data / "reref_diagnostics.json").write_text(json.dumps(DIAGNOSTICS))
    for module in ("si_common", "si_reref_offsets"):
        sys.modules.pop(module, None)
    import si_reref_offsets as module
    return module


def rows(tex):
    return [line for line in tex.splitlines() if line.rstrip().endswith(r"\\")]


def test_the_realized_outcomes_sit_under_the_realized_line(producer):
    tex = producer.coverage_table(producer.load())
    printed = rows(tex)
    realized = next(i for i, line in enumerate(printed)
                    if "Label sets realized into the build" in line)
    under = [line.split("&")[:2] for line in printed[realized + 1:realized + 4]]
    assert [(label.strip(), n.strip()) for label, n in under] == [
        (r"\quad with a non-zero offset", "1"), (r"\quad offset rounds to zero", "1"),
        (r"\quad too few residues, or no usable alignment", "1")]


def test_outcomes_that_do_not_account_for_every_label_set_are_refused(producer):
    d = producer.load()
    d["diagnostics"]["realized"]["outcome"]["offset"] = 2
    with pytest.raises(SystemExit, match="do not account"):
        producer.coverage_table(d)


def test_the_alphafold_row_cites_the_predictor_it_names(producer):
    tex = producer.coverage_table(producer.load())
    assert r"AlphaFold\cite{AlphaFold}" in tex


def test_si_reref_diagnostics_names_the_file_read(tmp_path, monkeypatch, producer):
    other = tmp_path / "reref_diagnostics_other.json"
    changed = {**DIAGNOSTICS, "realized": {"outcome": {"offset": 1, "rounds_to_zero": 1,
                                                       "no_correctable_shift": 1}}}
    other.write_text(json.dumps(changed))
    monkeypatch.setenv("SI_REREF_DIAGNOSTICS", str(other))
    for module in ("si_common", "si_reref_offsets"):
        sys.modules.pop(module, None)
    import si_reref_offsets as module
    assert other.resolve() == module.DIAGNOSTICS
    tex = module.coverage_table(module.load())
    assert r"no C$\alpha$, C$\beta$, C$'$ or N shift to correct" in tex
