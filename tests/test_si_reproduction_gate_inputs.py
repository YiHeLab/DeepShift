"""Table S18 reports the run whose inputs are complete, without the cells UCBShift's paper
excludes from its test set."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SI_BUILD = Path(__file__).resolve().parents[1] / "si_build"
ATOMS = ("N", "CA", "CB", "C", "H", "HA")
#: What the rerun on rebuilt structures measured, per nucleus: cells, MAE of the combined
#: output, RMSE of the structure module. The structure module's RMSE is the value its own
#: Table 3 publishes, so those rows print a zero difference.
RERUN = {"N": (100, 1.0923, 2.49), "CA": (90, 0.4332, 1.08), "CB": (80, 0.4870, 1.28),
         "C": (100, 0.6176, 1.5191), "H": (70, 0.1859, 0.44), "HA": (60, 0.1084, 0.25)}
PUBLISHED_X = {"N": 2.49, "CA": 1.08, "CB": 1.28, "C": 1.17, "H": 0.44, "HA": 0.25}
#: The same six nuclei as scored from a prediction cache made without hydrogens: the
#: numbers the frozen gate report carries, which this table does not print.
UNHYDROGENATED = {"N": 1.1792, "CA": 0.5237, "CB": 0.5541, "C": 0.7059,
                  "H": 0.2774, "HA": 0.1591}
#: Cells the paper's exclusions remove from the rerun, by chain and nucleus.
REMOVED = {"1RPJA:C": 15, "1BGF_:C": 5, "1VKBA:CB": 2}
SHIFTX2_RMSE = {"N": 1.5855, "CA": 0.5869, "CB": 0.7802,
                "C": 0.6682, "H": 0.3381, "HA": 0.1568}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SI_BUILD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(root: Path) -> Path:
    """The benchmark's frozen gate report: the SHIFTX2 block, the transfer-module
    agreement, and the committed-cache scoring the table leaves off the page."""
    path = root / "reproduction" / "reproduction_report.json"
    path.parent.mkdir(parents=True)
    cells = {atom: RERUN[atom][0] for atom in ATOMS}
    path.write_text(json.dumps({
        "ucb_committed_vs_target": {
            "UCBShift": {a: {"n": cells[a], "mae": UNHYDROGENATED[a]} for a in ATOMS},
            "X": {a: {"n": cells[a], "mae": UNHYDROGENATED[a]} for a in ATOMS}},
        "ucb_rerun_agreement": {"1abc": {"_Y": {"n": 40, "max_abs_diff": 0.0}}},
        "shiftx2_leaky_vs_S9": {
            "aggregate": {a: {"n": 20, "rmse": SHIFTX2_RMSE[a]} for a in ATOMS},
            "coverage": {"total": 61, "scored": 45},
            "per_protein": {"1def": {"n_cells": dict.fromkeys(ATOMS, 20),
                                     "rmse": dict(SHIFTX2_RMSE)}}},
        "gate": {"passed": True, "failures": [], "generated_utc": "2026-01-01T00:00:00Z"},
    }))
    return path


def _rerun(root: Path) -> Path:
    """The released code rerun on the released test structures with hydrogens rebuilt."""
    (root / "probe").mkdir(parents=True)
    (root / "score_ucb200.json").write_text(json.dumps({
        "published": {"UCBShift-X": PUBLISHED_X},
        "funnel": {"chains_with_new_preds": 3},
        "per_arm_own_cells": {
            "UCBShift_H": {a: {"n": RERUN[a][0], "mae": RERUN[a][1]} for a in ATOMS},
            "X_H": {a: {"n": RERUN[a][0], "rmse": RERUN[a][2]} for a in ATOMS}},
    }))
    (root / "probe" / "paper_exclusions_result.json").write_text(json.dumps({
        "arms": {"UCBShift_H": {a: {"n": _kept(a), "mae": RERUN[a][1]} for a in ATOMS},
                 "X_H": {a: {"n": _kept(a), "rmse": RERUN[a][2]} for a in ATOMS}},
        "removed": {"UCBShift_H": REMOVED, "X_H": REMOVED},
    }))
    (root / "probe" / "paper_exclusions.py").write_text("# the script that wrote it\n")
    return root


def _kept(atom):
    return RERUN[atom][0] - sum(n for key, n in REMOVED.items() if key.endswith(f":{atom}"))


def _pool61(root: Path) -> Path:
    """The 61-set pool Table S20 reads: SHIFTX2 as distributed on its own cells."""
    root.mkdir(parents=True)
    path = root / "pool_shiftx2_61_chains.json"
    path.write_text(json.dumps({
        "n_scorable_proteins": 60, "protein_exclusions": {"1kdb": "no refDB target"},
        "own_cells": {"shiftx2_asdistributed": {a: {"n": 30, "rmse": SHIFTX2_RMSE[a]}
                                                for a in ATOMS}}}))
    return path


def _released_test_set(root: Path) -> Path:
    (root / "test_targets").mkdir(parents=True)
    for name in ("1abc", "1def", "1ghi", "1jkl"):
        (root / "test_targets" / f"{name}.csv").write_text("RES_NUM,RESNAME\n")
    return root


@pytest.fixture
def gate(monkeypatch, tmp_path):
    competitor = tmp_path / "competitor"
    _report(competitor)
    _rerun(tmp_path / "rerun")
    _released_test_set(tmp_path / "cspred")
    pool = _pool61(tmp_path / "pool61")

    monkeypatch.setenv("DEEPSHIFT_RUN", str(tmp_path / "run"))
    monkeypatch.setenv("DEEPSHIFT_ARCHIVE", str(tmp_path / "archive"))
    monkeypatch.setenv("SI_COMPETITOR_ROOT", str(competitor))
    monkeypatch.setenv("SI_GATE_UCB200_RERUN", str(tmp_path / "rerun"))
    monkeypatch.setenv("SI_GATE_CSPRED_TM", str(tmp_path / "cspred"))
    monkeypatch.setenv("SI_UCB200_POOL61", str(pool))
    monkeypatch.setenv("SI_OUT", str(tmp_path / "out"))

    sys.path.insert(0, str(SI_BUILD))
    try:
        _load("si_common")
        _load("build_reproduction_gate").main()
    finally:
        sys.path.remove(str(SI_BUILD))
    out = tmp_path / "out"
    return {"tex": (out / "si_reproduction_gate.tex").read_text()}


def test_the_table_reports_the_run_on_structures_with_hydrogens(gate):
    """A prediction cache built from structures whose hydrogens are absent understates
    every proton nucleus, so the column is the rerun on rebuilt structures. The frozen
    gate report carries the cache's scoring, and none of it reaches the page."""
    for atom in ("N", "CA", "CB", "H", "HA"):
        assert f"& {RERUN[atom][1]:.4f} &" in gate["tex"], atom
        assert f"{UNHYDROGENATED[atom]:.4f}" not in gate["tex"], atom


def test_the_ucbshift_blocks_drop_the_cells_the_paper_excludes(gate):
    """Both UCBShift blocks are scored without the cells UCBShift's paper removes from its
    test set, as Tables S19 and S20 are: each row carries the cells left, the blocks are
    marked."""
    assert gate["tex"].count(rf"C$'$ & {_kept('C'):,} &") == 2
    assert gate["tex"].count(rf"C$\beta$ & {_kept('CB'):,} &") == 2
    assert r"\textbf{UCBShift$^{\dagger}$}" in gate["tex"]


def test_the_structure_module_is_reported_against_its_published_statistic(gate):
    """Their paper publishes the structure module as a root mean square error, so the
    block reports one and the Published column carries the value rather than a gap."""
    assert r"\textbf{UCBShift-X$^{\dagger}$} --- RMSE, ppm" in gate["tex"]
    assert "not published" not in gate["tex"]
    assert r"N & 100 & 2.49 & 2.4900 & $+$0.0000 \\" in gate["tex"]


def test_the_table_prints_five_columns_and_no_verdict(gate):
    """The admission check is decided against this project's own rescoring rather than
    against the papers, so no verdict of it belongs beside a published value."""
    assert r"\begin{tabular}{lrrrr}" in gate["tex"]
    assert r"Nucleus & Cells & Published & This work & Difference \\" in gate["tex"]
    for absent in ("Band", "not gated", "pass"):
        assert absent not in gate["tex"], absent


def test_shiftx2_is_read_from_the_pool_on_its_own_test_set(gate):
    """SHIFTX2's block is the alignment-matched pool Table S20 reads, with its cells and
    the proteins it covers on the page."""
    assert rf"N & 30 & 1.1169 & {SHIFTX2_RMSE['N']:.4f}" in gate["tex"]
    assert "on 60 of its 61 test proteins" in gate["tex"]


def test_a_cohort_larger_than_the_released_test_set_is_refused(monkeypatch, tmp_path):
    """Every protein scored is one of the authors' released test proteins."""
    competitor = tmp_path / "competitor"
    _report(competitor)
    _rerun(tmp_path / "rerun")
    scored = json.loads((tmp_path / "rerun" / "score_ucb200.json").read_text())
    scored["funnel"]["chains_with_new_preds"] = 5
    (tmp_path / "rerun" / "score_ucb200.json").write_text(json.dumps(scored))
    _released_test_set(tmp_path / "cspred")
    pool = _pool61(tmp_path / "pool61")
    monkeypatch.setenv("DEEPSHIFT_RUN", str(tmp_path / "run"))
    monkeypatch.setenv("DEEPSHIFT_ARCHIVE", str(tmp_path / "archive"))
    monkeypatch.setenv("SI_COMPETITOR_ROOT", str(competitor))
    monkeypatch.setenv("SI_GATE_UCB200_RERUN", str(tmp_path / "rerun"))
    monkeypatch.setenv("SI_GATE_CSPRED_TM", str(tmp_path / "cspred"))
    monkeypatch.setenv("SI_UCB200_POOL61", str(pool))
    monkeypatch.setenv("SI_OUT", str(tmp_path / "out"))
    sys.path.insert(0, str(SI_BUILD))
    try:
        _load("si_common")
        with pytest.raises(ValueError, match="the_cohort_is_within_the_released_test_set"):
            _load("build_reproduction_gate").main()
    finally:
        sys.path.remove(str(SI_BUILD))
