"""Table S11: what it counts and what it prints.

The ledger is built from this run alone: the fold-6 task table, the per-task outputs the
predictors wrote under the benchmark root, the coverage census measured on them, and the
backbone scoring report on the master cells.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SI_BUILD = Path(__file__).resolve().parents[1] / "si_build"
NUCLEI = ("HA", "H", "CA", "CB", "C", "N")
COLUMNS = [f"{n.lower()}_shift" for n in NUCLEI]
ARMS = ("ucbshift_full_sameprot", "sparta", "camshift", "nmrgnn")
SHORT_ARM = "camshift"            # below the floor on one nucleus
TASKS = 3
PER_NUCLEUS = 100
UNIVERSE = PER_NUCLEUS * len(NUCLEI)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SI_BUILD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _answer(*residues) -> str:
    """One task's output file: a status and the nuclei it labels, per residue."""
    return json.dumps({"status": "ok", "predictions": {
        str(index): dict(residue) for index, residue in enumerate(residues)}})


def _arm(directory: Path, answers):
    directory.mkdir(parents=True)
    for index, answer in enumerate(answers):
        (directory / f"task{index}.json").write_text(answer)


def _coverage(arm) -> dict:
    """Cells covered and whether the arm joins, per nucleus column: every arm covers
    every nucleus in full, bar the short arm's beta carbon."""
    short = arm == SHORT_ARM
    return {column: (PER_NUCLEUS // 2 if short and column == "cb_shift" else PER_NUCLEUS)
            for column in COLUMNS}


def _write_universe(directory: Path, sids=("t1", "t2", "t3")) -> None:
    """The census's scorable cells: UNIVERSE of them spread over ``sids``."""
    np.savez(directory / "arm_coverage.npz",
             sid=np.array([sids[i % len(sids)] for i in range(UNIVERSE)]))


def _census(arms) -> dict:
    return {
        "n_universe_cells": UNIVERSE, "n_structure_instances": TASKS,
        "coverage_npz": "arm_coverage.npz",
        "coverage_floor": 0.90,
        "universe_by_nucleus": dict.fromkeys(COLUMNS, PER_NUCLEUS),
        "arms_scanned": [*arms, "deepshift"],
        "arms": {arm: {
            "n_cells": sum(_coverage(arm).values()),
            "by_nucleus": {column: {"n_cells": cells, "n_universe": PER_NUCLEUS,
                                    "coverage": cells / PER_NUCLEUS,
                                    "joins": cells / PER_NUCLEUS >= 0.90}
                           for column, cells in _coverage(arm).items()},
        } for arm in (*arms, "deepshift")},
    }


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    """The two tables, built over a run root carrying every input they read."""
    run = tmp_path / "run"
    bench = run / "benchmark"
    (bench / "master_cells").mkdir(parents=True)
    (bench / "fold6_tasks.tsv").write_text(
        "structure_id\tsource\nt1\taf\nt2\tnmr\nt3\txtal\n")
    (bench / "master_cells" / "coverage_census.json").write_text(json.dumps(_census(ARMS)))
    _write_universe(bench / "master_cells")
    # Four fold-6 realizations (one label set realized twice), one of them with no
    # evaluable residue, so the note reconciles 4 structures with the 3 tasks.
    data = run / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "fold_assignments_labelset_identity90.json").write_text(json.dumps(
        {"ls_a": 6, "ls_b": 6, "ls_c": 6, "ls_d": 1}))
    for source, rows in {"af": ["ls_a,success", "ls_d,success"],
                         "nmr": ["ls_b,success", "ls_a,success"],
                         "xtal": ["ls_c,success", "ls_c,failed"]}.items():
        (data / f"build_log_{source}.csv").write_text(
            "label_set_id,status\n" + "\n".join(rows) + "\n")
    scored = bench / "scored" / "master"
    scored.mkdir(parents=True)
    # The master cells: the backbone report's 540 cells on two of the tasks, and side-chain
    # cells on the third, so the set spans more structures than its backbone part.
    master = bench / "master_cells" / "master_cells_pf14.npz"
    np.savez(master, sid=np.array(["t1"] * 270 + ["t2"] * 270 + ["t3"] * 10),
             atom=np.array([COLUMNS[i % len(COLUMNS)] for i in range(540)] + ["cg_shift"] * 10))
    report = scored / "score_tier2_master.json"
    report.write_text(json.dumps({
        "n_common_cells": 540, "n_common_structures": TASKS - 1,
        "master_cells": str(master),
        "methods": [*ARMS, "deepshift"],
        # what the backbone tables mark: the short arm on its beta carbon
        "arms_below_master_coverage": {SHORT_ARM: ["CB"]},
        "by_source": {"all": {"nuclei": {
            nucleus: {"deepshift": {"rmse": 0.5, "mae": 0.4},
                      "sparta": {"rmse": 0.9, "mae": 0.6},
                      "nmrgnn": {"rmse": 27.0, "mae": 1.0}}
            for nucleus in NUCLEI}}},
    }))
    # The side-chain table marks the arms it prints where this report scored them below
    # the floor; the census flags none of them, so only the report can say so.
    (scored / "score_by_nucleus_tier2.json").write_text(json.dumps({"by_nucleus": {
        "HD21": {"backbone": False, "arms": {
            "deepshift": {"below_master_coverage": False},
            "shiftx2_sameprot": {"below_master_coverage": True},
            "ucbshift2_sameprot": {"below_master_coverage": False}}},
        "CA": {"backbone": True, "arms": {
            "ucbshift2_sameprot": {"below_master_coverage": True}}}},
        "nuclei_below_the_floor": {"HH": {"deepshift": 40, "shiftx2_sameprot": 0}}}))
    backbone = {"resname": "ALA", "HA": 4.1, "H": 8.0}
    for name in ARMS:
        # SPARTA+ answers two tasks, the second labelling a nucleus the first does not
        # and one atom of a bound ligand.
        answers = ([_answer(backbone), _answer({"resname": "HEM", "CB": 30.0, "FE": 0.0})]
                   if name == "sparta" else [_answer(backbone)])
        _arm(bench / name, answers)

    monkeypatch.setenv("DEEPSHIFT_RUN", str(run))
    monkeypatch.setenv("DEEPSHIFT_ARCHIVE", str(tmp_path / "archive"))
    monkeypatch.setenv("DEEPSHIFT_BENCH_OUT", str(bench))
    monkeypatch.setenv("SI_FINAL_SCORES_DIR", str(scored))
    monkeypatch.setenv("SI_OUT", str(tmp_path / "out"))
    monkeypatch.delenv("DEEPSHIFT_COVERAGE_CENSUS", raising=False)
    monkeypatch.setattr(sys, "argv", ["si_competitor_ledger.py"])

    sys.path.insert(0, str(SI_BUILD))
    try:
        _load("si_common")
        module = _load("si_competitor_ledger")
        module.main()
    finally:
        sys.path.remove(str(SI_BUILD))
    out = tmp_path / "out"
    return {"run": run, "report": report, "bench": bench, "tmp": tmp_path,
            "tex": (out / "si_coverage_ledger.tex").read_text()}


def test_nuclei_returned_counts_the_whole_run_and_only_amino_acid_atoms(ledger):
    """A predictor labels whatever atoms a structure carries, so reading its nucleus set
    from one answer reports that structure's residues rather than the tool's vocabulary.
    The count is the distinct standard amino-acid atom names it writes anywhere in the
    run: SPARTA+ here labels HA and H on one task and CB on another, and the heme iron of
    the second is not a nucleus it predicts for a protein."""
    assert r"SPARTA+\cite{spartaplus2010} & 3 & 2 & 1 & 3 & 100.0 \\" in ledger["tex"]


def test_the_ledger_prints_the_coverage_the_master_cells_were_frozen_on(ledger):
    """The column the master rule is read off: the share of the scorable cells each
    predictor produced. The short arm covers half of one nucleus and is named as such."""
    assert r"Cells covered (\%)" in ledger["tex"]
    row = next(line for line in ledger["tex"].splitlines() if line.startswith("CAMSHIFT"))
    share = 100 * sum(_coverage(SHORT_ARM).values()) / UNIVERSE
    assert f"& {share:.1f} \\\\" in row
    assert ("In the backbone tables CAMSHIFT covers less than 90\\% of a nucleus it "
            "returns; in the side-chain table SHIFTX2 covers less than 90\\% of a nucleus "
            "it returns.") in ledger["tex"]


def test_the_scored_count_is_the_per_nucleus_reports_not_the_vocabulary(ledger):
    """DeepShift emits six nuclei here, but the per-nucleus report scores two and puts HH
    below its floor; the sentence prints the report's count and names what it dropped."""
    assert ("The accuracy tables score 2 of the 6 nuclei DeepShift emits; HH has too few "
            "cells (at most 40) to score.") in ledger["tex"]


def test_deepshift_is_covered_on_every_task_and_returns_every_scored_nucleus(ledger):
    """DeepShift's predictions come from the cache, so every task is covered by
    construction."""
    assert r"\textbf{DeepShift} & 3 & 3 & 0 & 6 & 100.0 \\" in ledger["tex"]
    assert r"\begin{tabular}{lrrrrr}" in ledger["tex"]


def test_a_predictor_whose_output_is_absent_fails_the_ledger_closed(ledger, monkeypatch):
    """A missing per-task directory would otherwise print a predictor that answered
    nothing, which reads as a result rather than as an absent run."""
    for path in (ledger["bench"] / "nmrgnn").glob("*.json"):
        path.unlink()
    (ledger["bench"] / "nmrgnn").rmdir()
    sys.path.insert(0, str(SI_BUILD))
    try:
        module = _load("si_competitor_ledger")
        with pytest.raises(FileNotFoundError, match="nmrgnn"):
            module.main()
    finally:
        sys.path.remove(str(SI_BUILD))


def test_the_note_reconciles_fold6_structures_with_tasks(ledger):
    """Table S2's realizations, the tasks that keep a residue, and the master set's span."""
    text = ledger["tex"]
    assert "Of the 4 fold-6 structure realizations" in text
    assert "1 keep no residue after exclusions" in text
    assert "spans 3 of the remaining 3" in text
    assert "600 residue--nucleus cells of 3 structure instances are scorable" in text
    assert "over 2 structure instances" in text


def test_scorable_structures_are_those_holding_a_scorable_cell(ledger, tmp_path):
    """The note counts the structures the census's cells lie on, not the tasks: a task
    that keeps residues but no scorable cell is not among them."""
    sys.path.insert(0, str(SI_BUILD))
    try:
        _load("si_common")
        module = _load("si_competitor_ledger")
    finally:
        sys.path.remove(str(SI_BUILD))
    _write_universe(tmp_path, sids=("t1", "t2"))
    census = _census(ARMS)
    _path, n = module.scorable_structures(tmp_path / "coverage_census.json", census)
    assert (n, census["n_structure_instances"]) == (2, 3)


def test_a_master_set_that_is_not_the_reports_cohort_is_refused(ledger):
    """The note counts the master set's structures from the master cells the backbone
    report names; cells whose backbone part is not that report's cohort are refused."""
    report = json.loads(ledger["report"].read_text())
    report["n_common_cells"] += 1
    ledger["report"].write_text(json.dumps(report))
    sys.path.insert(0, str(SI_BUILD))
    try:
        _load("si_common")
        module = _load("si_competitor_ledger")
        with pytest.raises(SystemExit, match="REFUSING"):
            module.main()
    finally:
        sys.path.remove(str(SI_BUILD))


def test_a_rerun_census_is_named_by_the_environment(ledger, monkeypatch):
    """A census re-run beside the frozen one, not over it, is read through
    DEEPSHIFT_COVERAGE_CENSUS."""
    census = _census(ARMS)
    census["arms"]["sparta"]["n_cells"] = UNIVERSE // 2
    rerun = ledger["tmp"] / "rerun" / "coverage_census.json"
    rerun.parent.mkdir()
    rerun.write_text(json.dumps(census))
    _write_universe(rerun.parent)
    monkeypatch.setenv("DEEPSHIFT_COVERAGE_CENSUS", str(rerun))
    sys.path.insert(0, str(SI_BUILD))
    try:
        _load("si_competitor_ledger").main()
    finally:
        sys.path.remove(str(SI_BUILD))
    out = ledger["tmp"] / "out"
    assert (r"SPARTA+\cite{spartaplus2010} & 3 & 2 & 1 & 3 & 50.0 \\"
            in (out / "si_coverage_ledger.tex").read_text())
