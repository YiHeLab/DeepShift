"""The master cell sets: who joins, what joining costs, and what a partial arm costs.

The fixture's run root carries four competitor arms of unequal coverage.
Every arm returns the alpha carbon. On the beta carbon one arm clears the floor with a
residue of every structure missing and one falls far below it. On the gamma carbon the
two retrieval arms return different slices and neither reaches the floor, and the other
two return nothing. On the delta carbon the wide arm clears the floor with a residue of
every structure missing and the narrow arm returns a handful. None of them returns the
arginine N-epsilon. That is one nucleus for each branch of the rule.
"""
import json
import sys

import numpy as np
import pytest
from benchmark_run import (
    SIDE_CHAIN_ALIASED,
    SIDE_CHAIN_ARGININES,
    SIDE_CHAIN_RESIDUES,
    SIDE_CHAIN_STRUCTURES,
    observed_ppm,
    write_competitor,
)

from benchmarks import (
    coverage_census,
    master_cells,
    score_by_nucleus,
    score_by_source_tier,
    score_paired_tier,
)
from benchmarks.config import BENCHMARK_ARMS
from benchmarks.master_cells import MASTER_COLUMNS, MASTER_COVERAGE_FLOOR
from deepshift.self_exclusion import DECLARED_MODES, ExclusionLevel

TIER = 2
TIER_MODE = next(iter(DECLARED_MODES[ExclusionLevel.SAME_PROTEIN]))
CLEAN_HITLOG = "9999.9zzzA\t9999\tx\tx\tx\t0.20\tkeep_eligible\n"
NARROW, WIDE = "ucbshift2_sameprot", "shiftx2_sameprot"
NEAR, SHORT = "sparta", "camshift"          # a backbone arm that joins, and one that does not
ARMS = (NARROW, WIDE, NEAR, SHORT)
RETRIEVAL = (NARROW, WIDE)
ATOMS = ("CA", "CB", "CG", "CD", "NE")
NARROW_DELTA_RESIDUES = 3
SHORT_FIRST_RESIDUE = 13                    # the beta carbons the short arm skips entirely


def _emits(tool, row, atom):
    """Which cells a competitor arm returns a value for; the shape of the census."""
    residue = int(row["struct_resnum"])
    stride = int(row["bmrb"]) + residue
    if atom == "CA":
        return True
    if atom == "CB":
        if tool == NEAR:                    # clears the floor, and still misses a residue
            return residue != 1
        if tool == SHORT:
            return residue >= SHORT_FIRST_RESIDUE
        return True
    if tool not in RETRIEVAL:
        return False                        # neither backbone arm labels a side chain here
    if atom == "CG":
        return stride % 7 != 0 if tool == NARROW else stride % 5 != 0
    if atom == "CD":
        # the wide arm clears the floor and still misses a residue of every structure
        return residue <= NARROW_DELTA_RESIDUES if tool == NARROW else residue != 1
    return False


def _write_arms(built):
    cells = built["cells"]
    cells.write_tasks(cells.build_tasks(), built["build"].out_root / "fold6_tasks.tsv")
    labels = {(row["bmrb"], row["struct_resnum"]): row for row in built["rows"]}
    by_sid = {}
    for row in cells.build_row_index()[1]:
        by_sid.setdefault(cells.structure_id(row), []).append(row)
    for tool in ARMS:
        for sid, rows in by_sid.items():
            write_competitor(
                built["build"].out_root, tool, sid,
                {row["struct_resnum"]: {
                    "resname": row["resname"],
                    **{atom: observed_ppm(labels[(row["bmrb"], row["struct_resnum"])],
                                          f"{atom.lower()}_shift") + 1.0
                       for atom in ATOMS if _emits(tool, row, atom)}}
                 for row in rows},
                mode=TIER_MODE if tool in RETRIEVAL else None,
                hitlog=CLEAN_HITLOG if tool in RETRIEVAL else None)


def _covered(built, atom, tools):
    """How many universe cells of one nucleus every named arm returns."""
    total = 0
    for row in built["cells"].build_row_index()[1]:
        if f"{atom.lower()}_shift" not in built["rows"][row["g"]]["observed"]:
            continue
        total += all(_emits(tool, row, atom) for tool in tools)
    return total


@pytest.fixture
def frozen(sidechain_benchmark, tmp_path):
    """One census of the fixture's two arms, and the master cells derived from it."""
    _write_arms(sidechain_benchmark)
    out = tmp_path / "master_cells"
    census = coverage_census.main(["--arms", *ARMS, "--out-dir", str(out)])
    master_cells.main(["--census", str(out / "coverage_census.json"),
                       "--out", str(out / "master_cells.npz")])
    return {"census": census, "path": out / "master_cells.npz",
            "provenance": master_cells.read_report(out / "master_cells.npz"),
            "built": sidechain_benchmark}


def test_the_universe_is_what_deepshift_can_score_after_the_fence(frozen):
    """One beta-carbon cell is fenced out of the run, and the aliased labels are gone."""
    census = frozen["census"]
    expected = (SIDE_CHAIN_STRUCTURES * SIDE_CHAIN_RESIDUES * 4    # four nuclei per residue
                + SIDE_CHAIN_STRUCTURES * SIDE_CHAIN_ARGININES    # the arginine N-epsilon
                - SIDE_CHAIN_ALIASED - 1)                         # aliased labels; the fence
    assert census["n_universe_cells"] == expected
    assert census["n_arg_ne_aliased_excluded"] == SIDE_CHAIN_ALIASED


def test_an_arm_that_returns_a_nucleus_everywhere_joins_it(frozen):
    """Every arm covers the alpha carbon in full, so they join and cost the set nothing."""
    node = frozen["provenance"]["by_nucleus"]["ca_shift"]
    assert sorted(node["joined"]) == sorted(["deepshift", *ARMS])
    assert node["n_master"] == node["n_universe"]


def test_an_arm_below_the_floor_does_not_shrink_the_master_set(frozen):
    """Neither arm reaches the floor on the gamma carbon, so the master set is the
    universe: a predictor that fails on a tenth of the cells never takes the rest down."""
    census = frozen["census"]["arms"]
    assert census[NARROW]["by_nucleus"]["cg_shift"]["coverage"] < MASTER_COVERAGE_FLOOR
    assert census[WIDE]["by_nucleus"]["cg_shift"]["coverage"] < MASTER_COVERAGE_FLOOR
    node = frozen["provenance"]["by_nucleus"]["cg_shift"]
    assert node["joined"] == ["deepshift"]
    assert node["n_master"] == node["n_universe"]


def test_a_marked_arm_never_joins_however_high_its_coverage(frozen, monkeypatch):
    """PLM-CS covers 91 % of fold 6 but its misses are the longest proteins, so it is
    scored on the master cells it has and never lets those proteins out of the set."""
    from benchmarks import config, master_cells
    monkeypatch.setattr(config, "MASTER_MARKED_ARMS", (WIDE,))
    joined = master_cells.joining_arms(frozen["census"], "ca_shift")
    assert WIDE not in joined and "deepshift" in joined
    assert frozen["census"]["arms"][WIDE]["by_nucleus"]["ca_shift"]["coverage"] >= MASTER_COVERAGE_FLOOR


def test_the_master_set_is_the_cells_every_joining_arm_returned(frozen):
    """On the delta carbon the wide arm joins and the narrow one does not, so the master
    set is exactly the wide arm's cells: a joining arm does shrink it, a partial one
    does not."""
    node = frozen["provenance"]["by_nucleus"]["cd_shift"]
    assert sorted(node["joined"]) == sorted(["deepshift", WIDE])
    assert NARROW in node["below_floor"]
    assert node["n_master"] == _covered(frozen["built"], "CD", [WIDE])
    assert node["n_master"] < node["n_universe"]


def test_a_nucleus_no_arm_returns_keeps_its_whole_universe(frozen):
    node = frozen["provenance"]["by_nucleus"]["ne_shift"]
    assert node["joined"] == ["deepshift"]
    assert node["n_master"] == node["n_universe"]
    assert node["n_universe"] == (SIDE_CHAIN_STRUCTURES * SIDE_CHAIN_ARGININES
                                  - SIDE_CHAIN_ALIASED)


def test_the_frozen_set_loads_as_the_four_named_columns(frozen):
    frame = master_cells.load_master(frozen["path"])
    assert list(frame.columns) == list(MASTER_COLUMNS)
    assert len(frame) == sum(node["n_master"] for node in
                             frozen["provenance"]["by_nucleus"].values())
    assert frame["residue"].dtype == np.int64
    assert set(frame["atom"]) <= set(frozen["provenance"]["by_nucleus"])


def test_the_report_records_the_census(frozen):
    report = frozen["provenance"]
    assert report["rule"]["floor"] == MASTER_COVERAGE_FLOOR
    assert report["coverage_census"]["arms"].keys() == frozen["census"]["arms"].keys()


def test_a_master_set_naming_an_unknown_nucleus_is_refused(frozen):
    frame = master_cells.load_master(frozen["path"])
    with pytest.raises(SystemExit, match="no column for"):
        master_cells.master_index(frame, ["ca_shift"])


@pytest.fixture
def scored(frozen, tmp_path):
    """The per-nucleus scorer run on the frozen master cells."""
    return score_by_nucleus.main(
        ["--tier", str(TIER), "--tools", NARROW, WIDE, "--boots", "200",
         "--master-cells", str(frozen["path"]),
         "--out", str(tmp_path / "master_scores.json")])


def test_every_arm_of_a_nucleus_is_scored_on_that_nucleus_s_master_cells(scored, frozen):
    """The alpha carbon: one cell set, and every arm's own count equals it."""
    node = scored["by_nucleus"]["CA"]
    assert node["n_cells"] == frozen["provenance"]["by_nucleus"]["ca_shift"]["n_master"]
    for arm in node["arms"].values():
        assert arm["n_cells"] == node["n_cells"]
        assert arm["below_master_coverage"] is False


def test_a_partial_arm_is_scored_on_the_master_cells_it_returned_and_is_marked(scored, frozen):
    """The gamma carbon: the master set is the whole universe, and each arm is scored on
    the part of it that arm returned, with its own count and a marker."""
    node = scored["by_nucleus"]["CG"]
    assert node["n_cells"] == frozen["provenance"]["by_nucleus"]["cg_shift"]["n_universe"]
    assert node["arms"]["deepshift"]["n_cells"] == node["n_cells"]
    for arm in (NARROW, WIDE):
        assert node["arms"][arm]["n_cells"] == _covered(frozen["built"], "CG", [arm])
        assert node["arms"][arm]["n_cells"] < node["n_cells"]
        assert node["arms"][arm]["below_master_coverage"] is True
    assert scored["arms_below_master_coverage"][NARROW] == sorted(
        scored["arms_below_master_coverage"][NARROW])


def test_a_partial_arm_never_shrinks_what_the_other_arms_are_scored_on(scored, frozen):
    """Without the master rule the gamma carbon would be scored on the cells both arms
    share; with it, DeepShift and each competitor keep every master cell they returned."""
    node = scored["by_nucleus"]["CG"]
    shared = _covered(frozen["built"], "CG", [NARROW, WIDE])
    assert shared < node["arms"][NARROW]["n_cells"]
    assert shared < node["arms"][WIDE]["n_cells"]


def test_the_report_names_the_frozen_set_it_scored(scored, frozen):
    assert scored["master_cells"] == str(frozen["path"])
    assert "master cell set" in scored["scope_rule"]
    sidecar = json.loads((frozen["path"].parent / "coverage_census.json").read_text())
    assert sidecar["coverage_floor"] == MASTER_COVERAGE_FLOOR


# --------- the marginal and paired backbone scorers on the same frozen cells ---------

@pytest.fixture
def backbone(frozen, tmp_path):
    """The marginal and paired backbone reports of one master-cell scoring run."""
    marginal, paired = tmp_path / "marginal.json", tmp_path / "paired.json"
    for module, out, extra in ((score_by_source_tier, marginal, []),
                               (score_paired_tier, paired,
                                ["--check_marginal", str(marginal)])):
        sys.argv = ["scorer", "--tier", str(TIER), "--tools", *ARMS, "--boot", "200",
                    "--master-cells", str(frozen["path"]), "--out", str(out), *extra]
        module.main()
    return json.loads(marginal.read_text()), json.loads(paired.read_text())


def test_the_marginal_scorer_scores_every_arm_on_the_nucleus_s_master_cells(backbone, frozen):
    """The beta carbon: one cell set for the run, each arm's own count beside its error."""
    marginal, _paired = backbone
    node = marginal["by_source"]["all"]["nuclei"]["CB"]
    assert node["n_cells"] == frozen["provenance"]["by_nucleus"]["cb_shift"]["n_master"]
    assert node["deepshift"]["n_cells"] == node["n_cells"]
    assert node[NEAR]["n_cells"] == node["n_cells"]        # it joined, so it lost nothing
    assert node[NEAR]["below_master_coverage"] is False
    assert node[SHORT]["n_cells"] < node["n_cells"]        # scored on what it returned
    assert node[SHORT]["below_master_coverage"] is True
    assert "CB" in marginal["arms_below_master_coverage"][SHORT]
    assert marginal["master_cells"] == str(frozen["path"])


def test_a_partial_arm_does_not_shrink_the_cells_the_others_are_scored_on(backbone, frozen):
    """Without the master rule every arm would lose the short arm's missing residues."""
    marginal, _paired = backbone
    node = marginal["by_source"]["all"]["nuclei"]["CB"]
    assert node["deepshift"]["n_cells"] > node[SHORT]["n_cells"]


def test_a_paired_difference_is_taken_on_the_cells_the_pair_shares(backbone):
    """DeepShift covers every master cell, so a pair's cells are the competitor's own."""
    marginal, paired = backbone
    node, against = marginal["by_source"]["all"]["nuclei"]["CB"], paired["all"]["CB"]
    assert against[SHORT]["n_cells"] == node[SHORT]["n_cells"]
    assert against[SHORT]["competitor"] == pytest.approx(node[SHORT]["rmse"])
    assert against[NEAR]["deepshift"] == pytest.approx(node["deepshift"]["rmse"])


# --------- the join a producer handed the frozen file makes on its own cells ---------

#: A master file written by hand: one record per (nucleus, cell), as the format names it.
HANDMADE = [("ca_shift", "b1_af_l1_a1_e1", "ls_a", 12),
            ("cb_shift", "b1_af_l1_a1_e1", "ls_a", 12),
            ("ca_shift", "b2_nmr_l1_a1_e1", "ls_b", 7)]


def _write_master(path, records):
    np.savez(path, atom=np.array([r[0] for r in records]),
             sid=np.array([r[1] for r in records]),
             label_set=np.array([r[2] for r in records]),
             residue=np.array([r[3] for r in records]))
    return path


def test_the_loader_returns_the_four_columns_the_contract_names(tmp_path):
    frame = master_cells.load_master(_write_master(tmp_path / "master.npz", HANDMADE))
    assert list(frame.columns) == list(MASTER_COLUMNS)
    assert len(frame) == len(HANDMADE)


def test_a_file_missing_a_column_is_refused_by_name(tmp_path):
    path = tmp_path / "partial.npz"
    np.savez(path, atom=np.array(["ca_shift"]), sid=np.array(["b1"]))
    with pytest.raises(SystemExit, match="label_set, residue"):
        master_cells.load_master(path)


def test_restricting_keeps_the_named_cells_and_only_those(tmp_path):
    master = master_cells.load_master(_write_master(tmp_path / "master.npz", HANDMADE))
    keys = master_cells.cell_keys(
        ["ca_shift", "cb_shift", "ca_shift", "n_shift"],
        ["b1_af_l1_a1_e1", "b1_af_l1_a1_e1", "b2_nmr_l1_a1_e1", "b1_af_l1_a1_e1"],
        [12, 12, 7, 12])
    kept = master_cells.restrict(keys, master, what="test")
    assert kept.tolist() == [True, True, True, False]


def test_a_key_the_master_set_names_none_of_stops_the_producer(tmp_path):
    """An empty intersection is a wrong file or a wrong key, never a cohort of no cells."""
    master = master_cells.load_master(_write_master(tmp_path / "master.npz", HANDMADE))
    keys = master_cells.cell_keys(["CA"], ["b1_af_l1_a1_e1"], [12])   # atom names, not columns
    with pytest.raises(SystemExit, match="names none of"):
        master_cells.restrict(keys, master, what="test")


def test_the_frozen_set_restricts_a_producer_s_own_cells(frozen):
    """The join every producer makes: its cells, keyed the one way, against the file."""
    frame = master_cells.load_master(frozen["path"])
    keys = master_cells.cell_keys(frame["atom"][:5], frame["sid"][:5], frame["residue"][:5])
    assert master_cells.restrict(keys, frame, what="master").all()


# --------- the one arm roster, reduced by what this run measured ---------

def _census(tmp_path):
    """A census of three arms: one nucleus two of them return, one only the third does."""
    path = tmp_path / "coverage_census.json"
    path.write_text(json.dumps({"arms": {
        "shiftx2_leaky": {"by_nucleus": {"cb_shift": {"coverage": 0.99}}},
        "shiftx2_sameprot": {"by_nucleus": {"cb_shift": {"coverage": 0.99},
                                            "hz_shift": {"coverage": 0.0}}},
        "sparta": {"by_nucleus": {"cb_shift": {"coverage": 1.0},
                                  "hz_shift": {"coverage": 0.4}}}}}))
    return path


def test_the_arm_roster_is_the_config_one_reduced_by_what_the_run_measured(tmp_path):
    """No producer keeps a second copy of the roster, and a producer of one nucleus does
    not load arms this run's census says return nothing of it."""
    census = _census(tmp_path)
    assert master_cells.arms_returning("cb_shift", census=census) == [
        "shiftx2_leaky", "shiftx2_sameprot", "sparta"]
    assert master_cells.arms_returning("hz_shift", census=census) == ["sparta"]


def test_the_reported_exclusion_selects_which_arms_of_a_method_are_returned(tmp_path):
    """``tiers`` is the exclusion level a float reports: dropping the leaky one leaves
    the same-protein arm of the same method."""
    census = _census(tmp_path)
    assert master_cells.arms_returning("cb_shift", tiers=(2,), census=census) == [
        "shiftx2_sameprot", "sparta"]


def test_with_no_census_the_whole_roster_stands(tmp_path):
    """A producer run before the census exists loads every reported arm and reduces the
    list by its own coverage, rather than inventing one."""
    assert (master_cells.arms_returning("cb_shift", census=tmp_path / "absent.json")
            == [arm for arm in BENCHMARK_ARMS if not arm.endswith("_selfexcl")])


def test_a_rerun_census_is_named_by_the_environment(tmp_path):
    """DEEPSHIFT_COVERAGE_CENSUS points the roster readers at a census re-run beside the
    frozen one, so the frozen census need not be overwritten."""
    import os
    import subprocess
    from pathlib import Path

    rerun = tmp_path / "coverage_census.json"
    found = subprocess.run(
        [sys.executable, "-c",
         "from benchmarks import master_cells; print(master_cells.CENSUS_PATH)"],
        env={**os.environ, "DEEPSHIFT_COVERAGE_CENSUS": str(rerun)},
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    assert found.stdout.strip() == str(rerun)
