"""The all-nucleus scorer: what each nucleus is scored on, and what a verdict claims.

The run root carries eight label sets of arginine and lysine residues and two competitor
arms of unequal coverage: both return the gamma carbon, on overlapping but different
residues; only the wider arm returns the delta carbon on more than a handful of them;
neither returns the arginine N-epsilon, four of whose labels are aliased.
"""

import pytest
from benchmark_run import (
    ARG_NE_COLUMN,
    SIDE_CHAIN_ALIASED,
    SIDE_CHAIN_ARGININES,
    SIDE_CHAIN_STRUCTURES,
    observed_ppm,
    write_competitor,
)

from benchmarks import score_by_nucleus
from benchmarks.score_by_nucleus import DEEPSHIFT, MIN_CELLS
from deepshift.quality_control import ARG_NE_OBSERVED_RANGE_PPM
from deepshift.self_exclusion import DECLARED_MODES, ExclusionLevel

TIER = 2
TIER_MODE = next(iter(DECLARED_MODES[ExclusionLevel.SAME_PROTEIN]))
CLEAN_HITLOG = "9999.9zzzA\t9999\tx\tx\tx\t0.20\tkeep_eligible\n"
NARROW, WIDE = "ucbshift2_sameprot", "shiftx2_sameprot"
OFFSET = {NARROW: 1.0, WIDE: 1.4}          # ppm this arm sits off every label it returns
NARROW_DELTA_RESIDUES = 3                  # too few delta-carbon cells to clear the floor
ATOMS = ("CA", "CB", "CG", "CD", "NE")


def _emits(tool, row, atom):
    """Which cells a competitor arm returns a value for."""
    residue = int(row["struct_resnum"])
    stride = int(row["bmrb"]) + residue
    if atom in ("CA", "CB"):
        return True
    if atom == "CG":
        return stride % 7 != 0 if tool == NARROW else stride % 5 != 0
    if atom == "CD":
        return residue <= NARROW_DELTA_RESIDUES if tool == NARROW else True
    return False


def _write_arms(built):
    """The fold-6 task table, and both competitor arms over every structure in it."""
    cells = built["cells"]
    cells.write_tasks(cells.build_tasks(), built["build"].out_root / "fold6_tasks.tsv")
    labels = {(row["bmrb"], row["struct_resnum"]): row for row in built["rows"]}
    by_sid = {}
    for row in cells.build_row_index()[1]:
        by_sid.setdefault(cells.structure_id(row), []).append(row)
    for tool in (NARROW, WIDE):
        for sid, rows in by_sid.items():
            write_competitor(
                built["build"].out_root, tool, sid,
                {row["struct_resnum"]: {
                    "resname": row["resname"],
                    **{atom: observed_ppm(labels[(row["bmrb"], row["struct_resnum"])],
                                          f"{atom.lower()}_shift") + OFFSET[tool]
                       for atom in ATOMS if _emits(tool, row, atom)}}
                 for row in rows},
                mode=TIER_MODE, hitlog=CLEAN_HITLOG)


def _cells_covered(built, atom, tools):
    """The cells DeepShift and every named arm cover for one nucleus.

    The arginine N-epsilon range is read on the observed label, as the scorer reads it.
    The run's fitted fence takes one backbone cell out of this run root and none of the
    side-chain cells counted here.
    """
    low, high = ARG_NE_OBSERVED_RANGE_PPM
    column = f"{atom.lower()}_shift"
    labels = {(row["bmrb"], row["struct_resnum"]): row for row in built["rows"]}
    total = 0
    for row in built["cells"].build_row_index()[1]:
        deposited = labels[(row["bmrb"], row["struct_resnum"])]
        if column not in deposited["observed"]:
            continue
        if column == ARG_NE_COLUMN and not low <= observed_ppm(deposited, column) <= high:
            continue
        total += all(_emits(tool, row, atom) for tool in tools)
    return total


@pytest.fixture
def report(sidechain_benchmark, tmp_path):
    """One scoring run of the fixture's run root, at the tier the manuscript deploys."""
    _write_arms(sidechain_benchmark)
    return score_by_nucleus.main(
        ["--tier", str(TIER), "--tools", NARROW, WIDE, "--boots", "200",
         "--out", str(tmp_path / "scores.json")])


def test_a_nucleus_both_competitors_emit_is_scored_on_the_cells_all_three_cover(
        sidechain_benchmark, report):
    node = report["by_nucleus"]["CG"]
    assert node["emitting_competitors"] == [NARROW, WIDE]
    assert set(node["arms"]) == {DEEPSHIFT, NARROW, WIDE}
    assert node["n_cells"] == _cells_covered(sidechain_benchmark, "CG", [NARROW, WIDE])
    assert node["n_cells"] < min(node["cells_each_arm_covers"].values())
    assert node["n_label_sets"] == SIDE_CHAIN_STRUCTURES


def test_a_nucleus_one_competitor_emits_is_scored_on_the_cells_that_pair_covers(
        sidechain_benchmark, report):
    node = report["by_nucleus"]["CD"]
    assert node["cells_each_arm_covers"][NARROW] < MIN_CELLS
    assert node["emitting_competitors"] == [WIDE]
    assert set(node["arms"]) == {DEEPSHIFT, WIDE}
    assert list(node["vs_deepshift"]) == [WIDE]
    assert node["n_cells"] == _cells_covered(sidechain_benchmark, "CD", [WIDE])


def test_a_nucleus_no_competitor_emits_is_scored_on_deepshift_alone(
        sidechain_benchmark, report):
    node = report["by_nucleus"]["NE"]
    assert node["emitting_competitors"] == []
    assert list(node["arms"]) == [DEEPSHIFT]
    assert node["vs_deepshift"] == {}
    assert node["n_cells"] == _cells_covered(sidechain_benchmark, "NE", [])
    assert report["scope_counts"]["side_chain_with_no_incumbent"] == ["NE"]


def test_an_arginine_n_epsilon_label_outside_the_observed_range_leaves_every_arm(report):
    """The fence reads the observed label and never the residual: every arm tracks the
    aliased labels as closely as the sound ones, so only a fence on the label removes them.
    """
    fence = report["exclusions_on_observed"]["arg_ne_sweep_width"]
    assert fence["accepted_ppm"] == list(ARG_NE_OBSERVED_RANGE_PPM)
    assert fence["n_cells_removed"] == SIDE_CHAIN_ALIASED
    node = report["by_nucleus"]["NE"]
    assert node["n_cells"] == (SIDE_CHAIN_STRUCTURES * SIDE_CHAIN_ARGININES
                               - SIDE_CHAIN_ALIASED)
    assert node["cells_each_arm_covers"][DEEPSHIFT] == node["n_cells"]


def test_the_verdict_is_a_win_when_the_interval_on_the_difference_excludes_zero(report):
    against = report["by_nucleus"]["CG"]["vs_deepshift"][WIDE]["rmse"]
    assert against["deepshift"] < against["competitor"]
    assert against["diff"] < 0 and against["ci95"][1] < 0
    assert against["verdict"] == "win"
    counted = report["tally"][WIDE]["side_chain"]
    assert counted["win"] == sum(1 for node in report["by_nucleus"].values()
                                 if not node["backbone"] and WIDE in node["vs_deepshift"])
    assert counted["tie"] == 0 and counted["loss"] == 0


def test_a_task_table_naming_no_cell_of_the_dumps_stops_the_run(sidechain_benchmark,
                                                                tmp_path):
    """A task table from another cohort names structures this cache fold has no row for,
    which is a cohort of nothing rather than a benchmark with nothing to say."""
    _write_arms(sidechain_benchmark)
    table = sidechain_benchmark["build"].out_root / "fold6_tasks.tsv"
    header, first = table.read_text().splitlines()[:2]
    table.write_text(f"{header}\n{first.replace('b100_', 'b999_', 1)}\n")
    with pytest.raises(SystemExit, match="nothing to score"):
        score_by_nucleus.main(["--tier", str(TIER), "--tools", NARROW, WIDE,
                               "--boots", "8", "--out", str(tmp_path / "empty.json")])
