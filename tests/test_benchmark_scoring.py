"""The competitor benchmark: residual convention, tiers, model identity, prochiral rule."""
import json
import sys

import numpy as np
import pytest
from benchmark_run import OBSERVED, write_competitor

from benchmarks import score_by_source, score_by_source_tier, score_paired_tier
from deepshift.domain import SHIFT_COLUMNS
from deepshift.self_exclusion import ExclusionLevel
from deepshift.statistics import DEFAULT_SEED


def test_residuals_are_raw():
    parser = score_by_source.build_parser()
    args = parser.parse_args(["--tools", "sparta", "--out", "x.json"])
    assert args.seed == DEFAULT_SEED
    assert score_by_source.RESIDUAL_CONVENTION.startswith("raw")
    with pytest.raises(SystemExit):
        parser.parse_args(["--tools", "sparta", "--out", "x.json", "--center"])


def test_every_self_exclusion_tier_can_be_scored_paired():
    tiers = sorted(score_by_source_tier.LEVEL_BY_TIER)
    assert tiers == [1, 2, 3]
    assert score_by_source_tier.LEVEL_BY_TIER[2] is ExclusionLevel.SAME_PROTEIN
    argv = ["--tier", "2", "--tools", "sparta", "--out", "x.json"]
    parser = score_paired_tier.build_parser()
    assert parser.parse_args(argv).tier == 2


def test_the_benchmark_resolves_prochiral_pairs_by_the_package_rule():
    """One rule for every arm: the benchmark's per-row residuals equal what
    evaluate.resolve gives the same row, and a deposition-ambiguous pair whose labels are
    written the other way round is scored on the swapped assignment."""
    from benchmarks.score_validation import _resolved_cells
    from deepshift import evaluate
    from deepshift.domain import SHIFT_COLUMNS, STANDARD_RESIDUES
    from deepshift.loss import build_loss_tensors

    columns = tuple(SHIFT_COLUMNS)
    partner = build_loss_tensors(columns, tuple(STANDARD_RESIDUES))[1].numpy()
    serine = STANDARD_RESIDUES.index("SER")
    hb2, hb3 = columns.index("hb2_shift"), columns.index("hb3_shift")
    assert partner[serine, hb2] == hb3
    true = np.zeros(len(columns))
    true[hb2], true[hb3] = 3.0, 4.0
    pred = np.full(len(columns), np.nan)
    pred[hb2], pred[hb3] = 4.05, 2.95               # the swapped assignment fits
    ambiguity = np.zeros(len(columns), dtype=int)
    ambiguity[hb2] = ambiguity[hb3] = 2
    cells = _resolved_cells(pred, true, [hb2, hb3], serine, partner, ambiguity)
    residuals = {j: prediction - target for j, (prediction, target) in cells.items()}
    assert residuals == pytest.approx({hb2: 0.05, hb3: -0.05})

    mask = np.zeros((1, len(columns)), dtype=bool)
    mask[0, [hb2, hb3]] = True
    resolved, effective = evaluate.resolve(
        np.where(mask, pred[None, :], 0.0), true[None, :], mask,
        ambiguity[None, :], np.array([serine]), columns)
    assert effective[0, hb2] and effective[0, hb3]
    assert (pred - resolved[0])[[hb2, hb3]] == pytest.approx([0.05, -0.05])


def _competitor(root, tool, sid, rows, offset, *, mode=None, hitlog=None):
    """One competitor arm's output for one structure instance, offset from every label."""
    write_competitor(
        root, tool, sid,
        {row["struct_resnum"]: {
            "resname": row["resname"],
            **{name[:-6].upper(): 50.0 + SHIFT_COLUMNS.index(name) + offset
               for name in OBSERVED}}
         for row in rows},
        mode=mode, hitlog=hitlog)


def _two_arm_cohort(benchmark):
    """A fold-6 task table and two competitor arms over it: one structure-only, one
    retrieval arm declaring the same-protein exclusion mode with a clean hit log."""
    cells = benchmark["cells"]
    tasks = cells.build_tasks()
    cells.write_tasks(tasks, benchmark["build"].out_root / "fold6_tasks.tsv")
    rows_by_sid = {}
    for row in cells.build_row_index()[1]:
        rows_by_sid.setdefault(cells.structure_id(row), []).append(row)
    for sid, rows in rows_by_sid.items():
        _competitor(benchmark["build"].out_root, "sparta", sid, rows, 1.0)
        _competitor(benchmark["build"].out_root, "shiftx2_sameprot", sid, rows, 2.0,
                    mode="same_protein_bmrb_uniprot",
                    hitlog="9999.9zzzA\t9999\tx\tx\tx\t0.20\tkeep_eligible\n")
    return ["sparta", "shiftx2_sameprot"]


def test_the_paired_scorer_reproduces_the_tier_2_cohort_the_marginal_scorer_scored(
        benchmark, monkeypatch, tmp_path):
    """Tier 2 is scoreable both ways, and the paired run's point estimates are the
    marginal run's: --check_marginal is the gate, and it passes on the same cohort."""
    from benchmarks import score_by_source_tier, score_paired_tier

    tools = _two_arm_cohort(benchmark)
    marginal = tmp_path / "score_tier2.json"
    paired = tmp_path / "score_tier2_paired.json"
    argv = ["--tier", "2", "--tools", *tools, "--boot", "16", "--out", str(marginal)]
    monkeypatch.setattr(sys, "argv", ["score_by_source_tier.py", *argv])
    score_by_source_tier.main()
    monkeypatch.setattr(sys, "argv", ["score_paired_tier.py", "--tier", "2", "--tools",
                                      *tools, "--boot", "16", "--out", str(paired),
                                      "--check_marginal", str(marginal)])
    score_paired_tier.main()

    marginal_report = json.loads(marginal.read_text())
    paired_report = json.loads(paired.read_text())
    assert paired_report["tier"] == 2
    assert (paired_report["n_common_structures"], paired_report["n_common_cells"]) == (
        marginal_report["n_common_structures"], marginal_report["n_common_cells"])
    for source, block in marginal_report["by_source"].items():
        for nucleus, arms in block["nuclei"].items():
            for tool in tools:
                assert paired_report[source][nucleus][tool]["competitor"] == pytest.approx(
                    arms[tool]["rmse"])
                assert paired_report[source][nucleus][tool]["deepshift"] == pytest.approx(
                    arms["deepshift"]["rmse"])


def _cohort_inputs(benchmark):
    """The task table, rows and self-exclusion authority a scorer builds a cohort from."""
    import csv

    from benchmarks import score_validation

    path = benchmark["build"].out_root / "fold6_tasks.tsv"
    with open(path) as stream:
        tasks = {task["structure_id"]: task
                 for task in csv.DictReader(stream, delimiter="\t")}
    rows_by_sid, cols, atom_of_col, partner = score_validation.fold6_rows(list(tasks))
    return tasks, rows_by_sid, cols, atom_of_col, partner, benchmark["cells"].SelfExcluder()


def _tier2_cells(benchmark, tool, tasks, rows_by_sid, cols, atom_of_col, partner, excluder):
    from benchmarks import score_by_source, score_by_source_tier

    return score_by_source.tool_cells(
        tool, list(tasks), rows_by_sid, cols, atom_of_col, partner, tasks, excluder,
        verify=lambda arm, sid, task, exc: score_by_source_tier.verify_tiered(
            arm, sid, task, exc, 2))


def test_a_structure_only_arm_is_scored_on_every_structure_it_predicted(benchmark):
    """The arm outside ``RETRIEVAL_ARMS`` leaves no retrieval log, and that is not a
    failure: every structure it predicted enters the cohort."""
    from deepshift.self_exclusion import RETRIEVAL_ARMS

    tools = _two_arm_cohort(benchmark)
    assert tools[0] not in RETRIEVAL_ARMS
    usable, unusable = _tier2_cells(benchmark, tools[0], *_cohort_inputs(benchmark))
    assert unusable == set()
    assert set(usable) == set(_cohort_inputs(benchmark)[0])


def test_an_unregistered_arm_that_leaks_its_own_deposition_is_dropped_from_the_cohort(
        benchmark):
    """A log is read whenever one exists: an arm the retrieval set does not name, caught
    keeping its own deposition, loses that structure instead of being trusted."""
    tools = _two_arm_cohort(benchmark)
    tasks, rows_by_sid, cols, atom_of_col, partner, excluder = _cohort_inputs(benchmark)
    leaking = sorted(tasks)[0]
    log = benchmark["build"].out_root / tools[0] / "hitlog"
    log.mkdir(parents=True, exist_ok=True)
    own = tasks[leaking]["bmrb_id"]
    (log / f"{leaking}.log").write_text(
        f"candidate.9zzzA\t{own}\tx\tx\tx\t0.20\tkeep_eligible\n")

    usable, unusable = _tier2_cells(benchmark, tools[0], tasks, rows_by_sid, cols,
                                    atom_of_col, partner, excluder)
    assert unusable == {leaking}
    assert leaking not in usable


def test_the_intervals_are_clustered_on_label_sets(
        benchmark, tmp_path, monkeypatch):
    """Structure instances of one label set share the deposited shifts, so the unit the
    intervals are reported at is the label set; both scorers say which unit they used."""
    import sys

    from benchmarks import score_by_source_tier, score_paired_tier

    tools = _two_arm_cohort(benchmark)
    marginal = tmp_path / "score_tier2.json"
    paired = tmp_path / "score_tier2_paired.json"
    monkeypatch.setattr(sys, "argv", ["score_by_source_tier.py", "--tier", "2", "--tools",
                                      *tools, "--boot", "16", "--out", str(marginal)])
    score_by_source_tier.main()
    monkeypatch.setattr(sys, "argv", ["score_paired_tier.py", "--tier", "2", "--tools",
                                      *tools, "--boot", "16", "--out", str(paired),
                                      "--check_marginal", str(marginal)])
    score_paired_tier.main()

    marginal_report = json.loads(marginal.read_text())
    paired_report = json.loads(paired.read_text())
    assert marginal_report["unit"] == paired_report["unit"] == "labelset"
    label_sets = {task["label_set_id"] for task in _cohort_inputs(benchmark)[0].values()}
    assert marginal_report["by_source"]["all"]["n_units"] == len(label_sets)


def test_an_arm_rerun_outside_the_benchmark_root_replaces_the_staged_one(
        benchmark, monkeypatch, tmp_path):
    """DEEPSHIFT_ARM_DIRS names a re-run of one arm: the scorer reads its outputs and hit
    logs from there instead of the staged directory."""
    import shutil

    from benchmarks import score_by_source_tier
    from deepshift.self_exclusion import ARM_DIRS_VAR

    tools = _two_arm_cohort(benchmark)
    inputs = _cohort_inputs(benchmark)
    staged = benchmark["build"].out_root / tools[1]
    rerun = tmp_path / "rerun" / "retrieval_arm_rerun"
    shutil.move(staged, rerun)
    assert _tier2_cells(benchmark, tools[1], *inputs)[0] == {}
    monkeypatch.setenv(ARM_DIRS_VAR, f"{tools[1]}={rerun}")
    usable, unusable = _tier2_cells(benchmark, tools[1], *inputs)
    assert unusable == set() and set(usable) == set(inputs[0])

    out = tmp_path / "score.json"
    monkeypatch.setattr(sys, "argv", ["score_by_source_tier.py", "--tier", "2", "--tools",
                                      *tools, "--boot", "8", "--out", str(out)])
    score_by_source_tier.main()
    assert tools[1] in json.loads(out.read_text())["by_source"]["all"]["nuclei"]["CA"]
