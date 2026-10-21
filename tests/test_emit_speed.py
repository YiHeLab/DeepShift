"""Table S27: what each block times, and how a startup that cannot be negative is fitted."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

SI_BUILD = Path(__file__).resolve().parents[1] / "si_build"

RESIDUE_COUNTS = list(range(100, 200, 5))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SI_BUILD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _emit_speed(monkeypatch, census_dir, out_dir):
    monkeypatch.setenv("SI_SPEED_CENSUS", str(census_dir))
    monkeypatch.setenv("SI_OUT", str(out_dir))
    sys.path.insert(0, str(SI_BUILD))
    try:
        _load("si_common")
        return _load("emit_speed")
    finally:
        sys.path.remove(str(SI_BUILD))


def _timings(runs, tag, startup, rate):
    """A per-structure record whose wall clock is startup plus a cost per residue."""
    records = [{"status": "ok", "n_aa": n, "wall_sec": startup + rate * n}
               for n in RESIDUE_COUNTS]
    path = runs / f"{tag}__full__r0.json"
    path.write_text(json.dumps({"records": records}))
    return path, records


def _census(tmp_path, module, *, scaling_dominated, batch_chains_finished=None):
    census_dir = tmp_path / "census"
    """A census in the shape emit_speed reads, with one arm's startup below zero.

    ``scaling_dominated`` names the arm whose wall clock is all cost per residue, so an
    unconstrained fit puts its startup below zero; every other arm gets a positive one.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    arms = []
    for index, (tag, _label) in enumerate(module.COLD):
        startup = -1.0 if tag == scaling_dominated else 0.5 + 0.1 * index
        rate = 0.05
        path, records = _timings(runs, tag, startup, rate)
        seconds = sum(r["wall_sec"] for r in records) / len(records)
        arms.append({
            "arm": tag, "method": tag, "hardware": "20 CPU", "cohort": "cohort",
            "n_structures_timed": len(records), "n_residues_total": sum(RESIDUE_COUNTS),
            "per_protein_mean_sec": seconds,
            "per_protein_mean_ci95": [seconds * 0.9, seconds * 1.1],
            "fixed_cost_intercept_sec": startup,
            "fixed_cost_intercept_ci95": [startup - 0.1, startup + 0.1],
            "marginal_ms_per_residue": 1000.0 * rate,
            "marginal_ms_per_residue_ci95": [1000.0 * rate * 0.9, 1000.0 * rate * 1.1],
            "per_residue_ms": 1000.0 * rate, "cold_start_sec": 0.2,
            "coverage_all240_fold6_xtal": {"n_ok": 240, "n_tasks": 240, "n_found": 240},
            "measured_parallelism": 1.0, "source_file": str(path),
        })
    passes = [(tag, "r0") for tag, _label, _hardware in module.BATCH_TOOLS]
    passes += [(tag, rep) for tag, _l, _h, reps in module.REPEATED for rep in reps]
    for order, (tag, rep) in enumerate(passes):
        chains = len(RESIDUE_COUNTS)
        finished = chains if batch_chains_finished is None else batch_chains_finished
        (runs / f"{tag}__batch__{rep}.json").write_text(json.dumps({
            "n_chains": chains, "n_outputs": finished, "n_residues": sum(RESIDUE_COUNTS),
            "total_sec": 2.0 + order}))
    _ledger(runs, module, verdict="kept")
    census = {"cohort_n": len(RESIDUE_COUNTS), "arms": arms}
    census_dir.mkdir(exist_ok=True)
    (census_dir / "speed_census.json").write_text(json.dumps(census))
    return census


def _ledger(runs, module, *, verdict):
    """The retime ledger, one row per pass, as its driver writes it."""
    rows = [f"run_batch_parallel_prep.py {tag.removesuffix('_prep8')} 8 {rep[1:]}"
            for tag, _l, _h, reps in module.REPEATED if tag.endswith("_prep8")
            for rep in reps]
    rows += [f"run_speed_census.py --arm {tag}" for tag, _l in module.COLD
             if tag.startswith("deepshift")]
    (runs / "ledger.tsv").write_text(
        "pass\tattempt\tstarted\tfinished\treturncode\tpeak_foreign_cores\tverdict\n"
        + "".join(f"{row}\t1\t-\t-\t0\t0.10\t{verdict}\n" for row in rows))


def _startup_column(table):
    """The startup cell of every per-invocation row, in printed order."""
    block = table.split(r"\midrule")[1]
    return [line.split("&")[3].strip()
            for line in block.splitlines() if line.startswith(r"\quad ")]


def test_a_startup_that_cannot_be_negative_is_fitted_under_that_constraint(
        monkeypatch, tmp_path):
    """Startup is a physical quantity, so no row prints a negative one. An arm whose
    plain fit lands below zero prints zero with the rate refit through the origin; every
    other arm is an interior solution and its printed numbers do not move."""
    module = _emit_speed(monkeypatch, tmp_path / "census", tmp_path / "out")
    census = _census(tmp_path, module, scaling_dominated="ucbshift_x")
    arms = {a["arm"]: a for a in census["arms"]}
    overrides = {k: module.constrained_fit(arms[k]["source_file"]) for k, _ in module.COLD}

    assert overrides["sparta"] is None
    bound = overrides["ucbshift_x"]
    assert bound["fixed_cost_intercept_sec"] == 0.0
    assert bound["fixed_cost_intercept_ci95"][0] >= 0.0
    # The constant the constraint removed is absorbed into the rate, so the refit rate is
    # below the plain slope, not equal to it.
    assert bound["marginal_ms_per_residue"] < arms["ucbshift_x"]["marginal_ms_per_residue"]

    table = module.build_table(census, arms, overrides, "caption")
    assert not [cell for cell in _startup_column(table) if cell.startswith("-")]
    assert f"{bound['fixed_cost_intercept_sec']:.2f}" in table
    assert f"{bound['marginal_ms_per_residue']:.2f}" in table


def test_the_batch_block_carries_every_tool_that_ships_a_batch_interface(
        monkeypatch, tmp_path):
    """A batch row is a cohort total through the tool's own batch interface, so the block
    is not one method's lane: every tool that ships one is in it, charged per protein and
    per residue from that one total, and a repeated pass prints the mean of its repeats."""
    module = _emit_speed(monkeypatch, tmp_path / "census", tmp_path / "out")
    census = _census(tmp_path, module, scaling_dominated="ucbshift_x")
    arms = {a["arm"]: a for a in census["arms"]}
    overrides = {k: module.constrained_fit(arms[k]["source_file"]) for k, _ in module.COLD}
    runs = module.census_runs(arms)

    batch = module.build_table(census, arms, overrides, "caption").split(r"\midrule")[2]
    printed = [line.split("&")[0].removeprefix(r"\quad ").removeprefix(r"\textbf{")
               .strip().removesuffix("}")
               for line in batch.splitlines() if line.startswith(r"\quad ")]
    payloads = [(label, module.load_batch(runs, tag)) for tag, label, _h in module.BATCH_TOOLS]
    payloads += [(label, module.repeated_payload(runs, tag, reps))
                 for tag, label, _h, reps in module.REPEATED]
    for label, payload in payloads:
        assert label in printed
        per_protein = payload["total_sec"] / payload["n_chains"]
        rate = 1000.0 * payload["total_sec"] / arms["sparta"]["n_residues_total"]
        assert re.search(rf"{re.escape(label)}}}? & .*{per_protein:.2f} & --- "
                         rf"& {rate:.2f} & ---", batch)


def test_a_missing_repeat_of_a_batch_pass_refuses_the_table(monkeypatch, tmp_path):
    """The REDUCE tools print their idle-machine repeats; until those passes are on disk
    no table is written, rather than one mixing them with an older pass."""
    module = _emit_speed(monkeypatch, tmp_path / "census", tmp_path / "out")
    census = _census(tmp_path, module, scaling_dominated="ucbshift_x")
    runs = module.census_runs({a["arm"]: a for a in census["arms"]})
    tag, _label, _hardware, reps = module.REPEATED[-1]
    (runs / f"{tag}__batch__{reps[0]}.json").unlink()
    with pytest.raises(SystemExit, match="batch passes not on disk"):
        module.main()


def test_a_batch_pass_that_left_chains_unfinished_is_not_a_row(monkeypatch, tmp_path):
    """A pass that did not finish the cohort is not comparable with one that did, so no
    table is written from it."""
    module = _emit_speed(monkeypatch, tmp_path / "census", tmp_path / "out")
    census = _census(tmp_path, module, scaling_dominated="ucbshift_x",
                     batch_chains_finished=1)
    arms = {a["arm"]: a for a in census["arms"]}
    with pytest.raises(SystemExit, match="REFUSING to emit"):
        module.load_batch(module.census_runs(arms), module.BATCH_TOOLS[0][0])


def test_a_retimed_pass_the_ledger_did_not_keep_refuses_the_table(monkeypatch, tmp_path):
    """A pass that something else ran underneath is contended; until a clean repeat is
    kept, no table is written from it."""
    module = _emit_speed(monkeypatch, tmp_path / "census", tmp_path / "out")
    census = _census(tmp_path, module, scaling_dominated="ucbshift_x")
    runs = module.census_runs({a["arm"]: a for a in census["arms"]})
    _ledger(runs, module, verdict="rejected")
    with pytest.raises(SystemExit, match="ledger did not keep"):
        module.main()


def test_a_row_without_a_label_prints_the_census_method_name(monkeypatch, tmp_path):
    """SHIFTX2 was timed with SHIFTY+ disabled; its row prints the census's own name for
    that configuration rather than the bare tool name."""
    module = _emit_speed(monkeypatch, tmp_path / "census", tmp_path / "out")
    census = _census(tmp_path, module, scaling_dominated="ucbshift_x")
    for arm in census["arms"]:
        if arm["arm"] == "shiftx2_noshifty":
            arm["method"] = "SHIFTX2, SHIFTY+ disabled"
    arms = {a["arm"]: a for a in census["arms"]}
    overrides = {k: module.constrained_fit(arms[k]["source_file"]) for k, _ in module.COLD}

    assert dict(module.cold_rows(arms))["shiftx2_noshifty"] == "SHIFTX2, SHIFTY+ disabled"
    cold = module.build_table(census, arms, overrides, "caption").split(r"\midrule")[1]
    assert r"\quad SHIFTX2, SHIFTY+ disabled & " in cold
