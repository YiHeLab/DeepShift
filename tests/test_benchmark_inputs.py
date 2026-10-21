"""What the fold-6 competitor benchmark reads, and what it refuses.

The benchmark resolves every scientific input from one run root and scores the cells the
run's own fold-6 prediction dumps declare observed. These tests pin the loader and the
fence against the run root ``benchmark_run`` builds.
"""
import numpy as np
import pytest
from benchmark_run import MEMBER_FOLDS, OBSERVED, _predictions, build_run

from deepshift import evaluate
from deepshift.domain import SHIFT_COLUMNS


def test_the_benchmark_reads_the_dumps_deepshift_evaluate_writes(benchmark):
    """The ensemble the scorers read is the release-schema member dumps, averaged in ppm
    over this cache's rows, in this cache's row order."""
    loaded = benchmark["cells"].load_pred_npz()
    members = [evaluate.read_predictions(path) for path in benchmark["dumps"]]
    assert loaded["cols"] == list(SHIFT_COLUMNS)
    np.testing.assert_allclose(loaded["pred"],
                               np.mean([m.pred_ppm for m in members], axis=0))
    np.testing.assert_array_equal(loaded["mask"], benchmark["fenced"])
    np.testing.assert_allclose(loaded["true_raw"], members[0].true_ppm)


def test_the_scored_cells_are_the_ones_the_run_fence_left(benchmark):
    """A cell the run's exclusion fence removed is scorable for nobody: it is absent from
    the observed cells every arm is compared on."""
    rows = benchmark["cells"].build_row_index()[1]
    ids = [benchmark["cells"].structure_id(row) for row in rows]
    by_sid = benchmark["score_validation"].fold6_rows(ids)[0]
    scored = {(row["struct_resnum"], column)
              for cells in by_sid.values() for row in cells
              for column in row["obs_cols"]}
    assert ("6", SHIFT_COLUMNS.index("cb_shift")) not in scored     # the flagged cell
    assert ("6", SHIFT_COLUMNS.index("ca_shift")) in scored
    assert len(scored) == int(benchmark["fenced"].sum())
    assert len(scored) == len(benchmark["rows"]) * len(OBSERVED) - 1


def test_the_benchmark_refuses_predictions_from_another_cache_fold(benchmark, tmp_path):
    """Row identity is checked, not assumed: dumps whose rows are not this cache's rows
    would be read straight onto the wrong residues."""
    cells = benchmark["cells"]
    other = build_run(tmp_path / "second")
    maps = dict(other["maps"])
    maps["label_residue_id"] = {key: value + "x"
                                for key, value in maps["label_residue_id"].items()}
    for member, path in zip(MEMBER_FOLDS, other["dumps"]):
        evaluate.write_predictions(
            _predictions(other["rows"], maps, other["fenced"], 0.1 * member), path)
    with pytest.raises(SystemExit, match="row identities differ"):
        cells.fold6_ensemble(other["dumps"])


def test_the_benchmark_resolves_every_authority_from_one_run_root(tmp_path, monkeypatch):
    """One variable names the run; nothing else is a root the operator has to know."""
    monkeypatch.setenv("DEEPSHIFT_RUN", str(tmp_path / "run"))
    monkeypatch.delenv("DEEPSHIFT_BENCH_OUT", raising=False)
    from benchmarks import config

    build = config.current()
    root = tmp_path / "run"
    assert build.cache == root / "work/unified_cache/fold_6"
    assert build.fold_map.parent == build.data == root / "data"
    assert build.uniprot_ledger.parent == root / "data"
    assert build.flags.parent == root / "p100_fit"
    assert build.out_root == root / "benchmark"
    assert list(build.members) == [root / f"final/intervals/fold6_member{fold}.npz"
                                   for fold in MEMBER_FOLDS]


def test_a_missing_run_root_names_the_variable_it_wants(monkeypatch):
    monkeypatch.delenv("DEEPSHIFT_RUN", raising=False)
    monkeypatch.delenv("DEEPSHIFT_BENCH_OUT", raising=False)
    from benchmarks import config

    build = config.current()
    assert "__DEEPSHIFT_RUN_REQUIRED__" in str(build.cache)
    assert any("__DEEPSHIFT_RUN_REQUIRED__" in gap for gap in config.missing(build))


def test_the_benchmark_requires_only_the_inputs_it_reads(tmp_path, monkeypatch):
    """`missing()` is the pre-flight for the files the benchmark opens, so a run holding
    every one of them reports no gap. A path required but never opened would stop a run
    over a file whose absence changes nothing."""
    import csv as csv_module

    from deepshift.data.uniprot_snapshot import LEDGER_COLUMNS

    run = tmp_path / "run"
    monkeypatch.setenv("DEEPSHIFT_RUN", str(run))
    monkeypatch.setenv("DEEPSHIFT_SIFTS_CSV", str(tmp_path / "sifts.csv"))
    monkeypatch.delenv("DEEPSHIFT_BENCH_OUT", raising=False)
    from benchmarks import config

    build = config.current()
    for directory in (build.structural, build.atom, build.data, *build.pdb_dirs.values(),
                      build.flags.parent):
        directory.mkdir(parents=True, exist_ok=True)
    build.fold_map.write_text("{}")
    build.flags.write_text("{}")
    (tmp_path / "sifts.csv").write_text("pdb,chain,uniprot\n")
    for name in ("id_maps.json", "global_to_struct_resid.json", "shift_mask.npy"):
        (build.structural / name).write_text("")
    (build.atom / "atom_count.npy").write_text("")
    for edge in config.EDGE_FILES:
        (build.data / edge).write_text("label_set_a,label_set_b\n")
    for source in ("af", "nmr", "xtal"):
        with (build.data / f"structure_build_log_{source}.csv").open("w", newline="") as fh:
            csv_module.writer(fh).writerow(("materialization_id", "pdb_path", "chain_id"))
    with build.uniprot_ledger.open("w", newline="") as fh:
        csv_module.writer(fh).writerow(LEDGER_COLUMNS)

    assert config.missing(build) == []


def test_the_fixture_keeps_the_benchmark_output_inside_its_own_run(tmp_path, monkeypatch):
    """A suite run in a shell that names a real benchmark directory writes nothing there:
    the fixture binds the output root under its temporary run before any module reads it."""
    from benchmark_run import bind, build_run

    monkeypatch.setenv("DEEPSHIFT_BENCH_OUT", str(tmp_path / "elsewhere"))
    built = bind(monkeypatch, build_run(tmp_path))
    assert built["build"].out_root == built["run"] / "benchmark"
    assert built["run"] / "benchmark" == built["cells"].OUT_ROOT
