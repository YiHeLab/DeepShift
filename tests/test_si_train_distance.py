"""The training-distance table measures the held-out fold against the training folds.

The search itself is replaced by a stub, so what is pinned here is the cohort (every
structure realizing the held-out fold), the two sequence sets handed to the search, and
the bin each measured identity lands in.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "si_build"))

import census_fixture as fx


@pytest.fixture
def producer(tmp_path, monkeypatch):
    monkeypatch.setenv("SI_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("DEEPSHIFT_RUN", str(tmp_path / "run"))
    sys.modules.pop("si_common", None)
    sys.modules.pop("emit_train_distance", None)
    import emit_train_distance as module

    # The second training set is a root the release shell exports; every test here
    # names its own, whatever shell it runs in.
    second = tmp_path / "second_training_set.fasta"
    second.write_text(">other\nACDEFGHIKL\n")
    monkeypatch.setenv(module.SECOND_SET_VAR, str(second))
    data_dir = fx.write_build(tmp_path / "run" / "data")
    fx.write_entity_fasta(data_dir)
    return module, tmp_path / "out", data_dir


def stub_search(module, monkeypatch, hits, seen=None):
    """The search, stubbed; ``seen`` records the first call, against our training folds."""
    def search(queries, targets, **kwargs):
        if seen is not None and not seen:
            seen.update({"queries": set(queries), "targets": set(targets),
                         "kwargs": kwargs})
        return {q: h for q, h in hits.items() if q in queries}, "stubbed search"
    monkeypatch.setattr(module, "run_best_hit_identity", search)


def run(module):
    sys.argv = ["emit_train_distance.py"]
    module.main()


def table(out: Path) -> str:
    return (out / "si_train_distance.tex").read_text()


def count_in_row(out: Path, label: str) -> int:
    """The structures counted against our training folds in the table row ``label``."""
    row = next(line for line in table(out).splitlines() if line.startswith(label + " &"))
    return int(re.sub(r"\D", "", row.split(" & ")[1]))


def test_the_cohort_is_every_structure_realizing_the_held_out_fold(producer, monkeypatch):
    module, out, _ = producer
    seen = {}
    stub_search(module, monkeypatch,
                {"le_5": {"target": "le_1", "identity": 0.72, "query_coverage": 0.95}},
                seen)
    run(module)
    assert seen["queries"] == {"le_5"}
    assert seen["targets"] == {"le_1", "le_2", "le_3", "le_4"}
    assert "Total & 1 & 100.0" in table(out)


@pytest.mark.parametrize("identity,label", [
    (0.29, r"Below 30\%"), (0.30, r"30--50\%"), (0.64, r"50--65\%"),
    (0.65, r"65--80\%"), (0.80, r"80--90\%"), (0.90, r"90\% and above"),
    (0.99, r"90\% and above"),
])
def test_a_measured_identity_lands_in_the_bin_it_opens(producer, monkeypatch,
                                                       identity, label):
    module, out, _ = producer
    stub_search(module, monkeypatch,
                {"le_5": {"target": "le_1", "identity": identity,
                          "query_coverage": 0.95}})
    run(module)
    counts = {name: count_in_row(out, name) for name in module.LABELS}
    assert counts[label] == 1
    assert sum(counts.values()) == 1


def test_a_structure_with_no_hit_is_counted_as_having_no_relative(producer, monkeypatch):
    module, out, _ = producer
    stub_search(module, monkeypatch, {})
    run(module)
    assert count_in_row(out, "No relative found") == 1
    assert "0 against our training folds" in table(out)


def test_a_structure_whose_entity_has_no_sequence_is_reported_unmeasured(
        producer, monkeypatch):
    module, out, data_dir = producer
    fx.write_entity_fasta(data_dir, {k: v for k, v in fx.ENTITY_SEQUENCES.items()
                                     if k != "le_5"})
    stub_search(module, monkeypatch, {})
    run(module)
    assert count_in_row(out, "No relative found") == 1
    assert "1 against our training folds" in table(out)


def test_the_best_hit_search_asks_for_identity_and_coverage_over_every_target(monkeypatch):
    """The instrument behind the table: one search, wide open, best hit per query."""
    from deepshift.data import sequence_clustering

    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        Path(command[4]).write_text(
            "q1\tt1\t0.40\t0.90\nq1\tt2\t0.80\t0.70\nq2\tt1\t0.55\t0.95\n")
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(sequence_clustering.subprocess, "run", fake_run)
    hits, command = sequence_clustering.run_best_hit_identity(
        {"q1": "ACDE", "q2": "EFGH"}, {"t1": "ACDE", "t2": "EFGH"},
        coverage=0.5, cov_mode=2, binary="mmseqs")

    assert hits["q1"] == {"target": "t2", "identity": 0.80, "query_coverage": 0.70}
    assert hits["q2"]["target"] == "t1"
    argv = seen["command"]
    assert argv[1] == "easy-search"
    assert argv[argv.index("--format-output") + 1] == "query,target,fident,qcov"
    assert int(argv[argv.index("-e") + 1]) == sequence_clustering.EVALUE
    assert argv[argv.index("--max-seqs") + 1] == "2"
    assert argv[argv.index("--max-accept") + 1] == "2"
    assert argv[argv.index("-c") + 1] == "0.5"
    assert argv[argv.index("--cov-mode") + 1] == "2"
    assert command == " ".join(argv)


def test_the_second_training_set_comes_from_its_root_when_no_request_names_one(
        producer, monkeypatch, tmp_path):
    """The deployed table carries a column against UCBShift's training set; the root
    supplies that FASTA so the producer regenerates it with no argument, and an explicit
    request replaces it."""
    module, _out, _data = producer
    fasta = tmp_path / "their_training.fasta"
    fasta.write_text(">a\nACDE\n")
    monkeypatch.setenv(module.SECOND_SET_VAR, str(fasta))
    assert module.second_sets([]) == [f"{module.SECOND_SET_NAME}={fasta}"]
    assert module.second_sets(["theirs=/elsewhere.fasta"]) == ["theirs=/elsewhere.fasta"]
    monkeypatch.delenv(module.SECOND_SET_VAR)
    # Without it the table would print one column fewer than the deployed one; it refuses.
    with pytest.raises(SystemExit, match=module.SECOND_SET_VAR):
        module.second_sets([])
