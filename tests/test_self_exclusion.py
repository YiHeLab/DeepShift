import json
from pathlib import Path
from typing import ClassVar

import pytest

from deepshift.self_exclusion import (
    ARM_DIRS_VAR,
    RETRIEVAL_ARMS,
    ExclusionLevel,
    arm_directory,
    verify_self_exclusion,
)


class Authority:
    pdb_uniprot: ClassVar[dict] = {("1ABC", "A"): {"P12345"}}

    def query_uniprots(self, *_args):
        return {"P12345"}


TASK = {
    "bmrb_id": "10",
    "exclude_bmrbs": "10,11",
    "source": "xtal",
    "accession": "1ABCA",
    "chain": "A",
}


def _log(root: Path, line: str):
    path = root / "tool" / "hitlog" / "target.log"
    path.parent.mkdir(parents=True)
    path.write_text(line)


@pytest.mark.parametrize(
    ("level", "bmrb", "identity", "pdb_chain", "passed"),
    [
        (ExclusionLevel.OWN_DEPOSITION, "10", 0.20, "2XYZA", False),
        (ExclusionLevel.OWN_DEPOSITION, "11", 0.99, "1ABCA", True),
        (ExclusionLevel.SAME_PROTEIN, "11", 0.20, "2XYZA", False),
        (ExclusionLevel.SAME_PROTEIN, "12", 0.99, "2XYZA", True),
        (ExclusionLevel.SAME_PROTEIN, "12", 0.20, "1ABCA", False),
        (ExclusionLevel.BROAD_HOMOLOGY, "12", 0.90, "2XYZA", False),
    ],
)
def test_the_three_exclusion_levels(tmp_path, level, bmrb, identity, pdb_chain, passed):
    root = tmp_path
    _log(root, f"candidate.{pdb_chain}\t{bmrb}\tx\tx\tx\t{identity}\tkeep_eligible\n")
    actual, _violations = verify_self_exclusion(
        "tool",
        "target",
        TASK,
        Authority(),
        output_root=root,
        retrieval=True,
        level=level,
    )
    assert actual is passed
def test_missing_retrieval_log_fails_closed(tmp_path):
    passed, violations = verify_self_exclusion(
        "tool",
        "target",
        TASK,
        Authority(),
        output_root=tmp_path,
        retrieval=True,
        level=ExclusionLevel.SAME_PROTEIN,
    )
    assert not passed
    assert violations[0]["reason_codes"] == ["missing_hitlog"]


def test_nonretrieval_arm_needs_no_hit_log(tmp_path):
    assert verify_self_exclusion(
        "tool",
        "target",
        TASK,
        Authority(),
        output_root=tmp_path,
        retrieval=False,
        level=ExclusionLevel.SAME_PROTEIN,
    ) == (True, [])


def test_an_empty_log_records_that_no_reference_was_used(tmp_path):
    _log(tmp_path, "")
    assert verify_self_exclusion(
        "tool",
        "target",
        TASK,
        Authority(),
        output_root=tmp_path,
        retrieval=True,
        level=ExclusionLevel.SAME_PROTEIN,
    ) == (True, [])


def test_a_log_with_no_recognised_record_fails_closed(tmp_path):
    _log(tmp_path, "unrecognized\n")
    passed, violations = verify_self_exclusion(
        "tool",
        "target",
        TASK,
        Authority(),
        output_root=tmp_path,
        retrieval=True,
        level=ExclusionLevel.SAME_PROTEIN,
    )
    assert not passed
    assert violations[0]["reason_codes"] == ["malformed_hitlog"]


def test_a_log_is_inspected_even_when_the_arm_is_not_declared_a_retriever(tmp_path):
    """An arm outside ``RETRIEVAL_ARMS`` that leaves a log recording its own deposition
    is refused: the arm's membership decides only whether a missing log is a failure,
    and that is how the benchmark derives ``retrieval``."""
    tool = "tool"
    assert tool not in RETRIEVAL_ARMS
    _log(tmp_path, "candidate.2XYZA\t10\tx\tx\tx\t0.20\tkeep_eligible\n")
    passed, violations = verify_self_exclusion(
        tool,
        "target",
        TASK,
        Authority(),
        output_root=tmp_path,
        retrieval=tool in RETRIEVAL_ARMS,
        level=ExclusionLevel.OWN_DEPOSITION,
    )
    assert not passed
    assert violations[0]["reason_codes"] == ["own_deposition"]


def test_every_arm_that_searches_a_transfer_library_fails_closed_without_a_log(tmp_path):
    """The frozen exclusion record names the arms that search UCBShift's or SHIFTY+'s
    library. Each must be a retrieval arm, or a structure it left no hit log for is
    scored instead of dropped (ucbshift2_selfexcl was missing until 2026-10-07)."""
    from benchmarks import config
    from benchmarks.score_by_source import REFERENCE

    frozen = json.loads((REFERENCE / "competitor_same_protein.json").read_text())
    searching = sorted({arm for arms in frozen["arms"].values() for arm in arms})
    assert set(searching) <= set(config.BENCHMARK_ARMS)
    for arm in searching:
        assert arm in RETRIEVAL_ARMS, arm
        passed, violations = verify_self_exclusion(
            arm,
            "target",
            TASK,
            Authority(),
            output_root=tmp_path,
            retrieval=arm in RETRIEVAL_ARMS,
            level=ExclusionLevel.BROAD_HOMOLOGY,
        )
        assert not passed, arm
        assert violations[0]["reason_codes"] == ["missing_hitlog"]


def test_an_arm_staged_elsewhere_is_read_from_its_own_directory(tmp_path, monkeypatch):
    """DEEPSHIFT_ARM_DIRS points one arm at a re-run outside the benchmark root: its
    outputs and hit logs are read there, and every other arm stays under the root."""
    rerun = tmp_path / "elsewhere" / "tool_rerun"
    (rerun / "hitlog").mkdir(parents=True)
    (rerun / "hitlog" / "target.log").write_text(
        "candidate.2XYZA\t10\tx\tx\tx\t0.20\tkeep_eligible\n")
    monkeypatch.setenv(ARM_DIRS_VAR, f"tool={rerun}")
    assert arm_directory(tmp_path / "root", "tool") == rerun.resolve()
    assert arm_directory(tmp_path / "root", "other") == tmp_path / "root" / "other"
    passed, violations = verify_self_exclusion(
        "tool", "target", TASK, Authority(), output_root=tmp_path / "root",
        retrieval=True, level=ExclusionLevel.OWN_DEPOSITION)
    assert not passed and violations[0]["reason_codes"] == ["own_deposition"]


def test_an_arm_directory_that_does_not_exist_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(ARM_DIRS_VAR, f"tool={tmp_path / 'absent'}")
    with pytest.raises(SystemExit, match=ARM_DIRS_VAR):
        arm_directory(tmp_path, "tool")
