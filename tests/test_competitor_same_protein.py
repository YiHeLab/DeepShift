"""No competitor reported with same-protein blocking can retrieve its target's own protein.

``benchmarks/reference/competitor_same_protein.json`` (fold 6) and
``ucb200_competitor_same_protein.json`` (UCBShift-200) name, per structure, every reference
of each transfer library that is the target's own protein by the sequence-inclusive rule,
and the exclusion list that blocks them. The first tests check the frozen files; the last
reads each retrieval arm's hit logs under ``DEEPSHIFT_BENCH_OUT`` and fails on any
same-protein reference a blocked arm kept as a candidate (UCBShift ``keep_eligible``) or
transferred from (SHIFTX2 ``HITLOG``).
"""
import csv
import json
import re

import pytest

from benchmarks import build_competitor_exclusions as bce
from benchmarks import config
from benchmarks.score_by_source import REFERENCE
from deepshift.self_exclusion import RETRIEVAL_ARMS as DECLARED_RETRIEVERS
from deepshift.self_exclusion import arm_directory

FOLD6 = json.loads((REFERENCE / bce.OUT_NAME).read_text())
UCB200 = json.loads((REFERENCE / "ucb200_competitor_same_protein.json").read_text())
RETRIEVAL_ARMS = [arm for arm in config.BENCHMARK_ARMS
                  if arm.startswith(("ucbshift", "shiftx2")) and not arm.endswith("_leaky")
                  and arm not in ("ucbshift_x", "shiftx2_noshifty")]
_BMR = re.compile(r"bmr(\d+)\.cs")


def library(arm):
    return "ucbshift" if arm.startswith("ucbshift") else "shifty"


def test_the_sequence_rule_reads_coverage_off_the_shorter_sequence():
    # BMRB 27945 is UCBShift reference 4186 plus an N-terminal GHM tag: 137 of 140 query
    # residues align, all of the 137-residue reference does.
    assert bce.passes(1.0, 4, 140, 140, 1, 137, 137)
    assert bce.passes(0.95, 1, 90, 100, 1, 90, 90)
    assert not bce.passes(0.949, 1, 100, 100, 1, 100, 100)
    assert not bce.passes(1.0, 1, 89, 100, 1, 89, 100)


@pytest.mark.parametrize("frozen", [FOLD6, UCB200], ids=["fold6", "ucb200"])
def test_every_listed_same_protein_reference_is_excluded(frozen):
    for lib, lists in frozen["exclude_bmrbs"].items():
        for sid, blocked in lists.items():
            missing = set(frozen["evidence"][lib].get(sid, {})) - set(blocked)
            assert not missing, f"{lib} {sid}: same-protein {sorted(missing)} not excluded"
    for lib, by_sid in frozen["evidence"].items():
        assert all(refs and all(refs.values()) for refs in by_sid.values()), lib


def test_the_audited_leaks_are_blocked():
    # 27945/27946 (the AlphaFold arm kept 4186), 52275-52277 (TrEMBL A0A8D2EN79 against
    # 4775's SwissProt Q13257), 36259 and 26340: references the UniProt-keyed list missed.
    for sid, bmrb in (("b27945_af_l1_a1_e1", "4186"), ("b27946_af_l1_a1_e1", "4186"),
                      ("b52275_af_l1_a1_e1", "4775"), ("b36259_af_l1_a1_e1", "6801"),
                      ("b26340_af_l1_a1_e1", "10288")):
        for lib in bce.LIBRARIES:
            if bmrb in FOLD6["evidence"][lib].get(sid, {}):
                assert bmrb in FOLD6["exclude_bmrbs"][lib][sid], (lib, sid, bmrb)
    assert "uniref100" in FOLD6["evidence"]["ucbshift"]["b52275_af_l1_a1_e1"]["4775"]


@pytest.fixture
def bench():
    out_root = config.current().out_root
    if not (out_root / "fold6_tasks.tsv").is_file():
        pytest.skip("competitor outputs are not on disk (DEEPSHIFT_BENCH_OUT)")
    return out_root


def test_task_lists_cover_every_structure_the_frozen_file_does_not_list(bench):
    with open(bench / "fold6_tasks.tsv") as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}
    for lib, by_sid in FOLD6["evidence"].items():
        for sid, refs in by_sid.items():
            if sid in FOLD6["exclude_bmrbs"][lib]:
                continue
            listed = {b for b in tasks[sid]["exclude_bmrbs"].split(",") if b}
            assert set(refs) <= listed | {tasks[sid]["bmrb_id"]}, (lib, sid)


@pytest.mark.parametrize("arm", RETRIEVAL_ARMS)
def test_no_same_protein_reference_is_retrievable(bench, arm):
    lib, hits = library(arm), []
    for sid, refs in FOLD6["evidence"][lib].items():
        log = arm_directory(bench, arm) / "hitlog" / f"{sid}.log"
        if not log.is_file():
            continue
        for line in log.read_text(errors="replace").splitlines():
            field = line.split("\t")
            if lib == "ucbshift" and len(field) == 7 and field[6] == "keep_eligible":
                bmrb = field[0].split(".")[0]
            elif lib == "shifty" and field[0] == "HITLOG" and len(field) >= 3:
                match = _BMR.match(field[2])
                bmrb = match.group(1) if match else None
            else:
                continue
            if bmrb in refs:
                hits.append(f"{sid} {bmrb} ({'+'.join(refs[bmrb])})")
    assert not hits, (f"{arm} can retrieve the target's own protein on "
                      f"{len({h.split()[0] for h in hits})} structures: "
                      + "; ".join(sorted(set(hits))[:8]))


@pytest.mark.parametrize("arm", config.BENCHMARK_ARMS)
def test_an_arm_that_left_hit_logs_is_declared_a_retrieval_arm(bench, arm):
    """An arm staged with hit logs retrieves, so a structure it logged nothing for must
    fail closed rather than be scored."""
    if (arm_directory(bench, arm) / "hitlog").is_dir():
        assert arm in DECLARED_RETRIEVERS
