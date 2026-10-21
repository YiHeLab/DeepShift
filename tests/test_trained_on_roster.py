"""A competitor is scored without every benchmark protein it trained on, not only its depositions."""
import json
from types import SimpleNamespace

import pytest

from benchmarks import config, score_by_source


def _tasks():
    # s1: its own BMRB is in the corpus. s2: another deposition of a protein in the corpus
    # (caught by UniProt or sequence). s3, s4: two structures of one unrelated label set.
    rows = [("s1", "1", "ls_a"), ("s2", "9", "ls_b"), ("s3", "7", "ls_c"),
            ("s4", "7", "ls_c")]
    return {s: {"structure_id": s, "bmrb_id": b, "label_set_id": ls} for s, b, ls in rows}


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    lib = tmp_path / "refdb"
    lib.mkdir()
    for b in ("1", "2"):
        (lib / f"bmr{b}.cs").write_text("#SEQ: ACDE\n")
    ref = tmp_path / "reference"
    ref.mkdir()
    monkeypatch.setattr(config, "current", lambda: SimpleNamespace(refdb_library=lib))
    monkeypatch.setattr(score_by_source, "REFERENCE", ref)

    def write(label_sets):
        (ref / config.ARM_SAME_PROTEIN_ROSTERS["plmcs"]).write_text(json.dumps(
            {"label_set_ids": label_sets}))
    return write


def test_same_protein_depositions_are_withheld_not_only_the_same_bmrb(corpus):
    corpus(["ls_a", "ls_b"])
    tasks = _tasks()
    kept, dropped = score_by_source.drop_trained_on("plmcs", {s: {} for s in tasks}, tasks)
    assert set(kept) == {"s3", "s4"}
    assert dropped["structure_instances"] == 2 and dropped["label_sets"] == 2
    assert dropped["depositions"] == 2 and dropped["same_bmrb_depositions"] == 1


def test_an_arm_without_a_training_corpus_keeps_everything(corpus):
    tasks = _tasks()
    cells = {s: {} for s in tasks}
    assert score_by_source.drop_trained_on("sparta", cells, tasks) == (cells, None)


def test_a_roster_missing_a_same_bmrb_label_set_is_refused(corpus):
    corpus(["ls_b"])
    tasks = _tasks()
    with pytest.raises(SystemExit, match="missing from its same-protein roster"):
        score_by_source.drop_trained_on("plmcs", {s: {} for s in tasks}, tasks)


def test_the_shipped_roster_holds_every_label_set_it_names_once():
    roster = json.loads((score_by_source.REFERENCE
                         / config.ARM_SAME_PROTEIN_ROSTERS["plmcs"]).read_text())
    ids = roster["label_set_ids"]
    assert len(ids) == len(set(ids)) == roster["n_label_sets"]
    assert set(roster["evidence"]) == set(ids)
    assert all(e["matched_refdb"] for e in roster["evidence"].values())
