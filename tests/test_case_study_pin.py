"""The case-study hold-out: pinned to fold 0 after the draw, never through it."""
import json

import pytest

from deepshift.data import folds


def test_the_pin_moves_only_the_seed_and_records_where_each_came_from():
    assignment = {"a": 1, "b": 2, "c": 3, "d": 6, "e": 0}
    moved = folds.pin_after_draw(assignment, {"a", "c", "e", "not_in_universe"})
    assert moved == {"a": 1, "c": 3}
    assert assignment == {"a": 0, "b": 2, "c": 0, "d": 6, "e": 0}


def test_a_seed_member_drawn_into_fold_6_is_refused_rather_than_moved():
    assignment = {"a": 1, "d": 6}
    with pytest.raises(RuntimeError, match="fold 6"):
        folds.pin_after_draw(assignment, {"a", "d"})
    assert assignment == {"a": 1, "d": 6}


def test_the_leakage_gate_passes_once_the_families_exclude_the_pinned_seed():
    assignment = {"a": 1, "b": 1, "c": 2}
    folds.pin_after_draw(assignment, {"a"})
    # a's relatives stay in fold 1; the gate over families without a sees one fold each.
    assert folds.verify_no_labelset_leakage(assignment, {"uniprot": [["b"], ["c"]]}) == {"uniprot": 0}
    # left in the families, the pinned label set would span fold 0 and a training fold.
    assert folds.verify_no_labelset_leakage(assignment, {"uniprot": [["a", "b"]]}) == {"uniprot": 1}


def test_the_packaged_authority_loads_and_a_bad_one_is_refused(tmp_path):
    packaged = folds.os.path.join(folds._ROOT, folds.CASE_STUDY_EXCLUDED_BASENAME)
    bmrbs, sequences = folds.load_case_study_exclusion(packaged)
    assert {"15397", "19257", "19253", "16363", "17760", "16657", "25500"} == bmrbs
    assert sequences and all(set(seq) <= set("ACDEFGHIKLMNPQRSTVWYX") for seq in sequences.values())
    bad = tmp_path / "case.json"
    bad.write_text(json.dumps({"bmrbs": ["1", "1"], "sequences": {"x": "ACDE"}}))
    with pytest.raises(ValueError, match="unique"):
        folds.load_case_study_exclusion(str(bad))
    bad.write_text(json.dumps({"bmrbs": ["1"], "sequences": {"x": "ACDEZ"}}))
    with pytest.raises(ValueError, match="invalid"):
        folds.load_case_study_exclusion(str(bad))
