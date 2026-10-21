import pytest

from deepshift.domain import (
    exact_polymer_sequence_sha256,
    is_sha256,
    label_residue_id,
    label_set_id,
)

SEQ = exact_polymer_sequence_sha256(["ALA", "GLY", "SER"])

def test_label_residue_id_is_stable_and_well_formed():
    a = label_residue_id(label_set_id("1001", "1", "1", "1", SEQ), 7, "ALA")
    b = label_residue_id(label_set_id("1001", "1", "1", "1", SEQ), 7, "ALA")
    assert a == b
    assert a.startswith("lr_") and is_sha256(a[3:])

def test_shift_list_and_entity_participate_in_key():
    base = label_residue_id(label_set_id("1001", "1", "1", "1", SEQ), 7, "ALA")
    assert base != label_residue_id(label_set_id("1001", "2", "1", "1", SEQ), 7, "ALA")
    assert base != label_residue_id(label_set_id("1001", "1", "1", "2", SEQ), 7, "ALA")

def test_sample_condition_participates_in_label_set_key():
    first = label_set_id("1001", "1", "1", "1", SEQ, "7")
    second = label_set_id("1001", "1", "1", "1", SEQ, "8")
    assert first != second

def test_label_set_id_fails_closed_on_non_sha256_sequence():
    with pytest.raises(ValueError):
        label_set_id("1001", "1", "1", "1", "not-a-sha256")
