"""Sample assembly: was a protein partner in the NMR tube?"""
import json

import pytest

from deepshift.data import assembly, structures


class FakeSaveframe:
    def __init__(self, tags): self.tags = tags
    def get_tag(self, name):
        if name not in self.tags:
            raise KeyError(name)
        return [self.tags[name]]


class FakeLoop:
    def __init__(self, ids): self.ids = ids
    def get_tag(self, names): return [[i] for i in self.ids]


class FakeEntry:
    """Duck-types the two pynmrstar accessors entry_assembly uses."""
    def __init__(self, entities, assembly_ids):
        self.entities = [FakeSaveframe(e) for e in entities]
        self.loops = [FakeLoop(assembly_ids)] if assembly_ids is not None else []
    def get_saveframes_by_category(self, cat): return self.entities
    def get_loops_by_category(self, cat): return self.loops


PROTEIN = {"ID": "1", "Type": "polymer", "Polymer_type": "polypeptide(L)"}
PROTEIN2 = {"ID": "2", "Type": "polymer", "Polymer_type": "polypeptide(L)"}
DNA = {"ID": "3", "Type": "polymer", "Polymer_type": "polydeoxyribonucleotide"}


@pytest.mark.parametrize("polymer_type,entity_type,expected", [
    ("polypeptide(L)", "polymer", True),
    ("polyribonucleotide", "polymer", False),
    ("polydeoxyribonucleotide", "polymer", False),
    (None, "polymer", True),          # undeclared polymer: assume protein
    (None, "non-polymer", False),
])
def test_is_protein(polymer_type, entity_type, expected):
    assert assembly.is_protein(polymer_type, entity_type) is expected


def test_one_protein_copy_is_monomer():
    e = FakeEntry([PROTEIN], ["1"])
    assert assembly.entry_assembly(e)["class"] == "MONOMER"


def test_a_nucleic_acid_partner_does_not_make_a_complex():
    # The cloud folds in protein partners only, so DNA in the tube is irrelevant to it.
    e = FakeEntry([PROTEIN, DNA], ["1", "3"])
    r = assembly.entry_assembly(e)
    assert r["class"] == "MONOMER" and r["protein_copies"] == 1


def test_two_copies_of_one_entity_is_homo():
    e = FakeEntry([PROTEIN], ["1", "1"])
    r = assembly.entry_assembly(e)
    assert r["class"] == "HOMO" and r["protein_entities"] == 1


def test_two_entities_is_hetero():
    e = FakeEntry([PROTEIN, PROTEIN2], ["1", "2"])
    assert assembly.entry_assembly(e)["class"] == "HETERO"


def test_missing_assembly_loop_is_not_a_class():
    assert assembly.entry_assembly(FakeEntry([PROTEIN], None))["class"] == "no_assembly_loop"


PARTNER = {"entity_id": "2", "copies": 1, "alignment_sequence": "GGGGGGGGGG"}


@pytest.mark.parametrize("record,expected", [
    ({"target_present_in_assembly": True, "partners": [PARTNER]}, True),
    ({"target_present_in_assembly": True, "partners": []}, False),
    ({"target_present_in_assembly": False, "partners": []}, False),
    ({"class": "missing", "target_present_in_assembly": False, "partners": []}, False),
])
def test_partner_in_sample(record, expected):
    assert assembly.partner_in_sample({"ls_7": record}, "ls_7") is expected


def test_unclassified_fails_to_no_partner():
    """An entry that cannot be classified must not manufacture a partner."""
    assert assembly.partner_in_sample({}, "ls_unknown") is False


def test_load_reads_the_label_set_records_only(tmp_path):
    path = tmp_path / "sample_assembly.json"
    path.write_text(json.dumps({"label_sets": {"ls_1": {"partners": []}}, "records": {}}))
    assert assembly.load(path) == {"ls_1": {"partners": []}}
    path.write_text(json.dumps({"records": {"7": {"class": "HETERO"}}}))
    with pytest.raises(ValueError, match="label_sets"):
        assembly.load(path)


def test_label_set_record_removes_the_target_copy_and_keeps_declared_partners():
    entry = {
        "class": "HETERO",
        "protein_members": [
            {"entity_id": "1", "copies": 2, "alignment_sequence": "AAAAAAAAAA"},
            {"entity_id": "2", "copies": 1, "alignment_sequence": "GGGGGGGGGG"},
        ],
    }
    record = assembly.label_set_assembly(
        entry, bmrb_id="7", label_set_id="ls_x", target_entity_id="1",
        target_sequence="AAAAAAAAAA",
    )
    assert record["target_present_in_assembly"] is True
    assert record["partners"] == [
        {"entity_id": "1", "copies": 1, "alignment_sequence": "AAAAAAAAAA"},
        {"entity_id": "2", "copies": 1, "alignment_sequence": "GGGGGGGGGG"},
    ]


def test_coordinate_chains_must_match_a_declared_sequence_and_copy_slot(monkeypatch):
    record = {
        "target_present_in_assembly": True,
        "partners": [
            {"entity_id": "2", "copies": 1, "alignment_sequence": "GGGGGGGGGG"},
        ],
    }
    model = object()
    monkeypatch.setattr(structures, "select_best_nmr_model_index", lambda *_: 0)
    monkeypatch.setattr(structures, "parse_models", lambda *_args, **_kwargs: [model])
    monkeypatch.setattr(
        structures, "chain_sequences_with_insertion_codes",
        lambda _model: {
            "A": ("AAAAAAAAAA", list(range(10))),
            "B": ("GGGGGGGGGG", list(range(10))),
            "C": ("GGGGGGGGGG", list(range(10))),
            "D": ("VVVVVVVVVV", list(range(10))),
        },
    )
    monkeypatch.setattr(
        structures, "partner_protein_residues",
        lambda *_args, **_kwargs: {
            (chain, 1, ""): {"chain": chain} for chain in ("B", "C", "D")
        },
    )
    matched = structures.sample_matched_partner_residues_for(
        "complex.pdb", "A", record, drop_hydrogens=True
    )
    assert matched == {("B", 1, ""): {"chain": "B"}}


def test_sample_partner_matching_maximizes_declared_slots(monkeypatch):
    record = {
        "target_present_in_assembly": True,
        "partners": [
            {"entity_id": "2", "copies": 1, "alignment_sequence": "A" * 20},
            {"entity_id": "3", "copies": 1, "alignment_sequence": "A" * 19 + "G"},
        ],
    }
    model = object()
    monkeypatch.setattr(structures, "select_best_nmr_model_index", lambda *_: 0)
    monkeypatch.setattr(structures, "parse_models", lambda *_args, **_kwargs: [model])
    monkeypatch.setattr(
        structures,
        "chain_sequences_with_insertion_codes",
        lambda _model: {
            "A": ("G" * 20, list(range(20))),
            # B fits both slots and prefers slot 2; C fits only slot 2. A greedy match
            # would consume slot 2 with B and incorrectly discard C.
            "B": ("A" * 20, list(range(20))),
            "C": ("A" * 19 + "T", list(range(20))),
        },
    )
    monkeypatch.setattr(
        structures,
        "partner_protein_residues",
        lambda *_args, **_kwargs: {
            (chain, 1, ""): {"chain": chain} for chain in ("B", "C")
        },
    )
    matched = structures.sample_matched_partner_residues_for(
        "complex.pdb", "A", record, drop_hydrogens=True
    )
    assert set(matched) == {("B", 1, ""), ("C", 1, "")}
