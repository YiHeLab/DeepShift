"""Build-stage contracts: the entity grain, the atom cloud, and what is excluded.

Every test here is a case that produces a wrong label when the contract slips: two
entities colliding at the same residue number, a modified residue tokenized as standard,
a metal-perturbed residue trained on, or the build and inference clouds drifting apart.
"""
import csv
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from deepshift import atom_cloud, prediction, structure
from deepshift.data import (
    atoms,
    build,
    crystal,
    datasets,
    discover,
    exclusion,
    fetch,
    reref,
    structures,
)


def _atom_line(record, serial, atom, residue, chain, rid, x, y, z, element):
    return (f"{record:<6}{serial:>5} {atom:^4} {residue:>3} {chain}{rid:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{20.0:>6.2f}          "
            f"{element:>2}\n")


def test_unterminated_last_nmr_model_is_not_silently_dropped(tmp_path):
    pdb = tmp_path / "unterminated-last-model.pdb"
    pdb.write_text(
        "MODEL        1\n"
        + _atom_line("ATOM", 1, "CA", "ALA", "A", 1, 0, 0, 0, "C")
        + "ENDMDL\n"
        + "MODEL        2\n"
        + _atom_line("ATOM", 2, "CA", "ALA", "A", 1, 1, 0, 0, "C")
    )
    models = structure.parse_models(pdb, chain="A")
    assert len(models) == 2
    assert models[1][("A", 1, "")]["atoms"]["CA"][0] == pytest.approx(1.0)


def test_entity_pivot_keeps_two_entities_at_one_residue_number():
    """Two entities that both number their first residue 1 must stay two candidates.
    A BMRB-keyed pivot would merge them into a single row."""
    long_rows = [
        {"bmrb_id": "7", "shift_list_id": "1", "entity_assembly_id": entity,
         "entity_id": entity, "sample_condition_id": "1", "comp_index_id": 1,
         "comp_id": residue, "atom_id": "CA", "value": value, "ambiguity_code": 1}
        for entity, residue, value in (("1", "ALA", 52.1), ("2", "SER", 58.3))
    ]
    sequences = {"1": {"scope": "full", "tokens": ["ALA"]},
                 "2": {"scope": "full", "tokens": ["SER"]}}
    candidates, quarantine = fetch.build_candidates_from_long(long_rows, sequences)
    assert not quarantine
    assert {c["entity_id"] for c in candidates} == {"1", "2"}
    assert sorted(c["residues"][1]["cells"]["ca"] for c in candidates) == [52.1, 58.3]
    assert all(c["n_conflict_cells"] == 0 for c in candidates)


def test_entity_polymer_comp_id_in_first_column_is_not_dropped():
    class Loop:
        data: ClassVar[list[list[str]]] = [["ALA", "1", "1"], ["GLY", "1", "2"]]

        @staticmethod
        def get_tag_names():
            return [
                "_Entity_poly_seq.Comp_ID",
                "_Entity_poly_seq.Entity_ID",
                "_Entity_poly_seq.Num",
            ]

    class Entry:
        @staticmethod
        def get_loops_by_category(category):
            return [Loop()] if category == "entity_poly_seq" else []

    assert fetch.parse_entity_poly_seq(Entry()) == {
        1: {
            "tokens": ["ALA", "GLY"],
            "nums": [1, 2],
            "scope": "full",
        }
    }


def test_sample_conditions_cannot_mint_duplicate_label_set_ids():
    common = {
        "bmrb_id": "7", "shift_list_id": "1", "entity_assembly_id": "1",
        "entity_id": "1", "comp_index_id": 1, "comp_id": "ALA",
        "atom_id": "CA", "value": 52.1, "ambiguity_code": 1,
    }
    rows = [
        {**common, "sample_condition_id": condition}
        for condition in ("1", "2")
    ]
    candidates, quarantine = fetch.build_candidates_from_long(
        rows, {"1": {"scope": "full", "tokens": ["ALA"]}}
    )
    assert not quarantine and len(candidates) == 2
    _, selection = fetch.select_representatives(candidates)
    assert len({row["label_set_id"] for row in selection}) == 2


def test_preallocated_empty_candidate_slots_do_not_prove_a_query():
    entity = {"ls_one": {"bmrb_id": "7"}}
    selected = {"af": set(), "nmr": set(), "xtal": set()}
    attempts = {"af": set(), "nmr": set(), "xtal": set()}
    accounting = discover.coverage_accounting(entity, selected, attempts)
    assert sorted(discover.coverage_gate_failures(accounting)) == [
        ("7", "af"), ("7", "nmr"), ("7", "xtal")
    ]


def test_materialization_cannot_succeed_with_zero_mapped_labels():
    selection = [{"label_set_id": "ls_test", "uniprot": "PTEST", "chain": "A"}]
    entity = {
        "ls_test": {
            "identity": {},
            "sequence": "AAAAAAAAAA",
            "seq_positions": list(range(1, 11)),
            "seq_tokens": ["ALA"] * 10,
            "residues": [{
                "comp_index_id": 99,
                "comp_id": "ALA",
                "label_residue_id": "lr_test",
                "cells": {"ca_shift": 52.1},
            }],
        }
    }

    def loader(source, accession, chain, entity_seq=None):
        return {
            "struct_res_ids": list(range(1, 11)),
            "struct_seq": "AAAAAAAAAA",
            "coord_sha256": "a" * 64,
            "model_index": 0,
            "preprocessing_state": "fixture",
            "chain": "A",
        }

    rows, log = datasets.materialize_source(
        "af", selection, entity, loader, build_id="fixture"
    )
    assert rows == []
    assert log[-1]["status"] == "excluded"
    assert log[-1]["reason"] == "zero_mapped_label_residues"
    assert log[-1]["n_mapped"] == 0


def test_materialization_requires_the_ranked_candidate_ledger(tmp_path):
    with pytest.raises(FileNotFoundError, match=re.escape("candidate_ledger.csv")):
        datasets.materialize_masters(
            tmp_path / "out",
            tmp_path / "missing-shifts.csv",
            tmp_path / "selection",
            lambda *args, **kwargs: None,
            build_id="fixture",
        )


def test_ranked_candidate_ids_must_be_unique(tmp_path):
    ledger = tmp_path / "candidate_ledger.csv"
    columns = [
        "label_set_id", "source", "candidate_id", "candidate_rank",
        "accession", "chain", "route",
    ]
    with ledger.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for label_set in ("ls_one", "ls_two"):
            writer.writerow({
                "label_set_id": label_set,
                "source": "af",
                "candidate_id": "duplicate",
                "candidate_rank": "1",
                "accession": "PTEST",
                "chain": "A",
                "route": "fixture",
            })
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        datasets.load_ranked_candidates(ledger, "af")


def test_shiftx_parser_rejects_malformed_numeric_output(tmp_path):
    output = tmp_path / "shiftx.out"
    output.write_text("1 A 1.0 2.0 broken 4.0 5.0 6.0\n")
    with pytest.raises(ValueError, match="malformed SHIFTX value"):
        reref.parse_shiftx_output(output)


def test_a_shift_row_contradicting_the_polymer_token_is_dropped():
    """The polymer sequence is authoritative: a shift row claiming a residue the entity
    does not declare at that position must never become a label."""
    long_rows = [{"bmrb_id": "7", "shift_list_id": "1", "entity_assembly_id": "1",
                  "entity_id": "1", "sample_condition_id": "1", "comp_index_id": 1,
                  "comp_id": "TRP", "atom_id": "CA", "value": 57.0,
                  "ambiguity_code": 1}]
    candidates, _ = fetch.build_candidates_from_long(
        long_rows, {"1": {"scope": "full", "tokens": ["ALA"]}})
    assert candidates[0]["residues"] == {}
    assert candidates[0]["n_residue_code_mismatch"] == 1


def test_nonstandard_contact_shell_is_flagged_but_never_tokenized():
    standard = {"residue_name": "ALA", "atoms": {
        "N": np.array([-1.5, 0, 0]), "CA": np.array([0, 0, 0]),
        "C": np.array([1.5, 0, 0]), "O": np.array([1.5, 1.2, 0])}}
    modified = {"residue_name": "SEP", "is_standard": False, "atoms": {
        "CA": np.array([4, 0, 0]), "P": np.array([4, -2, 0])}}
    built = atom_cloud.build_atom_cloud({10: standard, 11: modified})
    assert built.residue_ids == [10]
    assert built.nonstandard_contact.tolist() == [True]
    assert int(built.atom_mask.sum()) == 4
    # A flagged shell drops the sample row; it never introduces a new atom token.
    assert atoms.loader_atom_count(
        int(built.atom_mask.sum()), bool(built.nonstandard_contact[0])
    ) == 0
    assert exclusion.infer_element("P", "", "SEP") == "P"
    assert np.all(built.atom_type[0, :built.atom_mask.sum()] < len(atom_cloud.ATOM_TO_IDX))


def test_build_and_inference_atom_clouds_are_byte_identical():
    """One implementation, shared by the cache build and cache-free inference, so a
    prediction cannot be made on a differently-built cloud than the model trained on."""
    residues = {
        1: {"residue_name": "ALA", "is_standard": True, "atoms": {
            "N": np.array([0.0, 0.0, 0.0]), "CA": np.array([1.0, 0.0, 0.0]),
            "C": np.array([2.0, 0.0, 0.0]), "CB": np.array([1.0, 1.0, 0.0])}},
        9: {"residue_name": "GLY", "is_standard": True, "atoms": {
            "N": np.array([1.0, 4.0, 0.0]), "CA": np.array([1.0, 5.0, 0.0]),
            "C": np.array([1.0, 6.0, 0.0])}},
    }
    assert prediction.build_atom_cloud is atom_cloud.build_atom_cloud
    recipe = replace(atom_cloud.FINAL_ATOM_CLOUD_RECIPE, m_atoms=16)
    built = atom_cloud.build_atom_cloud(residues, recipe=recipe)
    inferred = prediction.build_atom_cloud(residues, recipe=recipe)
    for build_value, inference_value in zip(built.model_arrays, inferred.model_arrays):
        assert build_value.tobytes() == inference_value.tobytes()


def test_atom_recipe_excludes_hydrogen_and_deuterium_names():
    residues = {1: {"residue_name": "ALA", "atoms": {
        "CA": np.zeros(3), "C": np.ones(3), "H": np.array([0.0, 1.0, 0.0]),
        "1HD": np.array([0.0, 0.0, 1.0]), "D": np.array([1.0, 0.0, 0.0]),
    }}}
    built = atom_cloud.build_atom_cloud(residues)
    assert int(built.atom_mask.sum()) == 2
    assert structure.is_hydrogen("H")
    assert structure.is_hydrogen("1HD")
    assert structure.is_hydrogen("D")


def test_hetatm_cutoffs_and_mse_mapping(tmp_path):
    pdb = tmp_path / "metal.pdb"
    lines = []
    serial = 1
    for rid, center in ((1, 0.0), (2, 30.0)):
        for atom, dx, element in (("N", -1.5, "N"), ("CA", 0, "C"),
                                  ("C", 1.5, "C"), ("O", 2.3, "O")):
            lines.append(_atom_line("ATOM", serial, atom, "ALA", "A", rid,
                                    center + dx, 0, 0, element))
            serial += 1
    lines.append(_atom_line("HETATM", serial, "FE", "FE", "X", 1, 3.0, 0, 0, "FE"))
    lines.append("END\n")
    pdb.write_text("".join(lines))
    model = structures.select_best_nmr_model(str(pdb), "A")
    aa = structures.model_to_aa(model)
    # Only the residue within the iron cutoff is excluded; the distant one survives.
    assert exclusion.perturber_excluded_rids(str(pdb), "A", aa) == {1}

    mse = {("A", 4): {"residue_name": "MSE", "atoms": {
        "CA": np.zeros(3), "SE": np.ones(3)}}}
    mapped = structures.model_to_aa(mse, keep_nonstandard=True)
    assert mapped[4]["residue_name"] == "MET"
    assert "SD" in mapped[4]["atoms"] and mapped[4].get("is_standard", True)

    # The cutoffs in force are read from the packaged rule file, and no category the file
    # keeps usable is excluded.
    rule = json.loads(Path(exclusion.__file__).with_name(
        "hetatm_exclusion_rule.json").read_text())
    assert {name: item["cutoff_A"] for name, item in rule["exclude"].items()} == \
        exclusion.EXCLUDE_CUTOFFS
    assert not set(exclusion.EXCLUDE_CUTOFFS) & set(rule["benign_keep"])
    assert all(exclusion.categorize_entity(exclusion.Entity("X", 1, "", name, [
        {"name": name, "element": name, "xyz": np.zeros(3)}])) in exclusion.EXCLUDE_CUTOFFS
        for name in ("FE", "ZN", "MG"))


def test_the_structure_archive_is_linked_into_the_build(tmp_path):
    """Discovery and materialization read the build's own coordinate directories, so
    every file of the frozen archive is linked there; a second pass changes nothing, a
    link that drifted is re-pointed, and a regular file in the way is refused."""
    archive = tmp_path / "archive"
    for name in build.STRUCTURE_ARCHIVE_DIRS:
        (archive / name).mkdir(parents=True)
    (archive / "pdbs" / "1ABC.pdb").write_text("nmr\n")
    (archive / "crystal_pdbs" / "2def.pdb").write_text("xtal\n")
    (archive / "alphafold" / "AF-P01308-F1-model_v6.pdb").write_text("af\n")
    data = tmp_path / "data"
    (data / "alphafold").mkdir(parents=True)
    (data / "alphafold" / "bmrb_uniprot_mapping.json").write_text("{}")

    counts = build.link_structure_archive(archive, data)
    assert counts == {"pdbs": 1, "crystal_pdbs": 1, "alphafold": 1}
    assert (data / "pdbs" / "1ABC.pdb").read_text() == "nmr\n"
    assert (data / "alphafold" / "bmrb_uniprot_mapping.json").read_text() == "{}"
    assert build.link_structure_archive(archive, data) == counts

    elsewhere = tmp_path / "elsewhere.pdb"
    elsewhere.write_text("stale\n")
    (data / "pdbs" / "1ABC.pdb").unlink()
    (data / "pdbs" / "1ABC.pdb").symlink_to(elsewhere)
    build.link_structure_archive(archive, data)
    assert (data / "pdbs" / "1ABC.pdb").resolve() == (archive / "pdbs" / "1ABC.pdb").resolve()

    (data / "crystal_pdbs" / "2def.pdb").unlink()
    (data / "crystal_pdbs" / "2def.pdb").write_text("not a link\n")
    with pytest.raises(RuntimeError, match="not a link"):
        build.link_structure_archive(archive, data)
    partial = tmp_path / "archive_without_af"
    (partial / "pdbs").mkdir(parents=True)
    (partial / "crystal_pdbs").mkdir()
    with pytest.raises(RuntimeError, match="no alphafold/"):
        build.link_structure_archive(partial, tmp_path / "data2")


def test_every_deposition_is_looked_up_in_the_frozen_uniprot_snapshot(tmp_path):
    """A deposition the snapshot holds no candidate for is still a key of the discovery
    map, with an empty answer, so discovery counts it as queried and records the
    drop reason instead of failing its coverage gate."""
    ledger = tmp_path / "uniprot_mapping_candidates.csv"
    with open(ledger, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["bmrb_id", "accession", "mapping_source"])
        writer.writeheader()
        writer.writerow({"bmrb_id": "1004", "accession": "P01308", "mapping_source": "baseline"})
        writer.writerow({"bmrb_id": "1004", "accession": "A0A000", "mapping_source": "blast"})
        writer.writerow({"bmrb_id": "2053", "accession": "P00001", "mapping_source": "baseline"})
    af_map = build.uniprot_map_for_discovery(ledger, ["1004", "2053", "25409"])
    assert af_map == {"1004": ["P01308", "A0A000"], "2053": ["P00001"], "25409": []}

    entity_map = {"ls_a": {"bmrb_id": "1004"}, "ls_b": {"bmrb_id": "25409"}}
    queried = {lsid for lsid, rec in entity_map.items() if str(rec["bmrb_id"]) in af_map}
    assert queried == {"ls_a", "ls_b"}
    accounting = discover.coverage_accounting(
        entity_map, {"xtal": set(), "nmr": set(), "af": set()},
        {"xtal": queried, "nmr": queried, "af": queried})
    unmapped = next(r for r in accounting if r["bmrb_id"] == "25409" and r["source"] == "af")
    assert unmapped["n_queried"] == 1
    assert "no_uniprot_mapping" in unmapped["drop_reasons"]
    assert discover.coverage_gate_failures(accounting) == []


def test_crystal_path_convention_is_case_insensitive():
    """The crystal module contributes one thing to the build, the file-name convention:
    a PDB id in any case resolves to the lowercase file beside the root it is given."""
    assert crystal.crystal_pdb_path("/x", "1ABC") == "/x/1abc.pdb"
