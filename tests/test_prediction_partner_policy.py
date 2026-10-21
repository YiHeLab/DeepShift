"""Prediction admits only exact protein chains declared present in the NMR sample."""
import argparse

import pytest

from deepshift import prediction


def _residue(chain):
    return {
        "residue_name": "ALA",
        "auth_chain_id": chain,
        "is_standard": True,
        "atoms": {},
    }


def test_named_sample_partner_selection_is_exact(monkeypatch):
    all_partners = {
        ("B", 1, ""): _residue("B"),
        ("C", 2, ""): _residue("C"),
    }
    monkeypatch.setattr(
        prediction, "partner_protein_residues_for", lambda *args, **kwargs: all_partners
    )
    selected = prediction._named_sample_partner_residues_for(
        "structure.pdb", "A", ["B"], drop_hydrogens=True
    )
    assert set(selected) == {("B", 1, "")}


def test_named_sample_partner_selection_fails_on_wrong_chain(monkeypatch):
    monkeypatch.setattr(
        prediction,
        "partner_protein_residues_for",
        lambda *args, **kwargs: {("B", 1, ""): _residue("B")},
    )
    with pytest.raises(ValueError, match=r"requested=.*C.*available=.*B"):
        prediction._named_sample_partner_residues_for(
            "structure.pdb", "A", ["C"], drop_hydrogens=True
        )


def test_target_chain_cannot_be_named_as_partner():
    with pytest.raises(ValueError, match="cannot also be a sample partner"):
        prediction._named_sample_partner_residues_for(
            "structure.pdb", "A", ["A"], drop_hydrogens=True
        )


def test_predict_cli_requires_exact_partner_chain_names():
    parser = argparse.ArgumentParser()
    prediction.add_predict_arguments(parser)
    args = parser.parse_args([
        "--pdb", "structure.pdb", "--checkpoint", "model.pt", "--out", "out.csv",
        "--sample-partner-chain", "B", "--sample-partner-chain", "D",
    ])
    assert args.sample_partner_chain == ["B", "D"]
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--pdb", "structure.pdb", "--checkpoint", "model.pt", "--out", "out.csv",
            "--sample-contains-partner",
        ])
