"""Cache-free prediction: the atom cap and what the output says about it."""
import numpy as np
import pytest
import torch

from deepshift import prediction
from deepshift.atom_cloud import FINAL_ATOM_CLOUD_RECIPE, build_atom_cloud
from deepshift.domain import ATOM_TYPES
from deepshift.model import AtomTransformer
from deepshift.normalization import build_fixed_per_aa

COLUMNS = ("ca_shift", "cb_shift")
RECIPE = FINAL_ATOM_CLOUD_RECIPE.with_vocabulary(len(ATOM_TYPES), AtomTransformer.N_OFFSET_TYPES)


def _crowded_residues(n_residues, radius=3.0, seed=0):
    """Alanines piled inside one small ball, so every residue's 8 A shell holds them all."""
    rng = np.random.default_rng(seed)
    residues = {}
    for rid in range(1, n_residues + 1):
        center = rng.normal(size=3)
        center *= radius * rng.random() ** (1 / 3) / np.linalg.norm(center)
        residues[rid] = {
            "residue_name": "ALA", "is_standard": True, "auth_chain_id": "A",
            "atoms": {name: center + rng.normal(scale=0.3, size=3)
                      for name in ("N", "CA", "C", "O", "CB")},
        }
    return residues


def _checkpoint(path):
    torch.manual_seed(0)
    model = AtomTransformer(len(ATOM_TYPES), len(COLUMNS), dim=16, layers=1)
    raw = {column: {"mean": 0.0, "std": 1.0} for column in COLUMNS}
    raw["snuc"] = dict.fromkeys(COLUMNS, 1.0)
    torch.save({"dim": 16, "layers": 1, "shift_cols": list(COLUMNS),
                "stats": build_fixed_per_aa(raw, COLUMNS), "recipe": RECIPE.to_dict(),
                "model_state_dict": model.state_dict()}, path)
    return path


def test_a_cloud_over_the_cap_keeps_the_nearest_atoms():
    residues = _crowded_residues(120)
    built = build_atom_cloud(residues, recipe=RECIPE)
    assert built.precap_atom_count.max() == 600
    assert built.atom_mask.sum(axis=1).tolist() == [512] * 120
    kept = np.linalg.norm(built.atom_xyz[0][built.atom_mask[0]], axis=1)
    every = np.sort(np.linalg.norm(np.concatenate(
        [np.stack(list(r["atoms"].values())) for r in residues.values()]
    ) - residues[1]["atoms"]["CA"], axis=1))
    np.testing.assert_allclose(np.sort(kept), every[:512], atol=1e-5)


def test_prediction_warns_and_marks_truncated_residues(tmp_path, monkeypatch):
    residues = _crowded_residues(120)
    monkeypatch.setattr(prediction, "parse_structure", lambda *a, **k: residues)
    monkeypatch.setattr(prediction, "retained_insertion_partner_residues", lambda *a, **k: {})
    checkpoint = _checkpoint(tmp_path / "member.pt")
    with pytest.warns(UserWarning, match="120 of 120 residues have more than 512"):
        frame = prediction.predict_pdb("crowded.pdb", "A", [checkpoint], batch_size=64)
    assert frame["atom_cloud_truncated"].all()
    assert list(frame.columns[:3]) == ["residue_id", "residue_name", "atom_cloud_truncated"]

    sparse = dict(list(residues.items())[:3])
    monkeypatch.setattr(prediction, "parse_structure", lambda *a, **k: sparse)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = prediction.predict_pdb("sparse.pdb", "A", [checkpoint], batch_size=64)
    assert not frame["atom_cloud_truncated"].any()
