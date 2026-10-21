"""The model forward and the strict checkpoint loader."""
import numpy as np
import pytest
import torch

from deepshift import pipeline, prediction
from deepshift.atom_cloud import FINAL_ATOM_CLOUD_RECIPE
from deepshift.domain import ATOM_TYPES
from deepshift.model import AtomTransformer, _stable_attention, load_checkpoint
from deepshift.prediction import forward_z

PAD = len(ATOM_TYPES)
RECIPE = FINAL_ATOM_CLOUD_RECIPE.with_vocabulary(PAD, AtomTransformer.N_OFFSET_TYPES).to_dict()


def _stats(columns):
    return {column: {"mean": 0.0, "std": 1.0} for column in columns}


def _cloud():
    at = np.asarray([[68, 1, 0, PAD]], dtype=np.int64)
    off = np.asarray([[6, 6, 6, 17]], dtype=np.int64)
    xyz = np.asarray([[[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0]]], dtype=np.float32)
    return at, off, xyz, at != PAD


def _save(path, model, columns, **extra):
    payload = {"dim": 32, "layers": 1, "shift_cols": columns, "stats": _stats(columns),
               "recipe": RECIPE, "model_state_dict": model.state_dict(), **extra}
    torch.save(payload, path)
    return path


def test_a_loaded_checkpoint_forwards_bit_exactly_like_its_model(tmp_path):
    torch.manual_seed(4)
    columns = [f"s{i}" for i in range(3)]
    model = AtomTransformer(PAD, len(columns), dim=32, layers=1).eval()
    loaded = load_checkpoint(_save(tmp_path / "model.pt", model, columns))
    at, off, xyz, mask = _cloud()
    with torch.no_grad():
        expected = model(torch.from_numpy(at), torch.from_numpy(off),
                         torch.from_numpy(xyz), torch.from_numpy(mask))[0].numpy()
    assert forward_z(loaded, (at, off, xyz, mask)).tobytes() == expected.tobytes()
    assert (loaded.atom_pad_index, loaded.offset_pad_index) == (PAD, 17)
    assert prediction.checkpoint_recipe(loaded) == FINAL_ATOM_CLOUD_RECIPE.with_vocabulary(
        PAD, 17)


def test_a_checkpoint_without_a_complete_recipe_is_refused(tmp_path):
    columns = ["a", "b"]
    model = AtomTransformer(PAD, 2, dim=32, layers=1)
    torch.save({"dim": 32, "layers": 1, "shift_cols": columns, "stats": _stats(columns),
                "model_state_dict": model.state_dict()}, tmp_path / "unstamped.pt")
    with pytest.raises(ValueError, match="recipe"):
        load_checkpoint(tmp_path / "unstamped.pt")
    partial = {key: value for key, value in RECIPE.items() if key != "partner_policy"}
    _save(tmp_path / "partial.pt", model, columns, recipe=partial)
    with pytest.raises(ValueError, match="partner_policy"):
        load_checkpoint(tmp_path / "partial.pt")


def test_a_foreign_atom_vocabulary_is_refused(tmp_path):
    model = AtomTransformer(PAD + 6, 1, dim=32, layers=1)
    _save(tmp_path / "foreign.pt", model, ["a"])
    with pytest.raises(ValueError, match="atom vocabulary"):
        load_checkpoint(tmp_path / "foreign.pt")


def test_checkpoint_without_ppm_statistics_is_refused(tmp_path):
    model = AtomTransformer(PAD, 2, dim=32, layers=1)
    torch.save({"dim": 32, "layers": 1, "shift_cols": ["a", "b"], "stats": {},
                "recipe": RECIPE, "model_state_dict": model.state_dict()},
               tmp_path / "no_stats.pt")
    with pytest.raises(ValueError, match="global ppm transform"):
        load_checkpoint(tmp_path / "no_stats.pt")


def test_stable_attention_avoids_half_dot_product_overflow():
    # The raw fp16 q@k overflows to inf; scores/softmax/value must be reduced in fp32,
    # or a large-cloud batch silently returns NaN predictions under AMP.
    q = torch.full((1, 1, 1, 32), 300.0, dtype=torch.float16)
    k = torch.stack((torch.full((32,), 300.0), torch.full((32,), -300.0)))
    k = k.to(torch.float16).reshape(1, 1, 2, 32)
    v = torch.tensor([[[[1., 2.], [3., 4.]]]], dtype=torch.float16)
    assert not torch.isfinite(q @ k.transpose(-1, -2)).all()
    out = _stable_attention(q, k, v)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.tensor([[[[1., 2.]]]], dtype=torch.float16))


def test_forward_is_deterministic_on_fixed_cpu_tensor():
    torch.manual_seed(7)
    model = AtomTransformer(PAD, 3, dim=32, layers=1).eval()
    at, off, xyz, mask = (torch.from_numpy(a) for a in _cloud())
    with torch.no_grad():
        first = model(at, off, xyz, mask)[0]
        second = model(at, off, xyz, mask)[0]
    assert torch.equal(first, second)


def test_console_entrypoint_returns_zero_for_a_successful_rich_handler(monkeypatch):
    monkeypatch.setattr(prediction, "predict_command", lambda args: {"programmatic": "result"})
    assert pipeline.main([
        "predict", "--pdb", "fixture.pdb", "--checkpoint", "fixture.pt", "--out", "fixture.csv",
    ]) == 0
