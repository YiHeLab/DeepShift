"""Ensemble prediction: members average in ppm, and an interval calibration applies to
exactly the number of members it was fitted on."""
import argparse

import numpy as np
import pytest
import torch

from deepshift import calibration, prediction
from deepshift.atom_cloud import FINAL_ATOM_CLOUD_RECIPE
from deepshift.domain import ATOM_TYPES
from deepshift.model import AtomTransformer
from deepshift.normalization import build_fixed_per_aa

COLUMNS = ("ca_shift", "cb_shift", "n_shift", "h_shift", "c_shift", "ha_shift")
RECIPE = FINAL_ATOM_CLOUD_RECIPE.with_vocabulary(len(ATOM_TYPES), AtomTransformer.N_OFFSET_TYPES)


def _residues(n=3, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for rid in range(1, n + 1):
        center = rng.normal(size=3) * 2.0
        out[rid] = {"residue_name": "ALA", "is_standard": True, "auth_chain_id": "A",
                    "atoms": {name: center + rng.normal(scale=0.3, size=3)
                              for name in ("N", "CA", "C", "O", "CB")}}
    return out


def _member(path, seed):
    torch.manual_seed(seed)
    model = AtomTransformer(len(ATOM_TYPES), len(COLUMNS), dim=16, layers=1)
    raw = {column: {"mean": 0.0, "std": 1.0} for column in COLUMNS}
    raw["snuc"] = dict.fromkeys(COLUMNS, 1.0)
    payload = {"dim": 16, "layers": 1, "shift_cols": list(COLUMNS),
               "stats": build_fixed_per_aa(raw, COLUMNS), "recipe": RECIPE.to_dict(),
               "model_state_dict": model.state_dict()}
    torch.save(payload, path)
    return path


def _fitted():
    """A five-member calibration artifact."""
    spread = np.linspace(0.05, 1.0, 200)
    sign = np.where(np.arange(spread.size) % 2, -1.0, 1.0)
    errors, spreads = {}, {}
    for index, nucleus in enumerate(calibration.BACKBONE_NUCLEI, start=1):
        spreads[nucleus] = spread * (1.0 + index / 20.0)
        errors[nucleus] = sign * np.sqrt((0.15 * index) ** 2 + 1.7 * spreads[nucleus] ** 2)
    return calibration.fit_interval_calibration(errors, spreads, n_bins=4)


@pytest.fixture
def sparse(monkeypatch):
    monkeypatch.setattr(prediction, "parse_structure", lambda *a, **k: _residues())
    monkeypatch.setattr(prediction, "retained_insertion_partner_residues", lambda *a, **k: {})


@pytest.fixture
def ensemble(tmp_path):
    """Five members, as a training run writes them."""
    return [_member(tmp_path / f"fold{fold}.pt", seed=fold)
            for fold in calibration.MEMBER_FOLDS]


def test_distinct_members_average(tmp_path, sparse):
    members = [_member(tmp_path / f"m{i}.pt", seed=i) for i in (1, 2)]
    frame = prediction.predict_pdb("x.pdb", "A", members)
    assert len(frame) == 3


def test_intervals_are_returned_for_the_fitted_number_of_members(ensemble, sparse):
    frame, intervals = prediction.predict_pdb("x.pdb", "A", ensemble,
                                              return_intervals=True, calibration=_fitted())
    assert intervals is not None and len(frame) == 3


def test_a_calibration_requires_the_number_of_members_it_was_fitted_on(
        tmp_path, ensemble, sparse):
    with pytest.raises(ValueError, match="interval calibration requires 5 members"):
        prediction.predict_pdb("x.pdb", "A", ensemble[:2],
                               return_intervals=True, calibration=_fitted())


def test_replacing_predictions_removes_an_obsolete_interval_file(tmp_path, sparse):
    member = _member(tmp_path / "m.pt", seed=1)
    out = tmp_path / "pred.csv"
    stale = out.with_suffix(".intervals.csv")
    stale.write_text("residue_id,nucleus,ci90_halfwidth_ppm\n1,CA,0.5\n")
    args = argparse.Namespace(pdb="x.pdb", chain="A", checkpoint=[str(member)], batch_size=32,
                              cpu=True, sample_partner_chain=[], interval_calibration=None,
                              out=str(out))
    prediction.predict_command(args)
    assert out.is_file()
    assert not stale.exists()


def test_predict_command_reports_a_wrong_member_count_instead_of_crashing(tmp_path, sparse, capsys):
    """predict_pdb refuses a member-count mismatch with an exception, which is right for a
    library caller. The CLI has its own "no intervals sidecar" message for it."""
    member = _member(tmp_path / "m.pt", seed=1)
    calibration_path = tmp_path / "interval_calibration.json"
    calibration.save_calibration(_fitted(), calibration_path)
    out = tmp_path / "pred.csv"
    args = argparse.Namespace(pdb="x.pdb", chain="A", checkpoint=[str(member)], batch_size=32,
                              cpu=True, sample_partner_chain=[],
                              interval_calibration=str(calibration_path), out=str(out))
    prediction.predict_command(args)
    assert out.is_file()
    assert not out.with_suffix(".intervals.csv").exists()
    assert ("no intervals sidecar: the artifact requires five members; got 1 checkpoint(s)"
           in capsys.readouterr().out)
