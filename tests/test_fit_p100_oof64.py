"""Focused tests for the held-out p100 calibration support."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from deepshift.atom_cloud import FINAL_ATOM_CLOUD_RECIPE
from deepshift.domain import ATOM_TYPES

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_p100_oof64.py"
SPEC = importlib.util.spec_from_file_location("fit_p100_oof64", SCRIPT)
FITTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = FITTER
SPEC.loader.exec_module(FITTER)
RECIPE = FINAL_ATOM_CLOUD_RECIPE.with_vocabulary(len(ATOM_TYPES), 17).to_dict()


def test_model_valid_rows_replays_center_any_with_atom_counts(tmp_path):
    atom = tmp_path / "atom"
    atom.mkdir()
    (atom / "config.json").write_text(json.dumps({"recipe": RECIPE}))
    np.save(
        atom / "atom_offset.npy",
        np.asarray(
            [
                [6, 17, 17],
                [4, 6, 17],
                [4, 6, 17],
                [6, 6, 17],
            ],
            dtype=np.int8,
        ),
    )
    np.save(atom / "atom_count.npy", np.asarray([1, 1, 2, 0], dtype=np.int16))
    assert FITTER._model_valid_rows(
        atom, np.asarray([0, 1, 2, 3], dtype=np.int64)
    ).tolist() == [True, False, True, False]


def test_model_valid_rows_rejects_impossible_atom_count(tmp_path):
    atom = tmp_path / "atom"
    atom.mkdir()
    (atom / "config.json").write_text(json.dumps({"recipe": RECIPE}))
    np.save(atom / "atom_offset.npy", np.zeros((1, 2), dtype=np.int8))
    np.save(atom / "atom_count.npy", np.asarray([3], dtype=np.int16))
    with pytest.raises(ValueError, match="outside the atom array width"):
        FITTER._model_valid_rows(atom, np.asarray([0], dtype=np.int64))


def _calibration_manifest(tmp_path, heldout=1):
    cache = tmp_path / "work" / "unified_cache"
    recipe = {"m_atoms": 512, "heavy_cutoff": 8.0}
    for fold in FITTER.CALIBRATION_FOLDS:
        fold_dir = cache / f"fold_{fold}"
        (fold_dir / "atom").mkdir(parents=True)
        config = {
            "stats": {
                "normalization": "per-aa",
                "normalization_fit": {"heldout_fold": fold},
            },
        }
        (fold_dir / "config.json").write_text(json.dumps(config))
        (fold_dir / "atom" / "config.json").write_text(
            json.dumps({"recipe": recipe})
        )
    return {
        "arguments": {"fold": heldout},
        "protocol": {
            "checkpoint_selection": "final",
            "test_fold_used_for_selection": False,
            "epochs": 35,
        },
        "folds": {
            "heldout": [heldout],
            "train": [fold for fold in FITTER.CALIBRATION_FOLDS if fold != heldout],
        },
        "inputs": {
            "flags": {"policy": "unfiltered_calibration", "unique_cells": 0},
        },
        "recipe": recipe,
        "normalization": {
            "marker": "per-aa",
            "fit": {"heldout_fold": heldout},
        },
    }


def test_calibration_checkpoint_manifest_matches_the_cache(tmp_path):
    manifest = _calibration_manifest(tmp_path)
    FITTER._validate_calibration_checkpoint_manifest(
        manifest, run_root=tmp_path, heldout_fold=1, checkpoint_epoch=35
    )


def test_calibration_checkpoint_manifest_rejects_a_filtered_member(tmp_path):
    manifest = _calibration_manifest(tmp_path)
    manifest["inputs"]["flags"] = {"policy": "composed_exclusion", "unique_cells": 12}
    with pytest.raises(ValueError, match="unfiltered_calibration"):
        FITTER._validate_calibration_checkpoint_manifest(
            manifest, run_root=tmp_path, heldout_fold=1, checkpoint_epoch=35
        )


def test_swap_acceptance_is_mapped_back_to_the_deposited_name():
    pred = np.asarray([[1.0, 11.0]])
    deposited = np.asarray([[10.0, 1.0]])
    resolved = np.asarray([[1.0, 10.0]])
    accepted, swaps = FITTER._accepted_deposited_cells(
        pred=pred,
        deposited=deposited,
        resolved=resolved,
        effective_mask=np.asarray([[True, True]]),
        residue_code=np.asarray([0]),
        ambiguity=np.asarray([[2, 2]]),
        partner=np.asarray([[1, 0]]),
        thresholds=np.asarray([0.5, 0.5]),
    )
    assert swaps == 1
    # Prediction channel 0 accepts deposited channel 1; channel 1 misses deposited
    # channel 0. The interval fit must retain those deposited-name identities.
    assert accepted.tolist() == [[False, True]]


def test_two_stereospecific_codes_cannot_swap_for_p100_acceptance():
    pred = np.asarray([[1.0, 10.0]])
    deposited = np.asarray([[10.0, 1.0]])
    accepted, swaps = FITTER._accepted_deposited_cells(
        pred=pred,
        deposited=deposited,
        resolved=deposited,
        effective_mask=np.asarray([[True, True]]),
        residue_code=np.asarray([0]),
        ambiguity=np.asarray([[1, 1]]),
        partner=np.asarray([[1, 0]]),
        thresholds=np.asarray([0.5, 0.5]),
    )
    assert swaps == 0
    assert accepted.tolist() == [[False, False]]


def test_sanity_gate_refuses_an_empty_or_non_finite_span():
    healthy = {"ALA:ca_shift": {"lo": 48.0, "hi": 58.0},
               "ARG:ne_shift": {"lo": FITTER.ARG_NE_OBSERVED_RANGE_PPM[0],
                                "hi": FITTER.ARG_NE_OBSERVED_RANGE_PPM[1]}}
    assert FITTER.scientific_sanity(healthy, {}) == []
    empty = {**healthy, "GLY:ha2_shift": {"lo": 4.2, "hi": 3.9}}
    failures = FITTER.scientific_sanity(empty, {})
    assert [(f["gate"], f["pair"]) for f in failures] == [("fitted_span", "GLY:ha2_shift")]
    non_finite = {**healthy, "ALA:ha_shift": {"lo": float("nan"), "hi": 5.0}}
    assert [f["pair"] for f in FITTER.scientific_sanity(non_finite, {})] == ["ALA:ha_shift"]


def test_cutoffs_carry_the_fitted_span_and_report_it_beside_the_observed_width():
    residues, columns = ("ALA", "ARG"), ("ha_shift", "ne_shift")
    shape = (len(residues), len(columns))
    spans = FITTER.AcceptedSpans(
        population_n=np.array([[40, 0], [0, 12]]),
        accepted_n=np.array([[38, 0], [0, 12]]),
        observed_lo=np.array([[0.2, np.inf], [np.inf, 70.0]]),
        observed_hi=np.array([[9.5, -np.inf], [-np.inf, 95.0]]),
        accepted_lo=np.array([[3.1, np.inf], [np.inf, 80.0]]),
        accepted_hi=np.array([[5.9, -np.inf], [-np.inf, 88.0]]),
    )
    assert spans.population_n.shape == shape
    cutoffs, missing, widths = FITTER.build_cutoffs(spans, residues, columns)
    assert missing == []
    assert cutoffs["ALA:ha_shift"] == {"lo": 3.1, "hi": 5.9, "n": 40, "accepted_n": 38}
    assert (cutoffs["ARG:ne_shift"]["lo"], cutoffs["ARG:ne_shift"]["hi"]) == \
        FITTER.ARG_NE_OBSERVED_RANGE_PPM
    assert widths["ALA:ha_shift"]["fitted_span_ppm"] == pytest.approx(2.8)
    assert widths["ALA:ha_shift"]["observed_span_ppm"] == pytest.approx(9.3)
    assert widths["ARG:ne_shift"]["fitted_span_ppm"] == pytest.approx(8.0)
