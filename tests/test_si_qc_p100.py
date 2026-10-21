"""Table S6 publishes the rule's own row set, read through ``P100Policy``."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from deepshift.quality_control import ARG_NE_OBSERVED_RANGE_PPM, P100Policy

SI_BUILD = Path(__file__).resolve().parents[1] / "si_build"

COLUMNS = ("h_shift", "ca_shift", "ne_shift")
RESIDUES = ("PRO", "ALA", "ARG")


def _load_si_qc_p100():
    if str(SI_BUILD) not in sys.path:
        sys.path.insert(0, str(SI_BUILD))
    spec = importlib.util.spec_from_file_location("si_qc_p100", SI_BUILD / "si_qc_p100.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cache(root: Path, values, ambiguity=None) -> Path:
    """A two-fold cache observing every (residue, column) pair once, at ``values``, with
    the ambiguity codes ``ambiguity`` gives a pair (1, unambiguous, otherwise)."""
    cache = root / "unified_cache"
    for fold in (1, 2):
        fold_dir = cache / f"fold_{fold}" / "structural"
        fold_dir.mkdir(parents=True)
        shifts = np.array([[values[(residue, column)] for column in COLUMNS]
                           for residue in RESIDUES], dtype=np.float32)
        np.save(fold_dir / "shifts.npy", shifts)
        np.save(fold_dir / "shift_mask.npy", np.ones(shifts.shape, dtype=np.uint8))
        np.save(fold_dir / "ambiguity.npy", np.array(
            [[(ambiguity or {}).get((residue, column), 1) for column in COLUMNS]
             for residue in RESIDUES], dtype=np.int8))
        (fold_dir / "id_maps.json").write_text(json.dumps(
            {"comp_id": {str(i): residue for i, residue in enumerate(RESIDUES)}}))
        (cache / f"fold_{fold}" / "config.json").write_text(
            json.dumps({"shift_cols": list(COLUMNS)}))
    return cache


def test_table_s5_publishes_exactly_the_pairs_the_policy_accepts(tmp_path):
    """Every observed pair the policy gives an interval for is a row, and no other pair
    is: a chemically impossible pair and a pair the fitted authority does not cover are
    counted as observed and published nowhere."""
    module = _load_si_qc_p100()
    policy = P100Policy({"PRO:h_shift": {"lo": 7.8, "hi": 8.9},
                         "ALA:ca_shift": {"lo": 45.0, "hi": 65.0}})
    inside = {("PRO", "h_shift"): 8.2, ("ALA", "h_shift"): 8.2, ("ARG", "h_shift"): 8.2,
              ("PRO", "ca_shift"): 60.0, ("ALA", "ca_shift"): 55.0, ("ARG", "ca_shift"): 56.0,
              ("PRO", "ne_shift"): 84.0, ("ALA", "ne_shift"): 84.0, ("ARG", "ne_shift"): 84.0}
    spans, observed, removed = module.measure(policy, _cache(tmp_path, inside))

    expected = {f"{residue}:{column}"
                for residue in RESIDUES for column in COLUMNS
                if policy.accepted_range(residue, column) is not None}
    assert set(spans) == expected == {"ALA:ca_shift", "ARG:ne_shift"}
    assert spans["ARG:ne_shift"] == ARG_NE_OBSERVED_RANGE_PPM
    assert "PRO:h_shift" in observed and "PRO:h_shift" not in spans
    assert set(removed.values()) == {0}


def test_table_s5_counts_removals_over_every_fold_of_the_cache(tmp_path):
    """A deposited value outside its accepted interval is counted once per fold it is
    observed in, and only for pairs the policy accepts."""
    module = _load_si_qc_p100()
    policy = P100Policy({"ALA:ca_shift": {"lo": 45.0, "hi": 65.0}})
    outside = {("PRO", "h_shift"): 800.0, ("ALA", "h_shift"): 800.0, ("ARG", "h_shift"): 800.0,
               ("PRO", "ca_shift"): 900.0, ("ALA", "ca_shift"): 900.0, ("ARG", "ca_shift"): 900.0,
               ("PRO", "ne_shift"): 40.0, ("ALA", "ne_shift"): 40.0, ("ARG", "ne_shift"): 40.0}
    spans, observed, removed = module.measure(policy, _cache(tmp_path, outside))

    assert set(spans) == {"ALA:ca_shift", "ARG:ne_shift"}
    assert removed == {"ALA:ca_shift": 2, "ARG:ne_shift": 2}
    assert observed["PRO:h_shift"] == 2


def test_a_cell_the_fence_does_not_test_is_not_counted_as_removed(tmp_path):
    """The fence skips cells carrying an ambiguity-set code, so the table does too: such a
    cell is observed, and an out-of-interval value on it is not a removal."""
    module = _load_si_qc_p100()
    policy = P100Policy({"ALA:ca_shift": {"lo": 45.0, "hi": 65.0}})
    values = {(residue, column): 900.0 for residue in RESIDUES for column in COLUMNS}
    _spans, observed, removed = module.measure(
        policy, _cache(tmp_path, values, ambiguity={("ALA", "ca_shift"): 9}))
    assert observed["ALA:ca_shift"] == 2
    assert removed["ALA:ca_shift"] == 0
