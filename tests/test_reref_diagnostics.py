"""Each label set's rereferencing outcome is literal: a label set with no Calpha, Cbeta, C'
or N shift is filed as having none to correct, however far its alignment got."""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reref_diagnostics", REPO / "scripts" / "reref_diagnostics.py")
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)

STRUCT = ("x.pdb", "A")
# Three residues: below the five an alignment needs.
PREDS = {i: ("A", {"H": 8.0, "CA": 52.0}) for i in (1, 2, 3)}


def _outcome(body):
    bodies = {("ls", "af"): (body, "m1")}
    _record, outcome, detail, _fits = diag.diagnose(
        "ls", bodies, {"m1": STRUCT}, {STRUCT: PREDS}, diag.R.RULESEARCH_WINNER_DEFAULT)
    return outcome, detail


def test_protons_only_on_too_few_residues_is_no_correctable_shift():
    body = {i: ["A", {"HN": 8.0}] for i in (1, 2, 3)}
    assert _outcome(body) == ("no_correctable_shift", "protons_only")


def test_a_correctable_shift_on_too_few_residues_is_too_few_residues():
    body = {i: ["A", {"CA": 52.0}] for i in (1, 2, 3)}
    assert _outcome(body) == ("too_few_residues", "fewer_than_5_aligned")


def test_an_empty_body_has_no_backbone_shift_and_an_absent_one_no_structure():
    assert _outcome({}) == ("no_correctable_shift", "no_backbone_shift")
    _record, outcome, detail, _fits = diag.diagnose(
        "ls", {}, {}, {}, diag.R.RULESEARCH_WINNER_DEFAULT)
    assert (outcome, detail) == ("no_structure", "not_realized")


def test_each_candidate_source_records_its_unrounded_fit_and_residue_count():
    preds = {i: ("A", {"H": 8.0, "CA": 52.0}) for i in range(1, 7)}
    body = {i: ["A", {"CA": 51.0}] for i in range(1, 7)}
    _record, outcome, _detail, fits = diag.diagnose(
        "ls", {("ls", "af"): (body, "m1")}, {"m1": STRUCT}, {STRUCT: preds},
        diag.R.RULESEARCH_WINNER_DEFAULT)
    assert outcome == "offset"
    assert fits == {"af": {"CA": {"offset_ppm": 1.0, "residues": 6}}}
