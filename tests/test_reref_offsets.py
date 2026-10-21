"""The rereferencing offset itself: what it equals, which structure fits it, when it is
withheld.

Every ppm label the release ships passes through here. ``data/reref_offsets.json`` is
written by ``deepshift.data.reref``, read back by ``scripts/export_dataset.py`` (which
adds the offsets to the deposited values) and by ``deepshift.data.cache`` (which bakes
them into the training cache), and reported as Tables S3 and S4. So the arithmetic below
is asserted against numbers worked out by hand rather than against a golden file.

SHIFTX is not installed to run these. The module shells out to ``SHIFTX_BIN`` and parses
its table, so the tests point ``SHIFTX_BIN`` at a small stub that emits a table the test
wrote: exit status, stderr and output are then the test's to choose, which is what makes
the failure paths reachable at all.
"""
import csv
import json
import os
import stat
import subprocess
import textwrap

import pytest

from deepshift import domain
from deepshift.data import reref
from deepshift.data.reref_utils import SXCOLS

CFG = reref.RULESEARCH_WINNER_DEFAULT

# Ten residue types with ten distinct one-letter codes, so the numbering alignment
# between observed rows and predicted rows has exactly one answer.
TOKENS = ["ALA", "SER", "THR", "VAL", "LEU", "ILE", "ASP", "GLU", "LYS", "ASN"]
ONE = "ASTVLIDEKN"


# ---------------------------------------------------------------------------
# A stub SHIFTX, and the PDB/table pair it reads
# ---------------------------------------------------------------------------

STUB_SHIFTX = textwrap.dedent('''
    #!/usr/bin/env python3
    """Stand-in for the SHIFTX executable: emits the table named by a REMARK in the PDB.

    STUB_SHIFTX_MODE picks the failure to simulate; STUB_SHIFTX_LOG, when set, gets one
    line per invocation recording how many SSBOND records the input carried.
    """
    import os
    import sys

    _, mode, pdb_path, out_path = sys.argv[:4]
    text = open(pdb_path, errors="ignore").read()
    ssbond = [line for line in text.splitlines() if line.startswith("SSBOND")]
    malformed = [line for line in ssbond
                 if line[11:14].strip() != "CYS" or line[25:28].strip() != "CYS"]
    log = os.environ.get("STUB_SHIFTX_LOG")
    if log:
        with open(log, "a") as handle:
            handle.write("%s\\t%d\\t%d\\n" % (mode, len(ssbond), len(malformed)))

    behaviour = os.environ.get("STUB_SHIFTX_MODE", "ok")
    if behaviour == "exit3":
        sys.exit(3)
    if behaviour == "empty":
        open(out_path, "w").close()
        sys.exit(0)
    if malformed:
        sys.stderr.write("SSBOND specification format error\\n")
        sys.exit(1)
    table = next(line.split()[3] for line in text.splitlines()
                 if line.startswith("REMARK 999 TABLE"))
    open(out_path, "w").write(open(table).read())
''').lstrip()


@pytest.fixture
def shiftx(tmp_path, monkeypatch):
    """``SHIFTX_BIN`` pointed at the stub, plus its invocation log."""
    binary = tmp_path / "shiftx_stub.py"
    binary.write_text(STUB_SHIFTX)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "shiftx_calls.log"
    monkeypatch.setattr(reref, "SHIFTX_BIN", str(binary))
    monkeypatch.setenv("STUB_SHIFTX_LOG", str(log))
    return log


def shiftx_table(path, rows):
    """Write a SHIFTX table. ``rows`` is ``[(residue number, one-letter code, {col: ppm})]``
    over ``SXCOLS``; a column left out is written as SHIFTX's own 0.000 for absent."""
    lines = []
    for number, code, values in rows:
        cells = " ".join(f"{values.get(col, 0.0):9.3f}" for col in SXCOLS)
        lines.append(f"{number:>5d} {code} {cells}")
    path.write_text("\n".join(lines) + "\n")
    return path


def atom_line(serial, name, residue, chain, number, altloc=" "):
    return (f"ATOM  {serial:>5d} {name:<4s}{altloc}{residue:>3s} {chain}{number:>4d}    "
            f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00\n")


def write_pdb(path, table, *, tokens=TOKENS, chain="A", extra=()):
    """A CA-only PDB naming the SHIFTX table the stub should answer with."""
    lines = [f"REMARK 999 TABLE {table}\n", *extra]
    for index, token in enumerate(tokens, start=1):
        lines.append(atom_line(index, "CA", token, chain, index))
    lines.append("END\n")
    path.write_text("".join(lines))
    return path


# ---------------------------------------------------------------------------
# keepA_chain: what reaches SHIFTX
# ---------------------------------------------------------------------------

def test_the_chain_filter_keeps_one_chain_its_first_altloc_and_every_header_line(tmp_path):
    """SHIFTX is run on one chain. A second chain's atoms would be predicted as part of
    the same polymer, and a second conformer would give the same residue two sets of
    coordinates -- while the SSBOND and REMARK records, which SHIFTX reads, must survive."""
    source = tmp_path / "two_chains.pdb"
    source.write_text(
        "REMARK 999 TABLE shifts.tab\n"
        "SSBOND   1 CYS A    3    CYS A    9\n"
        + atom_line(1, "CA", "ALA", "A", 1)
        + atom_line(2, "CA", "SER", "A", 2, altloc="A")
        + atom_line(3, "CA", "SER", "A", 2, altloc="B")
        + atom_line(4, "CA", "THR", "B", 1)
        + "HETATM    5  O   HOH B   2       0.000   0.000   0.000  1.00  0.00\n"
        + "END\n")

    kept = reref.keepA_chain(str(source), want_chain="A")
    try:
        with open(kept) as handle:
            lines = handle.readlines()
    finally:
        os.remove(kept)

    assert lines[0].startswith("REMARK 999 TABLE")
    assert lines[1].startswith("SSBOND")
    assert lines[-1].strip() == "END"
    atoms = [line for line in lines if line[:6] in ("ATOM  ", "HETATM")]
    assert {line[21] for line in atoms} == {"A"}
    # The kept conformer is stripped of its altloc character, so downstream readers see
    # one unlabelled copy rather than a still-alternate one.
    assert [(line[17:20], line[16]) for line in atoms] == [("ALA", " "), ("SER", " ")]


def test_no_wanted_chain_keeps_every_chain(tmp_path):
    source = tmp_path / "all.pdb"
    source.write_text(atom_line(1, "CA", "ALA", "A", 1) + atom_line(2, "CA", "THR", "B", 1))
    kept = reref.keepA_chain(str(source), want_chain=None)
    try:
        with open(kept) as handle:
            assert {line[21] for line in handle} == {"A", "B"}
    finally:
        os.remove(kept)


# ---------------------------------------------------------------------------
# run_shiftx: a failure is a flag, never a silent empty prediction
# ---------------------------------------------------------------------------

def _pdb_with_table(tmp_path, name="s.pdb"):
    table = shiftx_table(tmp_path / "table.txt",
                         [(i, ONE[i - 1], {"CA": 50.0 + i}) for i in range(1, 11)])
    return write_pdb(tmp_path / name, table)


def test_a_shiftx_run_that_fails_is_told_apart_from_one_that_predicted_nothing(
        tmp_path, shiftx, monkeypatch):
    """A missing offset has to be attributable. The three outcomes -- predictions, a
    non-zero exit, and exit zero with an unparsable table -- carry three different
    records, and only the first yields predictions."""
    pdb = _pdb_with_table(tmp_path)

    preds, flags = reref.run_shiftx(str(pdb), "A", return_flags=True)
    assert flags == [] and len(preds) == 10

    monkeypatch.setenv("STUB_SHIFTX_MODE", "exit3")
    preds, flags = reref.run_shiftx(str(pdb), "A", return_flags=True)
    assert preds == {} and flags == ["shiftx_exit_3"]

    monkeypatch.setenv("STUB_SHIFTX_MODE", "empty")
    preds, flags = reref.run_shiftx(str(pdb), "A", return_flags=True)
    assert preds == {} and flags == ["shiftx_empty_output"]


def test_a_shiftx_failure_raises_when_the_caller_did_not_ask_for_flags(
        tmp_path, shiftx, monkeypatch):
    """Without ``return_flags`` there is nowhere to record the failure, so a subprocess
    that never ran must not come back as an empty prediction."""
    pdb = _pdb_with_table(tmp_path)
    monkeypatch.setattr(reref, "SHIFTX_BIN", str(tmp_path / "does_not_exist"))
    with pytest.raises(RuntimeError, match="SHIFTX failed"):
        reref.run_shiftx(str(pdb), "A")
    assert reref.run_shiftx(str(pdb), "A", return_flags=True)[0] == {}


def test_a_shiftx_timeout_is_recorded_rather_than_read_as_no_prediction(tmp_path, monkeypatch):
    def never_returns(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="shiftx", timeout=180)

    monkeypatch.setattr(reref.subprocess, "run", never_returns)
    assert reref.run_shiftx("x.pdb", "A", return_flags=True) == ({}, ["shiftx_timeout"])
    with pytest.raises(RuntimeError, match="timed out"):
        reref.run_shiftx("x.pdb", "A")


def test_a_malformed_ssbond_header_is_retried_without_it_and_the_valid_ones_are_kept(
        tmp_path, shiftx):
    """One nonstandard disulfide endpoint makes SHIFTX abort on the whole structure. The
    retry drops only the record SHIFTX cannot parse: dropping every SSBOND would change
    the chemistry of the structure the offset is fitted on."""
    table = shiftx_table(tmp_path / "table.txt",
                         [(i, ONE[i - 1], {"CA": 50.0 + i}) for i in range(1, 11)])
    pdb = write_pdb(tmp_path / "ss.pdb", table, extra=[
        "SSBOND   1 CYS A    3    CYS A    9\n",
        "SSBOND   2 HCS A    4    CYS A    8\n",
    ])

    preds, flags = reref.run_shiftx(str(pdb), "A", return_flags=True)
    assert len(preds) == 10
    assert flags == ["shiftx_removed_malformed_ssbond"]

    first, retry = [line.split("\t") for line in shiftx.read_text().splitlines()]
    assert (int(first[1]), int(first[2])) == (2, 1)     # both records, one unparsable
    assert (int(retry[1]), int(retry[2])) == (1, 0)     # the CYS-CYS record survived


# ---------------------------------------------------------------------------
# eval_structure: the offset is mean(prediction - observation)
# ---------------------------------------------------------------------------

def _body(observed, tokens=ONE):
    return {number: [tokens[number - 1], values] for number, values in observed.items()}


def _preds(predicted, tokens=ONE, shift=0):
    return {number + shift: (tokens[number - 1], values)
            for number, values in predicted.items()}


def test_the_offset_is_the_mean_of_prediction_minus_observation():
    """Sense and magnitude at once: predictions sit a fixed 0.6 ppm above the deposited
    carbons, so the offset that removes the deposition's referencing error is +0.6 and
    not -0.6. Getting the sign wrong doubles every error instead of removing it."""
    observed = {i: {"CA": 50.0 + i} for i in range(1, 11)}
    predicted = {i: {"CA": 50.0 + i + 0.6} for i in range(1, 11)}
    _overall, per, frac = reref.eval_structure(_body(observed), _preds(predicted))
    assert frac == 1.0
    assert per["CA"]["n"] == 10
    assert per["CA"]["avgdiff"] == pytest.approx(0.6)


def test_a_residue_whose_prediction_is_outside_the_window_does_not_move_the_offset():
    """The mean is taken over cells within the per-nucleus window. One wildly wrong
    prediction -- a mismodelled loop, a mis-numbered residue -- would otherwise drag a
    constant that is applied to every residue of the deposition."""
    observed = {i: {"CA": 50.0 + i} for i in range(1, 11)}
    predicted = {i: {"CA": 50.0 + i + 0.6} for i in range(1, 11)}
    predicted[10]["CA"] += 20.0                          # 20.6 ppm out, window is 5.0
    _overall, per, _frac = reref.eval_structure(_body(observed), _preds(predicted))
    assert per["CA"] == {"avgdiff": pytest.approx(0.6), "n": 9}


def test_a_nucleus_with_fewer_than_three_aligned_cells_gets_no_statistic():
    observed = {i: {"CA": 50.0 + i, "CB": 20.0 + i} for i in range(1, 11)}
    predicted = {i: {"CA": 50.0 + i + 0.6} for i in range(1, 11)}
    for i in (1, 2):
        predicted[i]["CB"] = 20.0 + i + 0.4
    _overall, per, _frac = reref.eval_structure(_body(observed), _preds(predicted))
    assert set(per) == {"CA"}


def test_residues_are_paired_by_the_numbering_shift_that_matches_the_sequence():
    """Deposited residue numbering and structure numbering routinely differ by a constant
    (an expression tag, a construct boundary). The offset is only meaningful if the two
    are paired on the shift that matches the sequence, and cells whose residue type
    disagrees after that shift are dropped."""
    observed = {i: {"CA": 50.0 + i} for i in range(1, 11)}
    predicted = {i: {"CA": 50.0 + i + 0.6} for i in range(1, 11)}
    _overall, per, frac = reref.eval_structure(_body(observed), _preds(predicted, shift=100))
    assert frac == 1.0
    assert per["CA"] == {"avgdiff": pytest.approx(0.6), "n": 10}

    # The same rows read against a structure whose sequence only partly agrees: the
    # mismatched residues are not paired, and the coverage the caller gates on drops.
    mismatched = "ASTVLIDEKN"[:7] + "WWW"
    _overall, per, frac = reref.eval_structure(
        _body(observed, tokens=mismatched), _preds(predicted))
    assert frac == pytest.approx(0.7)
    assert per["CA"] == {"avgdiff": pytest.approx(0.6), "n": 7}


def test_the_asterisk_shiftx_prefixes_some_residue_numbers_with_is_stripped(tmp_path):
    """SHIFTX marks some otherwise valid residue numbers with a leading ``*`` -- every row
    of PDB 1AIN. Left unstripped the row is skipped, and a structure can lose its whole
    prediction without any error being raised."""
    output = tmp_path / "shiftx.out"
    output.write_text("   *3 A     4.100     8.200   120.300    52.400    19.500   176.600\n")
    assert reref.parse_shiftx_output(output)[3][0] == "A"


def test_a_deposition_too_short_to_align_is_reported_as_uncovered():
    """Fewer than five paired residues is not enough to trust a numbering alignment, and
    an alignment guessed from three residues could pair the wrong ones and still look
    self-consistent. The coverage comes back as zero, which is under the driver's floor."""
    observed = {i: {"CA": 50.0 + i} for i in range(1, 5)}
    predicted = {i: {"CA": 50.0 + i + 0.6} for i in range(1, 5)}
    _overall, _per, frac = reref.eval_structure(_body(observed), _preds(predicted))
    assert frac == 0.0


def test_a_prediction_shiftx_did_not_make_is_not_read_as_zero_ppm(tmp_path):
    """SHIFTX writes 0.000 for a nucleus it did not predict. Read literally that is a
    50 ppm error on every carbon, which would swamp the mean it is averaged into."""
    observed = {i: {"CA": 50.0 + i} for i in range(1, 11)}
    predicted = {i: {"CA": 50.0 + i + 0.6} for i in range(1, 11)}
    predicted[4]["CA"] = 0.0
    parsed = reref.parse_shiftx_output(shiftx_table(
        tmp_path / "table.txt", [(i, ONE[i - 1], predicted[i]) for i in range(1, 11)]))
    _overall, per, _frac = reref.eval_structure(_body(observed), parsed)
    assert per["CA"] == {"avgdiff": pytest.approx(0.6), "n": 9}


# ---------------------------------------------------------------------------
# derive_rulesearch_offsets: the arithmetic on top of the statistics
# ---------------------------------------------------------------------------

def _stats(**per_nucleus):
    """``{nucleus: (mean, count)}`` in the candidate schema the arithmetic reads."""
    return {nucleus: {"mean_fixed": mean, "n_fixed": count}
            for nucleus, (mean, count) in per_nucleus.items()}


def test_the_two_aliphatic_carbons_share_one_count_weighted_offset():
    """CA and CB are one 13C reference channel, so they take one number -- weighted by
    how many cells each was measured on, not averaged evenly. Here the even average is
    0.50 and the weighted one 0.529, which rounds to a different grid point."""
    offsets = reref.derive_rulesearch_offsets(
        _stats(CA=(0.6, 9), CB=(0.4, 5)), CFG["arithmetic"])
    assert offsets["CA"] == offsets["CB"] == pytest.approx(0.55)


def test_the_other_two_channels_keep_their_own_offsets():
    offsets = reref.derive_rulesearch_offsets(
        _stats(CA=(0.6, 9), CB=(0.4, 5), CO=(0.12, 10), N=(-0.33, 10)),
        CFG["arithmetic"])
    assert offsets["CO"] == pytest.approx(0.10)
    assert offsets["N"] == pytest.approx(-0.35)
    assert offsets["CO"] != offsets["CA"]


def test_only_one_aliphatic_carbon_still_sets_the_channel():
    assert reref.derive_rulesearch_offsets(
        _stats(CA=(0.6, 9)), CFG["arithmetic"]) == {"CA": pytest.approx(0.6)}
    assert reref.derive_rulesearch_offsets(
        _stats(CB=(0.4, 5)), CFG["arithmetic"]) == {"CB": pytest.approx(0.4)}


def test_protons_are_never_given_an_offset_however_far_off_they_are():
    """1H referencing errors are insignificant and RefDB never applies a proton offset.
    A 3 ppm HA disagreement is a bad structure, not a bad reference."""
    offsets = reref.derive_rulesearch_offsets(
        _stats(CA=(0.6, 9), CB=(0.6, 9), HA=(0.3, 20), HN=(-3.0, 20)),
        CFG["arithmetic"])
    assert set(offsets) == {"CA", "CB"}


def test_a_nucleus_measured_on_too_few_cells_is_left_uncorrected():
    """min_cells is the whole reason a label set can be realized and still carry no
    offset -- the count reported under Table S4's realized line."""
    arithmetic = CFG["arithmetic"]
    assert arithmetic["min_cells"] == 3
    assert "CO" not in reref.derive_rulesearch_offsets(
        _stats(CA=(0.6, 9), CB=(0.6, 9), CO=(2.0, 2)), arithmetic)
    assert "CO" in reref.derive_rulesearch_offsets(
        _stats(CA=(0.6, 9), CB=(0.6, 9), CO=(2.0, 3)), arithmetic)


def test_an_offset_that_rounds_to_zero_is_not_recorded_at_all():
    """An offset of exactly zero is a no-op, and Table S3 counts label sets whose stored
    offset is non-zero; a stored 0.0 would inflate the corrected count with corrections
    that do nothing."""
    assert reref.derive_rulesearch_offsets(
        _stats(CA=(0.01, 9), CB=(0.01, 9)), CFG["arithmetic"]) == {}
    assert reref.derive_rulesearch_offsets(
        _stats(CA=(0.0, 9), CB=(0.0, 9)), CFG["arithmetic"]) == {}


def test_the_offsets_land_on_the_declared_grid():
    assert reref.OFFSET_GRID_PPM == 0.05
    for raw in (0.37, -1.23, 4.44, 0.061):
        offsets = reref.derive_rulesearch_offsets(_stats(CO=(raw, 9)), CFG["arithmetic"])
        assert offsets["CO"] / 0.05 == pytest.approx(round(offsets["CO"] / 0.05))


# ---------------------------------------------------------------------------
# Structure selection
# ---------------------------------------------------------------------------

def _candidate(name, *, score, qcov, source="af", chain="A"):
    return {"source": source, "accession": name, "chain": chain,
            "candidate_id": name, "qcov": qcov, "scores": {"meanr_all": score}}


def test_the_candidate_with_the_best_shiftx_agreement_wins():
    pool = [_candidate("low", score=0.70, qcov=1.0), _candidate("high", score=0.95, qcov=1.0)]
    assert reref.select_rulesearch_candidate(
        pool, CFG["selection"])["candidate_id"] == "high"


def test_a_candidate_covering_the_deposition_outranks_a_better_correlated_fragment():
    """The offset is one constant for the whole deposition, so a structure covering 60%
    of it fits that constant on 60% of the evidence however well those residues agree."""
    assert CFG["selection"]["selector"]["preference_gate"] == 0.95
    pool = [_candidate("fragment", score=0.99, qcov=0.60),
            _candidate("whole", score=0.80, qcov=0.96)]
    assert reref.select_rulesearch_candidate(
        pool, CFG["selection"])["candidate_id"] == "whole"
    # With nothing over the gate the preference does not apply and correlation decides.
    below = [_candidate("fragment", score=0.99, qcov=0.60),
             _candidate("other", score=0.80, qcov=0.70)]
    assert reref.select_rulesearch_candidate(
        below, CFG["selection"])["candidate_id"] == "fragment"


def test_a_candidate_without_a_finite_correlation_cannot_be_selected():
    """A structure that produced too few paired cells scores NaN. Selecting it would fit
    the offset on the evidence that was too thin to rank it."""
    pool = [_candidate("nan", score=float("nan"), qcov=1.0),
            _candidate("none", score=None, qcov=1.0),
            _candidate("real", score=0.10, qcov=1.0)]
    assert reref.select_rulesearch_candidate(
        pool, CFG["selection"])["candidate_id"] == "real"
    assert reref.select_rulesearch_candidate(pool[:2], CFG["selection"]) is None
    assert reref.select_rulesearch_candidate([], CFG["selection"]) is None


def test_candidates_within_the_tie_tolerance_break_towards_the_experimental_structure():
    tolerance = CFG["selection"]["tie_tolerance"]
    pool = [_candidate("model", score=0.900, qcov=1.0, source="af"),
            _candidate("crystal", score=0.900 - tolerance / 2, qcov=1.0, source="xtal"),
            _candidate("solution", score=0.900 - tolerance / 2, qcov=1.0, source="nmr")]
    assert reref.select_rulesearch_candidate(
        pool, CFG["selection"])["candidate_id"] == "crystal"
    # Outside the tolerance the correlation decides again, tie-break or not.
    clear = [_candidate("model", score=0.900, qcov=1.0, source="af"),
             _candidate("crystal", score=0.700, qcov=1.0, source="xtal")]
    assert reref.select_rulesearch_candidate(
        clear, CFG["selection"])["candidate_id"] == "model"


def test_tied_candidates_of_one_method_break_towards_the_wider_coverage():
    pool = [_candidate("narrow", score=0.9, qcov=0.96, source="xtal"),
            _candidate("wide", score=0.9, qcov=0.99, source="xtal")]
    assert reref.select_rulesearch_candidate(
        pool, CFG["selection"])["candidate_id"] == "wide"


def _realized(source, *, overall, frac, chain="A"):
    return {"source": source, "struct": (f"/{source}.pdb", chain),
            "per": {"CA": {"avgdiff": 0.1, "n": 9}}, "overall": overall, "frac": frac}


def test_the_selected_structure_is_the_one_that_agrees_with_the_deposition():
    candidates = [_realized("xtal", overall=0.60, frac=1.0),
                  _realized("af", overall=0.90, frac=1.0)]
    assert reref.select_reref_structure(candidates, CFG)["source"] == "af"


def test_structure_selection_falls_back_to_source_priority_when_nothing_can_be_ranked():
    """Every candidate scoring NaN is not a reason to leave the deposition uncorrected:
    the documented fallback is the experimental structure, crystal before solution
    before model."""
    candidates = [_realized("af", overall=float("nan"), frac=1.0),
                  _realized("nmr", overall=float("nan"), frac=1.0),
                  _realized("xtal", overall=float("nan"), frac=1.0)]
    assert reref.select_reref_structure(candidates, CFG)["source"] == "xtal"
    assert reref.select_reref_structure(candidates[:2], CFG)["source"] == "nmr"


# ---------------------------------------------------------------------------
# offsets_for_method: the adapter between the two
# ---------------------------------------------------------------------------

def test_the_realized_counts_are_what_weights_the_carbon_offset():
    per = {"CA": {"avgdiff": 0.6, "n": 9}, "CB": {"avgdiff": 0.4, "n": 5}}
    assert reref.offsets_for_method(per, CFG)["CA"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# The record, and the driver that writes reref_offsets.json
# ---------------------------------------------------------------------------

def test_no_computed_offset_and_an_empty_computed_one_are_different_records():
    """Table S4 separates them: 'with no offset' counts label sets the procedure could
    not reach, 'carrying at least one non-empty nucleus' counts the ones it corrected.
    A label set whose every nucleus fell below the cell floor belongs to neither."""
    label_set = domain.label_set_id("7", "1", "1", "1", domain.exact_polymer_sequence_sha256(TOKENS))
    absent = reref.offset_record_for_label_set(
        label_set, selected_list_id="1", alignment_sequence_sha256="a" * 64)
    empty = reref.offset_record_for_label_set(
        label_set, computed_offsets={}, selected_list_id="1",
        alignment_sequence_sha256="a" * 64)
    assert (absent["method"], absent["offsets"]) == ("none", {})
    assert (empty["method"], empty["offsets"]) == ("computed", {})


def _write_csv(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


SHIFT_COLUMNS = ["ca_shift", "cb_shift", "c_shift", "n_shift", "ha_shift", "h_shift"]


class Build:
    """The three CSVs plus catalogue the rereferencing driver reads, and the offsets it
    writes back."""

    def __init__(self, root):
        self.root = root
        self.catalog, self.shifts = [], []
        self.structures = {"af": [], "nmr": [], "xtal": []}
        self.build_logs = {"af": [], "nmr": [], "xtal": []}

    def label_set(self, list_id, *, tokens=TOKENS):
        exact = domain.exact_polymer_sequence_sha256(tokens)
        alignment = domain.alignment_sequence_from_tokens(tokens)
        identity = domain.label_set_id("7", list_id, "1", "1", exact)
        self.catalog.append({
            "logical_entity_id": domain.logical_entity_id("7", "1", exact),
            "label_set_id": identity, "bmrb_id": "7", "entity_id": "1",
            "shift_list_id": list_id, "entity_assembly_id": "1",
            "sample_condition_ids": "", "exact_polymer_sequence_sha256": exact,
            "alignment_sequence_sha256": domain.alignment_sequence_sha256(alignment),
            "sequence_scope": "full", "polymer_length": len(tokens),
            "polymer_seq_nums": ",".join(str(i) for i in range(1, len(tokens) + 1)),
            "polymer_token_sequence": " ".join(tokens), "alignment_sequence": alignment,
            "n_supported_residues": len(tokens), "n_supported_cells": len(tokens)})
        for index, token in enumerate(tokens, start=1):
            self.shifts.append({"label_set_id": identity, "bmrb_id": "7",
                                "shift_list_id": list_id, "comp_index_id": index,
                                "comp_id": token})
        return identity

    def realize(self, identity, source, materialization, observed, *, tokens=TOKENS):
        for number, values in sorted(observed.items()):
            self.structures[source].append({
                "label_set_id": identity, "bmrb_id": "7",
                "materialization_id": materialization, "comp_index_id": number,
                "comp_id": tokens[number - 1], "struct_res_id": number,
                **{column: values.get(column, "") for column in SHIFT_COLUMNS}})

    def structure(self, source, materialization, pdb, chain="A"):
        self.build_logs[source].append({
            "materialization_id": materialization, "pdb_path": str(pdb),
            "chain_id": chain, "status": "success"})

    def run(self, *extra):
        _write_csv(self.root / "entity_catalog.csv", list(self.catalog[0]), self.catalog)
        _write_csv(self.root / "chemical_shifts.csv", list(self.shifts[0]), self.shifts)
        arguments = ["--shifts_file", str(self.root / "chemical_shifts.csv"),
                     "--out", str(self.root / "reref_offsets.json")]
        for source, rows in self.structures.items():
            path = self.root / f"structure_data_{source}.csv"
            _write_csv(path, ["label_set_id", "bmrb_id", "materialization_id",
                              "comp_index_id", "comp_id", "struct_res_id", *SHIFT_COLUMNS],
                       rows)
            arguments += ["--structure-master", f"{source}:{path}"]
        for source, rows in self.build_logs.items():
            path = self.root / f"structure_build_log_{source}.csv"
            _write_csv(path, ["materialization_id", "pdb_path", "chain_id", "status"], rows)
            arguments += ["--build-log", f"{source}:{path}"]
        reref.entity_reref_main([*arguments, *extra])
        return (json.loads((self.root / "reref_offsets.json").read_text()),
                json.loads((self.root / "reref_provenance.json").read_text()))


@pytest.fixture
def build(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    return Build(root)


def test_two_label_sets_on_one_structure_each_get_their_own_offset(tmp_path, shiftx, build):
    """The grain is (label set, nucleus). Two depositions of the same protein are
    referenced independently, so sharing a structure -- and sharing the one SHIFTX run
    over it -- must not give them a shared offset.

    Predictions sit 0.6 ppm above the first deposition's carbons and 1.0 ppm above the
    second's; those are the two offsets.
    """
    predicted = {i: {"CA": 50.0 + i + 0.6, "CB": 20.0 + i + 0.4,
                     "C": 175.0 + i + 0.12, "N": 120.0 + i - 0.33,
                     "HA": 4.0 + i * 0.01 + 0.3, "H": 8.0 + i * 0.01 + 0.5}
                 for i in range(1, 11)}
    table = shiftx_table(tmp_path / "table.txt",
                         [(i, ONE[i - 1], predicted[i]) for i in range(1, 11)])
    pdb = write_pdb(tmp_path / "shared.pdb", table)
    build.structure("af", "mz_a", pdb)

    first = build.label_set("1")
    # CA on nine residues, CB on five: the weighted carbon mean is 0.529, the even one 0.5.
    build.realize(first, "af", "mz_a", {
        i: {"ca_shift": 50.0 + i,
            **({"cb_shift": 20.0 + i} if i <= 5 else {}),
            "c_shift": 175.0 + i, "n_shift": 120.0 + i,
            "ha_shift": 4.0 + i * 0.01, "h_shift": 8.0 + i * 0.01}
        for i in range(1, 10)})
    second = build.label_set("2")
    build.realize(second, "af", "mz_a", {
        i: {"ca_shift": 50.0 + i + 0.6 - 1.0} for i in range(1, 11)})

    table_json, provenance = build.run("--workers", "1")

    assert table_json[first]["offsets"] == {
        "CA": pytest.approx(0.55), "CB": pytest.approx(0.55),
        "CO": pytest.approx(0.10), "N": pytest.approx(-0.35)}
    assert table_json[second]["offsets"] == {"CA": pytest.approx(1.0)}
    assert table_json[first]["struct_used"] == "af:A"
    assert provenance["n_label_sets_with_offsets"] == 2
    # One SHIFTX run for the structure the two label sets share.
    assert len(shiftx.read_text().splitlines()) == 1


def test_a_label_set_with_no_realized_structure_keeps_its_deposited_values(shiftx, build):
    """Nothing to predict against is not an offset of zero: the record says so, and the
    coverage table counts it."""
    orphan = build.label_set("1")
    table_json, provenance = build.run("--workers", "1")
    assert table_json[orphan]["method"] == "none"
    assert table_json[orphan]["offsets"] == {}
    assert table_json[orphan]["struct_type"] is None
    assert provenance["n_label_sets_with_offsets"] == 0


def test_an_alignment_below_the_coverage_floor_is_not_used_for_an_offset(
        tmp_path, shiftx, build):
    """Under 0.8 sequence agreement the structure is not this deposition's protein, and a
    constant fitted on it would be noise applied to every residue."""
    # A structure whose last five residues are not the deposition's: five of ten
    # residues pair, which is under the floor.
    wrong = ["ALA", "SER", "THR", "VAL", "LEU", "TRP", "TRP", "TRP", "TRP", "TRP"]
    table = shiftx_table(tmp_path / "table.txt", [
        (i, domain.AA_3_TO_1[token], {"CA": 50.0 + i + 0.6})
        for i, token in enumerate(wrong, start=1)])
    pdb = write_pdb(tmp_path / "wrong.pdb", table, tokens=wrong)
    build.structure("af", "mz_a", pdb)
    identity = build.label_set("1")
    build.realize(identity, "af", "mz_a",
                  {i: {"ca_shift": 50.0 + i} for i in range(1, 11)})

    table_json, _provenance = build.run("--workers", "1")
    assert table_json[identity]["method"] == "none"
    assert table_json[identity]["offsets"] == {}


def test_a_shiftx_failure_leaves_the_deposition_uncorrected_and_says_which_structure(
        tmp_path, shiftx, build, monkeypatch):
    """A failed subprocess must never look like a deposition that needed no correction.
    The record is a no-op and the sidecar names the structure and the exit status."""
    monkeypatch.setenv("STUB_SHIFTX_MODE", "exit3")
    table = shiftx_table(tmp_path / "table.txt",
                         [(i, ONE[i - 1], {"CA": 50.0 + i}) for i in range(1, 11)])
    pdb = write_pdb(tmp_path / "s.pdb", table)
    build.structure("af", "mz_a", pdb)
    identity = build.label_set("1")
    build.realize(identity, "af", "mz_a", {i: {"ca_shift": 50.0 + i} for i in range(1, 11)})

    table_json, provenance = build.run("--workers", "1")
    assert table_json[identity]["method"] == "none"
    assert provenance["n_label_sets_with_offsets"] == 0
    assert [event["flags"] for event in provenance["shiftx_events"]] == [["shiftx_exit_3"]]
    assert provenance["shiftx_events"][0]["pdb_path"] == str(pdb)


def test_the_better_agreeing_structure_is_the_one_the_offset_is_fitted_on(
        tmp_path, shiftx, build):
    """Two realized sources, one constant. The crystal structure here reproduces the
    deposition's carbons to a constant and the model does not, so the selector must take
    the crystal -- and the offset must be the crystal's number."""
    good = shiftx_table(tmp_path / "good.txt", [
        (i, ONE[i - 1], {"CA": 50.0 + i + 0.6}) for i in range(1, 11)])
    noisy = shiftx_table(tmp_path / "noisy.txt", [
        (i, ONE[i - 1], {"CA": 50.0 + (9 - i) + 1.4}) for i in range(1, 11)])
    build.structure("xtal", "mz_x", write_pdb(tmp_path / "xtal.pdb", good))
    build.structure("af", "mz_f", write_pdb(tmp_path / "af.pdb", noisy))

    identity = build.label_set("1")
    observed = {i: {"ca_shift": 50.0 + i} for i in range(1, 11)}
    build.realize(identity, "xtal", "mz_x", observed)
    build.realize(identity, "af", "mz_f", observed)

    table_json, _provenance = build.run("--workers", "1")
    assert table_json[identity]["struct_used"] == "xtal:A"
    assert table_json[identity]["struct_type"] == "xtal"
    assert table_json[identity]["offsets"]["CA"] == pytest.approx(0.6)
