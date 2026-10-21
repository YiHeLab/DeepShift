"""The anti-correlated channel list: what may be excluded, and on whose evidence.

These are deposited chemical shift series whose ORDER is wrong -- the values sit inside
the nucleus's physical range, so the p100 fence cannot see them, but they run backwards
against what any predictor computes from the structure. Removing cells from a benchmark
is the most self-serving edit a method paper can make, so the contract is narrow: the
judges are competitors only, every channel is read back from the BMRB archive before it
counts, and the list is data rather than code.
"""
import json

import pytest

from benchmarks import master_cells

RECORD = json.loads(master_cells.ANTICORRELATED.read_text())


def test_deepshift_is_not_a_judge():
    """The whole point. If our own model could vote, the exclusion would be tunable to
    our own errors and the benchmark would be worthless."""
    judges = RECORD["rule"]["judges"]
    assert "deepshift" not in judges
    assert RECORD["rule"]["deepshift_is_not_a_judge"] is True
    for channel in RECORD["channels"]:
        assert "deepshift" not in channel["r_independent_judges"]


def test_the_flagged_set_is_the_shape_the_precedent_describes():
    """UCBShift removed a nucleus from named proteins, not whole proteins. Ours is the same
    shape. The rule is applied to every nucleus, so the concentration in C-prime is an
    outcome rather than an input: it must still hold, or the rule is selecting noise."""
    nuclei = [c["nucleus"] for c in RECORD["channels"]]
    assert set(nuclei) <= {"C", "H", "HA", "CA", "CB", "N"}
    assert nuclei.count("C") >= 0.8 * len(nuclei), nuclei


def test_every_channel_was_read_back_from_the_archive():
    """Every channel is checked. A channel that reproduces the archive says so with r = 1;
    one that does not must carry an archive_status explaining what was found. DeepShift's
    own correlation is recorded but never asserted on: it is not one of the judges."""
    for channel in RECORD["channels"]:
        assert channel["archive_checked"] is True, channel["bmrb"]
        if channel["archive_verified"]:
            assert channel["r_archive_vs_our_stored"] == pytest.approx(1.0, abs=1e-3), (
                f"BMRB {channel['bmrb']} {channel['nucleus']}: our stored values are not "
                "the deposited ones, so the defect may be in this package not the archive")
        else:
            assert channel.get("archive_status"), channel["bmrb"]


def test_a_channel_that_was_never_checked_is_refused(tmp_path, monkeypatch):
    payload = json.loads(json.dumps(RECORD))
    payload["channels"][0].pop("archive_checked", None)
    path = tmp_path / "anticorrelated_channels.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(master_cells, "ANTICORRELATED", path)
    with pytest.raises(SystemExit, match="never read back from the BMRB archive"):
        master_cells.anticorrelated_channels()


def test_a_channel_that_failed_the_archive_silently_is_refused(tmp_path, monkeypatch):
    """Failing to reproduce the archive is allowed, but only if the record says so."""
    payload = json.loads(json.dumps(RECORD))
    payload["channels"][0]["archive_verified"] = False
    payload["channels"][0].pop("archive_status", None)
    path = tmp_path / "anticorrelated_channels.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(master_cells, "ANTICORRELATED", path)
    with pytest.raises(SystemExit, match="carries no archive_status"):
        master_cells.anticorrelated_channels()


def test_the_list_satisfies_the_rule_it_states():
    """The file carries its own evidence, so it can be checked without rerunning anything.

    This checks the record against the rule it declares. It does NOT recompute a
    correlation from the cell data, which does not ship: the per-cell dump lives with the
    run, not in this repository. Re-deriving the list is `benchmarks.dump_paired_cells`
    followed by the script in the campaign's anticorr directory.
    """
    for channel in RECORD["channels"]:
        judged = [r for r in channel["r_independent_judges"].values() if r is not None]
        assert len(judged) >= 3, channel["bmrb"]
        assert all(r < 0 for r in judged), channel["bmrb"]
        assert len(judged) == 4, channel["bmrb"]
        assert channel["n_cells"] >= RECORD["rule"]["minimum_cells_per_channel"]


def test_the_loader_returns_every_channel():
    channels, record = master_cells.anticorrelated_channels()
    assert len(channels) == RECORD["n_channels"] == len(RECORD["channels"])
    assert sum(c["n_cells"] for c in record["channels"]) == RECORD["n_cells"]
