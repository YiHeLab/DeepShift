#!/usr/bin/env python3
"""The master cell sets: one frozen cell set per nucleus that every covering arm scored.

The rule, per nucleus:

  * ``U(nuc)`` is the universe: the cells of that nucleus DeepShift can score on held-out
    fold 6, after the run's composite p100 + structural fence, the ambiguity drop and
    prochiral resolution. It is what ``benchmarks.coverage_census`` measures.
  * ``coverage(arm, nuc)`` is the share of ``U(nuc)`` the arm emitted a finite value for.
  * An arm at or above :data:`MASTER_COVERAGE_FLOOR` **joins**, and ``Master(nuc)`` is the
    cells of ``U(nuc)`` every joining arm emitted.
  * An arm below the floor never shrinks ``Master(nuc)``. It is scored on
    ``Master(nuc)`` intersected with its own cells, and every row it appears in prints
    that cell count and carries a marker.

The set is derived once from the census and frozen, so every predictor-comparison float
on fold 6 reads the same cells from the same file instead of intersecting its own arms.

A producer handed a master file joins its own cells to it on (nucleus column, structure
instance, residue number) through :func:`cell_keys` and :func:`restrict`, which fail
closed on an empty intersection. A producer handed none falls back to its own arms'
intersection and says so; nothing here invents a cohort.

    python -m benchmarks.master_cells --census <dir>/coverage_census.json \\
        --out <dir>/master_cells.npz
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from benchmarks import config
from deepshift.chemistry import atom_from_shift_column

from . import cells as common
from .config import BENCHMARK_ARMS, arm_exclusion_tier

HERE = Path(__file__).resolve().parent
ANTICORRELATED = HERE / "reference" / "anticorrelated_channels.json"

#: Where ``benchmarks.coverage_census`` writes the census the master cells are frozen
#: from: beside them, under the benchmark outputs. ``DEEPSHIFT_COVERAGE_CENSUS`` names a
#: census written elsewhere, so a re-run census need not overwrite the frozen one.
CENSUS_VAR = "DEEPSHIFT_COVERAGE_CENSUS"
CENSUS_PATH = Path(os.environ.get(CENSUS_VAR)
                   or common.OUT_ROOT / "master_cells" / "coverage_census.json")

#: An arm covering at least this share of a nucleus's universe joins that nucleus's
#: master set. A predictor that fails on a tenth of the cells is still
#: reported, on the cells it did produce, and it does not take the others down with it.
MASTER_COVERAGE_FLOOR = 0.90

#: One record per (nucleus, cell). ``atom`` is the nucleus column (``ca_shift``),
#: ``sid`` the structure instance, ``residue`` its structure residue number; the three
#: together name the cell. ``label_set`` is the deposition the bootstrap resamples.
MASTER_COLUMNS = ("atom", "sid", "label_set", "residue")


_ATOM_COLUMN = {"C": "c_shift", "H": "h_shift", "HA": "ha_shift",
                "CA": "ca_shift", "CB": "cb_shift", "N": "n_shift"}


def anticorrelated_cells(frame) -> int:
    """How many rows of ``frame`` sit in a channel Methods excludes."""
    channels = json.loads(ANTICORRELATED.read_text())["channels"]
    hit = 0
    for channel in channels:
        column = _ATOM_COLUMN[channel["nucleus"]]
        hit += int(((frame["label_set"] == channel["label_set"])
                    & (frame["atom"] == column)).sum())
    return hit


def load_master(path, *, allow_anticorrelated=False):
    """The frozen master cells as a DataFrame of :data:`MASTER_COLUMNS`.

    A file missing one of the columns is refused by name rather than loaded into a
    cohort that silently lacks a coordinate of the cell.

    A file that still carries the anticorrelated channels is refused too. Those cells are
    excluded by Methods, and a producer handed the unfenced set scores them without
    saying so -- which is how Figure 2(a), Figure S4, Figure 4 and Table S23 came to be
    drawn on a cohort the benchmark tables do not use. ``allow_anticorrelated`` is for the
    one legitimate case, a deliberate comparison of the two cohorts, and has to be asked
    for by name.
    """
    import pandas as pd

    with np.load(path, allow_pickle=False) as dump:
        missing = [name for name in MASTER_COLUMNS if name not in dump.files]
        if missing:
            raise SystemExit(f"{path} carries no {', '.join(missing)}: it is not a "
                             f"master cell file ({', '.join(sorted(dump.files))})")
        frame = pd.DataFrame({
            "atom": dump["atom"].astype(str), "sid": dump["sid"].astype(str),
            "label_set": dump["label_set"].astype(str),
            "residue": dump["residue"].astype(np.int64),
        })
    if not allow_anticorrelated:
        carried = anticorrelated_cells(frame)
        if carried:
            raise SystemExit(
                f"{path} still carries {carried:,} cells of the anticorrelated channels "
                "Methods excludes. A float drawn on these does not match the benchmark "
                "tables. Use the fenced master cell set, or pass allow_anticorrelated "
                "when the unfenced cohort is the point of the measurement.")
    return frame


def cell_keys(atom, sid, residue) -> np.ndarray:
    """``(nucleus column, structure instance, residue number)`` as one string per cell.

    The three together name a cell; the label set is context a cell already implies.
    """
    return np.array([f"{a}|{s}|{r}" for a, s, r in
                     zip(np.asarray(atom).astype(str), np.asarray(sid).astype(str),
                         np.asarray(residue).astype(str))])


def master_keys(frame) -> set:
    """The master cells as the keys :func:`cell_keys` builds."""
    return set(cell_keys(frame["atom"], frame["sid"], frame["residue"]).tolist())


def restrict(keys: np.ndarray, master, *, what: str) -> np.ndarray:
    """A boolean mask keeping the cells of ``keys`` the master set names.

    An empty intersection is a wrong file or a wrong key, never a cohort of no cells, so
    it stops the producer instead of emptying a float.
    """
    kept = np.isin(keys, list(master_keys(master)))
    if not kept.any():
        raise SystemExit(
            f"the master cell file names none of the {len(keys):,} {what} cells: check "
            "that its `atom` column holds nucleus columns (ca_shift) and its `residue` "
            "column residue numbers of these structure instances")
    return kept


def master_index(frame, cols) -> dict:
    """``{structure_id: {(struct_resnum, column position)}}`` for a scorer's cell test.

    ``cols`` is the prediction dump's column list, so a master set naming a nucleus this
    cache does not carry is refused here rather than silently scored on fewer cells.
    """
    position = {column: index for index, column in enumerate(cols)}
    unknown = sorted(set(frame["atom"]) - set(position))
    if unknown:
        raise SystemExit(f"the master cells name nuclei this cache has no column for: {unknown}")
    index: dict = {}
    for atom, sid, residue in zip(frame["atom"], frame["sid"], frame["residue"]):
        index.setdefault(sid, set()).add((str(residue), position[atom]))
    return index


def arms_returning(*columns: str, tiers=(None, 2), census=None) -> list:
    """The reported arms that return any of these nuclei, from the one roster in config.

    Which nuclei an arm returns is a measurement of this run, so it is read from the
    coverage census; with no census on disk the whole roster at those exclusion levels
    is returned and the producer reduces it by its own coverage. ``tiers`` names the
    exclusion levels the float reports (:func:`config.arm_exclusion_tier`): the default
    is the leaky arms and the same-protein arms every table prints, and ``(2,)`` drops
    the leaky ones. No producer keeps a second copy of the roster.
    """
    arms = [arm for arm in BENCHMARK_ARMS if arm_exclusion_tier(arm) in tiers]
    path = Path(census) if census is not None else CENSUS_PATH
    if not path.exists():
        return arms
    measured = json.loads(path.read_text())["arms"]
    return [arm for arm in arms
            if any(measured.get(arm, {}).get("by_nucleus", {}).get(column, {})
                   .get("coverage", 0.0) > 0 for column in columns)]


def report_path(master_npz) -> Path:
    """The report frozen beside one master cell set."""
    return Path(str(master_npz).replace(".npz", "") + ".provenance.json")


def read_report(master_npz) -> dict:
    return json.loads(report_path(master_npz).read_text())


def below_coverage(report) -> dict:
    """``{arm: {atom name}}`` -- where an arm fell below the floor and carries a marker."""
    out: dict = {}
    for column, node in report["by_nucleus"].items():
        for arm in node["below_floor"]:
            out.setdefault(arm, set()).add(atom_from_shift_column(column))
    return out


# ----------------------------- deriving the set -----------------------------

def joining_arms(census: dict, column: str) -> list:
    """The arms that cover enough of one nucleus to join its master set.

    An arm in :data:`config.MASTER_MARKED_ARMS` never joins: it is scored on the master
    cells it emitted and its rows carry the marker, however high its coverage.
    """
    return [arm for arm, node in census["arms"].items()
            if arm not in config.MASTER_MARKED_ARMS
            and node["by_nucleus"].get(column, {}).get("coverage", 0.0) >= MASTER_COVERAGE_FLOOR]


def anticorrelated_channels() -> tuple[set, dict]:
    """(label set, nucleus) channels whose DEPOSITED values run backwards.

    A range fence cannot catch these: the values are inside the nucleus's physical range
    and only their ORDER is wrong, so the deposited series is negatively correlated with
    what any predictor computes from the structure. The channels are named in
    ``reference/anticorrelated_channels.json``, which records the rule, the correlation
    each independent judge measured, and the BMRB archive read-back. The read-back shows
    this package reproduces the deposited record; it does not by itself show the deposited
    values are wrong. DeepShift is not one of the judges.
    """
    record = json.loads(ANTICORRELATED.read_text())
    unchecked = [c for c in record["channels"] if not c.get("archive_checked")]
    if unchecked:
        raise SystemExit(
            "REFUSING: anticorrelated_channels.json lists channels that were never read "
            f"back from the BMRB archive: {[(c['bmrb'], c['nucleus']) for c in unchecked]}. "
            "A channel is excluded only once the deposited record itself has been checked.")
    for channel in record["channels"]:
        if not channel.get("archive_verified") and not channel.get("archive_status"):
            raise SystemExit(
                f"REFUSING: {channel['bmrb']} {channel['nucleus']} did not reproduce the "
                "archive and carries no archive_status saying what was found.")
    return {(c["label_set"], c["nucleus"]) for c in record["channels"]}, record


def derive(census: dict, coverage_npz, *, keep_anticorrelated: bool = False
           ) -> tuple[dict, dict, dict]:
    """Intersect the joining arms per nucleus, then drop the anti-correlated channels
    (unless ``keep_anticorrelated``: the universe the channels are derived on, and the
    cohort the without-exclusions comparison is scored on).
    -> (master arrays, per-nucleus record, exclusion record)."""
    with np.load(coverage_npz, allow_pickle=False) as dump:
        arms = [str(a) for a in dump["arms"]]
        covered = dump["covered"]
        universe = {name: dump[name] for name in MASTER_COLUMNS}
    row_of = {arm: index for index, arm in enumerate(arms)}

    atom = universe["atom"].astype(str)
    keep = np.zeros(len(atom), dtype=bool)
    by_nucleus = {}
    for column in sorted(set(atom)):
        cells = atom == column
        joined = joining_arms(census, column)
        inside = cells.copy()
        for arm in joined:
            inside &= covered[row_of[arm]]
        keep |= inside
        by_nucleus[column] = {
            "n_universe": int(cells.sum()), "n_master": int(inside.sum()),
            "joined": joined,
            "below_floor": {arm: census["arms"][arm]["by_nucleus"].get(column, {}).get("coverage", 0.0)
                            for arm in arms if arm not in joined},
        }

    # The deposited-order defect, removed last so `by_nucleus` still reports the coverage
    # intersection and the exclusion is a separate step.
    channels, record = anticorrelated_channels()
    nucleus = np.array([atom_from_shift_column(c) for c in atom])
    label_set = universe["label_set"].astype(str)
    bad = np.zeros(len(atom), dtype=bool)
    per_channel = []
    for key in sorted(channels):
        hit = keep & (label_set == key[0]) & (nucleus == key[1])
        per_channel.append({"label_set": key[0], "nucleus": key[1], "n_removed": int(hit.sum())})
        bad |= hit
    if keep_anticorrelated:
        bad[:] = False
    keep &= ~bad
    for column, node in by_nucleus.items():
        removed = int((bad & (atom == column)).sum())
        node["n_removed_anticorrelated"] = removed
        node["n_master"] = node["n_master"] - removed
    excluded = {"rule": record["rule"], "precedent": record["precedent"],
                "verification": record["verification"],
                "source": str(ANTICORRELATED),
                "n_channels": len(channels), "n_cells_removed": int(bad.sum()),
                "per_channel": per_channel}
    return {name: values[keep] for name, values in universe.items()}, by_nucleus, excluded


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", type=Path, required=True,
                    help="coverage_census.json written by benchmarks.coverage_census")
    ap.add_argument("--coverage", type=Path, default=None,
                    help="arm_coverage.npz beside the census (default: the census's own)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-anticorrelated", action="store_true",
                    help="write the set before the anticorrelated channels are removed: "
                         "the universe they are derived on, never a set the tables score")
    ns = ap.parse_args(argv)
    if ns.out.exists():
        raise SystemExit(f"refusing to overwrite a frozen cell set: {ns.out}")

    census = json.loads(ns.census.read_text())
    coverage_npz = ns.coverage or ns.census.with_name(census["coverage_npz"])
    master, by_nucleus, excluded = derive(census, coverage_npz,
                                          keep_anticorrelated=ns.keep_anticorrelated)
    if ns.keep_anticorrelated:
        excluded = {**excluded, "n_cells_removed": 0,
                    "kept": "--keep-anticorrelated: the channels below are still in this "
                            "set; n_removed is what removing them would take out"}

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ns.out, **master)
    report_path(ns.out).write_text(json.dumps({
        "what": ("the master cell set BEFORE the anticorrelated channels are removed: the "
                 "universe they are derived on and the without-exclusions comparison's "
                 "cohort; no table scores it" if ns.keep_anticorrelated else
                 "the cell set every predictor-comparison float on held-out fold 6 scores on, "
                 "one per nucleus, frozen"),
        "rule": {
            "universe": census["universe_rule"],
            "coverage": "share of the nucleus's universe an arm emitted a finite value for",
            "floor": MASTER_COVERAGE_FLOOR,
            "join": f"an arm at or above {MASTER_COVERAGE_FLOOR:.2f} joins the nucleus; the "
                    "master set is the cells every joining arm emitted",
            "below_floor": "an arm below the floor does not shrink the master set: it is "
                           "scored on the master set intersected with its own cells, and "
                           "every row it appears in prints that cell count and is marked",
            "anticorrelated": "channels whose deposited values are negatively correlated "
                              "with independent predictions are then removed; see the "
                              "anticorrelated_channels block",
        },
        "anticorrelated_channels": excluded,
        "arms": census["arms_scanned"],
        "n_cells": len(master["atom"]),
        "n_nuclei": len(by_nucleus),
        "by_nucleus": by_nucleus,
        "coverage_census": census,
    }, indent=1, default=str) + "\n")

    backbone = [f"{atom_from_shift_column(c)}={n['n_master']:,}/{n['n_universe']:,}"
                for c, n in by_nucleus.items() if atom_from_shift_column(c) in
                ("N", "H", "CA", "CB", "C", "HA")]
    print(f"wrote {ns.out}: {len(master['atom']):,} cells over {len(by_nucleus)} nuclei")
    print(f"  anti-correlated channels removed: {excluded['n_channels']} "
          f"({excluded['n_cells_removed']:,} cells)")
    print("  backbone master/universe: " + "  ".join(sorted(backbone)))


if __name__ == "__main__":
    main()
