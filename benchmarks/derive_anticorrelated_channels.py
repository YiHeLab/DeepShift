#!/usr/bin/env python3
"""Re-derive the anticorrelated channel list from the judges' fold-6 outputs, and check it.

``reference/anticorrelated_channels.json`` is the frozen list ``benchmarks.master_cells``
removes. This applies the rule that record states to the cells it was stated on and
compares the outcome with the record, channel by channel:

  * Universe: the master cell set before the exclusion (``benchmarks.master_cells``'s
    coverage intersection with the channels still in it), backbone cells only.
  * Values: ``score_by_source.build_cohort`` at tier 2, the cells, fence, prochiral
    resolution and residuals every scorer uses. The deposited series is DeepShift's
    resolved target; a judge's value is that target plus the judge's own residual, as
    ``benchmarks.dump_paired_cells`` writes it.
  * Rule: a channel (one label set, one nucleus) with at least the record's minimum of
    cells is excluded when every judge the record names correlates negatively (Pearson)
    with the deposited series over at least that many cells it predicted. DeepShift's
    correlation is recorded and plays no part.

``--fenced`` names the master set the benchmark scores; it must be the universe less the
derived channels. Writes ``--out`` (the derived list and its comparison with the record),
and the per-cell values beside it (``<out>.cells.npz``, ``dump_paired_cells``' format).
Exits non-zero, after writing, when the derived list differs from the record.

    python -m benchmarks.derive_anticorrelated_channels \\
        --master-cells <bench>/master_cells/master_cells.npz \\
        --fenced <bench>/master_cells/master_cells_pf14.npz \\
        --out <dir>/anticorrelated_channels_derived.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import cells as common
from . import master_cells as master
from . import score_by_source_tier as sbt
from .score_by_source import build_cohort

#: The self-exclusion level the retrieval judge is held to: the paper's.
TIER = 2
#: A series flatter than this carries no order to correlate.
FLAT_PPM = 1e-9
#: The frozen record prints each correlation to this many decimals.
RECORD_DECIMALS = 4
REFERENCE = "deepshift"


def channel_correlations(label_set, atom, target, values: dict, minimum: int) -> list[dict]:
    """Per channel of at least ``minimum`` cells, each arm's Pearson r with ``target``.

    An arm's r is taken on the cells it predicted and is None when those are fewer than
    ``minimum`` or flat. A channel whose deposited series is flat is not tested.
    """
    label_set, atom = np.asarray(label_set).astype(str), np.asarray(atom).astype(str)
    target = np.asarray(target, dtype=float)
    keys = np.char.add(np.char.add(label_set, "|"), atom)
    names, inverse = np.unique(keys, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse, minlength=len(names)))[:-1]
    out = []
    for name, index in zip(names, np.split(order, bounds), strict=True):
        if len(index) < minimum or target[index].std() < FLAT_PPM:
            continue
        observed = target[index]
        r = {}
        for arm, v in values.items():
            v = np.asarray(v, dtype=float)[index]
            ok = np.isfinite(v) & np.isfinite(observed)
            r[arm] = (float(np.corrcoef(v[ok], observed[ok])[0, 1])
                      if ok.sum() >= minimum and v[ok].std() > FLAT_PPM else None)
        ls, nucleus = str(name).rsplit("|", 1)
        out.append({"label_set": ls, "nucleus": nucleus, "n_cells": len(index),
                    "index": index, "r": r})
    return out


def excluded(channel: dict, judges) -> bool:
    """The record's rule: every judge has a correlation and every one is negative."""
    return all(channel["r"].get(j) is not None and channel["r"][j] < 0 for j in judges)


def summarize(channel: dict, judges, bmrb: str) -> dict:
    """One excluded channel in the record's form, plus the unrounded correlations."""
    r, d = channel["r"], RECORD_DECIMALS
    return {"bmrb": bmrb, "nucleus": channel["nucleus"], "label_set": channel["label_set"],
            "n_cells": channel["n_cells"],
            "r_independent_judges": {j: round(r[j], d) for j in judges},
            "mean_judge_r": round(float(np.mean([r[j] for j in judges])), d),
            "r_deepshift_not_used_in_rule": (None if r.get(REFERENCE) is None
                                             else round(r[REFERENCE], d)),
            "r_unrounded": r}


def compare(derived: list[dict], record: dict, judges) -> dict:
    """Where the derived channels differ from the record's, to its printed precision."""
    tol = 0.5 * 10 ** -RECORD_DECIMALS + 1e-12
    want = {(c["label_set"], c["nucleus"]): c for c in record["channels"]}
    got = {(c["label_set"], c["nucleus"]): c for c in derived}
    differ = []
    for key in sorted(set(want) & set(got)):
        w, g = want[key], got[key]
        fields = {"bmrb": (w["bmrb"], g["bmrb"]), "n_cells": (w["n_cells"], g["n_cells"])}
        fields |= {f"r_{j}": (w["r_independent_judges"][j], g["r_unrounded"][j])
                   for j in judges}
        fields["mean_judge_r"] = (w["mean_judge_r"],
                                  float(np.mean([g["r_unrounded"][j] for j in judges])))
        if REFERENCE in g["r_unrounded"] and "r_deepshift_not_used_in_rule" in w:
            fields["r_deepshift"] = (w["r_deepshift_not_used_in_rule"],
                                     g["r_unrounded"][REFERENCE])
        bad = {name: {"record": a, "derived": b} for name, (a, b) in fields.items()
               if (abs(a - b) > tol if isinstance(a, float) else a != b)}
        if bad:
            differ.append({"bmrb": w["bmrb"], "nucleus": w["nucleus"], "fields": bad})
    missing = [{"bmrb": want[k]["bmrb"], "nucleus": k[1]} for k in sorted(set(want) - set(got))]
    extra = [{"bmrb": got[k]["bmrb"], "nucleus": k[1], "n_cells": got[k]["n_cells"],
              "r_independent_judges": got[k]["r_independent_judges"]}
             for k in sorted(set(got) - set(want))]
    counts = {"n_channels": (record["n_channels"], len(derived)),
              "n_cells": (record["n_cells"], sum(c["n_cells"] for c in derived))}
    return {"reproduced": not (differ or missing or extra)
            and all(a == b for a, b in counts.values()),
            "precision": f"correlations to the record's {RECORD_DECIMALS} decimals "
                         f"(|derived - record| <= {tol:.6g}); counts exactly",
            "counts": {k: {"record": a, "derived": b} for k, (a, b) in counts.items()},
            "missing_from_derived": missing, "not_in_record": extra,
            "channels_that_differ": differ}


def fence_check(universe, fenced, flagged: set) -> dict:
    """Whether the fenced master set is exactly the universe less the derived channels."""
    nucleus = np.array([c[:-6].upper() for c in universe["atom"]])
    hit = np.array([(ls, n) in flagged for ls, n in zip(universe["label_set"], nucleus,
                                                         strict=True)])
    left = master.master_keys(universe[~hit])
    kept = master.master_keys(fenced)
    return {"n_universe": len(universe), "n_removed": int(hit.sum()), "n_fenced": len(fenced),
            "fenced_is_universe_less_channels": left == kept,
            "n_only_in_universe_less_channels": len(left - kept),
            "n_only_in_fenced": len(kept - left)}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-cells", dest="master_cells", type=Path, required=True,
                    help="the master cell set before the exclusion, channels still in it")
    ap.add_argument("--fenced", type=Path, default=None,
                    help="the master cell set the benchmark scores; checked to be the "
                         "universe less the derived channels")
    ap.add_argument("--record", type=Path, default=master.ANTICORRELATED,
                    help="the frozen channel list to compare with")
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main(argv=None) -> int:
    ns = build_parser().parse_args(argv)
    cells_out = ns.out.with_name(ns.out.name.removesuffix(".json") + ".cells.npz")
    for path in (ns.out, cells_out):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")

    record = json.loads(ns.record.read_text())
    judges = list(record["rule"]["judges"])
    minimum = int(record["rule"]["minimum_cells_per_channel"])
    if REFERENCE in judges:
        raise SystemExit(f"{ns.record} names {REFERENCE} as a judge")
    universe = master.load_master(ns.master_cells, allow_anticorrelated=True)
    if not master.anticorrelated_cells(universe):
        raise SystemExit(f"{ns.master_cells} carries none of the record's channels: it is "
                         "a fenced set, not the universe they were derived on")

    with open(common.OUT_ROOT / "fold6_tasks.tsv") as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}
    sbt.check_arm_policy(judges, [], TIER, tasks)
    cohort = build_cohort(
        argparse.Namespace(tools=judges, leaky=[], master_cells=str(ns.master_cells),
                           allow_anticorrelated=True),
        verify=lambda tool, sid, task, exc: sbt.verify_tiered(tool, sid, task, exc, TIER))

    target = cohort.target[REFERENCE]
    values = {arm: target + cohort.resid[arm] for arm in [*judges, REFERENCE]}
    label_set = np.array([c[1] for c in cohort.cells])
    channels = channel_correlations(label_set, cohort.atom, target, values, minimum)
    bmrb_of = {sid: tasks[sid]["bmrb_id"] for sid in set(cohort.sid_arr)}
    flagged = [summarize(c, judges, bmrb_of[cohort.sid_arr[c["index"][0]]])
               for c in channels if excluded(c, judges)]
    comparison = compare(flagged, record, judges)
    tested = Counter(c["nucleus"] for c in channels)
    by_nucleus = Counter(c["nucleus"] for c in flagged)

    fenced = None
    if ns.fenced:
        fenced = fence_check(universe, master.load_master(ns.fenced),
                             {(c["label_set"], c["nucleus"]) for c in flagged})
        fenced |= {"path": str(ns.fenced)}

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cells_out, arms=np.array([*judges, REFERENCE]), atom=cohort.atom,
        label_set=label_set, residue=cohort.residue,
        resnum=np.array([c[3][0] for c in cohort.cells]), sid=cohort.sid_arr,
        source=cohort.src, bmrb=np.array([bmrb_of[s] for s in cohort.sid_arr]),
        true_ppm=target, **{f"pred_{arm}": v for arm, v in values.items()})
    report = {
        "what": "the anticorrelated channel list re-derived from the judges' fold-6 "
                "outputs, and its comparison with the frozen record",
        "rule": record["rule"]["statement"], "judges": judges,
        "minimum_cells_per_channel": minimum,
        "deepshift_is_not_a_judge": True,
        "record": {"path": str(ns.record)},
        "universe": {"master_cells": str(ns.master_cells),
                     "n_backbone_cells": len(cohort.cells),
                     "n_structures": len(cohort.common_sids),
                     "cells": "score_by_source.build_cohort at tier 2 on these master cells: "
                              "backbone, fenced, per-arm prochiral resolution; target = "
                              "DeepShift's resolved target, judge = target + its residual"},
        "population": {"n_channels_tested": len(channels),
                       "channels_tested_by_nucleus": dict(sorted(tested.items())),
                       "test": f"channels of at least {minimum} cells whose deposited "
                               "series is not flat"},
        "n_channels": len(flagged), "n_cells": sum(c["n_cells"] for c in flagged),
        "channels_by_nucleus": dict(sorted(by_nucleus.items())),
        "channels": flagged,
        "matches_record": comparison,
        "fenced_master_cells": fenced,
        "cells_dump": cells_out.name,
    }
    ns.out.write_text(json.dumps(report, indent=1) + "\n")

    print(f"wrote {ns.out}: {report['n_channels']} channels, {report['n_cells']:,} cells "
          f"{report['channels_by_nucleus']} of {len(channels):,} tested")
    if fenced is not None:
        print(f"  fenced set is the universe less these channels: "
              f"{fenced['fenced_is_universe_less_channels']}")
    if not comparison["reproduced"]:
        print("the derived list differs from the frozen record:\n"
              + json.dumps({k: comparison[k] for k in ("counts", "missing_from_derived",
                                                       "not_in_record",
                                                       "channels_that_differ")}, indent=1),
              file=sys.stderr)
        return 1
    print(f"  reproduces {ns.record.name}: same {record['n_channels']} channels, same cells, "
          f"every correlation to {RECORD_DECIMALS} decimals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
