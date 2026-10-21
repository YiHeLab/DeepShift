"""Every arm's backbone residuals on the crystal + AlphaFold pool of fold 6, on one common
cell set, each retrieval arm verified at its own exclusion level.

Feeds two supplementary tables: gross-error rates at fixed thresholds (S16) and the
exclusion ladder (S19). The cohort rule is ``score_by_source.build_cohort`` (a cell is
scored only if every listed arm predicted it, or, with ``--master-cells``, the frozen
master cell set); NMR conformers are dropped afterwards, as those tables state. An arm
named ``*_leaky`` is scored with no exclusion enforced, ``*_sameprot`` at the reported
level (tier 2), ``*_selfexcl`` at the stricter level (tier 3), and a structure-only arm
has nothing to verify.

On the master cells an arm below the coverage floor is scored on the master cells it
emitted: its ``n_cells`` sits beside its rates and its ``below_master_coverage`` flag is
set, so a rate read against a smaller cell set is never read as one against the whole.

    python -m benchmarks.score_pool_af_xtal --leaky ucbshift_full_leaky shiftx2_leaky \\
        --tools <arms...> --out <scored>/pool_af_xtal.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import cells as common
from . import config
from . import score_by_source_tier as sbt
from .score_by_source import BACKBONE, build_cohort

#: The pool both tables state: crystal structures and AlphaFold models.
POOL_SOURCES = ["xtal", "af"]
THRESHOLDS = {"HA": (0.5, 1.0), "H": (0.5, 1.0), "CA": (2.0, 5.0), "CB": (2.0, 5.0),
              "C": (2.0, 5.0), "N": (5.0, 10.0)}


def tier_of(tool: str) -> int:
    """The exclusion level a non-leaky arm is verified at, read off its staged name."""
    return config.arm_exclusion_tier(tool)


#: Shifts are stored to a few decimals, so a residual can sit exactly on a threshold (a
#: 0.50 ppm miss). The targets come from the cache as float32, which moves such a tie by up
#: to ~8e-6 ppm either side (half a float32 step at 200 ppm). "Greater than" leaves the tie
#: out: the tolerance is above that rounding and far below any stored digit (0.001 ppm).
GROSS_TIE_TOLERANCE_PPM = 1e-5


def gross(residuals, threshold):
    """|residual| > threshold in the stored decimals: a residual on the threshold is not gross."""
    return np.abs(residuals) > threshold + GROSS_TIE_TOLERANCE_PPM


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", nargs="+", required=True)
    ap.add_argument("--leaky", nargs="*", default=[])
    ap.add_argument("--master-cells", dest="master_cells", default=None,
                    help="the frozen master cell set (benchmarks.master_cells)")
    ap.add_argument("--out", required=True)
    ns = ap.parse_args(argv)
    ns.extra = []
    leaky = set(ns.leaky)

    def verify(tool, sid, task, exc):
        if tool in leaky:
            return True, "leaky arm: no exclusion enforced"
        return sbt.verify_tiered(tool, sid, task, exc, tier_of(tool))

    cohort = build_cohort(ns, verify=verify)
    keep = np.isin(cohort.src, POOL_SOURCES)
    sids_kept = sorted(set(cohort.sid_arr[keep]))
    with open(common.OUT_ROOT / "fold6_tasks.tsv") as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}
    by_source = {s: sum(1 for x in sids_kept if tasks[x]["source"] == s) for s in POOL_SOURCES}

    per_nucleus = {}
    for atom in BACKBONE:
        m = keep & (cohort.atom == atom)
        t1, t2 = THRESHOLDS[atom]
        entry = {"n_cells": int(m.sum()), "thresholds_ppm": [t1, t2], "arms": {}}
        for arm in cohort.names:
            r = cohort.resid[arm][m]
            r = r[np.isfinite(r)]        # the master cells this arm emitted, and no others
            if not r.size:
                continue                 # an arm with no cell of this nucleus has no entry
            entry["arms"][arm] = {
                "n_cells": int(r.size),
                "below_master_coverage": cohort.flagged(arm, atom),
                "rmse": float(np.sqrt(np.mean(r ** 2))), "mae": float(np.mean(np.abs(r))),
                "gross_error_pct": {str(t1): 100.0 * float(np.mean(gross(r, t1))),
                                    str(t2): 100.0 * float(np.mean(gross(r, t2)))}}
        per_nucleus[atom] = entry

    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "what": "backbone residuals of every listed arm on the crystal + AlphaFold pool of "
                "fold 6, one cell set (score_by_source.build_cohort), raw residuals",
        "sources": POOL_SOURCES, "n_structures": len(sids_kept), "n_structures_by_source": by_source,
        "n_cells": int(keep.sum()), "arms": list(cohort.names),
        "cohort_rule": cohort.cohort_rule, "master_cells": cohort.master_cells,
        "arms_below_master_coverage": {a: sorted(n) for a, n in
                                       (cohort.below_master or {}).items()},
        "exclusion_level_by_arm": {a: ("none" if a in leaky else f"tier {tier_of(a)}")
                                   for a in cohort.names},
        "per_nucleus": per_nucleus,
    }, indent=1, default=str) + "\n")
    print(f"wrote {out}: {len(sids_kept)} structures {by_source}, {int(keep.sum()):,} cells, "
          f"{len(cohort.names)} arms")


if __name__ == "__main__":
    main()
