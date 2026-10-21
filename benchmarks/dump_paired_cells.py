"""Write the per-cell fold-6 dump the homology and self-lookup figures read.

The format is the one ``deepshift.cohorts.load_paired_cells`` reads (``arms``, ``atom``,
``label_set``, ``residue``, ``sid``, ``source``, ``bmrb``, ``true_ppm``, ``pred_<arm>``),
plus ``resnum``, the residue number that with ``sid`` and the nucleus names a cell -- what
a master cell file is joined on.

The cohort is ``score_by_source.build_cohort``: the paper ensemble against the named
competitor arms on exactly the backbone cells every one of them predicted, after the
fence and per-arm prochiral resolution, so the dump cannot compare different proteins.
With ``--master-cells`` those cells are drawn from the frozen master set instead of this
pair's own intersection, and a cell an arm of the pair did not emit leaves the dump: a
paired dump is two arms on one cell set, so it is the master set intersected with both.
``--tier`` names the self-exclusion level the retrieval arms are held to and nothing
else; ``--leaky`` admits an arm scored with no exclusion at all, on the same cells.

``true_ppm`` is the paper ensemble's resolved target; each competitor's ``pred_<arm>`` is
stored as ``true_ppm + its own residual`` so that ``pred_<arm> - true_ppm`` is exactly the
residual the benchmark scored for that arm (prochiral resolution is per arm).

    python -m benchmarks.dump_paired_cells --tools procs15 --out <root>/ds_procs15_cells.npz
    python -m benchmarks.dump_paired_cells --tier 2 --leaky shiftx2_leaky \
        --tools shiftx2_leaky shiftx2_sameprot shiftx2_noshifty --out <root>/selflookup.npz
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from . import cells as common
from .score_by_source import build_cohort
from .score_by_source_tier import LEVEL_BY_TIER, verify_tiered


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", nargs="+", required=True)
    ap.add_argument("--leaky", nargs="*", default=[],
                    help="arms scored without the self-exclusion drop, and so leaky")
    ap.add_argument("--tier", type=int, choices=sorted(LEVEL_BY_TIER), default=2,
                    help="the self-exclusion level the enforced retrieval arms are held "
                         "to; 2 is the exclusion the paper reports")
    ap.add_argument("--master-cells", dest="master_cells", default=None,
                    help="the frozen master cell set (benchmarks.master_cells)")
    ap.add_argument("--out", required=True)
    ns = ap.parse_args(argv)
    ns.extra = []

    cohort = build_cohort(ns, verify=lambda tool, sid, task, exc:
                          verify_tiered(tool, sid, task, exc, ns.tier))
    # Both arms on one cell set: on the master cells, drop the cells either arm is silent on.
    paired = np.all([np.isfinite(cohort.resid[arm]) for arm in cohort.names], axis=0)
    with open(common.OUT_ROOT / "fold6_tasks.tsv") as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}
    true = cohort.target["deepshift"][paired]
    label_set = np.array([c[1] for c in cohort.cells])[paired]
    resnum = np.array([c[3][0] for c in cohort.cells])[paired]
    bmrb = np.array([tasks[s]["bmrb_id"] for s in cohort.sid_arr])[paired]
    arms = list(cohort.names)
    payload = {
        "arms": np.array(arms), "atom": cohort.atom[paired], "label_set": label_set,
        "residue": cohort.residue[paired], "resnum": resnum, "sid": cohort.sid_arr[paired],
        "source": cohort.src[paired], "bmrb": bmrb, "true_ppm": true,
    }
    for arm in arms:
        payload[f"pred_{arm}"] = true + cohort.resid[arm][paired]
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"wrote {out}: {int(paired.sum()):,} cells, {len(set(cohort.sid_arr[paired]))} "
          f"structures, arms {arms}")


if __name__ == "__main__":
    main()
