#!/usr/bin/env python3
"""Why each label set did or did not receive a rereferencing offset.

Re-runs the reref stage's structure selection and offset arithmetic for every label set
of one build, with the release functions of ``deepshift.data.reref``, and records the
outcome of each and, per candidate structure, the unrounded offset of each Calpha, Cbeta,
C' and N fitted on it with the number of residues it was fitted over. The recomputed offsets must equal the stored ``reref_offsets.json``
record for record (method, offsets, structure); otherwise nothing is written. Writes
``reref_diagnostics.json`` beside the offsets (``--out`` names another file), which the
coverage table of ``si_build/si_reref_offsets.py`` reads.

    DEEPSHIFT_RUN=/path/to/run SHIFTX_BIN=/path/to/shiftx python scripts/reref_diagnostics.py

Outcomes, one per label set, decided in this order:

  no_structure          not realized into any source, or no built structure to compare with
  no_correctable_shift  no Calpha, Cbeta, C' or N shift deposited on an aligned residue
                        of any source
  offset                a non-zero offset on at least one of Calpha, Cbeta, C' and N
  rounds_to_zero        three or more residues fit an offset, which rounds to zero on
                        the 0.05 ppm grid (for Calpha/Cbeta, their joint mean does)
  too_few_residues      a Calpha, Cbeta, C' or N shift is deposited, but fewer than five
                        residues align to any structure, the alignment identity is below
                        0.8, or fewer than three residues of each nucleus fall inside the
                        agreement window
  shiftx_failed         SHIFTX produced no prediction for any built structure
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepshift.data import reref as R

SOURCES = ("af", "nmr", "xtal")
# How far a candidate structure got through the stage's compute_offset, least to most.
STAGES = ("no_rows", "empty_body", "no_structure", "shiftx_failed", "alignment", "window",
          "candidate")
FOLD_MAP = "fold_assignments_labelset_identity90.json"


def load(data: Path):
    """Each (label set, source) body with its built structure, as entity_reref_main reads them."""
    bodies = {}
    for source in SOURCES:
        master = pd.read_csv(data / f"structure_data_{source}.csv", low_memory=False,
                             dtype={"label_set_id": str, "bmrb_id": str})
        for lsid, rows in master.groupby("label_set_id"):
            bodies[(lsid, source)] = (R._label_set_body(rows),
                                      str(rows["materialization_id"].iloc[0]))
    built = {}
    for source in SOURCES:
        log = pd.read_csv(data / f"structure_build_log_{source}.csv", dtype=str)
        for _, row in log[log["status"] == "success"].iterrows():
            built[str(row["materialization_id"])] = (row["pdb_path"], str(row["chain_id"]))
    return bodies, built


def shiftx(structure):
    pdb, chain = structure
    if not os.path.isfile(pdb):
        return {}
    kept = R.keepA_chain(pdb, want_chain=chain)
    try:
        return R.run_shiftx(kept, chain, return_flags=True)[0]
    finally:
        os.remove(kept)


def correctable(body) -> bool:
    return any(n in values for _aa, values in body.values() for n in R.REREF_APPLY_NUCS)


def diagnose(lsid, bodies, built, predictions, cfg):
    """(record as the stage would store it, outcome, detail, fits) for one label set.

    ``fits`` holds, per candidate source, the unrounded mean(SHIFTX - observed) of each
    Calpha, Cbeta, C' and N fitted on it and the in-window residues it was fitted over."""
    reached, candidates = {}, []
    for source in R._REREF_SOURCE_PREFERENCE:
        body, mzid = bodies.get((lsid, source), (None, None))
        if body is None:
            reached[source] = ("no_rows", None)
        elif not body:
            reached[source] = ("empty_body", body)
        elif mzid not in built:
            reached[source] = ("no_structure", body)
        elif not predictions[built[mzid]]:
            reached[source] = ("shiftx_failed", body)
        else:
            overall, per, frac = R.eval_structure(body, predictions[built[mzid]])
            if frac < 0.8:
                reached[source] = ("alignment", body, frac)
            elif not per:
                reached[source] = ("window", body)
            else:
                reached[source] = ("candidate", body)
                candidates.append({"source": source, "struct": built[mzid], "per": per,
                                   "overall": overall, "frac": frac})
    if candidates:
        chosen = R.select_reref_structure(candidates, cfg)
        offsets = R.offsets_for_method(chosen["per"], cfg)
        record = {"method": "computed", "offsets": offsets,
                  "struct_used": f"{chosen['source']}:{chosen['struct'][1]}"}
    else:
        record = {"method": "none", "offsets": {}, "struct_used": None}
    fits = {c["source"]: {n: {"offset_ppm": c["per"][n]["avgdiff"], "residues": c["per"][n]["n"]}
                          for n in R.REREF_APPLY_NUCS if n in c["per"]}
            for c in candidates}
    return (record, *outcome(reached, candidates, record, chosen if candidates else None), fits)


def outcome(reached, candidates, record, chosen):
    """(outcome, detail): whether a correctable shift exists is decided before how far
    alignment got."""
    realized = [r[1] for r in reached.values() if r[0] != "no_rows"]
    if not realized:
        return "no_structure", "not_realized"
    if not any(correctable(body) for body in realized):
        return "no_correctable_shift", ("protons_only" if any(realized) else "no_backbone_shift")
    if candidates:
        if record["offsets"]:
            return "offset", None
        if any(n in chosen["per"] for n in R.REREF_APPLY_NUCS):
            return "rounds_to_zero", None
    furthest = max(reached.values(), key=lambda r: STAGES.index(r[0]))
    stage = furthest[0]
    if stage in ("no_structure", "shiftx_failed"):
        return stage, None
    if stage == "alignment":
        return "too_few_residues", ("fewer_than_5_aligned" if furthest[2] == 0.0
                                    else "identity_below_0.8")
    return "too_few_residues", "window"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=Path(os.environ.get("DEEPSHIFT_RUN", "__DEEPSHIFT_RUN_REQUIRED__"))
                    / "data", help="the build's data directory (default: $DEEPSHIFT_RUN/data)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write (default: reref_diagnostics.json in --data)")
    ap.add_argument("--workers", type=int, default=12, help="SHIFTX threads")
    args = ap.parse_args(argv)

    offsets_path = args.data / "reref_offsets.json"
    stored = json.loads(offsets_path.read_text())
    cfg = R.RULESEARCH_WINNER_DEFAULT
    realized = set(json.loads((args.data / FOLD_MAP).read_text()))

    bodies, built = load(args.data)
    wanted = sorted({built[mzid] for (lsid, _s), (body, mzid) in bodies.items()
                     if lsid in stored and body and mzid in built})
    print(f"SHIFTX on {len(wanted)} structures, {args.workers} threads", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        predictions = dict(zip(wanted, pool.map(shiftx, wanted)))

    per_label_set, mismatches = {}, []
    for lsid in sorted(stored):
        record, result, detail, fits = diagnose(lsid, bodies, built, predictions, cfg)
        if record != {k: stored[lsid].get(k) for k in record}:
            mismatches.append(lsid)
        per_label_set[lsid] = {"outcome": result, "detail": detail,
                               "struct_used": record["struct_used"], "fits": fits}
    if mismatches:
        raise SystemExit(f"{len(mismatches)} label sets do not reproduce the stored offsets, "
                         f"e.g. {mismatches[:5]}; nothing written")

    def counts(keys):
        return {"outcome": dict(Counter(per_label_set[k]["outcome"] for k in keys)),
                "detail": dict(Counter(f"{per_label_set[k]['outcome']}/{per_label_set[k]['detail']}"
                                       for k in keys))}

    out = args.out or args.data / "reref_diagnostics.json"
    out.write_text(json.dumps({
        "reproduces_stored_offsets": len(stored),
        "all": counts(stored),
        "realized": counts(sorted(realized & set(stored))),
        "per_label_set": per_label_set,
    }, indent=1) + "\n")
    print(f"{len(stored)} label sets reproduce the stored offsets -> {out}")
    print(json.dumps(counts(sorted(realized & set(stored))), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
