#!/usr/bin/env python3
"""Reflect each excluded C-prime channel about its own mean and score it against the judges.

``reference/anticorrelated_channels.json`` lists the twelve C-prime channels the benchmark
removes because all four independent judges correlate negatively with them. A series that
was deposited upside down about a centre point turns back into an ordinary one when it is
reflected, ``y -> 2 * mean(y) - y``. For each channel this aligns the deposited C-prime
values (the fold-6 cells of every structure instance of that label set, through
``score_validation.fold6_rows``) with the mean prediction of the four judges -- ProCS15,
SPARTA+, UCBShift-X and SHIFTX2; DeepShift is not used -- and records the Pearson r and
the RMSE before and after reflection. The same RMSE over every other fold-6 C-prime
channel with enough cells is the reference: what the judges' mean reaches on a channel
deposited correctly. ``data/README.md`` quotes the reflected RMSEs and that reference.

    python -m benchmarks.cprime_mirror_test --out benchmarks/reference/cprime_mirror_test.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import cells as common
from .score_validation import fold6_rows

HERE = Path(__file__).resolve().parent
RECORD = HERE / "reference" / "anticorrelated_channels.json"
JUDGES = ("procs15", "sparta", "ucbshift_x", "shiftx2_sameprot")
#: Fewer aligned cells than this carry no order worth correlating; the record's own minimum.
MINIMUM = 20


def judge_means(sid: str) -> dict[str, float]:
    """{structure residue label (number and any insertion code): mean C-prime of the judges
    that predicted it}."""
    by_residue: dict[str, list[float]] = {}
    for arm in JUDGES:
        path = common.OUT_ROOT / arm / f"{sid}.json"
        if not path.is_file():
            continue
        for resnum, cell in json.loads(path.read_text()).get("predictions", {}).items():
            if cell.get("C") is not None:
                by_residue.setdefault(str(resnum).strip(), []).append(cell["C"])
    return {k: float(np.mean(v)) for k, v in by_residue.items()}


def channel_cells(label_set, sids, tasks, rows_by_sid, j):
    """Deposited C-prime and the judges' mean on every structure instance of a label set."""
    observed, predicted = [], []
    for sid in sorted(s for s in sids if tasks[s]["label_set_id"] == label_set):
        judged = judge_means(sid)
        for row in rows_by_sid.get(sid, []):
            resnum = str(row["struct_resnum"]).strip()
            if j in row["obs_cols"] and resnum in judged:
                observed.append(row["true_raw"][j])
                predicted.append(judged[resnum])
    return np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)

    with open(common.OUT_ROOT / "fold6_tasks.tsv") as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}
    record = json.loads(RECORD.read_text())
    channels = {c["label_set"]: c for c in record["channels"] if c["nucleus"] == "C"}
    sids = sorted(tasks)
    rows_by_sid, cols, _atom_of_col, _partner = fold6_rows(sids)
    j = cols.index("c_shift")

    out, reference = [], []
    for label_set in sorted({t["label_set_id"] for t in tasks.values()} - set(channels)):
        y, x = channel_cells(label_set, sids, tasks, rows_by_sid, j)
        if len(y) >= MINIMUM:
            reference.append(float(np.sqrt(((y - x) ** 2).mean())))
    for label_set, channel in channels.items():
        y, x = channel_cells(label_set, sids, tasks, rows_by_sid, j)
        if len(y) < MINIMUM:
            continue
        reflected = 2 * y.mean() - y
        out.append({"bmrb": channel["bmrb"], "label_set": label_set, "n_cells": len(y),
                    "r": round(float(np.corrcoef(x, y)[0, 1]), 4),
                    "r_reflected": round(float(np.corrcoef(x, reflected)[0, 1]), 4),
                    "rmse_ppm": round(float(np.sqrt(((y - x) ** 2).mean())), 4),
                    "rmse_reflected_ppm": round(float(np.sqrt(((reflected - x) ** 2).mean())), 4)})
    out.sort(key=lambda c: c["rmse_reflected_ppm"])

    ns.out.write_text(json.dumps({
        "what": "each excluded C-prime channel against the mean of the four judges, as "
                "deposited and reflected about its own mean",
        "judges": list(JUDGES), "deepshift_used": False,
        "cells": "the channel's fold-6 cells on every structure instance of its label set "
                 "that at least one judge predicted",
        "unflagged_reference": {
            "what": "the same RMSE on every other fold-6 C-prime channel (label set) with at "
                    f"least {MINIMUM} aligned cells",
            "n_channels": len(reference),
            "median_rmse_ppm": round(float(np.median(reference)), 4),
            "p10_rmse_ppm": round(float(np.quantile(reference, 0.10)), 4),
            "p90_rmse_ppm": round(float(np.quantile(reference, 0.90)), 4)},
        "n_channels": len(out), "channels": out}, indent=1) + "\n")
    print(f"{len(out)} channels against {len(reference)} unflagged; wrote {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
