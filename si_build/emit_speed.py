#!/usr/bin/env python3
"""Emit tab:inference_speed, timing every method the same way.

A per-file command-line tool pays interpreter start, library import and model load once
per structure. A resident process pays them once for the whole cohort. Timing one method
warm and the others cold and sorting the result into one list compares two different
quantities, so this emits two blocks against one clock definition:

  One protein per invocation -- every method, ours included, pays its own startup. This
  is the like-for-like comparison and the honest answer to "how long to predict one
  protein". The DeepShift rows spawn the release's `deepshift predict` on the shipped
  checkpoints once per structure.

  Batch -- one cohort pass of all 203 chains. Every tool that ships a batch interface is
  driven through it over the same cohort, so no competitor number is withheld; a tool
  whose multi-file mode reads its inputs as frames of one system has no batch lane and no
  row. DeepShift's two rows are the throughput arm (run_speed_throughput.py): one
  invocation, an 8-process featurization pool, residues of different proteins packed
  into shared GPU batches of 256, clocked from process launch to the last CSV on disk.
  SPARTA+, LEGOLAS and UCBShift-X hydrogenate with REDUCE, and run it over the same 8
  workers (run_legolas_parallel_prep.py, run_batch_parallel_prep.py). Every batch row is
  charged the same way: total_sec = preparation + tool (for DeepShift, launch to last
  output), over 203 chains for seconds per protein and over the census residue count for
  milliseconds per residue. Repeated passes print their mean, and each pass ran on an
  idle machine (1-minute load below 1.0 for three minutes, no GPU process).

Per-invocation columns are seconds per protein (mean over structures, bootstrap 95 %
interval), the startup (intercept of wall clock on residue count, constrained >= 0) and
the marginal cost per residue (slope of the same fit). Batch columns are seconds per
protein and milliseconds per residue, both from one cohort total; startup is not
separable from one pass and prints '---'.

The caption is the authors', from ``si_build/reference_inputs/captions/``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import si_common

# The measured census is a finished measurement held in the
# archive; SI_SPEED_CENSUS names another copy of it.
CENSUS = Path(os.environ.get("SI_SPEED_CENSUS",
                             str(si_common.archive("speed_census"))))
OUT = si_common.OUT
# The authors' caption, as for every table: si_common.CAPTIONS, never regenerated.
CAPTION = si_common.CAPTIONS / "si_tables_speed.tex"

# Per-invocation block. Competitors are cold by construction; the two DeepShift rows are
# the coldproc arms on the shipped checkpoints, which spawn `deepshift predict` per
# structure exactly as the competitor adapters spawn their own tools. Methods measured in more than one
# configuration appear at their fastest, which is the reading most favourable to them.
# A label of None prints the census's own method name, which states the configuration.
COLD = [
    ("sparta", "SPARTA+"),
    ("legolas_cpu", "LEGOLAS"),
    ("shiftx2_noshifty", None),
    ("napshift", "NapShift"),
    ("deepshift_final_gpu_single_coldproc", "DeepShift, single model"),
    ("deepshift_final_gpu_ens_coldproc", "DeepShift, five-model ensemble"),
    ("nmrgnn_cpu", "NMRGNN"),
    ("ucbshift_x", "UCBShift-X, structure only"),
    ("procs15", "ProCS15"),
]
# Batch block: every tool with a shipped batch interface, driven through it over the same
# cohort. NMRGNN has no row because it ships no batch mode: its multi-file input reads the
# extra files as trajectory frames of one system, not as separate proteins. The passes
# below were repeated on an idle machine and print the mean of the listed repeats; the
# DeepShift rows are the concurrent-throughput arm, the three REDUCE tools the passes with
# REDUCE over 8 workers.
REPEATED = [
    ("deepshift_throughput_single", "DeepShift, single model", "GPU", ("r3", "r4")),
    ("deepshift_throughput_ens", "DeepShift, five-model ensemble", "GPU", ("r2", "r3")),
    ("legolas_batch_gpu_prep8", "LEGOLAS", "GPU", ("r1", "r2")),
    ("legolas_batch_cpu_prep8", "LEGOLAS", "20 CPU", ("r3", "r4")),
    ("sparta_batch_prep8", "SPARTA+", "20 CPU", ("r3", "r4")),
    ("ucbshift_x_batch_prep8", "UCBShift-X, structure only", "20 CPU", ("r3", "r4")),
]
# A timing pass prints only if the retime ledger (retime_idle_ledger.py, ledger.tsv beside
# the passes) kept it: the machine was idle before it and no CPU outside its process tree
# was used during it. These six passes were timed back to back on an idle machine and are
# not in the ledger.
IDLE_BEFORE_LEDGER = {
    "deepshift_throughput_single__batch__r3.json", "deepshift_throughput_single__batch__r4.json",
    "deepshift_throughput_ens__batch__r2.json", "deepshift_throughput_ens__batch__r3.json",
    "legolas_batch_gpu_prep8__batch__r1.json", "legolas_batch_gpu_prep8__batch__r2.json",
}
# Tools with no preparation step to parallelize: one pass each. The SHIFTX2 pass records
# no command line; run_speed_batch.py runs it with -n (--noshifty).
BATCH_TOOLS = [
    ("shiftx2_batch", "SHIFTX2, SHIFTY+ disabled", "20 CPU"),
    ("napshift_batch", "NapShift", "20 CPU"),
    ("procs15_batch", "ProCS15", "20 CPU"),
]
N_BOOT = 2000
SEED = 20260814


def clock_evidence(cohort_n: int) -> str:
    """What the caption's clock covers, block by block, as a comment beneath the float."""
    return "\n".join([
        "% Clock, by block: every per-invocation timing includes process start and model",
        "% load; per-invocation ms per residue is the slope of wall clock on residue count.",
        "% Each batch row is one cohort total (preparation + tool; for DeepShift, process",
        f"% launch to last CSV) divided by the {cohort_n} chains and by the census residue",
        "% count. NMRGNN has no batch lane: its multi-file mode",
        "% reads the extra files as trajectory frames of one system, not separate proteins.",
    ])


def constrained_fit(source_file, boots=N_BOOT, seed=SEED):
    """Startup and marginal rate refitted under a non-negative startup.

    Startup is a physical quantity and cannot be negative, so the intercept of wall clock
    on residue count is fitted subject to that. An arm whose plain intercept is already
    positive is an interior solution: this returns None for it and its printed numbers are
    the census values. Where the constraint binds -- an arm whose cost is dominated by
    size scaling, leaving the constant barely identified -- the estimate is a startup of
    zero with the rate refit through the origin, and the interval is a protein bootstrap
    of that same constrained rule.
    """
    records = [r for r in json.loads(Path(os.path.expandvars(source_file)).read_text())["records"]
               if r["status"] == "ok"]
    residues = np.array([r["n_aa"] for r in records], float)
    wall = np.array([r["wall_sec"] for r in records], float)

    def fit(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        if intercept >= 0.0:
            return intercept, slope
        return 0.0, float((y * x).sum() / (x * x).sum())

    intercept, slope = fit(residues, wall)
    if intercept > 0.0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(residues), size=(boots, len(residues)))
    replicates = np.array([fit(residues[d], wall[d]) for d in draws])
    startup_lo, startup_hi = np.percentile(replicates[:, 0], [2.5, 97.5])
    rate_lo, rate_hi = np.percentile(replicates[:, 1], [2.5, 97.5])
    return {"fixed_cost_intercept_sec": intercept,
            "fixed_cost_intercept_ci95": [float(startup_lo), float(startup_hi)],
            "marginal_ms_per_residue": 1000.0 * slope,
            "marginal_ms_per_residue_ci95": [1000.0 * float(rate_lo),
                                             1000.0 * float(rate_hi)],
            "rule": "non-negative-startup constrained fit; rate through the origin",
            "bootstrap": {"resamples": boots, "seed": seed, "unit": "protein"}}


def ledger_kept(runs: Path) -> set[str]:
    """The timing files whose pass the retime ledger kept, named as the drivers write them."""
    kept = set()
    ledger = runs / "ledger.tsv"
    rows = ledger.read_text().splitlines()[1:] if ledger.exists() else []
    for line in rows:
        command, *_rest, verdict = line.split("\t")
        if verdict != "kept":
            continue
        script, *argv = command.split()
        if script == "run_batch_parallel_prep.py":
            arm, workers, repeat = argv
            kept.add(f"{arm}_prep{workers}__batch__r{repeat}.json")
        else:                                   # run_speed_census.py --arm A [--subset N --repeat R]
            option = dict(zip(argv[::2], argv[1::2]))
            kept.add(f"{option['--arm']}__sub{option['--subset']}__r{option['--repeat']}.json"
                     if "--subset" in option else f"{option['--arm']}__full__r0.json")
    return kept


def census_runs(arms) -> Path:
    """The directory of timing records the census summarized, read from the census.

    Every arm names the record it was aggregated from, and each tool's batch cohort pass
    is written beside those records, so the batch block follows the census itself rather
    than a second root.
    """
    roots = {Path(os.path.expandvars(a["source_file"])).parent for a in arms.values()}
    if len(roots) != 1:
        raise SystemExit(
            f"REFUSING to emit -- the census names {len(roots)} run directories")
    return roots.pop()


def load_batch(runs: Path, tag: str, rep: str = "r0") -> dict:
    """One cohort pass through a tool's shipped batch interface.

    A pass that did not finish every chain is not a row comparable with the others, so
    nothing is written.
    """
    payload = json.loads((runs / f"{tag}__batch__{rep}.json").read_text())
    if payload["n_outputs"] != payload["n_chains"]:
        raise SystemExit(f"REFUSING to emit -- {tag} finished "
                         f"{payload['n_outputs']}/{payload['n_chains']} chains")
    return payload


def authors_caption() -> str | None:
    """The authors' caption, verbatim, from the captions directory. Never regenerated."""
    return CAPTION.read_text().strip("\n") if CAPTION.is_file() else None


def coverage(a) -> str:
    c = a.get("coverage_all240_fold6_xtal")
    return "---" if not c else f"{100.0 * c['n_ok'] / c['n_tasks']:.1f}\\%"


def row(a, label, startup_sec, startup_ci=None):
    """startup_ci is printed for the per-invocation block only, because the width of that
    interval is the reading: an arm whose cost is all size scaling has a startup the
    cohort barely identifies and prints the constrained zero with its upper bound, and an
    arm that is almost all fixed cost has a tight interval well away from zero."""
    lo, hi = a["per_protein_mean_ci95"]
    name = r"\textbf{" + label + "}" if label.startswith("DeepShift") else label
    startup = f"{startup_sec:.2f}"
    if startup_ci:
        startup += f" \\tiny[{startup_ci[0]:.2f}, {startup_ci[1]:.2f}]"
    return " & ".join([
        r"\quad " + name,
        a["hardware"],
        f"{a['per_protein_mean_sec']:.2f} \\tiny[{lo:.2f}, {hi:.2f}]",
        startup,
        f"{a['marginal_ms_per_residue']:.2f}",
        coverage(a),
    ]) + r" \\"


def batch_values(payload, n_residues) -> dict:
    """Both batch columns from the one cohort total (preparation + tool)."""
    return {"seconds_per_protein": payload["total_sec"] / payload["n_chains"],
            "ms_per_residue": 1000.0 * payload["total_sec"] / n_residues}


def repeated_payload(runs: Path, tag: str, reps) -> dict:
    """Mean total of the listed repeats of one batch arm."""
    passes = [load_batch(runs, tag, r) for r in reps]
    if len({p["n_chains"] for p in passes}) != 1:
        raise SystemExit(f"REFUSING to emit -- {tag} repeats cover different cohorts")
    return {"n_chains": passes[0]["n_chains"],
            "total_sec": float(np.mean([p["total_sec"] for p in passes])),
            "repeats": dict(zip(reps, passes))}


def batch_row(payload, label, hardware, n_residues) -> str:
    """One cohort pass (or the mean of repeats), so a point estimate with no interval.

    Both columns are the cohort total -- preparation plus tool, or for DeepShift process
    launch to last output -- over 203 chains and over the census residue count.
    """
    v = batch_values(payload, n_residues)
    name = r"\textbf{" + label + "}" if label.startswith("DeepShift") else label
    return " & ".join([
        r"\quad " + name,
        hardware,
        f"{v['seconds_per_protein']:.2f}",
        "---",
        f"{v['ms_per_residue']:.2f}",
        "---",
    ]) + r" \\"


def batch_denominator(arms) -> int:
    """The census residue count every batch row is divided by: the count of the 203-chain
    cohort, on which every printed census arm agrees (some unprinted arms timed a subset)."""
    counts = {arms[k]["n_residues_total"] for k, _ in COLD}
    if len(counts) != 1:
        raise SystemExit(f"REFUSING to emit -- census arms disagree on residues: {counts}")
    return counts.pop()


def cold_rows(arms) -> list[tuple[str, str]]:
    """The per-invocation arms and the label each prints."""
    return [(k, arms[k]["method"] if label is None else label) for k, label in COLD]


def build_table(census, arms, overrides, caption) -> str:
    runs = census_runs(arms)
    cohort_n = census["cohort_n"]
    body = [r"\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{4pt}"]
    body.append(r"\caption{" + caption + "}" if caption else
                r"\capneeded{Facts to assert are in the comment beneath the table.}")
    # \label immediately after \caption, then the evidence: nothing between the two.
    # No \resizebox: at six columns the natural width fits the text block, and
    # resizebox would upscale a narrow table to \textwidth, inflating the fonts.
    body += [r"\label{tab:inference_speed}", clock_evidence(cohort_n),
             r"\begin{tabular}{llrrrr}", r"\toprule",
             (r"Method & Hardware & \begin{tabular}{@{}c@{}}Seconds per\\protein\end{tabular} "
              r"& \begin{tabular}{@{}c@{}}Startup\\(s)\end{tabular} "
              r"& \begin{tabular}{@{}c@{}}Milliseconds\\per residue\end{tabular} "
              r"& \begin{tabular}{@{}c@{}}Chains it\\finishes\end{tabular} \\"),
             r"\midrule",
             r"\multicolumn{6}{@{}l}{\emph{One protein per invocation: every method pays "
             r"its own startup}} \\"]
    cold = sorted(((arms[k], lbl, overrides[k]) for k, lbl in cold_rows(arms)),
                  key=lambda t: t[0]["per_protein_mean_sec"])
    for arm, lbl, override in cold:
        a = {**arm, **(override or {})}
        body.append(row(a, lbl, a["fixed_cost_intercept_sec"],
                        a["fixed_cost_intercept_ci95"]))
    body += [r"\midrule",
             r"\multicolumn{6}{@{}l}{\emph{Batch: model loaded once for all "
             f"{cohort_n}" r" chains}} \\"]
    n_residues = batch_denominator(arms)
    payloads = [(repeated_payload(runs, k, reps), lbl, hw) for k, lbl, hw, reps in REPEATED]
    payloads += [(load_batch(runs, k), lbl, hw) for k, lbl, hw in BATCH_TOOLS]
    merged = [(batch_values(p, n_residues)["seconds_per_protein"],
               batch_row(p, lbl, hw, n_residues)) for p, lbl, hw in payloads]
    body += [line for _, line in sorted(merged, key=lambda t: t[0])]
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    census = json.loads((CENSUS / "speed_census.json").read_text())
    arms = {a["arm"]: a for a in census["arms"]}

    need = [k for k, _ in COLD if k not in arms]
    if need:
        raise SystemExit(f"REFUSING to emit -- arms not in the census: {need}")
    runs = census_runs(arms)
    retimed = [f"{k}__batch__{r}.json" for k, _, _, reps in REPEATED for r in reps]
    retimed += [Path(arms[k]["source_file"]).name for k, _ in COLD if k.startswith("deepshift")]
    passes = retimed + [f"{k}__batch__r0.json" for k, _, _ in BATCH_TOOLS]
    if absent := [p for p in passes if not (runs / p).is_file()]:
        raise SystemExit(f"REFUSING to emit -- batch passes not on disk in {runs}: {absent}")
    if unkept := [p for p in retimed if p not in ledger_kept(runs) | IDLE_BEFORE_LEDGER]:
        raise SystemExit(f"REFUSING to emit -- the retime ledger did not keep: {unkept}")

    # DeepShift's chains-finished figures are the census's (measured in July on an earlier
    # model); each printed row was measured again with the shipped model it times -- member
    # 1 alone for the single-model row, all five members for the ensemble -- and must agree.
    final_cover_paths = {"single": runs.parent / "deepshift_coverage240_final.json",
                         "ensemble": runs.parent / "deepshift_coverage240_final_ensemble.json"}
    final_cover = {k: json.loads(p.read_text()) for k, p in final_cover_paths.items()}
    for k, _label in COLD:
        if not k.startswith("deepshift"):
            continue
        c = arms[k].get("coverage_all240_fold6_xtal")
        m = final_cover["ensemble" if "_ens" in k else "single"]
        if not c or (c["n_ok"], c["n_tasks"]) != (m["n_ok"], m["n_tasks"]):
            raise SystemExit(f"REFUSING to emit -- {k} prints {c and c['n_ok']}/"
                             f"{c and c['n_tasks']} chains finished; the shipped model "
                             f"finishes {m['n_ok']}/{m['n_tasks']}")

    overrides = {k: constrained_fit(arms[k]["source_file"]) for k, _ in COLD}
    caption = authors_caption()
    banner = "\n".join([
        "% Numbers are read from the speed census the environment names.",
        "% Two blocks, because a warm timing and a cold one are not one ranking. The",
        "% per-invocation block times every method the way a per-file CLI runs, paying",
        "% full startup on every structure; the batch block drives every tool that ships",
        "% a batch interface over the same cohort with its model loaded once. Per-invocation",
        "% rows carry startup and the startup-free marginal rate; batch rows carry one",
        "% cohort total over chains and over residues, preparation included.",
        "% Startup is fitted under a non-negativity constraint; an arm where that",
        "% constraint binds prints the constrained refit.",
        "% The caption is the authors', from si_build/reference_inputs/captions/.",
    ])

    si_common.emit(
        "si_tables_speed", build_table(census, arms, overrides, caption), banner=banner,
        script="emit_speed.py", placeholder_caption=not caption)

    print(f"caption preserved: {'yes' if caption else 'NO -- emitted capneeded'}")


if __name__ == "__main__":
    main()
