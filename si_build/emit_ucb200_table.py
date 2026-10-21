"""Emit tab:published_testsets: every scored predictor on UCBShift's 200-protein
test set and on SHIFTX2's 61-protein test set, grouped by exclusion setting, each
set on the cells common to all of its printed rows.

Reads score_allpred_paper_exclusions.json ($SI_UCB200_ARTIFACTS_DIR, 200-set: the pooled
report with the cells UCBShift's paper removes from its own test set removed, as Tables S18
and S19 remove them; benchmarks/reference/ucb200_paper_exclusions.json) and
pool_shiftx2_61.json ($SI_UCB200_POOL61, 61-set). Every printed number comes from those two
files. Emission
refuses to run if any expected arm is absent -- no placeholder rows, ever.

UCBShift-Y is not printed: partial cell coverage; its released-value reproduction
is in the gate table. PLM-CS prints under its own block header: it is a
sequence-only model whose RefDB training corpus contains every test deposition of
both sets (verified against the released refDB tree), so its rows measure
memorization, not prediction.

Data inputs (score_allpred_paper_exclusions.json, pool_shiftx2_61.json) live outside the release repo --
experiment records, not release code -- and are resolved by environment; output goes to
si_common.OUT.
"""
import json
import os
import re

import si_common

ARTIFACTS_DIR = si_common.root("SI_UCB200_ARTIFACTS_DIR")   # holds the 200-set report
SCORES200 = ARTIFACTS_DIR / "score_allpred_paper_exclusions.json"
POOL61 = si_common.root("SI_UCB200_POOL61")                 # pool_shiftx2_61.json itself
# The 61-set prediction run's own log: one line per protein naming the members that
# predicted it, a single held-out member or all five. The run writes it beside the pool.
PREDICT_LOG = (si_common.root("SI_SHIFTX2_61_LOG") if os.environ.get("SI_SHIFTX2_61_LOG")
               else POOL61.parent.parent / "logs" / "shiftx2_61.log")


def member_assignment(log, pool61) -> dict:
    """How many of the proteins the printed row covers a single held-out member predicted,
    and how many the five-member ensemble did, read from the prediction log. The row covers
    the scorable proteins every printed arm predicted: the pool's exclusions and the
    proteins an incomplete printed arm is missing are out, and what is left must be the
    pool's common chains."""
    members = dict(re.findall(r"\]\s+(\S+)\s+members=\[([0-9, ]+)\]", log.read_text()))
    excluded = set(pool61["protein_exclusions"])
    for note in (pool61["incomplete_arms_excluded"] or {}).values():
        excluded |= set(re.findall(r"'([0-9A-Z_]{5})'", note.split("missing", 1)[-1]))
    shown = {pid: m for pid, m in members.items() if pid not in excluded}
    if len(shown) != pool61["common_chains"]:
        raise SystemExit(f"{len(shown)} proteins left after the exclusions, but the pool "
                         f"scores {pool61['common_chains']} common chains")
    n_all = sum(len(m.split(",")) == 5 for m in shown.values())
    return {"proteins_in_log": len(members), "not_in_the_printed_row": sorted(excluded),
            "proteins_in_the_printed_row": len(shown),
            "held_out_member": len(shown) - n_all, "five_member_ensemble": n_all}
NUCLEI = ["HA", "H", "CA", "CB", "C", "N"]          # print order
ATOMS61 = {"HA": "HA", "H": "H", "CA": "CA", "CB": "CB", "C": "C", "N": "N"}

BLOCKS_200 = [
    ("No transfer database", [
        (r"\textbf{DeepShift}$^{\dagger}$", "deepshift_ensemble"),
        ("UCBShift-X", "X_H"),
        ("SHIFTX2 (SHIFTY+ off)", "shiftx2_structonly"),
        ("SPARTA+", "spartaplus"),
        (r"CAMSHIFT$^{\S}$", "camshift"),
        ("NapShift", "napshift"),
        ("LEGOLAS", "legolas"),
        (r"NMRGNN$^{\S}$", "nmrgnn"),
        ("ProCS15", "procs15"),
        ("MLFF", "mlff_relaxed"),
    ]),
    ("No exclusion", [
        ("UCBShift", "UCBShift_H"),
        ("SHIFTX2", "shiftx2_full"),
    ]),
    ("Same-BMRB excluded", [
        ("UCBShift", "ucbshift_bmrbonly"),
        ("SHIFTX2", "shiftx2_selfexcl"),
    ]),
    (r"Same-BMRB, same-UniProt, and $\geq$90\% identity excluded", [
        ("UCBShift", "ucbshift_strict"),
        ("UCBShift 2.0", "ucbshift2"),
    ]),
    ("Training set contains every test deposition", [
        ("PLM-CS", "plmcs"),
    ]),
]

BLOCKS_61 = [
    ("No transfer database", [
        (r"\textbf{DeepShift}$^{\ddagger}$", "deepshift_heldout"),
        ("UCBShift-X", "ucbshift_x"),
        ("SHIFTX2 (SHIFTY+ off)", "shiftx2_structonly"),
        ("SPARTA+", "sparta"),
        (r"CAMSHIFT$^{\S}$", "camshift_fixed"),
        ("NapShift", "napshift"),
        ("LEGOLAS", "legolas"),
        (r"NMRGNN$^{\S}$", "nmrgnn_fixed"),
        ("ProCS15", "procs15"),
        ("MLFF", "mlff_relaxed"),
        ("PLM-CS", "plmcs"),
    ]),
    ("No exclusion", [
        ("UCBShift", "ucbshift_full_leaky"),
        ("SHIFTX2, as distributed", "shiftx2_asdistributed"),
    ]),
    ("Same-BMRB excluded", [
        ("UCBShift", "ucbshift_full_bmrbonly"),
        ("SHIFTX2", "shiftx2_selfexcl"),
    ]),
    (r"Same-BMRB, same-UniProt, and $\geq$90\% identity excluded", [
        ("UCBShift", "ucbshift_full_selfexcl"),
        ("UCBShift 2.0", "ucbshift2"),
    ]),
]


def row(vals, fmt):
    return " & ".join(fmt(v) for v in vals)


with open(SCORES200) as fh:
    d200 = json.load(fh)
common200 = d200["common_cells"]
with open(POOL61) as fh:
    d61 = json.load(fh)
common61 = d61["common"]

missing = [arm for _t, ent in BLOCKS_200 for _s, arm in ent if arm not in common200]
missing += [f"61:{arm}" for _t, ent in BLOCKS_61 for _s, arm in ent if arm not in common61]
if missing:
    raise SystemExit(f"REFUSING to emit -- arms not yet scored: {missing}")

counts200 = [common200["deepshift_ensemble"][n]["n"] for n in NUCLEI]
for _t, ent in BLOCKS_200:
    for _s, arm in ent:
        assert [common200[arm][n]["n"] for n in NUCLEI] == counts200, arm
counts61 = [common61["deepshift_heldout"][ATOMS61[n]]["n"] for n in NUCLEI]
for _t, ent in BLOCKS_61:
    for _s, arm in ent:
        assert [common61[arm][ATOMS61[n]]["n"] for n in NUCLEI] == counts61, arm

# the chains the common cells span, not the 60 with labels before the intersection
if "common_chains" not in d61:
    raise SystemExit(f"REFUSING to emit -- {POOL61} does not record the chains its common "
                     "cells span (common_chains)")
n_chains61 = d61["common_chains"]

banner = []
a = banner.append
a(f"% 200-set: {d200['paper_exclusions']['common_cells_removed']} cells UCBShift's paper "
  "removes are removed.")
a(f"% 61-set: arms missing proteins, which leave the common set: "
  f"{d61['incomplete_arms_excluded'] or 'none'}")
a("% UCBShift-Y omitted: partial cell coverage; its released-value reproduction is in the gate table.")
a("%")
a("% Caption source material (for the caption; not for the page)")
a("%   Shown: every scored predictor on UCBShift's published 200-protein test set and on SHIFTX2's")
a("%   61-protein test set -- the authors' structures, labels, merge and outlier rules -- grouped by")
a("%   what the predictor may retrieve. Within each set, cells common to every printed row. RMSE of")
a("%   the raw residual, ppm. 61-set labels merged by sequence alignment (author numbering is")
a("%   unreliable on 31 of 60 chains); 1KDBA absent: deposition missing from the released refDB.")
a("%   Dagger: the five-member ensemble (run_predict.json names its checkpoints). Double dagger: each protein with a >=90%")
a("%   training relative is scored by the one CV member that held that relative's fold out.")
a("%   Section sign: tail-dominated RMSE. PLM-CS: RefDB-trained sequence model. Every 200-set")
a("%   deposition is in its training corpus (hence its own block there); the 61-set is its own")
a("%   published test set, so that row is its honest condition -- the contrast measures the leak.")
a("%   MLFF row (both sets): relaxed protocol only -- Amber99/OpenMM relaxation, then")
a("%   polish_only hydrogens.")
a("%   1HSLB relaxed with its free-His ligand (HETATM HIS B 239) dropped for minimisation and")
a("%   re-attached, per the het rule.")

lines = []
a = lines.append
a(r"\begin{table}[H]\centering\small")
assign = member_assignment(PREDICT_LOG, d61)
a(r"\caption{Each predictor on the two published benchmark test sets. Some cells, as well as"
  r" the C$'$ entries of 3 anticorrelated proteins, are excluded from UCBShift's test set as in"
  r" the UCBShift paper\cite{UCBShift}. UCBShift contains most"
  r" of SHIFTX2's test proteins in its training set. PLM-CS contains all of UCBShift's test"
  r" data in its training set. These are not faults of the models, but they prevent an"
  r" accurate evaluation on these sets. The DeepShift row for the SHIFTX2 set is run on the"
  r" model that had the protein held out for " + str(assign["held_out_member"])
  + r" proteins with a training relative and uses the five-member ensemble for the other "
  + str(assign["five_member_ensemble"]) + r".}")
a(r"\label{tab:published_testsets}")
a(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}")
a(r"\toprule")
a(r"Predictor & H$\alpha$ & H$^{\mathrm{N}}$ & C$\alpha$ & C$\beta$ & C$'$ & N \\")
a(r"\midrule")
a(r"\multicolumn{7}{@{}l}{\emph{UCBShift's 200-protein test set, "
  + f"{d200['funnel']['common_chains']} chains" + r"}} \\")
a(r"\quad \emph{cells scored} & " + row(counts200, lambda v: f"{v:,}") + r" \\")
for title, entries in BLOCKS_200:
    a(r"\multicolumn{7}{@{}l}{\emph{" + title + r"}} \\")
    for stub, arm in entries:
        vals = [common200[arm][n]["rmse"] for n in NUCLEI]
        a(r"\quad " + stub + " & " + row(vals, lambda v: f"{v:.3f}") + r" \\")
a(r"\midrule")
a(r"\multicolumn{7}{@{}l}{\emph{SHIFTX2's 61-protein test set, "
  + f"{n_chains61} chains" + r"}} \\")
a(r"\quad \emph{cells scored} & " + row(counts61, lambda v: f"{v:,}") + r" \\")
for title, entries in BLOCKS_61:
    a(r"\multicolumn{7}{@{}l}{\emph{" + title + r"}} \\")
    for stub, arm in entries:
        vals = [common61[arm][ATOMS61[n]]["rmse"] for n in NUCLEI]
        a(r"\quad " + stub + " & " + row(vals, lambda v: f"{v:.3f}") + r" \\")
a(r"\bottomrule")
a(r"\end{tabular*}")
a(r"\vspace{0.4em}")
a(r"\begin{minipage}{\textwidth}\footnotesize\raggedright")
a(r"$^{\dagger}$the five-member ensemble\quad")
a(r"$^{\ddagger}$scored by the CV member holding out each protein's training relative\quad")
a(r"$^{\S}$tail-dominated RMSE")
a(r"\end{minipage}")
a(r"\end{table}")
si_common.emit(
    "si_published_testsets", "\n".join(lines), script="emit_ucb200_table.py",
    banner="\n".join(banner), placeholder_caption=False)
print(f"wrote si_published_testsets ({len(lines)} lines)")
