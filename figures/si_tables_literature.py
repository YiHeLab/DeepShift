#!/usr/bin/env python3
"""Table S16: the training, transfer and test corpora of the compared predictors.

Every competitor figure is the number the method's own authors state, read out of the
article that introduced the method. **The PDFs are parsed at generation time, not
hand-transcribed**: each number is captured by a regex and compared with the value
printed here, so a parse that drifts fails the build instead of shipping. The DeepShift
row is counted from the run.

``out/si_tables_literature.tex`` is the corpus table alone, one float and nothing around
it, since SI.tex ``\\input``s it inside a section of its own.

Usage:
    python figures/si_tables_literature.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import frozen

# The corpus table reads the articles, and they are not all filed in one place. DEEPSHIFT_PAPER_PATH is a colon-separated search path read left to
# right the way $PATH is, so a machine that keeps its PDFs elsewhere adds a directory
# instead of editing this file.
PAPER_PATH = tuple(Path(p).expanduser() for p in
                   os.environ.get("DEEPSHIFT_PAPER_PATH", "").split(os.pathsep) if p)

# The released transfer library UCBShift searches at prediction time, as chains on disk.
REFDB_CHAINS = frozen.env_path("DEEPSHIFT_REFDB_CHAINS")


def paper(name: str) -> Path:
    """The first copy of `name` on the paper search path.

    Fails naming the file and the path it looked on, because a corpus size that cannot
    be read out of its source must not be filled in from anywhere else.
    """
    for root in PAPER_PATH:
        if (found := root / name).is_file():
            return found
    raise SystemExit(f"{name} is on none of the directories in DEEPSHIFT_PAPER_PATH "
                     f"({', '.join(str(p) for p in PAPER_PATH) or 'unset'}). Every "
                     "number in tab:lit_corpora is read from its own paper; none is "
                     "typed in.")

# The DeepShift row is counted from the run itself, so its fold authority comes off
# frozen.RUN_ROOT with every other figure and not from a second variable.
DEEPSHIFT_FOLD_MAP = frozen.RUN_ROOT / "data/fold_assignments_labelset_identity90.json"

DEPLOYED_STEM = "si_tables_literature"


# --------------------------------------------------------------------------------
# tab:lit_corpora -- one block per predictor the paper compares against
# --------------------------------------------------------------------------------
# These figures come out of the papers and not the supplementary information,
# because that is where most of these methods state their corpus. Every one is captured
# from the PDF by a regular expression at generation time and compared with the value
# printed here, so a pattern that drifts onto a different number, or a paper that is
# re-typeset, stops the build instead of shipping a stale figure.
#
# Units are not converted between rows. The papers do not count the same thing: CamShift
# reports chemical shifts and gives no protein count, NapShift reports database entries,
# UCBShift reports proteins before duplicate chains are removed and chains after, and
# ProCS15 is fitted to DFT calculations and not trained on a protein set. The column
# header says so.

_TEXT: dict[Path, str] = {}


def pdftext(pdf: Path) -> str:
    """Whitespace-flattened text of a whole paper, cached.

    A corpus size is a sentence somewhere in a body whose pagination we have no reason to
    depend on, so the whole paper is read and not one page.
    """
    if pdf not in _TEXT:
        raw = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                             text=True, check=True).stdout
        _TEXT[pdf] = re.sub(r"\s+", " ", raw)
    return _TEXT[pdf]


def stated(pdf: Path, pattern: str, expected: int) -> str:
    """Assert the paper still states ``expected``, and return it with thousands commas.

    The regex must capture exactly one group, the number as the paper prints it. A build
    that cannot find it, or that finds a different number, stops here and never
    ships a figure nobody re-read.
    """
    m = re.search(pattern, pdftext(pdf))
    if not m:
        raise SystemExit(f"{pdf.name}: pattern {pattern!r} does not match. Correct the "
                         "pattern against the PDF; do not edit the number by hand.")
    got = int(m.group(1).replace(",", "").replace(" ", ""))
    if got != expected:
        raise SystemExit(f"{pdf.name}: pattern {pattern!r} now reads {got:,}, not "
                         f"{expected:,}. The source changed; check it before shipping.")
    return f"{expected:,}"


def quoted(pdf: Path, pattern: str, expected: str) -> str:
    """The same guarantee for a phrase, for the rows whose size is not a count."""
    m = re.search(pattern, pdftext(pdf))
    got = m.group(1) if m else None
    if got != expected:
        raise SystemExit(f"{pdf.name}: pattern {pattern!r} does not read "
                         f"{expected!r} (got {got!r}). Check the source.")
    return expected


def corpora_rows():
    """(method, [(role, size, source)]) for every predictor the paper compares against.

    A transfer library is a set of assigned proteins searched at prediction time, not
    training data, so it gets its own line and is never folded into the training
    row -- for the retrieval methods that is the number that bounds the claim.
    """
    # The released RefDB is the one entry counted and not quoted: neither UCBShift
    # paper states the size of the library its shipped code searches, and both versions
    # search the same released snapshot, so both rows cite the snapshot and not a
    # paper that does not carry the number.
    refdb = len(list(REFDB_CHAINS.glob("*.pdb")))
    if not refdb:
        raise SystemExit(f"{REFDB_CHAINS}: no chains found; the RefDB transfer-library "
                         "size is counted from this directory and must not be guessed")
    refdb_size = f"{refdb:,} chains"
    refdb_src = r"released \texttt{refDB}\cite{RefDB}"

    sparta = paper("SPARTA+.pdf")
    shiftx2 = paper("ShiftX2.pdf")
    ucb1 = paper("UCBShift.pdf")
    ucb2 = paper("ucbshift-2-0-bridging-the-gap-from-backbone-to-side-chain-protein-"
                 "chemical-shift-prediction-for-protein-structures-2.pdf")
    camshift = paper("camshift_2009_jacs.pdf")
    napshift = paper("Qi_2022_NapShift_hybrid_potentials_NMR.pdf")
    mlff = paper("2505.23354v5-2.pdf")
    efgcs = paper("efgcs_2024_ProteinScience_Gu_ESMFold_chemshift.pdf")
    nmrgnn = paper("nmrgnn.pdf")
    procs15 = paper("procs15.pdf")
    legolas = paper("legolas_main.pdf")
    plmcs = paper("plmcs.pdf")

    sparta_train = stated(sparta, r"containing (\d+) proteins for which high-resolution",
                          580)
    x_train = stated(shiftx2, r"final training dataset consisted of (\d+) high resolution",
                     197)
    x_test = stated(shiftx2, r"final testing dataset consisted of (\d+) high resolution",
                    61)
    y_lib = stated(shiftx2, r"currently contains (\d+) re-referenced protein assignments",
                   1903)
    u1_train = stated(ucb1, r"there were altogether (\d+) proteins in the training dataset",
                      852)
    u1_chains = stated(ucb1, r"protein chains in the training dataset decreased to (\d+)",
                       647)
    u1_test = stated(ucb1, r"(\d+) proteins with high-resolution", 200)
    u2_train = stated(ucb2, r"of the (\d+) protein structures are provided", 851)
    u2_test = stated(ucb2, r"test set consists of (\d+) structures", 200)
    cam_shifts = stated(camshift, r"extracted a total of (\d+ \d+) chemical shifts", 224036)
    cam_test2 = stated(camshift, r"second test comprises (\d+) structures", 28)
    nap_train = stated(napshift, r"decomposed into (\d+) entries for training", 2987)
    nap_test = stated(napshift, r"tested using a set of (\d+) proteins", 250)
    mlff_train = stated(mlff, r"split into (\d+) training and \d+ test proteins", 823)
    mlff_test = stated(mlff, r"split into \d+ training and (\d+) test proteins", 225)
    efg_bb = stated(efgcs, r"test dataset of (\d+) proteins that do not share the same "
                           r"amino acid sequence as the training dataset of backbone", 200)
    efg_sc = stated(efgcs, r"test dataset of (\d+) proteins that do not share the same "
                           r"amino acid sequence as the training dataset of side-chain", 61)

    # The four rows below were absent while their papers were not on disk. They are on
    # disk now, so every predictor this paper scores has a line and none is left out for
    # want of a source.
    gnn_train = stated(nmrgnn, r"paired dataset of (\d+) proteins with both X-ray", 2405)
    gnn_test = stated(nmrgnn, r"SHIFTX test dataset of (\d+) proteins", 62)
    # ProCS15 is fitted to quantum calculations, not trained on a protein set, so its
    # "training" cell is a magnitude of calculations and stays as the paper writes it.
    procs_dft = quoted(procs15, r"backbone terms are based on .?([\d.]+) million DFT "
                                r"calculations", "2.35")
    procs_test = quoted(procs15, r"we benchmark the NMR chemical shift predictions on "
                                 r"(Ubiquitin and GB3)", "Ubiquitin and GB3")
    # LEGOLAS takes the SHIFTX2 corpus, which is why its counts repeat the SHIFTX2 row's
    # 197 and 61 and are not an independent measurement; PLM-CS tests on SHIFTX2's 61.
    leg_train = stated(legolas,
                       r"SHIFTX2 data set consists of a training set of (\d+) proteins",
                       197)
    leg_test = stated(legolas,
                      r"training set of \d+ proteins and a test set of (\d+) proteins", 61)
    plm_files = stated(plmcs, r"training set includes (\d+) rereferenced protein", 2429)
    plm_train = stated(plmcs, r"resulting in a dataset of (\d+) proteins", 2359)
    plm_test = stated(plmcs, r"SHIFTX test dataset,\d* which contains (\d+) proteins", 61)

    S_SPARTA = r"Shen and Bax\cite{spartaplus2010}"
    S_X2 = r"Han \emph{et al.}\cite{shiftx2_2011}"
    S_U1 = r"Li \emph{et al.}\cite{UCBShift}"
    S_U2 = r"Ptaszek \emph{et al.}\cite{UCBShift2}"
    S_CAM = r"Kohlhoff \emph{et al.}\cite{camshift2009}"
    S_NAP = r"Qi \emph{et al.}\cite{napshift2022}"
    S_GNN = r"Yang \emph{et al.}\cite{nmrgnn2021}"
    S_PROCS = r"Larsen \emph{et al.}\cite{procs15_2015}"
    S_LEG = r"Darrows \emph{et al.}\cite{LEGOLAS}"
    S_PLM = r"Zhu \emph{et al.}\cite{PLM-CS}"
    S_MLFF = r"Bojan \emph{et al.}\cite{MLFF}"
    S_EFG = r"Gu \emph{et al.}\cite{efgcs2024}"
    NONE = "none"

    return [
        ("SPARTA+", [
            ("training set", f"{sparta_train} proteins", f"{S_SPARTA}, Methods"),
            ("transfer library", NONE, S_SPARTA),
            ("test set", "11 proteins", f"{S_SPARTA}, Table~1")]),
        ("SHIFTX+", [
            ("training set", f"{x_train} proteins", f"{S_X2}, SI Table~S1"),
            ("transfer library", NONE, f"{S_X2}, Methods"),
            ("test set", f"{x_test} proteins", f"{S_X2}, SI Table~S2")]),
        # SHIFTY+ has no test set of its own: it is a module of SHIFTX2, reported on the
        # SHIFTX2 test set and then only where its search finds a homolog. Printing 61
        # against it would overstate what it was scored on.
        ("SHIFTY+", [
            ("training set", NONE, f"{S_X2}, Methods"),
            ("transfer library", f"{y_lib} proteins", f"{S_X2}, SI Table~S7")]),
        ("SHIFTX2", [
            ("training set", f"{x_train} proteins", f"{S_X2}, Methods"),
            ("transfer library", f"{y_lib} proteins", f"{S_X2}, Methods"),
            ("test set", f"{x_test} proteins", f"{S_X2}, SI Table~S2")]),
        ("UCBShift", [
            ("training set", f"{u1_train} proteins ({u1_chains} chains)",
             f"{S_U1}, Results"),
            ("transfer library", refdb_size, refdb_src),
            ("test set", f"{u1_test} proteins", f"{S_U1}, Table~1")]),
        ("UCBShift 2.0", [
            ("training set", f"{u2_train} protein structures", f"{S_U2}, SI Table~S3"),
            ("transfer library", refdb_size, refdb_src),
            ("test set", f"{u2_test} structures", f"{S_U2}, Methods")]),
        ("CAMSHIFT", [
            ("training set", f"{cam_shifts} shifts", S_CAM),
            ("transfer library", NONE, S_CAM),
            ("test set", f"7 + {cam_test2} proteins", S_CAM)]),
        ("NapShift", [
            ("training set", f"{nap_train} entries", f"{S_NAP}, Methods"),
            ("transfer library", NONE, S_NAP),
            ("test set", f"{nap_test} proteins", f"{S_NAP}, Methods")]),
        ("NMRGNN", [
            ("training set", f"{gnn_train} proteins", f"{S_GNN}, Methods (RefDB dataset)"),
            ("transfer library", NONE, S_GNN),
            ("test set", f"{gnn_test} proteins",
             f"{S_GNN}, Methods (SHIFTX test set)")]),
        ("ProCS15", [
            ("training set", rf"none ($\sim${procs_dft} million DFT calculations)",
             f"{S_PROCS}, Abstract"),
            ("transfer library", NONE, S_PROCS),
            ("test set", "2 proteins",
             f"{S_PROCS}, Results ({procs_test[0].lower() + procs_test[1:]})")]),
        ("LEGOLAS", [
            ("training set", f"{leg_train} proteins",
             f"{S_LEG}, Training (SHIFTX2 training set)"),
            ("transfer library", NONE, S_LEG),
            ("test set", f"{leg_test} proteins",
             f"{S_LEG}, Training (SHIFTX2 test set)")]),
        ("PLM-CS", [
            ("training set", f"{plm_files} RefDB files ({plm_train} proteins after "
                             "length filter)", f"{S_PLM}, Datasets"),
            ("transfer library", NONE, S_PLM),
            ("test set", f"{plm_test} proteins",
             f"{S_PLM}, Results (SHIFTX test set)")]),
        ("MLFF predictor", [
            ("training set", f"{mlff_train} proteins", f"{S_MLFF}, Data"),
            ("transfer library", NONE, S_MLFF),
            ("test set", f"{mlff_test} proteins", f"{S_MLFF}, Figure~3")]),
        ("EFG-CS", [
            ("training set", "not stated", f"{S_EFG}, Datasets"),
            ("transfer library", "not stated", f"{S_EFG}, pipeline"),
            ("test set", f"{efg_bb} proteins for backbone, {efg_sc} for side chains",
             f"{S_EFG}, Tables~1 and~2")]),
    ]


# --------------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------------
def deepshift_corpus():
    """Our corpus, counted on the build the shipped model trained on.

    Every competitor row in this table is a genuine training set, so ours must be one
    too: the union of folds 1--5, what the five CV members collectively saw. The full
    realized corpus is the wrong number here, because it also counts fold 0 (the
    UCBShift-200 exclusion set) and fold 6 (held out), which no member
    trains on.

    Counted from the fold-assignment map and the realized caches, so a label set that was
    assigned to a training fold but never materialized cannot inflate the count.
    """
    root = frozen.RUN_ROOT
    assign = json.loads(DEEPSHIFT_FOLD_MAP.read_text())

    realized, bmrb = {}, {}
    for f in sorted({int(v) for v in assign.values()}):
        idm = root / f"work/unified_cache/fold_{f}/structural/id_maps.json"
        if not idm.exists():
            realized[f], bmrb[f] = None, None
            continue
        m = json.loads(idm.read_text())
        realized[f] = set(m["label_set_id"].values())
        bmrb[f] = {str(v) for v in m["bmrb_id"].values()}

    train = [f for f in (1, 2, 3, 4, 5) if f in realized]
    if len(train) != 5 or any(realized[f] is None for f in train):
        raise SystemExit("a training fold cache is missing; corpus size cannot be counted")
    train_ls = set().union(*(realized[f] for f in train))
    train_bmrb = set().union(*(bmrb[f] for f in train))
    all_ls = set().union(*(v for v in realized.values() if v is not None))

    return {
        "realized_label_sets_all_folds": len(all_ls),
        "train_label_sets": len(train_ls),
        "train_bmrb_entries": len(train_bmrb),
        "excluded_from_training": {
            "fold_0": len(realized.get(0) or ()),
            "fold_6": len(realized.get(6) or ()),
        },
    }


def corpora_table(rows, ds):
    """The tab:lit_corpora float: one block of three roles per method, DeepShift last."""
    body = []
    for method, entries in rows:
        for i, (role, size, source) in enumerate(entries):
            body.append(" & ".join([method if i == 0 else "", role, size, source])
                        + r" \\")
        body.append(r"\addlinespace")
    body.pop()                                     # no gap immediately before the rule
    body += [r"\midrule",
             rf"\textbf{{DeepShift}} & training set & "
             rf"\textbf{{{ds['train_label_sets']:,}}} realized label sets & "
             r"this work, folds 1--5 \\"]
    return [r"\begin{table}[H]\centering\small",
            r"\caption{Size of training, transfer, and test corpus for common chemical "
            r"shift prediction methods.}",
            r"\label{tab:lit_corpora}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llll}",
            r"\toprule",
            r"Method & Role & Size & Source \\",
            r"\midrule", *body,
            r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]


def corpora_document(rows, ds, sized, ucb2_train) -> list[str]:
    """The roster the supplement prints: one float, and the facts its caption needs.

    The float is the whole file. It is ``\\input`` inside a section of the supplement, so
    a sectioning command here would open a second one, and prose here would land in the
    middle of somebody else's argument.
    """
    L = corpora_table(rows, ds)
    # The DeepShift row is like-for-like with the competitors' training-set rows and
    # nothing else. The facts a caption needs are here, not in prose.
    L += [
        "% Every figure above is the number the method's own authors state, in the unit",
        "%   they state it in, taken from the paper that introduced the method. Units are",
        "%   not converted between rows. A transfer library is a set of assigned proteins",
        "%   searched at prediction time, not training data, so it is a line of its own.",
        "%   The one entry counted and not quoted is the released RefDB transfer",
        f"%   library ({sized['UCBShift']['transfer library']}): neither UCBShift paper",
        "%   states the size of the library its shipped code searches, and both versions",
        "%   search the same released snapshot, so both rows cite the snapshot.",
        "%   LEGOLAS takes the SHIFTX2 corpus, which is why its counts repeat the",
        "%   SHIFTX2 row's 197 and 61; PLM-CS tests on SHIFTX2's 61.",]
    L += [
        "% DeepShift row, counted not asserted, from $DEEPSHIFT_RUN:",
        f"%   {ds['train_label_sets']:,} realized label sets in folds 1--5 "
        f"({ds['train_bmrb_entries']:,} distinct BMRB depositions). That is the number in",
        "%   the table, because every competitor row is a training set and folds 1--5 are",
        "%   what the five CV members collectively trained on.",
        f"%   The full realized corpus is {ds['realized_label_sets_all_folds']:,} label "
        f"sets; it additionally contains fold 0 ({ds['excluded_from_training']['fold_0']:,},",
        f"%   the UCBShift-200 exclusion set) and fold 6 "
        f"({ds['excluded_from_training']['fold_6']:,}, held out), which no",
        "%   member trains on. Quoting the full corpus against a competitor training set",
        "%   would be a unit mismatch.",
        "%   Unit caveat: our count is label sets (deposition x entity x shift list); the",
        "%   competitor rows are proteins as their authors counted them. The distinct-BMRB",
        f"%   figure ({ds['train_bmrb_entries']:,}) is the closer analogue if one unit must "
        "be chosen.",
        f"%   Ratio against UCBShift 2.0's training set ("
        f"{sized['UCBShift 2.0']['training set']}): "
        f"{ds['train_label_sets'] / ucb2_train:.1f}x on label sets, "
        f"{ds['train_bmrb_entries'] / ucb2_train:.1f}x on distinct depositions.",
        f"%   SHIFTY+'s {sized['SHIFTY+']['transfer library']} is a transfer library",
        "%   searched at prediction time, not a trained-on set, so it does not bound the",
        "%   claim as scoped.",
        "",
    ]
    return L


def main():
    rows = corpora_rows()
    sized = {m: {role: size for role, size, _ in e} for m, e in rows}
    # The comparator the ratio in the roster's facts is taken against: UCBShift 2.0's
    # training set, read back off the table and never restated.
    ucb2_train = int(sized["UCBShift 2.0"]["training set"].split()[0].replace(",", ""))
    ds = deepshift_corpus()

    frozen.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = frozen.OUT_DIR / f"{DEPLOYED_STEM}.tex"
    path.write_text("\n".join(corpora_document(rows, ds, sized, ucb2_train)) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
