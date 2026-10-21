"""How far each held-out structure is from the training set it was held out of.

The held-out fold's structures are measured here, from the run named by ``$DEEPSHIFT_RUN``:
every structure's logical entity is searched against the entity sequences of the training
folds with one MMseqs pass, and the best hit's identity over the aligned region places the
structure in one of seven bins. Nothing is read from a deployed table.

A second training set -- another predictor's published training chains, say -- is measured
on the same cohort with the same instrument when ``--against`` names its FASTA, so the two
columns differ in the training set alone.

Usage:
    DEEPSHIFT_RUN=/path/to/a/run python emit_train_distance.py \\
        [--against "name=/path/to/their_training_chains.fasta"]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import si_common

#: The second training set the table measures the same cohort against, named by a root so
#: the deployed table regenerates without an argument; ``--against`` adds or replaces it.
SECOND_SET_VAR = "SI_UCB_TRAINING_FASTA"
SECOND_SET_NAME = "UCBShift's training set"


def second_sets(requested: list[str]) -> list[str]:
    """The NAME=FASTA requests to measure: the ones given, else the root's.

    The deployed table measures UCBShift's training set too, so a missing root refuses
    rather than printing a table with one column fewer.
    """
    if requested:
        return list(requested)
    path = si_common.root(SECOND_SET_VAR)
    if not path.exists():
        raise SystemExit(f"REFUSING: ${SECOND_SET_VAR} names no file ({path}); the table "
                         "measures the second training set too (or name one with --against)")
    return [f"{SECOND_SET_NAME}={path}"]

# Run directly (`python si_build/emit_train_distance.py`), so make the package importable
# without asking the caller to set PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deepshift.data import census as corpus
from deepshift.data import folds as fold_stage
from deepshift.data.cluster import MMSEQS_COVERAGE, MMSEQS_IDENTITY
from deepshift.data.discover import load_fasta
from deepshift.data.sequence_clustering import run_best_hit_identity

ENTITY_FASTA = "entity_query.fasta"
# Bin edges on best-hit identity; a structure with no hit takes the first label.
EDGES = (0.30, 0.50, 0.65, 0.80, 0.90)
LABELS = ("No relative found", r"Below 30\%", r"30--50\%", r"50--65\%", r"65--80\%",
          r"80--90\%", r"90\% and above")


def bin_of(identity: float | None) -> int:
    if identity is None:
        return 0
    return 1 + sum(1 for edge in EDGES if identity >= edge)


def entity_of_structure(ledger, fold_of, fold: int) -> list[tuple[str, str]]:
    """``(label set, logical entity)`` for every structure realizing that fold."""
    block = ledger[ledger["label_set_id"].map(fold_of) == fold]
    return list(zip(block["label_set_id"], block["logical_entity_id"], strict=True))


def measure(structures, sequences, targets, *, coverage, cov_mode, threads):
    """Bin every structure by its entity's best hit in ``targets``.

    Returns the seven bin counts, the number of structures whose entity carries no
    sequence to search with, and the search command.
    """
    queries = {entity: sequences[entity] for _, entity in structures
               if entity in sequences}
    hits, command = run_best_hit_identity(queries, targets, coverage=coverage,
                                          cov_mode=cov_mode, threads=threads)
    counts = [0] * len(LABELS)
    unmeasured = 0
    at_split_rule = 0
    for _, entity in structures:
        if entity not in sequences:
            unmeasured += 1
            counts[0] += 1
            continue
        hit = hits.get(entity)
        identity = None if hit is None else hit["identity"]
        counts[bin_of(identity)] += 1
        if (hit is not None and identity >= MMSEQS_IDENTITY
                and hit["query_coverage"] >= MMSEQS_COVERAGE):
            at_split_rule += 1
    return {"counts": counts, "unmeasured": unmeasured,
            "n_at_the_split_rule": at_split_rule, "n_targets": len(targets),
            "n_queries_searched": len(queries), "command": si_common.portable(command),
            "coverage_floor": coverage, "cov_mode": cov_mode}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=si_common.root("DEEPSHIFT_RUN"),
                    help="run root; the cohort and both sequence sets come from its data")
    ap.add_argument("--against", action="append", default=[], metavar="NAME=FASTA",
                    help=f"a second training set to measure the same cohort against; "
                         f"without it the FASTA {SECOND_SET_VAR} names is measured")
    ap.add_argument("--coverage", type=float, default=0.50,
                    help="query-coverage floor a hit must reach to be reported")
    ap.add_argument("--cov-mode", type=int, default=2,
                    help="MMseqs coverage mode the floor is applied in")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    data_dir = args.run / "data"
    ledger = corpus.read_realizations(data_dir)
    fold_of = corpus.read_fold_map(data_dir)
    corpus.bind_fold_map(fold_of, set(ledger["label_set_id"]))
    sequences = load_fasta(data_dir / ENTITY_FASTA)
    if not sequences:
        raise SystemExit(f"{data_dir / ENTITY_FASTA} carries no entity sequence")

    held_out = entity_of_structure(ledger, fold_of, fold_stage.HOLDOUT_FOLD)
    training_folds = tuple(range(1, fold_stage.N_FOLDS + 1))
    training_entities = {entity for label_set, entity in
                         zip(ledger["label_set_id"], ledger["logical_entity_id"],
                             strict=True)
                         if fold_of[label_set] in training_folds}
    if training_entities & {entity for _, entity in held_out}:
        raise SystemExit("a logical entity is both held out and in a training fold, so "
                         "the fold map and the ledger disagree about the split")
    ours = {entity: sequences[entity] for entity in sorted(training_entities)
            if entity in sequences}

    columns = [("our training folds", measure(
        held_out, sequences, ours, coverage=args.coverage, cov_mode=args.cov_mode,
        threads=args.threads))]
    for request in second_sets(args.against):
        name, _, path = request.partition("=")
        if not path:
            raise SystemExit(f"--against wants NAME=FASTA, not {request!r}")
        other = load_fasta(Path(path))
        if not other:
            raise SystemExit(f"{path} carries no sequence")
        columns.append((name, measure(held_out, sequences, other,
                                      coverage=args.coverage, cov_mode=args.cov_mode,
                                      threads=args.threads)))

    total = len(held_out)
    banner = "\n".join([
        f"%   cohort: every structure realizing fold {fold_stage.HOLDOUT_FOLD} "
        f"({total} structures)",
        "%   instrument: " + columns[0][1]["command"],
        "% The caption is a short title; the facts it must assert are listed in the",
        "% \\note{} beside the float.",
    ])

    column_spec = "l" + "rr" * len(columns)
    lines = [
        r"\begin{table}[H]\centering\small",
        r"\caption{Distance from every held-out structure to a training set}",
        r"\label{tab:train_distance}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + column_spec + "}",
        r"\toprule",
        " & " + " & ".join(r"\multicolumn{2}{c}{against " + name + "}"
                           for name, _ in columns) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                 for i in range(len(columns))),
        "Best identity to that training set" + " & Structures & (\\%)" * len(columns)
        + r" \\",
        r"\midrule",
    ]
    for index, label in enumerate(LABELS):
        cells = []
        for _, column in columns:
            count = column["counts"][index]
            cells += [si_common.tex_thousands(count), f"{100 * count / total:.1f}"]
        if index == len(LABELS) - 1:
            cells = [r"\textbf{" + cell + "}" for cell in cells]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        "Total & " + " & ".join([si_common.tex_thousands(total), "100.0"] * len(columns))
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        "}",
        r"\vspace{0.4em}",
        r"\begin{minipage}{\textwidth}\footnotesize\raggedright",
    ]
    unmeasured = ", ".join(
        f"{si_common.tex_thousands(column['unmeasured'])} against {name}"
        for name, column in columns)
    lines += [
        "Of the first row, these structures are unmeasured, not measured at zero, "
        "because the sequence file carries no entry for their entity: "
        + unmeasured + r".\par",
        "Identity is measured over the aligned region alone, at a query-coverage floor of "
        + f"{args.coverage:.2f}" + ". Of the structures measured against our training "
        "folds, " + si_common.tex_thousands(columns[0][1]["n_at_the_split_rule"])
        + " reach the identity and query coverage the split rule groups "
        "components on ($\\geq$"
        + f"{MMSEQS_IDENTITY:.2f}" + " identity at $\\geq$" + f"{MMSEQS_COVERAGE:.2f}"
        + r" coverage).\par",
        r"\end{minipage}",
        r"\end{table}",
    ]
    tex = "\n".join(lines) + "\n"

    si_common.emit(
        "si_train_distance", tex, banner=banner, script="emit_train_distance.py",
        placeholder_caption=False)
    print(tex)


if __name__ == "__main__":
    main()
