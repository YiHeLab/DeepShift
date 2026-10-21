"""The literature roster is ``\\input`` inside a section of the supplement, so it must be
one float and nothing else -- no sectioning command, no prose, no second table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))

import si_tables_literature as lit

CORPORA_ROWS = [
    ("SPARTA+", [("training set", "580 proteins", "Shen and Bax, Methods"),
                 ("test set", "11 proteins", "Shen and Bax, Table~1")]),
    ("UCBShift 2.0", [("training set", "851 protein structures", "Ptaszek, SI Table~S3"),
                      ("transfer library", "1,968 chains", "released refDB")]),
    ("UCBShift", [("transfer library", "1,968 chains", "released refDB")]),
    ("SHIFTY+", [("transfer library", "1,903 proteins", "Han, SI Table~S7")]),
]
CORPUS = {
    "train_label_sets": 10149,
    "train_bmrb_entries": 9793,
    "realized_label_sets_all_folds": 12710,
    "excluded_from_training": {"fold_0": 1461, "fold_6": 1100},
}


def roster():
    sized = {m: {role: size for role, size, _ in e} for m, e in CORPORA_ROWS}
    return lit.corpora_document(CORPORA_ROWS, CORPUS, sized, 851)


def printed(document):
    """The lines that reach the page: a %-comment is not one of them."""
    return [line for line in document if not line.startswith("%")]


def test_the_roster_is_one_float_with_nothing_around_it():
    body = "\n".join(printed(roster()))
    assert body.count(r"\begin{table}") == 1
    assert r"\label{tab:lit_corpora}" in body
    assert r"\section" not in body
