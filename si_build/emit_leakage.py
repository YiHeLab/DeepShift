#!/usr/bin/env python3
"""Build si_leakage.tex: our fold partition and UCBShift's split, side by side.

Folds 1-6, the folds the seeded draw separates, are checked against the eight criteria
they were built from, which puts a zero in every cell (fold 0 is filled by rule, is not
separated from the others, and is not in the build's edge tables); UCBShift's published 647-train / 200-test chain split is
measured on the same eight criteria in the columns beside it, so the zeroes have
something to sit against.

Our half is recomputed here from the build's own edge tables -- nothing is transcribed
from prose. Their half is read from ucb_split_criteria.json, written by measure_ucb_split.py in
this directory, which measures the same eight criteria through the same MMseqs function
the production fold builder calls.

Writes: si_leakage.tex.

Data inputs that live outside the release repo are named by environment variable and have
no default; output goes to si_common.OUT.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import si_common
from si_common import tex_thousands as thousands

# The experiment record holding this table's small input (ucb_split_criteria.json). It is
# not release code, so it is resolved by environment and never copied into the release repo.
ARTIFACTS_DIR = si_common.root("SI_LEAKAGE_ARTIFACTS_DIR")
# The edge tables are the build's own, so they follow DEEPSHIFT_RUN with every other
# data-side table, and need no second variable for the same directory.
BUILD = si_common.DATA
FOLDMAP = BUILD / 'fold_assignments_labelset_identity90.json'
THEIRS = ARTIFACTS_DIR / 'ucb_split_criteria.json'
OUT = si_common.OUT

# family key -> (edge table, the row stub as the reader should read it)
ROWS = [
    ('same_bmrb', 'labelset_edges_same_bmrb.csv',
     r'Same BMRB deposition'),
    ('uniprot', 'labelset_edges_uniprot.csv',
     r'Same UniProt accession'),
    ('exact_token', 'labelset_edges_exact_token.csv',
     r'Same sequence, modified residues and all'),
    ('alignment', 'labelset_edges_alignment.csv',
     r'Same sequence, modified residues read as standard ones'),
    ('mmseqs_cov0', 'labelset_edges_mmseqs_cov0.csv',
     r'MMseqs2\cite{mmseqs2}, $\geq$80\% identity and $\geq$90\% coverage'),
    ('mmseqs_containment', 'labelset_edges_mmseqs_containment.csv',
     r'MMseqs2, $\geq$80\% identity, shorter one $\geq$90\% covered'),
    ('modeled_region', 'labelset_edges_modeled_region.csv',
     r'Same modelled region, $\geq$80\% identity and $\geq$90\% coverage'),
    ('partner_chain', 'labelset_edges_partner_chain.csv',
     r'Solved together in one PDB\cite{PDB} entry'),
]


# The folds the edge tables cover: the seeded draw's six. Fold 0 is assigned by rule.
AUDITED_FOLDS = {'1', '2', '3', '4', '5', '6'}

# The modelled-region gate the row stub states; their measurement must have used it.
MODELED_REGION_GATE = {'identity': 0.8, 'coverage': 0.9, 'coverage_modes': [0, 1]}


def check_modeled_region_gate(theirs_mmseqs):
    """Refuse unless their measurement used the modelled-region gate the row stub states."""
    gate = {'identity': theirs_mmseqs.get('modeled_region_identity'),
            'coverage': theirs_mmseqs.get('modeled_region_coverage'),
            'coverage_modes': theirs_mmseqs.get('modeled_region_coverage_modes')}
    if gate != MODELED_REGION_GATE:
        raise SystemExit(f'modelled-region gate of their measurement is {gate}, '
                         f'not {MODELED_REGION_GATE}, which the row stub states')


def ours(path):
    """Groups, groups whose members do not all sit in one fold, and the label sets
    inside those groups. Read straight out of the build's own edge table."""
    folds, members = defaultdict(set), defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            assert row['fold'] in AUDITED_FOLDS, (path, row['fold'])
            folds[row['edge_group']].add(int(row['fold']))
            members[row['edge_group']].append(row['label_set_id'])
    crossing = [g for g, f in folds.items() if len(f) > 1]
    caught = {ls for g in crossing for ls in members[g]}
    return {'groups': len(folds),
            'members': sum(len(v) for v in members.values()),
            'crossing': len(crossing),
            'held_out_entries_caught': len(caught),
            'folds_touched_by_crossing_groups':
                sorted(sorted(folds[g]) for g in crossing)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    theirs = json.loads(THEIRS.read_text())
    check_modeled_region_gate(theirs['mmseqs'])
    n_labelsets = sum(str(f) in AUDITED_FOLDS for f in json.loads(FOLDMAP.read_text()).values())

    our_rows = {family: ours(BUILD / filename) for family, filename, _ in ROWS}

    body = []
    for family, _filename, stub in ROWS:
        o, t = our_rows[family], theirs['families'][family]
        body.append(
            f"{stub} & {thousands(o['groups'])} & {o['crossing']} & "
            f"\\textbf{{{o['held_out_entries_caught']}}} & "
            f"{thousands(t['groups'])} & {t['crossing']} & "
            f"\\textbf{{{t['test_chains_affected']}}} \\\\")

    split = theirs['split']
    n_train, n_test = split['train_chains_read'], split['test_chains_read']

    banner = '\n'.join([
        '% Eight criteria, one grain, with UCBShift\'s published train/test split measured',
        '%   beside ours so our zeroes have something to sit against.',
        '% Our half is recomputed here from the build\'s own',
        '%   edge tables. Nothing is transcribed from prose.',
        '% Their half: measure_ucb_split.py in the same directory, which runs the same eight',
        '%   criteria over their 647 training and 200 test chains, calling the same MMseqs',
        '%   function and the same binary the production fold builder calls.',
        '% The row stubs name what the builder does: row three groups on the exact polymer',
        '%   sequence with modified residues kept, row four on the same sequence with',
        '%   modified residues read as their standard parent. There is no alignment and no',
        '%   threshold in either.',
        '% No caption or sentence in this file was written by a generative AI tool.',
    ])

    tex = [
        r'\begin{table}[H]\centering\small',
        r'\caption{The same linkage checks run on both splits}',
        r'\label{tab:leakage}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lrrrrrr}',
        r'\toprule',
        (r' & \multicolumn{3}{c}{This work: ' + thousands(n_labelsets) +
         r' label sets in folds 1--6} & '
         r"\multicolumn{3}{c}{UCBShift's\cite{UCBShift} published split: " + thousands(n_train) +
         r' training and ' + thousands(n_test) + r' test chains} \\'),
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
        (r'Linkage criterion & Groups & On both sides & Held-out entries'
         r' & Groups & On both sides & Held-out entries \\'),
        r'\midrule',
        *body,
        r'\bottomrule',
        r'\end{tabular}',
        r'}',
        r'\vspace{0.4em}',
        r'\begin{minipage}{\textwidth}\footnotesize\raggedright',
        (r'A group is a set of entries that one criterion links. It sits \emph{on both '
         r'sides} when its members do not all fall in one fold (this work), or when it '
         r'holds a training chain and a test chain at once (UCBShift). The bold column '
         r'counts the held-out entries inside those groups: label sets out of ' +
         thousands(n_labelsets) + r' here, test chains out of ' + thousands(n_test) +
         r' there.\par'),
        (r'The left-hand columns check the shipped fold assignment against the criteria '
         r"it was built from; the right-hand columns measure UCBShift's published split "
         r'against the same eight.\par'),
        r'\end{minipage}',
        r'\end{table}',
        '',
    ]

    si_common.emit(
        'si_leakage', '\n'.join(tex), banner=banner, script='emit_leakage.py',
        placeholder_caption=False)

    for family, _, _stub in ROWS:
        o, t = our_rows[family], theirs['families'][family]
        print(f"{family:20s} ours {o['groups']:6d}/{o['crossing']}/"
              f"{o['held_out_entries_caught']}   theirs {t['groups']:5d}/"
              f"{t['crossing']}/{t['test_chains_affected']}")


if __name__ == '__main__':
    main()
