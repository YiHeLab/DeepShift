"""Build benchmarks/reference/plmcs_same_protein.json: the fold-6 label sets PLM-CS trained on.

PLM-CS trained on the released RefDB snapshot (``DEEPSHIFT_REFDB_LIBRARY``, one
``bmr<id>.cs`` per entry). A fold-6 label set is the same protein as one of those entries
when any of the following holds:

  bmrb      its own BMRB accession is the entry's;
  uniprot   it shares a UniProt accession with the entry under this build's mapping
            (``data/uniprot_mapping_candidates.csv``, ``source_final`` rows, plus the task
            table's ``uniprots`` column);
  sequence  its entity sequence aligns to a sequence of the entry at >= 95 % identity over
            >= 90 % of the shorter sequence (MMseqs2, exhaustive search).

An entry's sequences are each distinct ``#SEQ`` line plus the sequence its own shift rows
spell, gaps as X: some files repeat one entity's ``#SEQ`` under a second entity and four
carry none, so the rows are what was trained on.

    DEEPSHIFT_RUN=... DEEPSHIFT_REFDB_LIBRARY=... MMSEQS_BIN=... \\
        python -m benchmarks.build_same_protein_roster
"""
import csv
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from . import config
from .score_by_source import REFERENCE

ID_MIN, COV_MIN = 0.95, 0.90
SEARCH = ["--exhaustive-search", "1", "-e", "1000", "--alignment-mode", "3",
          "--seq-id-mode", "0", "--max-seqs", "100000",
          "--format-output", "query,target,fident,qstart,qend,qlen,tstart,tend,tlen"]
RULE = ("a fold-6 label set is withheld from PLM-CS if any RefDB entry PLM-CS trained on "
        "is the same BMRB, shares a UniProt accession under our build's mapping, or aligns "
        "at >=95% identity over >=90% of the shorter sequence")


def entry_sequences(path: Path) -> list[str]:
    seqs, rows = [], {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#SEQ:"):
            seq = line.split(":", 1)[1].strip().upper()
            if seq and seq not in seqs:
                seqs.append(seq)
        elif line and not line.startswith("#"):
            field = line.split(",")
            if len(field) >= 2 and field[0].lstrip("-").isdigit() and len(field[1]) == 1:
                rows.setdefault(int(field[0]), field[1].upper())
    if rows:
        spelled = "".join(rows.get(i, "X") for i in range(min(rows), max(rows) + 1))
        if spelled not in seqs:
            seqs.append(spelled)
    return seqs


def sequence_matches(entries: dict, label_sets: dict, mmseqs: str) -> set:
    """(label set, BMRB) pairs aligning at the identity and coverage the rule asks."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "refdb.fasta").write_text("".join(
            f">r{b}_{i}\n{s}\n" for b, seqs in entries.items() for i, s in enumerate(seqs)))
        (tmp / "fold6.fasta").write_text("".join(f">{ls}\n{s}\n"
                                                 for ls, s in label_sets.items()))
        subprocess.run([mmseqs, "easy-search", tmp / "fold6.fasta", tmp / "refdb.fasta",
                        tmp / "hits.m8", tmp / "work", *SEARCH],
                       check=True, capture_output=True)
        hits = (tmp / "hits.m8").read_text().splitlines()
    pairs = set()
    for line in hits:
        q, t, fident, qs, qe, ql, ts, te, tl = line.split("\t")
        qs, qe, ql, ts, te, tl = map(int, (qs, qe, ql, ts, te, tl))
        cover = (qe - qs + 1) / ql if ql <= tl else (te - ts + 1) / tl
        if float(fident) >= ID_MIN and cover >= COV_MIN:
            pairs.add((q, t[1:].rsplit("_", 1)[0]))
    return pairs


def main() -> None:
    build = config.current()
    library = build.refdb_library
    entries = {f.name[3:-3]: entry_sequences(f) for f in sorted(library.glob("bmr*.cs"))}
    with open(build.out_root / "fold6_tasks.tsv") as fh:
        tasks = list(csv.DictReader(fh, delimiter="\t"))
    bmrb_of = {t["label_set_id"]: t["bmrb_id"] for t in tasks}

    uniprot = defaultdict(set)
    with open(build.uniprot_ledger) as fh:
        for row in csv.DictReader(fh):
            if row["source_final"] == "True" and row["accession"]:
                uniprot[row["bmrb_id"]].add(row["accession"].split("-")[0])
    for t in tasks:
        uniprot[t["bmrb_id"]] |= {a.split("-")[0] for a in t["uniprots"].split(",") if a}

    csv.field_size_limit(10**9)
    with open(build.data / "entity_catalog.csv") as fh:
        sequences = {r["label_set_id"]: r["alignment_sequence"].upper()
                     for r in csv.DictReader(fh) if r["label_set_id"] in bmrb_of}
    by_sequence = sequence_matches(entries, sequences,
                                   os.environ.get("MMSEQS_BIN", "mmseqs"))

    evidence = {}
    for ls, bmrb in sorted(bmrb_of.items()):
        matched = {"bmrb": [bmrb] if bmrb in entries else [],
                   "uniprot": sorted(e for e in entries
                                     if uniprot.get(e) and uniprot[e] & uniprot[bmrb]),
                   "sequence": sorted(e for q, e in by_sequence if q == ls)}
        if any(matched.values()):
            evidence[ls] = {"bmrb_id": bmrb,
                            "matched_refdb": {k: v for k, v in matched.items() if v}}
    out = REFERENCE / config.ARM_SAME_PROTEIN_ROSTERS["plmcs"]
    out.write_text(json.dumps({
        "what": "fold-6 label sets of the same protein as an entry PLM-CS trained on",
        "rule": RULE,
        "thresholds": {"identity": ID_MIN, "coverage_of_shorter": COV_MIN},
        "corpus": "the released RefDB snapshot SHIFTX2 ships (DEEPSHIFT_REFDB_LIBRARY)",
        "corpus_files": len(entries),
        "n_label_sets": len(evidence),
        "n_bmrb": len({e["bmrb_id"] for e in evidence.values()}),
        "label_set_ids": sorted(evidence),
        "evidence": evidence,
    }, indent=1) + "\n")
    print(f"{out}: {len(evidence)} label sets")


if __name__ == "__main__":
    main()
