"""Build benchmarks/reference/competitor_same_protein.json: for every fold-6 structure, the
references in each competitor's transfer library that are the structure's own protein.

A retrieval arm reported at the same-protein level must not be able to retrieve any
deposition of the target protein. The lists the arms were first run with were keyed on the
task table: the query's BMRB, its UniProt component in this build, and references keyed to
its own PDB entry. That misses a library entry whose BMRB this build never mapped, or mapped
to an unreviewed accession while the entry's PDB chain carries the reviewed one (BMRB 27945
against UCBShift reference 4186: the query is 4186's sequence plus a three-residue tag). So a
reference is the same protein as a fold-6 structure when any of the following holds:

  bmrb      it is the structure's own BMRB entry;
  uniprot   one of its UniProt accessions is one of the structure's, or sits in the same
            UniRef100 cluster as one (the frozen map ``uniref100_map.json``: an unreviewed
            TrEMBL accession and the reviewed entry of the same sequence share a cluster);
  sequence  one of its sequences aligns to the structure's entity sequence or to the
            sequence of its chain at >= 95 % identity over >= 90 % of the shorter
            sequence (MMseqs2, exhaustive search).

Libraries, keyed by the BMRB id each tool's exclusion hook blocks on:

  ucbshift  the BLAST database UCBShift's Y module searches (``refDB.blastdb``; UCBShift 1.0
            and 2.0 ship the same one), dumped with ``blastdbcmd``. Entries are
            ``<bmrb>.<pdb><chain>``; the chain's SIFTS accessions are the entry's.
  shifty    what SHIFTY+ searches: the three FASTAs of ``shifty3/blastdb`` and every
            ``refdb/bmr<id>.cs`` (its ``#SEQ`` lines and the sequence its shift rows spell).

``exclude_bmrbs`` names, for every structure with a same-protein reference its task-table
list misses, that list plus every same-protein BMRB of the library; every other structure
keeps the task table's ``exclude_bmrbs`` (plus its own BMRB). The arms are run with it. ``tests/test_competitor_same_protein.py``
checks the frozen file and, where the competitor outputs are on disk, every hit log.

    DEEPSHIFT_RUN=... DEEPSHIFT_SIFTS_CSV=... DEEPSHIFT_REFDB_LIBRARY=... \\
    DEEPSHIFT_UCBSHIFT_BLASTDB=.../CSpred/refDB/refDB.blastdb BLASTDBCMD_BIN=... MMSEQS_BIN=... \\
        python -m benchmarks.build_competitor_exclusions [--fetch-uniref100]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from . import config
from .build_same_protein_roster import entry_sequences
from .score_by_source import REFERENCE

ID_MIN, COV_MIN = 0.95, 0.90
OUT_NAME = "competitor_same_protein.json"
UNIREF_NAME = "uniref100_map.json"
LIBRARIES = ("ucbshift", "shifty")
SEARCH = ["--exhaustive-search", "1", "-e", "1000", "--alignment-mode", "3",
          "--seq-id-mode", "0", "--max-seqs", "100000",
          "--format-output", "query,target,fident,qstart,qend,qlen,tstart,tend,tlen"]
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}
RULE = ("a library reference is the same protein as a fold-6 structure if it is the "
        "structure's BMRB entry, shares a UniProt accession or UniRef100 cluster with it, or "
        "aligns to its entity or chain sequence at >=95% identity over >=90% of the shorter "
        "sequence")


def passes(identity: float, qstart: int, qend: int, qlen: int,
           tstart: int, tend: int, tlen: int) -> bool:
    """The sequence criterion on one MMseqs2 alignment: identity over aligned columns, and
    the aligned span over the shorter of the two sequences."""
    cover = (qend - qstart + 1) / qlen if qlen <= tlen else (tend - tstart + 1) / tlen
    return identity >= ID_MIN and cover >= COV_MIN


def chain_sequence(pdb: str | Path, chain: str | None) -> str:
    """One-letter sequence of the first model's chain, from its CA records."""
    seq, seen, first = [], set(), None
    for line in Path(pdb).read_text(errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")) or line[12:16].strip() != "CA":
            continue
        ch = line[21]
        if chain in (None, "", "_"):
            first = ch if first is None else first
            if ch != first:
                continue
        elif ch != chain:
            continue
        key = (ch, line[22:27])
        if key in seen or line[17:20] not in AA3:
            continue
        seen.add(key)
        seq.append(AA3[line[17:20]])
    return "".join(seq)


def read_fasta(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        if line.startswith(">"):
            out.append([line[1:].split()[0] if line[1:].split() else "", ""])
        elif out:
            out[-1][1] += line.strip().upper()
    return [tuple(r) for r in out]


def ucbshift_library(blastdb: str, blastdbcmd: str) -> dict:
    """bmrb -> {"seqs": [...], "pdb_chains": [...]} from UCBShift's BLAST database."""
    dump = subprocess.run([blastdbcmd, "-db", blastdb, "-entry", "all"], check=True,
                          capture_output=True, text=True).stdout
    lib = defaultdict(lambda: {"seqs": [], "pdb_chains": []})
    for name, seq in read_fasta(dump):
        bmrb, pdb_chain = name.split(".", 1)
        lib[bmrb]["pdb_chains"].append(pdb_chain)
        if seq not in lib[bmrb]["seqs"]:
            lib[bmrb]["seqs"].append(seq)
    return dict(lib)


def shifty_library(refdb: Path) -> dict:
    lib = defaultdict(lambda: {"seqs": [], "pdb_chains": []})
    for kind in "chn":
        fasta = refdb.parent / "blastdb" / f"refdb.{kind}.fasta"
        if fasta.is_file():
            for name, seq in read_fasta(fasta.read_text()):
                bmrb = re.match(r"bmr(\d+)\.cs", name).group(1)
                if seq and seq not in lib[bmrb]["seqs"]:
                    lib[bmrb]["seqs"].append(seq)
    for f in sorted(refdb.glob("bmr*.cs")):
        for seq in entry_sequences(f):
            if seq not in lib[f.name[3:-3]]["seqs"]:
                lib[f.name[3:-3]]["seqs"].append(seq)
    return dict(lib)


def sifts(path: Path):
    by_chain, by_entry = defaultdict(set), defaultdict(set)
    with open(path) as fh:
        next(fh)
        for r in csv.DictReader(fh):
            acc = r["SP_PRIMARY"].split("-")[0]
            by_chain[(r["PDB"].upper(), r["CHAIN"])].add(acc)
            by_entry[r["PDB"].upper()].add(acc)
    return by_chain, by_entry


def pdb_uniprots(table, pdb: str, chain: str) -> set:
    """SIFTS accessions of a chain; a blank or unknown chain takes the whole entry."""
    by_chain, by_entry = table
    found = by_chain.get((pdb.upper(), chain), set()) if chain not in ("", "_", None) else set()
    return set(found) or set(by_entry.get(pdb.upper(), set()))


def fetch_uniref100(accessions: list[str]) -> dict:
    """accession -> UniRef100 cluster id, from the UniProt ID-mapping service."""
    api, out = "https://rest.uniprot.org/idmapping", {}
    for i in range(0, len(accessions), 500):
        data = urllib.parse.urlencode({"from": "UniProtKB_AC-ID", "to": "UniRef100",
                                       "ids": ",".join(accessions[i:i + 500])}).encode()
        job = json.load(urllib.request.urlopen(f"{api}/run", data=data, timeout=120))["jobId"]
        for _ in range(300):
            status = json.load(urllib.request.urlopen(f"{api}/status/{job}", timeout=120))
            if "results" in status or status.get("jobStatus") == "FINISHED":
                break
        text = urllib.request.urlopen(f"{api}/stream/{job}?format=tsv", timeout=300).read()
        for line in text.decode().splitlines()[1:]:
            acc, cluster = line.split("\t")[:2]
            out[acc] = cluster
    return {a: out.get(a) for a in accessions}


def sequence_matches(queries: dict, libs: dict, mmseqs: str) -> dict:
    """{(library, query structure, bmrb)} for every alignment meeting the rule."""
    hits = set()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "q.fasta").write_text("".join(f">{sid}|{i}\n{s}\n" for sid, seqs in
                                             queries.items() for i, s in enumerate(seqs)))
        for name, lib in libs.items():
            (tmp / f"{name}.fasta").write_text("".join(
                f">{b}|{i}\n{s}\n" for b, e in lib.items() for i, s in enumerate(e["seqs"])))
            subprocess.run([mmseqs, "easy-search", tmp / "q.fasta", tmp / f"{name}.fasta",
                            tmp / f"{name}.m8", tmp / "w", *SEARCH], check=True,
                           capture_output=True)
            for line in (tmp / f"{name}.m8").read_text().splitlines():
                q, t, fident, *span = line.split("\t")
                if passes(float(fident), *map(int, span)):
                    hits.add((name, q.rsplit("|", 1)[0], t.split("|")[0]))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch-uniref100", action="store_true",
                    help=f"refresh reference/{UNIREF_NAME} from rest.uniprot.org first")
    ns = ap.parse_args()
    build = config.current()
    blastdb = os.environ.get("DEEPSHIFT_UCBSHIFT_BLASTDB") or str(
        build.refdb_shifts.parent / "refDB.blastdb")
    blastdbcmd = os.environ.get("BLASTDBCMD_BIN") or "blastdbcmd"
    mmseqs = os.environ.get("MMSEQS_BIN", "mmseqs")
    libs = {"ucbshift": ucbshift_library(blastdb, blastdbcmd),
            "shifty": shifty_library(build.refdb_library)}

    with open(build.out_root / "fold6_tasks.tsv") as fh:
        tasks = {t["structure_id"]: t for t in csv.DictReader(fh, delimiter="\t")}
    ledger = defaultdict(set)
    with open(build.uniprot_ledger) as fh:
        for row in csv.DictReader(fh):
            if row["source_final"] == "True" and row["accession"]:
                ledger[row["bmrb_id"]].add(row["accession"].split("-")[0])
    table = sifts(build.sifts)
    csv.field_size_limit(10**9)
    with open(build.data / "entity_catalog.csv") as fh:
        entity = {}
        for r in csv.DictReader(fh):
            entity.setdefault(r["label_set_id"], r["alignment_sequence"].upper())

    query_acc, query_seqs = {}, {}
    for sid, t in tasks.items():
        acc = {a.split("-")[0] for a in t["uniprots"].split(",") if a} | ledger[t["bmrb_id"]]
        if t["source"] == "af":
            acc.add(t["accession"].split("-")[0])
        else:
            acc |= pdb_uniprots(table, t["accession"][:4], t["chain"])
        query_acc[sid] = acc
        seqs = [entity.get(t["label_set_id"], ""), chain_sequence(t["pdb_path"], t["chain"])]
        query_seqs[sid] = [s for i, s in enumerate(seqs) if s and s not in seqs[:i]]
    ref_acc = {}
    for name, lib in libs.items():
        ref_acc[name] = {}
        for bmrb, e in lib.items():
            chains = e["pdb_chains"] or libs["ucbshift"].get(bmrb, {}).get("pdb_chains", [])
            acc = set(ledger[bmrb])
            for pc in chains:
                acc |= pdb_uniprots(table, pc[:4], pc[4:5])
            ref_acc[name][bmrb] = acc

    uniref_path = REFERENCE / UNIREF_NAME
    wanted = sorted(set().union(*query_acc.values(), *(a for lib in ref_acc.values()
                                                       for a in lib.values())))
    if ns.fetch_uniref100:
        uniref_path.write_text(json.dumps({"service": "https://rest.uniprot.org/idmapping",
                                           "to": "UniRef100",
                                           "map": fetch_uniref100(wanted)}, indent=1) + "\n")
    uniref = json.loads(uniref_path.read_text())["map"]
    unmapped = [a for a in wanted if a not in uniref]
    if unmapped:
        raise SystemExit(f"{len(unmapped)} accessions are not in {uniref_path.name} "
                         f"(e.g. {unmapped[:3]}); rerun with --fetch-uniref100")

    def clusters(accs):
        return {uniref[a] for a in accs if uniref.get(a)}

    by_sequence = sequence_matches(query_seqs, libs, mmseqs)
    evidence = {name: {} for name in libs}
    exclude = {name: {} for name in libs}
    for sid, t in sorted(tasks.items()):
        qa, qc = query_acc[sid], clusters(query_acc[sid])
        for name, lib in libs.items():
            same = {}
            for bmrb in lib:
                crit = []
                if bmrb == t["bmrb_id"]:
                    crit.append("bmrb")
                if qa & ref_acc[name][bmrb]:
                    crit.append("uniprot")
                elif qc & clusters(ref_acc[name][bmrb]):
                    crit.append("uniref100")
                if (name, sid, bmrb) in by_sequence:
                    crit.append("sequence")
                if crit:
                    same[bmrb] = crit
            if same:
                evidence[name][sid] = dict(sorted(same.items(), key=lambda kv: int(kv[0])))
            listed = {b for b in t["exclude_bmrbs"].split(",") if b} | {t["bmrb_id"]}
            if set(same) - listed:
                exclude[name][sid] = sorted(listed | set(same), key=int)

    out = REFERENCE / OUT_NAME
    out.write_text(json.dumps({
        "what": "per fold-6 structure, the references of each competitor transfer library "
                "that are the structure's own protein, and the exclusion list that blocks them",
        "rule": RULE,
        "thresholds": {"identity": ID_MIN, "coverage_of_shorter": COV_MIN},
        "libraries": {name: {"entries": len(lib)} for name, lib in libs.items()},
        "arms": {"ucbshift": ["ucbshift_full_sameprot", "ucbshift_full_selfexcl",
                              "ucbshift2_sameprot", "ucbshift2_selfexcl", "ucbshift_full_leaky"],
                 "shifty": ["shiftx2_sameprot", "shiftx2_selfexcl", "shiftx2_leaky"]},
        "n_structures": len(tasks),
        "n_structures_with_same_protein_reference": {
            name: len(v) for name, v in evidence.items()},
        "evidence": evidence,
        "exclude_bmrbs": exclude,
    }, indent=1) + "\n")
    print(f"{out}: " + ", ".join(f"{name} {len(v)} structures with a same-protein reference"
                                 for name, v in evidence.items()))


if __name__ == "__main__":
    main()
