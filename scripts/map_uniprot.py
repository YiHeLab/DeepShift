#!/usr/bin/env python
"""Map BMRB depositions to UniProt accessions: the procedure behind the deposited snapshot.

The build never calls a web service for UniProt. It reads the frozen BMRB-to-UniProt
mapping (``--uniprot-snapshot``, deposited with the dataset), and this script is how that
mapping was made. It queries live services (BMRB, EBI SIFTS, UniProt ID mapping, EBI
BLAST), so running it today gives a mapping that differs where those services have
changed; replicate the paper from the deposited snapshot.

Two commands, run in this order:

``links``
    Each deposition's own UniProt cross-reference in its BMRB entry (the first
    ``Database_code`` in ``UNIPROT_DB_NAMES`` with an accession). Writes the baseline
    mapping, ``bmrb_uniprot_mapping.json`` in the snapshot.
``augment``
    The depositions the baseline leaves unmapped, in three steps: the UniProt
    accessions EBI's SIFTS table gives for the PDB entries linked to the deposition
    (``pairs.csv``); UniProt's ID-mapping service for linked PDB entries SIFTS does not
    list; then an EBI NCBI-BLAST search of the deposition's sequence against UniProtKB,
    taking the top hit when it aligns at 70 % identity or more. Where a deposition maps
    to several accessions, the alphabetically first is kept. Writes
    ``new_uniprot_mappings.json``, rewritten after each step and every 50 BLAST results,
    so an interrupted run resumes from it.

Both write one JSON object, BMRB id -> accession. The deposited snapshot was built up by
this procedure in rounds from March to July 2026, each round keeping the accessions earlier
rounds had found: it holds the accumulated baseline and one ``augment`` output, both of
7 July 2026, and a further ``augment`` run captured on 14 July 2026. The build's
``uniprot_snapshot`` stage merges the three.
"""
import argparse
import gzip
import json
import re
import shutil
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

BMRB_ENTRY_URL = "https://api.bmrb.io/v2/entry/{}?format=json"
SIFTS_URL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz"
IDMAPPING_URL = "https://rest.uniprot.org/idmapping"
BLAST_URL = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"

#: Database names under which BMRB entries cite a UniProt accession.
UNIPROT_DB_NAMES = {"UNIPROT", "UNP", "SP", "SWISSPROT", "SWISS-PROT", "TREMBL"}
UNIPROT_ACC_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
AA_3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
MIN_BLAST_IDENTITY = 70.0
MIN_BLAST_LENGTH = 20


def write_json(mapping, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True))


def bmrb_ids(shifts_csv):
    return set(pd.read_csv(shifts_csv, usecols=["bmrb_id"])["bmrb_id"].astype(str))


# ---------------------------------------------------------------------------- links

def fetch_entry(bmrb_id, retries=3):
    """The BMRB entry as JSON, or None when it does not exist or cannot be fetched."""
    request = urllib.request.Request(BMRB_ENTRY_URL.format(bmrb_id),
                                     headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None


def linked_accession(entry, bmrb_id):
    """The first UniProt accession the entry's database-link loops cite, or None."""
    body = entry.get(str(bmrb_id)) or entry.get("data")
    if body is None:
        body = next((v for v in entry.values() if isinstance(v, dict) and "saveframes" in v),
                    None)
    saveframes = body.get("saveframes", []) if isinstance(body, dict) else body or []
    for saveframe in saveframes:
        if not isinstance(saveframe, dict):
            continue
        for loop in saveframe.get("loops", []):
            db = acc = None                 # the last matching tag of each kind
            for i, tag in enumerate(t.split(".")[-1].lower() for t in loop.get("tags", [])):
                if tag == "database_code":
                    db = i
                if tag in ("accession_code", "entry_code"):
                    acc = i
            if db is None or acc is None:
                continue
            for row in loop.get("data", []):
                try:
                    name, accession = str(row[db]).strip().upper(), str(row[acc]).strip()
                except (IndexError, TypeError):
                    continue
                if name in UNIPROT_DB_NAMES and accession and accession != ".":
                    return accession
    return None


def links(args):
    mapping = {}
    for bmrb_id in sorted(bmrb_ids(args.shifts), key=int):
        entry = fetch_entry(bmrb_id)
        accession = linked_accession(entry, bmrb_id) if entry else None
        if accession:
            mapping[bmrb_id] = accession
    write_json(mapping, args.out)
    print(f"{len(mapping)} depositions carry a UniProt link -> {args.out}")


# -------------------------------------------------------------------------- augment

def sequences(shifts_csv, wanted):
    """Each deposition's one-letter sequence, in residue order, standard residues only."""
    table = pd.read_csv(shifts_csv, usecols=["bmrb_id", "residue_id", "residue_code"])
    table["bmrb_id"] = table["bmrb_id"].astype(str)
    table = table[table["bmrb_id"].isin(wanted)]
    out = {}
    for bmrb_id, group in table.groupby("bmrb_id"):
        group = group.drop_duplicates(subset=["residue_id"]).sort_values("residue_id")
        letters = "".join(AA_3_TO_1.get(code, "") for code in group["residue_code"])
        if letters:
            out[bmrb_id] = letters
    return out


def sifts_table(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = path.with_name(path.name + ".gz")
        urllib.request.urlretrieve(SIFTS_URL, archive)
        with gzip.open(archive, "rb") as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    sifts = pd.read_csv(path, comment="#", dtype=str)
    sifts.columns = [c.strip() for c in sifts.columns]
    pdb_to_uniprot = defaultdict(set)
    for pdb_id, accession in zip(sifts["PDB"].str.strip().str.lower(),
                                 sifts["SP_PRIMARY"].str.strip()):
        if pd.notna(accession) and accession:
            pdb_to_uniprot[pdb_id].add(accession)
    return pdb_to_uniprot


def assign(results, bmrb_to_pdbs, pdb_to_uniprot):
    """Map each deposition to the first of the accessions its PDB entries carry."""
    for bmrb_id, pdbs in bmrb_to_pdbs.items():
        if bmrb_id in results:
            continue
        accessions = set().union(*(pdb_to_uniprot.get(p, set()) for p in pdbs))
        if accessions:
            results[bmrb_id] = sorted(accessions)[0]


def idmapping(pdb_ids):
    """UniProt ID mapping, PDB -> UniProtKB, in batches of 100 PDB ids."""
    found = defaultdict(set)

    def collect(response):
        for item in response.json().get("results", []):
            target = item.get("to", {})
            accession = target.get("primaryAccession", "") if isinstance(target, dict) else ""
            if item.get("from") and accession:
                found[item["from"].lower()].add(accession)

    for start in range(0, len(pdb_ids), 100):
        batch = ",".join(p.upper() for p in pdb_ids[start:start + 100])
        try:
            job = requests.post(f"{IDMAPPING_URL}/run", timeout=30,
                                data={"from": "PDB", "to": "UniProtKB", "ids": batch})
            job_id = job.json().get("jobId") if job.status_code == 200 else None
            for _ in range(60 if job_id else 0):
                time.sleep(3)
                status = requests.get(f"{IDMAPPING_URL}/status/{job_id}", timeout=30,
                                      allow_redirects=False)
                if status.status_code == 303:
                    result = requests.get(status.headers.get("Location", ""), timeout=60)
                    if result.status_code == 200:
                        collect(result)
                    break
                try:
                    state = status.json().get("jobStatus")
                except ValueError:
                    continue
                if state == "FINISHED":
                    result = requests.get(f"{IDMAPPING_URL}/uniprotkb/results/{job_id}",
                                          timeout=60)
                    if result.status_code == 200:
                        collect(result)
                    break
                if state in ("FAILURE", "ERROR"):
                    break
        except (requests.RequestException, ValueError) as exc:
            print(f"  ID-mapping batch {start}: {exc}")
        time.sleep(0.5)
    return found


def blast(sequence, email):
    """The top UniProtKB hit of one EBI blastp search when it aligns at >= 70 % identity."""
    params = {"email": email, "program": "blastp", "database": "uniprotkb",
              "sequence": sequence, "stype": "protein", "matrix": "BLOSUM62",
              "exp": "1e-5", "alignments": "5"}
    job_id = None
    for _ in range(2):                      # one retry after a rate-limit refusal
        try:
            response = requests.post(f"{BLAST_URL}/run", data=params, timeout=60)
        except requests.RequestException:
            return None
        if response.status_code == 200:
            job_id = response.text.strip()
            break
        if response.status_code != 429:
            return None
        time.sleep(30)
    if job_id is None:
        return None
    for _ in range(120):
        time.sleep(5)
        try:
            status = requests.get(f"{BLAST_URL}/status/{job_id}", timeout=30).text.strip()
        except requests.RequestException:
            continue
        if status == "FINISHED":
            break
        if status in ("FAILURE", "ERROR", "NOT_FOUND"):
            return None
    else:
        return None
    try:
        hits = requests.get(f"{BLAST_URL}/result/{job_id}/json", timeout=60).json()["hits"]
    except (requests.RequestException, ValueError, KeyError):
        return None
    if not hits:
        return None
    best, hsps = hits[0], hits[0].get("hit_hsps", [])
    if hsps and hsps[0].get("hsp_identity", 0) < MIN_BLAST_IDENTITY:
        return None
    accession = best.get("hit_acc", "")
    return accession if UNIPROT_ACC_RE.match(accession) else None


def augment(args):
    baseline = json.loads(args.baseline.read_text())
    missing = bmrb_ids(args.shifts) - set(baseline)
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    print(f"{len(missing)} depositions without a UniProt link; {len(results)} already mapped")

    pairs = pd.read_csv(args.pairs)
    bmrb_to_pdbs = {str(e): [p.strip().lower() for p in str(ids).split(",") if p.strip()]
                    for e, ids in zip(pairs["Entry_ID"], pairs["pdb_ids"])
                    if str(e) in missing and str(e) not in results}
    if bmrb_to_pdbs:
        pdb_to_uniprot = sifts_table(args.sifts)
        assign(results, bmrb_to_pdbs, pdb_to_uniprot)
        write_json(results, args.out)
        unlisted = sorted({p for b, pdbs in bmrb_to_pdbs.items() if b not in results
                           for p in pdbs if p not in pdb_to_uniprot})
        assign(results, bmrb_to_pdbs, idmapping(unlisted))
        write_json(results, args.out)
        print(f"after SIFTS and ID mapping: {len(results)} mapped")

    seqs = sequences(args.shifts, missing - set(results))
    queue = sorted((b for b, s in seqs.items() if len(s) >= MIN_BLAST_LENGTH),
                   key=lambda b: len(seqs[b]))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for bmrb_id in queue:
            futures[pool.submit(blast, seqs[bmrb_id], args.email)] = bmrb_id
            time.sleep(0.5)                 # EBI asks for no more than ~2 submissions/s
        for done, future in enumerate(as_completed(futures), 1):
            if accession := future.result():
                results[futures[future]] = accession
            if done % 50 == 0:
                write_json(results, args.out)
    write_json(results, args.out)
    print(f"{len(results)} of {len(missing)} mapped -> {args.out}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    link = commands.add_parser("links", help="UniProt links of the BMRB entries themselves")
    link.add_argument("--shifts", type=Path, required=True,
                      help="the build's chemical_shifts.csv (its bmrb_id column)")
    link.add_argument("--out", type=Path, required=True, help="bmrb_uniprot_mapping.json")
    more = commands.add_parser("augment", help="SIFTS, UniProt ID mapping and EBI BLAST")
    more.add_argument("--shifts", type=Path, required=True,
                      help="the build's chemical_shifts.csv")
    more.add_argument("--baseline", type=Path, required=True,
                      help="the mapping `links` wrote")
    more.add_argument("--pairs", type=Path, required=True, help="BMRB-to-PDB pairs.csv")
    more.add_argument("--sifts", type=Path, required=True,
                      help="SIFTS pdb_chain_uniprot.csv; downloaded from EBI when absent")
    more.add_argument("--email", required=True, help="contact address EBI's job API requires")
    more.add_argument("--workers", type=int, default=10, help="concurrent BLAST jobs")
    more.add_argument("--out", type=Path, required=True, help="new_uniprot_mappings.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    links(args) if args.command == "links" else augment(args)


if __name__ == "__main__":
    main()
