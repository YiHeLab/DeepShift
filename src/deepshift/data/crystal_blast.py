#!/usr/bin/env python3
"""Crystal candidate BLAST stage.

The criterion is the one the paper states: the Protein Data Bank is searched for X-ray
crystal structures with identity >= 0.95 to their BMRB counterparts.

The stage takes no precomputed list. It builds the query FASTA from the current
chemical_shifts.csv, downloads pdb_seqres.txt.gz and builds the BLAST DB (reusing one
built in the same run), BLASTs every BMRB query with local blastp, and emits
blast_full.tsv (outfmt 6 plus qlen/slen) for the census stage.

Usage:
  python -m deepshift.data.crystal_blast \
      --shifts_file  data/chemical_shifts.csv \
      --work_dir     data/crystal_blast \
      --threads      20
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import time
import urllib.request

BLAST_PLUS_VERSION = "2.17.0+"

PDB_SEQRES_URL = os.environ.get(
    'PDB_SEQRES_URL',
    'https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz')
def _configured_executable(env_name, command):
    configured = os.environ.get(env_name)
    if configured:
        return configured
    found = shutil.which(command)
    if found:
        return found
    return command


BLASTP_BIN = _configured_executable('BLASTP_BIN', 'blastp')
MAKEBLASTDB_BIN = _configured_executable('MAKEBLASTDB_BIN', 'makeblastdb')


def require_blast_version(executable):
    """Require the pinned BLAST+ release."""
    try:
        first_line = subprocess.run(
            [executable, "-version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError) as exc:
        raise RuntimeError(f"BLAST+ {BLAST_PLUS_VERSION} is required: {executable}") from exc
    if BLAST_PLUS_VERSION not in first_line:
        raise RuntimeError(
            f"BLAST+ version mismatch: require {BLAST_PLUS_VERSION}, got {first_line!r}")
    return first_line


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


MIN_QUERY_LEN = 5

# Label-set payload fields carried alongside each query sequence; FASTA
# ids are label_set_ids and the drop log keeps the full key so nothing is silently
# reduced back to a bare BMRB.
IDENTITY_FIELDS = ('bmrb_id', 'entity_id', 'shift_list_id', 'entity_assembly_id',
                   'exact_polymer_sequence_sha256', 'logical_entity_id')


def entity_sequence_map(shifts_file, catalog_file=None):
    """Build one full query sequence per selected label set: the label set's declared
    entity sequence (one residue per polymer position), from the
    ``entity_catalog.csv`` full-sequence sidecar keyed by ``label_set_id``
    -- never the observed-only concatenation of shift-bearing rows
    (an entity labelled at positions 1, 3, 10 BLASTs its whole polymer, or a
    fragment/sibling chain could pass while disagreeing with most of it).

    ``catalog_file`` defaults next to ``shifts_file`` and raises when the sidecar is
    absent, rather than reconstructing it. Returns:

      seqs  = {label_set_id: {'sequence': str, <IDENTITY_FIELDS>}}
      drops = [{'label_set_id', <identity fields>, 'reason', 'seq_len'} ...]
    """
    from .fetch import default_entity_catalog_path, read_entity_catalog
    catalog_file = catalog_file or default_entity_catalog_path(shifts_file)
    if not os.path.isfile(catalog_file):
        raise ValueError(
            f'entity_catalog.csv full-sequence sidecar not found ({catalog_file}); '
            'the entity build must consume the catalog emitted by fetch. Refusing '
            'to reconstruct queries from observed-only shift rows.')
    log(f'reading entity catalog: {catalog_file}')
    catalog = read_entity_catalog(catalog_file)

    seqs, drops = {}, []
    for lsid in sorted(catalog):
        rec = catalog[lsid]
        ident = {f: str(rec.get(f, '')) for f in IDENTITY_FIELDS}
        seq = str(rec['alignment_sequence']).upper()
        if len(seq) < MIN_QUERY_LEN:
            drops.append({'label_set_id': lsid, **ident,
                          'reason': 'too_short', 'seq_len': len(seq)})
            continue
        if seq.count('X') == len(seq):
            drops.append({'label_set_id': lsid, **ident,
                          'reason': 'all_X', 'seq_len': len(seq)})
            continue
        seqs[lsid] = {'sequence': seq, **ident}
    return seqs, drops


def build_query_fasta(shifts_file, out_fasta, drops_out=None):
    """Write the label-set-grained query FASTA and (optionally) a drop log; FASTA
    ids are label_set_ids (entity_sequence_map), drop log keeps the full identity key."""
    seqs, drops = entity_sequence_map(shifts_file)
    with open(out_fasta, 'w') as fh:
        for lsid, rec in seqs.items():
            fh.write(f'>{lsid}\n{rec["sequence"]}\n')
    if drops_out is not None:
        with open(drops_out, 'w') as fh:
            json.dump({'n_written': len(seqs), 'n_dropped': len(drops),
                       'drops': drops}, fh, indent=2)
    log(f'query FASTA: {len(seqs)} label-set sequences written, '
        f'{len(drops)} label sets dropped (logged)')
    return len(seqs)


def ensure_seqres_db(work_dir, threads):
    """Download fresh pdb_seqres + makeblastdb. Returns DB prefix."""
    db_dir = os.path.join(work_dir, 'pdb_seqres_db')
    os.makedirs(db_dir, exist_ok=True)
    db_prefix = os.path.join(db_dir, 'pdb_seqres_prot')
    if os.path.isfile(db_prefix + '.pin'):
        log(f'BLAST DB already present this run: {db_prefix}.*  (reusing)')
        return db_prefix
    gz = os.path.join(db_dir, 'pdb_seqres.txt.gz')
    fasta_prot = os.path.join(db_dir, 'pdb_seqres_prot.fasta')
    log(f'downloading pdb_seqres: {PDB_SEQRES_URL}')
    urllib.request.urlretrieve(PDB_SEQRES_URL, gz)
    log(f'  downloaded {os.path.getsize(gz):,} bytes; gunzip + protein filter')
    n_all = n_prot = 0
    with gzip.open(gz, 'rt') as inp, open(fasta_prot, 'w') as outp:
        keep = False
        for line in inp:
            if line.startswith('>'):
                n_all += 1
                keep = 'mol:protein' in line
                if keep:
                    n_prot += 1
            if keep:
                outp.write(line)
    log(f'  pdb_seqres records: {n_all:,} total, {n_prot:,} protein kept')
    log('  makeblastdb -dbtype prot')
    subprocess.run([MAKEBLASTDB_BIN, '-in', fasta_prot, '-dbtype', 'prot',
                    '-out', db_prefix, '-title',
                    f'pdb_seqres_prot_{time.strftime("%Y-%m-%d")}'],
                   check=True)
    log(f'  built: {db_prefix}.*')
    return db_prefix


# Uniform all-query run for the entity build: one 5,000-hit blastp over the full
# label-set universe, so every query is scored against the same target depth.
MAX_TARGET_SEQS = 5000
MAX_HSPS = 1


def run_blast(query_fasta, db_prefix, out_tsv, threads):
    cols = ('qseqid sseqid pident length mismatch gapopen qstart qend '
            'sstart send evalue bitscore qlen slen')
    log(f'blastp (threads={threads}) -> {out_tsv}')
    t0 = time.time()
    subprocess.run([BLASTP_BIN, '-query', query_fasta, '-db', db_prefix,
                    '-out', out_tsv, '-outfmt', f'6 {cols}',
                    '-evalue', '1e-5', '-num_threads', str(threads),
                    '-max_hsps', str(MAX_HSPS),
                    '-max_target_seqs', str(MAX_TARGET_SEQS)], check=True)
    with open(out_tsv) as fh:
        n = sum(1 for _ in fh)
    log(f'  blast done in {time.time()-t0:.0f}s; {n:,} hit rows')
    with open(out_tsv + '.cols', 'w') as fh:
        fh.write(cols + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--shifts_file', default='data/chemical_shifts.csv')
    ap.add_argument('--work_dir', default='data/crystal_blast')
    ap.add_argument('--threads', type=int, default=20)
    ap.add_argument('--skip_query', action='store_true',
                    help='reuse an existing query.fasta in work_dir')
    args = ap.parse_args(argv)

    require_blast_version(BLASTP_BIN)
    require_blast_version(MAKEBLASTDB_BIN)

    os.makedirs(args.work_dir, exist_ok=True)
    query_fasta = os.path.join(args.work_dir, 'query.fasta')
    blast_tsv = os.path.join(args.work_dir, 'blast_full.tsv')

    if not (args.skip_query and os.path.isfile(query_fasta)):
        drops_out = os.path.join(args.work_dir, 'query_dropped.json')
        n_query = build_query_fasta(args.shifts_file, query_fasta, drops_out=drops_out)
        log(f'{n_query:,} query sequences')
    db_prefix = ensure_seqres_db(args.work_dir, args.threads)
    run_blast(query_fasta, db_prefix, blast_tsv, args.threads)
    log('DONE')


if __name__ == '__main__':
    main()
