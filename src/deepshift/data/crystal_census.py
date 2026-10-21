#!/usr/bin/env python3
"""Crystal-candidate census and RCSB method/resolution classification.

Blast-hit survey helpers consumed by discover.py, the method-neutral discovery
stage that routes every hit to the right per-source (X-ray / NMR / EM) master:
``census`` streams blast_full.tsv to per-label-set qualifying hits (thresholds
below); ``_fetch_batch``/classify query RCSB GraphQL for exptl method and resolution
(cached in pdb_method_resolution.json); ``is_xray``/``is_nmr``/``is_em``/``best_res``
are the method and resolution predicates. The per-source best pick is made in
discover.py.
"""
import json
import time
import urllib.request

PIDENT_MIN = 95.0
QCOV_MIN = 0.90
EVALUE_MAX = 1e-5
GRAPHQL = 'https://data.rcsb.org/graphql'
QUERY_TMPL = ('{{ entries(entry_ids: [{ids}]) {{ rcsb_id '
              'exptl {{ method }} '
              'rcsb_entry_info {{ resolution_combined }} }} }}')


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def census(blast_tsv):
    """blast_full.tsv (outfmt 6 + qlen slen) -> ({query_id: [hit dicts]}, pdb id set).
    Query id is whatever the FASTA carried (entity build: ``label_set_id``, never
    reduced to a bare BMRB). Records both query coverage (``qcov``) and subject
    coverage (``scov`` = aligned span / subject length) so a longer structure that
    merely contains the shifted construct is distinguishable from a partial match."""
    # column order written by crystal_blast.run_blast
    cols = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'qlen', 'slen']
    idx = {c: i for i, c in enumerate(cols)}
    hits = {}
    pdb_ids = set()
    n = kept = 0
    with open(blast_tsv) as f:
        for line in f:
            n += 1
            p = line.rstrip('\n').split('\t')
            if len(p) < len(cols):
                continue
            try:
                pident = float(p[idx['pident']])
                evalue = float(p[idx['evalue']])
                qstart = int(p[idx['qstart']])
                qend = int(p[idx['qend']])
                sstart = int(p[idx['sstart']])
                send = int(p[idx['send']])
                qlen = int(p[idx['qlen']])
                slen = int(p[idx['slen']])
            except ValueError:
                continue
            qcov = (qend - qstart + 1) / qlen if qlen else 0.0
            scov = (abs(send - sstart) + 1) / slen if slen else 0.0
            if pident < PIDENT_MIN or qcov < QCOV_MIN or evalue >= EVALUE_MAX:
                continue
            # sseqid like '1abc_A' (pdbid_chain) in pdb_seqres
            sid = p[idx['sseqid']]
            if '_' in sid:
                pdbid, chain = sid.split('_', 1)
            else:
                pdbid, chain = sid[:4], sid[4:5]
            pdbid = pdbid.lower()
            label_set_id = p[idx['qseqid']]
            hits.setdefault(label_set_id, []).append(
                {'label_set_id': label_set_id, 'pdbid': pdbid, 'chain': chain,
                 'pident': pident, 'qcov': round(qcov, 4),
                 'scov': round(scov, 4)})
            pdb_ids.add(pdbid)
            kept += 1
    log(f'census: {n:,} rows -> {kept:,} qualifying hits over '
        f'{len(hits):,} label sets, {len(pdb_ids):,} unique PDB ids')
    return hits, pdb_ids


def _fetch_batch(ids, retries=4):
    q = QUERY_TMPL.format(ids=','.join(f'"{i.upper()}"' for i in ids))
    payload = json.dumps({'query': q}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                GRAPHQL, data=payload,
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            out = {}
            for e in data.get('data', {}).get('entries', []) or []:
                pid = e['rcsb_id'].lower()
                methods = [m['method'] for m in (e.get('exptl') or [])]
                res = (e.get('rcsb_entry_info') or {}).get(
                    'resolution_combined')
                out[pid] = {'methods': methods, 'resolution': res}
            return out
        except Exception as ex:
            if attempt == retries - 1:
                log(f'  GraphQL batch failed after {retries}: {ex}')
                return {}
            time.sleep(2 * (attempt + 1))
    return {}


def is_xray(methods):
    return any('X-RAY DIFFRACTION' in m for m in methods)


def is_nmr(methods):
    """True NMR structure: solution OR solid-state NMR."""
    return any('NMR' in m for m in methods)


def is_em(methods):
    """Electron microscopy / electron crystallography."""
    return any(('ELECTRON MICROSCOPY' in m) or ('ELECTRON CRYSTALLOGRAPHY' in m)
               for m in methods)


def best_res(resolution):
    if not resolution:
        return None
    vals = [r for r in resolution if isinstance(r, (int, float))]
    return min(vals) if vals else None
