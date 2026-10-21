#!/usr/bin/env python3
"""Entity-grained, method-neutral structure discovery.

Scores every label set (BMRB x shift-list x entity-assembly x entity x seq-hash)
against each structure source. Nothing is keyed on a bare BMRB: a BMRB link is
projected outward to that BMRB's label sets and scored on the entity sequence,
never assigned identity 100.

Per-source best pick: NMR = union of pairs.csv links + homology hits;
crystal = X-ray only, lowest resolution first; AF = entity-alignment-validated
accession only (no blind fan-out); EM excluded from masters, kept in ledger.

Remote-metadata and local re-BLAST are separate flags. Method classification is
cached, resumable and uncapped: an unclassified PDB is ledgered
``metadata_unavailable`` and is not selected.

Outputs (--out_dir): discovery_selection_{af,nmr,xtal}.csv, candidate_ledger.csv,
af_disposition.csv, coverage_accounting.csv (per-BMRB x per-source
query/recover/drop), reconciliation.json, query_dropped.json,
discover_blast.tsv.

Coverage gate: ``main`` raises if any valid BMRB was never queried against a source.
"""
import argparse
import csv
import json
import os
import subprocess
import time
import urllib.request
from collections import Counter

from .. import domain as _id
from . import crystal_census as cc
from .crystal_blast import IDENTITY_FIELDS, entity_sequence_map
from .paths import BLASTP_BIN, DATA_DIR

AF_MODEL_VERSION = 'model_v6'
AF_FILES_URL = 'https://alphafold.ebi.ac.uk/files'
RCSB_BATCH = 100
RCSB_CHECKPOINT_EVERY = 20        # batches between cache checkpoints
AF_HEAD_CAP = 36                  # bounded sibling-recovery probes
MIN_IDENTITY = cc.PIDENT_MIN      # 95.0
MIN_QUERY_COVERAGE = cc.QCOV_MIN  # 0.90


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def _identity_of(entity_map, lsid):
    """The expanded label-set identity columns for one label set."""
    rec = entity_map[lsid]
    return {'label_set_id': lsid, **{f: rec.get(f, '') for f in IDENTITY_FIELDS}}


def query_sha256(sequence):
    """Hash of the entity query sequence, bound into every ledger row."""
    return _id.sha256_hex(str(sequence).upper().encode('utf-8'))


def load_fasta(path):
    seqs = {}
    if not os.path.isfile(path):
        return seqs
    key, buf = None, []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if key is not None:
                    seqs[key] = ''.join(buf)
                key, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if key is not None:
        seqs[key] = ''.join(buf)
    return seqs


def reblast(reblast_seqs, db_prefix, out_tsv, blastp_bin, threads, max_target):
    """Local blastp for the new/changed label-set queries only (label_set_id FASTA
    ids). Writes a standalone tsv with the same columns as blast_full.tsv."""
    fasta = out_tsv + '.query.fasta'
    with open(fasta, 'w') as fh:
        for lsid, seq in reblast_seqs.items():
            fh.write(f'>{lsid}\n{seq}\n')
    cols = ('qseqid sseqid pident length mismatch gapopen qstart qend '
            'sstart send evalue bitscore qlen slen')
    log(f'reblast: {len(reblast_seqs)} label-set queries -> {out_tsv} '
        f'(max_target_seqs={max_target})')
    t0 = time.time()
    subprocess.run([blastp_bin, '-query', fasta, '-db', db_prefix,
                    '-out', out_tsv, '-outfmt', f'6 {cols}',
                    '-evalue', '1e-5', '-max_hsps', '1', '-num_threads', str(threads),
                    '-max_target_seqs', str(max_target)], check=True)
    with open(out_tsv) as fh:
        n = sum(1 for _ in fh)
    log(f'  reblast done in {time.time()-t0:.0f}s; {n:,} hit rows')
    with open(out_tsv + '.cols', 'w') as fh:
        fh.write(cols + '\n')


def classify(pdb_ids, cache_path, out_cache_path):
    """Return (methcache, n_batches, unavailable); reuses canonical+stage caches,
    fetches unknowns in RCSB_BATCH batches with no cap. ``unavailable`` PDBs get
    ledgered metadata_unavailable and are never selected."""
    cache = {}
    if os.path.isfile(cache_path):
        with open(cache_path) as fh:
            cache = json.load(fh)
    if os.path.isfile(out_cache_path):
        with open(out_cache_path) as fh:
            cache.update(json.load(fh))
    todo = sorted(p for p in pdb_ids if p not in cache)
    log(f'classify: {len(pdb_ids):,} unique candidate PDBs, '
        f'{len(cache):,} cached, {len(todo):,} to fetch from RCSB (no cap)')
    n_batches = 0
    for i in range(0, len(todo), RCSB_BATCH):
        cache.update(cc._fetch_batch(todo[i:i + RCSB_BATCH]))
        n_batches += 1
        if n_batches % RCSB_CHECKPOINT_EVERY == 0:
            with open(out_cache_path, 'w') as fh:
                json.dump(cache, fh)
            log(f'  classified {min(i+RCSB_BATCH, len(todo)):,}/{len(todo):,}')
    with open(out_cache_path, 'w') as fh:
        json.dump(cache, fh)
    unavailable = sorted(p for p in pdb_ids if p not in cache)
    if unavailable:
        log(f'  WARNING: {len(unavailable):,} PDBs unclassified after fetch '
            f'-> metadata_unavailable (never selected)')
    return cache, n_batches, unavailable


def method_of(methcache, pdbid):
    """(tag, resolution). tag is None when the PDB is unclassified (never select)."""
    rec = methcache.get(pdbid)
    if rec is None:
        return None, None
    methods = rec.get('methods', []) or []
    res = cc.best_res(rec.get('resolution'))
    if cc.is_xray(methods):
        return 'xray', res
    if cc.is_nmr(methods):
        return 'nmr', res
    if cc.is_em(methods):
        return 'em', res
    if methods:
        return 'other', res
    return None, res


def af_model_on_disk(uniprot, af_dir):
    path = os.path.join(af_dir, f'AF-{uniprot}-F1-{AF_MODEL_VERSION}.pdb')
    return path if os.path.isfile(path) else None


def af_head_exists(uniprot, timeout=20):
    url = f'{AF_FILES_URL}/AF-{uniprot}-F1-{AF_MODEL_VERSION}.pdb'
    try:
        req = urllib.request.Request(url, method='HEAD')
        req.add_header('User-Agent', 'deepshift/0.1.0')
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _model_sequence(pdb_path):
    """Chain-A one-letter sequence + ordered residue ids for an AF model."""
    from ..structure import parse_pdb
    from .structures import _extract_chain_sequence
    residues = parse_pdb(pdb_path, chain_id='A')
    seq, rids, _ = _extract_chain_sequence(residues, 'A')
    return seq, rids


def validate_af_candidate(entity_seq, model_seq):
    """Align an AF model sequence to the entity query; return identity/query_coverage/
    aligned_length/passed against the shared MIN_IDENTITY/MIN_QUERY_COVERAGE contract."""
    from .alignment import ALIGNMENT_AMBIGUITY_MARGIN, frame_safe_alignments
    from .structures import _alignment_stats, label_residue_mapping
    if not entity_seq or not model_seq:
        return {'identity': 0.0, 'query_coverage': 0.0, 'aligned_length': 0,
                'ambiguous_mapping': False, 'passed': False}
    alignments = frame_safe_alignments(
        model_seq, entity_seq, score_margin=ALIGNMENT_AMBIGUITY_MARGIN)
    iterator = iter(alignments)
    try:
        aln = next(iterator)   # target=model, query=entity
    except StopIteration:
        return {'identity': 0.0, 'query_coverage': 0.0, 'aligned_length': 0,
                'ambiguous_mapping': False, 'passed': False}
    s = _alignment_stats(aln)
    ident_pct = 100.0 * s['identity']
    # Discovery and materialization must pick the same residue mapping; a
    # near-co-optimal alternate mapping here would desync them, so flag it.
    model_ids = list(range(len(model_seq)))
    entity_ids = list(range(len(entity_seq)))
    best_mapping = label_residue_mapping(aln, model_ids, entity_ids)
    ambiguous = False
    for _ in range(64):
        try:
            alternative = next(iterator)
        except StopIteration:
            break
        if label_residue_mapping(alternative, model_ids, entity_ids) != best_mapping:
            if float(aln.score - alternative.score) <= (
                    ALIGNMENT_AMBIGUITY_MARGIN + 1e-9):
                ambiguous = True
            break
    passed = (ident_pct >= MIN_IDENTITY
              and s['query_coverage'] >= MIN_QUERY_COVERAGE
              and s['n_aligned'] >= 10
              and not ambiguous)
    return {'identity': round(ident_pct, 3),
            'query_coverage': round(s['query_coverage'], 4),
            'aligned_length': s['n_aligned'],
            'ambiguous_mapping': ambiguous, 'passed': passed}


def gather_candidates(entity_map, hits, pairs_map, methcache):
    """Per label set, merge homology BLAST hits with pairs.csv BMRB->PDB links
    projected to every label set of that BMRB (identity/coverage None, never
    100; re-aligned at realization)."""
    cand = {lsid: [] for lsid in entity_map}
    seen = {lsid: set() for lsid in entity_map}
    for lsid, hh in hits.items():
        if lsid not in cand:
            continue
        for h in hh:
            key = (h['pdbid'], h['chain'], 'homology')
            if key in seen[lsid]:
                continue
            tag, res = method_of(methcache, h['pdbid'])
            cand[lsid].append({
                'pdb_id': h['pdbid'], 'chain': h['chain'], 'route': 'homology',
                'method': tag, 'resolution': res,
                'identity': h['pident'], 'coverage': h['qcov'],
                'subject_coverage': h.get('scov')})
            seen[lsid].add(key)
    bmrb_to_lsids = {}
    for lsid, rec in entity_map.items():
        bmrb_to_lsids.setdefault(str(rec.get('bmrb_id')), []).append(lsid)
    for bmrb, pdbs in pairs_map.items():
        for lsid in bmrb_to_lsids.get(str(bmrb), []):
            for pdbid in pdbs:
                key = (pdbid, '', 'pairs')
                if key in seen[lsid]:
                    continue
                tag, res = method_of(methcache, pdbid)
                cand[lsid].append({
                    'pdb_id': pdbid, 'chain': '', 'route': 'pairs',
                    'method': tag, 'resolution': res,
                    'identity': None, 'coverage': None,
                    'subject_coverage': None})
                seen[lsid].add(key)
    return cand


def _route_pref(route):
    return 0 if route == 'pairs' else 1


def _num(v):
    return v if v is not None else 0.0


def rank_xray(cands):
    """All X-ray candidates in the exact deterministic preference order."""
    return sorted((c for c in cands if c['method'] == 'xray'), key=lambda c: (
        c['resolution'] if c['resolution'] is not None else 999.0,
        -_num(c['coverage']), -_num(c['identity']),
        _route_pref(c['route']), c['pdb_id'], c['chain']))


def select_xray(cands):
    xs = rank_xray(cands)
    if not xs:
        return None, xs
    return xs[0], xs


def rank_nmr(cands):
    """The direct-plus-homology NMR union in deterministic preference order."""
    return sorted((c for c in cands if c['method'] == 'nmr'), key=lambda c: (
        _route_pref(c['route']), -_num(c['coverage']), -_num(c['identity']),
        c['pdb_id'], c['chain']))


def select_nmr(cands):
    ns = rank_nmr(cands)
    if not ns:
        return None, ns
    return ns[0], ns


def rank_af(af_cands):
    """Every mapped AF candidate in deterministic entity-fit preference order.

    Failed and unavailable candidates stay in the pool, so a label set with no
    realizable model still records what was considered."""
    return sorted(af_cands, key=lambda c: (
        -_num(c.get('identity')), -_num(c.get('query_coverage')),
        -_num(c.get('aligned_length')), str(c.get('mapping_confidence', '')),
        c['accession']))


def select_af(af_cands):
    """Best entity-validated AF candidate that is locally available, plus the full
    validated list; ranked by sequence preference alone, selection also
    requires availability."""
    ranked = [c for c in rank_af(af_cands) if c['passed']]
    for c in ranked:
        if c['available']:
            return c, ranked
    return None, ranked


def candidate_id(label_set_id, source, accession, chain, route):
    """Stable identity of one discovery candidate, independent of its rank."""
    payload = {'label_set_id': str(label_set_id), 'source': str(source),
               'accession': str(accession), 'chain': str(chain),
               'route': str(route)}
    return 'rc_' + _id.sha256_hex(payload)


def discover_entities(
        entity_map, hits, pairs_map, methcache, af_candidates, *,
        query_attempts=None):
    """Score every label set for each source; emit selection tables + ledger.

    af_candidates: {label_set_id: [{accession, identity, query_coverage,
    aligned_length, mapping_confidence, available, reason}]}, already
    entity-validated (see validate_af_candidate); empty list == no AF.

    Returns dict with xtal_rows, nmr_rows, af_rows, ledger, af_disposition,
    funnel, method_distribution, metadata_unavailable count."""
    cand = gather_candidates(entity_map, hits, pairs_map, methcache)
    xtal_rows, nmr_rows, af_rows, ledger, af_disp = [], [], [], [], []
    funnel = {s: {'candidates': 0, 'eligible': 0, 'selected': 0}
              for s in ('xtal', 'nmr', 'af')}
    n_metadata_unavailable = 0
    # label sets that got a best-pick for each source (feeds the coverage gate)
    selected_ls = {'xtal': set(), 'nmr': set(), 'af': set()}

    for lsid in sorted(entity_map):
        ident = _identity_of(entity_map, lsid)
        qsha = query_sha256(entity_map[lsid]['sequence'])
        cs = cand.get(lsid, [])

        best_x, xs = select_xray(cs)
        rank_x = {id(c): i for i, c in enumerate(xs, 1)}
        funnel['xtal']['candidates'] += len(cs)
        funnel['xtal']['eligible'] += len(xs)
        if best_x:
            funnel['xtal']['selected'] += 1
            selected_ls['xtal'].add(lsid)
            xtal_rows.append({**ident, 'source': 'xtal',
                              'pdb_id': best_x['pdb_id'], 'chain': best_x['chain'],
                              'method': 'X-RAY DIFFRACTION',
                              'resolution': best_x['resolution'] if best_x['resolution'] is not None else '',
                              'identity': best_x['identity'] if best_x['identity'] is not None else '',
                              'query_coverage': best_x['coverage'] if best_x['coverage'] is not None else '',
                              'subject_coverage': best_x['subject_coverage'] if best_x['subject_coverage'] is not None else '',
                              'route': best_x['route']})

        best_n, ns = select_nmr(cs)
        rank_n = {id(c): i for i, c in enumerate(ns, 1)}
        funnel['nmr']['candidates'] += len(cs)
        funnel['nmr']['eligible'] += len(ns)
        if best_n:
            funnel['nmr']['selected'] += 1
            selected_ls['nmr'].add(lsid)
            nmr_rows.append({**ident, 'source': 'nmr',
                             'pdb_id': best_n['pdb_id'], 'chain': best_n['chain'],
                             'method': 'NMR', 'resolution': '',
                             'identity': best_n['identity'] if best_n['identity'] is not None else '',
                             'query_coverage': best_n['coverage'] if best_n['coverage'] is not None else '',
                             'subject_coverage': best_n['subject_coverage'] if best_n['subject_coverage'] is not None else '',
                             'route': best_n['route']})

        # ---- AlphaFold ----
        afc = af_candidates.get(lsid, [])
        best_af, validated = select_af(afc)
        ranked_af = rank_af(afc)
        rank_a = {id(c): i for i, c in enumerate(ranked_af, 1)}
        funnel['af']['candidates'] += len(afc)
        funnel['af']['eligible'] += len(validated)
        if best_af:
            funnel['af']['selected'] += 1
            selected_ls['af'].add(lsid)
            af_rows.append({**ident, 'source': 'af',
                            'uniprot': best_af['accession'],
                            'af_model': f"AF-{best_af['accession']}-F1-{AF_MODEL_VERSION}.pdb",
                            'identity': best_af['identity'],
                            'query_coverage': best_af['query_coverage'],
                            'aligned_length': best_af['aligned_length'],
                            'mapping_confidence': best_af['mapping_confidence'],
                            'route': 'uniprot_entity_validated'})
        for c in afc:
            sel = bool(best_af) and c['accession'] == best_af['accession'] and c['available']
            if sel:
                reason = 'selected:af'
            elif c['passed'] and not c['available']:
                reason = 'af_recoverable_not_local'
            elif c['passed']:
                reason = 'outranked:af'
            else:
                reason = c.get('reason') or 'af_sequence_mismatch'
            af_disp.append({'label_set_id': lsid, 'bmrb_id': ident['bmrb_id'],
                            'uniprot': c['accession'],
                            'accession': c['accession'],
                            'selected': sel, 'reason': reason})
            cid = candidate_id(lsid, 'af', c['accession'], 'A', 'uniprot')
            ledger.append({**ident, 'query_sha256': qsha, 'source': 'af',
                           'candidate_id': cid, 'candidate_rank': rank_a[id(c)],
                           'method': 'ALPHAFOLD', 'accession': c['accession'],
                           'chain': 'A', 'route': 'uniprot',
                           'identity': c['identity'], 'query_coverage': c['query_coverage'],
                           'aligned_length': c.get('aligned_length', ''),
                           'mapping_confidence': c.get('mapping_confidence', ''),
                           'subject_coverage': '', 'resolution': '',
                           'availability': 'on_disk' if c['available'] else 'absent_local',
                           'discovery_selected': sel, 'selected': sel,
                           'reason': reason})

        # ---- structural candidate ledger ----
        sel_x = (best_x['pdb_id'], best_x['chain']) if best_x else None
        sel_n = (best_n['pdb_id'], best_n['chain']) if best_n else None
        for c in cs:
            key = (c['pdb_id'], c['chain'])
            selected, reason = False, ''
            if c['method'] == 'xray':
                selected, reason = (True, 'selected:xtal') if key == sel_x else (False, 'outranked:xtal')
            elif c['method'] == 'nmr':
                selected, reason = (True, 'selected:nmr') if key == sel_n else (False, 'outranked:nmr')
            elif c['method'] == 'em':
                reason = 'em_excluded_unvalidated_source'
            elif c['method'] == 'other':
                reason = 'other_method_excluded'
            else:   # unclassified method (in `unavailable` or never fetched)
                reason = 'metadata_unavailable'
                n_metadata_unavailable += 1
            lane = {'xray': 'xtal', 'nmr': 'nmr', 'em': 'em',
                    'other': 'other'}.get(c['method'], 'unknown')
            rank = rank_x.get(id(c), rank_n.get(id(c), ''))
            cid = candidate_id(lsid, lane, c['pdb_id'], c['chain'], c['route'])
            ledger.append({**ident, 'query_sha256': qsha, 'source': lane,
                           'candidate_id': cid, 'candidate_rank': rank,
                           'method': c['method'] or '', 'accession': c['pdb_id'],
                           'chain': c['chain'], 'route': c['route'],
                           'identity': c['identity'] if c['identity'] is not None else '',
                           'query_coverage': c['coverage'] if c['coverage'] is not None else '',
                           'aligned_length': '', 'mapping_confidence': '',
                           'subject_coverage': c['subject_coverage'] if c['subject_coverage'] is not None else '',
                           'resolution': c['resolution'] if c['resolution'] is not None else '',
                           'availability': '', 'discovery_selected': selected,
                           'selected': selected, 'reason': reason})

    method_dist = dict(Counter(r['method'] or 'none' for r in ledger))
    if query_attempts is None:
        # Fallback: only proves attempts with hit evidence (undercounts vs real query
        # sets); production passes explicit attempted-query sets from below.
        query_attempts = {
            'xtal': set(hits),
            'nmr': set(hits),
            'af': {lsid for lsid, candidates in af_candidates.items() if candidates},
        }
    accounting = coverage_accounting(entity_map, selected_ls, query_attempts)
    return {'xtal_rows': xtal_rows, 'nmr_rows': nmr_rows, 'af_rows': af_rows,
            'ledger': ledger, 'af_disposition': af_disp, 'funnel': funnel,
            'method_distribution': method_dist,
            'metadata_unavailable': n_metadata_unavailable,
            'coverage_accounting': accounting,
            'coverage_summary': coverage_summary(accounting)}


def coverage_accounting(entity_map, selected, query_attempts):
    """Per-BMRB x per-source (af/nmr/xtal) discovery coverage: for every
    valid BMRB, whether each label set was queried against all three sources,
    plus recovered/dropped counts. ``query_attempts`` must be real query evidence,
    not preallocated empty dicts.

    ``selected`` maps each source to label-set ids with a best pick; returns
    per-(bmrb, source) records sorted by (bmrb, source)."""
    agg = {}   # (bmrb, source) -> record

    def _rec(bmrb, source):
        r = agg.get((bmrb, source))
        if r is None:
            r = {'bmrb_id': bmrb, 'source': source, 'n_label_sets': 0,
                 'n_queried': 0, 'n_recovered': 0, 'n_dropped': 0,
                 'drop_reasons': {}}
            agg[(bmrb, source)] = r
        return r

    for lsid, meta in entity_map.items():
        bmrb = str(meta.get('bmrb_id'))
        per_source = {
            'xtal': (lsid in query_attempts.get('xtal', ()), 'no_xray_candidate'),
            'nmr': (lsid in query_attempts.get('nmr', ()), 'no_nmr_candidate'),
            'af': (lsid in query_attempts.get('af', ()), 'no_uniprot_mapping'),
        }
        for source, (queried, drop_reason) in per_source.items():
            r = _rec(bmrb, source)
            r['n_label_sets'] += 1
            if queried:
                r['n_queried'] += 1
            if lsid in selected.get(source, ()):
                r['n_recovered'] += 1
            elif queried:
                r['n_dropped'] += 1
                reason = drop_reason or f'no_{source}_candidate'
                r['drop_reasons'][reason] = r['drop_reasons'].get(reason, 0) + 1

    for r in agg.values():
        r['queried'] = r['n_queried'] > 0
    order = {'af': 0, 'nmr': 1, 'xtal': 2}
    return sorted(agg.values(),
                  key=lambda r: (r['bmrb_id'], order.get(r['source'], 9)))


def coverage_gate_failures(records):
    """(bmrb, source) pairs where a valid BMRB was never queried against a source."""
    return [(r['bmrb_id'], r['source']) for r in records
            if r['n_label_sets'] > 0 and r['n_queried'] == 0]


def assert_coverage_complete(records):
    """Coverage gate: raise if any valid BMRB was never queried against a source."""
    failures = coverage_gate_failures(records)
    if not failures:
        return
    by_src = {}
    for bmrb, source in failures:
        by_src.setdefault(source, []).append(bmrb)
    detail = '; '.join(
        f"{src}: {len(bs)} BMRB(s) never queried (e.g. {', '.join(sorted(bs)[:5])})"
        for src, bs in sorted(by_src.items()))
    raise RuntimeError(
        f"coverage-accounting gate FAILED: {len(failures)} (BMRB, "
        f"source) pair(s) were never queried -- {detail}")


def coverage_summary(records):
    """Roll the per-(bmrb, source) accounting up into a JSON-safe summary."""
    per_source = {}
    for src in ('af', 'nmr', 'xtal'):
        srows = [r for r in records if r['source'] == src]
        reasons = Counter()
        for r in srows:
            reasons.update(r['drop_reasons'])
        per_source[src] = {
            'bmrbs': len(srows),
            'queried': sum(1 for r in srows if r['n_queried'] > 0),
            'never_queried': sum(1 for r in srows if r['n_queried'] == 0),
            'recovered': sum(1 for r in srows if r['n_recovered'] > 0),
            'dropped_only': sum(1 for r in srows
                                if r['n_recovered'] == 0 and r['n_queried'] > 0),
            'label_sets': sum(r['n_label_sets'] for r in srows),
            'label_sets_recovered': sum(r['n_recovered'] for r in srows),
            'label_sets_dropped': sum(r['n_dropped'] for r in srows),
            'drop_reasons': dict(reasons)}
    failures = coverage_gate_failures(records)
    return {'n_bmrbs': len({r['bmrb_id'] for r in records}),
            'per_source': per_source,
            'gate_passed': not failures,
            'n_never_queried_pairs': len(failures)}


_ID_COLS = ['label_set_id', *list(IDENTITY_FIELDS)]
_XTAL_COLS = [*_ID_COLS, 'source', 'pdb_id', 'chain', 'method', 'resolution',
                         'identity', 'query_coverage', 'subject_coverage', 'route']
_AF_COLS = [*_ID_COLS, 'source', 'uniprot', 'af_model', 'identity',
                       'query_coverage', 'aligned_length', 'mapping_confidence', 'route']
_LEDGER_COLS = [*_ID_COLS, 'query_sha256', 'source', 'method', 'accession',
                           'candidate_id', 'candidate_rank', 'chain', 'route',
                           'identity', 'query_coverage', 'aligned_length',
                           'mapping_confidence',
                           'subject_coverage', 'resolution', 'availability',
                           'discovery_selected', 'selected', 'reason']
_AFDISP_COLS = ['label_set_id', 'bmrb_id', 'uniprot', 'accession', 'selected', 'reason']
_COVERAGE_COLS = ['bmrb_id', 'source', 'n_label_sets', 'n_queried', 'queried',
                  'n_recovered', 'n_dropped', 'drop_reasons']


def _fmt_reasons(reasons):
    """Compact 'reason:count;reason:count' rendering for the coverage CSV."""
    return ';'.join(f'{k}:{v}' for k, v in sorted(reasons.items()))


def _write(path, rows, cols):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore', restval='')
        w.writeheader()
        w.writerows(rows)
    log(f'  wrote {len(rows):,} rows -> {path}')


def write_outputs(out_dir, result):
    # Diagnostic picks only, not authoritative: materialization walks candidate_ledger.csv
    # and publishes selection_*.csv for the first candidate clearing every binding gate.
    _write(os.path.join(out_dir, 'discovery_selection_xtal.csv'),
           result['xtal_rows'], _XTAL_COLS)
    _write(os.path.join(out_dir, 'discovery_selection_nmr.csv'),
           result['nmr_rows'], _XTAL_COLS)
    _write(os.path.join(out_dir, 'discovery_selection_af.csv'),
           result['af_rows'], _AF_COLS)
    _write(os.path.join(out_dir, 'candidate_ledger.csv'), result['ledger'], _LEDGER_COLS)
    _write(os.path.join(out_dir, 'af_disposition.csv'), result['af_disposition'], _AFDISP_COLS)
    # Coverage-accounting gate output: per-BMRB x per-source query/recover/drop.
    cov_rows = [{**r, 'drop_reasons': _fmt_reasons(r['drop_reasons'])}
                for r in result['coverage_accounting']]
    _write(os.path.join(out_dir, 'coverage_accounting.csv'), cov_rows, _COVERAGE_COLS)


def build_af_candidates(entity_map, af_map, af_dir, *,
                        head_cap=AF_HEAD_CAP, allow_head=True):
    """Per label set, entity-validate every mapped AF accession. ``af_map`` maps a
    BMRB id to one or more UniProt accessions, each aligned to the label set's
    entity sequence; only a passing, locally-available model is selectable.
    Returns {label_set_id: [candidate dicts]} and the head-probe count."""
    out = {}
    n_head = 0
    for lsid, rec in entity_map.items():
        bmrb = str(rec.get('bmrb_id'))
        accs = af_map.get(bmrb)
        if accs is None:
            out[lsid] = []
            continue
        if isinstance(accs, str):
            accs = [accs]
        cands = []
        for i, acc in enumerate(dict.fromkeys(accs)):
            path = af_model_on_disk(acc, af_dir)
            if path:
                mseq, _ = _model_sequence(path)
                m = validate_af_candidate(rec['sequence'], mseq)
                cands.append({'accession': acc, 'identity': m['identity'],
                              'query_coverage': m['query_coverage'],
                              'aligned_length': m['aligned_length'],
                              'mapping_confidence': i, 'available': True,
                              'passed': m['passed'],
                              'reason': '' if m['passed'] else (
                                  'af_alignment_ambiguous'
                                  if m['ambiguous_mapping']
                                  else 'af_sequence_mismatch')})
            else:
                recoverable = False
                if allow_head and n_head < head_cap:
                    n_head += 1
                    recoverable = af_head_exists(acc)
                cands.append({'accession': acc, 'identity': 0.0,
                              'query_coverage': 0.0, 'aligned_length': 0,
                              'mapping_confidence': i, 'available': False,
                              'passed': False,
                              'reason': 'af_recoverable_remote' if recoverable
                              else 'af_model_absent_local'})
        out[lsid] = cands
    return out, n_head


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=DATA_DIR,
                    help='build data dir to read (default $SHIFT_DATA_ROOT)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--blastp_bin', default=BLASTP_BIN,
                    help='blastp executable (default $BLASTP_BIN, else PATH)')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--max_target', type=int, default=5000)
    ap.add_argument('--af_head_cap', type=int, default=AF_HEAD_CAP)
    ap.add_argument('--no_reblast', action='store_true',
                    help='skip the local re-BLAST of new/changed label-set queries')
    ap.add_argument('--no_remote_metadata', action='store_true',
                    help='skip all remote calls (RCSB classify + AF HEAD); cached only')
    ap.add_argument('--no_remote', action='store_true',
                    help='convenience: set both --no_reblast and --no_remote_metadata')
    args = ap.parse_args(argv)
    if args.no_remote:
        args.no_reblast = args.no_remote_metadata = True

    d = args.data_dir
    os.makedirs(args.out_dir, exist_ok=True)
    cb = os.path.join(d, 'crystal_blast')
    shifts = os.path.join(d, 'chemical_shifts.csv')
    remote = {'rcsb_batches': 0, 'af_heads': 0}

    # ---- 1. entity queries ----
    seqs, drops = entity_sequence_map(shifts)
    with open(os.path.join(args.out_dir, 'query_dropped.json'), 'w') as fh:
        json.dump({'n_written': len(seqs), 'n_dropped': len(drops), 'drops': drops},
                  fh, indent=2)
    entity_map = seqs   # {lsid: {sequence, <identity fields>}}
    log(f'queries: {len(seqs):,} label sets, {len(drops):,} dropped (logged)')

    orig = load_fasta(os.path.join(cb, 'query.fasta'))
    blast_full = os.path.join(cb, 'blast_full.tsv')
    # A prior FASTA is reusable only paired with its completed result file (alone, no hits).
    reuse = (
        {ls for ls, rec in seqs.items() if orig.get(ls) == rec['sequence']}
        if os.path.isfile(blast_full) else set()
    )
    reblast_set = {ls: seqs[ls]['sequence'] for ls in seqs if ls not in reuse}
    log(f'query reuse: {len(reuse):,} unchanged, {len(reblast_set):,} new/changed')

    # ---- 2. homology hits (reuse blast_full where unchanged) ----
    # A fresh build has no prior blast_full.tsv; absent file just means no reusable
    # prior hits (every query gets re-BLASTed).
    hits_all = cc.census(blast_full)[0] if os.path.isfile(blast_full) else {}
    hits = {ls: hh for ls, hh in hits_all.items() if ls in reuse}
    structural_query_attempts = set(reuse)
    if reblast_set and not args.no_reblast:
        rb_tsv = os.path.join(args.out_dir, 'discover_blast.tsv')
        reblast(reblast_set, os.path.join(cb, 'pdb_seqres_db', 'pdb_seqres_prot'),
                rb_tsv, args.blastp_bin, args.threads, args.max_target)
        hits_rb, _ = cc.census(rb_tsv)
        for ls, hh in hits_rb.items():
            if ls in reblast_set:
                hits[ls] = hh
        structural_query_attempts.update(reblast_set)
    elif reblast_set:
        log(f'  --no_reblast: skipping re-BLAST of {len(reblast_set)} queries')

    # ---- pairs (BMRB -> PDB), projected outward as candidates ----
    pairs_map = {}
    pairs_path = os.path.join(d, 'pairs.csv')
    if os.path.isfile(pairs_path):
        with open(pairs_path) as f:
            for row in csv.DictReader(f):
                b = row.get('Entry_ID')
                s = row.get('pdb_ids') or ''
                pset = {p.strip().lower() for p in s.split(',')
                        if p.strip() and p.strip().lower() != 'nan'}
                if pset:
                    pairs_map[b] = pset

    # ---- 3. classify candidate PDB universe ----
    pdb_universe = set()
    for hh in hits.values():
        pdb_universe |= {h['pdbid'] for h in hh}
    for ps in pairs_map.values():
        pdb_universe |= ps
    out_cache = os.path.join(args.out_dir, 'pdb_method_resolution.updated.json')
    canon_cache = os.path.join(cb, 'pdb_method_resolution.json')
    if args.no_remote_metadata:
        if os.path.isfile(canon_cache):
            with open(canon_cache) as fh:
                methcache = json.load(fh)
        else:
            methcache = {}
        unavailable = sorted(p for p in pdb_universe if p not in methcache)
        n_batches = 0
    else:
        methcache, n_batches, unavailable = classify(pdb_universe, canon_cache, out_cache)
    remote['rcsb_batches'] = n_batches

    # ---- 4. AlphaFold candidates (entity-aligned) ----
    af_map_path = os.path.join(d, 'alphafold', 'bmrb_uniprot_mapping.json')
    if os.path.isfile(af_map_path):
        with open(af_map_path) as fh:
            af_map = json.load(fh)
    else:
        af_map = {}
    af_dir = os.path.join(d, 'alphafold')
    af_candidates, n_head = build_af_candidates(
        entity_map, af_map, af_dir, head_cap=args.af_head_cap,
        allow_head=not args.no_remote_metadata)
    remote['af_heads'] = n_head
    # Only af_map's explicit keys count as "queried" -- absent BMRBs aren't provably
    # queried, and build_af_candidates' empty slots are not evidence either way.
    af_query_attempts = {
        lsid for lsid, rec in entity_map.items()
        if str(rec.get('bmrb_id')) in af_map
    }

    # ---- 5. discover + write ----
    result = discover_entities(entity_map, hits, pairs_map, methcache,
                               af_candidates, query_attempts={
                                   'xtal': structural_query_attempts,
                                   'nmr': structural_query_attempts,
                                   'af': af_query_attempts,
                               })
    write_outputs(args.out_dir, result)

    recon = {
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'grain': 'label_set_id',
        'universe_label_sets': len(seqs),
        'query_dropped': len(drops),
        'query_reused': len(reuse),
        'query_reblasted': len(reblast_set),
        'funnel': result['funnel'],
        'selected': {'af': len(result['af_rows']),
                     'nmr': len(result['nmr_rows']),
                     'crystal': len(result['xtal_rows'])},
        'method_distribution_over_ledger': result['method_distribution'],
        'coverage_accounting': result['coverage_summary'],
        'metadata_unavailable_ledger_rows': result['metadata_unavailable'],
        'pdbs_unclassified': len(unavailable),
        'remote_calls': remote,
        'defaults': {
            'identity_min_pct': MIN_IDENTITY,
            'query_coverage_min': MIN_QUERY_COVERAGE,
            'nmr_source': 'UNION(pairs.csv, homology-NMR hits)',
            'crystal_source': 'X-RAY DIFFRACTION only',
            'af_source': 'entity-aligned validated accession + local model',
            'em_policy': 'excluded from masters, retained in ledger',
            'rcsb_cap': 'none (resumable, fail-closed metadata_unavailable)'},
    }
    with open(os.path.join(args.out_dir, 'reconciliation.json'), 'w') as fh:
        json.dump(recon, fh, indent=2)
    log('reconciliation: ' + json.dumps(recon['selected']))
    # The coverage gate runs last, so every output is on disk before it raises on a
    # never-queried BMRB.
    log('coverage: ' + json.dumps({s: v['queried']
                                    for s, v in recon['coverage_accounting']
                                    ['per_source'].items()}))
    assert_coverage_complete(result['coverage_accounting'])
    log('DONE')
    return recon


if __name__ == '__main__':
    main()
