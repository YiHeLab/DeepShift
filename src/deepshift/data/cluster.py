#!/usr/bin/env python3
"""Entity-grained sequence clustering: one alignment sequence per logical entity,
not per BMRB. Edge families: exact-token/alignment hash equality plus two
MMseqs2 passes -- bidirectional and containment, id>=0.80/cov>=0.90 -- over
entities >=10 residues."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable

from ..domain import sha256_hex
from .config import DATA_DIR
from .sequence_clustering import (
    EVALUE,
    pairwise_edges_to_components,
    run_all_vs_all,
)

# Realized entities shorter than this are not clustered (quarantined by folds.py).
MIN_ENTITY_LEN = 10
# Two label sets share a training fold when MMseqs finds identity >= 0.80 at coverage
# >= 0.90 between them. Coverage mode 0 requires the floor on both sequences; mode 1
# requires it on the target alone, so a sequence wholly contained in a longer one is an
# edge too. Both modes run for every sequence family of the training folds.
MMSEQS_IDENTITY = 0.80
MMSEQS_COVERAGE = 0.90
MMSEQS_FAMILY_COVERAGE_MODE = {"mmseqs_cov0": 0, "mmseqs_containment": 1}
MMSEQS_COVERAGE_MODES = tuple(MMSEQS_FAMILY_COVERAGE_MODE.values())
SCHEMA_ALIGNMENT_EDGE = "deepshift.alignment-edge.v1"

# The entity-grained per-source masters produced by materialization.
SOURCE_MASTERS = ('structure_data_af.csv', 'structure_data_nmr.csv',
                  'structure_data_xtal.csv')


def find_build_logs(data_dir: str) -> dict[str, str]:
    """Materialization build logs (``build_log_{af,nmr,xtal}.csv``) under ``data_dir``;
    they carry the modeled region per label set."""
    out: dict[str, str] = {}
    for src in ('af', 'nmr', 'xtal'):
        p = os.path.join(data_dir, f'build_log_{src}.csv')
        if os.path.isfile(p):
            out[src] = p
    return out


def load_modeled_positions(
        build_log_paths: Iterable[str]) -> dict[tuple[str, str], list[int]]:
    """``{(label_set_id, source): [polymer positions]}`` from materialization build logs --
    the modeled region, authoritative over the sparse master shift rows.
    Raises if a build log is present but missing ``modeled_positions``: reconstructing
    the region from observed-only rows would understate it."""
    out: dict[tuple[str, str], list[int]] = {}
    for path in build_log_paths:
        if not os.path.isfile(path):
            continue
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            if not cols:
                continue
            if 'modeled_positions' not in cols:
                raise ValueError(
                    f'{path}: materialization build log has no modeled_positions '
                    f'column -- cannot derive the true modeled region for the '
                    f'leakage clustering. Rebuild '
                    f'materialization so it emits the full modeled domain.')
            for row in reader:
                if (row.get('status') or 'success') != 'success':
                    continue
                lsid, src = row.get('label_set_id'), row.get('source')
                if not lsid or not src:
                    continue
                raw = (row.get('modeled_positions') or '').strip()
                out[(lsid, src)] = sorted(
                    int(x) for x in raw.split(';') if x != '')
    return out


def collect_realized_entities(master_paths: Iterable[str],
                              catalog_path: str | None = None,
                              build_log_paths: Iterable[str] | None = None,
                              require_build_logs: bool = False) -> dict[str, dict]:
    """One record per realized logical entity (>=1 materialized row, any source);
    representative label_set_id is already 1:1 post materialization.

    Full sequence/length come from ``entity_catalog.csv``, not the sparse shift rows,
    which would compress the polymer and miss homology/exact edges. ``catalog_path``
    defaults next to the first master and raises when absent.

    Modeled sub-sequences likewise use the modeled positions from the
    materialization build logs (``build_log_{source}.csv``) against that same catalog
    sequence, never observed comp_id -- else an unlabeled residue vanishes and
    non-adjacent positions concatenate as if adjacent. ``require_build_logs=True``
    (production) makes a missing/columnless log fatal; a fixture without one falls back
    to the masters' label positions.

    Returns {logical_entity_id: record}, grouped by label_set_id, ordered by
    comp_index_id (entities never interleave).
    """
    from .fetch import default_entity_catalog_path, read_entity_catalog
    master_list = list(master_paths)
    if catalog_path is None:
        anchor = next((p for p in master_list if os.path.isfile(p)), None)
        catalog_path = default_entity_catalog_path(anchor) if anchor else None
    if not catalog_path or not os.path.isfile(catalog_path):
        raise ValueError(
            f'entity_catalog.csv full-sequence sidecar not found '
            f'({catalog_path}); clustering requires the FULL entity sequence '
            f'Refusing to reconstruct from observed-only shift rows.')
    catalog = read_entity_catalog(catalog_path)

    # Modeled-region provenance, authoritative when build logs are available.
    if build_log_paths is None:
        anchor_dir = os.path.dirname(os.path.abspath(catalog_path))
        build_log_paths = list(find_build_logs(anchor_dir).values())
    else:
        build_log_paths = list(build_log_paths)
    if require_build_logs and not build_log_paths:
        raise ValueError(
            'no materialization build_log_{af,nmr,xtal}.csv found; the modeled-'
            'region leakage clustering requires the true modeled domain and must '
            'not fall back to the sparse master shift rows.')
    modeled_positions = load_modeled_positions(build_log_paths)
    have_modeled_provenance = bool(modeled_positions)
    # label_set_id -> {identity, residues{comp_index_id: comp_id}, modeled:set, sources:set, shift_res:set}
    acc: dict[str, dict] = {}
    for path in master_list:
        if not os.path.isfile(path):
            continue
        with open(path, newline='') as handle:
            reader = csv.DictReader(handle)
            cols = reader.fieldnames or []
            if 'label_set_id' not in cols:
                raise ValueError(f'{path}: not entity-grained (no label_set_id)')
            shift_cols = [c for c in cols if c.endswith('_shift')]
            for row in reader:
                lsid = row.get('label_set_id')
                if not lsid:
                    continue
                rec = acc.get(lsid)
                if rec is None:
                    rec = acc[lsid] = {
                        'label_set_id': lsid,
                        'logical_entity_id': row.get('logical_entity_id', ''),
                        'bmrb_id': row.get('bmrb_id', ''),
                        'entity_id': row.get('entity_id', ''),
                        'exact_polymer_sequence_sha256':
                            row.get('exact_polymer_sequence_sha256', ''),
                        'residues': {}, 'modeled': set(),
                        # modeled_by_source: kept per-source so one source's
                        # positions don't dilute another source's modeled domain.
                        'modeled_by_source': defaultdict(set),
                        # structures: co-solved (source, accession) keys; chains sharing a
                        # key are partner chains of one complex.
                        'structures': set(),
                        'sources': set(), 'shift_res': set()}
                src = row.get('source', '')
                if src:
                    rec['sources'].add(src)
                accession = (row.get('accession') or '').strip()
                if src and accession:
                    rec['structures'].add((src, accession))
                try:
                    cix = int(row['comp_index_id'])
                except (KeyError, TypeError, ValueError):
                    continue
                rec['residues'].setdefault(cix, row.get('comp_id', 'UNK'))
                if (row.get('struct_res_id') not in (None, '', 'nan')
                        and row.get('mismatch_type') != 'gap_in_structure'):
                    rec['modeled'].add(cix)
                    if src:
                        rec['modeled_by_source'][src].add(cix)
                if any(row.get(c) not in (None, '', 'nan') for c in shift_cols):
                    rec['shift_res'].add(cix)

    entities: dict[str, dict] = {}
    for lsid, rec in acc.items():
        cat = catalog.get(lsid)
        if cat is None:
            raise ValueError(
                f'realized label set {lsid} has no entity_catalog.csv row -- '
                f'inconsistent full-sequence sidecar (fail-closed).')
        # Full entity sequence and length from the sidecar.
        entity_seq = cat['alignment_sequence']
        # polymer position -> authoritative one-letter code (never observed comp_id).
        pos_char = dict(zip(cat['seq_positions'], cat['alignment_sequence']))

        def _region_seq(positions, *, pos_char=pos_char):
            return ''.join(pos_char[p] for p in sorted(positions) if p in pos_char)

        # Per-source modeled positions: the modeled region from build logs when available
        # (else the masters' label positions, fixture fallback); sequence is always the catalog.
        if have_modeled_provenance:
            src_positions = {}
            for src in rec['sources']:
                if (lsid, src) in modeled_positions:
                    src_positions[src] = modeled_positions[(lsid, src)]
                elif require_build_logs:
                    # Realized (entity, source) with no modeled region in the build logs is
                    # inconsistent; refused, since dropping a mandatory edge is silent.
                    raise ValueError(
                        f'label set {lsid} is realized by source {src!r} in the '
                        f'masters but has no modeled_positions in the materialization '
                        f'build logs; refusing to drop its modeled region (would '
                        f'evade a modeled-region leakage edge). Rebuild the build '
                        f'logs so every realized (label set, source) is present.')
                else:
                    # Fixture fallback: the masters' label positions stand in for this
                    # source, which keeps it in the edge set.
                    src_positions[src] = sorted(rec['modeled_by_source'].get(src, ()))
        else:
            src_positions = {src: sorted(cixset)
                             for src, cixset in rec['modeled_by_source'].items()}
        modeled_by_source = {src: _region_seq(pos)
                             for src, pos in src_positions.items()}
        # Unioned modeled sub-sequence (length/quarantine/balance fallbacks). Leakage
        # edges use the per-source form above.
        union_positions = set()
        for pos in src_positions.values():
            union_positions.update(pos)
        modeled_seq = _region_seq(union_positions)
        le = rec['logical_entity_id'] or lsid
        out = {
            'logical_entity_id': le, 'label_set_id': lsid,
            'bmrb_id': rec['bmrb_id'], 'entity_id': rec['entity_id'],
            'exact_polymer_sequence_sha256': rec['exact_polymer_sequence_sha256'],
            'entity_sequence': entity_seq, 'modeled_sequence': modeled_seq,
            'modeled_by_source': modeled_by_source,
            'entity_length': cat['polymer_length'],
            'modeled_length': len(modeled_seq),
            'n_shift_residues': len(rec['shift_res']),
            # structures: partner-chain edges group entities sharing a
            # (source, accession) file.
            'structures': sorted(rec['structures']),
            'sources': sorted(rec['sources'])}
        # One representative per entity (smallest label_set_id), with the co-solved
        # structures unioned across every label set so no partner-chain signal is lost.
        prev = entities.get(le)
        if prev is None:
            entities[le] = out
        else:
            merged = sorted(set(map(tuple, prev['structures']))
                            | set(map(tuple, out['structures'])))
            rep = out if lsid < prev['label_set_id'] else prev
            rep['structures'] = merged
            entities[le] = rep
    return entities


def alignment_sequence_hash(sequence: str) -> str:
    """Hash of the uppercase alignment sequence (normalized-alignment equality edge),
    schema-tagged to avoid id collisions."""
    return sha256_hex({'schema': SCHEMA_ALIGNMENT_EDGE,
                       'sequence': str(sequence).upper()})


def hash_edge_groups(entities: dict[str, dict], key_fn) -> dict[str, list[str]]:
    """Group logical_entity_ids by a hash key (skips empty keys). Only multi-member
    groups are returned -- a lone entity is not an edge."""
    groups: dict[str, list[str]] = defaultdict(list)
    for le, rec in entities.items():
        k = key_fn(rec)
        if k:
            groups[k].append(le)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def entity_mmseqs_groups(entities: dict[str, dict], *, cov_mode: int,
                         mmseqs_binary=None, identity=MMSEQS_IDENTITY,
                         coverage=MMSEQS_COVERAGE) -> list[list[str]]:
    """Connected closure of exhaustive MMseqs pair edges over entity sequences."""
    seqs = {le: rec['entity_sequence'] for le, rec in entities.items()
            if rec['entity_length'] >= MIN_ENTITY_LEN}
    if not seqs:
        return []
    pairs = run_all_vs_all(
        seqs, identity=identity, coverage=coverage,
        cov_mode=cov_mode, binary=mmseqs_binary)
    return pairwise_edges_to_components(seqs, pairs)


def partner_chain_groups(entities: dict[str, dict],
                         partner_membership: dict[object, Iterable[str]] | None
                         = None) -> list[list[str]]:
    """Partner-chain leakage groups at logical-entity grain: chains co-solved in one
    physical structure share a coordinate file, so every co-solved chain, context and
    target alike, lands in one leakage component. Two signals are unioned: file-native
    co-solved chains sharing a ``(source, accession)`` on each record's ``structures``
    set (from :func:`collect_realized_entities`), and optional ``partner_membership``
    ``{complex_key -> logical_entity_ids}`` context metadata for partners admitted as
    context only (no shift rows of their own, and so invisible to file-native capture --
    a separately-solved ligand, say). Members absent from the realized universe are
    ignored. Each complex contributes one group (>=2 distinct entities), closed to a
    clique by the fold union-find; homo-oligomer copies of one entity emit no edge.
    No length gate applies -- a short peptide ligand still co-folds.
    """
    by_complex: dict[object, set[str]] = defaultdict(set)
    for le, rec in entities.items():
        for key in rec.get('structures') or ():
            by_complex[('file', tuple(key))].add(le)
    if partner_membership:
        known = set(entities)
        for complex_key, members in partner_membership.items():
            for le in members:
                if le in known:
                    by_complex[('context', complex_key)].add(le)
    groups, seen = [], set()
    for members in by_complex.values():
        if len(members) < 2:
            continue
        key = tuple(sorted(members))
        if key not in seen:          # distinct complexes can share a member set
            seen.add(key)
            groups.append(list(key))
    return sorted(groups)


def build_entity_edges(entities: dict[str, dict], *, mmseqs_binary=None,
                       partner_membership=None) -> dict[str, object]:
    """All entity-grained edge families keyed by logical_entity_id: ``exact_token``
    (same exact_polymer_sequence_sha256, modified residues kept), ``alignment`` (same
    one-letter sequence), ``mmseqs_cov0`` (bidirectional, id>=80% & cov>=90%),
    ``mmseqs_containment`` (cov-mode 1, same thresholds), and ``partner_chain``
    (co-solved in one structure 2a; ``partner_membership`` adds geometry-stage
    context-only partners).
    """
    return {
        'exact_token': hash_edge_groups(
            entities, lambda r: r['exact_polymer_sequence_sha256']),
        'alignment': hash_edge_groups(
            entities, lambda r: alignment_sequence_hash(r['entity_sequence'])),
        **{family: entity_mmseqs_groups(entities, cov_mode=mode, mmseqs_binary=mmseqs_binary)
           for family, mode in MMSEQS_FAMILY_COVERAGE_MODE.items()},
        'partner_chain': partner_chain_groups(
            entities, partner_membership=partner_membership),
    }


def write_entity_fasta(entities: dict[str, dict], path: str,
                       min_len: int = MIN_ENTITY_LEN) -> None:
    """Write the entity query FASTA (id=logical_entity_id), frozen in id order,
    realized entities >= min_len only."""
    with open(path, 'w') as fh:
        for le in sorted(entities):
            rec = entities[le]
            if rec['entity_length'] < min_len:
                continue
            fh.write(f'>{le}\n{rec["entity_sequence"]}\n')


def write_edge_tables(out_dir: str, entities: dict[str, dict],
                      edges: dict[str, object], settings: dict) -> dict:
    """Write one CSV per edge family plus entity_edges_provenance.json, which records
    the homolog axis and the edge-group counts. Returns that record."""
    os.makedirs(out_dir, exist_ok=True)
    for family in ('exact_token', 'alignment', 'mmseqs_cov0', 'mmseqs_containment',
                   'partner_chain'):
        groups = edges.get(family, {})
        members = groups.values() if isinstance(groups, dict) else groups
        keys = list(groups.keys()) if isinstance(groups, dict) else range(len(groups))
        path = os.path.join(out_dir, f'entity_edges_{family}.csv')
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['edge_group', 'logical_entity_id', 'label_set_id', 'bmrb_id'])
            for gid, ms in zip(keys, members):
                for le in ms:
                    rec = entities.get(le, {})
                    w.writerow([gid, le, rec.get('label_set_id', ''),
                                rec.get('bmrb_id', '')])
    prov = {**settings,
            'n_realized_entities': len(entities),
            'edge_group_counts': {
                f: len(edges.get(f) or ())
                for f in ('exact_token', 'alignment', 'mmseqs_cov0',
                          'mmseqs_containment', 'partner_chain')}}
    prov_path = os.path.join(out_dir, 'entity_edges_provenance.json')
    with open(prov_path, 'w') as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)
    return prov


def _find_masters(data_dir: str) -> list[str]:
    return [p for p in (os.path.join(data_dir, n) for n in SOURCE_MASTERS)
            if os.path.isfile(p)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Entity-grained sequence clustering.')
    ap.add_argument('--data_dir', default=DATA_DIR,
                    help='Directory holding the entity-grained source masters.')
    ap.add_argument('--out_dir', default=None,
                    help='Edge-table output dir (default: <data_dir>).')
    ap.add_argument('--mmseqs_bin', default=os.environ.get('MMSEQS_BIN'))
    args = ap.parse_args(argv)
    out_dir = args.out_dir or args.data_dir

    masters = _find_masters(args.data_dir)
    if not masters:
        print(f'ERROR: no entity-grained masters found under {args.data_dir}',
              file=sys.stderr)
        return 2
    print(f'Reading {len(masters)} master(s) for realized entities...')
    # Production requires build-log-derived modeled regions, never the sparse master
    # rows.
    entities = collect_realized_entities(masters, require_build_logs=True)
    print(f'  realized logical entities: {len(entities):,}')

    from .sequence_clustering import resolve_mmseqs
    mmseqs_bin = args.mmseqs_bin or resolve_mmseqs()
    fasta_path = os.path.join(out_dir, 'entity_query.fasta')
    write_entity_fasta(entities, fasta_path)
    print(f'  wrote FASTA -> {fasta_path}')

    edges = build_entity_edges(entities, mmseqs_binary=mmseqs_bin)
    settings = {'identity': MMSEQS_IDENTITY, 'coverage': MMSEQS_COVERAGE, 'evalue': EVALUE}
    prov = write_edge_tables(out_dir, entities, edges, settings)
    print('  edge groups: ' + json.dumps(prov['edge_group_counts']))
    print(f'  wrote edge tables -> {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
