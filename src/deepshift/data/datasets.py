#!/usr/bin/env python3
"""Bind chemical-shift labels to structure residues, entity-grained."""

import csv
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# Entity-grained materialization: consumes discovery's selection tables +
# entity-grained shifts, emits row-level structure_data_{af,nmr,xtal}.csv keyed by
# (label_set_id, source, structure_instance_id); every row's label ppm/ambiguity is
# copied verbatim from one authoritative row, so sources never disagree on a label.
SELECTION_ACCESSION = {'xtal': 'pdb_id', 'nmr': 'pdb_id', 'af': 'uniprot'}
DUPLICATE_TARGET_EPS = 0.001
DUPLICATE_TARGET_FRACTION = 0.90
DUPLICATE_TARGET_MIN_SHARED = 2


def _assigned_shift_cells(entity):
    """Canonical assigned cells keyed by polymer position, residue code, and atom."""
    cells = {}
    for residue in entity.get('residues', []):
        for column, raw in residue.get('cells', {}).items():
            if not str(column).endswith('_shift'):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            key = (int(residue['comp_index_id']),
                   str(residue['comp_id']).upper(),
                   str(column)[:-6].lower())
            cells[key] = value
    return cells


def _target_agreement(left, right, eps=DUPLICATE_TARGET_EPS):
    """Fraction/per-nucleus bit agreement over shared assigned atom keys."""
    shared = sorted(set(left) & set(right))
    by_nucleus = defaultdict(lambda: [0, 0])
    identical = 0
    for key in shared:
        atom = key[2]
        same = abs(left[key] - right[key]) <= eps
        by_nucleus[atom][0] += 1
        by_nucleus[atom][1] += int(same)
        identical += int(same)
    per_nucleus = {
        atom: {'n_shared': counts[0], 'n_identical': counts[1],
               'fraction_identical': counts[1] / counts[0]}
        for atom, counts in sorted(by_nucleus.items())
    }
    fraction = identical / len(shared) if shared else None
    return fraction, len(shared), identical, per_nucleus


def _geometry_payload(geometry):
    by_source = defaultdict(list)
    for source, structure_instance in sorted(geometry):
        by_source[source].append(structure_instance)
    return dict(sorted(by_source.items()))


def deduplicate_materialized_labelsets(entity_rows, source_results,
                                       fraction_cutoff=DUPLICATE_TARGET_FRACTION,
                                       value_eps=DUPLICATE_TARGET_EPS,
                                       min_shared_atoms=DUPLICATE_TARGET_MIN_SHARED):
    """Drop duplicate label sets, after structure binding.

    Within a (bmrb_id, exact sequence hash) group, an alternate is dropped only when its
    assigned-cell keys are a value-agreeing (``value_eps``) subset of a retained twin's
    (>= ``min_shared_atoms``) and its realized geometry is also a subset of the twin's --
    a different chain stays distinct even with a copied target vector.

    Returns ``(filtered_source_results, ledger_rows, dropped_label_sets)``; excluded
    rows carry ``reason=dropped_duplicate``.
    """
    geometries = defaultdict(set)
    for source, (rows, _logs) in source_results.items():
        for row in rows:
            lsid = str(row.get('label_set_id', ''))
            sid = str(row.get('structure_instance_id', ''))
            if lsid and sid:
                geometries[lsid].add((source, sid))

    groups = defaultdict(list)
    shift_cells = {}
    for lsid, entity in entity_rows.items():
        ident = entity.get('identity', {})
        groups[(str(ident.get('bmrb_id', '')),
                str(ident.get('exact_polymer_sequence_sha256', '')))].append(lsid)
        shift_cells[lsid] = _assigned_shift_cells(entity)

    ledger_rows, dropped = [], set()
    for (bmrb, seq_hash), labelsets in sorted(groups.items()):
        kept = []
        # Richer target (more shift cells, then more geometry) is considered first, so
        # a poorer twin can never delete it merely because its id sorts later.
        ordered_labelsets = sorted(
            labelsets, key=lambda ls: (
                -len(shift_cells[ls]), -len(geometries[ls]), ls))
        for alternate in ordered_labelsets:
            if not kept:
                kept.append(alternate)
                continue
            comparisons = []
            for retained in kept:
                fraction, n_shared, n_identical, per_nucleus = _target_agreement(
                    shift_cells[retained], shift_cells[alternate], eps=value_eps)
                same_geometry = bool(geometries[alternate]) and \
                    geometries[alternate].issubset(geometries[retained])
                alternate_is_subset = bool(shift_cells[alternate]) and \
                    set(shift_cells[alternate]).issubset(shift_cells[retained])
                comparisons.append((retained, fraction, n_shared, n_identical,
                                    per_nucleus, same_geometry,
                                    alternate_is_subset))
            # Highest target agreement first; ties resolve by retained label-set id.
            comparisons.sort(key=lambda item: (
                -(item[1] if item[1] is not None else -1.0), item[0]))
            qualifying = [
                item for item in comparisons
                if item[1] is not None
                and item[1] >= fraction_cutoff
                and item[2] >= min_shared_atoms
                and item[3] == item[2]
                and item[6]
            ]
            droppable = [item for item in qualifying if item[5]]
            if droppable:
                chosen = droppable[0]
                status, review = 'dropped_duplicate', ''
                dropped.add(alternate)
            else:
                chosen = qualifying[0] if qualifying else comparisons[0]
                if qualifying:
                    status = 'kept_different_geometry' \
                        if geometries[alternate] else 'kept_no_geometry'
                    review = 'bit_identical_targets_different_geometry' \
                        if geometries[alternate] else 'bit_identical_targets_no_geometry'
                elif chosen[1] is None:
                    status, review = 'kept_no_shared_atoms', ''
                elif chosen[2] < min_shared_atoms:
                    status, review = 'kept_insufficient_shared_atoms', ''
                elif chosen[3] == chosen[2] and not chosen[6]:
                    status, review = 'kept_unique_targets', ''
                else:
                    status, review = 'kept_distinct', ''
                kept.append(alternate)

            (retained, fraction, n_shared, n_identical, per_nucleus,
             same_geometry, _alternate_is_subset) = chosen
            alt_identity = entity_rows[alternate].get('identity', {})
            ret_identity = entity_rows[retained].get('identity', {})
            ledger_rows.append({
                'bmrb_id': bmrb,
                'exact_polymer_sequence_sha256': seq_hash,
                'logical_entity_id': alt_identity.get('logical_entity_id', ''),
                'retained_logical_entity_id': ret_identity.get('logical_entity_id', ''),
                'retained_label_set_id': retained,
                'alternate_label_set_id': alternate,
                'fraction_identical': '' if fraction is None else fraction,
                'n_shared_atoms': n_shared,
                'n_identical_atoms': n_identical,
                'per_nucleus_agreement_json': json.dumps(
                    per_nucleus, sort_keys=True, separators=(',', ':')),
                'retained_structure_instances_json': json.dumps(
                    _geometry_payload(geometries[retained]), sort_keys=True,
                    separators=(',', ':')),
                'alternate_structure_instances_json': json.dumps(
                    _geometry_payload(geometries[alternate]), sort_keys=True,
                    separators=(',', ':')),
                'same_structure_geometry': same_geometry,
                'status': status,
                'review_flag': review,
            })

    filtered = {}
    for source, (rows, logs) in source_results.items():
        kept_rows = [row for row in rows
                     if str(row.get('label_set_id', '')) not in dropped]
        kept_logs = []
        for row in logs:
            if str(row.get('label_set_id', '')) in dropped:
                row = {**row, 'status': 'excluded',
                       'reason': 'dropped_duplicate'}
            kept_logs.append(row)
        filtered[source] = (kept_rows, kept_logs)
    return filtered, ledger_rows, dropped




def load_ranked_candidates(path, source):
    """Read one source lane from discovery's ranked candidate ledger.

    Rows without a positive ``candidate_rank`` (method-excluded EM/other/unknown)
    are skipped -- never allowed into AF/NMR/X-ray fallback.
    """
    if not os.path.exists(path):
        return []
    # keep_default_na=False: a literal chain id of "NA" is real, not pandas' NA token.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'label_set_id', 'source', 'candidate_id', 'candidate_rank',
                'accession', 'chain', 'route'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f'{path} is not a ranked discovery ledger; missing {sorted(missing)}')
    duplicate_ids = sorted(
        df.loc[df['candidate_id'].duplicated(keep=False), 'candidate_id'].unique()
    )
    if duplicate_ids:
        raise ValueError(
            f"{path} contains duplicate candidate_id values: {duplicate_ids[:10]}"
        )
    rows = []
    for row in df[df['source'] == source].to_dict('records'):
        try:
            rank = int(row['candidate_rank'])
        except (TypeError, ValueError):
            continue
        if rank < 1:
            raise ValueError(f"invalid candidate rank {rank} for {row['candidate_id']}")
        row['candidate_rank'] = rank
        rows.append(row)
    return sorted(rows, key=lambda r: (r['label_set_id'], r['candidate_rank'],
                                       r['candidate_id']))


def entity_label_rows(shifts_file, catalog_file=None):
    """Group the entity-grained selected chemical_shifts.csv by label set.

    Returns ``{label_set_id: {'identity': {...}, 'sequence': str, 'seq_positions':
    [int,...], 'polymer_length': int, 'residues': [ordered {comp_index_id, comp_id,
    label_residue_id, cells: {col: value}}]}}``. ``sequence``/``seq_positions`` are the
    full entity alignment (from the ``entity_catalog.csv`` sidecar), not the
    observed-only shift rows, so a label maps to its own residue by ``comp_index_id``.
    The sidecar is required.
    """
    from .fetch import default_entity_catalog_path, read_entity_catalog
    # low_memory=False is required: per-chunk type inference segfaults pandas' C parser
    # at 64 columns x ~1.5M rows; low_memory=False reads all 1,509,465 rows.
    df = pd.read_csv(shifts_file, dtype={'bmrb_id': str, 'label_set_id': str},
                     low_memory=False)
    if 'label_set_id' not in df.columns:
        raise ValueError(f'{shifts_file} is not entity-grained (no label_set_id)')
    catalog_file = catalog_file or default_entity_catalog_path(shifts_file)
    if not os.path.exists(catalog_file):
        raise ValueError(
            f'entity_catalog.csv full-sequence sidecar not found ({catalog_file}); '
            'materialization requires the FULL entity sequence to map labels to '
            'true polymer positions. Refusing observed-only fallback.')
    catalog = read_entity_catalog(catalog_file)
    id_cols = ['bmrb_id', 'entity_id', 'shift_list_id', 'entity_assembly_id',
               'exact_polymer_sequence_sha256', 'logical_entity_id']
    label_cols = [c for c in df.columns
                  if c.endswith('_shift') or c.endswith('_ambiguity_code')]
    out = {}
    for lsid, g in df.groupby('label_set_id', sort=True):
        cat = catalog.get(lsid)
        if cat is None:
            raise ValueError(
                f'label set {lsid} has selected shifts but no entity_catalog.csv '
                f'row -- inconsistent full-sequence sidecar (fail-closed).')
        g = g.sort_values('comp_index_id')
        first = g.iloc[0]
        residues = []
        for _, r in g.iterrows():
            comp_id = str(r['comp_id']).upper()
            cells = {c: r[c] for c in label_cols if pd.notna(r[c])}
            residues.append({'comp_index_id': int(r['comp_index_id']),
                             'comp_id': comp_id,
                             'label_residue_id': r.get('label_residue_id'),
                             'cells': cells})
        out[lsid] = {
            'identity': {c: ('' if c not in g.columns or pd.isna(first[c])
                             else str(first[c])) for c in id_cols},
            'sequence': cat['alignment_sequence'],
            'seq_positions': cat['seq_positions'],
            'seq_tokens': cat.get('seq_tokens') or [],
            'polymer_length': cat['polymer_length'],
            'residues': residues}
    return out


def _is_insertion_coded(struct_rid):
    """True when a mapped structure residue id is a ``(seq, icode)`` insertion-code tuple.

    The integer-keyed atom cache and SHIFTX rereferencer can't represent an insertion
    code, so such a label residue is excluded.
    """
    return isinstance(struct_rid, tuple)


def materialize_source(source, selection_rows, entity_rows, structure_loader,
                       build_id, min_query_coverage=0.90, min_identity=0.95,
                       min_aligned_length=10):
    """Realize one source's selected label sets into row-level master + log.

    ``structure_loader(source, accession, chain, entity_seq)`` returns a dict with
    keys ``struct_res_ids``, ``struct_seq``, ``coord_sha256``, ``model_index``,
    ``preprocessing_state``, ``chain``, or ``None`` if unrealizable.

    Returns (master_rows, log_rows); a selected structure is excluded unless it clears
    every realization gate: coverage >= ``min_query_coverage``,
    identity >= ``min_identity``, >= ``min_aligned_length`` aligned residues. A label
    landing on an insertion-coded residue is excluded.
    """
    from .. import domain as _id
    from . import structures as _S
    from .alignment import ALIGNMENT_AMBIGUITY_MARGIN, frame_safe_alignments

    master_rows, log_rows = [], []
    realized_label_sets = set()

    def _rank(row, fallback):
        try:
            return int(row.get('candidate_rank', fallback))
        except (TypeError, ValueError):
            return fallback

    ordered = sorted(enumerate(selection_rows), key=lambda item: (
        str(item[1].get('label_set_id', '')), _rank(item[1], item[0] + 1),
        str(item[1].get('candidate_id', ''))))
    for original_index, sel in ordered:
        lsid = sel['label_set_id']
        ent = entity_rows.get(lsid)
        chain = sel.get('chain', '') or 'A'
        acc = (sel.get(SELECTION_ACCESSION[source], '')
               or sel.get('accession', ''))
        base_log = {'label_set_id': lsid, 'source': source, 'accession': acc,
                    'candidate_id': sel.get('candidate_id', ''),
                    'candidate_rank': _rank(sel, original_index + 1),
                    'route': sel.get('route', ''),
                    'requested_chain': sel.get('chain', ''),
                    'chain': chain, 'structure_instance_id': '',
                    'n_label_residues': 0, 'n_mapped': 0,
                    'n_aligned': '', 'query_coverage': '', 'identity': '',
                    'status': 'excluded'}
        if lsid in realized_label_sets:
            log_rows.append({**base_log, 'status': 'not_attempted',
                             'reason': 'lower_ranked_after_success'})
            continue
        if ent is None:
            log_rows.append({**base_log, 'reason': 'no_label_rows'})
            continue
        full_seq = ent['sequence']
        loaded = structure_loader(source, acc, chain, entity_seq=full_seq)
        if loaded is None:
            log_rows.append({**base_log, 'reason': 'structure_unavailable'})
            continue
        # Align the full entity sequence, not the observed-only subset, so a label at
        # comp_index_id=10 maps to residue 10, never the next match of a compressed
        # 3-mer.
        seq_positions = ent.get('seq_positions') or list(
            range(1, len(full_seq) + 1))
        # A conservative score contract that also rejects a near-co-optimal alternate
        # mapping, so a repeated-motif gap placement never binds shifts to arbitrary
        # coordinates.
        alignments = frame_safe_alignments(
            loaded['struct_seq'], full_seq,
            score_margin=ALIGNMENT_AMBIGUITY_MARGIN)
        iterator = iter(alignments)
        try:
            aln = next(iterator)
        except StopIteration:
            log_rows.append({**base_log, 'reason': 'alignment_fail'})
            continue
        stats = _S._alignment_stats(aln)
        used_chain = loaded['chain']
        # Mandatory realization gates: an unrelated equal-length
        # chain can align at near-1 coverage with very low identity, so identity is
        # checked too, not coverage alone.
        gate = {**base_log, 'chain': used_chain,
                'n_aligned': stats['n_aligned'],
                'query_coverage': round(stats['query_coverage'], 4),
                'identity': round(stats['identity'], 4)}
        if stats['identity'] < min_identity:
            log_rows.append({**gate, 'reason': 'low_identity'})
            continue
        if stats['query_coverage'] < min_query_coverage:
            log_rows.append({**gate, 'reason': 'low_query_coverage'})
            continue
        if stats['n_aligned'] < min_aligned_length:
            log_rows.append({**gate, 'reason': 'too_few_aligned'})
            continue
        sid = _id.structure_instance_id(
            source, acc, used_chain, loaded['model_index'],
            loaded['coord_sha256'], loaded['preprocessing_state'])
        # {polymer_position -> {struct_res_id, mismatch_type}} over the whole entity.
        mapping = {m['comp_index_id']: m for m in _S.label_residue_mapping(
            aln, loaded['struct_res_ids'], seq_positions)}
        # Ambiguity is judged on label-bearing positions: a co-optimal alternate
        # alignment invalidates this realization only if it re-places a shift-bearing
        # residue. Disagreement confined to unlabeled positions is kept.
        shift_bearing = {key[0] for key in _assigned_shift_cells(ent)}
        ambiguous = False
        for _ in range(64):
            try:
                alternative = next(iterator)
            except StopIteration:
                break
            margin = float(aln.score - alternative.score)
            if margin > ALIGNMENT_AMBIGUITY_MARGIN + 1e-9:
                # Beyond the co-optimality margin; alternatives arrive best-score-first,
                # so none that follow compete either.
                break
            alternative_mapping = {
                m['comp_index_id']: m for m in _S.label_residue_mapping(
                    alternative, loaded['struct_res_ids'], seq_positions)}
            if any(mapping.get(c) != alternative_mapping.get(c)
                   for c in shift_bearing):
                ambiguous = True
                break
        if ambiguous:
            log_rows.append({**gate, 'reason': 'ambiguous_alignment'})
            continue
        # mismatch_type is computed catalog-vs-structure, so it reads 'match' even when
        # a label row's comp_id disagrees with the catalog token. Every label code is
        # cross-checked against the token authority and a disagreement is refused,
        # covering any CSV not already validated at fetch time.
        pos_to_token = dict(zip(ent.get('seq_positions') or [],
                               ent.get('seq_tokens') or []))
        n_mapped = n_icode = n_code_mismatch = 0
        realization_rows = []
        for res in ent['residues']:
            m = mapping.get(res['comp_index_id'], {})
            struct_rid = m.get('struct_res_id')
            mismatch_type = m.get('mismatch_type', 'gap_in_structure')
            token = pos_to_token.get(res['comp_index_id'])
            if token is not None and str(token).upper() != str(res['comp_id']).upper():
                # Wrong-amino-acid label: never emit as a usable target.
                struct_rid, mismatch_type = None, 'label_residue_code_mismatch'
                n_code_mismatch += 1
            elif _is_insertion_coded(struct_rid):
                # Frame is correct, but the atom cache/SHIFTX reref can't key an
                # insertion code -- exclude explicitly, never mis-bind it.
                struct_rid, mismatch_type = None, 'insertion_code_unsupported'
                n_icode += 1
            elif struct_rid is not None:
                n_mapped += 1
            row = {
                'label_set_id': lsid,
                'label_residue_id': res['label_residue_id'],
                'source': source,
                'structure_instance_id': sid,
                'comp_index_id': res['comp_index_id'],
                'comp_id': res['comp_id'],
                'residue_code': res['comp_id'],
                'struct_res_id': '' if struct_rid is None else struct_rid,
                'mismatch_type': mismatch_type,
                'accession': acc, 'chain': used_chain,
                # Provenance so the cache's id_maps can recompute structure_instance_id
                # + materialization_id.
                'model_index': loaded['model_index'],
                'coordinate_file_sha256': loaded['coord_sha256'],
                'preprocessing_state': loaded['preprocessing_state'],
                'build_id': build_id,
                **ent['identity'],
                **res['cells'],           # label ppm + ambiguity, verbatim
            }
            realization_rows.append(row)
        # The modeled region is every polymer position this structure actually placed,
        # not just the labeled residues -- leakage clustering needs the full modeled
        # domain or it can compress the polymer and miss a hard homology edge.
        # Insertion-coded placements still count here; only the label is masked.
        modeled_positions = sorted(
            cix for cix, m in mapping.items()
            if m.get('struct_res_id') is not None
            and m.get('mismatch_type') != 'gap_in_structure')
        outcome = {
            **gate,
            'structure_instance_id': sid,
            'n_label_residues': len(ent['residues']),
            'n_mapped': n_mapped,
            'n_insertion_code_excluded': n_icode,
            'n_label_residue_code_mismatch': n_code_mismatch,
            'n_modeled_positions': len(modeled_positions),
            'modeled_positions': ';'.join(str(p) for p in modeled_positions),
        }
        if n_mapped == 0:
            # A structure can align globally yet every shift-bearing label lands on a
            # gap/insertion/wrong residue; publishing that as success would suppress a
            # lower-ranked candidate that might work.
            log_rows.append({
                **outcome,
                'status': 'excluded',
                'reason': 'zero_mapped_label_residues',
            })
            continue
        master_rows.extend(realization_rows)
        log_rows.append({**outcome, 'status': 'success', 'reason': 'kept'})
        realized_label_sets.add(lsid)
    return master_rows, log_rows


def _authoritative_selection(source, candidate, success_log):
    """Build the finalized selection row for one actually realized candidate."""
    ident = {c: candidate.get(c, '') for c in (
        'label_set_id', 'bmrb_id', 'entity_id', 'shift_list_id',
        'entity_assembly_id', 'exact_polymer_sequence_sha256',
        'logical_entity_id')}
    common = {
        **ident, 'candidate_id': candidate.get('candidate_id', ''),
        'candidate_rank': candidate.get('candidate_rank', ''), 'source': source,
        'chain': success_log.get('chain', ''),
        'requested_chain': candidate.get('chain', ''),
        'identity': (float(success_log.get('identity', 0.0)) * 100.0),
        'query_coverage': success_log.get('query_coverage', ''),
        'aligned_length': success_log.get('n_aligned', ''),
        'route': candidate.get('route', ''),
    }
    accession = candidate.get('accession', '')
    if source == 'af':
        return {**common, 'uniprot': accession,
                'af_model': f'AF-{accession}-F1-model_v6.pdb',
                'mapping_confidence': candidate.get('mapping_confidence', '')}
    method = 'NMR' if source == 'nmr' else 'X-RAY DIFFRACTION'
    return {**common, 'pdb_id': accession, 'method': method,
            'resolution': candidate.get('resolution', ''),
            'subject_coverage': candidate.get('subject_coverage', '')}


def _write_realization_ledger(path, rows):
    lead = ['label_set_id', 'logical_entity_id', 'bmrb_id', 'entity_id',
            'shift_list_id', 'entity_assembly_id',
            'exact_polymer_sequence_sha256', 'source', 'candidate_id',
            'candidate_rank', 'accession', 'requested_chain', 'chain', 'route',
            'status', 'reason', 'identity', 'query_coverage', 'n_aligned',
            'structure_instance_id']
    cols = set(lead)
    for row in rows:
        cols.update(row)
    ordered = lead + sorted(cols - set(lead))
    pd.DataFrame(rows, columns=ordered).to_csv(path, index=False)


def write_master_csv(path, rows):
    """Write row-level master with a stable identity-first column union."""
    lead = ['label_set_id', 'label_residue_id', 'source', 'structure_instance_id',
            'bmrb_id', 'entity_id', 'shift_list_id', 'entity_assembly_id',
            'exact_polymer_sequence_sha256', 'logical_entity_id',
            'comp_index_id', 'comp_id', 'residue_code', 'struct_res_id',
            'mismatch_type', 'accession', 'chain']
    cols = set()
    for r in rows:
        cols.update(r)
    ordered = [c for c in lead if c in cols] + sorted(c for c in cols if c not in lead)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction='ignore', restval='')
        w.writeheader()
        w.writerows(rows)


def materialize_masters(out_dir, shifts_file, selection_dir, structure_loader,
                        build_id, min_query_coverage=0.90, min_identity=0.95,
                        min_aligned_length=10):
    """Emit row-level structure_data_{af,nmr,xtal}.csv + build_log_{source}.csv.

    Reads discovery's per-source selection tables + entity-grained shifts; realization
    gates (coverage, identity, aligned length) are enforced per source.
    """
    ledger_path = os.path.join(selection_dir, 'candidate_ledger.csv')
    if not os.path.isfile(ledger_path):
        raise FileNotFoundError(
            "materialization requires discovery's ranked candidate_ledger.csv; "
            f"not found: {ledger_path}"
        )
    os.makedirs(out_dir, exist_ok=True)
    entity_rows = entity_label_rows(shifts_file)
    source_results = {}
    candidates_by_source = {}
    for source in ('af', 'nmr', 'xtal'):
        sel = load_ranked_candidates(ledger_path, source)
        rows, log_rows = materialize_source(
            source, sel, entity_rows, structure_loader, build_id,
            min_query_coverage=min_query_coverage, min_identity=min_identity,
            min_aligned_length=min_aligned_length)
        source_results[source] = (rows, log_rows)
        candidates_by_source[source] = sel

    source_results, dedup_rows, dropped = deduplicate_materialized_labelsets(
        entity_rows, source_results)
    dedup_columns = [
        'bmrb_id', 'exact_polymer_sequence_sha256', 'logical_entity_id',
        'retained_logical_entity_id', 'retained_label_set_id',
        'alternate_label_set_id', 'fraction_identical', 'n_shared_atoms',
        'n_identical_atoms', 'per_nucleus_agreement_json',
        'retained_structure_instances_json',
        'alternate_structure_instances_json', 'same_structure_geometry',
        'status', 'review_flag']
    pd.DataFrame(dedup_rows, columns=dedup_columns).to_csv(
        os.path.join(out_dir, 'label_set_dedup.csv'), index=False)

    summary = {'dedup': {'n_comparisons': len(dedup_rows),
                         'n_dropped': len(dropped)}}
    realization_rows = []
    for source in ('af', 'nmr', 'xtal'):
        rows, log_rows = source_results[source]
        write_master_csv(os.path.join(out_dir, f'structure_data_{source}.csv'), rows)
        pd.DataFrame(log_rows).to_csv(
            os.path.join(out_dir, f'build_log_{source}.csv'), index=False)
        candidate_by_id = {str(r.get('candidate_id', '')): r
                           for r in candidates_by_source[source]}
        finalized = []
        for log_row in log_rows:
            candidate = candidate_by_id.get(str(log_row.get('candidate_id', '')))
            combined = ({**candidate, **log_row} if candidate is not None
                        else dict(log_row))
            realization_rows.append(combined)
            if log_row.get('status') == 'success' and candidate is not None:
                finalized.append(_authoritative_selection(
                    source, candidate, log_row))
        # build.py assigns discovery_selection_* to discover, selection_* to this stage.
        selection_cols = None
        if source == 'af':
            selection_cols = [
                'label_set_id', 'bmrb_id', 'entity_id', 'shift_list_id',
                'entity_assembly_id', 'exact_polymer_sequence_sha256',
                'logical_entity_id', 'candidate_id', 'candidate_rank', 'source',
                'uniprot', 'af_model', 'chain', 'requested_chain', 'identity',
                'query_coverage', 'aligned_length', 'mapping_confidence', 'route']
        else:
            selection_cols = [
                'label_set_id', 'bmrb_id', 'entity_id', 'shift_list_id',
                'entity_assembly_id', 'exact_polymer_sequence_sha256',
                'logical_entity_id', 'candidate_id', 'candidate_rank', 'source',
                'pdb_id', 'chain', 'requested_chain', 'method', 'resolution',
                'identity', 'query_coverage', 'aligned_length',
                'subject_coverage', 'route']
        pd.DataFrame(finalized, columns=selection_cols).to_csv(
            os.path.join(out_dir, f'selection_{source}.csv'), index=False)
        summary[source] = {'candidates': len(candidates_by_source[source]),
                           'selected': len(finalized),
                           'materialized_label_sets':
                               sum(1 for r in log_rows if r['status'] == 'success'),
                           'rows': len(rows)}
    _write_realization_ledger(
        os.path.join(out_dir, 'realization_ledger.csv'), realization_rows)
    return summary
