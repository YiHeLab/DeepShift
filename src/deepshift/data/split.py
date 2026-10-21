#!/usr/bin/env python3
"""Route entity-grained master rows to the per-fold CSVs the cache stage reads.

Streams the AF/NMR/crystal source masters, keys each row on ``label_set_id``, and
routes it through the label-set fold map, failing closed on a non-entity-grained
master, a missing identity column, or an unmapped label_set_id."""

import argparse
import csv
import json
import os
from collections import Counter

csv.field_size_limit(10 ** 7)


def _read_header(path):
    with open(path, newline='') as handle:
        return next(csv.reader(handle))


def _parse_source_master(spec):
    if ':' not in spec:
        raise argparse.ArgumentTypeError(
            f"source master must be SOURCE:PATH, got {spec!r}")
    source, path = spec.split(':', 1)
    if source not in ('af', 'nmr', 'xtal'):
        raise argparse.ArgumentTypeError(
            f"source must be af, nmr, or xtal, got {source!r}")
    return source, path


# Each per-fold CSV preserves full identity (label set, entity, residue,
# structure/sample ids), ambiguity codes, and every shift column; a row missing a
# required id stops the build.

# Required on every production row. The cache keys on materialization_id and
# asserts one materialization per (sample_id, source).
REQUIRED_IDENTITY = ('label_set_id', 'logical_entity_id', 'label_residue_id',
                     'bmrb_id', 'entity_id', 'shift_list_id', 'entity_assembly_id',
                     'exact_polymer_sequence_sha256', 'comp_index_id', 'comp_id',
                     'source', 'structure_instance_id', 'sample_id',
                     'materialization_id')


def _union_header(master_paths):
    """Deterministic column union across the source masters (identity first).
    Every column name appears once."""
    lead = [*list(REQUIRED_IDENTITY), 'residue_code', 'struct_res_id', 'mismatch_type', 'accession', 'chain']
    seen, extra = set(lead), []
    present = set()
    for path in master_paths:
        for col in _read_header(path):
            present.add(col)
            if col not in seen:
                seen.add(col)
                extra.append(col)
    ordered = [c for c in lead if c in present] + extra
    return ordered, present


def route_labelset_masters(master_paths, ls_fold_map, out_dir,
                           dataset='unified', require_identity=True):
    """Route entity-grained master rows to per-fold CSVs by label_set_id, failing
    closed on a non-entity-grained master, a missing required identity column, or
    an unmapped label_set_id. Writes one CSV per fold."""
    os.makedirs(out_dir, exist_ok=True)
    body_header, present = _union_header(master_paths)
    # The cache builder selects rows by a 'split' (fold) column; each per-fold CSV
    # already holds only its fold's rows, so 'split' is that fold on every row.
    header = ['split', *body_header]
    if 'label_set_id' not in present:
        raise ValueError('masters are not entity-grained (no label_set_id)')
    if require_identity:
        missing = [c for c in REQUIRED_IDENTITY if c not in present]
        if missing:
            raise ValueError(f'masters lack required identity columns: {missing}')

    writers, handles, paths = {}, {}, {}
    counts = Counter()

    def writer_for(fold):
        if fold in writers:
            return writers[fold]
        final = os.path.join(out_dir, f'structure_data_{dataset}_fold_{fold}.csv')
        tmp = final + f'.tmp.{os.getpid()}'
        handle = open(tmp, 'w', newline='', encoding='utf-8')
        w = csv.writer(handle)
        w.writerow(header)
        handles[fold] = handle
        writers[fold], paths[fold] = w, (tmp, final)
        return w

    try:
        for path in master_paths:
            with open(path, newline='') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    lsid = row.get('label_set_id')
                    if not lsid:
                        raise ValueError(f'{path}: row missing label_set_id')
                    if require_identity:
                        for col in REQUIRED_IDENTITY:
                            if row.get(col) in (None, ''):
                                raise ValueError(
                                    f'{path}: row {lsid} missing required id {col!r}')
                    if lsid not in ls_fold_map:
                        raise ValueError(
                            f'{path}: label_set_id {lsid} absent from fold map '
                            '(no -1 drops allowed)')
                    fold = ls_fold_map[lsid]
                    writer_for(fold).writerow(
                        [fold] + [row.get(c, '') for c in body_header])
                    counts[fold] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for _fold, (tmp, final) in paths.items():
        os.replace(tmp, final)
    return {'per_fold_rows': dict(counts)}


def entity_main(argv=None):
    parser = argparse.ArgumentParser(description='Entity-grained fold split.')
    parser.add_argument('--source_master', action='append', type=_parse_source_master,
                        default=[], required=True, metavar='SOURCE:PATH')
    parser.add_argument('--fold_map', required=True,
                        help='label-set fold map (fold_assignments_labelset_identity90.json)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--dataset', default='unified')
    parser.add_argument('--allow_missing_identity', action='store_true',
                        help='relax the production requirement that every full '
                             'identity column be present')
    args = parser.parse_args(argv)
    with open(args.fold_map) as fh:
        ls_fold_map = {str(k): int(v) for k, v in json.load(fh).items()}
    master_paths = [path for _, path in args.source_master]
    summary = route_labelset_masters(
        master_paths, ls_fold_map, args.out_dir, dataset=args.dataset,
        require_identity=not args.allow_missing_identity)
    print('per-fold rows: ' + json.dumps(summary['per_fold_rows']))


def main(argv=None):
    return entity_main(argv)


if __name__ == '__main__':
    main()
