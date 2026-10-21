"""Suppress co-located protein chains assigned to a different data fold."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


def suppression_key(pdb_path, chain) -> str:
    """The map key for one target chain: the build log's structure basename and the
    chain. The producer and every consumer key on the build-log path as written."""
    return f"{os.path.basename(str(pdb_path))}\t{chain}"


def _rows(path):
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if (row.get("status") or "success") != "success":
                continue
            label_set = row.get("label_set_id")
            pdb_path = row.get("pdb_path")
            chain = row.get("chain_id")
            if label_set and pdb_path and chain:
                yield label_set, str(pdb_path), chain


def build_suppression_map(build_logs, fold_map):
    """Return ``suppression_key(structure, target_chain) -> partner chains`` plus cell
    accounting."""
    by_structure = defaultdict(set)
    paths_by_basename = defaultdict(set)
    n_rows = 0
    for path in build_logs:
        for label_set, structure, chain in _rows(path):
            by_structure[structure].add((label_set, chain))
            paths_by_basename[os.path.basename(structure)].add(structure)
            n_rows += 1

    collisions = {name: sorted(paths) for name, paths in paths_by_basename.items()
                  if len(paths) > 1}
    if collisions:
        examples = dict(sorted(collisions.items())[:5])
        raise ValueError(
            "partner suppression is keyed by structure basename, but distinct structure "
            f"paths share {len(collisions)} basename(s); examples={examples}"
        )

    suppression = defaultdict(set)
    cross_fold_pairs = 0
    multichain_structures = 0
    missing_fold = set()
    for structure, members in sorted(by_structure.items()):
        if len({chain for _label_set, chain in members}) < 2:
            continue
        multichain_structures += 1
        for target_label_set, target_chain in sorted(members):
            target_fold = fold_map.get(target_label_set)
            if target_fold is None:
                missing_fold.add(target_label_set)
                continue
            for partner_label_set, partner_chain in sorted(members):
                if partner_chain == target_chain:
                    continue
                partner_fold = fold_map.get(partner_label_set)
                if partner_fold is None:
                    missing_fold.add(partner_label_set)
                    continue
                if partner_fold != target_fold:
                    suppression[suppression_key(structure, target_chain)].add(partner_chain)
                    cross_fold_pairs += 1
    if missing_fold:
        raise ValueError(
            f"{len(missing_fold)} materialized label sets have no fold assignment; "
            f"examples={sorted(missing_fold)[:10]}"
        )
    result = {key: sorted(chains) for key, chains in sorted(suppression.items())}
    report = {
        "schema": "deepshift.partner-chain-suppression.v1",
        "rule": "suppress a co-located dataset entity when its fold differs from the target",
        "build_log_rows": n_rows,
        "structures": len(by_structure),
        "multichain_structures": multichain_structures,
        "cross_fold_pairs": cross_fold_pairs,
        "suppression_keys": len(result),
        "key_namespace": "unique structure basename plus target chain",
        "residual_scope": (
            "partner chains that are not dataset entities are not assigned a fold; "
            "sample matching still controls whether they enter the cloud"
        ),
    }
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-logs", nargs="+", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    fold_map = json.loads(args.fold_map.read_text())
    suppression, report = build_suppression_map(args.build_logs, fold_map)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(suppression, indent=2, sort_keys=True) + "\n")
    report_path = args.report or args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.out}: {report['suppression_keys']} target-chain keys, "
        f"{report['cross_fold_pairs']} cross-fold pairs"
    )


if __name__ == "__main__":
    main()
