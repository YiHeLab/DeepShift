"""Which protein partners were in the NMR tube? Read from the deposition.

The output is label-set specific.  It names the target entity and the sequence and copy
count of every remaining protein entity declared in the sample.  Atom construction then
admits only coordinate chains matching those declared partner slots.  Unknown metadata,
missing partner sequences, and excess coordinate chains fail to no partner.

The count that matters is PROTEIN copies in the BMRB ``_Entity_assembly`` loop:

    MONOMER  <=1 protein copy -- any structural partner is spurious for these shifts
    HOMO     >1 copy, one entity
    HETERO   more than one protein entity

Read the raw NMR-STAR, not a derived CSV, which collapses homo-oligomers to one copy.

This closes the crystal-lattice failure mode without relying on REMARK 350: a chain absent
from the NMR sample has no declared sequence/copy slot and cannot enter the cloud.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 2


def is_protein(polymer_type, entity_type) -> bool:
    p = (polymer_type or '').lower()
    t = (entity_type or '').lower()
    if 'polypeptide' in p:
        return True
    if any(k in p for k in ('polynucleotide', 'polyribonucleotide', 'polydeoxyribo', 'dna', 'rna')):
        return False
    return t == 'polymer'          # a polymer with no declared type is rare; assume protein


def _entry_sequences(entry) -> dict[str, str]:
    """Full one-letter entity sequences from the same frozen NMR-STAR entry."""
    try:
        from ..domain import alignment_sequence_from_tokens
        from .fetch import parse_entity_poly_seq
        parsed = parse_entity_poly_seq(entry)
    except Exception:
        return {}
    return {
        str(entity_id): alignment_sequence_from_tokens(record["tokens"])
        for entity_id, record in parsed.items()
        if entity_id is not None and record.get("tokens")
    }


def entry_assembly(entry) -> dict:
    """Classify one parsed ``pynmrstar.Entry`` and retain protein member identities."""
    etype, ptype = {}, {}
    for sf in entry.get_saveframes_by_category('entity') or []:
        try:
            eid = sf.get_tag('ID')[0]
        except Exception:
            continue
        for tag, into in (('Type', etype), ('Polymer_type', ptype)):
            try:
                into[eid] = sf.get_tag(tag)[0]
            except Exception:
                into[eid] = None

    loops = entry.get_loops_by_category('entity_assembly')
    if not loops:
        return {'class': 'no_assembly_loop'}
    rows = loops[0].get_tag(['Entity_ID'])
    ids = [str(r[0] if isinstance(r, (list, tuple)) else r) for r in rows]
    protein = [i for i in ids if is_protein(ptype.get(i), etype.get(i))]
    if len(protein) <= 1:
        cls = 'MONOMER'
    elif len(set(protein)) == 1:
        cls = 'HOMO'
    else:
        cls = 'HETERO'
    sequences = _entry_sequences(entry)
    members = []
    for entity_id in sorted(set(protein)):
        members.append({
            "entity_id": entity_id,
            "copies": protein.count(entity_id),
            "alignment_sequence": sequences.get(entity_id),
        })
    return {'class': cls, 'protein_copies': len(protein),
            'protein_entities': len(set(protein)), 'total_assembly_copies': len(ids),
            'protein_members': members}


def label_set_assembly(entry_record, *, bmrb_id, label_set_id, target_entity_id,
                       target_sequence) -> dict:
    """Declared partner slots after removing the target molecule itself."""
    target_entity_id = str(target_entity_id)
    target_sequence = str(target_sequence or "")
    partners = []
    target_seen = False
    missing_partner_sequences = []
    for member in entry_record.get("protein_members", ()):
        entity_id = str(member.get("entity_id"))
        copies = int(member.get("copies", 0))
        if entity_id == target_entity_id:
            target_seen = copies > 0
            copies -= 1
            sequence = member.get("alignment_sequence") or target_sequence
        else:
            sequence = member.get("alignment_sequence")
        if copies <= 0:
            continue
        if not sequence:
            missing_partner_sequences.append(entity_id)
            continue
        partners.append({
            "entity_id": entity_id,
            "copies": copies,
            "alignment_sequence": str(sequence),
        })
    return {
        "bmrb_id": str(bmrb_id),
        "label_set_id": str(label_set_id),
        "target_entity_id": target_entity_id,
        "target_alignment_sequence": target_sequence,
        "class": entry_record.get("class", "unknown"),
        "target_present_in_assembly": target_seen,
        "partners": partners if target_seen else [],
        "missing_partner_sequences": missing_partner_sequences,
    }


def _classify_one(args):
    import pynmrstar
    bmrb, root = args
    hits = sorted(Path(root).glob(f'bmr{bmrb}/*_3.str'))
    if not hits:
        return bmrb, {'class': 'missing'}
    try:
        return bmrb, entry_assembly(pynmrstar.Entry.from_file(str(hits[0])))
    except Exception as exc:
        return bmrb, {'class': 'parse_error', 'err': str(exc)[:120]}


def classify_snapshot(snapshot_dir, bmrb_ids, workers: int = 8) -> dict:
    """``{bmrb_id: record}`` over a BMRB snapshot of ``bmr<ID>/*_3.str`` files."""
    from multiprocessing import Pool

    work = [(str(b), str(snapshot_dir)) for b in bmrb_ids]
    with Pool(workers) as pool:
        return dict(pool.map(_classify_one, work, chunksize=32))


def load(path) -> dict:
    """Label-set records from ``sample_assembly.json``."""
    payload = json.loads(Path(path).read_text())
    if "label_sets" not in payload:
        raise ValueError(f"{path}: sample_assembly.json carries no label_sets")
    return {str(k): v for k, v in payload["label_sets"].items()}


def partner_in_sample(assembly: dict, key) -> bool:
    """True when this label set's sample held a protein partner: the target is in the
    declared assembly and at least one other protein slot remains. An unclassified
    deposition counts as having no partner."""
    record = assembly.get(str(key))
    return (isinstance(record, dict)
            and bool(record.get("target_present_in_assembly") and record.get("partners")))


def main(argv=None):
    import argparse
    import csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--entity-catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    with args.entity_catalog.open(newline="") as stream:
        catalog = list(csv.DictReader(stream))
    bmrb_ids = sorted({str(row["bmrb_id"]) for row in catalog})
    records = classify_snapshot(args.snapshot, bmrb_ids, workers=args.workers)
    label_sets = {
        str(row["label_set_id"]): label_set_assembly(
            records.get(str(row["bmrb_id"]), {}),
            bmrb_id=row["bmrb_id"],
            label_set_id=row["label_set_id"],
            target_entity_id=row["entity_id"],
            target_sequence=row["alignment_sequence"],
        )
        for row in catalog
    }
    counts = {}
    for record in records.values():
        name = record["class"]
        counts[name] = counts.get(name, 0) + 1
    payload = {
        "schema_version": SCHEMA_VERSION,
        "grain": "label_set_id",
        "rule": (
            "include only coordinate chains matching a remaining declared protein "
            "entity sequence and copy slot after removing the target molecule"
        ),
        "unknown_behavior": "exclude partner",
        "counts": dict(sorted(counts.items())),
        "records": records,
        "label_sets": label_sets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {len(records)} BMRB entries, classes={payload['counts']}")


if __name__ == "__main__":
    main()
