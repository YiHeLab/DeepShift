#!/usr/bin/env python3
"""Consolidate the protected UniProt snapshots into a many-candidate ledger.

Reads the immutable, digest-named snapshot files pinned by
the snapshot builder (baseline, prior_augmented, live_checkpoint
and/or live_final) and merges them into one lossless candidate ledger: every
BMRB->accession pair is kept as a separate candidate with its mapping source,
and a conflict (one BMRB mapping to different accessions) is recorded rather than
resolved -- the scientific selection happens later, in the entity alignment
contract. The stage is strictly offline, issuing no BLAST job or download.
"""

from __future__ import annotations

import argparse
import json
import os
import hashlib
import re

import pandas as pd

# Digest files are "<label>.n<rows>.<sha256>.json".
DIGEST_RE = re.compile(r"^(?P<label>.+)\.n(?P<rows>\d+)\.(?P<sha>[0-9a-f]{64})\.json$")

# Which snapshot labels represent a complete mapping and which a partial one.
FINAL_LABELS = {"baseline", "prior_augmented", "live_final"}

LEDGER_COLUMNS = ("bmrb_id", "accession", "mapping_source", "source_final",
                  "source_digest_file", "source_sha256")




def parse_str_str(data: bytes) -> dict:
    """Parse a string->string JSON object; reject malformed/truncated/typed."""
    obj = json.loads(data.decode("utf-8"))  # raises on truncated JSON
    if not isinstance(obj, dict):
        raise ValueError("mapping JSON is not an object")
    for k, v in obj.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("mapping must be string->string")
    return obj


def load_snapshot_files(snapshot_dir):
    """Loaded snapshot records for every digest file present: {label, rows, sha256,
    digest_file, is_final, mapping}. The on-disk sha256 in the filename is verified
    against the bytes; raises on a corrupt/truncated file so bad input can never
    enter the ledger.
    """
    records = []
    for name in sorted(os.listdir(snapshot_dir)):
        m = DIGEST_RE.match(name)
        if not m:
            continue
        path = os.path.join(snapshot_dir, name)
        with open(path, "rb") as fh:
            data = fh.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != m.group("sha"):
            raise ValueError(f"{name}: content sha256 {digest} != filename digest")
        mapping = parse_str_str(data)
        label = m.group("label")
        records.append({
            "label": label,
            "rows": len(mapping),
            "sha256": digest,
            "digest_file": name,
            "is_final": label in FINAL_LABELS,
            "mapping": mapping,
        })
    if not records:
        raise ValueError(f"no digest snapshot files found under {snapshot_dir}")
    return records


def build_candidate_ledger(records):
    """Merge snapshot records into a lossless candidate ledger + a summary. Returns
    ``(ledger_df, manifest)``, one row per distinct (bmrb_id, accession, mapping_source);
    conflicts (a BMRB with >1 distinct accession across ALL sources) are counted and
    listed in the summary.
    """
    seen = set()
    rows = []
    per_bmrb_acc = {}
    for rec in records:
        for bmrb, acc in rec["mapping"].items():
            acc = acc.strip()
            if not acc:
                continue
            key = (bmrb, acc, rec["label"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "bmrb_id": bmrb,
                "accession": acc,
                "mapping_source": rec["label"],
                "source_final": rec["is_final"],
                "source_digest_file": rec["digest_file"],
                "source_sha256": rec["sha256"],
            })
            per_bmrb_acc.setdefault(bmrb, set()).add(acc)

    rows.sort(key=lambda r: (r["bmrb_id"], r["accession"], r["mapping_source"]))
    ledger = pd.DataFrame(rows, columns=list(LEDGER_COLUMNS))

    conflicts = {b: sorted(a) for b, a in per_bmrb_acc.items() if len(a) > 1}
    manifest = {
        "n_candidate_rows": len(rows),
        "n_bmrb_ids": len(per_bmrb_acc),
        "n_conflicting_bmrb_ids": len(conflicts),
        "conflicting_bmrb_ids": conflicts,
    }
    return ledger, manifest


def consolidate(snapshot_dir, out_ledger, out_manifest):
    records = load_snapshot_files(snapshot_dir)
    ledger, manifest = build_candidate_ledger(records)

    os.makedirs(os.path.dirname(os.path.abspath(out_ledger)) or ".", exist_ok=True)
    with open(out_ledger, "wb") as fh:
        fh.write(ledger.to_csv(index=False).encode("utf-8"))
    with open(out_manifest, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot-dir", required=True,
                    help="pinned dir of immutable digest snapshot files")
    ap.add_argument("--out-ledger", required=True,
                    help="output candidate-ledger CSV")
    ap.add_argument("--out-manifest", required=True,
                    help="output summary JSON (counts and conflicts)")
    args = ap.parse_args(argv)
    manifest = consolidate(args.snapshot_dir, args.out_ledger, args.out_manifest)
    print(json.dumps({k: manifest[k] for k in (
        "n_candidate_rows", "n_bmrb_ids", "n_conflicting_bmrb_ids")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
