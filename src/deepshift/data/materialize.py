"""Materialization stage: realize discovery's per-source selection
tables into row-level ``structure_data_{af,nmr,xtal}.csv`` + build logs.

The CLI wrapper the entity build wires between ``discover`` and ``cluster``/``folds``: it
gives ``datasets.materialize_masters`` a real ``structure_loader`` over the seeded
AF/crystal/NMR coordinate dirs, then adds the ``residue_id`` alias plus the ``sample_id``
and ``materialization_id`` identities the label-set split and cache stages key on. It
walks discovery's ranked, source-scoped candidates and finalizes the first past every
gate.
"""
from __future__ import annotations

import argparse
import hashlib
import os

import pandas as pd

from .. import domain as _id
from . import structures as _S
from .alignment import align_sequences
from .crystal import crystal_pdb_path
from .datasets import materialize_masters


def _sha256_file(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _locate(source: str, accession: str, pdb_dir: str, af_dir: str,
            crystal_dir: str, af_version: str):
    """``(path, preprocessing_state)`` for one selected structure.

    Model index is resolved later from the conformer actually used, not fixed here."""
    if source == "af":
        p = os.path.join(af_dir, f"AF-{accession}-F1-model_{af_version}.pdb")
        return (p, "af_" + af_version) if os.path.isfile(p) else (None, "")
    if source == "xtal":
        p = crystal_pdb_path(crystal_dir, accession)
        return (p, "xray") if os.path.isfile(p) else (None, "")
    # nmr: pdbs/<ID>.pdb, case-insensitive (archive stores upper-case ids).
    for name in (f"{accession}.pdb", f"{accession.upper()}.pdb",
                 f"{accession.lower()}.pdb"):
        p = os.path.join(pdb_dir, name)
        if os.path.isfile(p):
            return p, "nmr_repr"
    return None, ""


def _score_chain(chain_seq, entity_seq):
    """(identity, n_aligned) of one chain against the entity, or None on failure."""
    aln = align_sequences(chain_seq, entity_seq)
    if aln is None:
        return None
    stats = _S._alignment_stats(aln)
    return stats["identity"], stats["n_aligned"]


def _pick_chain(chain_seqs, entity_seq):
    """Score every chain against the entity sequence and return the chain with the
    highest (identity, aligned length). ``chain_seqs`` is ``{chain: (sequence, res_ids)}``."""
    scored = []
    for ch, (seq, rids) in chain_seqs.items():
        sc = _score_chain(seq, entity_seq)
        if sc is not None:
            scored.append((sc[0], sc[1], ch, seq, rids))
    if not scored:
        return None
    best = max(scored, key=lambda x: (x[0], x[1], x[2]))
    return best[2], best[3], best[4]


def make_structure_loader(pdb_dir, af_dir, crystal_dir, af_version="v6"):
    """Return ``loader(source, accession, chain, entity_seq=None) -> dict | None``.

    Parses one coordinate file with insertion codes retained, reports the
    representative NMR model index, and scores every chain against the
    entity to pick it. Returns residue ids (integer author numbers, or
    ``(seq, icode)`` for insertion-coded ones so the materializer fails closed instead of
    silently dropping 10A), one-letter sequence, coordinate hash, model index, and
    preprocessing state.
    """

    def loader(source, accession, chain, entity_seq=None):
        if not accession:
            return None
        path, preproc = _locate(
            source, accession, pdb_dir, af_dir, crystal_dir, af_version)
        if not path:
            return None
        try:
            models = _S.parse_all_models_with_insertion_codes(path, chain_id=None)
        except Exception:
            return None
        if not models:
            return None
        # Chain choice uses model 0 (sequence identity is ensemble-invariant); the
        # representative model index is then resolved over that chain so downstream stages agree.
        chain_seqs0 = _S.chain_sequences_with_insertion_codes(models[0])
        picked = _pick_chain(chain_seqs0, entity_seq)
        if picked is None:
            return None
        used_chain, _seq0, _rids0 = picked

        model_index = 0
        if source == "nmr" and len(models) > 1:
            try:
                model_index = _S.select_best_nmr_model_index(path, used_chain)
            except Exception:
                model_index = 0
        rep_chain_seqs = _S.chain_sequences_with_insertion_codes(models[model_index])
        if used_chain not in rep_chain_seqs:
            return None
        seq, rids = rep_chain_seqs[used_chain]
        return {
            "struct_res_ids": rids,
            "struct_seq": seq,
            "coord_sha256": _sha256_file(path),
            "model_index": model_index,
            "preprocessing_state": preproc,
            "chain": used_chain,
        }

    return loader


def _augment_master(path: str, build_id: str) -> int:
    """Add residue_id + sample_id + materialization_id to one master CSV in place.

    residue_id aliases comp_index_id for the cache; sample_id and materialization_id are
    the canonical identities. Returns the row count (0 for an empty master)."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return 0
    try:
        df = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        # A zero-structure source writes a header-less master; treat as empty, don't crash.
        return 0
    if df.empty:
        return 0
    df["residue_id"] = df["comp_index_id"]
    sa = [_id.sample_id(le, src)
          for le, src in zip(df["logical_entity_id"], df["source"])]
    df["sample_id"] = sa
    df["materialization_id"] = [
        _id.materialization_id(s, ls, si, build_id)
        for s, ls, si in zip(sa, df["label_set_id"], df["structure_instance_id"])]
    df.to_csv(path, index=False)
    return len(df)


_SOURCE_DATASET = {"af": "alphafold", "nmr": "experimental", "xtal": "crystal"}


def write_atoms_build_log(master_path, modeled_log_path, source, out_path,
                          pdb_dir, af_dir, crystal_dir, af_version="v6"):
    """Emit a cache/atoms build log from a materialized master.

    The log is entity-grained: one row per ``materialization_id`` (entity x source x
    structure), not per (bmrb, source), so a multi-entity deposition sharing one source
    resolves each entity's own structure. ``bmrb_id`` is carried alongside; the atom stage
    keys on ``materialization_id`` when present, and two materializations sharing one structure carry
    the same pdb_path/chain. ``modeled_log_path`` is the authoritative per-label-set
    realization log from ``materialize_masters``, joined in here and never derived from
    sparse master rows, so unlabeled modeled regions aren't silently compressed and
    leakage checks stay intact.
    """
    dataset = _SOURCE_DATASET[source]
    cols = ["dataset", "materialization_id", "sample_id", "structure_instance_id",
            "label_set_id", "bmrb_id", "pdb_path", "chain_id", "status",
            "source", "accession", "n_modeled_positions", "modeled_positions"]
    rows = []
    if os.path.isfile(master_path) and os.path.getsize(master_path) > 0:
        try:
            df = pd.read_csv(master_path, dtype=str)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()   # a source with zero materialized structures
        if not df.empty:
            if not os.path.isfile(modeled_log_path):
                raise FileNotFoundError(
                    f"modeled-position build log not found: {modeled_log_path}")
            modeled_df = pd.read_csv(modeled_log_path, dtype=str).fillna("")
            required = {"label_set_id", "structure_instance_id",
                        "modeled_positions", "n_modeled_positions", "status"}
            missing = required - set(modeled_df.columns)
            if missing:
                raise ValueError(
                    f"{modeled_log_path} omits modeled-position fields: "
                    f"{sorted(missing)}")
            modeled = {}
            for _, log_row in modeled_df[modeled_df["status"] == "success"].iterrows():
                log_key = (str(log_row["label_set_id"]),
                           str(log_row["structure_instance_id"]))
                value = (str(log_row["n_modeled_positions"]),
                         str(log_row["modeled_positions"]))
                if log_key in modeled and modeled[log_key] != value:
                    raise ValueError(
                        f"conflicting modeled positions for {log_key} in "
                        f"{modeled_log_path}")
                modeled[log_key] = value
            key = "materialization_id" if "materialization_id" in df.columns \
                else "bmrb_id"
            seen = df.drop_duplicates(subset=[key])
            for _, r in seen.iterrows():
                path, _pre = _locate(source, r["accession"], pdb_dir, af_dir,
                                     crystal_dir, af_version)
                if not path:
                    continue
                log_key = (str(r.get("label_set_id", "")),
                           str(r.get("structure_instance_id", "")))
                if log_key not in modeled:
                    raise ValueError(
                        f"materialization {r.get('materialization_id', '')} has no "
                        f"successful modeled-position authority in {modeled_log_path}")
                n_modeled, modeled_positions = modeled[log_key]
                rows.append({"dataset": dataset,
                             "materialization_id": r.get("materialization_id", ""),
                             "sample_id": r.get("sample_id", ""),
                             "structure_instance_id": r.get("structure_instance_id", ""),
                             "label_set_id": r.get("label_set_id", ""),
                             "bmrb_id": r["bmrb_id"],
                             "pdb_path": path, "chain_id": r["chain"],
                             "status": "success", "source": source,
                             "accession": r["accession"],
                             "n_modeled_positions": n_modeled,
                             "modeled_positions": modeled_positions})
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shifts-file", required=True,
                    help="entity-grained selected chemical_shifts.csv (from fetch)")
    ap.add_argument("--selection-dir", required=True,
                    help="dir with discovery candidate_ledger.csv")
    ap.add_argument("--out-dir", required=True,
                    help="dir for structure_data_{source}.csv + build_log_{source}.csv")
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--alphafold-dir", required=True)
    ap.add_argument("--crystal-dir", required=True)
    ap.add_argument("--af-version", default="v6")
    ap.add_argument("--build-id", required=True)
    ap.add_argument("--min-query-coverage", type=float, default=0.90)
    ap.add_argument("--min-identity", type=float, default=0.95,
                    help="entity identity floor at realization")
    ap.add_argument("--min-aligned-length", type=int, default=10,
                    help="minimum aligned residues at realization")
    args = ap.parse_args(argv)

    loader = make_structure_loader(args.pdb_dir, args.alphafold_dir,
                                   args.crystal_dir, af_version=args.af_version)
    summary = materialize_masters(
        args.out_dir, args.shifts_file, args.selection_dir, loader,
        args.build_id, min_query_coverage=args.min_query_coverage,
        min_identity=args.min_identity,
        min_aligned_length=args.min_aligned_length)

    for source in ("af", "nmr", "xtal"):
        master = os.path.join(args.out_dir, f"structure_data_{source}.csv")
        n = _augment_master(master, args.build_id)
        summary[source]["augmented_rows"] = n
        nb = write_atoms_build_log(
            master, os.path.join(args.out_dir, f"build_log_{source}.csv"), source,
            os.path.join(args.out_dir, f"structure_build_log_{source}.csv"),
            args.pdb_dir, args.alphafold_dir, args.crystal_dir,
            af_version=args.af_version)
        summary[source]["atoms_build_log_rows"] = nb

    import json
    print("materialize: " + json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
