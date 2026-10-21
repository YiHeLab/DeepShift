"""Full-corpus census of the fixed final atom-cloud recipe.

Parses every realized structure with its sample-matched protein context and counts the
atoms each center gathers before the cap, so the number of centers the cap truncates is
known before any array is built.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from ..atom_cloud import FINAL_ATOM_CLOUD_RECIPE, census_atom_counts
from . import assembly, structures
from .atoms import choose_m_atoms
from .partner_suppression import suppression_key


def _realized_records(build_log_specs):
    """Unique ``(label_set_id, bmrb_id, pdb_path, chain)`` realized rows."""
    records = []
    entity_grained = True
    for spec in build_log_specs:
        _source, logpath, _dataset = spec.split(":")
        frame = pd.read_csv(logpath, dtype=str).fillna("")
        if "materialization_id" not in frame.columns:
            entity_grained = False
            break
        required = {"label_set_id", "bmrb_id", "pdb_path", "chain_id", "status"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{logpath} omits census fields: {sorted(missing)}")
        for _, row in frame[frame["status"] == "success"].iterrows():
            records.append((str(row["label_set_id"]), str(row["bmrb_id"]),
                            row["pdb_path"], str(row["chain_id"])))
    if entity_grained:
        return sorted(set(records))
    raise ValueError("the final atom census requires entity-grained structure build logs")


def run_census(
    build_log_specs,
    *,
    sample_assembly,
    suppressed_partner_chains,
    search_dirs=None,
    limit=None,
):
    """Return the fixed cap and the count statistics over every realized structure, with the
    sample-matched partner context the atom stage will build."""
    search_dirs = search_dirs or []
    assembly_map = (assembly.load(sample_assembly)
                    if isinstance(sample_assembly, (str, os.PathLike))
                    else dict(sample_assembly))
    if isinstance(suppressed_partner_chains, (str, os.PathLike)):
        with open(suppressed_partner_chains) as stream:
            suppress_map = json.load(stream)
    else:
        suppress_map = dict(suppressed_partner_chains)

    counts: list[int] = []
    n_ok = n_miss = 0
    missing: list[tuple[str, str, str]] = []   # (pdb_path, chain, reason)
    items = _realized_records(build_log_specs)
    if limit:
        items = items[:limit]
    for label_set_id, _bmrb_id, pdb_path, chain in tqdm(items, desc="census"):
        path = pdb_path
        if not os.path.isfile(path):
            base = os.path.basename(path)
            alt = None
            for d in search_dirs:
                if os.path.isfile(os.path.join(d, base)):
                    alt = os.path.join(d, base)
                    break
            if alt is None:
                alt = structures.resolve_pdb_path(os.path.splitext(base)[0], search_dirs)
            if alt is None:
                n_miss += 1
                missing.append((pdb_path, chain, "structure_path_missing"))
                continue
            path = alt
        try:
            aa = structures.model_to_aa(
                structures.select_best_nmr_model(path, chain), keep_nonstandard=True)
        except Exception:
            n_miss += 1
            missing.append((path, chain, "structure_unparseable"))
            continue
        if not aa:
            n_miss += 1
            missing.append((path, chain, "structure_empty"))
            continue
        insertion = structures.retained_insertion_partner_residues(path, chain)
        sample_record = assembly_map.get(str(label_set_id), {})
        cross = (structures.sample_matched_partner_residues_for(
            path, chain, sample_record, drop_hydrogens=True
        ) if assembly.partner_in_sample(assembly_map, label_set_id) else {})
        suppress = suppress_map.get(suppression_key(pdb_path, chain), ())
        counts.extend(census_atom_counts(
            aa,
            recipe=FINAL_ATOM_CLOUD_RECIPE,
            partner_residues={**insertion, **cross} or None,
            suppress_chains=suppress,
        ))
        n_ok += 1

    # Every realized structure must be seen for the census to be authoritative: one
    # missing, unparseable or empty structure could be the densest, and skipping it
    # would truncate the cap.
    if missing:
        preview = "; ".join(f"{p} (chain {c}): {r}" for p, c, r in missing[:10])
        more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
        raise RuntimeError(
            f"atom census could not resolve {len(missing)} realized structure(s) "
            f"(ok={n_ok}, missing={n_miss}); refusing to choose an atom cap over an "
            f"incomplete corpus -- a missing dense structure would truncate. Fix the "
            f"entity build logs / structure paths before proceeding (fail-closed). "
            f"First offenders: {preview}{more}")
    # An empty count set is refused rather than defaulted: if no structure yielded a CA
    # center, the census is not authoritative.
    if not counts:
        raise RuntimeError(
            f"atom census produced ZERO CA centers "
            f"(structures ok={n_ok}, missing={n_miss}); refusing to choose an "
            f"atom cap from an empty count set. Check the entity build logs / "
            f"structure paths before proceeding (fail-closed).")
    observed_max = int(max(counts))
    m_atoms = choose_m_atoms(counts)
    stats = {
        "n_structures_ok": n_ok, "n_structures_missing": n_miss,
        "n_centers": len(counts),
        "precap_max": observed_max,
        "precap_mean": (sum(counts) / len(counts)) if counts else 0.0,
        "chosen_m_atoms": int(m_atoms),
        "n_centers_over_cap": int(sum(count > m_atoms for count in counts)),
        "partner_policy": FINAL_ATOM_CLOUD_RECIPE.partner_policy,
        "drop_hydrogens": FINAL_ATOM_CLOUD_RECIPE.drop_hydrogens,
    }
    return m_atoms, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source_build_logs", nargs="+", required=True,
                    help="one or more 'source:logpath:dataset' specs")
    ap.add_argument("--pdb_search_dirs", nargs="+", default=[])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample_assembly", required=True)
    ap.add_argument("--leaky_partner_chains", required=True)
    ap.add_argument("--out", required=True, help="atom_census.json output path")
    args = ap.parse_args(argv)

    _m_atoms, stats = run_census(
        args.source_build_logs,
        sample_assembly=args.sample_assembly,
        suppressed_partner_chains=args.leaky_partner_chains,
        search_dirs=args.pdb_search_dirs,
        limit=args.limit,
    )
    tmp = args.out + f".tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(stats, fh, indent=2)
    os.replace(tmp, args.out)
    print("atom_census: " + json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()
