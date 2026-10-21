"""Structure parsing, chain sequences, NMR model selection and sample-matched partners.

The integer atom cache addresses residues by author number: `to_int_resid`/
`project_model_to_chain_seq` keep only the lowest insertion code per number and drop
the siblings, so a dropped standard sibling loses its heavy atoms from the 8 A
neighbour cloud unless retained as a `(chain, seq, icode)` partner.
"""

import os
import re
from collections import defaultdict

import numpy as np

from ..domain import AA_3_TO_1
from ..structure import (
    canonical_residues,
    parse_models,
    partner_protein_residues,
    project_model_to_chain_seq,
    representative_model_index,
    to_int_resid,
)
from .alignment import align_sequences


def _canonical_residue_one_letter(key, info):
    """One-letter code after the atom-stage's canonical residue mapping."""
    canonical = canonical_residues(
        {key: info}, keep_nonstandard=True, drop_hydrogens=False)
    if not canonical:
        return None
    residue = next(iter(canonical.values()))
    return AA_3_TO_1.get(
        residue["residue_name"], "X" if residue["is_standard"] else None)


def _extract_chain_sequence(residues, chain_id):
    """(sequence_str, res_ids, {res_id: CA coord}) for one chain of a parse_pdb residues dict, or (None, None, None) if empty."""
    items = sorted(
        ((rid, rdata) for (ch, rid), rdata in residues.items() if ch == chain_id),
        key=lambda x: x[0],
    )
    if not items:
        return None, None, None

    seq_chars = []
    res_ids = []
    ca_coords = {}

    for rid, rdata in items:
        aa = _canonical_residue_one_letter((chain_id, rid, ""), rdata)
        if aa is None:
            continue
        seq_chars.append(aa)
        res_ids.append(rid)
        ca = rdata['atoms'].get('CA')
        if ca is not None and np.all(np.isfinite(ca)):
            ca_coords[rid] = ca

    if not seq_chars:
        return None, None, None

    return ''.join(seq_chars), res_ids, ca_coords


SAMPLE_PARTNER_MIN_IDENTITY = 0.95
SAMPLE_PARTNER_MIN_QUERY_COVERAGE = 0.90
SAMPLE_PARTNER_MIN_ALIGNED_LENGTH = 10


def _alignment_stats(alignment):
    """Identity + coverage contract for one alignment (alignment[0] the structure chain
    /target, alignment[1] the shift assignment/query). Returns dict: n_match, n_aligned,
    identity, target_len, query_len, target_coverage, query_coverage."""
    target, query = str(alignment[0]), str(alignment[1])
    n_match = n_aligned = target_len = query_len = 0
    for ct, cq in zip(target, query):
        if ct != '-':
            target_len += 1
        if cq != '-':
            query_len += 1
        if ct != '-' and cq != '-':
            n_aligned += 1
            if ct == cq:
                n_match += 1
    return {
        'n_match': n_match, 'n_aligned': n_aligned,
        'target_len': target_len, 'query_len': query_len,
        'identity': (n_match / n_aligned) if n_aligned else 0.0,
        'target_coverage': (n_aligned / target_len) if target_len else 0.0,
        'query_coverage': (n_aligned / query_len) if query_len else 0.0,
    }


def label_residue_mapping(alignment, struct_res_ids, label_comp_index_ids):
    """Map an aligned structure chain (alignment[0]) onto label residues (alignment[1],
    same order as label_comp_index_ids). One record per label residue:
    {'comp_index_id', 'struct_res_id' (or None), 'mismatch_type' in 'match'/'mismatch'/
    'gap_in_structure'}; structure-only positions (gap_in_cs) are not emitted."""
    a_struct, a_label = str(alignment[0]), str(alignment[1])
    i_s = i_l = 0
    out = []
    for cs, cl in zip(a_struct, a_label):
        if cs != '-' and cl != '-':
            out.append({
                'comp_index_id': label_comp_index_ids[i_l],
                'struct_res_id': struct_res_ids[i_s],
                'mismatch_type': 'match' if cs == cl else 'mismatch'})
        elif cs == '-' and cl != '-':
            out.append({
                'comp_index_id': label_comp_index_ids[i_l],
                'struct_res_id': None,
                'mismatch_type': 'gap_in_structure'})
        if cs != '-':
            i_s += 1
        if cl != '-':
            i_l += 1
    return out


def select_best_nmr_model_index(pdb_path, chain_id):
    """Index of the representative NMR model for chain_id (single model = 0), computed over
    the insertion-preserving ensemble so the same conformer is checked and built."""
    return representative_model_index(
        parse_all_models_with_insertion_codes(pdb_path, chain_id=chain_id))


def parse_all_models_with_insertion_codes(pdb_path, chain_id=None):
    """All MODELs keyed by (chain, seq, icode), insertion codes retained (unlike
    project_model_to_chain_seq, which projects to (chain, seq) and drops 10A);
    chain_id=None keeps every chain."""
    return parse_models(pdb_path, chain=chain_id)


def partner_protein_residues_for(pdb_path, target_chain, *, drop_hydrogens=False):
    """Cross-chain standard partner-protein residues for the target's representative model,
    keyed by (chain, seq, icode): the representative-model index is computed on
    the target chain alone, then the partner-inclusive parse is indexed at that model so
    coords share it (HETATM/solvent/ligand/nucleic-acid dropped, MSE->MET); {} for a
    single-chain structure (bit-identical clouds)."""
    mi = select_best_nmr_model_index(pdb_path, target_chain)
    models_pc = parse_models(pdb_path, chain=target_chain, keep_partner_chains=True)
    if not models_pc or mi >= len(models_pc):
        return {}
    return partner_protein_residues(
        models_pc[mi], target_chain, keep_nonstandard=False, drop_hydrogens=drop_hydrogens)


def sample_matched_partner_residues_for(
    pdb_path,
    target_chain,
    sample_record,
    *,
    drop_hydrogens=False,
    min_identity=SAMPLE_PARTNER_MIN_IDENTITY,
    min_query_coverage=SAMPLE_PARTNER_MIN_QUERY_COVERAGE,
    min_aligned_length=SAMPLE_PARTNER_MIN_ALIGNED_LENGTH,
):
    """Cross-chain residues whose chain consumes a declared NMR-sample protein slot.

    Each slot is one remaining copy of a deposited protein entity after the target
    molecule has been removed.  A coordinate chain may consume at most one slot, and a
    slot may be consumed no more often than its deposited copy count.  Extra lattice
    chains, nonmatching chains, and incomplete/unknown sample records are excluded.
    """
    if not isinstance(sample_record, dict) or not sample_record.get(
            "target_present_in_assembly"):
        return {}
    capacities = {
        str(index): int(member.get("copies", 0))
        for index, member in enumerate(sample_record.get("partners", ()))
        if int(member.get("copies", 0)) > 0 and member.get("alignment_sequence")
    }
    if not capacities:
        return {}

    model_index = select_best_nmr_model_index(pdb_path, target_chain)
    models = parse_models(pdb_path, chain=target_chain, keep_partner_chains=True)
    if not models or model_index >= len(models):
        return {}
    model = models[model_index]
    chain_sequences = chain_sequences_with_insertion_codes(model)
    candidates = []
    for chain, (structure_sequence, _residue_ids) in chain_sequences.items():
        if str(chain) == str(target_chain):
            continue
        for index, member in enumerate(sample_record.get("partners", ())):
            slot = str(index)
            if slot not in capacities:
                continue
            deposited_sequence = str(member["alignment_sequence"])
            alignment = align_sequences(structure_sequence, deposited_sequence)
            if alignment is None:
                continue
            stats = _alignment_stats(alignment)
            if (stats["identity"] < min_identity
                    or stats["query_coverage"] < min_query_coverage
                    or stats["n_aligned"] < min_aligned_length):
                continue
            candidates.append((
                -stats["identity"], -stats["query_coverage"],
                -stats["n_aligned"], str(chain), slot,
            ))

    # Maximum-cardinality bipartite matching, with one node per declared copy. A greedy
    # best-hit pass can discard a valid partner: one chain may fit two sample entities
    # while a second chain fits only one, requiring the first chain to take its alternate.
    edges = defaultdict(list)
    best_score = {}
    for neg_identity, neg_coverage, neg_aligned, chain, slot in sorted(candidates):
        score = (neg_identity, neg_coverage, neg_aligned, slot)
        best_score[chain] = min(best_score.get(chain, score), score)
        for copy_index in range(capacities[slot]):
            edges[chain].append((score, (slot, copy_index)))
    occupied = {}

    def place(chain, seen):
        for _score, copy_slot in sorted(edges[chain]):
            if copy_slot in seen:
                continue
            seen.add(copy_slot)
            incumbent = occupied.get(copy_slot)
            if incumbent is None or place(incumbent, seen):
                occupied[copy_slot] = chain
                return True
        return False

    for chain in sorted(edges, key=lambda value: (best_score[value], value)):
        place(chain, set())
    selected_chains = set(occupied.values())

    partners = partner_protein_residues(
        model, target_chain, keep_nonstandard=False, drop_hydrogens=drop_hydrogens
    )
    return {key: residue for key, residue in partners.items()
            if str(key[0]) in selected_chains}


def chain_sequences_with_insertion_codes(model):
    """Per-chain (sequence, res_id list) from an insertion-code-keyed model, keeping
    every insertion sibling (unlike the integer projection) so the alignment never
    frame-shifts. Returns {chain: (sequence_str, [res_id or (seq, icode), ...])}."""
    by_chain = {}
    for key, info in model.items():
        ch, seq, icode = key[0], int(key[1]), (key[2] if len(key) >= 3 else "")
        by_chain.setdefault(ch, []).append((seq, icode, info))
    out = {}
    for ch, items in by_chain.items():
        items.sort(key=lambda x: (x[0], x[1]))
        seq_chars, res_ids = [], []
        for seq, icode, info in items:
            aa = _canonical_residue_one_letter((ch, seq, icode), info)
            if aa is None:
                continue
            seq_chars.append(aa)
            res_ids.append(seq if icode == "" else (seq, icode))
        if seq_chars:
            out[ch] = ("".join(seq_chars), res_ids)
    return out


def dropped_standard_insertion_residues(model_with_icodes):
    """Sorted (seq, icode) keys of standard residues the integer projection would
    drop from model_with_icodes (a single model, insertion codes retained)."""
    canon = canonical_residues(model_with_icodes, keep_nonstandard=True)
    by_seq: dict[int, list[str]] = {}
    for seq, icode in canon:
        by_seq.setdefault(int(seq), []).append(icode)
    dropped = []
    for seq, icodes in by_seq.items():
        kept = sorted(icodes)[0]           # lowest icode ('' < 'A'), as to_int_resid keeps
        for icode in icodes:
            if icode != kept and canon[(seq, icode)].get("is_standard", True):
                dropped.append((seq, icode))
    return sorted(dropped)


def retained_insertion_partner_residues(pdb_path, chain_id):
    """Dropped standard insertion siblings for pdb_path/chain_id, keyed by a (chain, seq,
    icode) tuple for use as a build_atom_cloud partner_residues entry; {} when the
    structure is integer-representable."""
    models = parse_all_models_with_insertion_codes(pdb_path, chain_id=chain_id)
    if not models:
        return {}
    model = models[representative_model_index(models)]
    dropped = set(dropped_standard_insertion_residues(model))
    if not dropped:
        return {}
    canonical = canonical_residues(model, keep_nonstandard=True, drop_hydrogens=False)
    partners = {}
    for (seq, icode), info in canonical.items():
        if (seq, icode) in dropped:
            chain = info.get("auth_chain_id", chain_id) or chain_id
            partners[(str(chain), int(seq), str(icode))] = info
    return partners


_INSERTION_RID_RE = re.compile(r"^\s*(-?\d+)\s*([A-Za-z])\s*$")


def parse_struct_res_id(rid):
    """Classify a stored struct_res_id binding: serialized either as an author string
    like "10A" or as a [seq, icode]/(seq, icode) pair (empty icode = plain integer). Returns
    ("int", author_number), ("insertion", (seq, icode)) for its own center, or None."""
    if rid is None or isinstance(rid, bool):
        return None
    if isinstance(rid, (list, tuple)):
        if len(rid) == 1:
            return parse_struct_res_id(rid[0])
        if len(rid) == 2:
            try:
                icode = str(rid[1]).strip()
                return ("insertion", (int(rid[0]), icode)) if icode else ("int", int(rid[0]))
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(rid, int):
        return ("int", int(rid))
    if isinstance(rid, float):
        return None if rid != rid else ("int", int(rid))   # NaN -> unbound
    text = str(rid).strip()
    if text in ("", "None", "nan", "NaN"):
        return None
    try:
        return ("int", int(text))
    except ValueError:
        pass
    match = _INSERTION_RID_RE.match(text)
    if match:
        return ("insertion", (int(match.group(1)), match.group(2)))
    return None


def select_best_nmr_model(pdb_path, chain_id):
    """NMR model closest to the median structure: superimpose all models onto Model 1
    (Kabsch, shared CA), then pick the lowest RMSD to the median CA; single-model files
    return that model directly (shared by dataset and inference so ensembles resolve to
    one conformer). Returns the residues dict {(chain, res_id): {...}}, or None."""
    models = parse_all_models_with_insertion_codes(pdb_path, chain_id=chain_id)
    if not models:
        return None
    selected = models[representative_model_index(models)]
    return project_model_to_chain_seq(selected)


def model_to_aa(model, keep_nonstandard=False):
    """Canonicalise a chain-filtered model (MSE->MET incl. SE->SD) into the integer-res_id
    dict the atom cache is built on. With keep_nonstandard=True, nonstandard residues are
    kept as context tagged is_standard=False."""
    return to_int_resid(canonical_residues(
        model, keep_nonstandard=keep_nonstandard, drop_hydrogens=False))


def resolve_pdb_path(pdb_id, search_dirs, chain_id=None, bmrb_id=None):
    """Find pdb_id across search_dirs, trying case and chain/bmrb-suffixed naming
    conventions; None if not found."""
    pdb_upper = pdb_id.upper()
    pdb_lower = pdb_id.lower()

    patterns = [
        f'{pdb_upper}.pdb',
        f'{pdb_lower}.pdb',
    ]
    if chain_id:
        patterns.extend([
            f'{pdb_upper}{chain_id}.pdb',
            f'{pdb_lower}{chain_id}.pdb',
        ])
    if bmrb_id:
        patterns.append(f'rcsb_{bmrb_id}.pdb')

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in patterns:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path

    return None
