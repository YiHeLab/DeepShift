"""Chemistry vocabulary and the entity-aware identity authority.

The dependency-free leaf of the package: it fixes the frozen vocabulary the model and
cache are built on (64 nuclei, 21 residues, an 88-atom alphabet) and mints every
scientific identifier. Vocabulary order is baked into trained checkpoints and cached
arrays; the identities define the leakage-join keys and the source-invariant label
identity.

Identity rule: canonical JSON (UTF-8, NFC strings, sorted keys, compact separators)
hashed with lowercase SHA-256. Numbering is left out of the polymer payload, so
renumbering is equal while a modified residue keeps its own token.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata

# Frozen chemistry vocabulary. Order is load-bearing: checkpoint embeddings and cached
# uint8 arrays index it.

STANDARD_RESIDUES = (
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR", "UNK",
)
RESIDUE_TO_IDX = {name: i for i, name in enumerate(STANDARD_RESIDUES)}

AA_3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# The 88-token atom alphabet; ATOM_TO_IDX values are the embedding indices baked into
# every checkpoint and the cache's uint8 atom_type array.
ATOM_TYPES = (
    "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3", "CG", "CG1",
    "CG2", "CH2", "CZ", "CZ2", "CZ3", "DD1", "H", "H1", "H2", "H3", "HA", "HA2",
    "HA3", "HB", "HB1", "HB2", "HB3", "HC", "HD1", "HD11", "HD12", "HD13", "HD2",
    "HD21", "HD22", "HD23", "HD3", "HE", "HE1", "HE2", "HE21", "HE22", "HE3",
    "HG", "HG1", "HG11", "HG12", "HG13", "HG2", "HG21", "HG22", "HG23", "HG3",
    "HH", "HH11", "HH12", "HH2", "HH21", "HH22", "HN1", "HN2", "HXT", "HZ", "HZ1",
    "HZ2", "HZ3", "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "O",
    "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "OXT", "SD", "SG",
)
ATOM_TO_IDX = {name: i for i, name in enumerate(ATOM_TYPES)}


# The 64 predicted nuclei; SHIFT_COLUMNS is the cache label-column and model-head
# order, exactly `sorted()` on the derived column names, so it cannot drift from the
# cache builder's data-driven parse. The 64 include eight Trp indole nuclei, five Arg
# guanidinium nuclei, Thr HG1 and Tyr HH.
#
# Two of those carry pathologies a reader must not mistake for structure:
#   * Arg NE aliases: a deposit folded by one 15N sweep width sits ~31 ppm off
#     (84.6 -> ~115.6 ppm, under the backbone N density), so observed values outside
#     78.976--90.212 ppm are excluded by quality_control.
#   * Thr HG1 and Tyr HH are labile hydroxyl protons; their BMRB sigmas (1.34 and
#     1.57 ppm) are exchange-dominated, not headroom the model can explain.
NUCLEI = (
    "C", "CA", "CB", "CD1", "CD2", "CD", "CE1", "CE2", "CE3", "CE", "CG1", "CG2",
    "CG", "CH2", "CZ2", "CZ3", "CZ", "H", "HA2", "HA3", "HA", "HB1", "HB2", "HB3",
    "HB", "HD11", "HD12", "HD13", "HD1", "HD21", "HD22", "HD23", "HD2", "HD3",
    "HE1", "HE21", "HE22", "HE2", "HE3", "HE", "HG11", "HG12", "HG13", "HG1",
    "HG21", "HG22", "HG23", "HG2", "HG3", "HG", "HH11", "HH12", "HH21", "HH22",
    "HH2", "HH", "HZ2", "HZ3", "HZ", "N", "ND2", "NE1", "NE2", "NE",
)
SHIFT_COLUMNS = tuple(f"{name.lower()}_shift" for name in NUCLEI)

# Every atom name a chemical-shift predictor can label on a standard residue: the 64
# nuclei predicted here, plus six that other predictors report and this one does not --
# the HN alias several tools write for the amide proton, the lysine HZ1 and NZ, the
# histidine ND1, and the two arginine guanidinium nitrogens. A predictor's output is
# read against this vocabulary, so an arm that also labels ligand, solvent or
# nucleic-acid atoms is credited with what it predicts for a protein and nothing else.
SHIFT_ATOM_NAMES = frozenset(NUCLEI) | frozenset(
    {"HN", "HZ1", "ND1", "NH1", "NH2", "NZ"})

# The three structure sources, with no umbrella "experimental" category.
VALID_SOURCES = ("af", "nmr", "xtal")

# Canonical serialization + hashing: the single rule under every identity. Schema
# strings are pinned, so a payload's hash is stable across the codebase.
_SCHEMA_POLYMER_TOKENS = "deepshift.polymer-token-sequence.v1"
_SCHEMA_ALIGNMENT_SEQ = "deepshift.alignment-sequence.v1"
_SCHEMA_LABEL_SET = "deepshift.label-set.v2"
_SCHEMA_LOGICAL_ENTITY = "deepshift.logical-entity.v1"
_SCHEMA_LABEL_RESIDUE = "deepshift.label-residue.v1"
_SCHEMA_STRUCTURE_INSTANCE = "deepshift.structure-instance.v1"
_SCHEMA_SAMPLE = "deepshift.sample.v1"
_SCHEMA_MATERIALIZATION = "deepshift.materialization.v1"


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _nfc_deep(obj):
    """NFC-normalize every string in a JSON-ish structure (keys and values)."""
    if isinstance(obj, str):
        return _nfc(obj)
    if isinstance(obj, dict):
        return {_nfc(k) if isinstance(k, str) else k: _nfc_deep(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nfc_deep(v) for v in obj]
    return obj


def canonical_bytes(obj) -> bytes:
    """Canonical JSON bytes: NFC strings, sorted keys, compact separators, UTF-8."""
    return json.dumps(_nfc_deep(obj), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_hex(obj) -> str:
    """Lowercase SHA-256 of raw bytes, or of the canonical JSON bytes of any object."""
    data = obj if isinstance(obj, (bytes, bytearray)) else canonical_bytes(obj)
    return hashlib.sha256(data).hexdigest()


def is_sha256(value) -> bool:
    """True iff value is 64 lowercase hex characters."""
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


# Input normalizers: two spellings of one id must hash identically.

def normalize_star_id(value) -> str:
    """A STAR id in canonical form: decimal ids lose leading zeros ("007"->"7");
    non-decimal ids pass through NFC-normalized and stripped. Raises on empty/None."""
    if value is None:
        raise ValueError("STAR id is None; resolve it before minting identity")
    if isinstance(value, bool):
        raise TypeError("STAR id must not be a bool")
    if isinstance(value, int):
        return str(value)
    text = _nfc(str(value)).strip()
    if not text:
        raise ValueError("STAR id is empty; resolve it before minting identity")
    return str(int(text)) if text.lstrip("+-").isdigit() else text


def normalize_bmrb_id(value) -> str:
    """BMRB id as a decimal string."""
    text = normalize_star_id(value)
    if not text.lstrip("+-").isdigit():
        raise ValueError(f"bmrb_id must be decimal; got {value!r}")
    return text


def validate_source(source) -> str:
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}; got {source!r}")
    return source


def _require_id(value, prefix) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"expected an id with prefix {prefix!r}; got {value!r}")
    if not is_sha256(value[len(prefix):]):
        raise ValueError(f"id {value!r} does not carry a valid sha256 body")
    return value


# The entity token sequence is hashed exactly (a modified residue keeps its own token).
# The alignment sequence used for PDB/AF alignment and MMseqs clustering is a one-letter
# projection through this pinned, self-checksummed map (nonstandard -> X), so an
# alignment-sequence hash is only equal across entities resolved the same way.
ALIGNMENT_PARENT_MAP_VERSION = "deepshift.alignment-parent-map.v1"
ALIGNMENT_PARENT_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
ALIGNMENT_PARENT_MAP_SHA256 = sha256_hex({
    "schema": "deepshift.alignment-parent-map",
    "version": ALIGNMENT_PARENT_MAP_VERSION,
    "map": ALIGNMENT_PARENT_MAP,
})


def exact_polymer_sequence_sha256(tokens) -> str:
    """Hash the ordered, uppercased NMR-STAR monomer tokens of a full polymer, without
    numbering. An empty or blank token list is rejected."""
    toks = [_nfc(str(t)).upper() for t in tokens]
    if not toks or any(t == "" for t in toks):
        raise ValueError("exact polymer sequence needs non-empty ordered tokens")
    return sha256_hex({"schema": _SCHEMA_POLYMER_TOKENS, "tokens": toks})


def alignment_sequence_from_tokens(tokens) -> str:
    """One-letter alignment sequence: project each full-polymer token through the
    pinned parent map ('X' for anything outside the standard 20)."""
    return "".join(ALIGNMENT_PARENT_MAP.get(_nfc(str(t)).upper().strip(), "X")
                   for t in tokens)


def alignment_sequence_sha256(alignment_sequence: str) -> str:
    """Hash a one-letter alignment sequence, bound to the pinned parent map (its
    version + checksum are in the payload, so the hash is map-specific)."""
    seq = _nfc(str(alignment_sequence)).upper()
    if not seq:
        raise ValueError("alignment sequence is empty")
    return sha256_hex({
        "schema": _SCHEMA_ALIGNMENT_SEQ,
        "sequence": seq,
        "parent_map_version": ALIGNMENT_PARENT_MAP_VERSION,
        "parent_map_sha256": ALIGNMENT_PARENT_MAP_SHA256,
    })


# The six entity-aware identity minters: the sole identity authority.

def label_set_id(bmrb_id, assigned_shift_list_id, entity_assembly_id,
                 entity_id, exact_seq_sha256, sample_condition_id=None) -> str:
    """ls_ + sha256 of one indivisible experimental label set.

    Sample condition participates: two assigned-shift loops matching on list, assembly,
    entity and sequence but differing in conditions are distinct measurements.
    """
    if not is_sha256(exact_seq_sha256):
        raise ValueError("exact_polymer_sequence_sha256 must be 64 lowercase hex")
    return "ls_" + sha256_hex({
        "schema": _SCHEMA_LABEL_SET,
        "bmrb_id": normalize_bmrb_id(bmrb_id),
        "assigned_shift_list_id": normalize_star_id(assigned_shift_list_id),
        "entity_assembly_id": normalize_star_id(entity_assembly_id),
        "entity_id": normalize_star_id(entity_id),
        "exact_polymer_sequence_sha256": exact_seq_sha256,
        "sample_condition_id": (
            None if sample_condition_id is None
            else normalize_star_id(sample_condition_id)
        ),
    })


def logical_entity_id(bmrb_id, entity_id, exact_seq_sha256) -> str:
    """le_ + sha256(bmrb, entity, seq). Picks one representative label set; the chosen
    assembly/list stay in label_set_id, not erased."""
    if not is_sha256(exact_seq_sha256):
        raise ValueError("exact_polymer_sequence_sha256 must be 64 lowercase hex")
    return "le_" + sha256_hex({
        "schema": _SCHEMA_LOGICAL_ENTITY,
        "bmrb_id": normalize_bmrb_id(bmrb_id),
        "entity_id": normalize_star_id(entity_id),
        "exact_polymer_sequence_sha256": exact_seq_sha256,
    })


def label_residue_id(label_set_id_value, comp_index_id, comp_id) -> str:
    """lr_ + sha256(label_set_id, comp_index_id, comp_id). Source-invariant, so a
    perturber exclusion propagates across an entity's AF/NMR/xtal realizations."""
    _require_id(label_set_id_value, "ls_")
    return "lr_" + sha256_hex({
        "schema": _SCHEMA_LABEL_RESIDUE,
        "label_set_id": label_set_id_value,
        "comp_index_id": int(comp_index_id),
        "comp_id": _nfc(str(comp_id)).upper(),
    })


def structure_instance_id(source, accession, auth_chain_id, model_index,
                          coordinate_file_sha256, preprocessing_state) -> str:
    """si_ + sha256(source, accession, chain, model, coord_file_sha, preproc)."""
    validate_source(source)
    if not is_sha256(coordinate_file_sha256):
        raise ValueError("coordinate_file_sha256 must be 64 lowercase hex chars")
    return "si_" + sha256_hex({
        "schema": _SCHEMA_STRUCTURE_INSTANCE,
        "source": source,
        "accession": _nfc(str(accession)),
        "auth_chain_id": _nfc(str(auth_chain_id)),
        "model_index": int(model_index),
        "coordinate_file_sha256": coordinate_file_sha256,
        "preprocessing_state": _nfc(str(preprocessing_state)),
    })


def sample_id(logical_entity_id_value, source) -> str:
    """sa_ + sha256(logical_entity_id, source). The entity/source grain."""
    validate_source(source)
    _require_id(logical_entity_id_value, "le_")
    return "sa_" + sha256_hex({
        "schema": _SCHEMA_SAMPLE,
        "logical_entity_id": logical_entity_id_value,
        "source": source,
    })


def materialization_id(sample_id_value, label_set_id_value,
                       structure_instance_id_value, build_id) -> str:
    """mz_ + sha256(sample_id, label_set_id, structure_instance_id, build_id). A
    coordinate/preprocessing change makes a new materialization without redefining
    the scientific sample."""
    _require_id(sample_id_value, "sa_")
    _require_id(label_set_id_value, "ls_")
    _require_id(structure_instance_id_value, "si_")
    return "mz_" + sha256_hex({
        "schema": _SCHEMA_MATERIALIZATION,
        "sample_id": sample_id_value,
        "label_set_id": label_set_id_value,
        "structure_instance_id": structure_instance_id_value,
        "build_id": _nfc(str(build_id)),
    })
