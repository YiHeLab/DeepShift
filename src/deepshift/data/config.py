"""Vocabulary and roots shared by the data stages.

The chemistry tables are the package's own (``deepshift.domain``); the roots come from
``paths``. Both are re-exported here so a stage imports one module.
"""

from ..domain import AA_3_TO_1, ATOM_TO_IDX, ATOM_TYPES, RESIDUE_TO_IDX, STANDARD_RESIDUES
from .paths import ALPHAFOLD_DIR, BIG_STORAGE, DATA_ARCHIVE, DATA_DIR, PDB_DIR

# How a label residue relates to the structure residue it was bound to. New kinds are
# appended so the stored indices stay stable. ``insertion_code_unsupported`` marks a
# label residue mapped to an insertion-coded structure residue, which the integer-keyed
# cache cannot address; ``label_residue_code_mismatch`` marks a deposited residue code
# that disagrees with the polymer sequence. Both are tagged and masked from the loss.
MISMATCH_TYPES = ['match', 'gap_in_cs', 'gap_in_structure', 'mismatch',
                  'protein_edge', 'UNK', 'insertion_code_unsupported',
                  'label_residue_code_mismatch']
MISMATCH_TO_IDX = {m: i for i, m in enumerate(MISMATCH_TYPES)}

__all__ = [
    "AA_3_TO_1",
    "ALPHAFOLD_DIR",
    "ATOM_TO_IDX",
    "ATOM_TYPES",
    "BIG_STORAGE",
    "DATA_ARCHIVE",
    "DATA_DIR",
    "MISMATCH_TO_IDX",
    "MISMATCH_TYPES",
    "PDB_DIR",
    "RESIDUE_TO_IDX",
    "STANDARD_RESIDUES",
]
