"""The public chemistry vocabulary used by caches, checkpoints, and predictors."""
from __future__ import annotations

from .domain import (
    AA_3_TO_1,
    ATOM_TO_IDX,
    ATOM_TYPES,
    NUCLEI,
    RESIDUE_TO_IDX,
    SHIFT_ATOM_NAMES,
    SHIFT_COLUMNS,
    STANDARD_RESIDUES,
)

# Schema-invalid labels are rejected before QC. They are not extreme values of a
# valid population and therefore do not belong to the p100 rule.
CHEMICALLY_IMPOSSIBLE_PAIRS = frozenset({("PRO", "h_shift")})


def atom_from_shift_column(column: str) -> str:
    """Convert a cache/model shift-column name to its atom name."""
    suffix = "_shift"
    if not column.endswith(suffix) or len(column) == len(suffix):
        raise ValueError(f"not a shift column: {column!r}")
    return column[: -len(suffix)].upper()


__all__ = [
    "AA_3_TO_1",
    "ATOM_TO_IDX",
    "ATOM_TYPES",
    "CHEMICALLY_IMPOSSIBLE_PAIRS",
    "NUCLEI",
    "RESIDUE_TO_IDX",
    "SHIFT_ATOM_NAMES",
    "SHIFT_COLUMNS",
    "STANDARD_RESIDUES",
    "atom_from_shift_column",
]
