"""Locations inside the X-EISD checkout and the MIXED-pool row layout, shared by the drivers.

The checkout is the Lincoff et al. repository, named by ``XEISD_HOME`` (see README.md).
"""
from __future__ import annotations

import os
from pathlib import Path

N_ENSEMBLE = 1700
N_RANDOM = 100000


def xeisd_home() -> Path:
    """The X-EISD checkout; a missing variable fails here, naming it."""
    value = os.environ.get("XEISD_HOME")
    if not value:
        raise SystemExit("XEISD_HOME must name the X-EISD checkout (see README.md)")
    return Path(value).expanduser().resolve()


def mixed_random_indices(xeisd: Path) -> list[int]:
    """The RANDOM-pool structures the paper's MIXED pool embeds, in its own row order.

    X-EISD's MIXED table is the 1,700 ENSEMBLE rows (indices 0..1699) followed by 1,700
    RANDOM rows; its first column carries each row's source index in its own pool.
    """
    path = xeisd / "data/back_calc_data/mixpool/drksh3_mixpool_CS.txt"
    with path.open() as stream:
        indices = [int(line.split(maxsplit=1)[0]) for line in stream if line.strip()]
    if len(indices) != 2 * N_ENSEMBLE:
        raise ValueError(f"{path}: expected {2 * N_ENSEMBLE} MIXED rows, found {len(indices)}")
    if indices[:N_ENSEMBLE] != list(range(N_ENSEMBLE)):
        raise ValueError(f"{path}: the first {N_ENSEMBLE} rows are not ENSEMBLE rows "
                         f"0..{N_ENSEMBLE - 1}")
    random_rows = indices[N_ENSEMBLE:]
    if len(set(random_rows)) != N_ENSEMBLE or not all(0 <= i < N_RANDOM for i in random_rows):
        raise ValueError(f"{path}: the RANDOM source indices are not {N_ENSEMBLE} unique "
                         "valid rows")
    return random_rows
