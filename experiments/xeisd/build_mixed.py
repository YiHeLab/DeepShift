#!/usr/bin/env python3
"""Assemble DeepShift's MIXED shift table from its ENSEMBLE and RANDOM rows.

X-EISD's MIXED pool is the 1,700 ENSEMBLE conformers followed by 1,700 named RANDOM
conformers, taken at the source indices and in the row order its own MIXED table carries.
MIXED therefore needs no prediction of its own: it re-orders rows already computed.
Reads ``drksh3_ENSEMBLE_CS_deepshift.txt`` and ``drksh3_RANDOM_MIXEDROWS_CS_deepshift.txt``
from ``--tables`` and writes ``drksh3_MIXED_CS_deepshift.txt`` beside them.

    XEISD_HOME=<checkout> python build_mixed.py --tables <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pools import N_ENSEMBLE, mixed_random_indices, xeisd_home


def rows_of(path: Path) -> dict[int, str]:
    """A shift table as {source index: the 262 values}."""
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            index, values = line.split(maxsplit=1)
            out[int(index)] = values
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True,
                        help="directory run_shift_tables.py wrote its tables to")
    args = parser.parse_args()

    order = mixed_random_indices(xeisd_home())
    ensemble = rows_of(args.tables / "drksh3_ENSEMBLE_CS_deepshift.txt")
    random_path = args.tables / "drksh3_RANDOM_MIXEDROWS_CS_deepshift.txt"
    random_rows = rows_of(random_path)
    if sorted(ensemble) != list(range(N_ENSEMBLE)):
        raise ValueError(f"the ENSEMBLE table is not exactly rows 0..{N_ENSEMBLE - 1}")
    missing = [index for index in order if index not in random_rows]
    if missing:
        raise ValueError(f"{random_path} lacks MIXED source rows, e.g. {missing[:5]}")

    destination = args.tables / "drksh3_MIXED_CS_deepshift.txt"
    destination.write_text(
        "".join(f"{index} {ensemble[index]}\n" for index in range(N_ENSEMBLE))
        + "".join(f"{index} {random_rows[index]}\n" for index in order))
    print(f"wrote {destination}: {N_ENSEMBLE} ENSEMBLE + {len(order)} indexed RANDOM rows")


if __name__ == "__main__":
    main()
