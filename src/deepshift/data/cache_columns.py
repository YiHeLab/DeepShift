"""Column parsing shared by the cache builder."""
from ..domain import SHIFT_COLUMNS


def parse_shift_columns(columns):
    return sorted([c for c in columns if c.endswith('_shift')])


def require_release_shift_columns(columns):
    """Return the frozen release target schema (raises on missing/extra); the release
    CLI must never infer its model width from whichever columns occur in one CSV."""
    found = tuple(parse_shift_columns(columns))
    expected = tuple(SHIFT_COLUMNS)
    if found != expected:
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        raise ValueError(
            "release cache requires the frozen 64-column shift schema; "
            f"missing={missing}, extra={extra}"
        )
    return list(expected)
