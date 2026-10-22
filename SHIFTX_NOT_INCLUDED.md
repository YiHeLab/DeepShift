# SHIFTX is not in this archive

SHIFTX is used by the rereferencing stage of the data build. It is not redistributable:
the implementation includes Numerical Recipes routines whose licence forbids
redistribution, so no copy of it appears here and none may be added to a public copy of
this archive.

The pipeline does not need one bundled. It downloads SHIFTX from upstream at build time and
checks it against a pinned SHA-256 before use; see `THIRD_PARTY_NOTICES.md`. To use a copy
you already have, pass it explicitly:

    deepshift data build ... --shiftx-bin /path/to/shiftx

SHIFTX is required only by the rereferencing stage of the build; the released dataset
already has the offsets applied.
