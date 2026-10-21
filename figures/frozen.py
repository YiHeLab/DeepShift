"""Frozen-artifact access for the figure scripts.

Every figure script reads a file that already exists on disk. Nothing here fabricates or
interpolates a value.

**No statistic is defined here.** Prochiral resolution, per-nucleus error, the
secondary-shift r and amplitude, and the clustered bootstrap all live in
``deepshift.evaluate``; a second implementation would be a second thing to keep right.

What this module owns is where an artifact lives: the run root and the directories under
it that the pipeline writes. A prediction dump is
loaded by ``evaluate.read_predictions`` and members are combined by ``evaluate.ensemble``,
which averages in ppm because each member has already denormalized with its own
checkpoint statistics.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent
REPO = FIGURES_DIR.parent
# Where a figure and its png land. Overridable, as si_build's SI_OUT is,
# so a regeneration against another scoring run is written elsewhere and never lands
# on the deployed file by accident.
OUT_DIR = Path(os.environ.get("DEEPSHIFT_FIGURES_OUT") or FIGURES_DIR / "out")

# Figure scripts are run directly (`python figures/fig_x.py`), so make the package
# importable without asking the caller to set PYTHONPATH. The repository root goes on
# too, for `benchmarks.master_cells`: the panels that compare predictors read the frozen
# master cell file through the benchmark's own loader rather than a second copy of it.
for directory in (REPO / "src", REPO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from deepshift.data import paths

# Every figure reads one finished CV run, resolved once here through the same env var
# the rest of the pipeline uses (paths.RUN_ROOT / $DEEPSHIFT_RUN). A release checkout
# never guesses where an artifact lives.
RUN_ROOT = (
    Path(paths.RUN_ROOT).resolve()
    if paths.RUN_ROOT
    else Path("__DEEPSHIFT_RUN_REQUIRED__")
)
BUILD_DIR = RUN_ROOT / "data"           # the build data dir the run was trained from
FINAL_DIR = RUN_ROOT / "final"          # the paper members and everything scored from them
PREDS_DIR = FINAL_DIR / "plddt_preds"   # the members' held-out dumps as refold_fold{k}
MEMBER_DUMPS_DIR = FINAL_DIR / "intervals"   # one held-out dump per member per scored fold
INTERVAL_CALIBRATION = FINAL_DIR / "interval_calibration.json"


def env_path(name: str, *parts: str) -> Path:
    """Resolve an artifact root from ``$name``, with no workstation fallback.

    A missing variable yields a visibly invalid sentinel path, so import and ``--help``
    still work and the first read fails naming the variable it needs.
    """
    value = os.environ.get(name)
    root = Path(value).expanduser().resolve() if value else Path(f"__{name}_REQUIRED__")
    return root.joinpath(*parts)


def member_dumps(fold: int, members=(1, 2, 3, 4, 5), *, directory=None) -> list[Path]:
    """One prediction dump per member for a fold every member held out, checked to exist.

    These are what ``deepshift evaluate --save-npz`` writes, in the order the members are
    named. Unset ``$DEEPSHIFT_RUN`` surfaces as the sentinel inside the reported path, so
    the failure names the variable instead of a bare missing file.
    """
    root = Path(directory) if directory is not None else MEMBER_DUMPS_DIR
    dumps = [root / f"fold{fold}_member{member}.npz" for member in members]
    missing = [str(p) for p in dumps if not p.exists()]
    if missing:
        raise SystemExit("missing member prediction dumps:\n  " + "\n  ".join(missing))
    return dumps


#: The deposited channels the benchmark excludes, as one (label set, nucleus column) set.
#: Kept beside the evidence that condemned them rather than typed out here.
ANTICORRELATED_CHANNELS = Path(__file__).resolve().parent.parent / (
    "benchmarks/reference/anticorrelated_channels.json")

_ATOM_COLUMN = {"C": "c_shift", "H": "h_shift", "HA": "ha_shift",
                "CA": "ca_shift", "CB": "cb_shift", "N": "n_shift"}


def anticorrelated_channels() -> set:
    """``(label_set, shift column)`` for every channel the benchmark drops."""
    record = json.loads(ANTICORRELATED_CHANNELS.read_text())
    return {(c["label_set"], _ATOM_COLUMN[c["nucleus"]]) for c in record["channels"]}


def drop_anticorrelated(predictions):
    """Clear the mask on channels whose deposited series every judge disagrees with.

    These are the cells Methods excludes: one deposition and one nucleus where all four
    of ProCS15, SPARTA+, UCBShift-X and SHIFTX2 correlate negatively with the deposited
    values. A figure drawn from the raw member dumps sees them unless it asks, which is
    how Figure 2(a) came to print a C-prime RMSE the benchmark tables do not use. Every
    producer that scores fold-6 cells calls this, so the fence is one rule in one place.

    Returns the predictions unchanged when the dump carries none of the channels.
    """
    from dataclasses import replace

    channels = anticorrelated_channels()
    cols = list(predictions.shift_cols)
    mask = predictions.mask.copy()
    effective = predictions.effective_mask.copy()
    dropped = 0
    for label_set, column in channels:
        if column not in cols:
            continue
        j = cols.index(column)
        rows = predictions.label_set == label_set
        dropped += int(effective[rows, j].sum())
        mask[rows, j] = False
        effective[rows, j] = False
    return replace(predictions, mask=mask, effective_mask=effective), dropped


def emit(fig, stem: str) -> None:
    """Write ``<stem>.pdf`` and ``<stem>.png`` to ``out/``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT_DIR / f"{stem}.pdf", OUT_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    print(f"wrote {pdf} ({pdf.stat().st_size:,} B)")
    print(f"wrote {png} ({png.stat().st_size:,} B)")
