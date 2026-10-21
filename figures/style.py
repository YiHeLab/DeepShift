"""House style for DeepShift manuscript figures.

Importing changes nothing; ``use_house_style()`` installs the rcParams, so no plot
depends on import order.

The palette was checked against a white page -- all-pairs CIEDE2000
normal and under simulated deuteranopia/protanopia (Vienot 1999), and WCAG contrast.
Worst normal dE 24.6 (floor 20), worst CVD dE 14.7 (floor 9), worst contrast 3.69:1.
Four hues plus a neutral is the ceiling: a fifth collides (green/amber dE 7.8 under
protanopia), so the fifth and sixth series take an existing hue plus a facet.

Roles are fixed, never cycled: ACCENT is DeepShift and is never lent to a competitor;
INK["primary"] is the observed/identity/bound role, never a series.

Typeface is Helvetica (Nimbus Sans, the URW clone) matching the hand-made manuscript
figures, maths included. Verified from an emitted PDF's embedded font names, since
matplotlib falls back silently.
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np

# Validated categorical slots, by role not by index. Assign by role; never cycle.
ACCENT = "#0e7490"    # cyan-700    -- DeepShift, in every figure, always
AMBER = "#b45309"     # amber-700   -- the UCBShift family (UCBShift-X = open face)
VIOLET = "#6d28d9"    # violet-700  -- the SHIFTX2 family (SHIFTX+ = same hue, dashed)
NEUTRAL = "#8a8481"   # warm grey   -- a baseline that is present but not the point
PALETTE = {"slot1": ACCENT, "slot2": AMBER, "slot3": VIOLET, "slot4": NEUTRAL}

# The competitor arms, in plot order, decided in one place and shared by every figure.
# `open_face` marks a family member drawn hollow instead of given its own hue.
ARM_STYLE = {
    "deepshift":  {"color": ACCENT,  "label": "DeepShift"},
    "ucbshift2":  {"color": AMBER,   "label": "UCBShift 2.0"},
    "ucbshift_x": {"color": AMBER,   "label": "UCBShift-X", "open_face": True},
    "shiftx2":    {"color": VIOLET,  "label": "SHIFTX2"},
    "sparta":     {"color": NEUTRAL, "label": "SPARTA+"},
}

# Chart chrome, and the reference role. Text wears an ink token, never a series colour.
INK = {
    "surface": "#ffffff",   # the page a printed figure lands on
    "primary": "#0f172a",   # slate-900 -- also the observed/truth/identity-line role
    "secondary": "#334155",
    "muted": "#64748b",     # axis labels, tick labels, value columns
    "grid": "#e2e8f0",      # hairline, solid, one shade off the surface
    "axis": "#cbd5e1",
    "faint": "#94a3b8",     # a de-emphasised series that is still a series
}

# Figure widths in inches. \textwidth is 6.5in: `article` on letterpaper with
# `\usepackage[margin=1in]{geometry}`, not the 4.79in an unmodified article gives.
# Build the canvas at the width it is placed at, and place at `width=\textwidth`;
# a 7.2in canvas at 0.92\textwidth scales by 0.83 and lands 8pt ticks at 6.6pt.
WIDTH = {"column": 3.18, "text": 6.5, "wide": 6.5}

def bar_style(color, open_face: bool = False) -> dict:
    """Face and edge for one arm's bars. An open face is a family member, not a hue."""
    if open_face:
        return {"color": INK["surface"], "edgecolor": color, "linewidth": 0.9}
    return {"color": color, "edgecolor": INK["primary"], "linewidth": 0.3}


ROW_HEIGHT = 0.155        # inches per row in a forest/dot chart
MARKER = 4.4              # marker diameter in points
LINE = 1.1                # data line / whisker weight in points
RING = 0.7                # surface-coloured ring on overlapping marks


def use_house_style() -> None:
    """Install the house rcParams. Explicit call, never an import side effect."""
    mpl.rcParams.update({
        # Nimbus Sans is the URW Helvetica clone; the rest is a metric-compatible
        # fallback chain, so a rebuild elsewhere does not silently change typeface.
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "Liberation Sans",
                            "DejaVu Sans"],
        # `stix` sets Greek in Times, and `stixsans` still resolves italics through
        # STIXGeneral-Italic; only `custom` keeps the whole expression in one family.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        # Sizes are page sizes, since the canvas is built at the placed width. 8pt is
        # the floor for anything in a panel; the caption beside it is 10pt.
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        # Recessive chrome: hairlines, solid, no dashes anywhere.
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK["secondary"],
        "axes.grid": False,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "text.color": INK["primary"],
        "legend.frameon": False,
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
        # Type 42 embeds TrueType, so a copy-editor can select the text.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 400,
    })


_GREEK = {"a": r"\alpha", "b": r"\beta", "g": r"\gamma", "d": r"\delta",
          "e": r"\epsilon", "z": r"\zeta", "h": r"\eta"}
_SPECIAL = {"c_shift": r"C$^{\prime}$", "n_shift": "N", "h_shift": r"H$^{\mathrm{N}}$"}

BACKBONE = ("n_shift", "h_shift", "ca_shift", "cb_shift", "c_shift", "ha_shift")


def nucleus_label(col: str) -> str:
    """``'hb2_shift'`` -> ``'H$\\beta$2'``. Amide H and carbonyl C keep their names."""
    if col in _SPECIAL:
        return _SPECIAL[col]
    token = col[:-6] if col.endswith("_shift") else col
    element, rest = token[0].upper(), token[1:]
    if rest and rest[0] in _GREEK:
        return f"{element}${_GREEK[rest[0]]}${rest[1:]}"
    return element + rest.upper()


def nucleus_element(col: str) -> str:
    """The NMR-active isotope a column belongs to, for faceting."""
    return {"H": "1H", "C": "13C", "N": "15N"}[col[0].upper()]


ELEMENT_LABEL = {"1H": r"$^{1}$H", "13C": r"$^{13}$C", "15N": r"$^{15}$N"}


def finish(ax, *, xgrid: bool = True, spines=("bottom",)) -> None:
    """Recessive axes: hairline grid on the value axis only, most spines removed."""
    for name, spine in ax.spines.items():
        spine.set_visible(name in spines)
    if xgrid:
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=INK["grid"], linewidth=0.5, linestyle="-")
    ax.tick_params(length=2.5, pad=2)


def forest(ax, labels, series, *, xlabel=None, value_column=None,
           value_header="", xlim=None):
    """A per-nucleus dot chart: one row per label, one dot per series, CI whiskers.

    ``series`` items are dicts of ``name``, ``values`` and optional ``lo``/``hi`` and
    ``color``. Returns the rows' y coordinates so a caller can annotate them.
    """
    slots = [ACCENT, AMBER, VIOLET]
    n_rows, n_series = len(labels), len(series)
    y = np.arange(n_rows, dtype=float)
    # Series are nudged apart inside a row so overlapping CIs stay readable.
    spread = 0.0 if n_series == 1 else 0.30
    offsets = np.linspace(spread / 2, -spread / 2, n_series)

    for k, (spec, dy) in enumerate(zip(series, offsets)):
        color = spec.get("color", slots[k % len(slots)])
        vals = np.asarray(spec["values"], dtype=float)
        if spec.get("lo") is not None:
            lo, hi = np.asarray(spec["lo"], float), np.asarray(spec["hi"], float)
            ax.hlines(y + dy, lo, hi, color=color, linewidth=LINE,
                      zorder=2, alpha=0.85)
        ax.plot(vals, y + dy, linestyle="none", marker="o", markersize=MARKER,
                color=color, markeredgecolor=INK["surface"], markeredgewidth=RING,
                label=spec["name"], zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK["primary"])
    ax.set_ylim(n_rows - 0.5, -0.5)          # first row at the top
    if xlim is not None:
        ax.set_xlim(*xlim)
    if xlabel:
        ax.set_xlabel(xlabel)
    finish(ax)

    if value_column is not None:
        x = 1.015
        for row, text in zip(y, value_column):
            ax.text(x, row, text, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=7.5, color=INK["muted"])
        if value_header:
            ax.text(x, -0.9, value_header, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=7.5, color=INK["secondary"])
    return y


def direct_label(ax, x, y, text, *, color, dx=0.0, dy=0.0, **kw):
    """Label a single mark; the label is the identity channel, so it takes the hue."""
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=8.0, color=color, **kw)


def panel_label(ax, text, *, x=-0.02, y=1.0):
    """Bold (a)/(b) marker, positioned in axes fraction."""
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold", color=INK["primary"])


def legend(target, *args, **kw):
    """A legend, always present for two or more series.

    ``target`` may be an Axes or a Figure; a figure-level legend keeps the key out of
    a dense data area. Extra positional arguments are the usual ``handles, labels``.
    """
    kw.setdefault("handletextpad", 0.4)
    kw.setdefault("borderaxespad", 0.2)
    leg = target.legend(*args, **kw)
    for text in leg.get_texts():
        text.set_color(INK["secondary"])
    return leg
