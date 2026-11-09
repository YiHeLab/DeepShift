"""Page and type for the figures built at Nature Methods' size: Figures 1 and 5, Extended Data Fig. 2.

These are drawn at the journal's double-column width, 183 mm, with 5 to 8 pt type, and every
panel is placed on the canvas by hand (no constrained layout). Colours are the house roles of
``style.py``; this module adds the page, the type sizes and three light tints of those roles.
"""
from __future__ import annotations

import matplotlib as mpl
import style

MM = 1 / 25.4
DOUBLE = 183 * MM          # inches

FS = {"text": 6.0, "small": 5.5, "tick": 5.5, "label": 6.0, "title": 6.5, "panel": 8.0, "note": 5.0}

ACCENT, AMBER, VIOLET, NEUTRAL = style.ACCENT, style.AMBER, style.VIOLET, style.NEUTRAL
INK, SECOND, MUTED, FAINT, GRID, AXIS = (
    style.INK[k] for k in ("primary", "secondary", "muted", "faint", "grid", "axis"))
LIGHT_ACCENT, LIGHT_AMBER, LIGHT_VIOLET = "#a5d8e4", "#f1c9a0", "#cdbdf0"


def use_journal_style() -> None:
    """Install the rcParams of these figures. Explicit call, never an import side effect."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold",
        "font.size": FS["text"], "axes.labelsize": FS["label"], "axes.titlesize": FS["title"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["small"],
        "axes.edgecolor": MUTED, "axes.linewidth": 0.5, "axes.labelcolor": SECOND,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "axes.labelpad": 1.5, "text.color": INK, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "lines.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def clean(ax, spines=("left", "bottom"), ygrid=False) -> None:
    """Keep only the named spines; an optional y grid behind the data."""
    for k, s in ax.spines.items():
        s.set_visible(k in spines)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
