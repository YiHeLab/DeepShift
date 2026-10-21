"""The graphical abstract: the pipeline on top, one measured result below.

The register is a table-of-contents graphic, not a poster. The published graphics for
comparable methods (UCBShift 2.0, LEGOLAS) are a labelled left-to-right flow with plain
noun labels and no assertions; the reader is shown what goes in, what comes out, and one
piece of evidence. Nothing here is phrased as a claim, because the title, the abstract
and the caption already carry the claims.

Three decisions worth stating, because each is why something is drawn the way it is.

  * The lower panel is the ring-current response, not an accuracy bar. A proton held over
    an aromatic ring is shielded by an amount that follows from the coordinates alone: no
    residue prior and no homolog supplies it. How closely a predictor tracks that curve is
    a statement an accuracy number cannot make.
  * The geometry is drawn beside the curve, so the horizontal axis needs no sentence to
    explain it. It is not a cartoon: it is the single most strongly shielded proton in the
    benchmark, the point at the right-hand end of the curve, ray traced from the deposited
    coordinates by figures/render_structures.py.
  * The strip repeats Figure 1's protein, Figure 1's cutoff and Figure 1's renders. A
    graphical abstract that shows a different protein from the figure it distils makes the
    reader check whether the two are the same argument.

Run with $DEEPSHIFT_RUN set. Writes to $DEEPSHIFT_FIGURES_OUT, or figures/out.
"""

import csv
import itertools
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import frozen
import matplotlib.pyplot as plt
from build_paper_figures import (
    CARD_FACE,
    FOLD1_CHECKPOINT,
    REFERENCE,
    house_style,
    query_links,
    readout_attention,
    render,
    render_manifest,
)
from style import ACCENT, AMBER, INK, VIOLET, WIDTH

HERE = Path(__file__).resolve().parent
OUT = frozen.OUT_DIR

RING = "campaign/cpu/fig_06_structure/ring_cells.npz"
CUTOFF = 8.0

# The strip is drawn from the protein Figure 1 is drawn from, at the same 8 Angstrom
# cutoff, so the two are one picture at two levels of detail rather than two pictures.
FIGURE_ONE = (REFERENCE / "1POH.pdb", "A")

# The ring-current test. An aromatic proton sitting over a neighbouring ring is shielded,
# and by how much is pure geometry: it is not a property of the residue type or of any
# homolog. The field is computed from the coordinates, so the horizontal axis is physics
# and the vertical axis is what each predictor does about it.
RING_BINS = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.65, 1.20])
RING_ARMS = [
    ("true_ppm", "observed", INK["primary"], 2.2),
    ("pred_deepshift", "DeepShift", ACCENT, 2.0),
    ("pred_shiftx2_sameprot", "SHIFTX2", VIOLET, 1.4),
    ("pred_ucbshift2_sameprot", "UCBShift 2.0", AMBER, 1.4),
]

def ring_response(run: Path):
    """Binned secondary shift against the computed ring-current field, per predictor.

    Secondary shift is taken against each atom type's own median, so the curve is the
    response to the field rather than a statement about which protons are in the set.
    """
    z = np.load(run / RING, allow_pickle=True)
    ring = z["ring_ppm"]
    atom = np.array([str(a) for a in z["atom"]])

    def secondary(v):
        out = np.full(len(v), np.nan)
        for a in np.unique(atom):
            m = (atom == a) & np.isfinite(v)
            if m.any():
                out[m] = v[m] - np.median(v[m])
        return out

    centres, counts, curves = [], [], {k: [] for k, *_r in RING_ARMS}
    for lo, hi in itertools.pairwise(RING_BINS):
        m = (ring >= lo) & (ring < hi)
        if m.sum() < 20:
            continue
        centres.append(float(np.median(ring[m])))
        counts.append(int(m.sum()))
        for key, *_r in RING_ARMS:
            curves[key].append(float(np.nanmean(secondary(z[key])[m])))
    return np.array(centres), np.array(counts), {k: np.array(v) for k, v in curves.items()}


def most_shielded(run: Path):
    """The benchmark's most strongly ring-shielded proton, and the structure it sits in.

    This one cell chooses everything drawn: it is the right-hand end of the curve below,
    and the protein it belongs to is the protein the strip above is drawn from. Resolving
    it goes through the build log the run was trained from, so the accession is the one
    the pipeline itself used, not one written down here.
    """
    z = np.load(run / RING, allow_pickle=True)
    i = int(np.argmax(z["ring_ppm"]))
    cell = {k: str(z[k][i]) for k in ("sid", "resnum", "atom", "residue", "label_set",
                                      "source")}
    cell["ring_ppm"] = float(z["ring_ppm"][i])
    if cell["source"] not in ("nmr", "xtal"):
        raise SystemExit(f"{cell['sid']} is an {cell['source']} model; the inset wants a "
                         "deposited structure, whose proton positions are experimental")

    log = run / "data" / f"build_log_{cell['source']}.csv"
    rows = [r for r in csv.DictReader(log.open())
            if r["label_set_id"] == cell["label_set"] and r["status"] == "success"]
    if len(rows) != 1:
        raise SystemExit(f"{log.name} has {len(rows)} successful rows for "
                         f"{cell['label_set']}; expected exactly one")
    cell["accession"] = rows[0]["accession"].upper()
    cell["chain"] = rows[0]["chain"]
    cell["path"] = run / "data" / "pdbs" / f"{cell['accession']}.pdb"
    if not cell["path"].exists():
        raise SystemExit(f"{cell['path']} is not in the run's structure store")
    return cell


def chain_residues(path: Path, chain: str):
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("x", str(path))
    return [r for r in structure[0][chain] if r.id[0] == " " and "CA" in r]


def neighbourhood(residues):
    """The real 8 Angstrom heavy-atom cloud around one residue, and its distance matrix."""
    target = residues[len(residues) // 2]
    heavy = [(a, a.element) for r in residues for a in r if a.element != "H"]
    coords = np.array([a.coord for a, _e in heavy])
    elements = np.array([e for _a, e in heavy])
    centre = np.array([a.coord for a in target if a.element != "H"]).mean(axis=0)
    keep = np.linalg.norm(coords - centre, axis=1) <= CUTOFF
    in_target = np.array([a.get_parent() is target for a, _e in heavy])
    xyz = coords[keep] - centre
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    return xyz, elements[keep], in_target[keep], dist, int(keep.sum())


def build(run: Path):
    house_style()
    renders = render_manifest()
    centres, counts, curves = ring_response(run)

    # The inset is a measured cell, not an illustration, so the shipped render has to be
    # the cell this run still chooses. If the run changes, the build stops here.
    cell = most_shielded(run)
    shown = renders["ring_pair"]
    if (shown["accession"] != cell["accession"]
            or shown["proton"] != f"{cell['residue']}{cell['resnum']} {cell['atom']}"):
        raise SystemExit(f"the rendered ring pair is {shown['proton']} in "
                         f"{shown['accession']}, this run's strongest field is "
                         f"{cell['residue']}{cell['resnum']} {cell['atom']} in "
                         f"{cell['accession']}; re-run figures/render_structures.py")

    residues = chain_residues(*FIGURE_ONE)
    _xyz, _elements, _in_target, dist, n_atoms = neighbourhood(residues)
    cloud = renders["atom_cloud"]
    if cloud["cutoff_angstrom"] != CUTOFF or cloud["atoms_drawn"] != n_atoms:
        raise SystemExit(f"the rendered cloud holds {cloud['atoms_drawn']} atoms at "
                         f"{cloud['cutoff_angstrom']} A, this distance matrix holds "
                         f"{n_atoms} at {CUTOFF} A; re-run figures/render_structures.py")

    query_weights, query_baseline, _n_residues, _n_shifts = readout_attention(
        FOLD1_CHECKPOINT, *FIGURE_ONE)

    member = frozen.member_dumps(6, (1,))[0]
    from deepshift import evaluate
    observed = evaluate.read_predictions(member)
    peaks = np.array([observed.true_ppm[observed.mask[:, j], j].mean()
                      for j in range(observed.true_ppm.shape[1])])
    n_heads = observed.true_ppm.shape[1]

    fig = plt.figure(figsize=(WIDTH["text"], 3.75))

    # Two rows. Upper is the pipeline, left to right; lower is the evidence. Every
    # coordinate derives from these numbers, so nothing can drift into anything else.
    MARG = 0.030
    TOP, TH = 0.660, 0.290          # the pipeline cards
    NOTE = 0.578                    # the one line of prose, under the cards
    BOT, BH = 0.050, 0.468          # the result card

    def card(x, y, w, h):
        fig.patches.append(mpl.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.009",
            transform=fig.transFigure, facecolor=CARD_FACE, edgecolor=INK["axis"],
            linewidth=0.75, zorder=1))

    def inset(x, y, w, h, pad=0.010):
        ax = fig.add_axes([x + pad, y + pad, w - 2 * pad, h - 2 * pad], zorder=3)
        ax.set_axis_off()
        ax.patch.set_alpha(0)
        return ax

    def cap(x, y, w, text):
        fig.text(x + w / 2, y - 0.028, text, ha="center", va="top", fontsize=6.6,
                 color=INK["secondary"], zorder=4)

    def chevron(x, y, *, w=0.010, hh=0.022):
        fig.patches.append(mpl.patches.Polygon(
            [[x, y + hh], [x + w, y], [x, y - hh]], closed=True,
            transform=fig.transFigure, facecolor="none", edgecolor=INK["faint"],
            linewidth=1.0, zorder=4))

    # ---- upper row: structure -> cloud -> distances -> queries -> shifts -------------
    N, GAP = 5, 0.0335
    CW = (1.0 - 2 * MARG - (N - 1) * GAP) / N
    xs = [MARG + i * (CW + GAP) for i in range(N)]

    render(inset(xs[0], TOP, CW, TH, pad=0.006), "structure")
    cap(xs[0], TOP, CW, "protein structure")

    render(inset(xs[1], TOP, CW, TH, pad=0.006), "atom_cloud")
    cap(xs[1], TOP, CW, "8 \u00c5 atom cloud")

    ax = inset(xs[2], TOP, CW, TH)
    # The same ramp Figure 1 draws this matrix in; one object, one colour scheme.
    ax.imshow(dist, interpolation="nearest", aspect="equal",
              cmap=mpl.colors.LinearSegmentedColormap.from_list(
                  "geom", ["#062f38", ACCENT, "#a5dbe6", INK["surface"]]))
    cap(xs[2], TOP, CW, "interatomic distances")

    query_links(inset(xs[3], TOP, CW, TH, pad=0.012), query_weights, query_baseline)
    cap(xs[3], TOP, CW, "nucleus queries")

    ax = inset(xs[4], TOP, CW, TH)
    rng = np.abs(np.sin(np.arange(len(peaks)) * 12.9898))
    ax.vlines(peaks, 0, 0.35 + 0.6 * rng, color=ACCENT, lw=0.8)
    ax.axhline(0, color=INK["axis"], lw=0.7)
    ax.set_xlim(peaks.max() + 18, -16)
    ax.set_ylim(-0.06, 1.08)
    cap(xs[4], TOP, CW, f"{n_heads} chemical shifts")

    for i in range(N):
        card(xs[i], TOP, CW, TH)
    for i in range(N - 1):
        chevron(xs[i] + CW + GAP / 2 - 0.005, TOP + TH / 2)

    fig.text(MARG, NOTE, "structure alone; no homology transfer", fontsize=7.0,
             color=INK["muted"], ha="left", va="top", zorder=4)

    # ---- lower row: the response to a field only the geometry knows about ------------
    card(MARG, BOT, 1.0 - 2 * MARG, BH)

    # The geometry the horizontal axis measures comes first, then the response to it, so
    # the panel reads left to right like the row above it.
    side_w = 0.205
    side_h = side_w * WIDTH["text"] / 3.75
    render(fig.add_axes([MARG, BOT + (BH - side_h) / 2, side_w, side_h], zorder=3),
           "ring_pair")

    ax = fig.add_axes([0.285, BOT + 0.105, 0.480, BH - 0.145], zorder=3)
    for key, name, colour, lw in RING_ARMS:
        y = curves[key]
        ax.plot(centres, y, "-", color=colour, linewidth=lw,
                zorder=3 if name == "observed" else 2)
        ax.scatter(centres, y, s=np.clip(counts / 26.0, 6, 34), color=colour,
                   zorder=4, linewidths=0)
        ax.annotate(name, (centres[-1], y[-1]), textcoords="offset points",
                    xytext=(7, 0), va="center", ha="left", fontsize=6.8, color=colour,
                    annotation_clip=False,
                    fontweight="bold" if name in ("observed", "DeepShift") else "normal")
    ax.axhline(0, color=INK["axis"], lw=0.7, zorder=1)
    ax.set_xlabel("ring-current shielding computed from the structure (ppm)",
                  fontsize=7.0, color=INK["secondary"], labelpad=2)
    ax.set_ylabel("secondary $^{1}$H shift (ppm)", fontsize=7.0,
                  color=INK["secondary"], labelpad=2)
    ax.tick_params(labelsize=6.4, colors=INK["secondary"], length=2.5, pad=1.5)
    # The axis stops where the data stop; the series labels sit outside it, so no
    # empty tail is drawn just to hold them.
    ax.set_xlim(-0.02, centres[-1] * 1.06)
    for side_name in ("top", "right"):
        ax.spines[side_name].set_visible(False)
    for side_name in ("left", "bottom"):
        ax.spines[side_name].set_color(INK["axis"])

    stem = "graphical_abstract"
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=400)
    print(f"wrote {OUT / stem}.pdf and .png")
    print(f"  strip drawn from {FIGURE_ONE[0].name} chain {FIGURE_ONE[1]}, "
          f"{n_heads} output heads")
    print(f"  inset: {shown['proton']} over {shown['shielding_ring']}, "
          f"{shown['distance_to_ring_centre_angstrom']} A at "
          f"{shown['angle_off_ring_normal_deg']} deg off the normal, "
          f"field {shown['ring_ppm']} ppm")


if __name__ == "__main__":
    sys.exit(build(frozen.RUN_ROOT))
