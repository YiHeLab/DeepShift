"""The four manuscript figures that are not about a competitor.

    Figure 1   method flow chart, six stations           after SHIFTX2 Fig. 1
    Figure 2   predicted vs observed, one panel/nucleus  after UCBShift 2.0 Fig. 2
    Figure S7  learning curve, RMSE against rows         after UCBShift Fig. S7
    Figure S8  error surface over (phi, psi)             after SPARTA+ Eq. 4

Each is anchored to a specific incumbent figure, never to taste: where the paper
reports something the field does not report, the form is taken from the nearest
incumbent figure answering a question of the same shape.

House rules, read off those figures: boxed axes, ticks out, a red y=x line in every
scatter, the metric printed in the panel title, a framed legend inside the axes, bars
anchored at zero. No figure carries a title band, a subtitle or explanatory prose --
that is the caption's job.

The competitor-facing figures are drawn elsewhere in this directory, one script each,
because each carries a self-exclusion tier this file has no reason to know about.

Figures 1 and 2 are drawn from the run named by ``$DEEPSHIFT_RUN``: its five fold-6
member prediction dumps and the fold-1 member's checkpoint. Figure S8 is drawn from the
same run's five held-out cross-validation dumps, through the torsion cell set
``$DEEPSHIFT_RAMACHANDRAN_OOF`` names, and Figure S7 from this run's data-fraction
ladder under ``$DEEPSHIFT_SCALING_LADDER``.
"""

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import frozen
import matplotlib.pyplot as plt

# Colour is decided in one place for the whole paper; see style.py for the validation
# these tokens carry. Never introduce a literal here.
from style import ACCENT, INK, PALETTE, WIDTH
from style import use_house_style as _house_rcparams

from deepshift import evaluate
from deepshift.calibration import EVALUATION_FOLD

HERE = Path(__file__).resolve().parent
OUT = frozen.OUT_DIR                      # $DEEPSHIFT_FIGURES_OUT, or figures/out
REFERENCE = HERE / "reference"
# The molecular panels are ray traced once by figures/render_structures.py and read back
# here, so the figures stay pure matplotlib and a reader needs no PyMOL to rebuild them.
RENDERS = REFERENCE / "renders"
# Figure S7's data-fraction ladder, retrained on this run: one directory per point,
# each holding the trainer's record and the point's evaluation on fold 6. Figure S8's
# pooled out-of-fold torsion cells are this run's own, written by
# benchmarks/dump_cv_torsions.py. Both are named by environment, with no fallback.
SCALING_LADDER = frozen.env_path("DEEPSHIFT_SCALING_LADDER")
RAMACHANDRAN_CELLS = frozen.env_path("DEEPSHIFT_RAMACHANDRAN_OOF", "cv_cells.npz")
# The fold-1 member's checkpoint: Figure 1 draws the trained radial basis and the
# distance-to-attention map out of it. A reproduction bundle carries no checkpoint, so
# without the run's own file the figure reads the shipped member, which holds the same
# tensors re-saved with run-relative paths.
FOLD1_CHECKPOINT = (frozen.FINAL_DIR / "fold1.final.pt"
                    if (frozen.FINAL_DIR / "fold1.final.pt").is_file()
                    else frozen.REPO / "models" / "deepshift_fold1.pt")

# Field ordering, after SHIFTX2 Fig. 2: 15N, 13Ca, 13Cb, 13C', 1HN, 1Ha.
BACKBONE = [
    ("n_shift", "N", r"$^{15}$N"),
    ("ca_shift", "CA", r"$^{13}$C$\alpha$"),
    ("cb_shift", "CB", r"$^{13}$C$\beta$"),
    ("c_shift", "C", r"$^{13}$C$^{\prime}$"),
    ("h_shift", "H", r"$^{1}$H$^{N}$"),
    ("ha_shift", "HA", r"$^{1}$H$\alpha$"),
]


def house_style():
    """The house rcParams, then the two this file needs to differ on.

    style.py turns constrained_layout on; every figure here places its own axes or
    calls tight_layout, and matplotlib warns and ignores one of them if both are
    active. Sizes come from style.py so nothing in a panel is below 8pt.
    """
    _house_rcparams()
    mpl.rcParams.update({
        "figure.constrained_layout.use": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel_letter(ax, letter, dx=-0.09, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def framed_legend(ax, **kw):
    kw.setdefault("frameon", True)
    kw.setdefault("framealpha", 1.0)
    kw.setdefault("edgecolor", "black")
    kw.setdefault("fancybox", False)
    kw.setdefault("handlelength", 1.2)
    leg = ax.legend(**kw)
    leg.get_frame().set_linewidth(0.6)
    return leg


def save(fig, stem):
    """Write the figure as pdf and png."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}")


def stats(pred, obs):
    """RMSE, MAE and the raw-ppm correlation this panel plots.

    r here is on raw ppm, not on the secondary shift: the panel is the
    predicted-vs-observed grid, and its r is the correlation of what is drawn. That makes
    it inflated by residue-type variance -- a residue-prior model scores 0.997 on Cbeta --
    which the deployed caption states. The secondary-shift correlation is Figure 2, panel (b).
    """
    err = pred - obs
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r": float(np.corrcoef(pred, obs)[0, 1]),
        "n": int(err.size),
    }


# ------------------------------------------------------------- figure 1 -----

def _learned_geometry(ckpt, n_points=400):
    """The trained radial basis and the distance-to-attention-bias map.

    Both come straight out of the checkpoint, so the curves in Fig. 1 are the
    model's own and not a schematic: pair_mlp is Linear(32, 16) -> GELU ->
    Linear(16, 8) and its output is the additive per-head attention bias.
    """
    import torch

    state = torch.load(ckpt, map_location="cpu",
                       weights_only=False)["model_state_dict"]
    centres = state["rbf.centers"].numpy()
    beta = float(state["rbf.beta"])
    grid = np.linspace(0, float(centres.max()), n_points)
    phi = np.exp(-beta * (grid[:, None] - centres[None, :]) ** 2)
    with torch.no_grad():
        h = torch.from_numpy(phi).float() @ state["pair_mlp.0.weight"].T \
            + state["pair_mlp.0.bias"]
        h = torch.nn.functional.gelu(h)
        bias = (h @ state["pair_mlp.2.weight"].T
                + state["pair_mlp.2.bias"]).numpy()
    return grid, phi, bias, state["atom_embed.weight"].numpy(), centres


# The six backbone nuclei, in the order the readout panel draws them, and the atoms of a
# residue they can read: its own backbone, plus one column for everything past CB.
BACKBONE_QUERIES = (("N", "N"), ("H", "HN"), ("CA", "CA"), ("HA", "HA"),
                    ("C", "C'"), ("CB", "CB"))
RESIDUE_ATOMS = ("N", "CA", "C", "O", "CB", "R")


def readout_attention(checkpoint, pdb, chain):
    """Where each nucleus query looks, measured on a real chain.

    The readout is cross-attention: one learned query per predicted nucleus, attending
    over the atoms of the residue being predicted -- and only those, the surrounding cloud
    having already been mixed in by the encoder. This runs the model once over every
    residue of the chain, averages the eight readout heads, and pools each residue's atoms
    into its own N, CA, C, O, CB and R, where R is everything past CB.

    Returned alongside is the mean over the six backbone queries, which is the baseline the
    panel divides by: an atom that every query attends to says nothing about any one of
    them. Runs on the CPU, because nothing here is a reported number -- it is a picture of
    a trained model's attention, not a metric.
    """
    import torch

    from deepshift.atom_cloud import build_atom_cloud
    from deepshift.domain import ATOM_TYPES, NUCLEI
    from deepshift.prediction import checkpoint_recipe, load_checkpoint
    from deepshift.structure import parse_structure

    loaded = load_checkpoint(checkpoint)
    model = loaded.model.eval()
    built = build_atom_cloud(parse_structure(pdb, chain, keep_nonstandard=True),
                             recipe=checkpoint_recipe(loaded))
    atom_type, offset, xyz, mask = built.model_arrays
    names = np.array([*ATOM_TYPES, "PAD"])

    encoded = {}
    handle = model.norm.register_forward_hook(
        lambda _m, _i, out: encoded.__setitem__("h", out.detach()))
    with torch.no_grad():
        model(torch.from_numpy(atom_type), torch.from_numpy(offset),
              torch.from_numpy(xyz), torch.from_numpy(mask))
        handle.remove()
        h = encoded["h"]
        n_res, n_atoms, _ = h.shape
        n_shifts, heads, dh = model.n_shifts, model.rd_heads, model.rd_dh
        q = model.rq(model.nucleus_embed.weight).view(1, n_shifts, heads, dh)
        q = q.expand(n_res, -1, -1, -1).transpose(1, 2)
        k = model.rk(h).view(n_res, n_atoms, heads, dh).transpose(1, 2)
        centre = torch.from_numpy((offset == type(model).CENTER_OFFSET) & mask)
        logits = (q.float() @ k.float().transpose(-1, -2)) / np.sqrt(dh)
        logits = logits.masked_fill(~centre[:, None, None, :],
                                    torch.finfo(torch.float32).min)
        weights = logits.softmax(-1).mean(1).numpy()          # (residues, 64, atoms)

    pooled = np.zeros((n_shifts, len(RESIDUE_ATOMS)))
    for i in range(n_res):
        for column in np.where(centre[i].numpy())[0]:
            name = str(names[atom_type[i, column]])
            slot = RESIDUE_ATOMS.index(name) if name in RESIDUE_ATOMS[:-1] else -1
            pooled[:, slot] += weights[i, :, column]
    pooled /= n_res
    drawn = np.array([pooled[NUCLEI.index(nucleus)] for nucleus, _label
                      in BACKBONE_QUERIES])
    return drawn, drawn.mean(axis=0), n_res, n_shifts


CARD_FACE = "#fbfcfd"          # one card colour for both figures, and the link casing


def query_links(ax, drawn, baseline, background=CARD_FACE):
    """The six backbone queries on the left, a residue's atoms on the right, linked.

    A link is drawn where a query attends to an atom more than the mean of the six does,
    and its width is that excess. Raw weight would draw the same six links every time,
    because the carbonyl oxygen is the atom every query leans on.

    Two links sharing an endpoint would leave the same point at the same angle and fuse
    into one stroke, so each end fans: the links at a node are stacked on a short vertical
    run, ordered by the height of their far end, which is also what keeps them from
    crossing each other on the way in. Each link leaves and arrives horizontally, so no
    two ever run together.

    Crossings are drawn, not blended. Every link is laid down twice, once as a casing in
    the card's own colour and once as the link, and the order is fixed -- the pale links
    first, then the accent, each group thinnest first. Two strokes that meet therefore
    always show the same one in front, and no pair of pale strokes can pile up into a
    third shade that means nothing.
    """
    excess = drawn / baseline
    links = [(row, column, excess[row, column])
             for row in range(len(BACKBONE_QUERIES))
             for column in range(len(RESIDUE_ATOMS))
             if excess[row, column] > 1.0]
    left_x, right_x = 0.30, 0.74
    left_y = np.linspace(0.90, 0.10, len(BACKBONE_QUERIES))
    right_y = np.linspace(0.90, 0.10, len(RESIDUE_ATOMS))
    # The fan's step has to clear the widest stroke, or two links at one node touch.
    pitch = 0.042

    def fan(keyed, rank_by):
        """Where each link meets one node: stacked, centred, far end highest first."""
        offsets = {}
        for members in keyed.values():
            members = sorted(members, key=rank_by, reverse=True)
            for i, link in enumerate(members):
                # i == 0 is the far end that sits highest, so it takes the highest slot.
                offsets[link] = ((len(members) - 1) / 2 - i) * pitch
        return offsets

    by_query, by_atom = {}, {}
    for link in links:
        by_query.setdefault(link[0], []).append(link)
        by_atom.setdefault(link[1], []).append(link)
    start = fan(by_query, lambda link: right_y[link[1]])
    finish = fan(by_atom, lambda link: left_y[link[0]])

    strongest = {row: int(np.argmax(excess[row])) for row in range(len(BACKBONE_QUERIES))}
    span = right_x - left_x
    ordered = sorted(links, key=lambda link: (link[1] == strongest[link[0]], link[2]))
    for order, (row, column, value) in enumerate(ordered):
        y0 = left_y[row] + start[(row, column, value)]
        y1 = right_y[column] + finish[(row, column, value)]
        curve = mpl.path.Path([(left_x, y0), (left_x + 0.42 * span, y0),
                               (right_x - 0.42 * span, y1), (right_x, y1)],
                              [mpl.path.Path.MOVETO, mpl.path.Path.CURVE4,
                               mpl.path.Path.CURVE4, mpl.path.Path.CURVE4])
        width = min((value - 1.0) * 2.2, 2.3)
        accent = column == strongest[row]
        ax.add_patch(mpl.patches.PathPatch(
            curve, facecolor="none", capstyle="round", linewidth=width + 1.2,
            edgecolor=background, zorder=2 + 2 * order))
        ax.add_patch(mpl.patches.PathPatch(
            curve, facecolor="none", capstyle="round", linewidth=width,
            edgecolor=ACCENT if accent else INK["grid"], zorder=3 + 2 * order))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for row, (_nucleus, label) in enumerate(BACKBONE_QUERIES):
        ax.text(left_x - 0.045, left_y[row], label, ha="right", va="center", fontsize=6.8,
                fontweight="bold", color=ACCENT)
    for column, label in enumerate(RESIDUE_ATOMS):
        ax.text(right_x + 0.045, right_y[column], label, ha="left", va="center",
                fontsize=6.8, color=INK["secondary"])


def render_manifest():
    """The ray-traced panels' manifest."""
    return json.loads((RENDERS / "renders.json").read_text())


def render(ax, name):
    """One ray-traced panel, centred in its card at its own aspect ratio."""
    ax.imshow(plt.imread(RENDERS / f"{name}.png"), interpolation="antialiased")
    ax.set_axis_off()


def fig01_architecture(pdb=None, chain="A", resid=None, cutoff=8.0):
    """The method overview: six stations, left to right, one card each.

    Every station is the same card and every label sits below it, in the gutter, so
    nothing has to be proportioned by hand and no picture can reach the text:

      * every station is the SAME card -- one size, one hairline border, one baseline, so
        nothing has to be proportioned by hand;
      * the label and the spec sit BELOW the card, in the gutter, where no picture can
        reach them;
      * a card holds a picture and nothing else.

    Each picture is rendered from something real -- ribbon and cloud from a deposited
    structure, the distance matrix from that cloud, the embedding from the trained
    checkpoint, the output spectrum from observed shifts.
    """
    from Bio.PDB import PDBParser

    renders = render_manifest()
    pdb = pdb or (REFERENCE / "1POH.pdb")
    structure = PDBParser(QUIET=True).get_structure("x", str(pdb))
    residues = [r for r in structure[0][chain] if r.id[0] == " " and "CA" in r]
    target = residues[len(residues) // 2] if resid is None else \
        next(r for r in residues if r.id[1] == resid)
    # The distance matrix is computed here; the cloud drawn beside it was ray traced
    # earlier. If the two ever stopped describing the same neighbourhood, the pair of
    # cards would be a lie, so the manifest is checked against this card's own arguments.
    # The cloud the model is handed (deepshift.atom_cloud): every heavy atom of each
    # residue with any heavy atom within the cutoff of the target residue.
    from deepshift.atom_cloud import FINAL_ATOM_CLOUD_RECIPE, target_cloud
    from deepshift.structure import parse_structure
    if FINAL_ATOM_CLOUD_RECIPE.heavy_cutoff != cutoff:
        raise SystemExit(f"this card is {cutoff} A; the model's cloud is "
                         f"{FINAL_ATOM_CLOUD_RECIPE.heavy_cutoff} A")
    cloud = target_cloud(parse_structure(pdb, chain, keep_nonstandard=True), target.id[1])
    drawn = renders["atom_cloud"]
    if (drawn["target"] != f"{target.get_resname()}{target.id[1]}"
            or drawn["cutoff_angstrom"] != cutoff or drawn["n_atoms"] != len(cloud)):
        raise SystemExit(f"the rendered cloud is {drawn['n_atoms']} atoms, "
                         f"{drawn['cutoff_angstrom']} A around {drawn['target']}; this card "
                         f"is {len(cloud)} atoms, {cutoff} A around "
                         f"{target.get_resname()}{target.id[1]}; re-run "
                         "figures/render_structures.py")
    sub = np.array([point for _r, _n, point in cloud], dtype=float)
    centre = np.asarray(target["CA"].coord, dtype=float)

    grid, _phi, bias, _atom_embed, centres = _learned_geometry(FOLD1_CHECKPOINT)
    query_weights, query_baseline, _n_residues, n_shifts = readout_attention(
        FOLD1_CHECKPOINT, pdb, chain)
    # The spectrum card needs the observed shift of each predicted nucleus, which every
    # member dump carries identically: the deposited labels of the held-out rows.
    member = frozen.member_dumps(EVALUATION_FOLD, (1,))[0]
    observed = evaluate.read_predictions(member)
    peaks = np.array([observed.true_ppm[observed.mask[:, j], j].mean()
                      for j in range(observed.true_ppm.shape[1])])

    fig = plt.figure(figsize=(WIDTH["text"], 2.42))

    # Six equal cards on one baseline. Everything below is derived from these four
    # numbers, so the layout cannot drift out of proportion.
    N = 6
    PW, GAP = 0.1355, 0.0294
    X0 = (1.0 - (N * PW + (N - 1) * GAP)) / 2
    XS = [X0 + i * (PW + GAP) for i in range(N)]
    PY, PH = 0.335, 0.615                       # card bottom, card height
    TITLE_Y, SPEC_Y = PY - 0.075, PY - 0.185    # label gutter, below every card

    def card(x):
        fig.patches.append(mpl.patches.FancyBboxPatch(
            (x, PY), PW, PH, boxstyle="round,pad=0,rounding_size=0.012",
            transform=fig.transFigure, facecolor=CARD_FACE,
            edgecolor=INK["axis"], linewidth=0.8, zorder=1))

    def label(x, title, spec):
        # Plain text only -- .upper() would mangle any mathtext in a title.
        assert "$" not in title, f"card title must not contain mathtext: {title!r}"
        fig.text(x + PW / 2, TITLE_Y, title.upper(), ha="center", va="top",
                 fontsize=7.6, fontweight="bold", color=INK["secondary"],
                 zorder=4)
        fig.text(x + PW / 2, SPEC_Y, spec, ha="center", va="top",
                 fontsize=7.6, color=INK["muted"], zorder=4, linespacing=1.25)

    def inset(x, pad_x=0.010, pad_y=0.035, w=None, h=None, dy=0.0):
        ax = fig.add_axes([x + pad_x, PY + pad_y + dy,
                           (w if w is not None else PW - 2 * pad_x),
                           (h if h is not None else PH - 2 * pad_y)], zorder=3)
        ax.set_axis_off()
        ax.patch.set_alpha(0)
        return ax

    def chevrons(x, n=2, w=0.0068, hh=0.020, gap=0.0092):
        """The connector: light chevrons, not arrows -- an arrowhead at this size
        reads as a blob, and there is nothing branching here to need one."""
        y = PY + PH / 2
        for k in range(n):
            xa = x + k * gap
            fig.add_artist(plt.Line2D(
                [xa, xa + w, xa], [y + hh / 2, y, y - hh / 2],
                lw=1.25, color=INK["axis"], solid_capstyle="round",
                transform=fig.transFigure, zorder=2))

    for x in XS:
        card(x)
    for xa in XS[:-1]:
        chevrons(xa + PW + (GAP - 0.0160) / 2)

    # 1 -- the deposited structure
    render(inset(XS[0], pad_x=0.004, pad_y=0.018), "structure")
    label(XS[0], "Structure", "experimental PDB\nor AlphaFold")

    # 2 -- the 8 A heavy-atom cloud around one residue
    render(inset(XS[1], pad_x=0.004, pad_y=0.018), "atom_cloud")
    label(XS[1], f"{cutoff:.0f} \u00c5 atom cloud",
          f"{len(cloud)} heavy atoms\naround {target.get_resname()}{target.id[1]}")

    # 3 -- the two channels the cloud feeds: what the model learned to do with a
    # distance, drawn over the distances it is given. The atom-type embedding used to sit
    # here as a two-component scatter; it was removed because 51 of its 88 rows are
    # hydrogen types this heavy-atom model never sees, and they still sit at their
    # initialisation (mean norm 19.63 against the 19.60 a random 384-vector gives).
    half = (PH - 0.052) / 2

    def toward_white(color, amount):
        rgb = np.array(mpl.colors.to_rgb(color))
        return tuple(rgb + (1.0 - rgb) * amount)

    axe = fig.add_axes([XS[2] + 0.018, PY + half + 0.040, PW - 0.036, half - 0.014],
                       zorder=3)
    axe.patch.set_alpha(0)
    for head in range(bias.shape[1]):
        axe.plot(grid, bias[:, head], lw=0.8, zorder=2 + head,
                 color=toward_white(ACCENT, 0.08 + 0.62 * head / (bias.shape[1] - 1)))
    axe.axhline(0, color=INK["axis"], lw=0.6, zorder=1)
    # No tick labels: at this measure they would have to be set below 7pt. The range
    # they would carry is in the card's spec instead.
    axe.set_xlim(0, float(centres.max()) + 0.2)
    axe.margins(y=0.10)          # the tallest head must not touch the card's border
    axe.set_xticks([])
    axe.set_yticks([])
    for side in ("top", "right", "left"):
        axe.spines[side].set_visible(False)
    axe.spines["bottom"].set_color(INK["axis"])

    axd = fig.add_axes([XS[2] + 0.030, PY + 0.030, PW - 0.060, half - 0.020], zorder=3)
    ramp = mpl.colors.LinearSegmentedColormap.from_list(
        "geom", ["#062f38", ACCENT, "#a5dbe6", INK["surface"]])
    order = np.argsort(np.linalg.norm(sub - centre, axis=1))
    # A symmetric matrix is drawn square, so the aspect is fixed rather than automatic.
    axd.imshow(np.linalg.norm(sub[order, None, :] - sub[None, order, :], axis=-1),
               cmap=ramp, interpolation="nearest", aspect="equal")
    axd.set_xticks([])
    axd.set_yticks([])
    for sp in axd.spines.values():
        sp.set_color(INK["axis"])
        sp.set_linewidth(0.5)
    label(XS[2], "Distances",
          f"learned bias, {bias.shape[1]} heads\n" + r"all pairs $d_{ab}$, RBF "
          f"{centres.min():.0f}\u2013{centres.max():.0f} " + r"$\AA$")

    # 4 -- one real encoder block (model.py AtomBlock)
    cx = XS[3] + PW / 2
    # A card is 0.88in wide here, so a full spelling does not fit at a legible size:
    # "self-attention" needs ~49pt inside a ~54pt box. The block keeps the structure of
    # the encoder and shortens the words to match the measure.
    bw = PW - 0.030

    def tbox(cy, h, lines, hi=False):
        fig.patches.append(mpl.patches.FancyBboxPatch(
            (cx - bw / 2, cy - h / 2), bw, h,
            boxstyle="round,pad=0,rounding_size=0.008",
            transform=fig.transFigure,
            facecolor="#e0f2f7" if hi else INK["surface"],
            edgecolor=ACCENT if hi else INK["axis"],
            linewidth=0.9 if hi else 0.7, zorder=4))
        for i, (txt, fs) in enumerate(lines):
            dy = (len(lines) - 1) * 0.026 / 2 - i * 0.026
            fig.text(cx, cy + dy, txt, ha="center", va="center", fontsize=fs,
                     fontweight="bold" if hi else "normal",
                     color=ACCENT if hi else INK["secondary"], zorder=5)

    # Every y below is inside [PY, PY+PH], so no plate and no residual arrow can leave
    # the card or cross the card's title.
    y_in = PY + PH - 0.040
    y_n1, y_at, y_p1 = 0.858, 0.772, 0.694
    y_n2, y_ff, y_p2, y_x6 = 0.632, 0.566, 0.492, 0.418
    fig.add_artist(plt.Line2D([cx, cx], [y_x6, y_in], lw=0.9, color=INK["axis"],
                              transform=fig.transFigure, zorder=3))
    for ya, yb in ((y_in, y_p1), (y_p1 - 0.018, y_p2)):
        sk = bw / 2 + 0.008          # the residual skip, around the outside
        fig.add_artist(plt.Line2D([cx, cx - sk, cx - sk, cx], [ya, ya, yb, yb],
                                  lw=0.8, color=INK["axis"],
                                  transform=fig.transFigure, zorder=3))
    for cy in (y_p1, y_p2):
        fig.text(cx, cy, "+", ha="center", va="center", fontsize=8,
                 fontweight="bold", color=INK["muted"], zorder=6,
                 bbox={"boxstyle": "circle,pad=0.16", "facecolor": INK["surface"],
                       "edgecolor": INK["axis"], "linewidth": 0.7})
    tbox(y_n1, 0.048, [("norm", 7.0)])
    tbox(y_at, 0.092, [("attention", 6.8), ("+ distance bias", 5.6)], hi=True)
    tbox(y_n2, 0.048, [("norm", 7.0)])
    tbox(y_ff, 0.052, [("feed-forward", 6.4)])
    fig.patches.append(mpl.patches.FancyBboxPatch(
        (cx - 0.024, y_x6 - 0.026), 0.048, 0.052,
        boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
        facecolor="#eef2f7", edgecolor=INK["axis"], linewidth=0.7, zorder=4))
    fig.text(cx, y_x6, r"$\times\,6$", ha="center", va="center", fontsize=7.2,
             fontweight="bold", color=INK["secondary"], zorder=5)
    label(XS[3], "Transformer", "6 layers \u00b7 8 heads\n384-dim")

    # 5 -- the readout: one learned query per predicted nucleus, attending over the
    # atoms of the residue it predicts. The six backbone queries are drawn against the
    # atoms they read; the amide proton's query reads the nitrogen it hangs off, and
    # HA's reads the alpha carbon.
    query_links(inset(XS[4], pad_x=0.010, pad_y=0.030), query_weights,
                query_baseline)
    label(XS[4], "Nucleus queries",
          f"6 backbone of {n_shifts}\nlinks: where each looks")

    # 6 -- the predicted spectrum, high ppm on the left as a spectrum is drawn
    axp = inset(XS[5], pad_x=0.012, pad_y=0.070)
    rng = np.abs(np.sin(np.arange(len(peaks)) * 12.9898))
    axp.vlines(peaks, 0, 0.35 + 0.6 * rng, color=ACCENT, lw=0.85)
    axp.axhline(0, color=INK["axis"], lw=0.8)
    # Pad both ends so the extreme peaks do not sit on the card's border.
    axp.set_xlim(peaks.max() + 18, -16)
    axp.set_ylim(-0.06, 1.08)
    label(XS[5], "Chemical shifts", "one predicted\nspectrum (ppm)")

    save(fig, "fig_01_architecture")


# ------------------------------------------------------------- figure 2 -----

def fig02_accuracy():
    """Predicted against observed, one panel per backbone nucleus."""
    dumps = frozen.member_dumps(EVALUATION_FOLD)
    members = evaluate.ensemble([evaluate.read_predictions(path) for path in dumps])
    # The same fence the benchmark tables use. Without it this panel prints a C-prime
    # RMSE of 1.08 against Table 1's 0.97, on cells Methods says are excluded.
    members, _dropped = frozen.drop_anticorrelated(members)
    pred, true = members.pred_ppm, members.resolved_ppm
    mask, cols = members.effective_mask, list(members.shift_cols)
    # Height is set by the float, not by taste: this panel and fig_03 share one float
    # whose text block is 9in tall, so the two renders plus the caption must fit inside
    # it or LaTeX reports an Overfull \vbox and the caption spills onto the next page.
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH["text"], 4.15))
    # Density ramps off the accent, so the one figure of the paper that is entirely
    # DeepShift is unmistakably in DeepShift's colour. White at the low end because a
    # log-binned hexbin is mostly empty and the sparse bins must not read as data.
    ramp = mpl.colors.LinearSegmentedColormap.from_list(
        "accent", [INK["surface"], "#a5dbe6", ACCENT, "#062f38"])

    for ax, (col, _atom, label) in zip(axes.ravel(), BACKBONE):
        j = cols.index(col)
        sel = mask[:, j]
        obs, prd = true[sel, j], pred[sel, j]
        s = stats(prd, obs)

        lo, hi = min(obs.min(), prd.min()), max(obs.max(), prd.max())
        pad = 0.04 * (hi - lo)
        lim = (lo - pad, hi + pad)

        ax.hexbin(obs, prd, gridsize=90, extent=(*lim, *lim), bins="log",
                  cmap=ramp, mincnt=1, linewidths=0)
        # y = x is the truth line, so it wears the reference ink, not a hue. Dashed
        # and under the data: it is the thing being met, not a series competing with it.
        ax.plot(lim, lim, color=INK["primary"], lw=0.8, ls=(0, (4, 2.2)), zorder=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(f"{label}   RMSE {s['rmse']:.2f}   R {s['r']:.3f}", pad=3)
        ax.set_xlabel("Observed shift (ppm)")
        ax.set_ylabel("Predicted shift (ppm)")
        ax.text(0.045, 0.955, f"MAE {s['mae']:.2f} ppm\nn = {s['n']:,}",
                transform=ax.transAxes, va="top", fontsize=8.0,
                color=INK["secondary"])

    panel_letter(axes[0, 0], "a", dx=-0.16)
    fig.tight_layout(pad=0.5, w_pad=1.1, h_pad=0.9)
    save(fig, "fig_02_accuracy")


# ------------------------------------------------------------- figure 7 -----

def fig07_scaling():
    """Learning curve: held-out RMSE against the number of training rows.

    Six points, at 5, 10, 25, 50, 75 and 100 %: five deterministic nested subsamples of the
    training rows and the paper member itself at the full corpus. Every point holds out
    fold 1 and is scored on fold 6, so one identical cell set carries the whole curve and
    the last point is the shipped model's own number. Row counts are read from each
    point's training record rather than restated here.
    """
    points = ["fraction_005", "fraction_010", "fraction_025", "fraction_050",
              "fraction_075", "fraction_100"]
    # Four categorical slots, not arm identities: this panel separates nuclei, and every
    # series on it is DeepShift. style.PALETTE is the slot table the house style defines
    # for exactly this case.
    s1, s2, s3, s4 = PALETTE["slot1"], PALETTE["slot2"], PALETTE["slot3"], PALETTE["slot4"]
    heavy = [("n_shift", r"$^{15}$N", s1), ("ca_shift", r"$^{13}$C$\alpha$", s2),
             ("cb_shift", r"$^{13}$C$\beta$", s3), ("c_shift", r"$^{13}$C$^{\prime}$", s4)]
    light = [("h_shift", r"$^{1}$H$^{N}$", s1), ("ha_shift", r"$^{1}$H$\alpha$", s2)]

    curves, rows = {}, {}
    for name in points:
        dump = SCALING_LADDER / name / "fold6.preds.npz"
        rows[name] = training_rows(SCALING_LADDER / name)
        # The same fence every fold-6 score uses (Methods): without it the 100% point
        # prints a C-prime RMSE of 1.10 against Figure 2(a)'s 0.96.
        preds, _dropped = frozen.drop_anticorrelated(evaluate.read_predictions(dump))
        for col, _lab, _c in heavy + light:
            scored = evaluate.score(preds, columns=[col], boots=0)[col]
            curves.setdefault(col, {})[rows[name]] = scored["rmse"]

    fig, (ax_c, ax_h) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, group in ((ax_c, heavy), (ax_h, light)):
        for col, lab, color in group:
            xs = sorted(curves[col])
            ax.plot(xs, [curves[col][x] for x in xs], "-o", color=color,
                    lw=1.1, ms=3.4, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("Training rows")
        ax.set_ylabel("RMSE (ppm)")
        framed_legend(ax, loc="upper right", ncol=2, columnspacing=0.8)
        ax.grid(lw=0.4, color="0.85")
        ax.set_axisbelow(True)

    panel_letter(ax_c, "a", dx=-0.15)
    panel_letter(ax_h, "b", dx=-0.15)
    fig.tight_layout(pad=0.5, w_pad=1.4)
    save(fig, "fig_07_scaling")


def training_rows(point: Path) -> int:
    """The rows one ladder point trained on, read from the trainer's own record.

    ``train_used`` is counted after the fence and the fraction, so it is the number the
    x axis means; a fraction restated here would drift from the run the moment the
    corpus changed.
    """
    record = json.loads((point / "metrics.json").read_text())
    return int(record["manifest"]["samples"]["train_used"])


# ------------------------------------------------------------- figure 8 -----

def fig08_ramachandran(min_cells=30, step=5):
    """Prediction error as a surface over (phi, psi), after SPARTA+ Eq. 4.

    Pooled over the out-of-fold predictions of all five cross-validation members, which
    is an order of magnitude more cells than fold 6 alone and is what supports 5-degree
    bins. The cells and their torsions are ``benchmarks/dump_cv_torsions.py``'s.
    """
    with np.load(RAMACHANDRAN_CELLS, allow_pickle=True) as z:
        cv = {k: z[k] for k in z.files}
    names = [str(s) for s in cv["atom_names"]]

    edges = np.arange(-180, 180 + step, step)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3))

    for ax, (col, _atom, label) in zip(axes.ravel(), BACKBONE):
        sel = cv["atom"] == names.index(col)
        phi, psi = cv["phi"][sel], cv["psi"][sel]
        err = cv["abs_err"][sel]
        count, _, _ = np.histogram2d(phi, psi, bins=[edges, edges])
        total, _, _ = np.histogram2d(phi, psi, bins=[edges, edges], weights=err)
        with np.errstate(invalid="ignore", divide="ignore"):
            mae = total / count
        surface = np.ma.masked_where(count < min_cells, mae)

        # Colour on the 2nd-98th percentile of the drawn bins: at 5 degrees the
        # sparsest corners of the plane would otherwise set the whole scale.
        drawn = surface.compressed()
        mesh = ax.pcolormesh(edges, edges, surface.T, cmap="viridis",
                             vmin=np.percentile(drawn, 2),
                             vmax=np.percentile(drawn, 98), rasterized=True)
        cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046, extend="both")
        cb.ax.tick_params(labelsize=6, width=0.6)
        cb.outline.set_linewidth(0.6)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_aspect("equal")
        ax.set_title(f"{label}   MAE {err.mean():.2f} ppm", pad=3)
        ax.set_xlabel(r"$\phi$ (deg)")
        ax.set_ylabel(r"$\psi$ (deg)")

    fig.tight_layout(pad=0.5, w_pad=0.9, h_pad=0.9)
    save(fig, "fig_08_ramachandran")


if __name__ == "__main__":
    house_style()
    print("building:")
    fig01_architecture()
    fig02_accuracy()
    fig07_scaling()
    fig08_ramachandran()
