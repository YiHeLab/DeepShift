"""Cache-free DeepShift inference from a PDB file."""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .atom_cloud import AtomCloudRecipe, build_atom_cloud
from .calibration import IntervalCalibration, halfwidth, load_calibration
from .chemistry import atom_from_shift_column
from .data.norm import get_center
from .data.structures import (
    partner_protein_residues_for,
    retained_insertion_partner_residues,
)
from .model import LoadedCheckpoint, load_checkpoint
from .normalization import denormalize
from .structure import is_hydrogen, parse_models, parse_structure


def checkpoint_recipe(loaded: LoadedCheckpoint) -> AtomCloudRecipe:
    """The atom recipe stamped in a checkpoint, with that checkpoint's vocabulary."""
    recipe = AtomCloudRecipe.from_mapping(loaded.recipe)
    return recipe.with_vocabulary(loaded.atom_pad_index, loaded.offset_pad_index)


@torch.no_grad()
def forward_z(checkpoint: LoadedCheckpoint, arrays, *, batch_size: int = 32) -> np.ndarray:
    """Batched z-space forward at training-eval precision: autocast on CUDA, fp32 on
    CPU, output cast to fp32, so inference and evaluate() return the same numbers."""
    at, off, xyz, mask = arrays
    device = next(checkpoint.model.parameters()).device
    out = []
    for start in range(0, len(at), batch_size):
        sl = slice(start, start + batch_size)
        ctx = torch.autocast("cuda") if device.type == "cuda" else nullcontext()
        with ctx:
            pred, _ = checkpoint.model(
                torch.from_numpy(at[sl]).to(device), torch.from_numpy(off[sl]).to(device),
                torch.from_numpy(xyz[sl]).to(device), torch.from_numpy(mask[sl]).to(device))
        out.append(pred.float().cpu().numpy())
    return np.concatenate(out)


def predict_pdb(pdb_path: str | Path, chain: str = "A",
                checkpoints: Iterable[str | Path | LoadedCheckpoint] = (), *,
                batch_size: int = 32,
                sample_partner_chains: Iterable[str] = (),
                return_intervals: bool = False,
                calibration: IntervalCalibration | dict | str | Path | None = None):
    """Predict every shift a checkpoint declares, per residue, straight from a PDB:
    parse once, build the checkpoint-recipe atom cloud, run each fold model, denormalize
    with that model's own stats, average in ppm. Cache-free and retrieval-free.

    ``return_intervals`` returns ``(frame, intervals)``. Intervals require an explicit
    fold-0 calibration artifact and exactly as many model members as it was fitted on.
    ``sample_partner_chains`` names the exact protein chains known to have been present in
    the NMR sample; all other coordinate chains are excluded."""
    loaded = [x if isinstance(x, LoadedCheckpoint) else load_checkpoint(x) for x in checkpoints]
    if not loaded:
        raise ValueError("at least one checkpoint is required")
    if any(x.shift_columns != loaded[0].shift_columns for x in loaded[1:]):
        raise ValueError("checkpoint shift-column mismatch")
    vocab = {(x.atom_pad_index, x.offset_pad_index) for x in loaded}
    if len(vocab) != 1:
        raise ValueError("checkpoint vocabulary mismatch")
    recipes = [checkpoint_recipe(x) for x in loaded]
    if any(r != recipes[0] for r in recipes[1:]):
        raise ValueError("checkpoint recipe mismatch")
    recipe = recipes[0]
    # Averaging in ppm requires every member to invert the same normalization; two
    # normalizations are two ppm scales, and their mean is neither.
    normalizations = {x.stats.get("normalization") for x in loaded}
    if len(normalizations) != 1:
        raise ValueError(f"checkpoint normalization mismatch: {sorted(map(str, normalizations))}")

    # Three rules, all mirroring the cache build so the cloud matches training.
    # 1. keep_nonstandard: a modified residue must own its author number, or an insertion
    #    sibling (10A next to a modified 10) takes the integer slot and is returned below
    #    as a retained partner, entering every neighbouring cloud twice.
    aa = parse_structure(pdb_path, chain, keep_nonstandard=True)  # drops hydrogens (no-H model)
    if aa is None:
        return None
    # 2. The integer residue contract keeps one residue per author number, so insertion
    #    siblings must be re-added as partners or their heavy atoms leave the neighbours'
    #    8 A clouds. {} without insertion codes, so those clouds stay bit-identical.
    siblings = {key: {**info, "atoms": {name: xyz for name, xyz in info["atoms"].items()
                                        if not is_hydrogen(name)}}
                for key, info in retained_insertion_partner_residues(pdb_path, chain).items()}
    # 3. Admit only the explicitly named protein chains that were present in the NMR
    #    sample. A structure-level yes/no switch would also admit crystal-lattice chains.
    partners = _named_sample_partner_residues_for(
        pdb_path, chain, sample_partner_chains, drop_hydrogens=True
    )
    siblings = {**partners, **siblings}
    built = build_atom_cloud(aa, recipe=recipe,
                             partner_residues=siblings or None)
    if built is None:
        return None
    truncated = truncated_residues(built, recipe)
    member_ppm = np.stack([denormalize(forward_z(
        x, built.model_arrays, batch_size=batch_size
    ), built.residue_codes, x.stats, x.shift_columns) for x in loaded])
    ppm = member_ppm.mean(axis=0)
    frame = pd.DataFrame({
        "residue_id": built.residue_ids,
        "residue_name": built.residue_names,
        "atom_cloud_truncated": truncated,
    })
    for i, column in enumerate(loaded[0].shift_columns):
        values = ppm[:, i].copy()
        absent = np.array([not nucleus_exists(n, column) for n in built.residue_names])
        values[absent] = np.nan
        frame[column] = values
    if return_intervals:
        fitted = load_calibration(calibration) if calibration is not None else None
        if fitted is not None and len(loaded) != len(fitted.member_folds):
            raise ValueError(f"interval calibration requires {len(fitted.member_folds)} "
                             f"members; got {len(loaded)}")
        intervals = (
            interval_table(
                member_ppm,
                built.residue_ids,
                built.residue_names,
                loaded[0].shift_columns,
                calibration=fitted,
            )
            if fitted is not None
            else None
        )
        return frame, intervals
    return frame


def truncated_residues(built, recipe: AtomCloudRecipe) -> np.ndarray:
    """Residues whose contact cloud held more atoms than the recipe's cap.

    The cloud keeps the atoms nearest the target residue and drops the rest. A truncated
    residue is predicted from an incomplete shell, so the output marks it and a warning
    names the count.
    """
    if recipe.m_atoms is None:
        return np.zeros(len(built.residue_ids), dtype=bool)
    over = np.asarray(built.precap_atom_count) > int(recipe.m_atoms)
    if over.any():
        warnings.warn(
            f"{int(over.sum())} of {over.size} residues have more than {recipe.m_atoms} heavy "
            f"atoms within {recipe.heavy_cutoff} A (largest {int(built.precap_atom_count.max())}); "
            f"the {recipe.m_atoms} nearest the target residue were kept for each",
            stacklevel=3,
        )
    return over


def _named_sample_partner_residues_for(
    pdb_path: str | Path,
    target_chain: str,
    sample_partner_chains: Iterable[str],
    *,
    drop_hydrogens: bool,
) -> dict:
    """Return only caller-declared protein partner chains, failing on any mismatch."""
    if isinstance(sample_partner_chains, str):
        sample_partner_chains = (sample_partner_chains,)
    requested = {str(value).strip() for value in sample_partner_chains}
    requested.discard("")
    target = str(target_chain).strip()
    if target in requested:
        raise ValueError(f"target chain {target_chain!r} cannot also be a sample partner")
    if not requested:
        return {}

    all_partners = partner_protein_residues_for(
        pdb_path, target_chain, drop_hydrogens=drop_hydrogens
    )
    available = {str(key[0]) for key in all_partners}
    missing = requested - available
    if missing:
        raise ValueError(
            "sample partner chain(s) are not protein chains in the selected structure "
            f"model: requested={sorted(missing)!r}, available={sorted(available)!r}"
        )
    return {key: value for key, value in all_partners.items()
            if str(key[0]) in requested}


# The six columns the shipped calibration constants cover. Side chains get no interval:
# members disagreeing on which of a prochiral pair a label names is a labelling
# artefact, not uncertainty.
BACKBONE_INTERVAL_COLUMNS = ("n_shift", "h_shift", "ca_shift", "cb_shift", "c_shift",
                             "ha_shift")


def interval_table(
    member_ppm: np.ndarray,
    resids,
    names,
    shift_columns,
    *,
    calibration: IntervalCalibration | dict | str | Path,
) -> pd.DataFrame:
    """Long-form 90% prediction intervals, one row per chemically-present backbone cell.

    Spread is the ddof=1 SD across members, the definition the constants were fitted
    on; the interval is prediction +/- ci90_halfwidth_ppm."""
    sd = member_ppm.std(axis=0, ddof=1)
    blocks = []
    for i, column in enumerate(shift_columns):
        if column not in BACKBONE_INTERVAL_COLUMNS:
            continue
        present = np.array([nucleus_exists(n, column) for n in names])
        blocks.append(pd.DataFrame({
            "residue_id": np.asarray(resids)[present],
            "residue_name": np.asarray(names)[present],
            "nucleus": atom_from_shift_column(column),
            "ensemble_sd_ppm": sd[present, i]}))
    table = (pd.concat(blocks, ignore_index=True)
             .sort_values(["residue_id", "nucleus"], ignore_index=True))
    table["ci90_halfwidth_ppm"] = halfwidth(
        table["nucleus"], table["ensemble_sd_ppm"], calibration=calibration
    )
    return table


# Chemically-absent (residue, nucleus) cells, blanked in every output; the head emits
# all 64 columns, so glycine CB would otherwise print the denormalized global mean. The
# base rule is the per-(residue, nucleus) center table, plus two corrections counted on
# real training labels: NEVER_LABELLED pairs the table admits (methyl pseudo-atom alias
# or artifact) but that carry zero labels, and ALSO_REAL lysine zeta-ammonium protons,
# which are labelled but absent from the table.
NEVER_LABELLED = {("GLY", "HA"), ("PRO", "H"), ("THR", "HG11"), ("THR", "HG12"),
                  ("THR", "HG13"), ("ILE", "HG11"), ("ILE", "HD21"), ("ILE", "HD22"),
                  ("ILE", "HD23")}
ALSO_REAL = {("LYS", "HZ2"), ("LYS", "HZ3")}


def nucleus_exists(residue_name: str, column: str) -> bool:
    pair = (residue_name, atom_from_shift_column(column))
    if pair in NEVER_LABELLED:
        return False
    return pair in ALSO_REAL or get_center(residue_name, column) is not None


def _no_residues_reason(pdb_path, chain: str) -> str:
    """Why the parse came back empty: wrong chain id, mmCIF and an unreadable file all
    give the same empty result, so the message names the cause, not "no usable residues"."""
    with open(pdb_path, errors="replace") as stream:
        head = stream.read(4096)
    if head.lstrip().startswith("data_") or "_atom_site." in head:
        return "this is mmCIF; DeepShift reads PDB format (files.rcsb.org/download/XXXX.pdb)"
    present = sorted({info["auth_chain_id"] or "(blank)"
                      for model in parse_models(pdb_path) for info in model.values()})
    if not present:
        return "no ATOM/HETATM records found"
    if chain not in present:
        return f"chain {chain!r} is not in the file; chains present: {present}"
    return f"chain {chain!r} has no standard amino-acid residue with a CA"


def predict_command(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    loaded = [load_checkpoint(path, device=device) for path in args.checkpoint]
    calibration = args.interval_calibration
    if calibration is not None:
        # A wrong member count reports the "no intervals" message below rather than
        # predict_pdb's refusal.
        fitted = load_calibration(calibration)
        if len(loaded) != len(fitted.member_folds):
            calibration = None
    result = predict_pdb(args.pdb, args.chain, loaded, batch_size=args.batch_size,
                         sample_partner_chains=args.sample_partner_chain,
                         return_intervals=True, calibration=calibration)
    if result is None:
        raise RuntimeError(f"no usable residues in {args.pdb}: "
                           f"{_no_residues_reason(args.pdb, args.chain)}")
    frame, intervals = result
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    if intervals is not None:
        sidecar = out.with_suffix(".intervals.csv")
        intervals.to_csv(sidecar, index=False)
        print(f"wrote {sidecar}: 90% prediction intervals for the backbone nuclei "
              f"(prediction +/- ci90_halfwidth_ppm)")
    else:
        reason = (
            "no --interval-calibration artifact was supplied"
            if args.interval_calibration is None
            else f"the artifact requires five members; got {len(loaded)} checkpoint(s)"
        )
        print(f"no intervals sidecar: {reason}")
        # A stale interval file would pair these predictions with half-widths fitted for
        # other members.
        stale = out.with_suffix(".intervals.csv")
        if stale.exists():
            stale.unlink()
            print(f"removed {stale}: it belonged to the predictions this run replaced")
    return frame


def add_predict_arguments(parser):
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--interval-calibration",
        help="fold-0 variance/conformal artifact for the five-member model",
    )
    parser.add_argument(
        "--sample-partner-chain",
        action="append",
        default=[],
        metavar="CHAIN",
        help=("include this exact protein chain because it was present in the NMR sample; "
              "repeat for each sample partner"),
    )
    parser.add_argument("--out", required=True)
    parser.set_defaults(handler=predict_command)
