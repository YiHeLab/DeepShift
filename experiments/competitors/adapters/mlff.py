"""MLFF adapter: MACE-OFF23 descriptors of a relaxed chain, the released heads, ppm.

MLFF (Bojan et al., arXiv:2505.23354) feeds the frozen MACE-OFF23-large descriptor of a 5 A
local environment to one graph-network head per nucleus (N CA C H HA CB CG CD CD2 CG2 CE).
The model, the environments and the heads are the authors' own, imported from their
repository; only the descriptor call is replaced by a force-free equivalent that batches
identical cut-outs, which does not change a descriptor.

The input is an Amber99SB-relaxed chain (``relax.py``). REDUCE ``-build`` is run on it without
``-Trim``: it flips Asn/Gln/His where needed and keeps the hydrogens the relaxation placed.
The heads predict a secondary shift; the absolute shift adds a random-coil constant, from the
authors' ``utils/get_rc_shifts`` for the six backbone nuclei and from ``RC_SIDECHAIN`` for the
side-chain nuclei, for which their code ships none (the table is recovered from the
``rc_<atom>`` columns of their released training targets, ``target_fixed.zip``: a pure
function of residue type, 28 (residue, atom) keys, none multi-valued).

Reads ``MLFF_HOME`` (the repository checkout holding ``MLFF_representation/`` and
``models/all_cs_models/cs_mace``), ``MACE_MODEL`` (default
``~/.cache/mace/MACE-OFF23_large.model``) and ``REDUCE_BIN``.
"""
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import resname_map, tool_path

RMAX = 5.0         # environment radius, A
BACKBONE = {'N', 'CA', 'C', 'H', 'HA', 'CB'}
RC_SIDECHAIN = {
    'CG': {'ARG': 27.2, 'ASN': 177.0, 'ASP': 179.7, 'GLN': 33.8, 'GLU': 36.3,
           'HIS': 133.53, 'LEU': 27.0, 'LYS': 24.8, 'MET': 32.1, 'PHE': 138.8,
           'PRO': 27.4, 'TRP': 111.7, 'TYR': 129.69},
    'CD': {'ARG': 43.4, 'GLN': 180.3, 'GLU': 183.6, 'LYS': 29.1, 'PRO': 50.6},
    'CD2': {'HIS': 119.9, 'LEU': 23.5, 'PHE': 131.7, 'TRP': 126.5, 'TYR': 132.85},
    'CG2': {'ILE': 17.5, 'THR': 21.6, 'VAL': 21.2},
    'CE': {'LYS': 42.1, 'MET': 17.0},
}


def build_predictor(device):
    """-> (predictor, descriptor function, authors' get_rc_shifts, load_struct, struct_to_dataframe).

    The authors' ``CSPredictor`` loads MACE-OFF23-large and every head found under
    ``models/all_cs_models/cs_mace/<atom>/``.
    """
    home = Path(tool_path('MLFF_HOME'))
    repo = home / 'MLFF_representation'
    heads = home / 'models' / 'all_cs_models' / 'cs_mace'
    model = Path(os.environ.get('MACE_MODEL')
                 or Path.home() / '.cache' / 'mace' / 'MACE-OFF23_large.model')
    sys.path.insert(0, str(repo))
    warnings.filterwarnings('ignore')
    from cs_predict import CSPredictor
    from utils.get_rc_shifts import get_rc_shifts
    from utils.process_data import load_struct, struct_to_dataframe

    paths = sorted(str(p) for atom in sorted(os.listdir(heads)) if (heads / atom).is_dir()
                   for p in (heads / atom).glob('cs_mace_with_clip_*_epoch_2000'))
    if not paths:
        raise SystemExit(f'no trained heads under {heads}')
    predictor = CSPredictor(mace_model_path=str(model) if model.is_file() else None,
                            cs_model_paths=paths, device=device)
    return (predictor, make_descriptor_fn(predictor.descriptors_model), get_rc_shifts,
            load_struct, struct_to_dataframe)


def make_descriptor_fn(calc):
    """MACE invariant descriptors without the force autograd graph ``get_descriptors`` builds."""
    from e3nn import o3
    from mace.modules.utils import extract_invariant

    model = calc.models[0]
    irreps = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps.lmax
    n_invariant = irreps.dim // (l_max + 1) ** 2
    n_layers = int(model.num_interactions)
    per_layer = [irreps.dim] * n_layers
    per_layer[-1] = n_invariant               # the last layer has no equivariant features
    to_keep = int(np.sum(per_layer[:n_layers]))

    def run(atoms):
        batch = calc._atoms_to_batch(atoms)
        with torch.no_grad():
            feats = model(batch.to_dict(), compute_force=False)['node_feats']
            feats = extract_invariant(feats, num_layers=n_layers,
                                      num_features=n_invariant, l_max=l_max)
            return feats[:, :to_keep].detach().cpu().numpy()
    return run


def receptive_field(predictor):
    """How far apart two cut-outs must sit to be independent: r_max x message-passing layers."""
    model = predictor.descriptors_model.models[0]
    return float(getattr(model, 'r_max', 5.0)) * int(getattr(model, 'num_interactions', 2))


def pack(records, separation):
    """Cut-outs on a cubic grid, ``separation`` apart: (ase.Atoms, offset of each cut-out)."""
    from ase import Atoms

    side = math.ceil(len(records) ** (1 / 3))
    max_span = max(float(np.ptp(np.stack(r['env']['coords'].to_numpy()), axis=0).max())
                   for r in records)
    stride = max_span + separation + 2.0
    symbols, coords, offsets, cursor = [], [], [], 0
    for k, rec in enumerate(records):
        xyz = np.stack(rec['env']['coords'].to_numpy()).astype(np.float64)
        xyz -= (xyz.max(0) + xyz.min(0)) / 2
        xyz += np.array([k % side, (k // side) % side, k // (side * side)],
                        dtype=np.float64) * stride
        offsets.append(cursor)
        symbols.extend(rec['env']['element'].tolist())
        coords.append(xyz)
        cursor += len(xyz)
    packed = np.concatenate(coords, axis=0)
    # MACE measures distances in float32, so a box centred on the origin keeps precision
    # that one pushed out to large coordinates loses.
    packed -= packed.mean(0)
    return Atoms(symbols=symbols, positions=packed), offsets


def cutout_key(env):
    """Byte identity of an ordered cut-out: same atoms, same order, same coordinates."""
    rid = np.asarray(env['res_id'].to_numpy(), dtype=np.int64)
    xyz = np.stack(env['coords'].to_numpy()).astype(np.float32, copy=False)
    return rid.tobytes(), xyz.tobytes(), tuple(env['element'].tolist())


def descriptors(predictor, run, frame, batch_size):
    """{res_id: {atom: descriptor}} for every residue; one forward per ``batch_size`` cut-outs.

    Identical cut-outs share a forward. The target-atom row is selected as the authors'
    ``get_descriptors`` does it, including the ``indices[0] - env.index[0]`` offset that
    makes a descriptor row correspond to its atom.
    """
    environments = predictor.extract_environments(frame, rmax=RMAX)
    groups = {}
    for res_id, env in environments.items():
        groups.setdefault(cutout_key(env), {'env': env, 'targets': []})['targets'].append(res_id)
    records = list(groups.values())
    separation = receptive_field(predictor) if batch_size > 1 else 0.0

    out = {}
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        if len(chunk) == 1:
            result = run(predictor.compute_atoms(chunk[0]['env']))
            offsets = [0]
        else:
            packed, offsets = pack(chunk, separation)
            result = run(packed)
        for rec, offset in zip(chunk, offsets):
            env = rec['env']
            for res_id in rec['targets']:
                per_atom = {}
                for atom_type in predictor.atom_types:
                    idx = env[(env['res_id'] == res_id)
                              & (env['atom'] == atom_type)].index.tolist()
                    if not idx:
                        continue
                    row = offset + (idx[0] - env.index[0])
                    if row < len(result):
                        per_atom[atom_type] = torch.tensor(
                            result[row], dtype=torch.float32, device=predictor.device)
                out[res_id] = per_atom
    return out


def polish_hydrogens(relaxed_pdb, out_pdb):
    """REDUCE ``-build`` without ``-Trim``: flip optimisation that keeps the existing H."""
    reduce_bin = os.environ.get('REDUCE_BIN', 'reduce')
    with open(out_pdb, 'w') as fh:
        subprocess.run([reduce_bin, '-build', '-Quiet', str(relaxed_pdb)],
                       stdout=fh, stderr=subprocess.DEVNULL, check=False)
    if not (Path(out_pdb).exists() and Path(out_pdb).stat().st_size):
        raise RuntimeError(f'REDUCE produced no output for {relaxed_pdb}')


def to_ppm(secondary, frame, relaxed_pdb, get_rc_shifts):
    """Absolute ppm per author residue number, with the secondary shift under ``<ATOM>_secondary``.

    Their parser keys residues on the number alone, so an insertion-code sibling shares a
    key with its neighbour: the first residue seen keeps it.
    """
    residues = frame[['res_id', 'name']].drop_duplicates()
    residues = residues.drop_duplicates('res_id', keep='first').sort_values('res_id')
    rc = get_rc_shifts(''.join(residues['name']), res_ids=list(residues['res_id']),
                       format_1letter=True).set_index('res_id')
    names = resname_map(relaxed_pdb)
    preds = {}
    for key, shifts in secondary.items():
        res_id = int(key)
        rid = str(res_id)
        resname = names.get(rid) or ''
        cell = {'resname': resname}
        for atom, value in shifts.items():
            value = float(value)
            if not math.isfinite(value):
                continue
            cell[atom + '_secondary'] = value
            if atom in BACKBONE:
                base = rc[atom].get(res_id) if res_id in rc.index else None
            else:
                base = RC_SIDECHAIN.get(atom, {}).get(resname)
            if base is None or base != base:
                continue          # the nucleus does not exist for this residue (PRO H, GLY CB)
            cell[atom] = value + float(base)
        if len(cell) > 1:
            preds[rid] = cell
    return preds


_PREDICTOR = None


def predict(relaxed_pdb, workdir, batch_size=8, device=None):
    """Predictions for one relaxed chain: ``{author resnum: {'resname', '<ATOM>', ...}}``."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = build_predictor(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    predictor, run, get_rc_shifts, load_struct, struct_to_dataframe = _PREDICTOR

    prepared = Path(workdir) / 'prepared.pdb'
    polish_hydrogens(relaxed_pdb, prepared)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        frame = struct_to_dataframe(load_struct(str(prepared)))
    secondary = predictor.predict_cs(descriptors(predictor, run, frame, batch_size))
    secondary = {str(int(k)): v for k, v in secondary.items()}
    preds = to_ppm(secondary, frame, relaxed_pdb, get_rc_shifts)
    if not preds:
        raise RuntimeError('no predictions')
    return preds
