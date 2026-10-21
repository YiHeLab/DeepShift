"""The MLFF/MACE chemical-shift predictor of Bojan et al. (arXiv:2505.23354), driven here.

The released ``CSPredictor`` computes every MACE descriptor twice and builds a force graph
it never reads. ``descriptors_fast`` takes the same environments and the same target-atom
rows from the predictor's own methods and runs only the invariant forward pass, with
cut-outs that repeat byte for byte computed once and several cut-outs packed into one
forward. ``build_predictor`` checks the force-free path against the library's on a random
cluster before use.

    MLFF_HOME    checkout of https://github.com/mb012/MLFF_representation
    MLFF_MODELS  the ``cs_mace`` directory of the released heads (one folder per atom)
    REDUCE_BIN   REDUCE executable

The MACE-OFF23 large descriptor model is downloaded by ``mace.calculators.mace_off``.
"""
from __future__ import annotations

import contextlib
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch


def env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"set {name}")
    return Path(value)


@contextlib.contextmanager
def inside(repo: Path):
    """Their modules resolve ``data/`` relative to their root."""
    previous = Path.cwd()
    os.chdir(repo)
    try:
        yield
    finally:
        os.chdir(previous)


def make_descriptor_fn(calc):
    """Invariant MACE descriptors without the force autograd graph."""
    from e3nn import o3
    from mace.modules.utils import extract_invariant

    model = calc.models[0]
    irreps = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps.lmax
    n_invariant = irreps.dim // (l_max + 1) ** 2
    n_layers = int(model.num_interactions)
    per_layer = [irreps.dim] * n_layers
    per_layer[-1] = n_invariant        # the last layer has no equivariant features
    to_keep = int(np.sum(per_layer[:n_layers]))

    def run(atoms):
        batch = calc._atoms_to_batch(atoms)
        with torch.no_grad():
            feats = model(batch.to_dict(), compute_force=False)["node_feats"]
            feats = extract_invariant(feats, num_layers=n_layers,
                                      num_features=n_invariant, l_max=l_max)
            return feats[:, :to_keep].detach().cpu().numpy()
    return run


def verify_descriptor_fn(calc, run, tol: float = 1e-4) -> None:
    """The force-free path must equal the library path on a random cluster."""
    from ase import Atoms

    rng = np.random.default_rng(0)
    symbols = ["C", "N", "O", "H"] * 12
    xyz = rng.normal(scale=3.0, size=(len(symbols), 3)).astype(np.float64)
    atoms = Atoms(symbols=symbols, positions=xyz - xyz.mean(0))
    reference = np.asarray(calc.get_descriptors(atoms, invariants_only=True, num_layers=-1))
    error = float(np.abs(reference - run(atoms)).max())
    if error > tol:
        raise RuntimeError(f"force-free descriptors differ from the library by {error:.3e}")


def build_predictor(device: str):
    """-> (predictor, force-free descriptor function)."""
    repo, heads = env_path("MLFF_HOME"), env_path("MLFF_MODELS")
    sys.path.insert(0, str(repo))
    warnings.filterwarnings("ignore")
    with inside(repo):
        from cs_predict import CSPredictor
        paths = sorted(str(p) for atom in sorted(os.listdir(heads))
                       for p in (heads / atom).glob("cs_mace_with_clip_*_epoch_2000"))
        if not paths:
            raise SystemExit(f"no trained heads under {heads}")
        predictor = CSPredictor(cs_model_paths=paths, device=device)
    run = make_descriptor_fn(predictor.descriptors_model)
    verify_descriptor_fn(predictor.descriptors_model, run)
    return predictor, run


def reduce_polish(pdb_in: Path, pdb_out: Path) -> Path:
    """Flip-optimise REDUCE on a relaxed structure, keeping the hydrogens it carries.

    Stripping them first would discard the minimised hydrogens the relaxation produced.
    """
    with pdb_out.open("w") as stream:
        subprocess.run([str(env_path("REDUCE_BIN")), "-build", "-Quiet", str(pdb_in)],
                       stdout=stream, stderr=subprocess.DEVNULL, check=False)
    if not pdb_out.stat().st_size:
        raise RuntimeError(f"REDUCE produced no output for {pdb_in}")
    return pdb_out


def receptive_field(predictor) -> float:
    """Separation at which packed cut-outs cannot see each other: r_max x interactions."""
    model = predictor.descriptors_model.models[0]
    return float(getattr(model, "r_max", 5.0)) * int(getattr(model, "num_interactions", 2))


def _pack(records, separation: float):
    """Lay cut-outs on a cubic grid, far enough apart to be mutually invisible."""
    from ase import Atoms

    side = math.ceil(len(records) ** (1 / 3))
    max_span = max(float(np.ptp(np.stack(r["env"]["coords"].to_numpy()), axis=0).max())
                   for r in records)
    stride = max_span + separation + 2.0
    symbols, coords, offsets, cursor = [], [], [], 0
    for k, rec in enumerate(records):
        xyz = np.stack(rec["env"]["coords"].to_numpy()).astype(np.float64)
        xyz -= (xyz.max(0) + xyz.min(0)) / 2
        xyz += np.array([k % side, (k // side) % side, k // (side * side)],
                        dtype=np.float64) * stride
        offsets.append(cursor)
        symbols.extend(rec["env"]["element"].tolist())
        coords.append(xyz)
        cursor += len(xyz)
    packed = np.concatenate(coords, axis=0)
    # MACE computes distances in float32: a centred box keeps the mantissa bits that a
    # grid pushed out to large coordinates would lose.
    packed -= packed.mean(0)
    return Atoms(symbols=symbols, positions=packed), offsets


def _cutout_key(env):
    """Byte identity of the ordered cut-out: same atoms, same order, same coordinates."""
    rid = np.asarray(env["res_id"].to_numpy(), dtype=np.int64)
    xyz = np.stack(env["coords"].to_numpy()).astype(np.float32, copy=False)
    return (rid.tobytes(), xyz.tobytes(), tuple(env["element"].tolist()))


def descriptors_fast(predictor, run, frame, batch_size: int):
    """``CSPredictor.get_descriptors``' contract, {res_id: {atom: tensor}}, one forward each.

    The target-atom row is indexed as in ``cs_predict.py`` (``indices[0] - env.index[0]``).
    """
    environments = predictor.extract_environments(frame, rmax=5.0)
    groups: dict = {}
    for res_id, env in environments.items():
        groups.setdefault(_cutout_key(env), {"env": env, "targets": []})["targets"].append(res_id)

    out: dict = {}
    records = list(groups.values())
    separation = receptive_field(predictor) if batch_size > 1 else 0.0
    for start in range(0, len(records), max(1, batch_size)):
        chunk = records[start:start + max(1, batch_size)]
        if len(chunk) == 1:
            descriptors = run(predictor.compute_atoms(chunk[0]["env"]))
            offsets = [0]
        else:
            packed, offsets = _pack(chunk, separation)
            descriptors = run(packed)
        for rec, offset in zip(chunk, offsets, strict=True):
            env = rec["env"]
            for res_id in rec["targets"]:
                per_atom = {}
                for atom_type in predictor.atom_types:
                    idx = env[(env["res_id"] == res_id)
                              & (env["atom"] == atom_type)].index.tolist()
                    if not idx:
                        continue
                    row = offset + (idx[0] - env.index[0])
                    if row < len(descriptors):
                        per_atom[atom_type] = torch.tensor(
                            descriptors[row], dtype=torch.float32, device=predictor.device)
                out[res_id] = per_atom
    return out
