"""The AtomTransformer: structure-only 8Å atom cloud → per-nucleus chemical shifts.

Owns the exact forward computation the trained weights encode and the strict loader
every consumer uses. A checkpoint carries its own normalization stats and its complete
atom-cloud recipe, and `load_checkpoint` refuses one that does not.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import torch
from torch import nn

from .atom_cloud import FINAL_ATOM_CLOUD_RECIPE, AtomCloudRecipe
from .domain import ATOM_TO_IDX


def _stable_attention(q, k, v, *, mask=None, bias=None):
    """Evaluate attention reductions in FP32, even inside an AMP region.

    Scaling an FP16 ``q @ k`` after the matmul is too late for overflow in the
    accumulation itself. Returns the original value dtype, leaving the AMP contract
    unchanged.
    """
    value_dtype = v.dtype
    with torch.autocast(device_type=q.device.type, enabled=False):
        q32, k32, v32 = q.float(), k.float(), v.float()
        logits = (q32 @ k32.transpose(-1, -2)) / math.sqrt(q.size(-1))
        if bias is not None:
            logits = logits + bias.float()
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        out = logits.softmax(-1) @ v32
    return out.to(value_dtype)


class RBFExpansion(nn.Module):
    def __init__(self, n_rbf: int = 32, dmin: float = 0.0, dmax: float = 12.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(dmin, dmax, n_rbf))
        self.register_buffer("beta", torch.tensor(1.0 / (((dmax - dmin) / n_rbf) ** 2)))

    def forward(self, distance):
        return torch.exp(-self.beta * (distance.unsqueeze(-1) - self.centers) ** 2)


class AtomBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int = 4, dropout: float = 0.3):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff_mult * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, key_mask):
        b, m, d = x.shape
        y = self.norm1(x)
        qkv = self.qkv(y).view(b, m, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        out = _stable_attention(q, k, v, mask=key_mask[:, None, None, :], bias=bias)
        out = out.transpose(1, 2).reshape(b, m, d)
        x = x + self.drop(self.proj(out))
        return x + self.ff(self.norm2(x))


class AtomTransformer(nn.Module):
    """Structure-only architecture: atom embeddings + distance-derived attention, a
    nucleus-query readout over the center residue's atoms, one head per shift.
    """

    CENTER_OFFSET = FINAL_ATOM_CLOUD_RECIPE.center_offset_code
    REF_ATOM_TYPES = tuple(ATOM_TO_IDX[name] for name in ("N", "CA", "C"))
    N_OFFSET_TYPES = FINAL_ATOM_CLOUD_RECIPE.pad_offset_index

    def __init__(self, n_atom_types: int, n_shifts: int, *, dim: int = 384, heads: int = 8,
                 layers: int = 6, n_rbf: int = 32, dropout: float = 0.3):
        super().__init__()
        self.n_shifts, self.heads = n_shifts, heads
        self.atom_embed = nn.Embedding(n_atom_types + 1, dim, padding_idx=n_atom_types)
        self.rbf = RBFExpansion(n_rbf)
        self.pair_mlp = nn.Sequential(nn.Linear(n_rbf, 2 * heads), nn.GELU(),
                                      nn.Linear(2 * heads, heads))
        self.geom_rbf = RBFExpansion(n_rbf)
        self.geom_proj = nn.Linear(3 * n_rbf, dim)
        self.blocks = nn.ModuleList([AtomBlock(dim, heads, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.nucleus_embed = nn.Embedding(n_shifts, dim)
        self.rd_heads, self.rd_dh = heads, dim // heads
        self.rq = nn.Linear(dim, dim)
        self.rk = nn.Linear(dim, dim)
        self.rv = nn.Linear(dim, dim)
        self.rout = nn.Linear(dim, dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(dim, 1))

    def _centered_xyz(self, atom_xyz, atom_mask):
        # Distances must not depend on where the structure sits in space. atom_xyz is
        # absolute fp16 (mean |x| ~22Å), whose rounding step grows with magnitude, and a
        # live PDB thousands of Å from the origin loses even fp32 cdist to cancellation.
        # Subtracting the masked centroid in fp32 first keeps distances translation-
        # invariant.
        xyz = atom_xyz.float()
        w = atom_mask.unsqueeze(-1).float()
        centroid = (xyz * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1.0)
        return xyz - centroid

    def _refs(self, atom_type, atom_offset, xyz):
        # Reference each atom to the center residue's N/CA/C (xyz already centered). A
        # missing anchor returns a zero ref and hit=False, and forward() masks that
        # channel, so an absent anchor cannot leak the frame.
        refs, hits = [], []
        center = atom_offset == self.CENTER_OFFSET
        for atom in self.REF_ATOM_TYPES:
            match = center & (atom_type == atom)
            hit = match.any(1, keepdim=True)
            index = match.float().argmax(1)
            ref = torch.gather(xyz, 1, index[:, None, None].expand(-1, 1, 3))
            refs.append(ref * hit.unsqueeze(-1))
            hits.append(hit)
        return refs, hits

    def forward(self, atom_type, atom_offset, atom_xyz, atom_mask):
        b, m = atom_type.shape
        h = self.atom_embed(atom_type)
        xyz = self._centered_xyz(atom_xyz, atom_mask)
        refs, hits = self._refs(atom_type, atom_offset, xyz)
        dref = torch.stack([(xyz - r).norm(dim=-1) for r in refs], dim=-1)
        miss = torch.cat(hits, dim=1).logical_not()[:, None, :]      # (b,1,3): anchor absent
        dref = dref.masked_fill(miss.expand(-1, m, -1), 0.0)         # constant -> translation-invariant
        h = h + self.geom_proj(self.geom_rbf(dref).flatten(-2))
        dist = torch.cdist(xyz, xyz)
        bias = self.pair_mlp(self.rbf(dist)).permute(0, 3, 1, 2)
        for block in self.blocks:
            h = block(h, bias, atom_mask)
        h = self.norm(h)
        center = (atom_offset == self.CENTER_OFFSET) & atom_mask
        s, heads, dh = self.n_shifts, self.rd_heads, self.rd_dh
        q = self.rq(self.nucleus_embed.weight).view(1, s, heads, dh).expand(b, -1, -1, -1).transpose(1, 2)
        k = self.rk(h).view(b, m, heads, dh).transpose(1, 2)
        v = self.rv(h).view(b, m, heads, dh).transpose(1, 2)
        out = _stable_attention(q, k, v, mask=center[:, None, None, :])
        out = out.transpose(1, 2).reshape(b, s, heads * dh)
        pred = self.head(self.rout(out)).squeeze(-1)
        return pred, center.any(1, keepdim=True).expand(-1, s)


@dataclass
class LoadedCheckpoint:
    model: AtomTransformer
    stats: Mapping
    shift_columns: tuple[str, ...]
    recipe: Mapping | None      # stamped atom-cloud recipe; inference rebuilds the identical cloud
    path: Path
    atom_pad_index: int
    offset_pad_index: int
    manifest: Mapping | None = None


def _validated_stats(stats, shift_columns):
    """Require a real ppm transform for every checkpoint output.

    A missing cell would fall through to mean=0/std=1 and return z scores named ppm.
    Checkpoints are an external artifact boundary, so this fails before construction.
    """
    if not isinstance(stats, Mapping):
        raise ValueError("checkpoint stats must be a mapping")
    for column in shift_columns:
        cell = stats.get(column)
        if not isinstance(cell, Mapping) or "mean" not in cell or "std" not in cell:
            raise ValueError(f"checkpoint stats missing global ppm transform for {column}")
        try:
            mean, std = float(cell["mean"]), float(cell["std"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint stats for {column} are not numeric") from exc
        if not isfinite(mean) or not isfinite(std) or std <= 0:
            raise ValueError(
                f"checkpoint stats for {column} need finite mean and positive std"
            )
    return stats


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> LoadedCheckpoint:
    """Strict-load a checkpoint over the atom alphabet this package defines.

    Every state-dict key must load, and the stats and the complete atom-cloud recipe
    travel with the file.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"dim", "layers", "shift_cols", "stats", "model_state_dict", "recipe"}
    if missing := required - checkpoint.keys():
        raise ValueError(f"checkpoint metadata missing: {sorted(missing)}")
    recipe = AtomCloudRecipe.from_mapping(checkpoint["recipe"]).to_dict()
    shift_columns = tuple(str(column) for column in checkpoint["shift_cols"])
    if not shift_columns or len(set(shift_columns)) != len(shift_columns):
        raise ValueError("checkpoint shift_cols must be non-empty and unique")
    if ("n_shifts" in checkpoint
            and int(checkpoint["n_shifts"]) != len(shift_columns)):
        raise ValueError("checkpoint n_shifts disagrees with shift_cols")
    stats = _validated_stats(checkpoint["stats"], shift_columns)

    state = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model_state_dict"].items()}
    dim = int(checkpoint["dim"])
    if "atom_embed.weight" not in state:
        raise ValueError("checkpoint has no atom_embed.weight")
    embed_shape = tuple(state["atom_embed.weight"].shape)
    n_atoms = embed_shape[0]
    if n_atoms != len(ATOM_TO_IDX) + 1 or embed_shape[1] != dim:
        raise ValueError(
            f"checkpoint atom vocabulary has {n_atoms} tokens and width {embed_shape[1]}; "
            f"this package defines {len(ATOM_TO_IDX)} atom names plus padding and dim {dim}")
    pad = AtomTransformer.N_OFFSET_TYPES

    model = AtomTransformer(n_atoms - 1, len(shift_columns),
                            dim=dim, layers=int(checkpoint["layers"]))
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"checkpoint mismatch: {result}")
    model.to(device).eval()

    return LoadedCheckpoint(
        model=model, stats=stats, shift_columns=shift_columns,
        recipe=recipe, path=path,
        atom_pad_index=n_atoms - 1, offset_pad_index=pad,
        manifest=checkpoint.get("manifest"))
