"""Train one held-out-fold member of the structure-only atom transformer."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, default_collate

from .atom_cloud import AtomCloudRecipe
from .domain import ATOM_TYPES, SHIFT_COLUMNS, STANDARD_RESIDUES
from .loss import (
    AMBIGUITY_SET_CODES,
    build_loss_tensors,
    hierarchical_loss,
    hierarchical_weight_recipe,
    swap_elements,
)
from .model import AtomTransformer
from .normalization import (
    BACKBONE_SHIFTS,
    STD_FLOOR,
    build_aa_tables,
    build_fixed_per_aa,
    denormalize,
    normalize,
)

# Lean cache dataset: mmap reader over fold_*/structural + fold_*/atom.

_ATOM_KEYS = ("atom_type", "atom_offset", "atom_xyz", "atom_mask")


def _model_inputs(batch, device, *, non_blocking=False):
    """The forward's keyword arguments for one batch."""
    return {k: batch[k].to(device, non_blocking=non_blocking) for k in _ATOM_KEYS}


class LeanCacheDataset(Dataset):
    """Reader for the entity multichain ppm cache.

    Structural labels and the atom cloud are memory-mapped; stored ppm converts to per-AA
    z at the single model boundary, and p100 flags mask the affected label cells. Each
    sample emits the atom cloud, the z target and its mask, ambiguity codes, residue and
    source identity, and the entity ids needed for grouping.
    """
    def __init__(self, fold_dirs, stats, shift_cols, *, sources=None):
        self.fold_dirs = [Path(x) for x in fold_dirs]
        self.shift_cols = tuple(shift_cols)
        self.parts = []
        self.index = []
        self._flag_drop = None
        source_names = None
        for fd in self.fold_dirs:
            cfg = json.loads((fd / "config.json").read_text())
            if tuple(cfg["shift_cols"]) != self.shift_cols:
                raise ValueError(f"shift-column mismatch in {fd}")
            if not (cfg.get("has_ppm_shifts") or cfg.get("shift_storage") == "ppm_float32"):
                raise ValueError(f"{fd} is not a ppm cache; only ppm_float32 is supported")
            sd, ad = fd / "structural", fd / "atom"
            p = {
                "shifts": np.load(sd / "shifts.npy", mmap_mode="r"),
                "smask": np.load(sd / "shift_mask.npy", mmap_mode="r"),
                "ambig": np.load(sd / "ambiguity.npy", mmap_mode="r") if (sd / "ambiguity.npy").exists() else None,
                "resid": np.load(sd / "residue_idx.npy", mmap_mode="r"),
                "atype": np.load(ad / "atom_type.npy", mmap_mode="r"),
                "aoff": np.load(ad / "atom_offset.npy", mmap_mode="r"),
                "axyz": np.load(ad / "atom_xyz.npy", mmap_mode="r"),
                "acnt": np.load(ad / "atom_count.npy", mmap_mode="r"),
                "src": np.load(sd / "source_idx.npy", mmap_mode="r") if (sd / "source_idx.npy").exists() else None,
                "bm": json.loads((sd / "bmrb_mapping.json").read_text()) if (sd / "bmrb_mapping.json").exists() else None,
                "g2r": json.loads((sd / "global_to_resid.json").read_text()) if (sd / "global_to_resid.json").exists() else None,
                # Entity-grained identity maps {id_class: {global_idx: id}}.
                "idmaps": json.loads((sd / "id_maps.json").read_text()) if (sd / "id_maps.json").exists() else None,
            }
            fold_source_names = cfg.get("source_names")
            if source_names is None:
                source_names = fold_source_names
            elif fold_source_names != source_names:
                raise ValueError(
                    f"{fd}: source_names {fold_source_names!r} differ from "
                    f"{source_names!r} in the other folds"
                )
            pi = len(self.parts)
            self.parts.append(p)
            eligible = (p["acnt"][:] > 0) & p["smask"][:].any(1)
            self.index.extend((pi, int(g)) for g in np.flatnonzero(eligible))
        if not self.parts:
            raise ValueError("at least one cache fold is required")
        # Expose ambiguity codes only when every fold carries them, so a batch is never
        # a ragged mix of coded and code-less samples.
        self.has_ambiguity = all(p["ambig"] is not None for p in self.parts)
        self.max_atoms = self.parts[0]["atype"].shape[1]
        means, stds, _ = build_aa_tables(stats, self.shift_cols)
        self.pa_mean = torch.from_numpy(means[:len(STANDARD_RESIDUES)].astype(np.float32))
        self.pa_std = torch.from_numpy(stds[:len(STANDARD_RESIDUES)].astype(np.float32))
        if sources is not None:
            if not source_names:
                raise ValueError("source filter requested but cache has no source_names")
            missing_source = [
                str(fd) for fd, part in zip(self.fold_dirs, self.parts)
                if part["src"] is None
            ]
            if missing_source:
                raise ValueError(
                    "source filter requested but source_idx.npy is missing from "
                    f"{missing_source}"
                )
            codes = {name: i for i, name in enumerate(source_names)}
            if unknown := sorted(set(sources) - codes.keys()):
                raise ValueError(f"unknown sources {unknown}; expected {source_names}")
            keep = {codes[x] for x in sources}
            self.index = [(pi, g) for pi, g in self.index
                          if int(self.parts[pi]["src"][g]) in keep]

    def __len__(self):
        return len(self.index)

    def atom_counts(self):
        return np.asarray([self.parts[pi]["acnt"][g] for pi, g in self.index], dtype=np.int32)

    def effective_counts(self, group_weight, positions=None):
        """Train-fold-only effective positive count per shift column.

        Each observed, un-flagged cell contributes its equivalence-group weight, so a
        methyl triplicate counts once. ``positions`` restricts the tally to the exact
        population the loss sees after the cap and validation carve-out.
        """
        gw = (group_weight.detach().cpu().numpy() if isinstance(group_weight, torch.Tensor)
              else np.asarray(group_weight, dtype=np.float64))
        n_aa = gw.shape[0]
        counts = np.zeros(len(self.shift_cols), dtype=np.float64)
        rows = self.index if positions is None else (self.index[p] for p in positions)
        for pi, g in rows:
            p = self.parts[pi]
            smask = p["smask"][g].astype(bool).copy()
            if self._flag_drop is not None and (cols := self._flag_drop[pi].get(g)) is not None:
                smask[cols] = False
            aa = int(p["resid"][g])
            aa = aa if 0 <= aa < n_aa else n_aa - 1
            counts += gw[aa] * smask
        return counts

    def row_identity(self, pi, g):
        """Every identity id the cache carries for one stored row: entity/label/sample/
        materialization/label-residue ids plus BMRB, residue label, and source."""
        p = self.parts[pi]
        ident = {"bmrb": (p["bm"] or {}).get(str(g)),
                 "residue": (p["g2r"] or {}).get(str(g)),
                 "source": int(p["src"][g]) if p["src"] is not None else 0}
        if maps := p["idmaps"]:
            for name in ("logical_entity_id", "label_set_id", "sample_id",
                         "materialization_id", "label_residue_id"):
                ident[name] = maps.get(name, {}).get(str(g))
        return ident

    def restrict_to_materializations(self, keep):
        """Keep only rows belonging to the given materialization ids. Returns rows kept.

        A materialization is one (label set, source) structure build, so this selects
        whole structures, the unit a corpus-composition experiment varies. Fails closed
        in both directions: an id addressing nothing, like an empty result, means the
        manifest and the cache disagree.
        """
        keep = set(keep)
        if not keep:
            raise ValueError("empty materialization manifest; omit the flag to train on all rows")
        seen, index = set(), []
        for pi, g in self.index:
            maps = (self.parts[pi]["idmaps"] or {}).get("materialization_id")
            if not maps:
                raise ValueError(f"{self.fold_dirs[pi]} carries no materialization_id map, "
                                 "so a materialization manifest cannot be applied")
            m = maps.get(str(g))
            if m in keep:
                seen.add(m)
                index.append((pi, g))
        if missing := (keep - seen):
            raise ValueError(
                f"{len(missing)} of {len(keep)} requested materializations matched no row "
                f"in {[str(d) for d in self.fold_dirs]} (e.g. {sorted(missing)[:3]}); the "
                "manifest was built against a different cache or a different fold split")
        self.index = index
        return len(index)

    def apply_flags(self, flags):
        """Mask p100-flagged label cells through the cache's global->label-residue map,
        so a flag masks exactly one entity's one residue cell. Returns the cell count.

        Empty flags mean "no fence" and are a no-op. A fence that was asked for but
        reached nothing fails closed instead, since a returned 0 is indistinguishable
        from a clean cache: a missing label-residue map, or ids/columns that do not
        address this cache, raises.
        """
        self._flag_drop = [{} for _ in self.parts]
        if not flags:
            return 0
        col_idx = {c: i for i, c in enumerate(self.shift_cols)}
        named = {c for cols in flags.values() for c in cols if isinstance(c, str)}
        if unknown := sorted(named - set(SHIFT_COLUMNS)):
            raise ValueError(f"the flags file names {unknown}, which are not shift columns; "
                             "those cells were fenced nowhere")
        dropped = 0
        present_requested = set()
        transitioned = set()
        for pi, g in self.index:
            lr_map = (self.parts[pi]["idmaps"] or {}).get("label_residue_id")
            if not lr_map:
                raise ValueError(f"{self.fold_dirs[pi]} carries no label_residue_id map, "
                                 "so the requested flag fence cannot be applied")
            label_residue_id = lr_map.get(str(g))
            requested = flags.get(label_residue_id, ())
            present_requested.update(
                (label_residue_id, column) for column in requested if column in col_idx
            )
            cols = [
                col_idx[column]
                for column in requested
                if column in col_idx and bool(self.parts[pi]["smask"][g, col_idx[column]])
            ]
            if cols:
                self._flag_drop[pi][g] = np.asarray(sorted(set(cols)), dtype=np.int64)
                dropped += len(self._flag_drop[pi][g])
                transitioned.update(
                    (label_residue_id, self.shift_cols[column])
                    for column in self._flag_drop[pi][g]
                )
        if unmatched := sorted(present_requested - transitioned):
            raise ValueError(
                f"{len(unmatched)} requested flag cells address label residues in this "
                "cache but no observed cell (possibly already masked); for example "
                f"{unmatched[:3]}"
            )
        if not dropped:
            raise ValueError(f"{len(flags)} flagged label residues matched no cell in "
                             f"{[str(d) for d in self.fold_dirs]}; the flags file and the "
                             "cache disagree on label-residue ids or shift columns")
        return dropped

    def __getitem__(self, i):
        pi, g = self.index[i]
        p = self.parts[pi]
        n = int(p["acnt"][g])
        mask = np.zeros(self.max_atoms, dtype=bool)
        mask[:n] = True
        smask = torch.from_numpy(p["smask"][g].astype(bool))
        if self._flag_drop is not None and (cols := self._flag_drop[pi].get(g)) is not None:
            smask[cols] = False
        ppm = torch.from_numpy(p["shifts"][g].astype(np.float32))
        aa = int(p["resid"][g])
        safe_aa = aa if 0 <= aa < len(STANDARD_RESIDUES) else len(STANDARD_RESIDUES) - 1
        target = (ppm - self.pa_mean[safe_aa]) / self.pa_std[safe_aa]
        item = {
            "atom_type": torch.from_numpy(p["atype"][g].astype(np.int64)),
            "atom_offset": torch.from_numpy(p["aoff"][g].astype(np.int64)),
            "atom_xyz": torch.from_numpy(p["axyz"][g].astype(np.float32)),
            "atom_mask": torch.from_numpy(mask),
            "shift_target": torch.where(smask, target, torch.zeros_like(target)),
            "shift_mask": smask,
            "res_idx": aa,
            "source": int(p["src"][g]) if p["src"] is not None else 0,
        }
        if self.has_ambiguity:
            item["ambiguity"] = torch.from_numpy(p["ambig"][g].astype(np.int64))
        return item


def trim_collate(batch):
    """Collate, then trim the atom axis to the batch's real (mask-padded) width."""
    out = default_collate(batch)
    full = out["atom_mask"].shape[1]
    real = int(out["atom_mask"].sum(1).max().item())
    width = min(full, max(64, ((real + 63) // 64) * 64))
    if width < full:
        for key in _ATOM_KEYS:
            out[key] = out[key][:, :width].contiguous()
    return out


class LengthBucketBatchSampler(Sampler):
    """Shuffle, then sort within mega-chunks so a batch holds similar-length clouds."""
    def __init__(self, lengths, batch_size, seed=0, mega=64):
        self.lengths = np.asarray(lengths)
        self.bs, self.seed, self.mega, self.epoch = int(batch_size), int(seed), int(mega), 0
        if self.bs <= 0:
            raise ValueError("batch_size must be positive")
        self._nb = (len(self.lengths) + self.bs - 1) // self.bs

    def __len__(self):
        return self._nb

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.permutation(len(self.lengths))
        batches, mega_size = [], self.bs * self.mega
        for start in range(0, len(idx), mega_size):
            chunk = idx[start:start + mega_size]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for offset in range(0, len(chunk), self.bs):
                batches.append(chunk[offset:offset + self.bs].tolist())
        for order in rng.permutation(len(batches)):
            yield batches[order]


def _nested_fraction(dataset, frac, seed):
    """Keep a ``frac`` share of the training rows, nested across fractions.

    A seeded permutation prefix contains the 25% arm in the 50% arm, leaving data volume
    as the ladder's only variable; independent draws per size would not be nested, and
    the ladder could fall non-monotone from sampling luck.
    """
    if frac <= 0 or frac >= 1:
        return dataset
    n = len(dataset)
    count = max(1, round(n * frac))
    keep = np.random.default_rng(seed).permutation(n)[:count]
    return Subset(dataset, keep.tolist())


def _lengths(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset.atom_counts()[np.asarray(dataset.indices)]
    return dataset.atom_counts()


def _effective_counts(dataset, group_weight):
    """Effective positive counts over the exact rows in *dataset* (base or Subset)."""
    if isinstance(dataset, Subset):
        return dataset.dataset.effective_counts(group_weight, positions=dataset.indices)
    return dataset.effective_counts(group_weight)


def dataset_identity(dataset):
    """Per-row identity dicts in the dataset's own (loader) order."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return [base.row_identity(*base.index[i]) for i in dataset.indices]
    return [dataset.row_identity(*dataset.index[i]) for i in range(len(dataset))]


def _ambiguity_eval_targets(pred_z, target_z, mask, aa, ambiguity, partner, center, scale):
    """Resolve prochiral swaps and unsupported set codes for evaluation, mirroring
    ``swap_elements``. Returns ``(resolved_target_z, effective_mask)``.

    A both-observed pair is relabelled when that lowers the pair's summed |error|, unless
    both cells are confidently stereospecific (both code-1); partner targets re-encode
    into the prediction channel so an equivalent swap counts once, and cells coded 4/5/6/9
    drop from the mask. ``ambiguity=None`` -> identity.
    """
    resolved = np.array(target_z, dtype=np.float64, copy=True)
    eff = np.asarray(mask, dtype=bool).copy()
    if ambiguity is None:
        return resolved, eff
    ambiguity = np.asarray(ambiguity)
    partner = np.asarray(partner)
    center = np.asarray(center, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    n_aa = partner.shape[0]
    aa = np.asarray(aa)
    aa_safe = np.where((aa >= 0) & (aa < n_aa), aa, n_aa - 1)
    part = partner[aa_safe]                                   # (N, S) partner col or -1
    c = center[aa_safe]
    s = np.maximum(scale[aa_safe], STD_FLOOR)
    eff &= ~np.isin(ambiguity, AMBIGUITY_SET_CODES)             # set codes never score
    _, S = pred_z.shape
    cols = np.arange(S)[None, :]
    safe_part = np.where(part >= 0, part, 0)
    y = target_z * s + c                                      # ppm per cell
    y_partner = np.take_along_axis(y, safe_part, axis=1)
    cross = (y_partner - c) / s                               # partner ppm re-encoded here
    code_j = np.take_along_axis(ambiguity, safe_part, axis=1)
    mask_j = np.take_along_axis(np.asarray(mask, dtype=bool), safe_part, axis=1)
    both_obs = (part >= 0) & np.asarray(mask, dtype=bool) & mask_j
    self_set = np.isin(ambiguity, AMBIGUITY_SET_CODES)
    partner_set = np.isin(code_j, AMBIGUITY_SET_CODES)
    swappable = both_obs & ~self_set & ~partner_set & ~((ambiguity == 1) & (code_j == 1))
    lead = swappable & (part > cols)                          # process each pair once (i < j)
    pred_j = np.take_along_axis(pred_z, safe_part, axis=1)
    cross_j = np.take_along_axis(cross, safe_part, axis=1)    # cross target for partner col
    cost_direct = np.abs(pred_z - target_z) + np.abs(pred_j - np.take_along_axis(target_z, safe_part, axis=1))
    cost_cross = np.abs(pred_z - cross) + np.abs(pred_j - cross_j)
    do_swap = lead & (cost_cross < cost_direct)
    rows, cis = np.nonzero(do_swap)
    cjs = part[rows, cis]
    resolved[rows, cis] = cross[rows, cis]
    resolved[rows, cjs] = cross_j[rows, cis]
    return resolved, eff


@torch.no_grad()
def evaluate(model, loader, device, stats, shift_cols, return_predictions=False,
             partner=None):
    """Score the held-out fold. Prochiral pairs are resolved in ppm by the package rule
    (a both-observed pair is relabelled when that lowers its summed absolute error, unless
    both cells are code 1; set codes drop), then per-nucleus z-MAE/MAE/RMSE are computed
    over the resolved targets so an equivalent swap is not counted twice."""
    # A column with no normalization entry must raise, not be skipped: skipping drops the
    # nucleus from every reported metric and from the backbone average epochs select on.
    if missing := [c for c in shift_cols if c not in stats]:
        raise ValueError(f"stats carry no entry for {missing}; those nuclei would be "
                         "dropped from the report without a word")
    model.eval()
    predictions, targets, masks, residues, ambigs = [], [], [], [], []
    have_ambiguity = False
    for batch in loader:
        target, smask, aa = batch.pop("shift_target"), batch.pop("shift_mask"), batch["res_idx"]
        amb = batch.pop("ambiguity") if "ambiguity" in batch else None
        inputs = _model_inputs(batch, device)
        ctx = autocast("cuda") if device.type == "cuda" else nullcontext()
        with ctx:
            pred, valid = model(**inputs)
        predictions.append(pred.float().cpu())
        targets.append(target)
        masks.append(smask & valid.cpu())
        residues.append(aa)
        if amb is not None:
            have_ambiguity = True
            ambigs.append(amb)
    pred, target = torch.cat(predictions), torch.cat(targets)
    mask, aa = torch.cat(masks), torch.cat(residues)
    pred_np, target_np, mask_np, aa_np = pred.numpy(), target.numpy(), mask.numpy(), aa.numpy()
    ambiguity_np = np.concatenate([a.numpy() for a in ambigs]) if have_ambiguity else None
    pred_ppm = denormalize(pred_np, aa_np, stats, shift_cols)
    raw_ppm = denormalize(target_np, aa_np, stats, shift_cols)
    if have_ambiguity and partner is not None:
        # The resolution the evaluator and every scorer apply: in ppm, no model's scale.
        flat = np.zeros((len(STANDARD_RESIDUES), len(shift_cols)))
        true_ppm, eff_mask_np = _ambiguity_eval_targets(
            pred_ppm, np.where(mask_np, raw_ppm, 0.0), mask_np, aa_np, ambiguity_np,
            partner, flat, flat + 1.0)
    else:
        true_ppm, eff_mask_np = raw_ppm.astype(np.float64), mask_np
    resolved_z = normalize(true_ppm, aa_np, stats, shift_cols)
    resolved_t = torch.from_numpy(np.asarray(resolved_z, dtype=np.float32))
    eff_mask = torch.from_numpy(eff_mask_np)
    error = pred_ppm - true_ppm
    per_z, per_mae, per_rmse = {}, {}, {}
    for i, col in enumerate(shift_cols):
        m = eff_mask[:, i]
        if m.any():
            per_z[col] = float((pred[:, i][m] - resolved_t[:, i][m]).abs().mean())
            mn = eff_mask_np[:, i]
            per_mae[col] = float(np.abs(error[mn, i]).mean())
            per_rmse[col] = float(np.sqrt(np.square(error[mn, i]).mean()))
    backbone = [c for c in shift_cols if c in BACKBONE_SHIFTS and c in per_z]
    metrics = {"backbone_z_mae": sum(per_z[c] for c in backbone) / len(backbone),
               "per_shift_mae": per_mae, "per_shift_z": per_z, "per_shift_rmse": per_rmse}
    if return_predictions:
        # Carry the raw labels, the ambiguity codes and the exact scored population
        # (resolved targets + the set-code-reduced mask), so the NPZ reproduces the
        # reported metric instead of drifting on the raw labels.
        metrics["predictions"] = {
            "pred_ppm": pred_ppm,
            "true_ppm": raw_ppm,
            "resolved_true_ppm": true_ppm,
            "mask": mask_np, "effective_mask": eff_mask_np, "residue_code": aa_np,
            "pred_z": pred_np, "target_z": target_np, "resolved_target_z": resolved_z,
            "ambiguity": (ambiguity_np if ambiguity_np is not None
                          else np.zeros_like(mask_np, dtype=np.int64))}
    return metrics


@dataclass(frozen=True)
class TrainingRecipe:
    epochs: int = 35
    batch_size: int = 128
    learning_rate: float = 3e-4
    eta_min_fraction: float = 0.05
    weight_decay: float = 0.0
    dimension: int = 384
    layers: int = 6
    heads: int = 8
    dropout: float = 0.3
    seed: int = 0
    checkpoint_every: int = 5
    huber_delta: float = 1.0
    gradient_clip: float = 1.0
    checkpoint_selection: str = "final"
    deterministic: bool = True


FINAL_TRAINING_RECIPE = TrainingRecipe()
MODEL_TAG = "deepshift_structonly"
GRAD_CLIP = FINAL_TRAINING_RECIPE.gradient_clip


def realized_training_recipe(manifest) -> TrainingRecipe:
    """The training recipe a checkpoint manifest records."""
    protocol, model, loss = manifest["protocol"], manifest["model"], manifest["loss"]
    return TrainingRecipe(
        epochs=int(protocol["epochs"]), batch_size=int(protocol["batch_size"]),
        learning_rate=float(protocol["learning_rate"]),
        eta_min_fraction=float(protocol["eta_min"]) / float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]), dimension=int(model["dim"]),
        layers=int(model["layers"]), heads=int(protocol["heads"]),
        dropout=float(protocol["dropout"]), seed=int(protocol["seed"]),
        checkpoint_every=int(protocol["ckpt_every"]),
        huber_delta=float(loss["huber_delta"]), gradient_clip=float(protocol["gradient_clip"]),
        checkpoint_selection=str(protocol["checkpoint_selection"]),
        deterministic=bool(protocol["determinism"]["deterministic_algorithms"]))


# Settings outside the recipe that also change what a member learns from; a paper
# member records every one of them at its default.
_NO_ABLATION = (
    ("protocol", "data_frac", 0.0),
    ("inputs", "train_materializations", None),
    ("arguments", "train_sources", None),
)


def verify_final_training_recipe(manifest, declared_ablations=()) -> TrainingRecipe:
    """Require a checkpoint manifest to record the final training recipe.

    Every field, and every ablation switch, must match -- except the switches an ablation arm
    declares in ``declared_ablations`` as ``block.key`` names (``protocol.data_frac`` for a
    data-fraction point, ``arguments.train_sources`` for a source-restricted member). Each
    declared switch must actually be away from its default, so a declaration cannot be
    used to wave a paper member through. Returns the recipe the manifest realized.
    """
    realized = realized_training_recipe(manifest)
    differing = []
    for field in fields(TrainingRecipe):
        have, want = getattr(realized, field.name), getattr(FINAL_TRAINING_RECIPE, field.name)
        same = (math.isclose(have, want, rel_tol=1e-6) if isinstance(want, float)
                else have == want)
        if not same:
            differing.append(f"{field.name}={have!r} (final {want!r})")
    known = {f"{block}.{key}" for block, key, _ in _NO_ABLATION}
    unknown = sorted(set(declared_ablations) - known)
    if unknown:
        raise ValueError(f"not ablation switches: {unknown}; known: {sorted(known)}")
    unrealized = set(declared_ablations)
    for block, key, default in _NO_ABLATION:
        have = manifest.get(block, {}).get(key)
        if have == default:
            continue
        if f"{block}.{key}" in unrealized:
            unrealized.discard(f"{block}.{key}")
        else:
            differing.append(f"{block}.{key}={have!r} (final {default!r})")
    if unrealized:
        raise ValueError("declared ablations the checkpoint does not realize: "
                         + ", ".join(sorted(unrealized)))
    if manifest["model"].get("tag") != MODEL_TAG:
        differing.append(f"model.tag={manifest['model'].get('tag')!r} (final {MODEL_TAG!r})")
    if differing:
        raise ValueError("checkpoint was not trained under the final recipe: "
                         + "; ".join(differing))
    return realized

# cuBLAS workspace configurations under which GEMM is deterministic.
_CUBLAS_DETERMINISTIC = (":4096:8", ":16:8")


def configure_determinism(deterministic):
    """Pin deterministic kernels, and return the settings applied for the manifest.

    Must run before the first CUDA call: ``CUBLAS_WORKSPACE_CONFIG`` is read once at
    cuBLAS init and is a no-op afterwards. A nondeterministic op raises rather than warns,
    and TF32 goes off with it, its matmul not being reproducible across kernel choices.
    """
    if not deterministic:
        return {"deterministic_algorithms": False}
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in _CUBLAS_DETERMINISTIC:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"deterministic_algorithms": True, "allow_tf32": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def load_atom_recipe(cache, folds):
    """The atom-cloud recipe stamped into the checkpoint.

    Every participating fold must carry ``FINAL_ATOM_CLOUD_RECIPE`` with the model's own
    vocabulary: the model is trained on that cloud and inference rebuilds it from the
    stamp, so any other recipe would train a model whose input no released path builds.
    """
    cache = Path(cache)
    from .data.atoms import shipped_atom_cloud_recipe

    expected = shipped_atom_cloud_recipe()
    for fold in sorted(set(folds)):
        fold_dir = cache / f"fold_{fold}"
        cfg = json.loads((fold_dir / "atom" / "config.json").read_text())
        if "recipe" not in cfg:
            raise ValueError(f"{fold_dir / 'atom/config.json'} carries no atom recipe")
        stamped = AtomCloudRecipe.from_mapping(cfg["recipe"])
        if stamped != expected:
            raise ValueError(
                f"fold {fold} atom recipe {stamped.to_dict()} is not the final recipe "
                f"{expected.to_dict()}")
    return expected.to_dict()


def _optimizer_step(loss, model, optimizer, scaler):
    """Backpropagate and step only when every unscaled gradient is finite."""
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if not bool(torch.isfinite(grad_norm)):
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.update()          # lower the scale as after a scaler-skipped step
        return False
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return True


def _real_model(model):
    return getattr(model, "_orig_mod", model)


def _log(message):
    """Emit trainer progress immediately, even when stdout is redirected."""
    print(message, flush=True)


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@dataclass
class Prepared:
    """Everything ``train`` needs before a model exists: the cohort, the loaders, and
    the loss weighting frozen on the exact rows the loss will see."""

    device: torch.device
    cache: Path
    determinism: dict
    train_folds: list
    participating: list
    shift_cols: list
    stats: dict
    fit: dict
    atom_recipe: dict
    train_ds: object
    eval_ds: object
    train_loader: object
    eval_loader: object
    group_weight: object
    partner: object
    backbone_col_np: object
    sidechain_weight_np: object
    loss_recipe: dict
    train_restricted: object
    train_flagged: int
    eval_flagged: int
    train_before: int
    eval_before: int


@dataclass
class Built:
    """The model and the tensors the training step needs on the device."""

    model: object
    n_parameters: int
    optimizer: object
    scheduler: object
    scaler: object
    eta_min: float
    group_weight: object
    partner: object
    backbone_col: object
    sidechain_weight: object
    swap_center: object
    swap_scale: object
    means: object
    stds: object


def _prepare(args) -> Prepared:
    """Validate the run, load the folds, and freeze the loss weighting.

    Everything here happens before a weight exists, and every check is one that must
    fail before a night of training, never on the first forward pass.
    """
    if args.fold not in range(1, 6):
        raise ValueError("training fold must be 1..5")
    if args.epochs <= 0 or args.dim <= 0 or args.layers <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, dim, layers, and batch-size must all be positive")
    # Before the device is resolved: torch.cuda.is_available() initializes CUDA, after
    # which the cuBLAS workspace setting can no longer take effect.
    determinism = configure_determinism(args.deterministic)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = Path(args.cache)
    # A member trains on the four cross-validation folds it does not hold out.
    train_folds = [f for f in range(1, 6) if f != args.fold]
    participating = sorted({args.fold, *train_folds})

    cfg = json.loads((cache / f"fold_{args.fold}" / "config.json").read_text())
    shift_cols = list(cfg["shift_cols"])
    # The cache's column order is authoritative (stamped into the checkpoint, replayed
    # at inference), but the vocabulary must be the shipped one: another nucleus set trains
    # a head the released ensemble cannot average with, invisibly when the counts agree.
    if set(shift_cols) != set(SHIFT_COLUMNS):
        raise ValueError(
            f"cache nucleus vocabulary is not the shipped one: "
            f"extra {sorted(set(shift_cols) - set(SHIFT_COLUMNS))}, "
            f"missing {sorted(set(SHIFT_COLUMNS) - set(shift_cols))}")
    raw_stats = cfg["stats"]
    fit = raw_stats.get("normalization_fit") or {}
    expected = sorted(f for f in range(1, 6) if f != args.fold)
    if (fit.get("policy") != "cv_train_folds_only" or fit.get("heldout_fold") != args.fold
            or sorted(fit.get("fit_folds", [])) != expected
            or len(raw_stats.get("snuc", {})) != len(shift_cols)):
        raise ValueError(f"fold {args.fold} cache lacks verified train-only S_NUC provenance")
    stats = build_fixed_per_aa(raw_stats, shift_cols)
    atom_recipe = load_atom_recipe(cache, participating)

    train_ds = LeanCacheDataset([cache / f"fold_{f}" for f in train_folds], stats, shift_cols,
                                sources=args.train_sources)
    eval_ds = LeanCacheDataset([cache / f"fold_{args.fold}"], stats, shift_cols)
    # A materialization manifest restricts which structures the training folds contribute
    # and is never applied to the held-out fold, so every arm of a composition experiment
    # is scored on identical evaluation cells.
    train_restricted = None
    if args.train_materializations:
        manifest = json.loads(Path(args.train_materializations).read_text())
        train_restricted = train_ds.restrict_to_materializations(manifest)
    flags = json.loads(Path(args.exclude_flagged).read_text())
    train_flagged = train_ds.apply_flags(flags)
    eval_flagged = eval_ds.apply_flags(flags)
    train_before, eval_before = len(train_ds), len(eval_ds)

    # Equivalence-group weights (methyl 1/N) and prochiral partner map.
    group_weight, partner = build_loss_tensors(shift_cols, STANDARD_RESIDUES)
    train_ds = _nested_fraction(train_ds, args.data_frac, args.seed)
    # Side-chain weights freeze on exactly the rows the loss trains on.
    effective_counts = _effective_counts(train_ds, group_weight)
    loss_recipe, backbone_col_np, sidechain_weight_np = hierarchical_weight_recipe(
        effective_counts, shift_cols)

    sampler = LengthBucketBatchSampler(_lengths(train_ds), args.batch_size, args.seed)
    workers = args.workers if device.type == "cuda" else 0
    loader_common = {"num_workers": workers, "pin_memory": device.type == "cuda",
                     "persistent_workers": workers > 0, "collate_fn": trim_collate}
    if workers:
        loader_common["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_common)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, **loader_common)

    return Prepared(
        device=device, cache=cache, determinism=determinism,
        train_folds=train_folds, participating=participating,
        shift_cols=shift_cols, stats=stats, fit=fit, atom_recipe=atom_recipe,
        train_ds=train_ds, eval_ds=eval_ds, train_loader=train_loader,
        eval_loader=eval_loader, group_weight=group_weight,
        partner=partner, backbone_col_np=backbone_col_np,
        sidechain_weight_np=sidechain_weight_np, loss_recipe=loss_recipe,
        train_restricted=train_restricted,
        train_flagged=train_flagged, eval_flagged=eval_flagged,
        train_before=train_before, eval_before=eval_before)


def _build(args, prep: Prepared) -> Built:
    """Construct the model, the optimizer and the loss tensors, on the device."""
    shift_cols, stats, device = prep.shift_cols, prep.stats, prep.device
    group_weight, partner = prep.group_weight, prep.partner
    backbone_col_np, sidechain_weight_np = prep.backbone_col_np, prep.sidechain_weight_np
    model = AtomTransformer(len(ATOM_TYPES), len(shift_cols), dim=args.dim,
                            heads=FINAL_TRAINING_RECIPE.heads,
                            layers=args.layers, dropout=args.dropout).to(device)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.compile:
        _log("torch.compile: enabled; AMP numerics may differ slightly from eager")
        model = torch.compile(model)
    eta_min = (args.lr * FINAL_TRAINING_RECIPE.eta_min_fraction
               if args.eta_min is None else args.eta_min)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta_min)
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    group_weight, partner = group_weight.to(device), partner.to(device)
    backbone_col = torch.from_numpy(backbone_col_np).to(device)
    sidechain_weight = torch.from_numpy(sidechain_weight_np).to(device)
    # Per-AA center/scale for the swap loss, re-encoding a partner target into the
    # prediction channel's own coordinates.
    means, stds, _ = build_aa_tables(stats, shift_cols)
    swap_center = torch.from_numpy(means.astype(np.float32)).to(device)
    swap_scale = torch.from_numpy(stds.astype(np.float32)).to(device)

    return Built(model=model, n_parameters=n_parameters,
                 optimizer=optimizer, scheduler=scheduler, scaler=scaler, eta_min=eta_min,
                 group_weight=group_weight, partner=partner, backbone_col=backbone_col,
                 sidechain_weight=sidechain_weight, swap_center=swap_center,
                 swap_scale=swap_scale, means=means, stds=stds)


def _manifest(args, prep: Prepared, built: Built) -> dict:
    """The run record stamped into every checkpoint this run writes."""
    stats, fit = prep.stats, prep.fit
    train_ds, eval_ds = prep.train_ds, prep.eval_ds
    determinism = prep.determinism
    train_folds = prep.train_folds
    atom_recipe, loss_recipe = prep.atom_recipe, prep.loss_recipe
    train_before, eval_before = prep.train_before, prep.eval_before
    train_flagged, eval_flagged = prep.train_flagged, prep.eval_flagged
    train_restricted = prep.train_restricted
    n_parameters, eta_min = built.n_parameters, built.eta_min
    flag_payload = json.loads(Path(args.exclude_flagged).read_text())
    flag_cells = sum(len(columns) for columns in flag_payload.values())
    manifest = {
        "arguments": {k: v for k, v in vars(args).items()
                      if k not in ("handler", "cache", "out", "exclude_flagged",
                                   "train_materializations")},
        "model": {"tag": MODEL_TAG, "n_parameters": n_parameters,
                  "dim": args.dim, "layers": args.layers,
                  "token_vocabulary": list(ATOM_TYPES)},
        "protocol": {
            "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
            "eta_min": eta_min, "weight_decay": args.weight_decay, "dropout": args.dropout,
            "heads": FINAL_TRAINING_RECIPE.heads,
            "gradient_clip": FINAL_TRAINING_RECIPE.gradient_clip,
            "ckpt_every": args.ckpt_every, "workers": args.workers,
            "data_frac": args.data_frac,
            "seed": args.seed, "compiled": bool(args.compile),
            # The delivered model is the last epoch, and the held-out fold is scored once,
            # after it, so it chooses nothing. Every consumer checks these two.
            "checkpoint_selection": "final", "test_fold_used_for_selection": False,
            "determinism": determinism,
        },
        "folds": {"train": train_folds, "heldout": [args.fold]},
        "normalization": {"marker": stats.get("normalization"), "fit": fit},
        "recipe": atom_recipe,
        "loss": {"weighting": "hier60_40", "huber_delta": float(args.huber_delta),
                 "effective_counts_scope": "train_rows_after_data_fraction_and_flags",
                 **loss_recipe},
        "inputs": {
            "shift_storage": "ppm_float32",
            "flags": {"policy": ("unfiltered_calibration" if not flag_payload
                                 else "composed_exclusion"),
                      "unique_cells": flag_cells},
            "train_materializations": (
                {"rows_kept": train_restricted} if args.train_materializations else None),
        },
        "samples": {"train_before_cap": train_before, "train_used": len(train_ds),
                    "eval_before_cap": eval_before,
                    "eval_used": len(eval_ds), "train_flagged_cells_masked": train_flagged,
                    "eval_flagged_cells_masked": eval_flagged},
    }
    return manifest


def _plain(value):
    """Return ``value`` with every numpy scalar and array replaced by a Python number or
    a tensor, so a checkpoint that carries it loads under ``weights_only=True``."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(v) for v in value)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resume_state(epoch, built: Built, history: list) -> dict:
    """Everything beyond the weights that continuing after ``epoch`` needs: the optimizer
    and schedule, the history so far, and every random stream the next epoch draws
    from."""
    numpy_state = np.random.get_state()
    return {
        "epoch": int(epoch),
        "optimizer": built.optimizer.state_dict(),
        "scheduler": built.scheduler.state_dict(),
        "scaler": built.scaler.state_dict() if built.scaler else None,
        "history": _plain(history),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": {"kind": numpy_state[0],
                      "keys": torch.from_numpy(np.asarray(numpy_state[1]).astype(np.int64)),
                      "pos": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                      "cached_gaussian": float(numpy_state[4])},
            "python": random.getstate(),
        },
    }


def _restore(last_path: Path, args, built: Built, manifest: dict) -> dict:
    """Load the last checkpoint's weights, optimizer, schedule and random streams, refuse
    a checkpoint from a different run, and record the resumption in the manifest."""
    saved = torch.load(last_path, map_location="cpu", weights_only=True)
    state = saved.get("resume")
    if state is None:
        raise ValueError(f"{last_path} carries no resume state")
    if state["epoch"] >= args.epochs:
        raise ValueError(f"{last_path} already reached epoch {state['epoch']} of {args.epochs}")
    stamped = saved["manifest"]
    for key in ("model", "folds", "recipe", "loss", "normalization"):
        if stamped[key] != manifest[key]:
            raise ValueError(f"cannot resume: '{key}' differs from the checkpoint's")
    if realized_training_recipe(stamped) != realized_training_recipe(manifest):
        raise ValueError("cannot resume: the training recipe differs from the checkpoint's")
    if stamped["inputs"]["flags"] != manifest["inputs"]["flags"]:
        raise ValueError("cannot resume: inputs 'flags' differ from the checkpoint's")
    _real_model(built.model).load_state_dict(saved["model_state_dict"])
    built.optimizer.load_state_dict(state["optimizer"])
    built.scheduler.load_state_dict(state["scheduler"])
    if built.scaler is not None and state["scaler"] is not None:
        built.scaler.load_state_dict(state["scaler"])
    rng = state["rng"]
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    numpy_state = rng["numpy"]
    np.random.set_state((numpy_state["kind"], numpy_state["keys"].numpy().astype(np.uint32),
                         numpy_state["pos"], numpy_state["has_gauss"],
                         numpy_state["cached_gaussian"]))
    random.setstate(rng["python"])
    manifest["resumed"] = {
        "from_epoch": state["epoch"],
        "times": int((stamped.get("resumed") or {}).get("times", 0)) + 1}
    return state


def _run_epochs(args, prep: Prepared, built: Built, manifest: dict, paths,
                resume: dict | None = None) -> dict:
    """Train every epoch, then score the last one on the held-out fold. Returns what the
    result and the dumps are built from."""
    last_path, final_path = paths
    device = prep.device
    shift_cols, stats = prep.shift_cols, prep.stats
    train_ds, eval_ds = prep.train_ds, prep.eval_ds
    train_loader, eval_loader = prep.train_loader, prep.eval_loader
    train_folds, atom_recipe = prep.train_folds, prep.atom_recipe
    model, optimizer = built.model, built.optimizer
    scheduler, scaler = built.scheduler, built.scaler
    group_weight, partner = built.group_weight, built.partner
    backbone_col, sidechain_weight = built.backbone_col, built.sidechain_weight
    swap_center, swap_scale = built.swap_center, built.swap_scale
    history, metrics, eval_predictions = [], None, None
    start_epoch = 1
    if resume is not None:
        start_epoch = resume["epoch"] + 1
        history = resume["history"]
        # The sampler seeds each epoch's order from its own epoch counter.
        train_loader.batch_sampler.epoch = resume["epoch"]

    def payload(epoch, backbone_z_mae):
        # The stamped recipe + stats make the checkpoint loadable by load_checkpoint and
        # rebuild the identical cloud at inference.
        return {"model_state_dict": _real_model(model).state_dict(), "model": MODEL_TAG,
                "dim": args.dim, "layers": args.layers, "n_shifts": len(shift_cols),
                "shift_cols": shift_cols, "stats": stats, "epoch": epoch,
                "backbone_z_mae": backbone_z_mae, "recipe": atom_recipe, "manifest": manifest}

    _log(f"train folds={train_folds} train={len(train_ds):,} "
         f"heldout={len(eval_ds):,} device={device}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total, batches, skipped, started = 0.0, 0, 0, time.time()
        for batch in train_loader:
            target = batch.pop("shift_target").to(device)
            smask = batch.pop("shift_mask").to(device)
            aa = batch["res_idx"].to(device)
            ambiguity = batch["ambiguity"].to(device) if "ambiguity" in batch else None
            inputs = _model_inputs(batch, device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            ctx = autocast("cuda") if scaler else nullcontext()
            with ctx:
                prediction, valid = model(**inputs)
                element, eq_w, eff_mask = swap_elements(
                    prediction, target, smask & valid, aa, group_weight, partner,
                    args.huber_delta, swap_center, swap_scale, ambiguity)
                loss = hierarchical_loss(element, eq_w, eff_mask, backbone_col, sidechain_weight)
            if not bool(torch.isfinite(loss)) or loss.grad_fn is None:
                skipped += 1
                continue
            if not _optimizer_step(loss, model, optimizer, scaler):
                skipped += 1
                continue
            total += loss.item()
            batches += 1
        if not batches:
            raise RuntimeError(f"all {skipped} training batches were skipped (non-finite)")
        scheduler.step()

        # The held-out fold is touched once, after the last epoch: it selects nothing.
        if epoch == args.epochs:
            metrics = evaluate(model, eval_loader, device, stats, shift_cols,
                               return_predictions=True, partner=partner.cpu().numpy())
            eval_predictions = metrics.pop("predictions")
        record = {"epoch": epoch, "loss": total / batches, "time": time.time() - started,
                  "skipped_batches": skipped}
        if metrics:
            record.update(metrics)
        history.append(record)
        _log(f"[{MODEL_TAG}] ep {epoch}/{args.epochs} loss {record['loss']:.4f}"
             + (f" bbz {metrics['backbone_z_mae']:.4f}" if metrics else "")
             + f" skip {skipped}")

        if (args.ckpt_every > 0 and epoch % args.ckpt_every == 0) or epoch == args.epochs:
            epoch_bbz = metrics["backbone_z_mae"] if metrics else float("nan")
            _atomic_torch_save({**payload(epoch, epoch_bbz),
                                "resume": _resume_state(epoch, built, history)},
                               last_path)
    _atomic_torch_save(payload(args.epochs, metrics["backbone_z_mae"]), final_path)
    return {"metrics": metrics, "eval_predictions": eval_predictions, "history": history}


def train(args):
    """Train one cross-validation member and write its checkpoints, metrics and dumps.

    Four phases, in order: settle the cohort and the loss weighting, build the model,
    stamp the run record, then train and score the last epoch. Each is a function below;
    this is the order they run in and the values they hand on.
    """
    prep = _prepare(args)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # --out names the member: the result is <member>.json and the checkpoints and the
    # prediction dump sit beside it under the same stem.
    if out.suffix in {".pt", ".pth", ""}:
        out = out.with_suffix(".json")
    base = out.with_suffix("")
    last_path, final_path = (Path(f"{base}{suffix}") for suffix in (".last.pt", ".final.pt"))
    built = _build(args, prep)
    manifest = _manifest(args, prep, built)
    resume = None
    if args.resume and last_path.exists():
        resume = _restore(last_path, args, built, manifest)
        _log(f"resuming after epoch {resume['epoch']} from {last_path}")
    outcome = _run_epochs(args, prep, built, manifest, (last_path, final_path), resume)

    shift_cols, eval_ds = prep.shift_cols, prep.eval_ds
    eval_predictions = outcome["eval_predictions"]
    result = {"model": MODEL_TAG,
              "selected_backbone_z_mae": outcome["metrics"]["backbone_z_mae"],
              "selection": "final", "selected_epoch": args.epochs,
              "ckpt": str(final_path), "last_ckpt": str(last_path), "final_ckpt": str(final_path),
              "resumed": manifest.get("resumed"),
              "manifest": manifest, "loss": manifest["loss"], "history": outcome["history"]}
    identity = dataset_identity(eval_ds)
    # Fixed-width Unicode keeps the NPZ loadable with allow_pickle=False; object
    # arrays turn a numeric artifact into arbitrary-code deserialization.
    def strings(name):
        return np.asarray(
            [str(r.get(name) or "") for r in identity], dtype=np.str_
        )
    preds_path = Path(f"{base}.preds.npz")
    np.savez(preds_path,
             pred_ppm=eval_predictions["pred_ppm"], true_ppm=eval_predictions["true_ppm"],
             resolved_true_ppm=eval_predictions["resolved_true_ppm"],
             mask=eval_predictions["mask"], effective_mask=eval_predictions["effective_mask"],
             residue_code=eval_predictions["residue_code"], pred_z=eval_predictions["pred_z"],
             target_z=eval_predictions["target_z"],
             resolved_target_z=eval_predictions["resolved_target_z"],
             ambiguity=eval_predictions["ambiguity"],
             shift_cols=np.asarray(shift_cols, dtype=np.str_),
             source=np.array([r.get("source", 0) for r in identity], dtype=np.int64),
             bmrb=strings("bmrb"), residue=strings("residue"),
             logical_entity_id=strings("logical_entity_id"), label_set_id=strings("label_set_id"),
             sample_id=strings("sample_id"), materialization_id=strings("materialization_id"),
             label_residue_id=strings("label_residue_id"),
             checkpoint_name=np.asarray(final_path.name, dtype=np.str_),
             checkpoint_epoch=np.asarray(args.epochs, dtype=np.int64),
             checkpoint_selection=np.asarray("final", dtype=np.str_))
    result["preds_npz"] = str(preds_path)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def add_train_arguments(parser):
    parser.add_argument("--cache", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=FINAL_TRAINING_RECIPE.epochs)
    parser.add_argument("--batch-size", type=int, default=FINAL_TRAINING_RECIPE.batch_size)
    parser.add_argument("--lr", type=float, default=FINAL_TRAINING_RECIPE.learning_rate)
    parser.add_argument("--eta-min", type=float)
    parser.add_argument("--weight-decay", type=float, default=FINAL_TRAINING_RECIPE.weight_decay)
    parser.add_argument("--dim", type=int, default=FINAL_TRAINING_RECIPE.dimension)
    parser.add_argument("--layers", type=int, default=FINAL_TRAINING_RECIPE.layers)
    parser.add_argument("--dropout", type=float, default=FINAL_TRAINING_RECIPE.dropout)
    parser.add_argument("--data-frac", type=float, default=0.0,
                        help="train on this nested share of the training rows (0 = all)")
    parser.add_argument("--seed", type=int, default=FINAL_TRAINING_RECIPE.seed)
    parser.add_argument("--ckpt-every", type=int,
                        default=FINAL_TRAINING_RECIPE.checkpoint_every)
    parser.add_argument("--huber-delta", type=float, default=FINAL_TRAINING_RECIPE.huber_delta)
    parser.add_argument("--exclude-flagged", required=True)
    parser.add_argument("--train-materializations", default=None,
                        help="JSON list of materialization ids; restricts which structures "
                             "the TRAINING folds contribute. The held-out fold is never "
                             "restricted, so arms stay scored on identical cells.")
    parser.add_argument("--train-sources", nargs="+",
                        help="structure sources the TRAINING folds contribute; the held-out "
                             "fold is never filtered")
    parser.add_argument("--compile", action="store_true")
    # Deterministic kernels by default so two runs of one command are comparable; turning
    # them off is recorded in the run metadata.
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                        default=FINAL_TRAINING_RECIPE.deterministic)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", required=True)
    # Every checkpoint written at --ckpt-every carries the optimizer, the schedule and
    # the random streams, so an interrupted member continues from its last one and ends
    # where the uninterrupted run would have.
    parser.add_argument("--resume", action="store_true",
                        help="continue from <out>.last.pt when it exists")
    parser.set_defaults(handler=train)
