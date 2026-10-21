#!/usr/bin/env python
"""Build the memory-mapped, per-fold label cache atom-transformer training reads,
so training never reparses CSVs/PDBs each epoch.

Cache layout per fold: <output_dir>/fold_{k}/{config.json, samples.npy,
structural/{shifts,shift_mask,ambiguity,residue_idx,
source_idx,materialization_idx}.npy, id_maps.json, ...}.
"""

import argparse
import gc
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import domain as identity
from ..chemistry import CHEMICALLY_IMPOSSIBLE_PAIRS
from ..structure import classify_atom
from . import norm
from .cache_columns import require_release_shift_columns
from .config import (
    ATOM_TO_IDX,
    BIG_STORAGE,
    MISMATCH_TO_IDX,
    RESIDUE_TO_IDX,
)
from .reref_utils import offset_columns

# Entity-grained cache identity: rows carry the full canonical identity
# minted in domain.py; the cache groups by materialization_id and keys residue
# lookups on comp_index_id.

# The canonical identity fields every materialized row must carry.
ROW_IDENTITY_FIELDS = (
    "materialization_id", "sample_id", "label_set_id", "structure_instance_id",
    "logical_entity_id", "source", "comp_index_id", "comp_id", "label_residue_id",
)


def _require_row_identity(row):
    missing = [f for f in ROW_IDENTITY_FIELDS if f not in row]
    if missing:
        raise ValueError(f"materialized row missing identity fields {missing}")
    identity._require_id(row["materialization_id"], "mz_")
    identity._require_id(row["sample_id"], "sa_")
    identity._require_id(row["label_set_id"], "ls_")
    identity._require_id(row["structure_instance_id"], "si_")
    identity._require_id(row["logical_entity_id"], "le_")
    identity._require_id(row["label_residue_id"], "lr_")
    identity.validate_source(row["source"])


def group_by_materialization(rows):
    """Group materialized rows by materialization_id."""
    groups: dict[str, list] = {}
    for row in rows:
        _require_row_identity(row)
        groups.setdefault(row["materialization_id"], []).append(row)
    return groups


def assert_unique_materialization_key(rows):
    """Every materialization_id must map to a single
    (sample_id, label_set_id, structure_instance_id) triple."""
    seen: dict[str, tuple] = {}
    for row in rows:
        _require_row_identity(row)
        key = (row["sample_id"], row["label_set_id"], row["structure_instance_id"])
        prior = seen.setdefault(row["materialization_id"], key)
        if prior != key:
            raise ValueError(
                f"materialization {row['materialization_id']} maps to two identity "
                f"triples: {prior} vs {key}")
    return seen


def assert_single_materialization_per_sample_source(rows):
    """A release may not hold a second materialization for one
    (logical_entity_id, source) -- a coordinate change replaces, never adds."""
    seen: dict[tuple, str] = {}
    for row in rows:
        _require_row_identity(row)
        key = (row["logical_entity_id"], row["source"])
        mzid = seen.setdefault(key, row["materialization_id"])
        if mzid != row["materialization_id"]:
            raise ValueError(
                f"second materialization for sample {key}: "
                f"{mzid} vs {row['materialization_id']}")
    return seen


def refuse_duplicate_comp_indices(rows):
    """Within one materialization, comp_index_id must be unique (residue lookups
    key on comp_index_id, so a duplicate would silently collide)."""
    for mzid, group in group_by_materialization(rows).items():
        seen = set()
        for row in group:
            comp = int(row["comp_index_id"])
            if comp in seen:
                raise ValueError(
                    f"duplicate comp_index_id {comp} within materialization {mzid}")
            seen.add(comp)


# Minted-id classes stored 1:1 per global index.
ID_MAP_ID_FIELDS = (
    "materialization_id", "sample_id", "label_set_id", "structure_instance_id",
    "logical_entity_id", "label_residue_id", "source", "comp_index_id", "comp_id",
)
# Expanded payload fields: every hashed id recomputes from id_maps alone. Absent
# fields store '' and are never omitted, so the residue-row grain stays 1:1.
ID_MAP_PAYLOAD_FIELDS = (
    "bmrb_id", "entity_id", "shift_list_id", "entity_assembly_id",
    "exact_polymer_sequence_sha256", "accession", "chain", "model_index",
    "coordinate_file_sha256", "preprocessing_state", "build_id",
)


def build_global_id_maps(rows):
    """Exact global maps for every ID class plus the expanded identity fields, keyed by
    the row's global index. Always written."""
    names = ID_MAP_ID_FIELDS + ID_MAP_PAYLOAD_FIELDS
    maps = {name: {} for name in names}
    for global_index, row in enumerate(rows):
        _require_row_identity(row)
        key = str(global_index)
        for name in ID_MAP_ID_FIELDS:
            maps[name][key] = (int(row["comp_index_id"])
                               if name == "comp_index_id" else row[name])
        for name in ID_MAP_PAYLOAD_FIELDS:
            v = row.get(name, "")
            maps[name][key] = "" if v is None else v
    return maps




def _fold_csv_dtypes(columns):
    """dtypes for a fold CSV: geometry columns float32, but ``*_shift`` columns stay
    float64 until rereferencing + the final single float32 round -- reading them as
    float32 first changes the baked target by an ulp."""
    dtypes = {'bmrb_id': str}
    dtypes.update({c: 'float32' for c in columns
                   if c.startswith(('dist_', 'cross_', 'sc_'))})
    dtypes.update({c: 'float64' for c in columns if c.endswith('_shift')})
    return dtypes


def _read_csv_chunked(path, usecols, dtype, chunksize=200_000):
    """Row-chunks of a wide CSV, for the caller to concatenate.

    Chunking is not a tuning knob: the C parser materializes ALL columns before applying
    usecols, so an unchunked read of a ~4.5k-column fold tokenizes the whole row and OOMs.
    Nor is low_memory=False -- per-chunk dtype inference SEGFAULTS the C parser on the sparse
    64-column shift table at 1.5M rows (measured: low_memory=True crashes, False reads
    1,509,465 rows). Neither changes the result.
    """
    return list(pd.read_csv(path, usecols=usecols, dtype=dtype,
                            low_memory=False, chunksize=chunksize))


REREF_KEY_COL = 'label_set_id'    # the only key the fold-array writer can look an offset up by
# The split stage names its fold CSVs structure_data_<DATASET>_fold_<f>.csv.
DATASET = 'unified'


# Rebuild-root isolation: a stale inherited SHIFT_CACHE_DIR/--output_dir must never
# rmtree a live cache; assert_safe_cache_dir gates every rmtree/write.
_DEFAULT_PROTECTED_CACHE_DIRS = ()


def _protected_cache_dirs():
    """Protected live-cache roots configured by the caller."""
    env = os.environ.get('SHIFT_PROTECTED_CACHE_DIRS')
    if env is not None:
        return tuple(p for p in env.split(os.pathsep) if p)
    return _DEFAULT_PROTECTED_CACHE_DIRS


def _is_within(path, root):
    """True if `path` equals or is a descendant of `root` (symlinks resolved)."""
    rp = os.path.realpath(path)
    rr = os.path.realpath(root)
    try:
        return os.path.commonpath([rp, rr]) == rr
    except ValueError:
        # Different drives / mixed absolute-relative — treat as not-within.
        return False


def assert_safe_cache_dir(cache_dir, rebuild_root=None):
    """Refuse (hard error) to rmtree/write into a protected live cache, or -- with an
    explicit rebuild root -- anywhere outside it. Called before the rmtree in
    build_cache_for_fold_entity and early in main() for a fail-fast."""
    target = os.path.realpath(cache_dir)
    if os.path.lexists(cache_dir) and not rebuild_root:
        raise SystemExit(
            f"REFUSING to replace existing cache at {target} without an explicit "
            "--rebuild_root / SHIFT_REBUILD_ROOT. The cache builder removes an "
            "existing target before rebuilding, so the deletion boundary must be "
            "named and checked."
        )
    for prot in _protected_cache_dirs():
        if _is_within(target, prot):
            raise SystemExit(
                f"REFUSING to build/delete cache at {target}: it resolves inside "
                f"protected live cache {os.path.realpath(prot)}. Point --output_dir "
                f"/ SHIFT_CACHE_DIR at a fresh rebuild root (or clear "
                f"SHIFT_PROTECTED_CACHE_DIRS if this root is the intended target).")
    if rebuild_root and not _is_within(target, rebuild_root):
        raise SystemExit(
            f"REFUSING to build/delete cache at {target}: it is not under the "
            f"declared rebuild root {os.path.realpath(rebuild_root)} "
            f"(--rebuild_root / SHIFT_REBUILD_ROOT).")


# Excluded from mean/std iff |value - median| > K * (1.4826 * MAD) per (residue,
# nucleus), K below -- keeps garbage deposits (e.g. sentinel -172/1567 ppm N) out of
# the stats so a poisoned label can't inflate a nucleus's std. Value QC lives here and
# in the flags stage; the parse stages do not mutate labels.
_STATS_MAD_K = {'ca_shift': 6.0, 'c_shift': 6.0, 'n_shift': 6.0,
                'h_shift': 6.0, 'cb_shift': 10.0, 'ha_shift': 8.0}
_STATS_MAD_K_DEFAULT = 6.0   # sidechain nuclei


def _robust_keep_mask(vals, col):
    """Boolean mask of values to keep for stats (robust MAD gate). vals: 1-D np array (no NaN)."""
    if len(vals) < 10:
        return np.ones(len(vals), dtype=bool)
    med = np.median(vals)
    mad = 1.4826 * np.median(np.abs(vals - med))
    if mad < 1e-6:
        return np.ones(len(vals), dtype=bool)
    k = _STATS_MAD_K.get(col, _STATS_MAD_K_DEFAULT)
    return np.abs(vals - med) <= k * mad


def _robust_mean_std(vals, col):
    """mean/std computed on the MAD-curated subset of `vals` (a pandas Series or array)."""
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return 0.0, 1.0
    keep = _robust_keep_mask(v, col)
    vk = v[keep]
    if len(vk) == 0:
        vk = v
    return float(vk.mean()), float(vk.std())


def _reref_values(df, col, reref_offsets, key_col='label_set_id'):
    """Return float64 observed values with the build's reref correction applied.

    ``key_col`` (label_set_id) scopes each offset to its own label set, never a
    sibling's. Missing that column with offsets in hand raises: the scale would
    otherwise fit one referencing convention and the targets another."""
    # copy=True is load-bearing: to_numpy() on a float64 column returns a view, and
    # this function runs twice per column -- an uncopied view double-applies the offset.
    values = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float64,
                                                              copy=True)
    if reref_offsets and key_col not in df.columns:
        raise ValueError(
            f"rereference offsets were supplied but the frame carries no {key_col!r} "
            f"column to join them on; the fit would silently use unreferenced ppm")
    if reref_offsets:
        offsets = df[key_col].astype(str).map(
            lambda b: reref_offsets.get(b, {}).get(col, 0.0)
        ).to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        values[valid] += offsets[valid]
    return values


def _flagged_keys_by_col(flagged):
    """Group flagged cells by shift column (input schema ``{label_residue_id:
    [shift_col, ...]}``); each ``by_col[col]`` is a set of
    ``label_residue_id`` strings."""
    by_col: dict = {}
    for owner, cells in (flagged or {}).items():
        for cell in cells:
            by_col.setdefault(str(cell), set()).add(str(owner))
    return by_col


def _flag_keep_mask(flag_keys, bad, n):
    """Boolean keep mask (True = keep) dropping rows whose ``label_residue_id``
    join key is in ``bad``."""
    return np.fromiter((key not in bad for key in flag_keys),
                       dtype=bool, count=n)


def compute_normalization_stats(df, shift_cols,
                                reref_offsets=None, flagged=None,
                                tiny_stats_floor=False, reref_key_col='label_set_id'):
    """Global mean/std per shift column and the per-nucleus target scale S_NUC from the
    training rows. Shift stats use the MAD-curated subset (see _robust_mean_std) so a
    physically impossible deposited value cannot inflate a nucleus's std."""
    stats = {}
    flagged_by_col = _flagged_keys_by_col(flagged)
    # Per-row join key: entity p100 flags join through the row's label_residue_id.
    if flagged_by_col and 'label_residue_id' not in df.columns:
        raise ValueError(
            "label-residue p100 flags require a 'label_residue_id' column to "
            "join flagged cells, but the training frame has none")
    flag_keys = (df['label_residue_id'].astype(str).to_numpy()
                 if 'label_residue_id' in df.columns else None)
    residue_code = df['residue_code'].astype(str).str.upper().to_numpy()
    snuc = {}
    snuc_counts = {}
    for col in shift_cols:
        if col in df.columns:
            values = _reref_values(df, col, reref_offsets, reref_key_col)
            keep = np.isfinite(values)
            bad = flagged_by_col.get(col)
            if bad:
                keep &= _flag_keep_mask(flag_keys, bad, len(df))
            vals = values[keep]
            if len(vals) > 0:
                m, sd = _robust_mean_std(vals, col)
                stats[col] = {'mean': m, 'std': max(sd, 0.1)}
            else:
                stats[col] = {'mean': 0.0, 'std': 1.0}

            # Per-model target scale: robust MAD of secondary shift, fit only on
            # this model's non-held-out folds. Centers are the fixed RC/BMRB values
            # in norm.py; cells without a center are excluded from this fit.
            center_by_aa = {aa: norm.get_center(aa, col)
                            for aa in set(residue_code)}
            centers = np.fromiter(
                ((center_by_aa.get(aa) if center_by_aa.get(aa) is not None
                  else np.nan) for aa in residue_code),
                dtype=np.float64, count=len(df))
            sk = keep & np.isfinite(centers)
            secondary = values[sk] - centers[sk]
            if len(secondary) < 10 and not tiny_stats_floor:
                raise ValueError(
                    f'cannot fit S_NUC[{col}] from only {len(secondary)} valid training cells')
            med = float(np.median(secondary)) if len(secondary) else 0.0
            scale = (float(1.4826 * np.median(np.abs(secondary - med)))
                     if len(secondary) else 0.1)
            if tiny_stats_floor:
                scale = max(scale, 0.1)
            if not np.isfinite(scale) or scale < 0.1:
                raise ValueError(f'invalid fitted S_NUC[{col}]={scale!r}')
            snuc[col] = scale
            snuc_counts[col] = len(secondary)

    stats['snuc'] = snuc
    stats['snuc_counts'] = snuc_counts
    stats['normalization'] = 'secondary_snuc_train_folds_v1'
    return stats


# Identity columns a fold CSV must carry. The cache is keyed by
# materialization_id, never the (bmrb, source) composite, so a multi-entity
# deposition doesn't collapse to one structure per (bmrb, source).
ENTITY_CACHE_COLS = (
    "materialization_id", "sample_id", "label_set_id", "structure_instance_id",
    "logical_entity_id", "label_residue_id", "comp_index_id", "comp_id",
)

# Columns only an entity-grained master carries (comp_index_id/comp_id excluded --
# deposition-grained masters can carry those too). Any of them marks an entity master.
ENTITY_SIGNAL_COLS = (
    "materialization_id", "sample_id", "label_set_id", "structure_instance_id",
    "logical_entity_id", "label_residue_id",
)


def _cache_is_entity_grained(df) -> bool:
    """True iff the df is a complete entity-grained master.

    A frame carrying any entity-identity column but missing one of ENTITY_CACHE_COLS
    is refused, since the (bmrb, source) grouping it would fall back to collapses
    sibling entities onto one cache row."""
    present_signal = [c for c in ENTITY_SIGNAL_COLS if c in df.columns]
    if not present_signal:
        return False  # deposition-grained master
    missing = [c for c in ENTITY_CACHE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"df carries entity-identity columns {present_signal} but is missing "
            f"required entity columns {missing}; refusing to fall back to the "
            f"(bmrb, source) grouping. Rebuild the "
            f"masters/split so every entity column is present, or drop them all.")
    return True


def _mat_group_sort_key(group_key, source_order):
    """Deterministic materialization ordering: bmrb (numeric), source, ids."""
    mzid, bmrb, source, lsid = group_key
    try:
        bnum = int(str(bmrb))
    except (TypeError, ValueError):
        bnum = 1 << 62
    return (bnum, source_order.get(str(source), len(source_order)), str(lsid), mzid)


@dataclass
class FoldArrays:
    """Everything one fold's array pass produces: the per-row structural arrays, the
    row-indexed identity/lookup maps, and the counts the writer reports."""
    residue_idx: np.ndarray
    mismatch_idx: np.ndarray
    shifts: np.ndarray
    shift_mask: np.ndarray
    ambiguity: np.ndarray
    source_idx: np.ndarray
    mat_idx: np.ndarray
    identity_rows: list = field(default_factory=list)   # per-global-index identity (id_maps)
    idx_to_bmrb: dict = field(default_factory=dict)
    global_to_resid: dict = field(default_factory=dict)         # global -> comp_index_id
    global_to_struct_resid: dict = field(default_factory=dict)  # global -> struct_res_id
    materialization_ids: list = field(default_factory=list)     # index == materialization code
    samples_list: list = field(default_factory=list)
    n_bad_res: int = 0
    n_bad_targets: int = 0

    @property
    def total_residues(self):
        return len(self.residue_idx)

    @property
    def n_mats(self):
        return len(self.materialization_ids)


def _allocate_fold_arrays(total_residues, n_shifts) -> FoldArrays:
    """Zeroed per-row arrays for one fold."""
    return FoldArrays(
        residue_idx=np.zeros(total_residues, dtype=np.int32),
        mismatch_idx=np.zeros(total_residues, dtype=np.int32),
        shifts=np.zeros((total_residues, n_shifts), dtype=np.float32),
        shift_mask=np.zeros((total_residues, n_shifts), dtype=bool),
        ambiguity=np.zeros((total_residues, n_shifts), dtype=np.int8),
        source_idx=np.zeros(total_residues, dtype=np.int8),
        mat_idx=np.zeros(total_residues, dtype=np.int32),
    )


def _ordered_materialization_groups(df, source_order):
    """Group rows by materialization_id -- one entity x source x structure per group --
    in the deterministic (bmrb, source, ids) order the cache is laid out in."""
    df = df.copy()
    df['comp_index_id'] = pd.to_numeric(df['comp_index_id'], errors='coerce')
    # groupby drops null keys silently: a blank materialization_id would vanish
    # before _require_row_identity ever saw it, shrinking total_residues unnoticed.
    n_keyless = int(df['materialization_id'].isna().sum())
    if n_keyless:
        raise ValueError(
            f"{n_keyless} row(s) carry no materialization_id; grouping would drop "
            f"them from the fold cache without a word. Rebuild the split so every "
            f"row has its full identity.")
    groups = {}
    for mzid, gdf in df.groupby('materialization_id', sort=False):
        r0 = gdf.iloc[0]
        key = (str(mzid), str(r0['bmrb_id']) if 'bmrb_id' in gdf.columns else '',
               str(r0['source']), str(r0['label_set_id']))
        groups[key] = gdf.sort_values('comp_index_id').reset_index(drop=True)
    return groups, sorted(groups, key=lambda k: _mat_group_sort_key(k, source_order))


def _identity_row(gdf, i, source, bare_bmrb, comp):
    """One row's canonical identity, with the expanded payload so id_maps recomputes every
    hash. Fields the master does not carry store ''."""
    def cell(col):
        return '' if col not in gdf.columns else (
            '' if pd.isna(gdf[col].iloc[i]) else str(gdf[col].iloc[i]))
    row = {name: str(gdf[name].iloc[i]) for name in (
        "materialization_id", "sample_id", "label_set_id", "structure_instance_id",
        "logical_entity_id", "label_residue_id")}
    row["source"] = str(source)
    row["comp_index_id"] = comp
    row["comp_id"] = str(gdf['comp_id'].iloc[i]).upper()
    row["bmrb_id"] = str(bare_bmrb)
    row.update({name: cell(name) for name in (
        "entity_id", "shift_list_id", "entity_assembly_id",
        "exact_polymer_sequence_sha256", "accession", "chain", "model_index",
        "coordinate_file_sha256", "preprocessing_state", "build_id")})
    return row


def _fill_residue_rows(arrays, gdf, start, source, bare_bmrb):
    """Residue/mismatch codes, the back-compat lookups, and one identity row per residue."""
    n = len(gdf)
    comp_idx = gdf['comp_index_id'].to_numpy()
    struct_rid = pd.to_numeric(gdf.get('struct_res_id'), errors='coerce') \
        if 'struct_res_id' in gdf.columns else pd.Series([np.nan] * n)
    residue_codes = gdf['residue_code'].fillna('UNK').astype(str).str.upper().to_numpy() \
        if 'residue_code' in gdf.columns else \
        gdf['comp_id'].astype(str).str.upper().to_numpy()
    mismatch = gdf['mismatch_type'].fillna('UNK').astype(str).to_numpy() \
        if 'mismatch_type' in gdf.columns else np.array(['match'] * n)

    for i in range(n):
        g = start + i
        comp = int(comp_idx[i]) if np.isfinite(comp_idx[i]) else -(g + 1)
        arrays.residue_idx[g] = RESIDUE_TO_IDX.get(residue_codes[i], RESIDUE_TO_IDX['UNK'])
        arrays.mismatch_idx[g] = MISMATCH_TO_IDX.get(mismatch[i], MISMATCH_TO_IDX['UNK'])
        arrays.idx_to_bmrb[str(g)] = str(bare_bmrb)
        arrays.global_to_resid[str(g)] = comp
        sr = struct_rid.iloc[i]
        arrays.global_to_struct_resid[str(g)] = int(sr) if pd.notna(sr) else None
        arrays.identity_rows.append(_identity_row(gdf, i, source, bare_bmrb, comp))


def _bake_shift_columns(arrays, gdf, start, key, shift_cols, ls_off, ambig_col_for):
    """Bake one materialization's deposited ppm: rereference by label set, mask the
    non-finite, carry the ambiguity codes. ``ambig_col_for`` is {} when the frame codes none."""
    mzid, _bmrb, _source, lsid = key
    n = len(gdf)
    for si, col in enumerate(shift_cols):
        if col not in gdf.columns:
            continue
        vals = gdf[col].to_numpy(dtype=np.float64, copy=True)
        invalid_infinities = np.isinf(vals)
        if invalid_infinities.any():
            raise ValueError(
                f"materialization {mzid} column {col} contains "
                f"{int(invalid_infinities.sum())} infinite deposited shift(s)"
            )
        valid = np.isfinite(vals)
        residue_column = "residue_code" if "residue_code" in gdf.columns else "comp_id"
        residue_names = gdf[residue_column].fillna("UNK").astype(str).str.upper().to_numpy()
        valid &= np.asarray([
            (residue, col) not in CHEMICALLY_IMPOSSIBLE_PAIRS
            for residue in residue_names
        ])
        off = ls_off.get(col)
        if off is not None:
            offset = float(off)
            if not np.isfinite(offset):
                raise ValueError(
                    f"label set {lsid} has a non-finite rereferencing offset "
                    f"for {col}: {off!r}"
                )
            vals = np.where(valid, vals + offset, vals)
            if not np.all(np.isfinite(vals[valid])):
                raise ValueError(
                    f"label set {lsid} rereferencing produced non-finite {col}"
                )
        arrays.shift_mask[start:start + n, si] = valid
        arrays.shifts[start:start + n, si] = np.where(valid, vals, 0.0)
        acol = ambig_col_for.get(col)
        if acol is not None and acol in gdf.columns:
            acodes = pd.to_numeric(gdf[acol], errors='coerce').to_numpy()
            acodes = np.where(valid & ~np.isnan(acodes), acodes, 0.0)
            arrays.ambiguity[start:start + n, si] = acodes.astype(np.int8)


def _loss_mask_unusable_residues(arrays):
    """Loss-mask gap/mismatch/insertion-code/wrong-residue-code residues -- the structure
    residue is absent, disagrees, isn't representable in the integer atom cache,
    or the label code contradicts the polymer token; none may train against a wrong
    atom cloud or wrong amino-acid label."""
    bad = np.isin(arrays.mismatch_idx,
                  [MISMATCH_TO_IDX['gap_in_structure'], MISMATCH_TO_IDX['mismatch'],
                   MISMATCH_TO_IDX['insertion_code_unsupported'],
                   MISMATCH_TO_IDX['label_residue_code_mismatch']])
    arrays.n_bad_res = int(bad.sum())
    arrays.n_bad_targets = int(arrays.shift_mask[bad].sum())
    arrays.shift_mask[bad, :] = False
    print(f"      Loss-mask: excluded {arrays.n_bad_res:,} gap/mismatch residues "
          f"({arrays.n_bad_targets:,} shift targets)")


def _fill_fold_arrays(df, fold, shift_cols, source_names, reref_offsets,
                      flagged) -> FoldArrays:
    """Fill one fold's arrays from the materialized frame. Everything here writes arrays;
    nothing here touches disk."""
    source_to_code = {s: i for i, s in enumerate(source_names)}
    groups, ordered_keys = _ordered_materialization_groups(df, source_to_code)
    total_residues = sum(len(g) for g in groups.values())
    print(f"    Building entity cache for fold {fold}...")
    print(f"      Materializations (entity x source): {len(ordered_keys)}")
    print(f"      Total residues: {total_residues:,}")
    print(f"      Shift columns: {len(shift_cols)}")

    arrays = _allocate_fold_arrays(total_residues, len(shift_cols))
    ambig_col_for = {col: col[:-len('_shift')] + '_ambiguity_code' for col in shift_cols}
    if not any(ac in df.columns for ac in ambig_col_for.values()):
        ambig_col_for = {}
    col_to_si = {c: i for i, c in enumerate(shift_cols)}

    start = 0
    for mat_code, key in enumerate(tqdm(ordered_keys, desc="      Processing entity")):
        mzid, bare_bmrb, source, lsid = key
        gdf = groups[key]
        arrays.materialization_ids.append(mzid)
        n = len(gdf)
        # An unlisted source is refused: defaulting it to code 0 would stamp its rows
        # as the first source (af), and a source-stratified metric would then report
        # crystal rows as AlphaFold.
        if source not in source_to_code:
            raise ValueError(
                f"materialization {mzid} has source {source!r}, which is not in "
                f"source_names {list(source_names)}; refusing to code it as "
                f"{source_names[0]!r}")
        arrays.source_idx[start:start + n] = source_to_code[source]
        arrays.mat_idx[start:start + n] = mat_code
        _fill_residue_rows(arrays, gdf, start, source, bare_bmrb)
        _bake_shift_columns(arrays, gdf, start, key, shift_cols,
                            (reref_offsets or {}).get(str(lsid), {}), ambig_col_for)
        start += n

    # Flags apply to exactly the flagged (label_residue_id, col) cells.
    if flagged:
        for g, row in enumerate(arrays.identity_rows):
            for col in flagged.get(row["label_residue_id"], ()):
                si = col_to_si.get(col)
                if si is not None:
                    arrays.shift_mask[g, si] = False

    _loss_mask_unusable_residues(arrays)

    for g in range(total_residues):
        if arrays.shift_mask[g].any():
            arrays.samples_list.append((g, int(arrays.mat_idx[g])))
    return arrays


def _stats_for_json(stats, shift_cols):
    """The stats block copied into the fold config."""
    out = {}
    for col in shift_cols:
        if col in stats:
            out[col] = {'mean': float(stats[col]['mean']),
                        'std': float(stats[col]['std'])}
    for key in ('snuc', 'snuc_counts', 'normalization', 'normalization_fit'):
        if key in stats:
            out[key] = stats[key]
    return out


def _fold_config(arrays, shift_cols, stats, no_h, source_names):
    """The fold's config.json payload, including the entity marker."""
    config = {
        'n_atom_types': len(ATOM_TO_IDX),
        'n_shifts': len(shift_cols),
        'has_cross_features': False,
        'total_residues': arrays.total_residues,
        'n_proteins': arrays.n_mats,
        'n_materializations': arrays.n_mats,
        'n_samples': len(arrays.samples_list),
        'shift_cols': shift_cols,
        'shift_storage': 'ppm_float32',
        'shift_storage_version': 1,
        'has_ppm_shifts': True,
        'entity_grained': True,
        'stats': _stats_for_json(stats, shift_cols),
        'no_h': bool(no_h),
        'source_names': list(source_names),
    }
    return config


def _write_fold_cache(cache_path, arrays, fold, shift_cols, stats, no_h, source_names):
    """Serialize one filled fold: the identity assertions, structural arrays, exact global
    id maps and config. Nothing here changes an array."""
    assert_unique_materialization_key(arrays.identity_rows)
    assert_single_materialization_per_sample_source(arrays.identity_rows)
    refuse_duplicate_comp_indices(arrays.identity_rows)
    id_maps = build_global_id_maps(arrays.identity_rows)

    total_residues, n_mats = arrays.total_residues, arrays.n_mats
    sd = cache_path / 'structural'
    np.save(sd / 'residue_idx.npy', arrays.residue_idx)
    np.save(sd / 'shifts.npy', arrays.shifts)
    np.save(sd / 'shift_mask.npy', arrays.shift_mask)
    np.save(sd / 'source_idx.npy', arrays.source_idx)
    np.save(sd / 'materialization_idx.npy', arrays.mat_idx)
    # Ambiguity is always written so swap-loss has its codes; an all-zero
    # array means the fold carried no coded cells.
    np.save(sd / 'ambiguity.npy', arrays.ambiguity)
    with open(sd / 'bmrb_mapping.json', 'w') as f:
        json.dump(arrays.idx_to_bmrb, f)
    with open(sd / 'global_to_resid.json', 'w') as f:
        json.dump(arrays.global_to_resid, f)
    with open(sd / 'global_to_struct_resid.json', 'w') as f:
        json.dump(arrays.global_to_struct_resid, f)
    with open(sd / 'materialization_ids.json', 'w') as f:
        json.dump(arrays.materialization_ids, f)
    with open(sd / 'id_maps.json', 'w') as f:
        json.dump(id_maps, f)
    # {global_idx: label_residue_id} map that quality_control.derive_p100_flags
    # consumes; without it p100 falls back to collision-prone BMRB/residue keys.
    with open(sd / 'global_to_label_residue.json', 'w') as f:
        json.dump(id_maps['label_residue_id'], f)

    samples = np.array(arrays.samples_list, dtype=np.int32)
    np.save(cache_path / 'samples.npy', samples)
    n_samples = len(arrays.samples_list)
    print(f"      Samples with shifts: {n_samples:,} / {total_residues:,} residues")
    print(f"      Wrote id_maps.json ({len(arrays.identity_rows):,} rows), "
          f"materialization_idx ({n_mats} materializations)")

    with open(cache_path / 'config.json', 'w') as f:
        json.dump(_fold_config(arrays, shift_cols, stats, no_h, source_names), f, indent=2)

    return {'fold': fold, 'n_proteins': n_mats, 'n_materializations': n_mats,
            'total_residues': total_residues, 'n_samples': n_samples,
            'residues_no_shifts': total_residues - n_samples}


def build_cache_for_fold_entity(
    df: pd.DataFrame,
    fold: int,
    shift_cols: list,
    stats: dict,
    cache_dir: str,
    reref_offsets: dict | None = None,
    flagged: dict | None = None,
    no_h: bool = False,
    source_names: list | None = None,
    rebuild_root: str | None = None,
):
    """Build the entity-grained lean label cache for one fold.

    Rows group by ``materialization_id`` (one entity x source x structure = one
    span). Residue lookups key on ``comp_index_id``; ``global_to_struct_resid.json``
    maps each label residue to its structure residue so the atom stage binds per
    entity, not a collided (bmrb, source) residue number. Also writes
    ``structural/id_maps.json`` (every id, row-indexed) and
    ``materialization_idx.npy``/``materialization_ids.json``.

    ``reref_offsets`` keys on ``label_set_id`` and ``flagged`` on ``label_residue_id``.
    """
    cache_path = Path(cache_dir)
    assert_safe_cache_dir(str(cache_path), rebuild_root)
    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True)
    (cache_path / 'structural').mkdir()

    if source_names is None:
        source_names = sorted(df['source'].dropna().astype(str).unique())
    arrays = _fill_fold_arrays(df, fold, shift_cols, source_names, reref_offsets,
                               flagged)
    summary = _write_fold_cache(cache_path, arrays, fold, shift_cols, stats,
                                no_h, source_names)
    del arrays
    gc.collect()
    return summary


def add_cache_arguments(parser):
    """Cache-stage CLI surface, separate from ``main`` so the defaults are testable
    (matching ``add_build_arguments``/``add_train_arguments``)."""
    parser.add_argument(
        '--data_dir', default='./data',
        help='Directory containing the compiled CSV (default: ./data)',
    )
    parser.add_argument(
        '--output_dir', default=None,
        help='Output directory for caches (default: <data_dir>/cache)',
    )
    parser.add_argument(
        '--rebuild_root', default=os.environ.get('SHIFT_REBUILD_ROOT'),
        help='Optional fresh rebuild root (or via SHIFT_REBUILD_ROOT). When set, '
             'every fold cache dir must resolve under it or the build is refused. '
             'Guards against a stale --output_dir/SHIFT_CACHE_DIR that would rmtree '
             'a live cache. A protected live cache (default '
             'SHIFT_PROTECTED_CACHE_DIRS) is refused '
             'unconditionally, rebuild-root or not.',
    )
    parser.add_argument(
        '--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
        help='Which folds to build caches for (default: 1 2 3 4 5)',
    )
    parser.add_argument(
        '--tiny_stats_floor', action='store_true',
        help='Small online smoke builds only: permit fewer than 10 target cells '
             'and floor fitted S_NUC at 0.1. Production builds must leave this off.',
    )
    parser.add_argument(
        '--reref_offsets', default=None,
        help='Optional reref_offsets.json (RefDB re-referencing): per-protein '
             'per-nucleus ppm offsets ADDED to observed shifts before z-norm. '
             'Non-circular (reads the canonical table, not a prior reref cache). '
             'Keyed bmrb -> {"applied": {N/C/CA/CB: ppm}}; remapped to *_shift cols.')
    parser.add_argument(
        '--flagged', default=None,
        help='Explicit QC authority. An empty JSON object declares an unfiltered p100 '
             'calibration cache; the final build supplies the composed p100-plus-structural '
             'map. It affects normalization only; shift_mask remains the observation authority.')
    parser.add_argument(
        '--no_h', action='store_true',
        help='Heavy-atom-only structural features, the only supported mode; '
             'H *shift* targets are kept, H atom references are not stored.')
    parser.add_argument(
        '--source_build_logs', nargs='+', default=None,
        help='Unified cache: per-source build_log specs as source:logpath:dataset '
             '(e.g. af:/.../build_log.csv:alphafold). Presence marks a unified '
             'build (source_names default [af,nmr,xtal]).')
    parser.add_argument(
        '--source_names', nargs='+', default=None,
        help='Unified cache: canonical ordered source names (default '
             "['af','nmr','xtal']). Sets the source_idx code order + "
             "config['source_names'].")


@dataclass
class Schema:
    """The fold CSV's column surface: what every later pass selects on."""
    columns: list
    shift_cols: list
    fold_col: str


@dataclass
class Curation:
    """The named scientific inputs a build may be curated with. A path that resolves to
    nothing is refused rather than treated as an absent input."""
    reref_offsets: dict | None
    flagged: dict


def _resolve_output(args):
    """Default the cache root, then refuse a protected/out-of-root one before any
    makedirs/rmtree. Guarding the parent rejects every fold dir; the same gate re-runs per
    fold before each rmtree (defense in depth)."""
    if args.output_dir is None:
        # Caches are large, so they default to the big-storage root.
        ds_name = os.path.basename(os.path.abspath(args.data_dir))
        args.output_dir = os.path.join(BIG_STORAGE, f'{ds_name}_cache')
    assert_safe_cache_dir(args.output_dir, args.rebuild_root)
    return args.output_dir


def _fold_csv(args, fold):
    """The per-fold CSV the split stage writes."""
    return os.path.join(args.data_dir, f'structure_data_{DATASET}_fold_{fold}.csv')


def _resolve_header_csv(args):
    """A per-fold CSV to read column headers from; every fold shares one header."""
    for f in list(args.folds) + list(range(0, 7)):
        cand = _fold_csv(args, f)
        if os.path.exists(cand):
            return cand
    candidates = [f for f in os.listdir(args.data_dir) if f.endswith('.csv')]
    raise FileNotFoundError(
        f"Cannot find dataset CSV in {args.data_dir}. "
        f"Available CSVs: {candidates}"
    )


def _discover_schema(csv_path) -> Schema:
    """Header-only read: the release shift targets and which column carries the fold
    assignment."""
    print(f"Reading column headers from {csv_path}...")
    all_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    print(f"  {len(all_columns)} columns total")

    try:
        shift_cols = require_release_shift_columns(all_columns)
    except ValueError as exc:
        raise SystemExit(f"ERROR: invalid release target schema: {exc}") from exc

    fold_col = 'split' if 'split' in all_columns else 'fold'
    if fold_col not in all_columns:
        raise ValueError("CSV must contain a 'split' or 'fold' column")
    return Schema(columns=all_columns, shift_cols=shift_cols, fold_col=fold_col)


def _validate_inputs(args):
    """Refuse to start when a requested fold CSV is missing."""
    missing = [_fold_csv(args, f) for f in args.folds if not os.path.isfile(_fold_csv(args, f))]
    if missing:
        raise SystemExit(
            "ERROR: requested fold CSV(s) missing; run the split stage first:\n  "
            + "\n  ".join(missing))


def _load_light_frame(args, schema):
    """The light columns the normalization fit needs, without the thousands of geometry
    columns. Folds 1..5 are always read."""
    print(f"  Shift columns:    {len(schema.shift_cols)}")

    light_cols = {'bmrb_id', 'residue_id', 'residue_code', schema.fold_col,
                  'source',             # 'source' kept if present (unified cache)
                  'label_set_id',       # entity reref keys stats on label_set_id
                  'label_residue_id'}   # entity p100 flags join the S_NUC fit on this
    light_cols.update(schema.shift_cols)
    light_cols = sorted(c for c in light_cols if c in schema.columns)

    print(f"\nLoading {len(light_cols)} light columns for normalization stats...")
    fold_files = [_fold_csv(args, f) for f in range(1, 6)]
    if missing := [f for f in fold_files if not os.path.exists(f)]:
        raise SystemExit(f"ERROR: normalization always reads folds 1..5, but {missing} "
                         "are missing. Re-run the split stage.")
    light_parts = []
    for ff in fold_files:
        usecols = [c for c in light_cols if c in pd.read_csv(ff, nrows=0).columns]
        chunks = _read_csv_chunked(ff, usecols, {'bmrb_id': str})
        light_parts.extend(chunks)
        print(f"    {os.path.basename(ff)}: {sum(len(c) for c in chunks):,} rows")
    light_df = pd.concat(light_parts, ignore_index=True)
    del light_parts
    gc.collect()
    print(f"  {len(light_df):,} residues from {light_df['bmrb_id'].nunique():,} proteins")

    fold_counts = light_df.groupby(schema.fold_col)['bmrb_id'].nunique()
    print("  Fold distribution:")
    for f_val, cnt in fold_counts.items():
        print(f"    Fold {f_val}: {cnt} proteins")
    return light_df


def _load_reref_offsets(args):
    """reref_offsets.json: keyed by label_set_id -> per-nucleus offset dict; offset_columns()
    spreads each offset over the *_shift columns of its reference channel. Two shapes
    accepted: {ls_...: {'offsets': {CA,CB,CO,N: v}}} (reref.entity_reref_main) or
    {ls_...: {'applied': {CA,CB,C,N: v}}}."""
    if not args.reref_offsets:
        return None
    with open(args.reref_offsets) as fh:
        raw = json.load(fh)
    offsets = {}
    for key, rec in raw.items():
        applied = rec.get('offsets', rec.get('applied', rec)) if isinstance(rec, dict) else {}
        cols = offset_columns(applied)
        if cols:
            offsets[str(key)] = cols
    # Entity-grained reref keys on label_set_id (ls_...) only; a bmrb-keyed table
    # would silently reach the stats fit but never the stored ppm (while still
    # stamping reref_baked=True), so any non-ls_ key is refused here.
    stray = sorted(k for k in offsets if not str(k).startswith('ls_'))
    if stray:
        raise SystemExit(
            f"--reref_offsets {args.reref_offsets!r} is not label-set keyed "
            f"(e.g. {stray[:3]}); the entity cache applies offsets by "
            f"label_set_id only, so these would silently reach the "
            f"normalization fit but never the stored ppm.")
    print(f"  Re-referencing ON ({REREF_KEY_COL}): {len(offsets):,} "
          f"label sets/proteins with offsets from {args.reref_offsets}")
    return offsets


def _load_flag_table(args):
    """Load an explicit QC authority, including the empty calibration authority."""
    if args.flagged:
        path = Path(args.flagged)
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"--flagged authority is missing or empty: {path}")
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise SystemExit(f"--flagged authority must contain a JSON object: {path}")
        mode = "unfiltered calibration" if not loaded else f"{len(loaded):,} flagged keys"
        print(f"  QC authority: {mode} from {path}")
        return {str(key): value for key, value in loaded.items()}
    return {}


def _load_curation_inputs(args) -> Curation:
    """The two named curation artifacts; an empty one is refused."""
    return Curation(reref_offsets=_load_reref_offsets(args), flagged=_load_flag_table(args))


def _fit_fold_stats(fold, args, light_df, schema, curation):
    """Fit this model's target units, which depend only on its four optimization folds:
    fold_k fits on folds 1..5 excluding k (0=UCB-200 and 6=holdout are always
    excluded)."""
    excluded = {fold, 0, 6}
    train_mask = ~light_df[schema.fold_col].isin(excluded)
    print(f"    Train-only stats: excluding folds {sorted(excluded)} "
          f"(held-out {fold} + UCB-200 fold 0 + holdout fold 6)")
    # Target units may not be fit on an empty frame: under --tiny_stats_floor every
    # per-column guard is satisfied by the 0.1 floor, so an empty frame would write
    # S_NUC=floor.
    if not train_mask.any():
        raise SystemExit(
            f"ERROR: fold {fold} has NO training rows to fit normalization on "
            f"(folds present: "
            f"{sorted(light_df[schema.fold_col].dropna().unique().tolist())}).")
    stats = compute_normalization_stats(
        light_df[train_mask], schema.shift_cols,
        reref_offsets=curation.reref_offsets, flagged=curation.flagged,
        tiny_stats_floor=args.tiny_stats_floor,
        reref_key_col=REREF_KEY_COL)
    fit_folds = sorted(int(x) for x in
                       light_df.loc[train_mask, schema.fold_col].dropna().unique())
    stats['normalization_fit'] = {
        'policy': 'cv_train_folds_only',
        'heldout_fold': int(fold),
        'fit_folds': fit_folds,
        'always_excluded_folds': [0, 6],
        'reref_applied_before_fit': bool(curation.reref_offsets),
        'flagged_cells_excluded': bool(curation.flagged),
        'tiny_stats_floor': bool(args.tiny_stats_floor),
    }
    n_train = train_mask.sum()
    print(f"    Stats from {light_df.loc[train_mask, 'bmrb_id'].nunique()} "
          f"training proteins ({n_train:,} residues)")
    return stats


def _load_fold_frame(args, fold):
    """This fold's full-width rows from the per-fold CSV the split stage wrote.

    Under --no_h the ~1.5k H-involving dist columns are discarded anyway, so they are never
    read (usecols) -- the dominant RAM win, letting the 1.2M-row fold fit in <94 GB; float32
    on the remaining geometry ~halves what is left. Shift targets stay float64 until
    rereferencing and the final single float32 round.
    """
    fold_csv = _fold_csv(args, fold)
    print(f"    Loading per-fold file: {fold_csv}")
    hdr = list(pd.read_csv(fold_csv, nrows=0).columns)
    if args.no_h:
        def _is_hdist(c):
            if not c.startswith('dist_'):
                return False
            pp = c.split('_')
            return len(pp) >= 3 and (classify_atom(pp[1]) == 'hydrogen'
                                     or classify_atom(pp[2]) == 'hydrogen')
        usecols = [c for c in hdr if not _is_hdist(c)]
        print(f"    --no_h fold read: {len(hdr)} -> {len(usecols)} cols "
              f"(dropped {len(hdr) - len(usecols)} H-dist before load)")
    else:
        usecols = hdr
    parts = _read_csv_chunked(fold_csv, usecols, _fold_csv_dtypes(usecols),
                              chunksize=100_000)
    fold_df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return fold_df


def _build_one_fold(fold, args, light_df, schema, curation, source_names):
    """Fit this fold's stats, load its rows, and write its cache."""
    print(f"\n{'=' * 50}")
    print(f"  FOLD {fold}")
    print(f"{'=' * 50}")

    stats = _fit_fold_stats(fold, args, light_df, schema, curation)
    t0 = time.time()
    fold_df = _load_fold_frame(args, fold)
    print(f"    Loaded {fold_df['bmrb_id'].nunique()} proteins, "
          f"{len(fold_df):,} residues in {time.time()-t0:.1f}s")

    # Build cache: entity-grained -- asserted below before grouping by
    # materialization_id so a multi-entity deposition never collapses.
    if not _cache_is_entity_grained(fold_df):
        raise SystemExit(
            f"fold {fold} CSV is not entity-grained (missing entity identity "
            f"columns); rebuild the split so every entity column is present.")
    t0 = time.time()
    summary = build_cache_for_fold_entity(
        df=fold_df,
        fold=fold,
        shift_cols=schema.shift_cols,
        stats=stats,
        cache_dir=os.path.join(args.output_dir, f'fold_{fold}'),
        reref_offsets=curation.reref_offsets,
        # p100 flags feed the S_NUC fit above but do not mask shift_mask: the cache
        # stays the unmasked observation authority, so flags can be recomputed against
        # the observed shifts.
        flagged=None,
        no_h=args.no_h,
        source_names=source_names,
        rebuild_root=args.rebuild_root,
    )
    summary['build_time_seconds'] = time.time() - t0
    print(f"    Cache built in {summary['build_time_seconds']:.1f}s")

    del fold_df
    gc.collect()
    return summary


def _print_summary(summaries):
    """Per-fold counts to the console."""
    print("\n" + "=" * 60)
    for fold, summary in sorted(summaries.items()):
        print(f"\n  Fold {fold}:")
        print(f"    Proteins:                {summary['n_proteins']}")
        print(f"    Total residues:          {summary['total_residues']:,}")
        print(f"    Samples (with shifts):   {summary['n_samples']:,}")
        print(f"    Residues excluded (no shifts): {summary['residues_no_shifts']:,}")
        print(f"    Build time:              {summary['build_time_seconds']:.1f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Build memory-mapped structure-only training caches.',
    )
    add_cache_arguments(parser)
    args = parser.parse_args()
    _resolve_output(args)

    # source_idx is canonical, never derived per fold: otherwise code 1 could mean
    # nmr in one fold and xtal in another, which a source-stratified metric reads as real.
    source_names = (args.source_names if args.source_names is not None
                    else list(identity.VALID_SOURCES))

    print("=" * 60)
    print("Training Cache Builder (structure-only)")
    print("=" * 60)
    print(f"  Data directory:        {args.data_dir}")
    print(f"  Output directory:      {args.output_dir}")
    print(f"  Folds:                 {args.folds}")
    print()

    schema = _discover_schema(_resolve_header_csv(args))
    _validate_inputs(args)
    light_df = _load_light_frame(args, schema)

    os.makedirs(args.output_dir, exist_ok=True)
    curation = _load_curation_inputs(args)

    summaries = {}
    for fold in args.folds:
        summaries[fold] = _build_one_fold(fold, args, light_df, schema, curation, source_names)
    del light_df

    _print_summary(summaries)
    print("\nDone!")


if __name__ == "__main__":
    main()
