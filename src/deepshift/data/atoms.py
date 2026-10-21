"""Build the final contact-only atom arrays for the atom transformer.

Uses the same PDB and representative NMR model as the structural cache and the same
recipe implementation as prediction. Writes aligned arrays under ``<fold>/atom/``:
    atom_type.npy        (N, M)   int16   atom-type idx (pad = n_atom_types)
    atom_offset.npy      (N, M)   int8    residue-offset code (pad = n_offset)
    atom_xyz.npy         (N, M, 3) float16 coords (model uses only pairwise dist)
    atom_count.npy       (N,)     int16   #real atoms (mask = arange < count)
    nucleus_atom_idx.npy (N, S)   int16   index into M of each shift nucleus, -1 absent
    precap_atom_count.npy (N,)    int32   atoms gathered before the M cap
    atom_truncated.npy   (N,)     bool    precap exceeded M

The cap M is fixed. A cloud with more than M atoms keeps the target residue's own atoms
and then the atoms nearest the target CA, in the distance order the recipe sorts them,
and drops the farthest; the row is counted and the count is written to atom/config.json.
"""
import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..atom_cloud import FINAL_ATOM_CLOUD_RECIPE, build_atom_cloud
from ..chemistry import atom_from_shift_column
from . import assembly, config, exclusion, structures
from .config import ATOM_TO_IDX
from .partner_suppression import suppression_key

PAD_ATOM = len(ATOM_TO_IDX)
PAD_OFF = 17            # the padding id of the residue-offset channel, part of the recipe


def shipped_atom_cloud_recipe():
    """The final recipe carrying the vocabulary this pipeline stamps.

    One spelling of that vocabulary, in one place: the atom alphabet's own length and the
    cross-offset alphabet's own size. The atom stage stamps it into every cache, the
    trainer requires it of every fold it reads and the staging step requires it of every
    checkpoint it ships, so no step can accept a vocabulary another would not write.
    """
    return FINAL_ATOM_CLOUD_RECIPE.with_vocabulary(PAD_ATOM, PAD_OFF)


def atom_cloud_recipe(args):
    """The final recipe with the cache/checkpoint vocabulary attached. The stage builds
    exactly one recipe; any other is refused."""
    final = shipped_atom_cloud_recipe()
    recipe = replace(final, m_atoms=args.m_atoms, partner_policy=args.partner_sample_policy)
    if recipe != final:
        raise SystemExit(
            f"atom stage recipe {recipe.to_dict()} differs from the final recipe "
            f"{final.to_dict()}")
    return recipe


def choose_m_atoms(precap_counts):
    """The fixed final cap. A census maximum above it means truncation of the farthest
    atoms for the rows over the cap; that is warned with the maximum and the row count."""
    counts = np.asarray(precap_counts, dtype=np.int64)
    cap = int(FINAL_ATOM_CLOUD_RECIPE.m_atoms)
    over = int((counts > cap).sum()) if counts.size else 0
    if over:
        warnings.warn(
            f"census maximum {int(counts.max())} exceeds M={cap}: {over} of {counts.size} "
            f"centers keep their {cap} nearest atoms", RuntimeWarning, stacklevel=2)
    return cap

def loader_atom_count(real_count, perturber_excluded):
    """0 for a census-excluded sample, so the loader, which keeps only atom_count>0
    rows, drops it; the true count survives in raw_atom_count.npy."""
    return 0 if perturber_excluded else int(real_count)


def finalize_atom_rejections(rejections, out_dir, allow_atom_rejections):
    """Write the rejection ledger and enforce the rejection gate.
    Returns the ledger; raises SystemExit(2) on any rejection unless explicitly allowed."""
    n_dropped_globals = int(sum(r['n_globals'] for r in rejections))
    ledger = {
        'stage': 'atoms', 'n_rejected_materializations': len(rejections),
        'n_dropped_globals': n_dropped_globals,
        'allow_atom_rejections': bool(allow_atom_rejections),
        'rejections': rejections,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_tmp = out_dir / f'rejections.json.tmp.{os.getpid()}'
    with open(ledger_tmp, 'w') as fh:
        json.dump(ledger, fh, indent=2)
    os.replace(ledger_tmp, out_dir / 'rejections.json')
    if rejections:
        print(f'\n  atom-stage rejections: {len(rejections)} materialization(s) '
              f'dropping {n_dropped_globals:,} realized cache rows '
              f'-> {out_dir}/rejections.json')
        if not allow_atom_rejections:
            print('\nFATAL: realized materializations/residues were '
                  'dropped by the atom stage (missing/unparseable structure, empty '
                  'cloud, or a shift-bearing residue that '
                  'never got a cloud -- see rejections.json for exact per-reason '
                  'counts). These rows would silently vanish (atom_count=0) AFTER '
                  'fold balancing, changing the CV distribution. Fix the structures '
                  'or pass --allow_atom_rejections to accept the logged drops.',
                  file=sys.stderr)
            raise SystemExit(2)
    return ledger


def load_materialization_build_log(specs):
    """Entity-grained build logs (from materialize.write_atoms_build_log) as
    {materialization_id: (pdb_path, chain_id)}, so each entity resolves its own structure
    and a multi-entity deposition does not collapse to one structure per (bmrb, source)."""
    import pandas as pd
    out = {}
    for spec in specs:
        parts = spec.split(':')
        if len(parts) != 3:
            raise ValueError(
                f"--source_build_logs spec must be source:logpath:dataset, got '{spec}'")
        _source, logpath, _dataset = parts
        if not os.path.isfile(logpath):
            raise FileNotFoundError(f"build_log not found: {logpath}")
        df = pd.read_csv(logpath, dtype=str)
        if 'materialization_id' not in df.columns:
            raise ValueError(
                f"{logpath} is not entity-grained (no materialization_id column); "
                "rebuild the atoms build log with the entity materialize stage")
        df = df[df['status'] == 'success']
        for _, r in df.iterrows():
            mzid = str(r['materialization_id'])
            if mzid:
                out[mzid] = (r['pdb_path'], str(r['chain_id']))
    return out


def add_atom_arguments(parser):
    """Atom-stage CLI surface, kept out of ``main`` so the defaults are testable."""
    parser.add_argument('--cache_dir', required=True, help='<fold>/ (parent of structural/)')
    parser.add_argument('--source_build_logs', nargs='+', required=True,
                        help="one or more 'source:logpath:dataset' specs naming the "
                             "materialization build logs; each materialization resolves "
                             "its own structure")
    parser.add_argument('--pdb_search_dirs', nargs='+',
                        default=[config.PDB_DIR, config.ALPHAFOLD_DIR,
                                 os.path.join(config.DATA_ARCHIVE, 'pdbs'),
                                 os.path.join(config.DATA_ARCHIVE, 'alphafold')])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--m_atoms', type=int, choices=[512], default=512,
                        help='fixed final atom-cloud cap')
    parser.add_argument('--partner_sample_policy', choices=['sample_matched'],
                        default='sample_matched',
                        help='include protein partners only when present in the NMR sample')
    parser.add_argument('--sample_assembly', required=True,
                        help='label-set-grained sample_assembly.json from '
                             'deepshift.data.assembly')
    parser.add_argument('--leaky_partner_chains', required=True,
                        help='JSON {suppression_key: [chain_ids]} of cross-fold dataset '
                             'partner chains to suppress; an empty object suppresses none')
    parser.add_argument('--allow_atom_rejections', action='store_true',
                        help='Permit realized materializations to be dropped by the '
                             'atom stage (missing/unparseable structure or empty '
                             'cloud). Without this flag ANY such drop is FATAL: a '
                             'silent atom_count=0 removes a '
                             'realized sample AFTER fold balancing, changing the CV '
                             'distribution. The rejection ledger is always written.')


def _load_partner_authorities(ap, args):
    """The held-out-chain suppression map and the NMR sample-composition map."""
    with open(args.leaky_partner_chains) as fh:
        suppress_map = json.load(fh)
    if not isinstance(suppress_map, dict):
        ap.error("--leaky_partner_chains must contain a JSON object")
    assembly_map = assembly.load(args.sample_assembly)
    print(f'  partner policy: sample_matched ({len(assembly_map)} label sets classified; '
          f'suppress map: {len(suppress_map)} target chains)')
    return suppress_map, assembly_map


@dataclass
class CacheBinding:
    """The fold cache this build writes into: shape, materialization groups, and the
    identity maps binding label rows to structure residues."""
    cd: Path
    sd: Path
    shift_cols: list
    nuc_atoms: list
    n_res: int
    log: dict
    grp2g: dict
    groups: list
    grp2labelset: dict
    bind: dict
    lr_by_g: dict
    src_by_g: dict
    le_by_g: dict


def _load_cache_binding(args):
    """Resolve each materialization's structure and the cache rows it owns.

    Rows group by materialization_id and bind to the structure by struct_res_id: a
    threaded peptide's comp_index is not its structure residue number.
    """
    cd = Path(args.cache_dir)
    sd = cd / 'structural'
    with open(cd / 'config.json') as fh:
        cfg = json.load(fh)
    shift_cols = cfg['shift_cols']
    nuc_atoms = [atom_from_shift_column(c) for c in shift_cols]
    n_res = np.load(sd / 'residue_idx.npy', mmap_mode='r').shape[0]
    print(f'cache residues: {n_res:,}  shift targets: {len(shift_cols)}  '
          f'M_atoms={args.m_atoms}')

    log = load_materialization_build_log(args.source_build_logs)
    with open(sd / 'materialization_ids.json') as fh:
        mat_ids = json.load(fh)
    mat_idx = np.load(sd / 'materialization_idx.npy')
    with open(sd / 'global_to_struct_resid.json') as fh:
        bind = json.load(fh)
    grp2g = {}
    for g in range(n_res):
        mzid = mat_ids[int(mat_idx[g])]
        grp2g.setdefault(str(mzid), []).append(g)
    groups = [m for m in mat_ids if m in grp2g]   # cache layout order
    # The sample authority is label-set-grained; every row in one materialization must
    # resolve to the same label set or partner selection is undefined.
    with open(sd / 'id_maps.json') as fh:
        id_maps = json.load(fh)
    label_by_global = id_maps.get('label_set_id', {})
    grp2labelset = {}
    for materialization in groups:
        values = {label_by_global.get(str(g)) for g in grp2g[materialization]}
        values.discard(None)
        if len(values) != 1:
            raise ValueError(
                f"materialization {materialization} maps to {len(values)} label sets"
            )
        grp2labelset[materialization] = values.pop()
    if args.limit:
        groups = groups[:args.limit]

    # Perturber exclusion is derived per structure file, and an AlphaFold model carries no
    # cofactor: a label residue excluded in an experimental realization is excluded in
    # every realization of its entity. An entity realized only as AlphaFold has no
    # experimental structure that could reveal a perturber and is kept, tagged af_only.
    return CacheBinding(
        cd=cd, sd=sd, shift_cols=shift_cols, nuc_atoms=nuc_atoms, n_res=n_res,
        log=log, grp2g=grp2g, groups=groups, bind=bind,
        lr_by_g=id_maps.get('label_residue_id', {}),
        src_by_g=id_maps.get('source', {}),
        le_by_g=id_maps.get('logical_entity_id', {}),
        grp2labelset=grp2labelset)


@dataclass
class AtomArrays:
    """Everything one build accumulates: per-row output arrays, rejection ledgers, counters."""
    type: np.ndarray
    off: np.ndarray
    xyz: np.ndarray
    cnt: np.ndarray
    raw_cnt: np.ndarray
    nuc: np.ndarray
    excluded: np.ndarray
    local_excluded: np.ndarray
    af_only: np.ndarray
    precap: np.ndarray
    trunc: np.ndarray
    # Group rejects also land in rejected_globals so the residue check cannot double-count.
    rejections: list = field(default_factory=list)
    rejected_globals: set = field(default_factory=set)
    # Label residues perturber-excluded in some experimental realization (entity union).
    excluded_label_residues: set = field(default_factory=set)
    n_ok: int = 0
    n_miss: int = 0
    n_sample_absent: int = 0
    suppression_keys_applied: set = field(default_factory=set)

    @property
    def m_atoms(self):
        return self.type.shape[1]


def _allocate_outputs(binding, m_atoms) -> AtomArrays:
    """Per-row output arrays for one fold: ``cnt`` is zeroed for excluded rows so the loader
    drops them, ``raw_cnt`` keeps the true gathered count, ``precap`` the count before the
    m_atoms cap, ``af_only`` tags rows whose entity has no experimental realization."""
    n, s = binding.n_res, len(binding.shift_cols)
    return AtomArrays(
        type=np.full((n, m_atoms), PAD_ATOM, dtype=np.int16),
        off=np.full((n, m_atoms), PAD_OFF, dtype=np.int8),
        xyz=np.zeros((n, m_atoms, 3), dtype=np.float16),
        cnt=np.zeros(n, dtype=np.int16),
        raw_cnt=np.zeros(n, dtype=np.int16),
        nuc=np.full((n, s), -1, dtype=np.int16),
        excluded=np.zeros(n, dtype=bool),
        local_excluded=np.zeros(n, dtype=bool),
        af_only=np.zeros(n, dtype=bool),
        precap=np.zeros(n, dtype=np.int32),
        trunc=np.zeros(n, dtype=bool),
    )


def _reject(arrays, binding, grp_key, reason):
    """Ledger a group the atom stage could not build."""
    gs = binding.grp2g.get(grp_key, [])
    arrays.rejected_globals.update(gs)
    arrays.rejections.append({'group': (list(grp_key) if isinstance(grp_key, tuple)
                                        else str(grp_key)),
                              'reason': reason,
                              'n_globals': len(gs)})


def _store_pass(arrays, binding, built_result, rid_to_globals, is_excluded):
    """Store one build pass's rows. A center is an integer author number (main pass) or a
    ``(chain, seq, icode)`` insertion-coded residue (label-center pass); ``rid_to_globals``
    is keyed by whichever the pass produced. Records the per-file perturber flag; the entity
    union is resolved after the whole corpus is seen."""
    for row_i, rid in enumerate(built_result.residue_ids):
        gs = rid_to_globals.get(rid)
        if not gs:
            continue
        real_cnt = int(built_result.atom_mask[row_i].sum())
        locally_excluded = is_excluded(rid)
        precap = int(built_result.precap_atom_count[row_i])
        for g in gs:
            arrays.raw_cnt[g] = real_cnt
            arrays.local_excluded[g] = locally_excluded
            arrays.precap[g] = precap
            arrays.trunc[g] = precap > arrays.m_atoms
            # bx is centred on the target CA, so fp16 quantises local offsets, not absolutes.
            arrays.type[g] = built_result.atom_type[row_i].astype(np.int16, copy=False)
            arrays.off[g] = built_result.atom_offset[row_i].astype(np.int8, copy=False)
            arrays.xyz[g] = built_result.atom_xyz[row_i].astype(np.float16, copy=False)
            arrays.nuc[g] = built_result.nucleus_atom_idx[row_i]
            # Entity union: perturbed in an experimental realization, excluded in all.
            if locally_excluded and binding.src_by_g.get(str(g)) in ('nmr', 'xtal'):
                lr = binding.lr_by_g.get(str(g))
                if lr is not None:
                    arrays.excluded_label_residues.add(lr)


def _build_group(grp, args, binding, arrays, suppress_map, assembly_map):
    """Gather and store one protein group's atom clouds. ``grp`` is (bmrb, source) in unified
    mode, else a bare bmrb -- the same key the build_log uses, so each instance gets its own
    PDB."""
    entry = binding.log.get(grp)
    if entry is None:
        arrays.n_miss += 1
        _reject(arrays, binding, grp, 'no_build_log_entry')
        return
    pdb_path, chain = entry
    suppress_key = suppression_key(pdb_path, chain)
    if not os.path.isfile(pdb_path):
        base = os.path.basename(pdb_path)
        alt = None
        for d in args.pdb_search_dirs:
            if os.path.isfile(os.path.join(d, base)):
                alt = os.path.join(d, base)
                break
        if alt is None:
            alt = structures.resolve_pdb_path(os.path.splitext(base)[0], args.pdb_search_dirs)
        if alt is None:
            arrays.n_miss += 1
            _reject(arrays, binding, grp, 'structure_path_missing')
            return
        pdb_path = alt
    try:
        # keep_nonstandard: modified residues stay for census categorization, but geometry
        # still skips them as centers and tokens; mapped MSE is standard MET.
        aa = structures.model_to_aa(
            structures.select_best_nmr_model(pdb_path, chain), keep_nonstandard=True)
        # The integer-keyed atom cache keeps only the lowest insertion code per author
        # number, so 10 and 10A cannot both be integer centers. The dropped sibling's heavy
        # atoms stay in neighbour geometry as a (chain, seq, icode) partner residue.
        partner_residues = structures.retained_insertion_partner_residues(pdb_path, chain)
        # Cross-chain partner protein residues, pinned to the same model as `aa`; {} when
        # the NMR sample held no protein partner.
        label_set_id = binding.grp2labelset.get(grp)
        sample_record = assembly_map.get(str(label_set_id), {})
        fold_partners = assembly.partner_in_sample(assembly_map, label_set_id)
        if not fold_partners:
            arrays.n_sample_absent += 1
        try:
            cross_partners = (structures.sample_matched_partner_residues_for(
                pdb_path, chain, sample_record, drop_hydrogens=True
            ) if fold_partners else {})
        except Exception as exc:
            # Reverting one failed structure to a single-chain cloud would leave a mixed,
            # unrecorded representation under one config; ledger it and let the gate decide.
            arrays.n_miss += 1
            _reject(arrays, binding, grp,
                    f"partner_structure_unparseable:{type(exc).__name__}")
            return
        # Partner chains whose label set sits in another fold leave the cloud.
        suppress_set = set(suppress_map.get(suppress_key, ()))
        if suppress_set:
            arrays.suppression_keys_applied.add(suppress_key)
    except Exception:
        arrays.n_miss += 1
        _reject(arrays, binding, grp, 'structure_unparseable')
        return
    if not aa:
        arrays.n_miss += 1
        _reject(arrays, binding, grp, 'structure_empty')
        return
    arrays.n_ok += 1
    excluded_rids = exclusion.perturber_excluded_rids(pdb_path, chain, aa)
    # A struct_res_id is either a plain integer author number (a main-pass center) or an
    # insertion-coded residue like 10A, built via its (chain, seq, icode) tuple key.
    int_bind, ins_bind = {}, {}
    for gg in binding.grp2g[grp]:
        parsed = structures.parse_struct_res_id(binding.bind.get(str(gg)))
        if parsed is None:
            continue
        kind, value = parsed
        (int_bind if kind == 'int' else ins_bind).setdefault(value, []).append(gg)

    # Split retained insertion siblings into labelled centers (a shift binds to them, so
    # they need their own cloud) and unlabelled neighbour partners; the main pass keeps
    # every sibling as a partner, so residues around 10A still gather its heavy atoms.
    # Only target-chain siblings qualify -- cross-chain partners do not enter here.
    labelled_centers = {k: v for k, v in (partner_residues or {}).items()
                        if str(k[0]) == str(chain) and (int(k[1]), str(k[2])) in ins_bind}
    unlabelled_partners = {k: v for k, v in (partner_residues or {}).items()
                           if str(k[0]) == str(chain)
                           and (int(k[1]), str(k[2])) not in ins_bind}

    recipe = atom_cloud_recipe(args)
    built = build_atom_cloud(
        aa, recipe=recipe, nucleus_atoms=binding.nuc_atoms,
        partner_residues={**(partner_residues or {}), **cross_partners} or None,
        suppress_chains=suppress_set or None)
    if built is None:
        _reject(arrays, binding, grp, 'empty_atom_cloud')
        return
    _store_pass(arrays, binding, built, int_bind, lambda rid: int(rid) in excluded_rids)

    # Label-center pass: each labelled insertion residue becomes its own center, with the
    # rest of the chain as non-center partners, so a shift on 10A gets a cloud, not a mask.
    if labelled_centers:
        aa_partners = {(str(chain), int(seq), ''): info for seq, info in aa.items()}
        ins_built = build_atom_cloud(
            labelled_centers, recipe=recipe, nucleus_atoms=binding.nuc_atoms,
            partner_residues={**aa_partners, **unlabelled_partners, **cross_partners} or None,
            suppress_chains=suppress_set or None)
        if ins_built is not None:
            _store_pass(
                arrays, binding, ins_built,
                {k: ins_bind.get((int(k[1]), str(k[2])), [])
                 for k in ins_built.residue_ids},
                lambda rid: int(rid[1]) in excluded_rids)


def _finalize_exclusions(arrays, binding):
    """Resolve the entity union and zero the excluded rows: a label residue is excluded
    iff some experimental realization of its entity perturbed it. Zeroed rows
    (atom_count==0) are dropped by the loader, with their gathered count preserved in
    raw_atom_count."""
    experimental_entities = {binding.le_by_g.get(str(g)) for g in range(binding.n_res)
                             if binding.src_by_g.get(str(g)) in ('nmr', 'xtal')}
    for g in range(binding.n_res):
        is_excluded = binding.lr_by_g.get(str(g)) in arrays.excluded_label_residues
        arrays.af_only[g] = binding.le_by_g.get(str(g)) not in experimental_entities
        arrays.excluded[g] = is_excluded
        arrays.cnt[g] = loader_atom_count(arrays.raw_cnt[g], is_excluded)
        if is_excluded:
            arrays.type[g] = PAD_ATOM
            arrays.off[g] = PAD_OFF
            arrays.xyz[g] = 0
            arrays.nuc[g] = -1


def _check_residue_completeness(arrays, binding):
    """At residue grain, ledger every shift-bearing sample left at atom_count=0. A
    mapped struct_res_id that never became a built center (missing CA, integer-projection
    collision, residue absent from the chosen model) leaves the row at 0, and the loader
    would drop it after fold balancing without a record. Gap/mismatch/insertion label
    residues are already loss-masked, so the one sanctioned zero-count sample is the
    census perturber exclusion."""
    smask = np.load(binding.sd / 'shift_mask.npy', mmap_mode='r')
    shift_bearing = np.asarray(smask).any(axis=1)
    g2grp = {g: grp for grp, gs in binding.grp2g.items() for g in gs}
    uncaught: dict = {}
    for g in range(binding.n_res):
        if not shift_bearing[g] or arrays.excluded[g] or arrays.cnt[g] > 0 \
                or g in arrays.rejected_globals:
            continue
        grp_key = g2grp.get(g)
        uncaught.setdefault(grp_key, []).append(g)
    for grp_key in sorted(uncaught, key=lambda k: str(k)):
        arrays.rejections.append({
            'group': (list(grp_key) if isinstance(grp_key, tuple) else str(grp_key)),
            'reason': 'sample_residue_no_atom_cloud',
            'n_globals': len(uncaught[grp_key])})


def _atom_config(args, arrays, binding, suppress_map):
    """The atom arrays' config: the cloud recipe and the stage's counts."""
    recipe = atom_cloud_recipe(args)
    return {
        'recipe': recipe.to_dict(),
        'precap_max': int(arrays.precap.max()) if binding.n_res else 0,
        'n_truncated': int(arrays.trunc.sum()),
        'leaky_partner_chains_map': os.path.basename(args.leaky_partner_chains),
        'n_suppress_map_structures': len(suppress_map),
        'suppression_keys_applied': len(arrays.suppression_keys_applied),
        'sample_partner_sequence_match': {
            'identity_min': structures.SAMPLE_PARTNER_MIN_IDENTITY,
            'query_coverage_min': structures.SAMPLE_PARTNER_MIN_QUERY_COVERAGE,
            'aligned_residues_min': structures.SAMPLE_PARTNER_MIN_ALIGNED_LENGTH,
            'copy_count_cap': True,
            'unknown_or_excess_chain_behavior': 'exclude',
        },
        'n_perturber_excluded': int(arrays.excluded.sum()),
        'perturber_exclusion_scope': 'entity_union',
        'n_af_only_entity': int(arrays.af_only.sum()),
        'entity_grained': True,
    }


def _write_atom_arrays(ad, arrays, atom_config):
    """Serialize the atom sidecar: loader arrays, census sidecars, and their config."""
    ad.mkdir(exist_ok=True)
    np.save(ad / 'atom_type.npy', arrays.type)
    np.save(ad / 'atom_offset.npy', arrays.off)
    np.save(ad / 'atom_xyz.npy', arrays.xyz)
    np.save(ad / 'atom_count.npy', arrays.cnt)
    np.save(ad / 'nucleus_atom_idx.npy', arrays.nuc)
    np.save(ad / 'perturber_excluded.npy', arrays.excluded)
    np.save(ad / 'raw_atom_count.npy', arrays.raw_cnt)    # true count before exclusion
    np.save(ad / 'precap_atom_count.npy', arrays.precap)  # census: count before the cap
    np.save(ad / 'atom_truncated.npy', arrays.trunc)      # census: rows where precap > cap
    np.save(ad / 'af_only_entity.npy', arrays.af_only)    # AF-only rows (kept, unvetted)
    atom_config_tmp = ad / f'config.json.tmp.{os.getpid()}'
    with open(atom_config_tmp, 'w') as fh:
        json.dump(atom_config, fh, indent=2)
    os.replace(atom_config_tmp, ad / 'config.json')


def _report(args, arrays, ad, t0):
    """One-screen build summary."""
    filled = (arrays.cnt > 0).mean()
    nuc_cov = (arrays.nuc >= 0).any(1).mean()
    n_excl = int(arrays.excluded.sum())
    # cnt is already zeroed for excluded rows, so key the fraction off raw_atom_count.
    has_raw = (arrays.raw_cnt > 0).any()
    excluded_frac = arrays.excluded[arrays.raw_cnt > 0].mean() if has_raw else 0.0
    median = int(np.median(arrays.cnt[arrays.cnt > 0])) if (arrays.cnt > 0).any() else 0
    print(f'\n  proteins ok={arrays.n_ok} miss={arrays.n_miss}  residues filled={filled:.3f}  '
          f'median atoms={median}  '
          f'nucleus-coverage={nuc_cov:.3f}  perturber_excluded={excluded_frac:.4f} '
          f'(excluded {n_excl} samples)  ({(time.time()-t0)/60:.1f} min)')
    print(f'  sample-matched partners: {arrays.n_sample_absent} materializations built '
          f'single-chain (no protein partner in the NMR sample)')
    print(f'  wrote {ad}/  (atom_type {arrays.type.shape}, '
          f'nucleus_atom_idx {arrays.nuc.shape})')


def _check_suppression_applied(arrays, binding, suppress_map):
    """A non-empty map that addresses this fold's structures must have matched."""
    addressed = {suppression_key(*binding.log[grp]) for grp in binding.groups
                 if grp in binding.log} & set(suppress_map)
    if addressed and not arrays.suppression_keys_applied:
        raise SystemExit(
            f'partner suppression map addresses {len(addressed)} target chains of this '
            f'fold and none was applied; example key {sorted(addressed)[0]!r}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_atom_arguments(ap)
    args = ap.parse_args(argv)
    suppress_map, assembly_map = _load_partner_authorities(ap, args)
    binding = _load_cache_binding(args)
    # The assembly map must share a key space with the cache groups: a total miss would
    # build every materialization single-chain and report a normal result.
    keys = [binding.grp2labelset.get(g) for g in binding.groups]
    known = sum(1 for k in keys if k is not None and str(k) in assembly_map)
    complex_n = sum(1 for k in keys
                    if k is not None and assembly.partner_in_sample(assembly_map, k))
    print(f'  sample-matched coverage: {known}/{len(keys)} groups found in the assembly '
          f'map, {complex_n} with a protein partner in the sample')
    if known == 0:
        raise SystemExit(
            'sample_matched: not one cache group resolves to a label set in '
            f'--sample_assembly. Example group key: {binding.groups[0]!r}')
    arrays = _allocate_outputs(binding, args.m_atoms)

    t0 = time.time()
    for grp in tqdm(binding.groups, desc='proteins'):
        _build_group(grp, args, binding, arrays, suppress_map, assembly_map)
    _finalize_exclusions(arrays, binding)
    _check_residue_completeness(arrays, binding)
    _check_suppression_applied(arrays, binding, suppress_map)
    if arrays.trunc.any():
        warnings.warn(
            f"{int(arrays.trunc.sum())} of {binding.n_res} rows exceed M={arrays.m_atoms} "
            f"(precap maximum {int(arrays.precap.max())}); each keeps its "
            f"{arrays.m_atoms} nearest atoms", RuntimeWarning, stacklevel=2)

    ad = binding.cd / 'atom'
    _write_atom_arrays(ad, arrays, _atom_config(args, arrays, binding, suppress_map))
    _report(args, arrays, ad, t0)
    finalize_atom_rejections(arrays.rejections, ad, args.allow_atom_rejections)


if __name__ == '__main__':
    sys.exit(main())
