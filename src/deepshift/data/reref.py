#!/usr/bin/env python
"""Rereferencing offset builder, label-set grained.

Produces reref_offsets.json: a per-label-set, per-nucleus correction added to the
observed shifts. Every offset is mean(SHIFTX prediction - observed) over the label
set's own aligned structure. CA/CB share one aliphatic offset, CO and N are
independent, HA/HN are never corrected.

Invoked by the build as:
  python -m deepshift.data.reref \
      --shifts_file chemical_shifts.csv \
      --structure-master <src>:... --build-log <src>:... --out reref_offsets.json

The SHIFTX executable is SHIFTX_BIN, else <SHIFT_TOOL_DIR>/shiftx/shiftx.
"""
import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import numpy as np

from .. import domain as _identity
from .external import ensure_shiftx
from .reref_utils import CUTOFF, HDR2SX, NUCS, SXCOLS, THREE2ONE, best_shift

COL2HDR = {'ca_shift': 'CA', 'cb_shift': 'CB', 'c_shift': 'CO',
           'n_shift': 'N', 'ha_shift': 'HA', 'h_shift': 'HN'}

# Pinned SHIFTX binary (env first, else the SHIFT_TOOL_DIR default).
SHIFTX_BIN = os.environ.get('SHIFTX_BIN') or os.path.join(
    os.environ.get('SHIFT_TOOL_DIR', os.path.join(os.getcwd(), '.deepshift-tools')),
    'shiftx', 'shiftx')

# The four nuclei that reref may adjust; HA/HN are never applied.
REREF_APPLY_NUCS = ('CA', 'CB', 'CO', 'N')


def _atomic_json(path, value, **kwargs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.tmp.',
                               dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, 'w') as fh:
            json.dump(value, fh, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def keepA_chain(pdb_path, want_chain=None):
    out = []
    with open(pdb_path, errors='ignore') as fh:
        for ln in fh:
            if ln[:6] in ('ATOM  ', 'HETATM'):
                if want_chain and ln[21] != want_chain:
                    continue
                if ln[16] != ' ':
                    if ln[16] != 'A':
                        continue
                    ln = ln[:16] + ' ' + ln[17:]
            out.append(ln)
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as tf:
        tf.writelines(out)
    return tf.name


def parse_shiftx_output(path):
    """Parse SHIFTX output, including its documented ``*`` residue marker: SHIFTX prefixes
    some otherwise-valid residue numbers (e.g. every row of PDB 1AIN) with ``*``, which
    must be stripped before parsing as an integer."""
    preds = {}
    with open(path) as fh:
        for line in fh:
            fields = line.split()
            if len(fields) != 8:
                continue
            residue_token = fields[0].lstrip('*')
            if not residue_token.lstrip('-').isdigit() or not (len(fields[1]) == 1 and fields[1].isalpha()):
                continue
            vals = {}
            for col, token in zip(SXCOLS, fields[2:]):
                try:
                    value = float(token)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}: malformed SHIFTX value {token!r} for "
                        f"residue {residue_token} column {col}"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(
                        f"{path}: non-finite SHIFTX value {token!r} for "
                        f"residue {residue_token} column {col}"
                    )
                vals[col] = None if value == 0.0 else value
            preds[int(residue_token)] = (fields[1], vals)
    return preds


def run_shiftx(pdb_path, chain, return_flags=False):
    mode = '1' + (chain if chain and chain.strip() else 'A')
    with tempfile.NamedTemporaryFile(suffix='.out', delete=False) as tf:
        tmp = tf.name
    sanitized_pdb = None
    flags = []
    try:
        proc = subprocess.run([SHIFTX_BIN, mode, pdb_path, tmp], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, timeout=180)
        preds = parse_shiftx_output(tmp) if proc.returncode == 0 else {}
        # A nonstandard SSBOND endpoint (e.g. HCS in 6R28) makes SHIFTX abort with a
        # specific diagnostic; retry without the malformed header only when that exact
        # parser failure is identified, keeping valid SSBOND records otherwise.
        if not preds and 'SSBOND specification format error' in (proc.stderr or ''):
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as sf:
                sanitized_pdb = sf.name
                with open(pdb_path, errors='ignore') as source:
                    sf.writelines(
                        line for line in source
                        if (not line.startswith('SSBOND') or
                            (line[11:14].strip() == 'CYS' and line[25:28].strip() == 'CYS'))
                    )
            open(tmp, 'w').close()
            retry = subprocess.run(
                [SHIFTX_BIN, mode, sanitized_pdb, tmp],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180,
            )
            preds = parse_shiftx_output(tmp) if retry.returncode == 0 else {}
            if preds:
                flags.append('shiftx_removed_malformed_ssbond')
            elif retry.returncode:
                flags.append(f'shiftx_retry_exit_{retry.returncode}')
            else:
                # Exit 0 with nothing parsed still records a flag, so every retry
                # outcome is represented.
                flags.append('shiftx_empty_output')
        elif proc.returncode:
            flags.append(f'shiftx_exit_{proc.returncode}')
        elif not preds:
            flags.append('shiftx_empty_output')
        return (preds, flags) if return_flags else preds
    except subprocess.TimeoutExpired as err:
        flags.append('shiftx_timeout')
        if return_flags:
            return {}, flags
        raise RuntimeError(f"SHIFTX timed out on {pdb_path} chain {chain}") from err
    except (OSError, ValueError) as exc:
        flags.append(f'shiftx_error:{type(exc).__name__}:{exc}')
        if return_flags:
            return {}, flags
        raise RuntimeError(
            f"SHIFTX failed on {pdb_path} chain {chain}: {exc}"
        ) from exc
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
        if sanitized_pdb and os.path.exists(sanitized_pdb):
            os.remove(sanitized_pdb)


def eval_structure(body, preds):
    """Returns (pooled_corr, per{nuc:{avgdiff,n}}, frac). offset sense = mean(pred-obs)."""
    s, frac = best_shift(body, preds)
    per = {}
    pp, po = [], []
    for n in NUCS:
        sx = HDR2SX[n]
        ps, os_ = [], []
        for sid, (aa, vals) in body.items():
            if n not in vals:
                continue
            pr = preds.get(sid + s)
            if not pr or pr[0] != aa:
                continue
            pv = pr[1].get(sx)
            if pv is None:
                continue
            if abs(pv - vals[n]) <= CUTOFF[n]:
                ps.append(pv)
                os_.append(vals[n])
        if len(ps) >= 3:
            a, b = np.array(ps), np.array(os_)
            per[n] = {'avgdiff': float(np.mean(a - b)), 'n': len(ps)}
            if a.std() > 0 and b.std() > 0:
                pp.extend((a - a.mean()) / a.std())
                po.extend((b - b.mean()) / b.std())
    overall = float(np.corrcoef(pp, po)[0, 1]) if len(pp) >= 5 and np.std(pp) > 0 and np.std(po) > 0 else float('nan')
    return overall, per, frac


# The offset recipe: structure selection plus offset arithmetic. Every realized source
# of a label set is a candidate; the selector ranks candidates by the correlation of
# SHIFTX with the observed shifts, preferring those covering at least 95% of the
# deposition; the offset is the mean of (prediction - observed) on the windowed cells of
# each nucleus measured on at least ``min_cells`` of them, CA and CB sharing one
# count-weighted constant, rounded to a 0.05 ppm grid.
RULESEARCH_WINNER_DEFAULT = {
    'selection': {
        'selector': {'metric': 'meanr_all', 'preference_gate': 0.95},
        'tie_tolerance': 0.001,
    },
    'arithmetic': {'min_cells': 3},
}

#: The grid every applied offset is rounded to, ppm.
OFFSET_GRID_PPM = 0.05


def _eval_per_to_nucleus_stats(per):
    """Adapt eval_structure() per-nucleus output to the arithmetic's schema: the
    cutoff-windowed mean(pred - obs) and its count."""
    return {nuc: {'mean_fixed': d.get('avgdiff'), 'n_fixed': int(d.get('n', 0))}
            for nuc, d in (per or {}).items() if d}


def derive_rulesearch_offsets(nucleus_stats, arithmetic):
    """Per-nucleus applied offsets from candidate stats.

    nucleus_stats: {nucleus: {mean_fixed, n_fixed}}; arithmetic carries min_cells. A
    nucleus measured on fewer cells is left uncorrected. CA/CB share one count-weighted
    offset when both are measured; when only one is, it alone is offset and the other is
    left uncorrected (data/cacb_unshared_offsets.tsv lists the released cells this leaves);
    CO/N are independent; HA/HN are never referenced.
    Offsets land on the 0.05 ppm grid, and one that rounds to zero is not recorded."""
    values, counts = {}, {}
    for nuc in REREF_APPLY_NUCS:
        stats = (nucleus_stats or {}).get(nuc) or {}
        value, count = stats.get('mean_fixed'), stats.get('n_fixed', 0)
        if value is not None and count >= arithmetic['min_cells']:
            values[nuc] = float(value)
            counts[nuc] = int(count)

    result = {}
    if 'CA' in values and 'CB' in values:
        value = ((counts['CA'] * values['CA'] + counts['CB'] * values['CB'])
                 / (counts['CA'] + counts['CB']))
        result.update({'CA': value, 'CB': value})
    for nuc in ('CA', 'CB', 'CO', 'N'):
        if nuc in values and nuc not in result:
            result[nuc] = values[nuc]

    final = {}
    for nuc, value in result.items():
        value = float(np.round(value / OFFSET_GRID_PPM) * OFFSET_GRID_PPM)
        # A numerically exact zero is a no-op, not a derived correction.
        if abs(value) > 1e-15:
            final[nuc] = value
    return final

# Structure selection.
_RULESEARCH_METHOD_RANK = {'xtal': 0, 'nmr': 1, 'af': 2, 'other': 3}


def _rulesearch_method_rank(candidate):
    return _RULESEARCH_METHOD_RANK.get(candidate['source'], 4)


def _rulesearch_tie_key(candidate):
    """Among tied candidates: experimental method first, then higher coverage, then
    stable identity."""
    return (_rulesearch_method_rank(candidate), -candidate['qcov'],
            candidate['accession'].upper(), candidate['chain'], candidate['candidate_id'])


def select_rulesearch_candidate(pool, selection):
    """Pick one candidate: the highest SHIFTX-observed correlation among candidates whose
    correlation is finite, preferring those covering at least ``preference_gate`` of the
    deposition; ties within ``tie_tolerance`` break by method rank and coverage."""
    if not pool:
        return None
    selector = selection['selector']
    metric = selector['metric']
    scored = []
    for candidate in pool:
        score = candidate['scores'].get(metric)
        if score is None or not math.isfinite(score):
            continue
        scored.append((score, candidate))
    if not scored:
        return None
    preferred = [(score, candidate) for score, candidate in scored
                 if candidate['qcov'] + 1e-12 >= selector['preference_gate']]
    if preferred:
        scored = preferred
    best = max(score for score, _candidate in scored)
    tied = [candidate for score, candidate in scored
            if best - score <= selection['tie_tolerance'] + 1e-15]
    return min(tied, key=_rulesearch_tie_key)


# Glue for the entity reref driver (realized eval_structure candidates).
def _realized_candidate_to_harness(cand):
    """Wrap an eval_structure realized candidate as a selection dict for the selector."""
    return {
        'source': cand['source'],
        'accession': str(cand['struct'][0]), 'chain': str(cand['struct'][1]),
        'candidate_id': f"{cand['source']}:{cand['struct'][1]}",
        'qcov': float(cand['frac']),
        'scores': {'meanr_all': cand['overall']},
    }


def select_reref_structure(candidates, cfg):
    """Pick one realized eval_structure candidate with the reference-rule selector.
    candidates: [{source, struct=(path,chain), per, overall, frac, ...}]; the selector
    ranks by correlation/qcov, falling back to a source-priority pick when it abstains."""
    preference = {s: i for i, s in enumerate(_REREF_SOURCE_PREFERENCE)}
    by_priority = min(candidates, key=lambda c: preference.get(c['source'], 99))
    pool, back = [], {}
    for orig in candidates:
        harness = _realized_candidate_to_harness(orig)
        pool.append(harness)
        back[harness['candidate_id']] = orig
    picked = select_rulesearch_candidate(pool, cfg['selection'])
    if picked is None:
        return by_priority
    return back[picked['candidate_id']]


def offsets_for_method(per, cfg):
    """Applied offsets from an eval_structure per-nucleus dict via the reference rule."""
    return derive_rulesearch_offsets(_eval_per_to_nucleus_stats(per), cfg['arithmetic'])


def offset_record_for_label_set(label_set_id, *, computed_offsets=None,
                                selected_list_id, alignment_sequence_sha256,
                                struct_type=None) -> dict:
    """One reref_offsets.json record: the computed offsets, or a no-op ('none')."""
    _identity._require_id(label_set_id, 'ls_')
    if computed_offsets is not None:
        offsets, method = computed_offsets, 'computed'
    else:
        offsets, method = {}, 'none'
    return {
        'label_set_id': label_set_id,
        'offsets': offsets,
        'method': method,
        'assigned_shift_list_id': _identity.normalize_star_id(selected_list_id),
        'alignment_sequence_sha256': alignment_sequence_sha256,
        'struct_type': struct_type,
    }


def write_label_set_offsets(table, path):
    """Emit reref_offsets.json keyed by label_set_id (list/condition/method
    provenance preserved in every record)."""
    _atomic_json(path, {k: table[k] for k in sorted(table)}, indent=1)


def _parse_source_spec(spec):
    """'source:path' -> (source, path). Path may itself contain ':' (Windows-free)."""
    source, _, path = spec.partition(':')
    if source not in _identity.VALID_SOURCES or not path:
        raise ValueError(f"expected 'source:path' with source in "
                         f"{_identity.VALID_SOURCES}, got {spec!r}")
    return source, path


# Experimental structures reference-align best; AF is the documented fallback.
_REREF_SOURCE_PREFERENCE = ('xtal', 'nmr', 'af')


def _label_set_body(g):
    """observed dict {struct_res_id: [aa1, {nuc: ppm}]} for one label set, keyed by
    struct_res_id (the aligned structure residue) to line up with SHIFTX predictions,
    which key by structure residue number."""
    body = {}
    for _, row in g.iterrows():
        sid = row.get('struct_res_id')
        if sid in (None, '', 'nan') or (isinstance(sid, float) and np.isnan(sid)):
            continue
        aa1 = THREE2ONE.get(str(row['comp_id']).upper())
        if aa1 is None:
            continue
        vals = {}
        for col, hdr in COL2HDR.items():
            v = row.get(col, '')
            if v in ('', 'None', None) or (isinstance(v, float) and np.isnan(v)):
                continue
            try:
                vals[hdr] = float(v)
            except (ValueError, TypeError):
                continue
        if vals:
            body[int(float(sid))] = [aa1, vals]
    return body


def entity_reref_main(argv=None):
    """Label-set-keyed rereferencing driver. Runs SHIFTX on each label set's own
    aligned structures, aligns observed against predicted by struct_res_id, and takes
    mean(pred - obs) per nucleus under the offset recipe. A label set with no aligned
    structure, a SHIFTX failure, or under 0.8 alignment coverage gets a no-op record.
    reref_offsets.json is keyed by label_set_id with method, list and sequence
    provenance."""
    global SHIFTX_BIN
    import pandas as pd

    ap = argparse.ArgumentParser(description='Label-set-grained rereferencing.')
    ap.add_argument('--shifts_file', required=True,
                    help='entity-grained selected chemical_shifts.csv (sequence source)')
    ap.add_argument('--out', required=True, help='reref_offsets.json (label-set keyed)')
    ap.add_argument('--structure-master', dest='structure_masters', action='append',
                    default=[], help="source:structure_data_<source>.csv (repeatable). "
                    "Carries struct_res_id + observed shifts per label residue.")
    ap.add_argument('--build-log', dest='build_logs', action='append', default=[],
                    help="source:structure_build_log_<source>.csv (repeatable). "
                    "materialization_id -> (pdb_path, chain).")
    ap.add_argument('--workers', type=int, default=8,
                    help='threads for the SHIFTX pre-warm.')
    a = ap.parse_args(argv)

    rulesearch_cfg = deepcopy(RULESEARCH_WINNER_DEFAULT)

    seq_df = pd.read_csv(a.shifts_file, dtype={'label_set_id': str, 'bmrb_id': str},
                     low_memory=False)
    if 'label_set_id' not in seq_df.columns:
        raise SystemExit(f'{a.shifts_file} is not entity-grained (no label_set_id)')

    # Each label set's alignment-sequence identity is the full entity sequence from
    # entity_catalog.csv, the one sequence authority every stage reads.
    from .fetch import default_entity_catalog_path, read_entity_catalog
    catalog_path = default_entity_catalog_path(a.shifts_file)
    if not os.path.isfile(catalog_path):
        raise SystemExit(f'entity_catalog.csv not found beside {a.shifts_file}')
    catalog = read_entity_catalog(catalog_path)

    # Per-source structure masters (struct_res_id + observed shifts + accession).
    masters = {}
    for spec in a.structure_masters:
        source, path = _parse_source_spec(spec)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            # low_memory=False for the same reason as the shifts file above: these
            # masters carry the sparse shift columns too.
            masters[source] = pd.read_csv(path, dtype={'label_set_id': str,
                                                       'bmrb_id': str},
                                          low_memory=False)
    # materialization_id -> (pdb_path, chain), from the atoms build logs.
    mat_struct = {}
    for spec in a.build_logs:
        _source, path = _parse_source_spec(spec)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            bl = pd.read_csv(path, dtype=str)
            for _, r in bl[bl.get('status', 'success') == 'success'].iterrows():
                mzid = str(r.get('materialization_id', ''))
                if mzid:
                    mat_struct[mzid] = (r['pdb_path'], str(r['chain_id']))

    # SHIFTX is memoized per structure so sibling label sets sharing a structure
    # do not re-run it (the offset is still each label set's own mean(pred-obs)).
    shiftx_cache = {}
    shiftx_seconds = 0.0
    # Every repair/failure event is retained.  A missing offset must be
    # distinguishable from a structure whose SHIFTX subprocess failed.
    shiftx_events = []

    def shiftx_for(pdb_path, chain):
        nonlocal shiftx_seconds
        key = (pdb_path, chain)
        if key in shiftx_cache:
            return shiftx_cache[key]
        preds = {}
        if os.path.isfile(pdb_path):
            vp = keepA_chain(pdb_path, want_chain=chain)
            try:
                t0 = time.time()
                preds, sx_flags = run_shiftx(vp, chain, return_flags=True)
                shiftx_seconds += time.time() - t0
                if sx_flags:
                    shiftx_events.append({'pdb_path': pdb_path, 'chain': chain,
                                          'flags': sx_flags})
            finally:
                with contextlib.suppress(OSError):
                    os.remove(vp)
        shiftx_cache[key] = preds
        return preds

    def compute_offset(lsid):
        """Structure selection for one label set: evaluate every allowed realized
        source, rank them by correlation and coverage, and return the selected
        structure's per-nucleus stats (with counts) plus a 'source:chain' tag."""
        candidates = []
        for source in _REREF_SOURCE_PREFERENCE:
            m = masters.get(source)
            if m is None:
                continue
            g = m[m['label_set_id'] == lsid]
            if g.empty:
                continue
            body = _label_set_body(g)
            if not body:
                continue
            mzid = str(g['materialization_id'].iloc[0]) if 'materialization_id' in g else ''
            struct = mat_struct.get(mzid)
            if struct is None:
                continue
            preds = shiftx_for(*struct)
            if not preds:
                continue
            overall, per, frac = eval_structure(body, preds)
            if frac < 0.8 or not per:
                continue
            cand = {'source': source, 'struct': struct, 'per': per,
                    'overall': overall, 'frac': frac}
            candidates.append(cand)
        if not candidates:
            return None, None
        chosen = select_reref_structure(candidates, rulesearch_cfg)
        return chosen['per'], f"{chosen['source']}:{chosen['struct'][1]}"

    if not os.path.exists(SHIFTX_BIN):
        SHIFTX_BIN = str(ensure_shiftx())
        print(f'  SHIFTX_BIN={SHIFTX_BIN}')

    # SHIFTX pre-warm: fill the memoized per-structure cache in parallel; the serial
    # loop below computes every offset from that cache.
    if a.workers > 1 and mat_struct and masters:
        master_groups = {src: {str(k): grp for k, grp in m.groupby('label_set_id')}
                         for src, m in masters.items()}
        wanted: dict[tuple, None] = {}
        for lsid in map(str, seq_df['label_set_id'].unique()):
            for source in _REREF_SOURCE_PREFERENCE:
                gg = master_groups.get(source, {}).get(lsid)
                if gg is None or gg.empty or not _label_set_body(gg):
                    continue
                mzid = (str(gg['materialization_id'].iloc[0])
                        if 'materialization_id' in gg.columns else '')
                struct = mat_struct.get(mzid)
                if struct is not None:
                    wanted[(struct[0], struct[1])] = None
                    break
        if wanted:
            print(f'  SHIFTX pre-warm: {len(wanted)} distinct structures on '
                  f'{a.workers} workers...', flush=True)

            def _prewarm(pdb_path, chain):
                # A structure that fails here is retried by the serial loop.
                with contextlib.suppress(Exception):
                    shiftx_for(pdb_path, chain)

            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                for fut in as_completed([ex.submit(_prewarm, p, c)
                                         for (p, c) in wanted]):
                    fut.result()
            print(f'  SHIFTX pre-warm done ({len(shiftx_cache)} structures cached, '
                  f'{shiftx_seconds:.1f}s SHIFTX).', flush=True)

    table = {}
    provenance = {}
    n_labelsets = 0
    t_start = time.time()
    for lsid, g in seq_df.groupby('label_set_id', sort=True):
        n_labelsets += 1
        g = g.sort_values('comp_index_id')
        cat = catalog.get(str(lsid))
        if cat is None or not _identity.is_sha256(
                cat.get('alignment_sequence_sha256', '')):
            raise SystemExit(
                f'label set {lsid} has selected shifts but no valid '
                f'entity_catalog.csv alignment_sequence_sha256')
        if not cat['alignment_sequence']:
            continue
        list_id = str(g.iloc[0].get('shift_list_id', '1'))
        aseq_sha = cat['alignment_sequence_sha256']
        computed_offsets = None
        per_stats, struct_used = compute_offset(lsid)
        struct_type = (struct_used.split(':')[0] if struct_used else None)
        if per_stats:
            computed_offsets = offsets_for_method(per_stats, rulesearch_cfg)
        rec = offset_record_for_label_set(
            lsid, computed_offsets=computed_offsets,
            selected_list_id=list_id, alignment_sequence_sha256=aseq_sha,
            struct_type=struct_type)
        rec['struct_used'] = struct_used
        table[lsid] = rec
        provenance[lsid] = {'method': rec['method'], 'struct_used': struct_used}

    write_label_set_offsets(table, a.out)
    elapsed = time.time() - t_start
    from collections import Counter
    by_method = Counter(r['method'] for r in table.values())
    n_struct = len(shiftx_cache)
    # "applied" means the record carries a nonzero offset -- a 'computed' record can
    # still hold {} (every nucleus below min_cells, or rounded to zero), which must not
    # count.
    n_applied = sum(1 for r in table.values() if r['offsets'])
    per_ls = (shiftx_seconds / max(1, n_struct))
    print(f'reref (entity): {len(table)} label sets '
          f'({dict(by_method)}); {n_applied} with a nonzero offset; '
          f'SHIFTX on {n_struct} distinct structures in {shiftx_seconds:.1f}s '
          f'(total stage {elapsed:.1f}s) -> {a.out}')
    print(f'  COST: ~{per_ls:.2f}s SHIFTX / distinct structure; '
          f'{shiftx_seconds / max(1, len(table)):.2f}s / label set (this run)')
    prov_path = os.path.join(os.path.dirname(os.path.abspath(a.out)),
                             'reref_provenance.json')
    _atomic_json(prov_path, {
        'stage': 'entity_reref', 'grain': 'labelset',
        'rulesearch_config': rulesearch_cfg,
        'n_label_sets': len(table), 'methods': dict(by_method),
        # A label set with selected shifts but no catalog alignment sequence is skipped
        # above (no record, no correction) -- the gap between walked and written must
        # stay visible.
        'n_label_sets_seen': n_labelsets,
        'n_label_sets_without_alignment_sequence': n_labelsets - len(table),
        'n_label_sets_with_offsets': n_applied,
        'n_distinct_structures': n_struct,
        'shiftx_events': sorted(
            shiftx_events, key=lambda r: (r['pdb_path'], r['chain'])),
        'shiftx_seconds': round(shiftx_seconds, 2),
        'seconds_per_structure': round(per_ls, 3),
        'stage_seconds': round(elapsed, 2),
        'per_label_set': provenance,
    }, indent=1)
    print(f'  reref provenance sidecar -> {prov_path}')
    return table


def main():
    entity_reref_main(sys.argv[1:])


if __name__ == '__main__':
    main()
