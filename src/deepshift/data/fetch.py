#!/usr/bin/env python3
"""
Bulk BMRB fetch and entity-aware chemical-shift parse.

1. Fetch: rsync pulls only top-level ``bmr<ID>/bmr<ID>_3.str`` files into a dated
   snapshot dir, hard-linking unchanged files from the preceding snapshot
   (``--link-dest``). Snapshots are retained, so a reparse needs no download.

2. Parse: pivots each entry's ``_Atom_chem_shift`` loop (via pynmrstar) into
   per-(list, assembly, entity) candidates; the 80%-coverage-floor representative
   rule picks one label set per logical entity and emits ``entity_catalog.csv``,
   ``label_set_selection.csv``, and ``chemical_shifts.csv``.

3. Summary: requires an exact _3.str per API id (missing==0, failed==0) before
   ``ok``, written atomically to ``fetch_manifest.json``.

Network fetch is opt-in (``--fetch``); otherwise the parse runs offline.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime

import pandas as pd

from ..domain import NUCLEI

BMRB_LIST_URL = 'https://api.bmrb.io/v2/list_entries?database=macromolecules'
RSYNC_URL = 'rsync://rsync.bmrb.io/macromolecules/'
STAR_URL = 'https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr{id}/bmr{id}_3.str'


# Target atoms (lowercase column stem) derived from domain.NUCLEI -> the
# <atom>_shift / <atom>_ambiguity_code columns, so the two stay the same width.
TARGET_ATOMS = frozenset(name.lower() for name in NUCLEI)

NULLS = (None, '.', '?', '')

# Standard protein monomers (three-letter). Used by the representative rule's
# residue-coverage floor; noncanonical/modified tokens do not count toward R.
STANDARD_COMP_IDS = frozenset([
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
])

# The six backbone assay atoms (tie-break 3 in the representative rule).
BACKBONE_ATOMS = frozenset(['h', 'ha', 'ha2', 'ha3', 'c', 'ca', 'cb', 'n'])

DUP_VALUE_EPS = 1e-6                     # identical-atom-key value tolerance (ppm)

# NMR-STAR assigned-shift ambiguity codes (BMRB dictionary). 1 unique, 2/3
# geminal/aromatic prochiral pairs, 4/5/6 intra/inter-residue/other, 9 unknown.
# Anything else is a QC error: dropped and counted against the candidate.
VALID_AMBIGUITY_CODES = frozenset({1, 2, 3, 4, 5, 6, 9})




def _to_int(v, default=None):
    """Parse a NMR-STAR field to int; return default on null/non-numeric."""
    if v in NULLS:
        return default
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _atomic_write(path, write_fn):
    """Write via <path>.tmp then os.replace so a reader never sees a half file."""
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        write_fn(f)
    os.replace(tmp, path)


def list_api_entry_ids(url=BMRB_LIST_URL, retries=3):
    """Fetch the authoritative macromolecule entry-id set from api.bmrb.io.

    A terminal failure raises rather than returning a partial id set."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                'Accept': 'application/json',
                'User-Agent': 'deepshift/0.1.0',
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            if isinstance(data, dict):
                for key in ('data', 'result', 'entry_ids'):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if not isinstance(data, list) or not data:
                raise ValueError('unexpected list_entries payload')
            return sorted({str(x) for x in data}, key=lambda s: (len(s), s))
        except Exception as e:  # retried then re-raised
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f'failed to fetch BMRB entry list from {url}: {last}')


def _dated_snapshot_dir(raw_root, when=None):
    when = when or datetime.now(UTC).strftime('%Y%m%d')
    return os.path.join(raw_root, f'snapshot_{when}')


def _latest_prior_snapshot(raw_root, exclude):
    """Newest existing snapshot dir under raw_root (for --link-dest), or None."""
    if not os.path.isdir(raw_root):
        return None
    cands = []
    for name in os.listdir(raw_root):
        p = os.path.join(raw_root, name)
        if name.startswith('snapshot_') and os.path.isdir(p) and p != exclude:
            cands.append(p)
    return sorted(cands)[-1] if cands else None


def rsync_snapshot(raw_root, when=None, link_dest=None, dry_run=False):
    """rsync the top-level _3.str archive into a dated snapshot dir, hard-linking
    unchanged files from the newest prior snapshot via --link-dest; ``dry_run``
    prints the command without running it."""
    snapshot = _dated_snapshot_dir(raw_root, when)
    os.makedirs(snapshot, exist_ok=True)
    if link_dest is None:
        link_dest = _latest_prior_snapshot(raw_root, exclude=snapshot)

    cmd = [
        'rsync', '-a', '--prune-empty-dirs',
        '--include=/bmr*/',
        '--include=/bmr*/bmr*_3.str',
        '--exclude=/bmr*/*/',
        '--exclude=*',
    ]
    if link_dest:
        cmd.append(f'--link-dest={os.path.abspath(link_dest)}')
    cmd += [RSYNC_URL, snapshot + os.sep]

    print('  rsync:', ' '.join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)
    return snapshot


def fetch_selected_snapshot(raw_root, entry_ids, when=None):
    """Download an explicit small set of immutable NMR-STAR entries."""
    snapshot = _dated_snapshot_dir(raw_root, when)
    os.makedirs(snapshot, exist_ok=True)
    for entry_id in entry_ids:
        directory = os.path.join(snapshot, f'bmr{entry_id}')
        destination = os.path.join(directory, f'bmr{entry_id}_3.str')
        if os.path.isfile(destination):
            continue
        os.makedirs(directory, exist_ok=True)
        url = STAR_URL.format(id=entry_id)
        print(f'  fetch {url} -> {destination}')
        urllib.request.urlretrieve(url, destination)
    return snapshot


def _id_sort_key(v):
    """Total order over STAR IDs that may be numeric or an inferred string.

    Numerics sort first by value; inferred (``fc:``/``uncat:``) ids sort after by
    text -- avoids a ``TypeError`` when a real numeric id mixes with an inferred one."""
    s = str(v)
    return (0, int(s)) if s.lstrip('+-').isdigit() else (1, s)


def _entity_assembly_authority(entry):
    """Authoritative assembly<->entity maps from the STAR ``_Entity_assembly`` loop
    (``ID`` -> ``Entity_ID``): recovers a shift row's missing entity/assembly id
    instead of quarantining it (e.g. BMRB 19266: 1,362 rows carry Entity_ID=1 with a
    null Entity_assembly_ID, while the loop declares assembly ID=1 -> Entity_ID=1).

    Returns ``(asm_to_entities, entity_to_asms)``, both empty when the loop is
    absent/unparseable so the caller falls back to shift-row inference."""
    asm_to_entities = defaultdict(set)
    entity_to_asms = defaultdict(set)
    if entry is None:
        return asm_to_entities, entity_to_asms
    try:
        loops = entry.get_loops_by_category('entity_assembly')
    except Exception:  # absent/odd category => no authority
        return asm_to_entities, entity_to_asms
    for loop in loops:
        idx = {t.split('.')[-1]: i for i, t in enumerate(loop.get_tag_names())}
        asm_j, ent_j = idx.get('ID'), idx.get('Entity_ID')
        if asm_j is None or ent_j is None:
            continue
        for row in loop.data:
            a, e = _to_int(row[asm_j]), _to_int(row[ent_j])
            if a is not None and e is not None:
                asm_to_entities[a].add(e)
                entity_to_asms[e].add(a)
    return asm_to_entities, entity_to_asms


def _resolve_entity_assembly_ids(rows, entry=None):
    """Fill a row's missing entity/assembly ID via a proven-unique assembly<->entity
    relationship: fills the missing side only when the other
    side maps to exactly one partner, consulting the ``_Entity_assembly`` authority
    when the shift rows alone can't resolve it. Stays None (quarantined downstream,
    never defaulted to 1) if both sources remain ambiguous."""
    asm_to_entities = defaultdict(set)
    entity_to_asms = defaultdict(set)
    for r in rows:
        a, e = r['entity_assembly_id'], r['entity_id']
        if a is not None and e is not None:
            asm_to_entities[a].add(e)
            entity_to_asms[e].add(a)
    # Shift-row evidence wins; the _Entity_assembly authority only fills what rows left ambiguous.
    auth_asm_to_entities, auth_entity_to_asms = _entity_assembly_authority(entry)
    for r in rows:
        if r['entity_id'] is None and r['entity_assembly_id'] is not None:
            ents = asm_to_entities.get(r['entity_assembly_id'])
            if not (ents and len(ents) == 1):
                ents = auth_asm_to_entities.get(r['entity_assembly_id'])
            if ents and len(ents) == 1:
                r['entity_id'] = next(iter(ents))
        if r['entity_assembly_id'] is None and r['entity_id'] is not None:
            asms = entity_to_asms.get(r['entity_id'])
            if not (asms and len(asms) == 1):
                asms = auth_entity_to_asms.get(r['entity_id'])
            if asms and len(asms) == 1:
                r['entity_assembly_id'] = next(iter(asms))


def _sf_tag(sf, name):
    """First value of a saveframe tag (by unprefixed name), or None."""
    try:
        vals = sf.get_tag(name)
    except Exception:  # missing tag => None
        return None
    return vals[0] if vals else None


def _iter_shift_loops(entry):
    """Yield ``(list_id, sample_condition_id, loop)`` for every shift loop.

    Iterates ``assigned_chemical_shifts`` saveframes (never globs) so the list ID
    and ``Sample_condition_list_ID`` travel with the loop; falls back to categorized
    loops for the rare entry whose loop sits outside a categorized saveframe."""
    yielded = False
    for si, sf in enumerate(entry.get_saveframes_by_category('assigned_chemical_shifts'),
                            start=1):
        loop = None
        for lp in sf.loops:
            if (lp.category or '').lstrip('_').lower() == 'atom_chem_shift':
                loop = lp
                break
        if loop is None:
            continue
        # List identity comes from the saveframe ID tag, else a namespaced framecode id -- never the bare enumeration index, which collides across entries.
        list_id = _to_int(_sf_tag(sf, 'ID'))
        if list_id is None:
            list_id = f'fc:{sf.name}' if getattr(sf, 'name', None) else f'fc:sf{si}'
        sample_cond = _to_int(_sf_tag(sf, 'Sample_condition_list_ID'))
        yielded = True
        yield list_id, sample_cond, loop
    if yielded:
        return
    for li, loop in enumerate(entry.get_loops_by_category('atom_chem_shift'), start=1):
        # Uncategorized loop with no assigned-shift saveframe: namespace the inferred id so it never masquerades as a real numeric list id.
        yield f'uncat:{li}', None, loop


def parse_entry_shifts(str_path, bmrb_id, all_lists=False, entry=None):
    """Parse one ``bmr<ID>_3.str``'s ``_Atom_chem_shift`` loop(s) entity-aware.

    Emits one long-format row per assigned shift with the full identity key; rows are
    not collapsed on (residue, comp), so two entities sharing a comp_index_id stay
    distinct. Default: only the list with the most valid rows (deposition-grain).
    ``all_lists=True`` (entity-grain) instead returns rows from every list, each
    with a ``sample_condition_id``, for the representative rule to pivot and score
    every (list, assembly) candidate. ``entry`` may be pre-parsed to skip a re-parse.

    Returns ``(rows, prov)``; prov has keys {'chosen_shift_list_id',
    'all_shift_list_ids', 'per_list_row_counts', 'sample_condition_list_ids',
    'per_list_sample_condition_ids', 'n_rows'}. No shift rows -> ([], prov-with-empties).
    """
    import pynmrstar  # local import: keeps module import cheap / offline-friendly

    if entry is None:
        entry = pynmrstar.Entry.from_file(str_path)

    parsed_rows = []  # flat; entity/assembly ids may be None until resolved
    for list_id_base, sample_cond_base, loop in _iter_shift_loops(entry):
        idx = {t.split('.')[-1]: i for i, t in enumerate(loop.get_tag_names())}

        def cell(row, name, _idx=idx):
            j = _idx.get(name)
            return row[j] if j is not None else None

        for row in loop.data:
            val = cell(row, 'Val')
            if val in NULLS:
                continue
            try:
                value = float(val)
            except (ValueError, TypeError):
                continue
            # textual 'nan'/'inf'/'-inf' parse to real floats; reject so a nonfinite value is never an observed cell.
            if not math.isfinite(value):
                continue

            # comp_index_id is canonical; fall back to Seq_ID when absent.
            comp_index_id = _to_int(cell(row, 'Comp_index_ID'))
            if comp_index_id is None:
                comp_index_id = _to_int(cell(row, 'Seq_ID'))
            if comp_index_id is None:
                continue  # unusable without a residue index

            comp_id = cell(row, 'Comp_ID')
            atom_id = cell(row, 'Atom_ID')
            if comp_id in NULLS or atom_id in NULLS:
                continue

            # Missing ids stay None rather than defaulting to (1,1,1); they are resolved below only via a proven-unique assembly<->entity relationship, else quarantined.
            eaid = _to_int(cell(row, 'Entity_assembly_ID'))
            eid = _to_int(cell(row, 'Entity_ID'))
            # Prefer the row-level assigned-list id, else the saveframe ID.
            list_id = _to_int(cell(row, 'Assigned_chem_shift_list_ID'),
                              default=list_id_base)
            sample_cond = _to_int(cell(row, 'Sample_condition_list_ID'),
                                  default=sample_cond_base)

            parsed_rows.append({
                'bmrb_id': str(bmrb_id),
                'shift_list_id': list_id,
                'entity_assembly_id': eaid,
                'entity_id': eid,
                'comp_index_id': comp_index_id,
                'comp_id': str(comp_id).upper(),
                'atom_id': str(atom_id).upper(),
                'value': value,
                'ambiguity_code': _to_int(cell(row, 'Ambiguity_code')),
                'sample_condition_id': sample_cond,
            })

    _resolve_entity_assembly_ids(parsed_rows, entry=entry)

    by_list = {}
    list_sample_cond = defaultdict(set)
    for r in parsed_rows:
        if r['sample_condition_id'] is not None:
            list_sample_cond[r['shift_list_id']].add(r['sample_condition_id'])
        by_list.setdefault(r['shift_list_id'], []).append(r)

    per_list = {lid: len(rows) for lid, rows in by_list.items()}
    ordered_lists = sorted(per_list, key=_id_sort_key)
    prov = {
        'chosen_shift_list_id': None,
        'all_shift_list_ids': [str(l) for l in ordered_lists],
        'per_list_row_counts': {str(k): per_list[k] for k in ordered_lists},
        'per_list_sample_condition_ids': {
            str(k): sorted(list_sample_cond.get(k, ())) for k in ordered_lists
        },
        'sample_condition_list_ids': [],
        'n_rows': 0,
    }
    if not by_list:
        return [], prov

    if all_lists:
        # Entity-grain path: keep every list; the representative rule scores each (list, assembly, entity) candidate, in list-id order.
        rows = [r for lid in ordered_lists for r in by_list[lid]]
        prov['n_rows'] = len(rows)
        prov['sample_condition_list_ids'] = sorted(
            {c for cs in list_sample_cond.values() for c in cs})
        return rows, prov

    # Most rows wins; ties keep the smaller list id (first maximal element of the id-ordered list).
    chosen = max(ordered_lists, key=lambda lid: per_list[lid])
    prov['chosen_shift_list_id'] = chosen
    prov['sample_condition_list_ids'] = sorted(list_sample_cond.get(chosen, ()))
    prov['n_rows'] = per_list[chosen]
    return by_list[chosen], prov


def parse_entity_poly_seq(entry):
    """Return ``{entity_id: {'tokens': [...], 'scope': 'full'|'observed_only'}}``.

    Reads each entity's full ``_Entity_poly_seq`` (ordered by ``Num``), falling back
    to ``_Entity_comp_index`` when absent; tokens are uppercased NMR-STAR codes
    (never collapsed to X). No full declared polymer -> ``observed_only`` (never
    hashed as an exact sequence)."""
    out = {}
    for cat, num_tag in (('entity_poly_seq', 'Num'), ('entity_comp_index', 'ID')):
        for loop in entry.get_loops_by_category(cat):
            idx = {t.split('.')[-1]: i for i, t in enumerate(loop.get_tag_names())}
            eid_j = idx.get('Entity_ID')
            num_j = idx.get(num_tag)
            # Column 0 is a valid index; ``or`` here would treat it as absent and mis-pick Mon_ID or drop the whole polymer loop.
            comp_j = idx.get('Comp_ID')
            if comp_j is None:
                comp_j = idx.get('Mon_ID')
            if comp_j is None:
                continue
            per_entity = defaultdict(list)
            for row in loop.data:
                # never synthesize entity 1 -- a present-but-null Entity_ID row is skipped; a loop with no Entity_ID column keys tokens under the None sentinel.
                if eid_j is not None:
                    eid = _to_int(row[eid_j])
                    if eid is None:
                        continue
                else:
                    eid = None
                comp = row[comp_j]
                if comp in NULLS:
                    continue
                num = _to_int(row[num_j]) if num_j is not None else len(per_entity[eid]) + 1
                per_entity[eid].append((num if num is not None else 0, str(comp).upper()))
            for eid, pairs in per_entity.items():
                if eid in out:
                    continue  # poly_seq (first loop) wins over comp_index fallback
                pairs.sort(key=lambda p: p[0])
                out[eid] = {'tokens': [c for _, c in pairs],
                            'nums': [n for n, _ in pairs],
                            'scope': 'full'}
    return out


def _merge_entity_sequences(acc, bmrb_id, entity_seqs):
    """Accumulate per-entry polymer sequences keyed by ``(bmrb_id, entity_id)``:
    a bare STAR entity_id is unique only within one deposition, so keying by
    (bmrb, entity_id) stops a later BMRB's entity 1 from overwriting an earlier
    one."""
    for eid, seq in entity_seqs.items():
        acc[(str(bmrb_id), eid)] = seq
    return acc


def _supported_cells(residues):
    """(n_finite_target_cells, R, n_backbone_cells) over a candidate's residues."""
    n_cells = n_backbone = 0
    covered_positions = set()
    for comp_index_id, res in residues.items():
        comp_id = str(res['comp_id']).upper()
        has = False
        for atom, value in res['cells'].items():
            # only finite observed cells count toward n_cells / R / backbone.
            if value is None or not math.isfinite(value):
                continue
            n_cells += 1
            has = True
            if atom in BACKBONE_ATOMS:
                n_backbone += 1
        if has and comp_id in STANDARD_COMP_IDS:
            covered_positions.add(comp_index_id)
    return n_cells, len(covered_positions), n_backbone


def _candidate_sort_key(cand):
    """Lexicographic rank. Larger is better for the first three
    scores, so negate; then fewest conflicts; then id order; then payload sha."""
    list_id = cand['shift_list_id']
    asm_id = cand['entity_assembly_id']

    return (
        -cand['n_cells'],
        -cand['R'],
        -cand['n_backbone'],
        cand['n_conflict_cells'],
        _representative_id_key(list_id),
        _representative_id_key(asm_id),
        tuple(_representative_id_key(v)
              for v in cand.get('sample_condition_ids', [])),
        cand['payload_sha256'],
    )


def _representative_id_key(value):
    s = str(value)
    return (0, int(s)) if s.lstrip('+-').isdigit() else (1, s)


def _candidate_payload(cand):
    observed = []
    for comp_index_id, residue in cand.get('residues', {}).items():
        comp_id = str(residue.get('comp_id', '')).upper()
        for atom_id, value in residue.get('cells', {}).items():
            if value is None or not math.isfinite(value):
                continue
            observed.append([
                int(comp_index_id), comp_id, str(atom_id).lower(), float(value)])
    observed.sort(key=lambda cell: (cell[0], cell[2], cell[1], cell[3]))
    conditions = sorted(
        (str(value) for value in cand.get('sample_condition_ids', [])),
        key=_representative_id_key)
    return {
        'bmrb_id': str(cand['bmrb_id']),
        'shift_list_id': str(cand['shift_list_id']),
        'entity_assembly_id': str(cand['entity_assembly_id']),
        'entity_id': str(cand['entity_id']),
        'exact_seq_sha256': cand['exact_seq_sha256'],
        'sample_condition_ids': conditions,
        'observed_cells': observed,
        'n_conflict_cells': int(cand.get('n_conflict_cells', 0)),
    }


def _candidate_payload_sha256(cand):
    from .. import domain as _id
    return _id.sha256_hex(_candidate_payload(cand))


def select_representatives(candidates, coverage_floor=0.80):
    """Choose one label set per logical entity, at an 80% coverage floor.

    ``candidates``: dicts with 'bmrb_id', 'shift_list_id', 'entity_assembly_id',
    'entity_id', 'exact_seq_sha256', 'residues', 'n_conflict_cells', and optional
    'sample_condition_ids'; never unioned or averaged.

    Returns ``(selected, rows)``: ``selected`` maps logical_entity_id -> the winner
    (with label_set_id added); ``rows`` is the full label_set_selection.csv payload
    (every candidate + scores + selected/reason + alternate_label_set_disagrees when
    a common cell diverges beyond 1e-6)."""
    from .. import domain as _id

    # Group candidates by logical entity (bmrb, entity_id, exact_seq_sha256).
    by_entity = defaultdict(list)
    for cand in candidates:
        cand = dict(cand)
        cand['n_cells'], cand['R'], cand['n_backbone'] = _supported_cells(cand['residues'])
        cand['payload_sha256'] = _candidate_payload_sha256(cand)
        le = _id.logical_entity_id(cand['bmrb_id'], cand['entity_id'],
                                   cand['exact_seq_sha256'])
        cand['logical_entity_id'] = le
        condition_ids = list(cand.get('sample_condition_ids', ()))
        if len(condition_ids) > 1:
            raise ValueError(
                "one candidate carries multiple sample conditions despite the "
                "condition-grained pivot"
            )
        cand['label_set_id'] = _id.label_set_id(
            cand['bmrb_id'], cand['shift_list_id'], cand['entity_assembly_id'],
            cand['entity_id'], cand['exact_seq_sha256'],
            condition_ids[0] if condition_ids else None)
        by_entity[le].append(cand)

    selected = {}
    rows = []
    for le, cands in by_entity.items():
        rmax = max(c['R'] for c in cands)
        floor = _ceil(rmax * coverage_floor)  # ceil(0.80 * Rmax)
        eligible = [c for c in cands if c['n_cells'] > 0 and c['R'] >= floor]
        if not eligible:
            for c in cands:
                rows.append(_selection_row(c, selected=False,
                                           reason='no_supported_cells_or_below_floor',
                                           rmax=rmax, floor=floor,
                                           coverage_fraction=coverage_floor))
            continue
        winner = min(eligible, key=_candidate_sort_key)
        selected[le] = winner
        for c in cands:
            is_win = c is winner
            reason = 'selected' if is_win else (
                'eligible_not_selected' if c in eligible else 'below_coverage_floor')
            disagrees = (not is_win) and _cells_disagree(winner['residues'],
                                                          c['residues'])
            rows.append(_selection_row(c, selected=is_win, reason=reason,
                                       rmax=rmax, floor=floor,
                                       coverage_fraction=coverage_floor,
                                       alternate_disagrees=disagrees))
        # Flag the winner if any retained alternate disagrees on a common cell.
        if any(r['alternate_label_set_disagrees'] for r in rows
               if r['logical_entity_id'] == le and not r['selected']):
            for r in rows:
                if r['logical_entity_id'] == le and r['selected']:
                    r['alternate_label_set_disagrees'] = True
    return selected, rows


def _ceil(x):
    return math.ceil(x - 1e-9)


def _detect_numbering_offset(residues, num_to_token):
    """Detect one global constant numbering offset between observed comp_index_ids
    and the authoritative polymer numbering (author-vs-entity numbering; measured:
    BMRB 18001 +1, 50088 -4, 2200 -1) -- a numbering convention, not a residue-code
    contradiction.

    Returns delta ``d`` (``polymer_num = comp_index_id + d``) that best reconciles
    observed codes with polymer tokens, or ``0`` when no offset both beats the
    literal alignment and reconciles a strict majority (a clean deposition is
    never perturbed)."""
    observed = [(cix, str(res.get('comp_id', '')).upper())
                for cix, res in residues.items()]
    n = len(observed)
    if n < 2:
        return 0  # a single residue "matches" some offset trivially; no evidence
    # Every (polymer_num - comp_index_id) where the polymer token equals the
    # observed code is a candidate offset; tally[d] counts residues a single global d reconciles.
    tally = Counter()
    for cix, comp_id in observed:
        for num, token in num_to_token.items():
            if token == comp_id:
                tally[num - cix] += 1
    if not tally:
        return 0
    base = tally.get(0, 0)          # literal alignment (delta 0)
    if base == n:
        return 0                    # already perfectly aligned; never perturb
    # Most-reconciled offset wins, but only if it is the unique best one: in a
    # repetitive polymer two deltas can reconcile the same count, and picking either
    # would coin-flip-mis-bind labels. A tie falls back to the literal alignment
    # (delta 0), dropping the disagreeing residue instead of mis-binding it.
    best_d = max(tally, key=lambda d: (tally[d], -abs(d)))
    m = tally[best_d]
    if sum(1 for d, count in tally.items() if count == m) > 1:
        return 0
    if best_d != 0 and m > base and m >= 2 and m * 2 > n:
        return best_d
    return 0


def _validate_residue_codes(residues, tokens, nums):
    """Check observed residue codes against the polymer authority (``tokens``/``nums``).

    A residue must sit at a declared position and carry the exact token there, else it
    is dropped without being minted as a label, and counted so the representative rule
    penalizes the candidate. A constant author-vs-entity numbering offset is detected
    and applied first, so an offset deposition realigns instead of being dropped.

    Returns ``(kept_residues, n_code_mismatch, n_offpolymer)``."""
    num_to_token = dict(zip(nums, tokens))
    delta = _detect_numbering_offset(residues, num_to_token)
    kept = {}
    n_code_mismatch = n_offpolymer = 0
    for cix, res in residues.items():
        polymer_num = cix + delta
        token = num_to_token.get(polymer_num)
        comp_id = str(res.get('comp_id', '')).upper()
        if token is None:
            # Not in the declared polymer -- cannot be validated, so it is dropped.
            n_offpolymer += 1
            continue
        if str(token).upper() != comp_id:
            # Disagrees with the authoritative polymer token (wrong amino acid); dropped.
            n_code_mismatch += 1
            continue
        # Renumber to the true polymer position (materialization keys on polymer Num); delta==0 keeps cix.
        kept[polymer_num] = res
    return kept, n_code_mismatch, n_offpolymer


def _pivot_candidate_rows(rows):
    """Deterministically pivot one condition's raw observations: conflict accounting
    depends only on the observation multiset, not row order. A residue-code
    contradiction invalidates the whole position; a value conflict invalidates that
    cell (near-identical duplicates keep the smallest value as a stable representative)."""
    by_position = defaultdict(list)
    for row in rows:
        by_position[row['comp_index_id']].append(row)

    residues = {}
    n_conflict_cells = 0
    for comp_index_id, observations in sorted(by_position.items()):
        comp_ids = {row['comp_id'] for row in observations}
        if len(comp_ids) != 1:
            # Every raw observation at a contradictory residue position is lost.
            n_conflict_cells += len(observations)
            continue

        comp_id = next(iter(comp_ids))
        by_atom = defaultdict(list)
        for row in observations:
            by_atom[row['atom']].append(row)

        cells, ambiguities = {}, {}
        for atom, atom_rows in sorted(by_atom.items()):
            # Invalid ambiguity metadata is independently countable even when the numeric cell also value-conflicts.
            n_conflict_cells += sum(
                row['ambiguity_code'] is not None
                and row['ambiguity_code'] not in VALID_AMBIGUITY_CODES
                for row in atom_rows)
            values = sorted(float(row['value']) for row in atom_rows)
            if values[-1] - values[0] > DUP_VALUE_EPS:
                n_conflict_cells += len(values) - 1
                continue

            cells[atom] = values[0]
            valid_ambiguities = {
                row['ambiguity_code'] for row in atom_rows
                if row['ambiguity_code'] in VALID_AMBIGUITY_CODES
            }
            if len(valid_ambiguities) > 1:
                n_conflict_cells += 1
                ambiguities[atom] = None
            else:
                ambiguities[atom] = next(iter(valid_ambiguities), None)

        # A completely voided position contributes no emitted label residue.
        if cells:
            residues[comp_index_id] = {
                'comp_id': comp_id, 'cells': cells, 'ambig': ambiguities,
            }
    return residues, n_conflict_cells


def build_candidates_from_long(long_rows, entity_sequences):
    """Pivot long-format shift rows into per-(list, assembly, entity) candidates, each
    hashed from ``entity_sequences[entity_id]``'s full polymer tokens. No full
    declared sequence -> returned in ``quarantine`` (never given a synthetic hash).

    Returns ``(candidates, quarantine)``."""
    from .. import domain as _id

    entity_sequences = entity_sequences or {}
    # a row missing entity_id resolves to the sole declared entity only when the entry proves exactly one full entity; otherwise None stays quarantined below.
    declared = {eid: s for eid, s in entity_sequences.items()
                if s and s.get('scope') == 'full' and s.get('tokens')}
    real_ids = [eid for eid in declared if eid is not None]
    sole_entity = real_ids[0] if (len(declared) == 1 and len(real_ids) == 1) else None
    grouped = defaultdict(list)
    quarantine = []
    for r in long_rows:
        atom = str(r['atom_id']).lower()
        if atom not in TARGET_ATOMS:
            continue
        value = r.get('value')
        # a nonfinite value is never an observed cell.
        if value is None or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            continue
        eid = r['entity_id']
        if eid is None and sole_entity is not None:
            eid = sole_entity
        asm = r['entity_assembly_id']
        lst = r['shift_list_id']
        cond = r.get('sample_condition_id')
        # unresolved identity is quarantined, never keyed as (...,1,1).
        if eid is None or asm is None or lst is None:
            quarantine.append({
                'bmrb_id': str(r['bmrb_id']),
                'entity_id': '' if eid is None else str(eid),
                'shift_list_id': '' if lst is None else str(lst),
                'entity_assembly_id': '' if asm is None else str(asm),
                'reason': 'unresolved_identity_missing_id'})
            continue
        # sample_condition_id is part of the candidate key, so different conditions become distinct candidates, never unioned.
        key = (str(r['bmrb_id']), lst, asm, eid, cond)
        comp_id = str(r['comp_id']).upper()
        grouped[key].append({
            'comp_index_id': r['comp_index_id'], 'comp_id': comp_id,
            'atom': atom, 'value': value,
            'ambiguity_code': r.get('ambiguity_code'),
        })

    candidates = []
    for (bmrb_id, list_id, asm_id, entity_id, cond), rows in grouped.items():
        residues, n_conflict_cells = _pivot_candidate_rows(rows)
        # exact entity-id authority only -- a sibling id's (or None's) full sequence is never borrowed, even as sole shift-bearing entity.
        seq = entity_sequences.get(entity_id)
        if not seq or seq.get('scope') != 'full' or not seq.get('tokens'):
            quarantine.append({
                'bmrb_id': bmrb_id, 'entity_id': str(entity_id),
                'shift_list_id': str(list_id), 'entity_assembly_id': str(asm_id),
                'reason': 'sequence_scope_observed_only'})
            continue
        tokens = seq['tokens']
        # Full polymer positions (Entity_poly_seq Num). Fall back to 1..N only if
        # the source loop carried no numbering; comp_index_id then aligns to 1..N.
        nums = seq.get('nums') or list(range(1, len(tokens) + 1))
        # validate every observed (comp_index_id, comp_id) against the polymer's
        # authoritative token before minting a label; a disagreeing or off-polymer
        # residue is dropped here, so no wrong-code label reaches the cache.
        residues, n_code_mismatch, n_offpolymer = _validate_residue_codes(
            residues, tokens, nums)
        n_conflict_cells += n_code_mismatch + n_offpolymer
        alignment_sequence = _id.alignment_sequence_from_tokens(tokens)
        candidates.append({
            'bmrb_id': bmrb_id,
            'shift_list_id': list_id,
            'entity_assembly_id': asm_id,
            'entity_id': entity_id,
            'exact_seq_sha256': _id.exact_polymer_sequence_sha256(tokens),
            # Full entity sequence on the candidate: the winner keeps it so the catalog sidecar is keyed by label_set_id, never re-concatenated downstream.
            'polymer_tokens': list(tokens),
            'polymer_seq_nums': [int(n) for n in nums],
            'alignment_sequence': alignment_sequence,
            'alignment_sequence_sha256':
                _id.alignment_sequence_sha256(alignment_sequence),
            'residues': residues,
            'n_conflict_cells': n_conflict_cells,
            # Residues dropped for code mismatch or for falling off the polymer.
            'n_residue_code_mismatch': n_code_mismatch,
            'n_offpolymer_residues': n_offpolymer,
            # One condition per candidate (part of the key); never unioned across conditions.
            'sample_condition_ids': [] if cond is None else [cond],
        })
    return candidates, quarantine


def selected_shift_records(selected, entity_sequences):
    """Wide-format residue rows for the selected label sets: one row per (label set,
    residue) with label_set_id/logical_entity_id/label_residue_id + identity
    columns, then the ``<atom>_shift`` cells; sorted by (label_set_id, comp_index_id)."""
    from .. import domain as _id
    records = []
    for le, cand in selected.items():
        lsid = cand['label_set_id']
        for cix in sorted(cand['residues']):
            res = cand['residues'][cix]
            comp_id = str(res['comp_id']).upper()
            rec = {
                'label_set_id': lsid,
                'logical_entity_id': le,
                'label_residue_id': _id.label_residue_id(lsid, cix, comp_id),
                'bmrb_id': str(cand['bmrb_id']),
                'entity_id': str(cand['entity_id']),
                'shift_list_id': str(cand['shift_list_id']),
                'entity_assembly_id': str(cand['entity_assembly_id']),
                'exact_polymer_sequence_sha256': cand['exact_seq_sha256'],
                'comp_index_id': cix,
                'comp_id': comp_id,
                'residue_id': cix,          # back-compat alias
                'residue_code': comp_id,
            }
            ambig = res.get('ambig', {})
            for atom, value in res['cells'].items():
                if value is not None:
                    rec[f'{atom}_shift'] = value
                    code = ambig.get(atom)
                    if code is not None:
                        rec[f'{atom}_ambiguity_code'] = code
            records.append(rec)
    records.sort(key=lambda r: (r['label_set_id'], r['comp_index_id']))
    return records


ENTITY_CATALOG_COLUMNS = (
    'logical_entity_id', 'label_set_id', 'bmrb_id', 'entity_id', 'shift_list_id',
    'entity_assembly_id', 'sample_condition_ids', 'exact_polymer_sequence_sha256',
    'alignment_sequence_sha256', 'sequence_scope', 'polymer_length',
    'polymer_seq_nums', 'polymer_token_sequence', 'alignment_sequence',
    'n_supported_residues', 'n_supported_cells')


def entity_catalog_records(selected, entity_sequences=None):
    """One row per selected logical entity: the authoritative entity-sequence
    sidecar keyed by ``label_set_id``, carrying the full
    polymer tokens, true positions, and alignment sequence + hash from the selected
    candidate itself, not a BMRB-global map, which would be overwritten across depositions.
    ``entity_sequences`` is accepted for backward compatibility but unused."""
    rows = []
    for le, cand in selected.items():
        tokens = cand.get('polymer_tokens', [])
        nums = cand.get('polymer_seq_nums') or list(range(1, len(tokens) + 1))
        alignment_sequence = cand.get('alignment_sequence', '')
        rows.append({
            'logical_entity_id': le,
            'label_set_id': cand['label_set_id'],
            'bmrb_id': str(cand['bmrb_id']),
            'entity_id': str(cand['entity_id']),
            'shift_list_id': str(cand['shift_list_id']),
            'entity_assembly_id': str(cand['entity_assembly_id']),
            'sample_condition_ids': ','.join(
                str(value) for value in cand.get('sample_condition_ids', [])
            ),
            'exact_polymer_sequence_sha256': cand['exact_seq_sha256'],
            'alignment_sequence_sha256': cand.get('alignment_sequence_sha256', ''),
            'sequence_scope': 'full',
            'polymer_length': len(tokens),
            'polymer_seq_nums': ','.join(str(n) for n in nums),
            'polymer_token_sequence': ' '.join(str(t) for t in tokens),
            'alignment_sequence': alignment_sequence,
            'n_supported_residues': cand['R'],
            'n_supported_cells': cand['n_cells'],
        })
    rows.sort(key=lambda r: r['logical_entity_id'])
    return rows


def read_entity_catalog(path):
    """Load the entity_catalog.csv sidecar into ``{label_set_id: record}``, each
    record carrying the full entity sequence:

        {'alignment_sequence': str, 'seq_positions': [int,...] (true polymer Num),
         'polymer_length': int, 'exact_polymer_sequence_sha256': str,
         'alignment_sequence_sha256': str, 'logical_entity_id'/'bmrb_id'/'entity_id': str}

    Raises if a row's declared polymer_length disagrees with its sequence/position
    lengths; a corrupt sidecar is refused rather than used to mis-map labels."""
    import csv as _csv

    from .. import domain as _id
    out = {}
    with open(path, newline='') as fh:
        for row in _csv.DictReader(fh):
            lsid = row.get('label_set_id')
            if not lsid:
                raise ValueError('entity_catalog.csv contains a row without label_set_id')
            if lsid in out:
                raise ValueError(f'entity_catalog.csv repeats label_set_id {lsid}')
            aln = (row.get('alignment_sequence') or '').strip()
            nums_raw = (row.get('polymer_seq_nums') or '').strip()
            positions = [int(x) for x in nums_raw.split(',') if x != ''] \
                if nums_raw else list(range(1, len(aln) + 1))
            tokens = [t for t in (row.get('polymer_token_sequence') or '').split()
                      if t]
            plen = int(row['polymer_length']) if row.get('polymer_length') else len(aln)
            if len(aln) != plen or len(positions) != plen:
                raise ValueError(
                    f'entity_catalog.csv row {lsid}: polymer_length={plen} but '
                    f'alignment_sequence has {len(aln)} chars / {len(positions)} '
                    f'positions -- corrupt full-sequence sidecar')
            # 3-letter tokens are the authoritative residue codes; a token count that disagrees with polymer_length is refused, since validating against it would mis-map.
            if tokens and len(tokens) != plen:
                raise ValueError(
                    f'entity_catalog.csv row {lsid}: polymer_length={plen} but '
                    f'polymer_token_sequence has {len(tokens)} tokens -- corrupt '
                    'full-sequence sidecar')
            if not tokens:
                raise ValueError(
                    f'entity_catalog.csv row {lsid}: polymer_token_sequence is required '
                    'to verify scientific identity'
                )
            actual_exact = _id.exact_polymer_sequence_sha256(tokens)
            declared_exact = row.get('exact_polymer_sequence_sha256', '')
            if actual_exact != declared_exact:
                raise ValueError(
                    f'entity_catalog.csv row {lsid}: exact sequence hash mismatch'
                )
            actual_alignment = _id.alignment_sequence_sha256(aln)
            declared_alignment = row.get('alignment_sequence_sha256', '')
            if actual_alignment != declared_alignment:
                raise ValueError(
                    f'entity_catalog.csv row {lsid}: alignment sequence hash mismatch'
                )
            actual_logical = _id.logical_entity_id(
                row.get('bmrb_id'), row.get('entity_id'), actual_exact
            )
            if actual_logical != row.get('logical_entity_id'):
                raise ValueError(
                    f'entity_catalog.csv row {lsid}: logical_entity_id mismatch'
                )
            condition_ids = [
                value for value in (row.get('sample_condition_ids') or '').split(',')
                if value
            ]
            if len(condition_ids) > 1:
                raise ValueError(
                    f'entity_catalog.csv row {lsid}: multiple sample conditions'
                )
            actual_label_set = _id.label_set_id(
                row.get('bmrb_id'), row.get('shift_list_id'),
                row.get('entity_assembly_id'), row.get('entity_id'), actual_exact,
                condition_ids[0] if condition_ids else None,
            )
            if actual_label_set != lsid:
                raise ValueError(
                    f'entity_catalog.csv row {lsid}: label_set_id mismatch'
                )
            out[lsid] = {
                'alignment_sequence': aln,
                'seq_positions': positions,
                'seq_tokens': tokens,
                'polymer_length': plen,
                'exact_polymer_sequence_sha256':
                    row.get('exact_polymer_sequence_sha256', ''),
                'alignment_sequence_sha256':
                    row.get('alignment_sequence_sha256', ''),
                'logical_entity_id': row.get('logical_entity_id', ''),
                'bmrb_id': row.get('bmrb_id', ''),
                'entity_id': row.get('entity_id', ''),
                'shift_list_id': row.get('shift_list_id', ''),
                'entity_assembly_id': row.get('entity_assembly_id', ''),
                'sample_condition_ids': condition_ids,
            }
    return out


def default_entity_catalog_path(sibling_file):
    """entity_catalog.csv living next to ``sibling_file`` (shifts CSV / master)."""
    return os.path.join(os.path.dirname(os.path.abspath(sibling_file)),
                        'entity_catalog.csv')


def write_entity_tables(out_dir, selected, selection_rows, entity_sequences):
    """Atomically write entity_catalog.csv, label_set_selection.csv and the
    selected chemical_shifts.csv with all identities/hashes."""
    os.makedirs(out_dir, exist_ok=True)

    catalog = entity_catalog_records(selected, entity_sequences)
    _atomic_write(os.path.join(out_dir, 'entity_catalog.csv'),
                  lambda f: pd.DataFrame(
                      catalog, columns=list(ENTITY_CATALOG_COLUMNS)
                  ).to_csv(f, index=False))

    _atomic_write(os.path.join(out_dir, 'label_set_selection.csv'),
                  lambda f: pd.DataFrame(selection_rows).to_csv(f, index=False))

    records = selected_shift_records(selected, entity_sequences)
    ident_cols = ['label_set_id', 'logical_entity_id', 'label_residue_id',
                  'bmrb_id', 'entity_id', 'shift_list_id', 'entity_assembly_id',
                  'exact_polymer_sequence_sha256', 'comp_index_id', 'comp_id',
                  'residue_id', 'residue_code']
    present = set()
    for rec in records:
        present.update(rec)
    shift_cols = sorted(c for c in present if c.endswith('_shift'))
    ambig_cols = sorted(c for c in present if c.endswith('_ambiguity_code'))
    df = pd.DataFrame(records, columns=ident_cols + shift_cols + ambig_cols)
    _atomic_write(os.path.join(out_dir, 'chemical_shifts.csv'),
                  lambda f: df.to_csv(f, index=False))
    return {'n_logical_entities': len(catalog),
            'n_label_set_candidates': len(selection_rows),
            'n_selected_residue_rows': len(records)}


def _cells_disagree(winner_res, alt_res, eps=DUP_VALUE_EPS):
    """True iff any observed cell common to both differs by more than eps."""
    for cix, ares in alt_res.items():
        wres = winner_res.get(cix)
        if wres is None:
            continue
        for atom, av in ares['cells'].items():
            wv = wres['cells'].get(atom)
            if av is None or wv is None:
                continue
            if abs(av - wv) > eps:
                return True
    return False


def _selection_row(cand, *, selected, reason, rmax, floor, coverage_fraction,
                   alternate_disagrees=False):
    from .. import domain as _id
    payload = _candidate_payload(cand)
    return {
        'logical_entity_id': cand['logical_entity_id'],
        'label_set_id': cand['label_set_id'],
        'bmrb_id': str(cand['bmrb_id']),
        'entity_id': str(cand['entity_id']),
        'shift_list_id': str(cand['shift_list_id']),
        'entity_assembly_id': str(cand['entity_assembly_id']),
        'exact_polymer_sequence_sha256': cand['exact_seq_sha256'],
        'n_supported_cells': cand['n_cells'],
        'n_supported_residues': cand['R'],
        'n_supported_backbone_cells': cand['n_backbone'],
        'n_conflict_cells': cand.get('n_conflict_cells', 0),
        'r_max': rmax,
        'coverage_floor': floor,
        'coverage_fraction': coverage_fraction,
        'sample_condition_ids': ','.join(
            str(s) for s in cand.get('sample_condition_ids', [])),
        # Complete candidate ledger: a reader can recompute the payload hash and the winner, and compare alternates.
        'candidate_payload_json': _id.canonical_bytes(payload).decode('utf-8'),
        'payload_sha256': cand['payload_sha256'],
        'selected': bool(selected),
        'reason': reason,
        'alternate_label_set_disagrees': bool(alternate_disagrees),
    }


def entry_star_path(snapshot_dir, bmrb_id):
    """Exact path of an entry's raw file: <snapshot>/bmr<ID>/bmr<ID>_3.str."""
    return os.path.join(snapshot_dir, f'bmr{bmrb_id}', f'bmr{bmrb_id}_3.str')


def build_entity_tables_from_snapshot(snapshot_dir, api_ids, out_dir,
                                      coverage_floor=0.80):
    """Entity-mode production entrypoint: parses the snapshot once and
    emits entity_catalog.csv, label_set_selection.csv, and the entity-grained
    chemical_shifts.csv. Every assigned shift list pivots into per-(list, assembly,
    entity) candidates; the 80%-floor representative rule chooses one label set per
    logical entity. Sequence-less entities are quarantined. Returns (stats, manifest)."""
    import pynmrstar

    os.makedirs(out_dir, exist_ok=True)
    api_ids = sorted({str(x) for x in api_ids}, key=lambda s: (len(s), s))

    all_cands, all_quarantine = [], []
    entity_sequences = {}    # (bmrb, entity_id) -> {tokens, scope}; provenance only
    parsed, failed, missing, empty = [], [], [], []

    for i, bmrb in enumerate(api_ids):
        if (i + 1) % 2000 == 0:
            print(f'  parsed {i + 1}/{len(api_ids)} entries...', flush=True)
        path = entry_star_path(snapshot_dir, bmrb)
        if not os.path.isfile(path):
            missing.append(bmrb)
            continue
        try:
            entry = pynmrstar.Entry.from_file(path)
            long_rows, _prov = parse_entry_shifts(path, bmrb, all_lists=True, entry=entry)
            entity_seqs = parse_entity_poly_seq(entry)
        except Exception as e:  # recorded here, refused by the caller
            failed.append({'bmrb_id': bmrb, 'error': f'{type(e).__name__}: {e}'})
            continue
        parsed.append(bmrb)
        if not long_rows:
            empty.append(bmrb)
            continue
        cands, quarantine = build_candidates_from_long(long_rows, entity_seqs)
        all_cands.extend(cands)
        all_quarantine.extend(quarantine)
        # entity_id is unique only within a BMRB; keying by (bmrb, entity_id) stops a later deposition's entity 1 from overwriting an earlier one's polymer length.
        _merge_entity_sequences(entity_sequences, bmrb, entity_seqs)

    selected, selection_rows = select_representatives(
        all_cands, coverage_floor=coverage_floor)
    stats = write_entity_tables(out_dir, selected, selection_rows, entity_sequences)

    _atomic_write(os.path.join(out_dir, 'quarantine_observed_only.csv'),
                  lambda f: pd.DataFrame(
                      all_quarantine,
                      columns=['bmrb_id', 'entity_id', 'shift_list_id',
                               'entity_assembly_id', 'reason']).to_csv(f, index=False))

    manifest = {
        'stage': 'S1_fetch_entity',
        'coverage_floor': coverage_floor,
        'n_api_ids': len(api_ids), 'n_parsed': len(parsed),
        'n_missing_files': len(missing), 'missing_ids': missing,
        'n_failed': len(failed), 'failed': failed,
        'n_empty_no_shift': len(empty),
        'n_label_set_candidates': stats['n_label_set_candidates'],
        'n_logical_entities': stats['n_logical_entities'],
        'n_selected_residue_rows': stats['n_selected_residue_rows'],
        'n_observed_only_quarantine': len(all_quarantine),
        'ok': (len(missing) == 0 and len(failed) == 0),
    }
    _atomic_write(os.path.join(out_dir, 'fetch_manifest.json'),
                  lambda f: json.dump(manifest, f, indent=1))
    print(f'  entity fetch: {stats["n_logical_entities"]:,} logical entities, '
          f'{stats["n_selected_residue_rows"]:,} residue rows, '
          f'{len(all_quarantine):,} observed-only quarantined')
    return stats, manifest


def _load_api_ids(args, snapshot_dir):
    """Resolve the API id set: fetch it (--fetch) or load a saved copy (offline)."""
    if args.api_ids_file:
        with open(args.api_ids_file) as f:
            ids = json.load(f)
        return [str(x) for x in ids]
    saved = os.path.join(snapshot_dir, 'list_entries.json')
    if os.path.isfile(saved):
        with open(saved) as f:
            return [str(x) for x in json.load(f)]
    raise SystemExit(
        f'ERROR: no API id list. Run with --fetch to download it, or point '
        f'--api-ids-file at a saved list (looked for {saved}).')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True,
                    help='data dir for chemical_shifts.csv + fetch_manifest.json')
    ap.add_argument('--raw-root', default=None,
                    help='where dated raw snapshots live (default: <out>/raw)')
    ap.add_argument('--snapshot', default=None,
                    help='explicit snapshot dir to parse (default: latest under raw-root)')
    ap.add_argument('--fetch', action='store_true',
                    help='hit the network: rsync the snapshot AND fetch the API id list')
    ap.add_argument('--api-ids-file', default=None,
                    help='saved API id list json for offline parse (default: '
                         '<snapshot>/list_entries.json)')
    ap.add_argument('--bmrb-ids', nargs='+', default=None,
                    help='fetch only these entries (small online build)')
    ap.add_argument('--date', default=None,
                    help='snapshot date stamp YYYYMMDD (default: today, UTC)')
    ap.add_argument('--link-dest', default=None,
                    help='prior snapshot to hard-link from (default: newest under raw-root)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the rsync command without running it')
    ap.add_argument('--coverage-floor', type=float, default=0.80,
                    help='representative-rule residue coverage floor')
    ap.add_argument('--limit-bmrbs', default=None,
                    help='file with one BMRB id per line; restrict the parse to '
                         'this subset (smoke build)')
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    raw_root = os.path.abspath(args.raw_root) if args.raw_root else os.path.join(out, 'raw')

    if args.fetch:
        if args.bmrb_ids:
            print('[1/3] download selected raw _3.str entries')
            api_ids = [str(value) for value in args.bmrb_ids]
            snapshot = fetch_selected_snapshot(raw_root, api_ids, when=args.date)
            print('[2/3] use selected entry-id list')
        else:
            print('[1/3] rsync raw _3.str snapshot')
            snapshot = rsync_snapshot(raw_root, when=args.date,
                                      link_dest=args.link_dest, dry_run=args.dry_run)
            print('[2/3] fetch API entry-id list')
            api_ids = list_api_entry_ids()
        with open(os.path.join(snapshot, 'list_entries.json'), 'w') as f:
            json.dump(api_ids, f)
    else:
        if args.snapshot:
            snapshot = args.snapshot
        elif args.date:
            snapshot = _dated_snapshot_dir(raw_root, args.date)
        else:
            snapshot = _latest_prior_snapshot(raw_root, exclude='')
        if not snapshot or not os.path.isdir(snapshot):
            raise SystemExit(f'ERROR: no snapshot to parse under {raw_root}. '
                             f'Run with --fetch first or pass --snapshot.')
        print(f'[1/3] using existing snapshot {snapshot}')
        print('[2/3] loading saved API entry-id list')
        api_ids = _load_api_ids(args, snapshot)
    if args.limit_bmrbs:
        with open(args.limit_bmrbs) as fh:
            keep = {line.strip() for line in fh if line.strip()
                    and not line.startswith('#')}
        api_ids = [b for b in api_ids if str(b) in keep]
        # include ids present on disk even if absent from the saved API list (smoke subsets pin ids directly).
        on_disk = {b for b in keep
                   if os.path.isfile(entry_star_path(snapshot, b))}
        api_ids = sorted(set(api_ids) | on_disk, key=lambda s: (len(str(s)), str(s)))
        print(f'  --limit-bmrbs: restricted to {len(api_ids):,} ids')

    print(f'  {len(api_ids):,} API ids')

    print('[3/3] entity-mode parse -> representative selection + entity tables')
    stats, manifest = build_entity_tables_from_snapshot(
        snapshot, api_ids, out, coverage_floor=args.coverage_floor)
    if not manifest['ok']:
        print(f"\nERROR (fail-closed): missing_files={manifest['n_missing_files']} "
              f"failed={manifest['n_failed']}. entity tables incomplete.")
        sys.exit(1)
    print(f"\nOK (entity): {stats['n_logical_entities']:,} logical entities "
          f"-> {os.path.join(out, 'chemical_shifts.csv')}")


if __name__ == '__main__':
    main()
