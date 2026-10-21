"""Fold-6 competitor benchmark: the mapping from predictions to comparable cells.

The authoritative translation from the run's fold-6 prediction dumps to
(BMRB, source, structure residue number, nucleus) cells, plus per-structure task
construction and self-exclusion. Every competitor is scored on the cells this
module names, so that a comparison is cell-for-cell and never cohort-for-cohort.

The scored cells are the ones the ensemble dumps declare observed: the exclusion fence
was applied when those predictions were written.

Roots come from :mod:`benchmarks.config`; nothing here knows a filesystem.
"""
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from deepshift import evaluate

from . import config

BUILD = config.current()
RUN = BUILD.run
CACHE = BUILD.cache
SD = BUILD.structural
AD = BUILD.atom
DATA = BUILD.data
MEMBERS = BUILD.members
FOLD_MAP = BUILD.fold_map
FLAGS = BUILD.flags
OUT_ROOT = BUILD.out_root
PDB_DIRS = BUILD.pdb_dirs
UNIPROT_LEDGER = BUILD.uniprot_ledger
SIFTS = BUILD.sifts
REFDB_SHIFTS = BUILD.refdb_shifts
EDGE_FILES = list(config.EDGE_FILES)


# ----------------------- predictions + cell mapping -----------------------

def fold6_ensemble(paths=None):
    """The ppm ensemble of one arm's five fold-6 member dumps, over this cache's rows.

    The dumps are read through the package reader, so the benchmark scores the schema
    ``deepshift evaluate --save-npz`` writes and nothing else. Row identity is checked, not
    assumed: the ensemble must carry this cache's eligible rows, in order.
    """
    paths = list(paths or MEMBERS)
    ensemble = evaluate.ensemble([evaluate.read_predictions(path) for path in paths])
    _, rows = build_row_index()
    expected = np.asarray([r['row_key'] for r in rows])
    if not np.array_equal(ensemble.row_key, expected):
        raise SystemExit(
            f"{len(expected)} eligible rows in {CACHE}, {ensemble.n_rows} in the "
            f"prediction dumps, and their row identities differ; the dumps were not "
            f"written from this cache fold")
    return ensemble


@lru_cache(maxsize=1)
def load_pred_npz():
    """The fold-6 ensemble as the arrays the scorers read, all in ppm."""
    ensemble = fold6_ensemble()
    return {'true_resolved': ensemble.resolved_ppm, 'pred': ensemble.pred_ppm,
            'mask': ensemble.effective_mask, 'true_raw': ensemble.true_ppm,
            'amb': ensemble.ambiguity, 'aa': ensemble.residue_code,
            'cols': list(ensemble.shift_cols)}


def g_order():
    acnt = np.load(AD / 'atom_count.npy')
    smask = np.load(SD / 'shift_mask.npy')
    return np.where((acnt > 0) & smask.any(1))[0]


def load_idmaps():
    with open(SD / 'id_maps.json') as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def build_row_index():
    idm = load_idmaps()
    go = g_order()
    with open(SD / 'global_to_struct_resid.json') as fh:
        g2sr = json.load(fh)
    rows = []
    for g in go:
        gs = str(int(g))
        materialization_id = idm['materialization_id'][gs]
        label_residue_id = idm['label_residue_id'][gs]
        rows.append({
            'g': int(g),
            'bmrb': str(idm['bmrb_id'][gs]),
            'source': idm['source'][gs],
            'materialization_id': materialization_id,
            'label_residue_id': label_residue_id,
            # The identity a prediction dump carries for the same row.
            'row_key': f"{materialization_id}|{label_residue_id}",
            'label_set_id': idm['label_set_id'][gs],
            'struct_resnum': str(g2sr[gs]),
            'resname': idm['comp_id'][gs],
            'accession': idm['accession'][gs],
            'chain': idm['chain'][gs],
            'model_index': str(idm.get('model_index', {}).get(gs, '0')),
            'entity_id': idm.get('entity_id', {}).get(gs, '1'),
            'shift_list_id': idm.get('shift_list_id', {}).get(gs, '1'),
            'entity_assembly_id': idm.get('entity_assembly_id', {}).get(gs, '1'),
        })
    return go, rows


def structure_id(r):
    return "b{bmrb}_{source}_l{shift_list_id}_a{entity_assembly_id}_e{entity_id}".format(**r)


def resolve_pdb(source, pdb_path):
    bn = Path(pdb_path).name
    for _s, d in PDB_DIRS.items():
        if (d / bn).exists():
            return str(d / bn)
    return str(PDB_DIRS[source] / bn)


def load_build_logs():
    bl = {}
    for src in ('af', 'nmr', 'xtal'):
        with open(DATA / f'structure_build_log_{src}.csv') as fh:
            for r in csv.DictReader(fh):
                bl[r['materialization_id']] = r
    return bl


# ----------------------- self-exclusion (this run's authorities) -----------------------

def _load_uniprot_ledger():
    """BMRB -> {UniProt accession}, from the run's consolidated candidate ledger.

    Every final-source candidate is kept, so a deposition the sources disagree about
    excludes references to either accession, not only to the one that won.
    """
    mapping = {}
    if not UNIPROT_LEDGER.exists():
        return mapping
    with open(UNIPROT_LEDGER) as fh:
        for row in csv.DictReader(fh):
            if str(row['source_final']).strip().lower() == 'true':
                mapping.setdefault(str(row['bmrb_id']), set()).add(row['accession'])
    return mapping


def _load_pdb_uniprot():
    m = {}
    if not SIFTS.exists():
        return m
    with open(SIFTS) as fh:
        reader = csv.DictReader(l for l in fh if not l.startswith('#'))
        for row in reader:
            m.setdefault((row['PDB'].upper(), row['CHAIN']), set()).add(row['SP_PRIMARY'])
    return m


def _load_edge_linkage():
    """BMRB -> set(BMRBs sharing any uniprot/same_bmrb edge_group) in this run.

    A component links depositions of one protein even when they carry different
    accessions, which a per-deposition map cannot express.
    """
    link = {}
    for ef in EDGE_FILES:
        p = DATA / ef
        if not p.exists():
            continue
        groups = {}
        with open(p) as fh:
            for r in csv.DictReader(fh):
                groups.setdefault(r['edge_group'], set()).add(str(r['bmrb_id']))
        for members in groups.values():
            for b in members:
                link.setdefault(b, set()).update(members)
    return link


def _refdb_bmrbs_by_uniprot(pdb_uniprot):
    """UniProt -> {refDB BMRB} (for refDB entries not in our data)."""
    rev = {}
    for f in REFDB_SHIFTS.glob('*.csv'):
        stem = f.stem
        if '.' not in stem:
            continue
        bmrb, pdbchain = stem.split('.', 1)
        pdb = pdbchain[:4].upper()
        chain = pdbchain[4:5] if len(pdbchain) > 4 else '_'
        for u in pdb_uniprot.get((pdb, chain), set()) | pdb_uniprot.get((pdb, '_'), set()):
            rev.setdefault(u, set()).add(bmrb)
    return rev


class SelfExcluder:
    def __init__(self):
        self.frozen = _load_uniprot_ledger()
        self.pdb_uniprot = _load_pdb_uniprot()
        self.link = _load_edge_linkage()
        self.refdb_rev = _refdb_bmrbs_by_uniprot(self.pdb_uniprot)

    def query_uniprots(self, bmrb, source, accession, chain):
        u = set()
        if source == 'af' and accession:
            u.add(accession)
        u |= self.frozen.get(str(bmrb), set())
        if source in ('nmr', 'xtal') and accession:
            u |= self.pdb_uniprot.get((accession.upper(), chain), set())
            u |= self.pdb_uniprot.get((accession.upper(), '_'), set())
        # UniProts of every linked BMRB (captures multi-UniProt components)
        for lb in self.link.get(str(bmrb), set()):
            u |= self.frozen.get(lb, set())
        return u

    def exclude_bmrbs(self, bmrb, source, accession, chain):
        excl = {str(bmrb)}
        # 1) this run's edge linkage (same-UniProt + same-BMRB components)
        excl |= self.link.get(str(bmrb), set())
        # 2) refDB entries sharing the query's UniProt(s) (for refs not in our data)
        u = self.query_uniprots(bmrb, source, accession, chain)
        for up in u:
            excl |= self.refdb_rev.get(up, set())
        return sorted(excl, key=lambda x: (len(x), x)), sorted(u)


def build_tasks():
    _go, rows = build_row_index()
    bl = load_build_logs()
    exc = SelfExcluder()
    tasks = {}
    for r in rows:
        mid = r['materialization_id']
        if mid in tasks:
            tasks[mid]['n_cells'] += 1
            continue
        blr = bl[mid]
        excl, unis = exc.exclude_bmrbs(r['bmrb'], r['source'], r['accession'], blr['chain_id'])
        tasks[mid] = {
            'structure_id': structure_id(r), 'materialization_id': mid,
            'label_set_id': r['label_set_id'],
            'bmrb_id': r['bmrb'], 'source': r['source'], 'accession': r['accession'],
            'pdb_path': resolve_pdb(r['source'], blr['pdb_path']), 'chain': blr['chain_id'],
            'model_index': r['model_index'], 'uniprots': ','.join(unis),
            'exclude_bmrbs': ','.join(excl),
            'n_cells': 1}
    return tasks


TASK_COLUMNS = ['structure_id', 'materialization_id', 'label_set_id', 'bmrb_id', 'source',
                'accession', 'pdb_path', 'chain', 'model_index', 'uniprots',
                'exclude_bmrbs', 'n_cells']


def write_tasks(tasks, path=None):
    """The per-structure task table every scorer reads its cohort from."""
    path = Path(path) if path is not None else OUT_ROOT / 'fold6_tasks.tsv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=TASK_COLUMNS, delimiter='\t')
        w.writeheader()
        for t in sorted(tasks.values(), key=lambda x: (x['source'], int(x['bmrb_id']))):
            w.writerow(t)
    return path


if __name__ == '__main__':
    tasks = build_tasks()
    outp = write_tasks(tasks)
    from collections import Counter
    bysrc = Counter(t['source'] for t in tasks.values())
    n_labelsets = len({t['label_set_id'] for t in tasks.values()})
    print(f"wrote {outp}: {len(tasks)} tasks by_source={dict(bysrc)} "
          f"distinct_bmrb={len({t['bmrb_id'] for t in tasks.values()})} label_sets={n_labelsets}")
