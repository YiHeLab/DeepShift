"""UCBShift 1.0 and 2.0 adapter (CSpred).

Arms, all on the single-chain PDB the driver extracts:

  ucbshift_x               UCBShift-X only (structure/ML, no homology database)
  ucbshift_full_leaky      X plus Y (homology transfer), nothing blocked
  ucbshift_full_sameprot   X plus Y, every deposition of the query's protein blocked
  ucbshift_full_selfexcl   X plus Y, the same list blocked and every reference at >=90 %
                           identity dropped
  ucbshift2_sameprot       UCBShift 2.0 (side chains), same-protein list blocked
  ucbshift2_selfexcl       UCBShift 2.0, same list plus the 90 % identity cutoff

CSpred needs explicit hydrogens and will not add any, so the chain goes through REDUCE
(``-Trim`` then ``-build``, residue numbers preserved). The self-retrieval blocking is the
``UCBY_EXCLUDE_BMRBS`` / ``UCBY_SELF_IDENT_CUTOFF`` / ``UCBY_HITLOG`` hooks added by
``patches/ucbshift.patch`` and ``patches/ucbshift2.patch``. The hit log has one line per
BLAST candidate: target, BMRB id, score, matched length, target length, identity, and the
decision ``keep_eligible``, ``drop_bmrb`` or ``drop_identity``. Version 1.0 writes the
backbone nuclei H, HA, C, CA, CB, N; version 2.0 writes 47 side-chain atom names, mapped
to the benchmark's 64 nuclei (methyl pseudoatoms become the first enumerated proton).

Reads ``UCBSHIFT_HOME`` (patched UCBShift 1.0 checkout), ``UCBSHIFT2_HOME`` (patched
UCBShift 2.0 checkout), ``UCBSHIFT2_MODELS`` (its downloaded model directory) and
``UCBSHIFT_PYTHON`` (the interpreter of the UCBShift environment).
"""
import contextlib
import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import hydrogenate, tool_path

BACKBONE = ['H', 'HA', 'C', 'CA', 'CB', 'N']
THREE = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
         'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}

# The benchmark's 64 nuclei.
NUCLEI = frozenset({
    'C', 'CA', 'CB', 'CD1', 'CD2', 'CD', 'CE1', 'CE2', 'CE3', 'CE', 'CG1', 'CG2', 'CG', 'CH2',
    'CZ2', 'CZ3', 'CZ', 'H', 'HA2', 'HA3', 'HA', 'HB1', 'HB2', 'HB3', 'HB', 'HD11', 'HD12',
    'HD13', 'HD1', 'HD21', 'HD22', 'HD23', 'HD2', 'HD3', 'HE1', 'HE21', 'HE22', 'HE2', 'HE3',
    'HE', 'HG11', 'HG12', 'HG13', 'HG1', 'HG21', 'HG22', 'HG23', 'HG2', 'HG3', 'HG', 'HH11',
    'HH12', 'HH21', 'HH22', 'HH2', 'HH', 'HZ2', 'HZ3', 'HZ', 'N', 'ND2', 'NE1', 'NE2', 'NE',
})

# UCBShift 2.0 emits methyl pseudoatoms where the benchmark enumerates the three equivalent
# protons, and the same name means different atoms in different residues. Names are
# resolved per residue first; whatever they do not cover must be a benchmark nucleus.
METHYL = {'ALA': {'HB': 'HB1'}, 'VAL': {'HG1': 'HG11', 'HG2': 'HG21'}, 'THR': {'HG2': 'HG21'},
          'ILE': {'HG2': 'HG21', 'HD1': 'HD11'}, 'LEU': {'HD1': 'HD11', 'HD2': 'HD21'},
          'MET': {'HE': 'HE1'}}
SINGLE = {'VAL': {'HB': 'HB'}, 'ILE': {'HB': 'HB'}, 'THR': {'HB': 'HB'}, 'LEU': {'HG': 'HG'}}
BARE = {'LYS': {'HG': 'HG2', 'HD': 'HD2', 'HE': 'HE2'}, 'ARG': {'HG': 'HG2', 'HD': 'HD2'},
        'PRO': {'HG': 'HG2', 'HD': 'HD2'}, 'GLN': {'HG': 'HG2'}, 'GLU': {'HG': 'HG2'}}
DIRECT = frozenset({
    'C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CG', 'CG1', 'CG2', 'CZ', 'H',
    'HA', 'HB2', 'HB3', 'HD2', 'HD21', 'HD22', 'HD3', 'HE1', 'HE2', 'HE3', 'HE21', 'HE22',
    'HG2', 'HG3', 'HG12', 'HG13', 'HZ', 'N', 'ND2', 'NE2', 'HD1',
})


def ucb_to_nucleus(atom, resname):
    """The benchmark nucleus of a UCBShift 2.0 atom name on a residue, or None."""
    atom = atom.upper()
    for table in (METHYL, SINGLE, BARE):
        if atom in table.get(resname, {}):
            return table[resname][atom]
    if atom == 'HD1' and resname in ('LEU', 'ILE'):
        return 'HD11'
    if atom in DIRECT:
        return atom
    return atom if atom in NUCLEI else None


def parse_backbone(out_csv, suffix):
    """CSpred CSV -> {residue number: {resname, atom: ppm}} for the ``suffix`` columns."""
    if not Path(out_csv).exists():
        return {}
    preds = {}
    with open(out_csv) as fh:
        for row in csv.DictReader(fh):
            number = row.get('RESNUM')
            if number is None:
                continue
            cell = {'resname': row.get('RESNAME', '')}
            for atom in BACKBONE:
                value = row.get(atom + suffix, '')
                if value not in ('', 'nan', 'NaN', None):
                    with contextlib.suppress(ValueError):
                        cell[atom] = float(value)
            if len(cell) > 1:
                preds[str(int(float(number)))] = cell
    return preds


def parse_sidechain(out_csv):
    """UCBShift 2.0 CSV -> {residue number: {resname, nucleus: ppm}}."""
    if not Path(out_csv).exists():
        return {}
    with open(out_csv) as fh:
        rows = list(csv.DictReader(fh))
    atoms = sorted({c[:-len('_UCBShift')] for c in (rows[0] if rows else {})
                    if c.endswith('_UCBShift')})
    preds = {}
    for row in rows:
        resname = (row.get('RESNAME') or '').strip().upper()
        number = (row.get('RESNUM') or '').strip()
        if resname not in THREE or not number:
            continue
        cell = {'resname': resname}
        for atom in atoms:
            raw = row.get(f'{atom}_UCBShift', '')
            if raw in ('', None, 'nan', 'NaN'):
                continue
            nucleus = ucb_to_nucleus(atom, resname)
            if nucleus is None:
                continue
            try:
                cell[nucleus] = float(raw)
            except ValueError:
                continue
        if len(cell) > 1:
            preds[str(int(float(number)))] = cell
    return preds


def hitlog_stats(hitlog):
    """(highest identity among kept references, references dropped) from a hit log."""
    max_kept, n_dropped = None, 0
    if hitlog and Path(hitlog).exists():
        with open(hitlog) as fh:
            for line in fh:
                fields = line.rstrip('\n').split('\t')
                if len(fields) < 7:
                    continue
                if fields[6] in ('drop_bmrb', 'drop_identity'):
                    n_dropped += 1
                elif fields[6] == 'keep_eligible':
                    identity = float(fields[5])
                    max_kept = identity if max_kept is None else max(max_kept, identity)
    return max_kept, n_dropped


def run_cspred(tree, prepared, out_csv, workdir, *, shiftx_only=False, models=None,
               exclude_bmrbs='', identity_cutoff=None, hitlog=None):
    """CSpred on a hydrogenated PDB, with the exclusion hooks set from the arguments."""
    python = tool_path('UCBSHIFT_PYTHON')
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join([str(Path(python).parent),
                                   f'{tree}/bins/ncbi-blast-2.9.0+/bin',
                                   f'{tree}/bins/mTM-align', env.get('PATH', '')])
    env['PYTHONPATH'] = str(tree)
    env['CUDA_VISIBLE_DEVICES'] = ''
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[var] = '1'
    env['TMPDIR'] = str(workdir)
    env['UCBY_EXCLUDE_BMRBS'] = exclude_bmrbs or ''
    env.pop('UCBY_SELF_IDENT_CUTOFF', None)
    if identity_cutoff is not None:
        env['UCBY_SELF_IDENT_CUTOFF'] = str(identity_cutoff)
    env.pop('UCBY_HITLOG', None)
    if hitlog:
        Path(hitlog).parent.mkdir(parents=True, exist_ok=True)
        Path(hitlog).unlink(missing_ok=True)
        env['UCBY_HITLOG'] = str(hitlog)
    cmd = [python, f'{tree}/CSpred.py', str(prepared), '--output', str(out_csv), '--pH', '5']
    if shiftx_only:
        cmd.append('--shiftx_only')
    if models:
        cmd += ['--models', str(models) + '/']
    subprocess.run(cmd, cwd=str(workdir), env=env, capture_output=True, timeout=7200,
                   text=True, check=False)


def predict(tool, chain_pdb, chain, bmrb_id, exclude_bmrbs='', exclude_uniprots='',
            identity_cutoff=0.90, workdir=None, hitlog=None):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # CSpred derives intermediate file names from the input stem inside a shared
    # working directory, so every job gets its own directory and a unique stem.
    prepared = workdir / f'{workdir.name}.pdb'
    if not hydrogenate(chain_pdb, prepared):
        return {}, {'mode': 'error', 'error': 'reduce_no_output'}
    out_csv = workdir / 'ucbshift.csv'
    second = tool.startswith('ucbshift2')
    tree = tool_path('UCBSHIFT2_HOME' if second else 'UCBSHIFT_HOME')
    models = tool_path('UCBSHIFT2_MODELS') if second else None

    if tool == 'ucbshift_x':
        run_cspred(tree, prepared, out_csv, workdir, shiftx_only=True)
        return parse_backbone(out_csv, '_X'), {
            'mode': 'not_applicable', 'fired': False, 'max_kept_ref_identity': None,
            'identity_cutoff': None}
    if tool == 'ucbshift_full_leaky':
        run_cspred(tree, prepared, out_csv, workdir, hitlog=hitlog)
        max_kept, _ = hitlog_stats(hitlog)
        return parse_backbone(out_csv, '_UCBShift'), {
            'mode': 'none_leaky', 'fired': False, 'max_kept_ref_identity': max_kept,
            'identity_cutoff': None, 'n_refs_dropped_self_or_homolog': 0}
    if tool in ('ucbshift_full_sameprot', 'ucbshift_full_selfexcl',
                'ucbshift2_sameprot', 'ucbshift2_selfexcl'):
        cutoff = identity_cutoff if tool.endswith('_selfexcl') else None
        run_cspred(tree, prepared, out_csv, workdir, models=models,
                   exclude_bmrbs=exclude_bmrbs, identity_cutoff=cutoff, hitlog=hitlog)
        preds = parse_sidechain(out_csv) if second else parse_backbone(out_csv, '_UCBShift')
        max_kept, n_dropped = hitlog_stats(hitlog)
        return preds, {
            'mode': 'bmrb_list+identity_cutoff' if cutoff else 'same_protein_bmrb_uniprot',
            'excluded_bmrbs': exclude_bmrbs.split(',') if exclude_bmrbs else [],
            'identity_cutoff': cutoff, 'fired': n_dropped > 0,
            'n_refs_dropped_self_or_homolog': n_dropped, 'max_kept_ref_identity': max_kept}
    raise ValueError(f'unknown ucbshift tool {tool}')
