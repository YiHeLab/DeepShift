"""CamShift adapter (PLUMED CS2BACKBONE).

CamShift is an empirical forward model over the query structure with no shift database or
homology transfer, so the arm is structure-only. Hydrogens come from REDUCE (every
deposited H stripped, then rebuilt), so all structure sources reach CamShift alike.
CS2BACKBONE needs contiguous 1..N residue numbering, so the template is renumbered and the
index-to-author-number map is carried back to key the predictions. It reads its parameter
file only from a ``data/`` directory beside the template, hence the per-job layout below.

Reads ``PLUMED_BIN`` (the ``plumed`` executable, PLUMED 2.10 with the isdb module) and
``CAMSHIFT_DB`` (``camshift.db``).
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import hydrogenate, resname_map, tool_path

ATOM_MAP = {'ca': 'CA', 'cb': 'CB', 'co': 'C', 'ha': 'HA', 'hn': 'H', 'nh': 'N'}


def renumber_protein(src, dst):
    """Contiguous 1..N template for CS2BACKBONE: (n_atoms, {index: (resname, author number)})."""
    serial = 0
    residues = {}
    residue_numbers = {}
    seen_atoms = set()
    with open(src) as fin, open(dst, 'w') as out:
        for line in fin:
            if line[:6].strip() != 'ATOM':
                continue
            if line[16].strip() not in ('', 'A', '1'):
                continue
            try:
                authnum = int(line[22:26])
            except ValueError:
                continue
            original_key = (line[21], line[22:27])
            atom_key = (original_key, line[12:16].strip())
            if atom_key in seen_atoms:
                continue
            seen_atoms.add(atom_key)
            resid = residue_numbers.setdefault(original_key, len(residue_numbers) + 1)
            serial += 1
            line = line[:6] + f'{serial:5d}' + line[11:22] + f'{resid:4d} ' + line[27:]
            out.write(line)
            residues[resid] = (line[17:20].strip().upper(), authnum)
        out.write('TER\nEND\n')
    return serial, residues


def predict(tool, chain_pdb, chain, bmrb_id, exclude_bmrbs='', exclude_uniprots='',
            identity_cutoff=0.90, workdir=None, hitlog=None):
    plumed = tool_path('PLUMED_BIN')
    camshift_db = tool_path('CAMSHIFT_DB')
    workdir = Path(workdir)
    (workdir / 'data').mkdir(parents=True, exist_ok=True)
    reduced = workdir / 'reduced.pdb'
    if not hydrogenate(chain_pdb, reduced):
        return {}, {'mode': 'error', 'error': 'reduce_no_output'}
    template = workdir / 'data' / 'template.pdb'
    natoms, residues = renumber_protein(reduced, template)
    if not natoms:
        return {}, {'mode': 'error', 'error': 'renumber_empty'}
    shutil.copy2(camshift_db, workdir / 'data' / 'camshift.db')

    # CS2BACKBONE reads these files to decide which residue/nucleus cells to emit. NOEXP
    # keeps the sentinel 1.0 values out of the calculation; the first and last residue are
    # commented out because CamShift does not predict terminal residues.
    ordered = sorted(residues)
    for stem in ('CA', 'CB', 'C', 'HA', 'H', 'N'):
        with open(workdir / 'data' / (stem + 'shifts.dat'), 'w') as fh:
            for i, resid in enumerate(ordered):
                prefix = '#' if i in (0, len(ordered) - 1) else ''
                fh.write(f'{prefix}{resid} 1.0\n')

    dat = workdir / 'plumed.dat'
    dat.write_text(
        f'cs: CS2BACKBONE ATOMS=1-{natoms} DATADIR=data TEMPLATE=template.pdb '
        'NEIGH_FREQ=1 NOPBC NOEXP\n'
        'PRINT ARG=cs.* STRIDE=1 FILE=colvar FMT=%14.7f\n')
    run = subprocess.run([plumed, 'driver', '--mf_pdb', str(template), '--plumed', str(dat),
                          '--trajectory-stride', '1'], cwd=str(workdir),
                         capture_output=True, text=True, timeout=1800, check=False)
    colvar = workdir / 'colvar'
    if run.returncode or not colvar.exists():
        return {}, {'mode': 'structure_only', 'excluded': [],
                    '_plumed_stderr': (run.stdout + '\n' + run.stderr)[-400:]}
    lines = colvar.read_text().splitlines()
    if len(lines) < 2:
        return {}, {'mode': 'structure_only', 'excluded': [], '_error': 'no_components'}

    # The header is '#! FIELDS time <component...>'; the data row is '<time> <value...>'.
    names = lines[0].split()[3:]
    values = [float(x) for x in lines[1].split()[1:]]
    resnames = resname_map(chain_pdb)
    preds = {}
    for name, value in zip(names, values, strict=False):
        parts = name.split('.', 1)[-1].split('-')
        if len(parts) < 3 or parts[0] not in ATOM_MAP:
            continue
        try:
            idx = int(parts[-1])
        except ValueError:
            continue
        if idx not in residues:
            continue
        authnum = str(residues[idx][1])
        preds.setdefault(authnum, {'resname': resnames.get(authnum, residues[idx][0])})[
            ATOM_MAP[parts[0]]] = value
    return preds, {'mode': 'structure_only', 'excluded': []}
