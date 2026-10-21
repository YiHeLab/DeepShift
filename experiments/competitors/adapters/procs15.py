"""ProCS15 adapter.

ProCS15 is a QM/DFT-parameterised forward model: it computes isotropic shieldings from the
query structure alone (no shift database, no homology), so the arm is structure-only. It
adds missing atoms and hydrogens itself and keeps deposited ones, so the chain PDB is fed
as extracted.

Shieldings become ppm with the fixed per-nucleus linear map ``delta = a * sigma + b`` in
``procs15_fixed_scale.json`` (fitted once on 1UBQ against BMRB 6457). Nuclei: N, CA, CB, C,
H (amide), HA, with glycine HA2/HA3 kept apart; a shielding of exactly 0 is the predictor's
terminal/unpredicted sentinel and is dropped.

The predictor spends about 18 s loading a 17 GB parameter table and about 0.1 s per
structure, so :func:`predict_many` runs many structures in one call; its output is byte-identical
to one call per structure. It needs about 27 GB of RAM per process.

Reads ``PROCS15_BIN`` (``procs15_predictor`` of the PHAISTOS build) and ``PROCS15_DATA``
(the ``ProCSnumpyfiles/`` folder).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import resname_map, tool_path

SCALE = json.loads((Path(__file__).resolve().parent / 'procs15_fixed_scale.json').read_text())

# .procs atom name -> (output key, key in the scale file)
PROCS_MAP = {'CA': ('CA', 'CA'), 'CB': ('CB', 'CB'), 'C': ('C', 'C'), 'N': ('N', 'N'),
             'H': ('H', 'HN'), 'HA': ('HA', 'HA'), 'HA2': ('HA2', 'HA'), 'HA3': ('HA3', 'HA')}
STANDARD = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU',
            'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}


def index_to_authnum(pdb_path):
    """{1-based residue index in ProCS15's order: author number plus insertion code}.

    ProCS15 numbers the standard residues of the first chain 1..N in order of appearance.
    """
    order, seen, first_chain = [], set(), None
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith('ATOM') or line[17:20].strip() not in STANDARD:
                continue
            chain = line[21]
            if first_chain is None:
                first_chain = chain
            if chain != first_chain:
                break
            key = (chain, line[22:26].strip(), line[26].strip())
            if key not in seen:
                seen.add(key)
                order.append(key[1] + key[2])
    return {i + 1: number for i, number in enumerate(order)}


def parse_procs(procs_path, remap):
    out = {}
    with open(procs_path) as fh:
        for line in fh:
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 6:
                continue
            try:
                idx = int(fields[1])
                value = float(fields[5])
            except ValueError:
                continue
            atom = fields[3]
            if atom not in PROCS_MAP or value == 0.0:
                continue
            key, scale_key = PROCS_MAP[atom]
            a, b = SCALE[scale_key]
            out.setdefault(str(remap.get(idx, idx)), {})[key] = a * value + b
    return out


def _convert(procs_path, chain_pdb):
    raw = parse_procs(procs_path, index_to_authnum(chain_pdb))
    names = resname_map(chain_pdb)
    return {rid: {'resname': names.get(rid, ''), **atoms} for rid, atoms in raw.items()}


def predict_many(chains, workdir, timeout=7200):
    """Predictions for ``{structure_id: chain PDB}`` from batched predictor calls.

    A malformed structure aborts the whole predictor process. The predictor follows the
    argument order, so the first structure without a ``.procs`` file is the one it died on;
    it is dropped and the structures behind it run again. Returns ``{structure_id:
    predictions}`` for every structure the predictor handled.
    """
    binary = tool_path('PROCS15_BIN')
    data = tool_path('PROCS15_DATA')
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    local = {}
    for sid, src in chains.items():
        local[sid] = workdir / f'{sid}.pdb'
        shutil.copy(src, local[sid])
    done = {}
    pending = list(local)
    while pending:
        subprocess.run([binary, '--data-folder', data, '--pdb-files',
                        *[str(local[sid]) for sid in pending]],
                       cwd=str(workdir), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                       timeout=timeout, check=False)
        missing = [sid for sid in pending if not local[sid].with_suffix('.procs').exists()
                   or not local[sid].with_suffix('.procs').stat().st_size]
        for sid in pending:
            if sid not in missing:
                done[sid] = _convert(local[sid].with_suffix('.procs'), chains[sid])
        if not missing:
            break
        pending = missing[1:]
    return done


def predict(tool, chain_pdb, chain, bmrb_id, exclude_bmrbs='', exclude_uniprots='',
            identity_cutoff=0.90, workdir=None, hitlog=None):
    done = predict_many({'query': chain_pdb}, workdir, timeout=3600)
    return done.get('query', {}), {'mode': 'structure_only', 'excluded': []}
