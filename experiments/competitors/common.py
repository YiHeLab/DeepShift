"""Helpers shared by the competitor drivers and adapters.

Tool locations are read from environment variables (``<TOOL>_HOME`` for an install
directory, ``<TOOL>_PYTHON`` for the interpreter of a tool's own environment,
``<TOOL>_BIN`` for one executable); a missing variable stops the run naming it.
"""
import os
import subprocess
from pathlib import Path


def tool_path(var, default=None):
    """The path in ``$var``, or ``default``; stops naming the variable when neither is set."""
    value = os.environ.get(var) or default
    if not value:
        raise SystemExit(f"{var} is not set (see experiments/competitors/README.md)")
    return value


def extract_chain(src, chain, dst, model_index=0):
    """Write one chain of the selected model to ``dst``, as the training cache reads it.

    ``model_index`` is the 0-based position of the model in parse order. Per residue, the
    alternate location with the highest summed occupancy is kept (ties go to the lower
    letter) together with every blank-altloc atom, and the altloc column is blanked.
    A single-model file is model 0.
    """
    model_index = int(model_index)
    with open(src) as fh:
        lines = fh.readlines()
    has_models = any(line[:6].strip() == 'MODEL' for line in lines)
    first = None
    cur_model = -1
    in_target = not has_models
    residues = {}
    order = []
    for line in lines:
        rec = line[:6].strip()
        if rec == 'MODEL':
            cur_model += 1
            in_target = cur_model == model_index
            continue
        if rec == 'ENDMDL':
            in_target = False
            continue
        if not in_target or rec not in ('ATOM', 'HETATM'):
            continue
        cur = line[21] if len(line) > 21 else ' '
        if chain in ('', '_', 'nan', None):
            if first is None:
                first = cur
            if cur != first:
                continue
        elif cur != chain:
            continue
        altloc = line[16:17].strip()
        atom_name = line[12:16].strip()
        icode = line[26:27]
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        try:
            occ = float(line[54:60])
        except ValueError:
            occ = 1.0
        key = (cur, resseq, icode)
        if key not in residues:
            residues[key] = []
            order.append(key)
        residues[key].append((altloc, atom_name, occ, line))
    out = []
    for key in order:
        records = residues[key]
        alt_occ = {}
        for altloc, _name, occ, _line in records:
            if altloc:
                alt_occ[altloc] = alt_occ.get(altloc, 0.0) + occ
        best = sorted(alt_occ, key=lambda a: (-alt_occ[a], a))[0] if alt_occ else None
        kept = {}
        for altloc, name, _occ, line in records:
            if altloc == '' or altloc == best:
                kept[name] = line[:16] + ' ' + line[17:]
        out.extend(kept.values())
    Path(dst).write_text(''.join(out) + 'TER\nEND\n')


def hydrogenate(chain_pdb, out_pdb):
    """Strip every deposited H, then rebuild them with REDUCE (residue numbers preserved).

    Returns True when REDUCE wrote a non-empty file. Every source therefore reaches a
    predictor that needs explicit H with REDUCE-placed hydrogens and nothing else.
    """
    reduce_bin = os.environ.get('REDUCE_BIN', 'reduce')
    trimmed = str(out_pdb) + '.trim'
    with open(trimmed, 'w') as fh:
        subprocess.run([reduce_bin, '-Trim', '-Quiet', str(chain_pdb)],
                       stdout=fh, stderr=subprocess.DEVNULL, check=False)
    with open(out_pdb, 'w') as fh:
        subprocess.run([reduce_bin, '-build', '-Quiet', trimmed],
                       stdout=fh, stderr=subprocess.DEVNULL, check=False)
    return Path(out_pdb).exists() and Path(out_pdb).stat().st_size > 0


def resname_map(chain_pdb):
    """Author residue number (str, no insertion code) -> 3-letter residue name, first seen."""
    names = {}
    with open(chain_pdb) as fh:
        for line in fh:
            if line[:6].strip() not in ('ATOM', 'HETATM'):
                continue
            try:
                rid = str(int(line[22:26]))
            except ValueError:
                continue
            names.setdefault(rid, line[17:20].strip())
    return names


def count_h(path):
    """(hydrogen atoms, atoms) from the element column of a PDB file."""
    h = n = 0
    with open(path) as fh:
        for line in fh:
            if line[:6].strip() in ('ATOM', 'HETATM'):
                n += 1
                if line[76:78].strip() in ('H', 'D'):
                    h += 1
    return h, n
