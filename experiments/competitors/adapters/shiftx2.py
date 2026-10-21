"""SHIFTX2 adapter.

SHIFTX2 combines SHIFTX+ (structure, machine learning) with SHIFTY+ (homology transfer from
the RefDB library). It runs in a Python 2.7 environment. SHIFTX strips and regenerates
hydrogens itself, so the chain is fed without REDUCE; waters are dropped.

Arms:

  shiftx2_leaky     SHIFTY+ on, nothing blocked
  shiftx2_sameprot  SHIFTY+ on, every deposition of the query's protein blocked
  shiftx2_selfexcl  SHIFTY+ on, the same list blocked and every hit at >=90 % identity dropped
  shiftx2_noshifty  SHIFTX+ only (``-n``)

The blocking is the ``SHIFTY_EXCLUDE_BMRBS`` / ``SHIFTY_SELF_IDEN`` / ``SHIFTY_LOG_HIT``
hooks added to ``shifty3/shifty3.py`` by ``patches/shiftx2_shifty3.patch``. With
``SHIFTY_LOG_HIT`` set, SHIFTY+ writes one ``HITLOG<TAB>shifttype<TAB>entry<TAB>identity``
line to stderr per reference it uses; that stderr is the arm's hit log.

Reads ``SHIFTX2_HOME`` (the ``shiftx2-linux`` directory) and ``SHIFTX2_PYTHON`` (the Python
2.7 interpreter of the SHIFTX2 environment; its directory also supplies ``java``).
"""
import contextlib
import csv
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import tool_path

# SHIFTX2 atom names -> the benchmark's
ATOM_MAP = {'CO': 'C', 'HN': 'H'}

# Seconds. The largest AlphaFold models (1,200-1,700 residues) need hours; a run that
# times out writes no output.
TIMEOUT = 21600


def extract_without_waters(src, chain, dst):
    """Single chain, waters dropped."""
    first = None
    lines = []
    with open(src) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ('ATOM', 'HETATM', 'TER', 'ANISOU'):
                continue
            cur = line[21] if len(line) > 21 else ' '
            if chain in ('', '_', 'nan', None):
                if first is None and rec in ('ATOM', 'HETATM'):
                    first = cur
                keep = cur == first
            else:
                keep = cur == chain
            if keep and not (rec == 'HETATM' and line[17:20].strip() == 'HOH'):
                lines.append(line)
    Path(dst).write_text(''.join(lines) + 'TER\nEND\n')


def run_group(cmd, *, cwd, env, timeout):
    """subprocess.run with a timeout that kills what it started.

    SHIFTX2 is a shell/perl front end that spawns Java and ``caleffect`` children. Plain
    ``subprocess.run(timeout=...)`` kills only the immediate child, and the grandchildren
    keep running. The child therefore leads its own session and the whole process group is
    signalled on timeout, SIGTERM first and SIGKILL after a grace period.
    """
    with subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, start_new_session=True) as child:
        try:
            out, err = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_group(child.pid)
            out, err = child.communicate()
            raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err) from None
    return subprocess.CompletedProcess(cmd, child.returncode, out, err)


def kill_group(pid):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        if sig is signal.SIGTERM:
            time.sleep(5)


def parse_cs(cs_path):
    """SHIFTX2 CSV output -> {residue number: {resname, atom: ppm}}."""
    preds = {}
    lines = [line for line in Path(cs_path).read_text().splitlines() if line.strip()]
    if not lines:
        return preds
    header = [h.strip().upper() for h in lines[0].split(',')]
    if 'ATOMNAME' in header or ('NUM' in header and 'ATOMNAME' in ','.join(header)):
        for row in csv.DictReader(lines):
            row = {(k or '').strip().upper(): (v or '').strip() for k, v in row.items()}
            num = row.get('NUM') or row.get('RESNUM') or row.get('RES_NUM')
            atom = row.get('ATOMNAME') or row.get('ATOM')
            shift = row.get('SHIFT') or row.get('CS')
            res = row.get('RES') or row.get('RESNAME') or ''
            if num is None or atom is None or shift in (None, ''):
                continue
            try:
                num = str(int(float(num)))
                value = float(shift)
            except ValueError:
                continue
            preds.setdefault(num, {'resname': res})[ATOM_MAP.get(atom.upper(), atom.upper())] = value
        return preds
    # Tabular sectioned format: Num RES CA CB CO N H HA ...
    cols = None
    for line in lines:
        parts = line.split()
        if parts and parts[0].upper() in ('NUM', 'NUM.'):
            cols = [c.upper() for c in parts[1:]]
            continue
        if cols and parts and re.match(r'^\d+$', parts[0]):
            cell = {'resname': parts[1] if len(parts) > 1 else ''}
            for col, value in zip(cols, parts[2:]):
                with contextlib.suppress(ValueError):
                    cell[ATOM_MAP.get(col, col)] = float(value)
            preds[parts[0]] = cell
    return preds


def hitlog_stats(stderr_text):
    """(highest identity among used references as a fraction, number used)."""
    max_used, n = None, 0
    for line in stderr_text.splitlines():
        if not line.startswith('HITLOG'):
            continue
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        try:
            identity = float(parts[3])
        except ValueError:
            continue
        fraction = identity / 100.0 if identity > 1.5 else identity
        max_used = fraction if max_used is None else max(max_used, fraction)
        n += 1
    return max_used, n


def predict(tool, chain_pdb, chain, bmrb_id, exclude_bmrbs='', exclude_uniprots='',
            identity_cutoff=0.90, workdir=None, hitlog=None):
    home = tool_path('SHIFTX2_HOME')
    python = tool_path('SHIFTX2_PYTHON')
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pdb = workdir / 'query.pdb'
    extract_without_waters(chain_pdb, chain, pdb)
    tmp = workdir / 'sxtmp'
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ)
    env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)

    cmd = [python, 'shiftx2.py', '-i', str(pdb), '-z', str(tmp), '-f', 'CSV', '-a', 'ALL']
    selfexcl = {'mode': 'not_applicable', 'fired': False, 'max_kept_ref_identity': None,
                'identity_cutoff': None}
    blocked = exclude_bmrbs.split(',') if exclude_bmrbs else []
    if tool == 'shiftx2_noshifty':
        cmd.append('-n')
    elif tool == 'shiftx2_leaky':
        env['SHIFTY_LOG_HIT'] = '1'
        selfexcl = {'mode': 'none_leaky', 'fired': False, 'identity_cutoff': None}
    elif tool == 'shiftx2_sameprot':
        # SHIFTY_SELF_IDEN stays unset, so SHIFTY+ applies no identity filter: homologs of
        # other proteins stay usable at every identity.
        env['SHIFTY_EXCLUDE_BMRBS'] = exclude_bmrbs or ''
        env.pop('SHIFTY_SELF_IDEN', None)
        env['SHIFTY_LOG_HIT'] = '1'
        selfexcl = {'mode': 'same_protein_bmrb_uniprot', 'fired': None,
                    'excluded_bmrbs': blocked, 'identity_cutoff': None}
    elif tool == 'shiftx2_selfexcl':
        env['SHIFTY_EXCLUDE_BMRBS'] = exclude_bmrbs or ''
        env['SHIFTY_SELF_IDEN'] = str(identity_cutoff * 100.0)    # percent
        env['SHIFTY_LOG_HIT'] = '1'
        selfexcl = {'mode': 'bmrb_list+identity_cutoff', 'excluded_bmrbs': blocked,
                    'identity_cutoff': identity_cutoff}
    else:
        raise ValueError(f'unknown shiftx2 tool {tool}')

    proc = run_group(cmd, cwd=home, env=env, timeout=TIMEOUT)
    produced = Path(str(pdb) + '.cs')
    preds = parse_cs(produced) if produced.exists() and produced.stat().st_size else {}

    if tool != 'shiftx2_noshifty':
        max_used, n_used = hitlog_stats(proc.stderr)
        selfexcl.update(max_kept_ref_identity=max_used, n_refs_used=n_used)
        if hitlog:
            Path(hitlog).parent.mkdir(parents=True, exist_ok=True)
            Path(hitlog).write_text(proc.stderr)
    if not preds:
        selfexcl['_shiftx2_stderr_tail'] = proc.stderr[-400:]
    return preds, selfexcl
