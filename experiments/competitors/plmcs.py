"""Run PLM-CS over the fold-6 task table: sequence in, six backbone shifts per residue out.

PLM-CS (Zhu et al., Digital Discovery 4:331-337, 2025) reads a sequence only: a frozen
ESM-2 650M embedding feeds a two-layer transformer with one regression head per nucleus
(CA, CB, C, HA, H, N). It reads no coordinates and retrieves no reference shifts, so there
is nothing to exclude; its training set is fixed inside the released checkpoints.

Each task's chain is extracted from its structure, the residues of its ATOM records give the
sequence (non-standard residues become ``X``), and the package's own ``predict_from_seq`` runs
once per distinct sequence with the ESM-2 backbone loaded once. The package pads the embedding
to 512 rows and builds its positional encoder for 512 positions, so a longer chain would be
silently truncated; such a chain is written as ``failed`` instead.

Writes ``$DEEPSHIFT_BENCH_OUT/plmcs/<structure_id>.json``. Run in the interpreter of the
environment that has ``plm-cs`` (the PyPI package, with its ``*-ckpt`` checkpoint packages)
and ``fair-esm`` (``$PLMCS_PYTHON``); ESM-2 weights are fetched into ``$TORCH_HOME``
(default ``~/.cache/torch``) on first use:

    $PLMCS_PYTHON experiments/competitors/plmcs.py
"""
import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from common import extract_chain

MAX_LEN = 512
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
# The package writes lowercase column names; the benchmark uses PDB atom names.
COLUMN_TO_ATOM = {'ca': 'CA', 'cb': 'CB', 'c': 'C', 'ha': 'HA', 'h': 'H', 'n': 'N'}
SELF_EXCLUSION = {
    'mode': 'not_applicable', 'excluded_bmrbs': [], 'identity_cutoff': None, 'fired': False,
    'max_kept_ref_identity': None, 'input_modality': 'sequence_only',
    'training_set': 'fixed inside the released checkpoints (RefDB with the SHIFTX test '
                    'proteins removed); cannot be filtered per query',
}


def chain_residues(chain_pdb):
    """[(author residue number + insertion code, residue name, one letter)] from ATOM records.

    HETATM records are skipped, so a modified residue deposited as HETATM leaves a gap.
    """
    residues, seen = [], set()
    with open(chain_pdb) as fh:
        for line in fh:
            if line[:4] != 'ATOM':
                continue
            key = (line[21], line[22:27])
            if key in seen:
                continue
            seen.add(key)
            resname = line[17:20].strip().upper()
            residues.append((line[22:26].strip() + line[26:27].strip(), resname,
                             THREE_TO_ONE.get(resname, 'X')))
    return residues


def predict_sequences(sequences):
    """{sequence: DataFrame of the six heads' outputs} with ESM-2 loaded once."""
    import esm
    import plm_cs.CS_predict as cs_predict

    held = {}
    load = esm.pretrained.esm2_t33_650M_UR50D

    def load_once():
        if 'model' not in held:
            held['model'] = load()
        return held['model']

    esm.pretrained.esm2_t33_650M_UR50D = load_once
    cs_predict.esm.pretrained.esm2_t33_650M_UR50D = load_once
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for position, sequence in enumerate(sequences, 1):
            csv_path = Path(td) / 'prediction.csv'
            try:
                cs_predict.predict_from_seq(sequence, str(csv_path))
                out[sequence] = pd.read_csv(csv_path)
            except Exception as error:  # reported per structure
                out[sequence] = f'{type(error).__name__}: {error}'
            if position % 25 == 0 or position == len(sequences):
                print(f'[{position}/{len(sequences)}] sequences', flush=True)
    return out


def main():
    bench_out = Path(os.environ.get('DEEPSHIFT_BENCH_OUT', ''))
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tasks', type=Path, default=bench_out / 'fold6_tasks.tsv',
                        help='fold-6 task table (default $DEEPSHIFT_BENCH_OUT/fold6_tasks.tsv)')
    parser.add_argument('--out', type=Path, default=bench_out / 'plmcs',
                        help='output directory (default $DEEPSHIFT_BENCH_OUT/plmcs)')
    parser.add_argument('--only', nargs='+', default=None, metavar='STRUCTURE_ID',
                        help='run just these structure ids')
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    with open(args.tasks) as stream:
        tasks = list(csv.DictReader(stream, delimiter='\t'))
    if args.only:
        tasks = [t for t in tasks if t['structure_id'] in set(args.only)]
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in tasks
             if not (args.out / f"{t['structure_id']}.json").exists()
             or not (args.out / f"{t['structure_id']}.json").stat().st_size]

    residues = {}
    with tempfile.TemporaryDirectory() as td:
        for task in tasks:
            chain_pdb = Path(td) / 'chain.pdb'
            extract_chain(task['pdb_path'], task['chain'], chain_pdb,
                          task.get('model_index') or 0)
            residues[task['structure_id']] = chain_residues(chain_pdb)
    sequences = sorted({''.join(r[2] for r in rs) for rs in residues.values()
                        if 0 < len(rs) <= MAX_LEN})
    print(f'{len(tasks)} structures, {len(sequences)} distinct sequences to predict', flush=True)
    frames = predict_sequences(sequences) if sequences else {}

    failed = 0
    for task in tasks:
        sid = task['structure_id']
        rs = residues[sid]
        sequence = ''.join(r[2] for r in rs)
        began = time.time()
        predictions, error = {}, None
        if not rs:
            error = 'no_ATOM_residues_in_chain'
        elif len(rs) > MAX_LEN:
            error = f'sequence_too_long: {len(rs)} residues > {MAX_LEN}'
        elif isinstance(frames[sequence], str):
            error = f'plmcs_no_prediction: {frames[sequence][:200]}'
        elif len(frames[sequence]) != len(rs):
            error = f'length_mismatch: {len(frames[sequence])} rows for {len(rs)} residues'
        else:
            for (rid, resname, _), row in zip(rs, frames[sequence].to_dict('records')):
                cell = {'resname': resname}
                for column, atom in COLUMN_TO_ATOM.items():
                    value = row.get(column)
                    if value is not None and math.isfinite(float(value)):
                        cell[atom] = float(value)
                if len(cell) > 1:
                    predictions[rid] = cell
            if not predictions:
                error = 'no_finite_values'
        record = {'status': 'ok' if predictions else 'failed',
                  'runtime_sec': round(time.time() - began, 3),
                  'self_exclusion': {**SELF_EXCLUSION, 'sequence_length': len(rs)},
                  'predictions': predictions}
        if error:
            record['error'] = error
            failed += 1
        partial = args.out / f'{sid}.json.part'
        partial.write_text(json.dumps(record, indent=1))
        partial.replace(args.out / f'{sid}.json')
    print(f'wrote {len(tasks) - failed} predictions, {failed} failed (see "error")', flush=True)


if __name__ == '__main__':
    main()
