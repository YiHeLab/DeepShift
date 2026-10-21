"""Filesystem roots for the data stages, each an environment variable with a default.

  SHIFT_DATA_ROOT          the build's data directory            default: data
  SHIFT_PDB_DIR            experimental PDB files                default: <data>/pdbs
  SHIFT_ALPHAFOLD_DIR      AlphaFold models                      default: <data>/alphafold
  SHIFT_BIG_STORAGE        caches and other large outputs        default: ./big
  SHIFT_DATA_ARCHIVE       the structure archive                 default: <big>/data_archive
  SHIFT_UCB_TEST_DIR       UCBShift-200 test PDBs                default: ./data/ucb_test_pdbs
  SHIFT_UCB_TEST_TARGETS   UCBShift-200 test targets, JSON       no default
  BLASTP_BIN               the blastp executable                 default: from PATH
  DEEPSHIFT_RUN            a finished run's outputs              no default

This module imports only the standard library so every stage can load it cheaply.
"""

import os
import shutil

# DataLoader workers talk over AF_UNIX sockets, whose path is capped at 108 characters;
# a long inherited TMPDIR overflows the cap and hangs any loader with workers.
if len(os.environ.get("TMPDIR", "")) > 60 or not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = "/tmp"


def _env(name, default):
    """The variable's value, or ``default`` when it is unset or empty."""
    val = os.environ.get(name)
    if val is None or val == '':
        return default
    return val


DATA_DIR = _env('SHIFT_DATA_ROOT', 'data')
PDB_DIR = _env('SHIFT_PDB_DIR', os.path.join(DATA_DIR, 'pdbs'))
ALPHAFOLD_DIR = _env('SHIFT_ALPHAFOLD_DIR', os.path.join(DATA_DIR, 'alphafold'))
BIG_STORAGE = _env('SHIFT_BIG_STORAGE', './big')
DATA_ARCHIVE = _env('SHIFT_DATA_ARCHIVE', os.path.join(BIG_STORAGE, 'data_archive'))

# The packaged UCBShift-200 exclusion list is used directly; the test directory is only
# read to regenerate that list from the source PDBs.
UCB_TEST_DIR = _env('SHIFT_UCB_TEST_DIR', './data/ucb_test_pdbs')

# The UCBShift-200 test targets are the exclusion authority. They are a third-party file
# with no canonical location, so there is no default to guess.
UCB_TEST_TARGETS = _env('SHIFT_UCB_TEST_TARGETS', '')

# MMseqs2 has its own resolver (sequence_clustering.resolve_mmseqs) that can provision a
# pinned copy; blastp cannot, so BLASTP_BIN or `data build --blastp-bin` names it.
BLASTP_BIN = _env('BLASTP_BIN', shutil.which('blastp') or 'blastp')

# One finished run: its members, prediction dumps and build data. Every producer in the
# repository resolves the run through this variable, and there is no default.
RUN_ROOT = _env('DEEPSHIFT_RUN', None)
