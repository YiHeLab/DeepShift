"""Crystal PDBs live at ``<DATA_DIR>/crystal_pdbs/<pdb_id>.pdb`` (lowercased); resolved via :func:`crystal_pdb_path`, not built here (see discover -> materialize)."""
from __future__ import annotations

import os


def crystal_pdb_path(crystal_dir: str, pdb_id: str) -> str:
    return os.path.join(crystal_dir, f'{str(pdb_id).lower()}.pdb')
