"""The BMRB-to-UniProt mapping the fold stage reads.

The mapping is split across two JSON files under ``<data_dir>/alphafold``: the curated
baseline and the additions found when the dataset was assembled. Both are frozen inputs
of a build; nothing in the package writes them.
"""

import json
from pathlib import Path

MAPPING_FILENAMES = ("bmrb_uniprot_mapping.json", "new_uniprot_mappings.json")


def load_merged_mapping(search_dirs, required=False):
    """The union of the two mapping files under each of ``search_dirs``.

    The baseline wins over the additions and an earlier directory wins over a later one;
    empty accessions are dropped. With ``required`` the absence of both files everywhere
    is an error, because the fold stage cannot draw its shared-UniProt edges without them.
    """
    dirs = [Path(d) for d in search_dirs]
    merged = {}
    found_any = False
    for d in dirs:
        for fname in MAPPING_FILENAMES:
            p = d / fname
            if p.is_file():
                found_any = True
                with open(p) as f:
                    for k, v in json.load(f).items():
                        if v:
                            merged.setdefault(str(k), v)
    if not found_any and required:
        looked = "\n  ".join(str(d / n) for d in dirs for n in MAPPING_FILENAMES)
        raise FileNotFoundError(
            "BMRB->UniProt mapping not found. Expected bmrb_uniprot_mapping.json "
            "and/or new_uniprot_mappings.json at:\n  " + looked)
    return merged
