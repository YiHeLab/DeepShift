"""Census-derived nonstandard-entity exclusions for training rows."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources

import numpy as np
from scipy.spatial import cKDTree

from ..domain import STANDARD_RESIDUES
from ..structure import parse_models
from . import structures

WATER_NAMES = {"HOH", "DOD", "WAT", "H2O"}
HEME_NAMES = {
    "HEM", "HEC", "HEA", "HEB", "HEO", "HEV", "HDD", "HAS", "DHE",
    "CP3", "MHM", "SRM", "VER", "HNI", "HES",
}
SUGAR_NAMES = {
    "NAG", "NDG", "MAN", "BMA", "GLC", "BGC", "GAL", "GLA", "FUC", "FUL",
    "XYS", "XYP", "ARA", "RIB", "FRU", "SIA", "SLB", "G6D", "MMA", "RAM",
    "TRE", "SUC", "MAL", "LBT", "LAC", "NGA", "A2G",
}
SOLVENT_NAMES = {
    "GOL", "EDO", "PEG", "PGE", "PG4", "P6G", "MPD", "DMS", "DTT", "BME",
    "MES", "MPO", "TRS", "BIS", "HEP", "CAP", "CAC", "ACT", "ACY", "FMT",
    "ACE", "EOH", "IPA", "MOH", "BU3", "BOG", "LDA", "SDS",
    "DDM", "LMT", "C8E", "C10", "C12", "C14", "C16", "DAO", "OLA", "OLC",
    "SO4", "SUL", "NO3", "CIT", "TAR", "MAL", "OXA", "IMD", "IOD", "SCN",
    "CL", "BR", "F", "NH4", "URE", "1PE", "2PE", "3PE", "DEP", "DMF",
}
NUCLEOTIDE_NAMES = {
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "CTP", "CDP", "CMP", "UTP",
    "UDP", "UMP", "TTP", "TMP", "NAD", "NAP", "NAI", "FAD", "FMN", "SAM",
    "SAH", "COA", "ACP", "PO4", "SEP", "TPO", "PTR", "PCA", "PLP", "PMP",
    "A", "C", "G", "U", "I", "DA", "DC", "DG", "DT", "DU", "DI",
}
PARAMAGNETIC_OR_TRANSITION = {
    "FE", "CU", "MN", "CO", "NI", "CR", "V", "TI", "MO", "W", "RU", "RH",
    "PD", "OS", "IR", "PT", "RE", "CE", "PR", "ND", "PM", "SM", "EU", "GD",
    "TB", "DY", "HO", "ER", "TM", "YB", "LU", "U",
}
OTHER_METALS = {
    "LI", "NA", "K", "RB", "CS", "BE", "MG", "CA", "SR", "BA", "AL", "GA",
    "IN", "SN", "PB", "BI", "CD", "HG", "AG", "AU", "Y",
}

# The packaged rule: for each category the exclusion cutoff in angstrom and the census
# evidence behind it. A residue with a backbone heavy atom closer than the cutoff to any
# heavy atom of an entity of that category is excluded.
_RULE = json.loads(resources.files("deepshift.data").joinpath(
    "hetatm_exclusion_rule.json").read_text())
EXCLUDE_CUTOFFS = {name: float(item["cutoff_A"]) for name, item in _RULE["exclude"].items()}
BACKBONE = ("N", "CA", "C", "O")


@dataclass
class Entity:
    chain: str
    resid: int
    icode: str
    resname: str
    atoms: list[dict]

    @property
    def elements(self) -> list[str]:
        return [a["element"] for a in self.atoms]

    @property
    def heavy_atoms(self) -> list[dict]:
        return [a for a in self.atoms if a["element"] not in {"H", "D"}]


def infer_element(atom_name: str, element_field: str, resname: str = "") -> str:
    """Element from the PDB element column, or from the atom name when it is blank."""
    e = element_field.strip().upper()
    if e:
        return e
    raw = atom_name.strip().upper()
    rn = resname.strip().upper()
    if rn in PARAMAGNETIC_OR_TRANSITION | OTHER_METALS | {"ZN"} and raw == rn:
        return rn
    raw = raw.lstrip("0123456789")
    if raw.startswith("CL"):
        return "CL"
    if raw.startswith("BR"):
        return "BR"
    if raw.startswith("SE"):
        return "SE"
    return raw[:1] or "X"


def _element_columns(path: str) -> dict[tuple, str]:
    """The PDB element column of every coordinate record, keyed by
    (chain, seq, icode, atom name). Blank where the file carries none."""
    elements: dict[tuple, str] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if line[:6].strip() not in {"ATOM", "HETATM"} or len(line) < 27:
                continue
            try:
                seq = int(line[22:26])
            except ValueError:
                continue
            key = (line[21].strip(), seq, line[26].strip(), line[12:16].strip())
            elements.setdefault(key, line[76:78] if len(line) >= 78 else "")
    return elements


def nonstandard_entities(pdb_path: str, model_index: int) -> list[Entity]:
    """Nonstandard residues of one model, all chains, with alternate locations resolved
    by the structure module's highest-summed-occupancy rule."""
    models = parse_models(pdb_path, chain=None)
    if model_index >= len(models):
        return []
    elements = _element_columns(pdb_path)
    entities = []
    for (chain, seq, icode), info in models[model_index].items():
        resname = info["residue_name"].upper()
        if resname in STANDARD_RESIDUES:
            continue
        atoms = []
        for name, xyz in info["atoms"].items():
            xyz = np.asarray(xyz, dtype=np.float64)
            if not np.all(np.isfinite(xyz)):
                continue
            element = infer_element(name, elements.get((chain, seq, icode, name), ""), resname)
            atoms.append({"name": name, "element": element, "xyz": xyz})
        entity = Entity(chain, seq, icode, resname, atoms)
        if entity.heavy_atoms:
            entities.append(entity)
    return entities


def _aromatic_like(entity: Entity) -> bool:
    """Coordinate/element heuristic for a large conjugated carbon-rich ligand."""
    heavy = entity.heavy_atoms
    if len(heavy) < 10:
        return False
    ring_atoms = [a for a in heavy if a["element"] in {"C", "N"}]
    carbons = [a for a in ring_atoms if a["element"] == "C"]
    if len(carbons) < 6 or len(ring_atoms) < 6:
        return False
    xyz = np.asarray([a["xyz"] for a in ring_atoms])
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    adjacency = (dist >= 1.15) & (dist <= 1.82)
    keep = np.ones(len(ring_atoms), dtype=bool)
    while True:
        degree = adjacency[:, keep][keep].sum(1)
        drop_local = degree < 2
        if not drop_local.any():
            break
        current = np.where(keep)[0]
        keep[current[drop_local]] = False
        if keep.sum() < 6:
            return False
    core = np.where(keep)[0]
    if len(core) < 6:
        return False
    core_xyz = xyz[core]
    _, _, vt = np.linalg.svd(core_xyz - core_xyz.mean(0), full_matrices=False)
    plane_rms = float(np.sqrt(np.mean(((core_xyz - core_xyz.mean(0)) @ vt[-1]) ** 2)))
    return plane_rms <= 0.35


def categorize_entity(entity: Entity) -> str:
    rn = entity.resname.upper()
    elems = set(entity.elements)
    heavy_n = len(entity.heavy_atoms)
    if rn in WATER_NAMES:
        return "water"
    if rn == "MSE":
        return "mse"
    if rn in HEME_NAMES or ("FE" in elems and heavy_n >= 20 and _aromatic_like(entity)):
        return "heme_porphyrin"
    if "ZN" in elems or rn == "ZN":
        return "zinc"
    if elems & PARAMAGNETIC_OR_TRANSITION or (rn in PARAMAGNETIC_OR_TRANSITION and heavy_n <= 2):
        return "paramagnetic_transition_metal"
    if elems & OTHER_METALS or (rn in OTHER_METALS and heavy_n <= 2):
        return "other_metal"
    if "P" in elems or rn in NUCLEOTIDE_NAMES:
        return "phosphate_nucleotide"
    if rn in SUGAR_NAMES:
        return "sugar"
    if rn in SOLVENT_NAMES:
        return "organic_solvent_buffer_detergent"
    if _aromatic_like(entity):
        return "large_aromatic_ligand"
    return "other"


def perturber_excluded_rids(pdb_path: str, chain: str, aa: dict[int, dict]) -> set[int]:
    """Return standard residue ids inside a census-derived exclusion cutoff."""
    entities = nonstandard_entities(
        pdb_path, structures.select_best_nmr_model_index(pdb_path, chain))
    by_category = defaultdict(list)
    for entity in entities:
        category = categorize_entity(entity)
        if category in EXCLUDE_CUTOFFS:
            by_category[category].extend(atom["xyz"] for atom in entity.heavy_atoms)
    trees = {category: cKDTree(np.asarray(coords, dtype=np.float64))
             for category, coords in by_category.items()}

    excluded = set()
    for rid, info in aa.items():
        if not info.get("is_standard", True):
            continue
        atoms = info.get("atoms", {})
        points = np.asarray([atoms[name] for name in BACKBONE
                             if name in atoms and np.all(np.isfinite(atoms[name]))],
                            dtype=np.float64)
        if not len(points):
            continue
        for category, tree in trees.items():
            distances, _ = tree.query(points, k=1)
            if float(np.min(distances)) < EXCLUDE_CUTOFFS[category]:
                excluded.add(int(rid))
                break
    return excluded
