"""Contact-only atom clouds used by both cache construction and prediction."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from .domain import ATOM_TO_IDX, RESIDUE_TO_IDX, STANDARD_RESIDUES
from .structure import is_hydrogen

BACKBONE_ATOMS = ("N", "CA", "C", "O")
# The cloud the paper reports: heavy atoms only, centred on the target CA, with the
# NMR-sample partners. A recipe stamped with anything else is refused.
REPORTED_CHOICES = {"coordinate_frame": "target_ca_centered_fp16", "drop_hydrogens": True,
                    "partner_policy": "sample_matched"}


@dataclass(frozen=True)
class AtomCloudRecipe:
    """Complete, serializable definition of an atom-cloud representation.

    ``partner_policy`` records which protein chains the caller is allowed to provide.
    The geometry function cannot infer NMR sample composition; the cache builder must
    select the partners before calling it and stamp that decision here.
    """

    m_atoms: int | None = 512
    heavy_cutoff: float = 8.0
    center_offset_code: int = 6
    spatial_offset_code: int = 12
    pad_atom_index: int = len(ATOM_TO_IDX)
    pad_offset_index: int = 17
    coordinate_frame: Literal["target_ca_centered_fp16"] = "target_ca_centered_fp16"
    drop_hydrogens: Literal[True] = True
    partner_policy: Literal["sample_matched"] = "sample_matched"
    spatial_scheme: Literal["contact_only_8a"] = "contact_only_8a"

    def __post_init__(self):
        for name, reported in REPORTED_CHOICES.items():
            if getattr(self, name) != reported:
                raise ValueError(f"atom recipe {name}={getattr(self, name)!r} is not the "
                                 f"final recipe's {reported!r}")
        if self.m_atoms is not None and int(self.m_atoms) <= 0:
            raise ValueError(f"m_atoms must be positive or None; got {self.m_atoms}")
        if not np.isfinite(self.heavy_cutoff) or self.heavy_cutoff <= 0:
            raise ValueError(f"heavy_cutoff must be positive; got {self.heavy_cutoff}")
        if self.pad_atom_index < len(ATOM_TO_IDX):
            raise ValueError(
                f"pad_atom_index={self.pad_atom_index} overlaps the "
                f"{len(ATOM_TO_IDX)}-token atom alphabet"
            )
        real_offsets = max(self.center_offset_code, self.spatial_offset_code)
        if self.pad_offset_index <= real_offsets:
            raise ValueError(
                f"pad_offset_index={self.pad_offset_index} must exceed {real_offsets}"
            )

    def with_vocabulary(self, atom_pad_index: int, offset_pad_index: int):
        """Return this recipe with the vocabulary owned by one checkpoint/cache."""
        return replace(
            self,
            pad_atom_index=int(atom_pad_index),
            pad_offset_index=int(offset_pad_index),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> AtomCloudRecipe:
        """Read a complete stamped recipe. A missing field raises rather than taking a
        default: a cache or checkpoint that does not say how its cloud was built cannot
        be reproduced."""
        if not isinstance(values, Mapping):
            raise ValueError("atom recipe must be a mapping of every recipe field")
        names = [field.name for field in fields(cls)]
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(f"atom recipe is missing {missing}")
        return cls(
            m_atoms=None if values["m_atoms"] is None else int(values["m_atoms"]),
            heavy_cutoff=float(values["heavy_cutoff"]),
            center_offset_code=int(values["center_offset_code"]),
            spatial_offset_code=int(values["spatial_offset_code"]),
            pad_atom_index=int(values["pad_atom_index"]),
            pad_offset_index=int(values["pad_offset_index"]),
            coordinate_frame=str(values["coordinate_frame"]),
            drop_hydrogens=bool(values["drop_hydrogens"]),
            partner_policy=str(values["partner_policy"]),
            spatial_scheme=str(values["spatial_scheme"]),
        )


FINAL_ATOM_CLOUD_RECIPE = AtomCloudRecipe()

@dataclass(frozen=True)
class AtomCloud:
    """Atom arrays plus residue metadata and census fields for one structure."""

    atom_type: np.ndarray
    atom_offset: np.ndarray
    atom_xyz: np.ndarray
    atom_mask: np.ndarray
    residue_ids: list
    residue_names: list[str]
    residue_codes: np.ndarray
    nucleus_atom_idx: np.ndarray | None
    nonstandard_contact: np.ndarray
    precap_atom_count: np.ndarray

    @property
    def model_arrays(self):
        return self.atom_type, self.atom_offset, self.atom_xyz, self.atom_mask


def _finite_atoms(residue: dict) -> list[np.ndarray]:
    return [
        np.asarray(coordinate, dtype=np.float64)
        for name, coordinate in residue["atoms"].items()
        if not is_hydrogen(name)
        and coordinate is not None
        and np.all(np.isfinite(coordinate))
    ]


def _residue_sort_rank(key):
    """Deterministic ordering across target integers and partner tuple keys."""
    if isinstance(key, tuple):
        chain = key[0]
        seq = key[1] if len(key) > 1 else 0
        icode = key[2] if len(key) > 2 else ""
        return (1, str(chain), int(seq), str(icode))
    return (0, int(key), "", "")


def contact_partner_map(
    residues: dict[int, dict],
    residue_ids: list[int],
    *,
    recipe: AtomCloudRecipe = FINAL_ATOM_CLOUD_RECIPE,
    partner_residues: dict | None = None,
) -> dict[int, list]:
    """Non-self residues with any atom inside the recipe's contact cutoff."""
    coordinates: list[np.ndarray] = []
    owners: list = []
    for residue_id in residue_ids:
        for coordinate in _finite_atoms(residues[residue_id]):
            coordinates.append(coordinate)
            owners.append(residue_id)
    if partner_residues:
        for partner_key, residue in partner_residues.items():
            for coordinate in _finite_atoms(residue):
                coordinates.append(coordinate)
                owners.append(partner_key)
    if not coordinates:
        return {residue_id: [] for residue_id in residue_ids}

    tree = cKDTree(np.asarray(coordinates, dtype=np.float64))
    result: dict[int, list] = {}
    for residue_id in residue_ids:
        target = _finite_atoms(residues[residue_id])
        hits: set[int] = set()
        if target:
            for indices in tree.query_ball_point(
                np.asarray(target, dtype=np.float64), r=recipe.heavy_cutoff
            ):
                hits.update(indices)
        partners = {owners[index] for index in hits} if hits else set()
        partners.discard(residue_id)
        result[residue_id] = sorted(partners, key=_residue_sort_rank)
    return result


def _gather_rows(residues, residue_ids, partners, recipe, partner_residues=None):
    """Gather uncapped, deterministically ordered atoms for every valid center."""
    lookup = residues if not partner_residues else {**residues, **partner_residues}
    rows = []
    for residue_id in residue_ids:
        residue = residues[residue_id]
        if not residue.get("is_standard", True):
            continue
        center = residue["atoms"].get("CA")
        if center is None or not np.all(np.isfinite(np.asarray(center, dtype=np.float64))):
            continue
        center = np.asarray(center, dtype=np.float32)
        contact_set = partners[residue_id]
        shell = any(not lookup[item].get("is_standard", True) for item in contact_set)
        members = [(residue_id, True)] + [(partner, False) for partner in contact_set]

        atoms = []
        for other, is_center in members:
            if not lookup[other].get("is_standard", True):
                continue
            code = (recipe.center_offset_code if is_center
                    else recipe.spatial_offset_code)
            for name, coordinate in lookup[other]["atoms"].items():
                if is_hydrogen(name):
                    continue
                atom_type = ATOM_TO_IDX.get(name)
                point = np.asarray(coordinate, dtype=np.float32)
                if atom_type is None or not np.all(np.isfinite(point)):
                    continue
                distance = float(np.linalg.norm(point - center))
                atoms.append((atom_type, code, point, distance, is_center, other, name))
        if not atoms:
            continue
        atoms.sort(
            key=lambda item: (
                0 if item[4] else 1,
                item[3],
                _residue_sort_rank(item[5]),
                item[6],
            )
        )
        rows.append((residue_id, residue, center, shell, atoms))
    return rows


def target_cloud(residues: dict[int, dict], residue_id: int, *,
                 recipe: AtomCloudRecipe = FINAL_ATOM_CLOUD_RECIPE) -> list[tuple]:
    """The uncapped atoms the model is handed for one target residue, in model order,
    as (residue id, atom name, coordinate). For a figure that draws the cloud."""
    contacts = contact_partner_map(residues, sorted(residues), recipe=recipe)
    rows = _gather_rows(residues, [residue_id], contacts, recipe)
    return [(owner, name, point) for _t, _c, point, _d, _ic, owner, name in rows[0][4]] \
        if rows else []


def _filtered_partners(partner_residues, suppress_chains):
    if not partner_residues:
        return None
    if not suppress_chains:
        return partner_residues
    suppressed = set(suppress_chains)
    return {key: value for key, value in partner_residues.items()
            if key[0] not in suppressed} or None


def census_atom_counts(
    residues: dict[int, dict],
    *,
    recipe: AtomCloudRecipe | None = None,
    partner_residues: dict | None = None,
    suppress_chains=None,
) -> list[int]:
    """Uncapped atom count for each valid center under exactly one recipe."""
    if recipe is None:
        recipe = replace(FINAL_ATOM_CLOUD_RECIPE, m_atoms=None)
    uncapped = replace(recipe, m_atoms=None)
    partners = _filtered_partners(partner_residues, suppress_chains)
    residue_ids = sorted(residues)
    contacts = contact_partner_map(
        residues, residue_ids, recipe=uncapped, partner_residues=partners
    )
    return [len(row[4]) for row in
            _gather_rows(residues, residue_ids, contacts, uncapped, partners)]


def build_atom_cloud(
    residues: dict[int, dict],
    *,
    recipe: AtomCloudRecipe = FINAL_ATOM_CLOUD_RECIPE,
    nucleus_atoms: list[str] | tuple[str, ...] | None = None,
    partner_residues: dict | None = None,
    suppress_chains=None,
) -> AtomCloud | None:
    """Build the recipe-defined contact-only cloud for every standard CA center."""
    partners = _filtered_partners(partner_residues, suppress_chains)
    residue_ids = sorted(residues)
    contacts = contact_partner_map(
        residues, residue_ids, recipe=recipe, partner_residues=partners
    )
    rows = _gather_rows(residues, residue_ids, contacts, recipe, partners)
    if not rows:
        return None

    precap = [len(row[4]) for row in rows]
    cap = max(precap) if recipe.m_atoms is None else int(recipe.m_atoms)
    rows_type, rows_offset, rows_xyz, rows_mask = [], [], [], []
    rows_nucleus, rows_shell, metadata = [], [], []
    for residue_id, residue, center, shell, atoms in rows:
        rows_shell.append(shell)
        capped = atoms[:cap]
        atom_types = np.full(cap, recipe.pad_atom_index, dtype=np.int64)
        offsets = np.full(cap, recipe.pad_offset_index, dtype=np.int64)
        xyz = np.zeros((cap, 3), dtype=np.float32)
        mask = np.zeros(cap, dtype=bool)
        center_tokens: dict[str, int] = {}
        for index, (atom_type, code, point, _, is_center, _rid, name) in enumerate(capped):
            atom_types[index], offsets[index] = atom_type, code
            xyz[index] = point - center
            mask[index] = True
            if is_center:
                center_tokens[name] = index
        rows_type.append(atom_types)
        rows_offset.append(offsets)
        rows_xyz.append(xyz)
        rows_mask.append(mask)
        if nucleus_atoms is not None:
            rows_nucleus.append(np.asarray(
                [center_tokens.get(name, -1) for name in nucleus_atoms], dtype=np.int16
            ))
        name = residue["residue_name"]
        metadata.append((
            residue_id,
            name,
            RESIDUE_TO_IDX.get(name, len(STANDARD_RESIDUES) - 1),
        ))

    residue_values, names, codes = zip(*metadata)
    return AtomCloud(
        atom_type=np.stack(rows_type),
        atom_offset=np.stack(rows_offset),
        atom_xyz=np.stack(rows_xyz),
        atom_mask=np.stack(rows_mask),
        residue_ids=list(residue_values),
        residue_names=list(names),
        residue_codes=np.asarray(codes, dtype=np.int64),
        nucleus_atom_idx=(np.stack(rows_nucleus) if nucleus_atoms is not None else None),
        nonstandard_contact=np.asarray(rows_shell, dtype=bool),
        precap_atom_count=np.asarray(precap, dtype=np.int64),
    )
