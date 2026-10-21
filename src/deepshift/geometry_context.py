"""Geometry classifications used by the structural consistency checks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atom_cloud import FINAL_ATOM_CLOUD_RECIPE, AtomCloudRecipe
from .domain import ATOM_TO_IDX

CYS_REDUCED_MAX_PPM = 32.0
CYS_OXIDIZED_MIN_PPM = 35.0
DISULFIDE_BOND_MAX_ANGSTROM = 2.3
PRO_CIS_MIN_PPM = 8.0
PRO_TRANS_MAX_PPM = 6.0
OMEGA_CIS_MAX_DEGREES = 30.0
OMEGA_TRANS_MIN_DEGREES = 150.0
PEPTIDE_CN_RANGE_ANGSTROM = (1.20, 1.45)
CA_C_RANGE_ANGSTROM = (1.40, 1.70)


@dataclass(frozen=True)
class StructuralState:
    shift_state: str | None
    geometry_state: str | None

    @property
    def conflicts(self) -> bool:
        return (
            self.shift_state is not None
            and self.geometry_state is not None
            and self.shift_state != self.geometry_state
        )


def dihedral_degrees(p0, p1, p2, p3) -> float:
    """Signed dihedral angle in degrees."""
    p0, p1, p2, p3 = (np.asarray(point, dtype=np.float64)
                      for point in (p0, p1, p2, p3))
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    norm = np.linalg.norm(b1)
    if norm == 0:
        return float("nan")
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _bonded_neighbor(atom_xyz, candidates, origin, span):
    """The candidate atom nearest ``origin``, when that distance is a bond length.

    A contact-only cloud carries no residue numbering, so a covalent partner is found
    by its bond: nearest first, then the bond-length window, so a second residue packed
    against the same atom cannot pass as bonded.
    """
    if not len(candidates):
        return None
    distances = np.linalg.norm(atom_xyz[candidates] - origin, axis=1)
    nearest = int(np.argmin(distances))
    low, high = span
    if not low < distances[nearest] < high:
        return None
    return atom_xyz[candidates[nearest]]


def cysteine_shift_state(cb_shift_ppm: float) -> str | None:
    """Reduced/oxidized assignment outside the Cys Cβ deadband."""
    if cb_shift_ppm < CYS_REDUCED_MAX_PPM:
        return "reduced"
    if cb_shift_ppm > CYS_OXIDIZED_MIN_PPM:
        return "oxidized"
    return None


def cysteine_geometry_state(
    atom_type,
    atom_offset,
    atom_xyz,
    *,
    recipe: AtomCloudRecipe = FINAL_ATOM_CLOUD_RECIPE,
) -> str | None:
    """Oxidized when the center Cys SG has a non-center SG within 2.3 Å."""
    atom_type = np.asarray(atom_type)
    atom_offset = np.asarray(atom_offset)
    atom_xyz = np.asarray(atom_xyz, dtype=np.float64)
    sg = ATOM_TO_IDX["SG"]
    center = np.flatnonzero(
        (atom_type == sg) & (atom_offset == recipe.center_offset_code)
    )
    if not len(center):
        return None
    partners = np.flatnonzero(
        (atom_type == sg) & (atom_offset != recipe.center_offset_code)
    )
    origin = atom_xyz[center[0]]
    bonded = any(
        np.linalg.norm(atom_xyz[index] - origin) < DISULFIDE_BOND_MAX_ANGSTROM
        for index in partners
    )
    return "oxidized" if bonded else "reduced"


def cysteine_state(atom_type, atom_offset, atom_xyz, cb_shift_ppm, *, recipe=None):
    """Shift and geometry assignments for one cysteine residue."""
    selected = recipe or FINAL_ATOM_CLOUD_RECIPE
    return StructuralState(
        cysteine_shift_state(float(cb_shift_ppm)),
        cysteine_geometry_state(atom_type, atom_offset, atom_xyz, recipe=selected),
    )


def proline_shift_state(cb_shift_ppm: float, cg_shift_ppm: float) -> str | None:
    """Cis/trans assignment from the deposited Pro Cβ−Cγ difference."""
    difference = cb_shift_ppm - cg_shift_ppm
    if difference >= PRO_CIS_MIN_PPM:
        return "cis"
    if difference <= PRO_TRANS_MAX_PPM:
        return "trans"
    return None


def proline_omega_degrees(
    atom_type,
    atom_offset,
    atom_xyz,
    *,
    recipe: AtomCloudRecipe = FINAL_ATOM_CLOUD_RECIPE,
) -> float | None:
    """Absolute Pro ω located from peptide-bond geometry in a contact-only cloud."""
    atom_type = np.asarray(atom_type)
    atom_offset = np.asarray(atom_offset)
    atom_xyz = np.asarray(atom_xyz, dtype=np.float64)
    center_code = recipe.center_offset_code
    n_atom, ca_atom, c_atom = (ATOM_TO_IDX[name] for name in ("N", "CA", "C"))
    center_n = np.flatnonzero((atom_type == n_atom) & (atom_offset == center_code))
    center_ca = np.flatnonzero((atom_type == ca_atom) & (atom_offset == center_code))
    if not (len(center_n) and len(center_ca)):
        return None
    n_xyz = atom_xyz[center_n[0]]
    ca_xyz = atom_xyz[center_ca[0]]
    neighbor_c = np.flatnonzero((atom_type == c_atom) & (atom_offset != center_code))
    previous_c = _bonded_neighbor(atom_xyz, neighbor_c, n_xyz, PEPTIDE_CN_RANGE_ANGSTROM)
    if previous_c is None:
        return None
    neighbor_ca = np.flatnonzero((atom_type == ca_atom) & (atom_offset != center_code))
    previous_ca = _bonded_neighbor(atom_xyz, neighbor_ca, previous_c, CA_C_RANGE_ANGSTROM)
    if previous_ca is None:
        return None
    omega = dihedral_degrees(previous_ca, previous_c, n_xyz, ca_xyz)
    return None if not np.isfinite(omega) else abs(omega)


def proline_geometry_state(omega_degrees: float | None) -> str | None:
    """Cis/trans assignment outside the ω deadband."""
    if omega_degrees is None or not np.isfinite(omega_degrees):
        return None
    if omega_degrees <= OMEGA_CIS_MAX_DEGREES:
        return "cis"
    if omega_degrees >= OMEGA_TRANS_MIN_DEGREES:
        return "trans"
    return None


def proline_state(
    atom_type,
    atom_offset,
    atom_xyz,
    cb_shift_ppm,
    cg_shift_ppm,
    *,
    recipe=None,
) -> StructuralState:
    """Shift and geometry assignments for one proline residue."""
    selected = recipe or FINAL_ATOM_CLOUD_RECIPE
    omega = proline_omega_degrees(
        atom_type, atom_offset, atom_xyz, recipe=selected
    )
    return StructuralState(
        proline_shift_state(float(cb_shift_ppm), float(cg_shift_ppm)),
        proline_geometry_state(omega),
    )


def backbone_torsions_degrees(
    atom_type,
    atom_offset,
    atom_xyz,
    *,
    recipe: AtomCloudRecipe = FINAL_ATOM_CLOUD_RECIPE,
) -> tuple[float, float]:
    """``(phi, psi)`` of the cloud's center residue, in degrees, NaN where undefined.

    phi needs the preceding residue's C and psi the following residue's N. Neither is
    labelled in a contact-only cloud, so each is located by its peptide bond -- the
    nearest non-center C to the center N, and the nearest non-center N to the center C,
    both inside ``PEPTIDE_CN_RANGE_ANGSTROM``. A terminal residue, or one whose neighbour
    the 8 A cloud does not carry, yields NaN on that angle rather than a guess.
    """
    atom_type = np.asarray(atom_type)
    atom_offset = np.asarray(atom_offset)
    atom_xyz = np.asarray(atom_xyz, dtype=np.float64)
    center_code = recipe.center_offset_code
    n_atom, ca_atom, c_atom = (ATOM_TO_IDX[name] for name in ("N", "CA", "C"))
    center = {}
    for name, index in (("N", n_atom), ("CA", ca_atom), ("C", c_atom)):
        found = np.flatnonzero((atom_type == index) & (atom_offset == center_code))
        if not len(found):
            return float("nan"), float("nan")
        center[name] = atom_xyz[found[0]]

    previous_c = _bonded_neighbor(
        atom_xyz, np.flatnonzero((atom_type == c_atom) & (atom_offset != center_code)),
        center["N"], PEPTIDE_CN_RANGE_ANGSTROM)
    next_n = _bonded_neighbor(
        atom_xyz, np.flatnonzero((atom_type == n_atom) & (atom_offset != center_code)),
        center["C"], PEPTIDE_CN_RANGE_ANGSTROM)
    phi = (float("nan") if previous_c is None else
           dihedral_degrees(previous_c, center["N"], center["CA"], center["C"]))
    psi = (float("nan") if next_n is None else
           dihedral_degrees(center["N"], center["CA"], center["C"], next_n))
    return phi, psi
