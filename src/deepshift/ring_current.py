"""The aromatic ring-current field at a proton, computed from coordinates alone.

The figure that uses this asks whether a predictor responds to a field the structure
already fixes, so the field must be computed from the structure and never from a shift.
One ring contributes

    sigma = -B * f * (1 - 3 cos^2 theta) / r^3

with ``r`` the distance from the ring centre to the point, ``theta`` the angle between
that vector and the ring normal, ``B`` benzene's intensity and ``f`` the ring's intensity
relative to benzene. ``sigma`` is a SHIELDING: it is positive above the ring face, and the
ppm shift the nucleus sees is its negative. The sum runs over every ring within
:data:`RING_CUTOFF_ANGSTROM`, excluding the proton's own residue.

Aromatic protons are not in the coordinates -- the pipeline builds hydrogen-free clouds --
so each is placed at :data:`CH_BOND_ANGSTROM` from its heavy parent, along the line from
the parent's own ring centre through the parent. The form written above is the point
dipole.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Ring vertices and the ring's intensity relative to benzene (Case 1995; Wishart).
#: Tryptophan carries two fused rings and contributes both.
AROMATIC_RINGS = {
    "PHE": ((("CG", "CD1", "CE1", "CZ", "CE2", "CD2"), 1.00),),
    "TYR": ((("CG", "CD1", "CE1", "CZ", "CE2", "CD2"), 0.81),),
    "HIS": ((("CG", "ND1", "CE1", "NE2", "CD2"), 0.69),),
    "TRP": ((("CG", "CD1", "NE1", "CE2", "CD2"), 0.57),
            (("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"), 1.02)),
}

#: Benzene's ring-current intensity, ppm A^3.
BENZENE_INTENSITY_PPM_A3 = 5.4548e-6 * 1e6
#: A ring further than this from the point contributes nothing.
RING_CUTOFF_ANGSTROM = 12.0
#: Where an aromatic proton sits from its heavy parent.
CH_BOND_ANGSTROM = 1.08

#: The aromatic CH and NH sites, and the heavy atom each proton hangs off.
AROMATIC_PROTONS = {
    "PHE": {"HD1": "CD1", "HD2": "CD2", "HE1": "CE1", "HE2": "CE2", "HZ": "CZ"},
    "TYR": {"HD1": "CD1", "HD2": "CD2", "HE1": "CE1", "HE2": "CE2"},
    "TRP": {"HD1": "CD1", "HE1": "NE1", "HE3": "CE3", "HZ2": "CZ2", "HZ3": "CZ3",
            "HH2": "CH2"},
    "HIS": {"HD2": "CD2", "HE1": "CE1"},
}


@dataclass(frozen=True)
class Ring:
    """One aromatic ring of one residue, as the field sum needs it."""

    center: np.ndarray
    normal: np.ndarray
    intensity: float
    residue: object


def ring_geometry(residues) -> list[Ring]:
    """Every aromatic ring of ``{residue key: {'residue_name', 'atoms'}}``.

    The centre is the mean of the ring's vertices and the normal is the least-variance
    direction of those vertices, so a ring missing an atom is skipped rather than
    guessed at.
    """
    rings = []
    for key, residue in residues.items():
        atoms = residue["atoms"]
        for names, intensity in AROMATIC_RINGS.get(residue["residue_name"], ()):
            if not all(name in atoms for name in names):
                continue
            points = np.asarray([atoms[name] for name in names], dtype=np.float64)
            center = points.mean(0)
            normal = np.linalg.svd(points - center)[2][2]
            rings.append(Ring(center, normal, float(intensity), key))
    return rings


def ring_current_shielding(point, rings, *, skip=None) -> float:
    """The shielding in ppm at ``point``, summed over every ring except ``skip``'s."""
    point = np.asarray(point, dtype=np.float64)
    total = 0.0
    for ring in rings:
        if ring.residue == skip:
            continue
        offset = point - ring.center
        distance = float(np.linalg.norm(offset))
        if distance < 1e-6 or distance > RING_CUTOFF_ANGSTROM:
            continue
        cosine = float(offset @ ring.normal) / distance
        total += (-BENZENE_INTENSITY_PPM_A3 * ring.intensity
                  * (1.0 - 3.0 * cosine * cosine) / distance ** 3)
    return total


def proton_position(parent_xyz, own_rings) -> np.ndarray | None:
    """Where an aromatic proton sits: outward from its own ring, through its parent.

    ``own_rings`` are the rings of the proton's own residue. None when the parent sits
    on a ring centre, which no aromatic ring does.
    """
    parent_xyz = np.asarray(parent_xyz, dtype=np.float64)
    if not own_rings:
        return None
    nearest = min(own_rings, key=lambda ring: np.linalg.norm(parent_xyz - ring.center))
    outward = parent_xyz - nearest.center
    length = float(np.linalg.norm(outward))
    if length < 1e-6:
        return None
    return parent_xyz + CH_BOND_ANGSTROM * outward / length
