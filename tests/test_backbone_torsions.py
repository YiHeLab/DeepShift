"""phi and psi read off a contact-only cloud, where no residue is numbered."""
import numpy as np

from deepshift.atom_cloud import FINAL_ATOM_CLOUD_RECIPE
from deepshift.domain import ATOM_TO_IDX
from deepshift.geometry_context import backbone_torsions_degrees, dihedral_degrees

CENTER = FINAL_ATOM_CLOUD_RECIPE.center_offset_code
SPATIAL = FINAL_ATOM_CLOUD_RECIPE.spatial_offset_code
PEPTIDE = 1.33


def residue(previous_c=True, next_n=True):
    """One centre residue, with the neighbours a peptide bond away that it asks for."""
    nitrogen = np.array([0.0, 0.0, 0.0])
    alpha = np.array([1.458, 0.0, 0.0])
    carbon = np.array([2.0, 1.4, 0.0])
    types = [ATOM_TO_IDX["N"], ATOM_TO_IDX["CA"], ATOM_TO_IDX["C"]]
    offsets = [CENTER, CENTER, CENTER]
    points = [nitrogen, alpha, carbon]
    before = nitrogen + PEPTIDE * np.array([-0.6, -0.8, 0.0])
    after = carbon + PEPTIDE * np.array([0.5, 0.6, 0.62])/np.linalg.norm([0.5, 0.6, 0.62])
    if previous_c:
        types.append(ATOM_TO_IDX["C"])
        offsets.append(SPATIAL)
        points.append(before)
    if next_n:
        types.append(ATOM_TO_IDX["N"])
        offsets.append(SPATIAL)
        points.append(after)
    return (np.asarray(types), np.asarray(offsets), np.asarray(points),
            {"N": nitrogen, "CA": alpha, "C": carbon, "prev_C": before, "next_N": after})


def test_the_angles_are_the_dihedrals_of_the_atoms_the_search_should_find():
    types, offsets, points, named = residue()
    phi, psi = backbone_torsions_degrees(types, offsets, points)
    assert phi == dihedral_degrees(named["prev_C"], named["N"], named["CA"], named["C"])
    assert psi == dihedral_degrees(named["N"], named["CA"], named["C"], named["next_N"])


def test_a_missing_neighbour_yields_nan_rather_than_a_guess():
    types, offsets, points, _named = residue(previous_c=False)
    phi, psi = backbone_torsions_degrees(types, offsets, points)
    assert np.isnan(phi)
    assert np.isfinite(psi)

    types, offsets, points, _named = residue(next_n=False)
    phi, psi = backbone_torsions_degrees(types, offsets, points)
    assert np.isfinite(phi)
    assert np.isnan(psi)


def test_an_atom_packed_against_the_backbone_is_not_taken_for_a_bonded_neighbour():
    """The search takes the nearest candidate and then requires a bond length, so a
    non-bonded contact leaves the angle undefined instead of inventing one."""
    types, offsets, points, named = residue(previous_c=False)
    types = np.append(types, ATOM_TO_IDX["C"])
    offsets = np.append(offsets, SPATIAL)
    points = np.vstack([points, named["N"] + np.array([3.5, 0.0, 0.0])])
    phi, _psi = backbone_torsions_degrees(types, offsets, points)
    assert np.isnan(phi)


def test_a_cloud_without_the_centre_backbone_has_no_angles():
    types, offsets, points, _named = residue()
    keep = offsets != CENTER
    phi, psi = backbone_torsions_degrees(types[keep], offsets[keep], points[keep])
    assert np.isnan(phi) and np.isnan(psi)
