"""The aromatic ring-current field: its sign, its reach, and what it excludes."""
import numpy as np
import pytest

from deepshift import ring_current

RADIUS = 1.39


def benzene(key="ring", center=(0.0, 0.0, 0.0)):
    """One phenylalanine ring in the xy plane, as ``ring_geometry`` reads residues."""
    names = ("CG", "CD1", "CE1", "CZ", "CE2", "CD2")
    center = np.asarray(center, dtype=float)
    atoms = {name: center + RADIUS * np.array([np.cos(angle), np.sin(angle), 0.0])
             for name, angle in zip(names, np.linspace(0, 2 * np.pi, 6, endpoint=False))}
    return {key: {"residue_name": "PHE", "atoms": atoms}}


def test_the_ring_is_found_with_its_centre_and_its_normal():
    rings = ring_current.ring_geometry(benzene())
    assert len(rings) == 1
    assert np.allclose(rings[0].center, [0.0, 0.0, 0.0], atol=1e-9)
    assert abs(abs(rings[0].normal[2]) - 1.0) < 1e-9
    assert rings[0].intensity == 1.0


def test_a_ring_missing_an_atom_is_skipped_rather_than_guessed_at():
    residues = benzene()
    residues["ring"]["atoms"].pop("CZ")
    assert ring_current.ring_geometry(residues) == []


def test_tryptophan_contributes_both_of_its_fused_rings():
    assert len(ring_current.AROMATIC_RINGS["TRP"]) == 2


def test_the_shielding_is_positive_above_the_face_and_negative_in_the_plane():
    """Above a ring the proton is shielded, which is an upfield shift once negated."""
    rings = ring_current.ring_geometry(benzene())
    above = ring_current.ring_current_shielding([0.0, 0.0, 3.0], rings)
    edge = ring_current.ring_current_shielding([5.0, 0.0, 0.0], rings)
    assert above > 0
    assert edge < 0
    assert abs(above) > abs(edge)


def test_the_field_falls_off_as_one_over_r_cubed():
    rings = ring_current.ring_geometry(benzene())
    near = ring_current.ring_current_shielding([0.0, 0.0, 3.0], rings)
    far = ring_current.ring_current_shielding([0.0, 0.0, 6.0], rings)
    assert near / far == pytest.approx(8.0, rel=1e-6)


def test_a_ring_past_the_cutoff_contributes_nothing():
    rings = ring_current.ring_geometry(benzene())
    beyond = ring_current.RING_CUTOFF_ANGSTROM + 1.0
    assert ring_current.ring_current_shielding([0.0, 0.0, beyond], rings) == 0.0


def test_the_protons_own_residue_is_excluded():
    """A proton never feels its own ring, and skipping is by residue, not by ring."""
    residues = {**benzene("own"), **benzene("other", center=(0.0, 0.0, 4.0))}
    rings = ring_current.ring_geometry(residues)
    point = [0.0, 0.0, 1.0]
    both = ring_current.ring_current_shielding(point, rings)
    without = ring_current.ring_current_shielding(point, rings, skip="own")
    assert both != without
    assert without == pytest.approx(
        ring_current.ring_current_shielding(
            point, [r for r in rings if r.residue == "other"]))


def test_the_proton_is_placed_outward_from_its_own_ring_centre():
    """No hydrogen is read from the coordinates, so the position is constructed."""
    rings = ring_current.ring_geometry(benzene())
    parent = np.array([RADIUS, 0.0, 0.0])
    placed = ring_current.proton_position(parent, rings)
    assert np.allclose(placed, [RADIUS + ring_current.CH_BOND_ANGSTROM, 0.0, 0.0])
    assert ring_current.proton_position(parent, []) is None


def test_every_aromatic_proton_hangs_off_an_atom_of_its_own_residues_rings():
    """A site whose parent is not on a ring would be placed from nothing."""
    for residue, protons in ring_current.AROMATIC_PROTONS.items():
        vertices = {name for names, _intensity in ring_current.AROMATIC_RINGS[residue]
                    for name in names}
        assert set(protons.values()) <= vertices, residue
