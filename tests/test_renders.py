"""The ray-traced panels are frozen images; the manifest must describe them."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RENDERS = REPO / "figures" / "reference" / "renders"
MANIFEST = RENDERS / "renders.json"


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip("no renders shipped; run figures/render_structures.py")
    return json.loads(MANIFEST.read_text())


def test_every_manifest_image_is_on_disk(manifest):
    for name, image in manifest["images"].items():
        assert (RENDERS / image["file"]).exists(), f"{name} is in the manifest but not on disk"


def test_no_image_is_shipped_without_a_manifest_entry(manifest):
    listed = {image["file"] for image in manifest["images"].values()}
    on_disk = {p.name for p in RENDERS.glob("*.png")}
    assert on_disk == listed, "an unlisted render is not in the manifest"


def test_the_ring_panel_is_a_shielding_geometry(manifest):
    """A proton beside a ring is deshielded; only one over its face is shielded.

    The panel sits next to an axis labelled shielding, so a geometry that would shift the
    other way would make the axis a lie.
    """
    ring = manifest["ring_pair"]
    assert ring["angle_off_ring_normal_deg"] < 54.7    # the magic angle: sign changes here
    assert 0 < ring["distance_to_ring_centre_angstrom"] < 5.0
    assert ring["ring_ppm"] > 0


def test_the_cloud_panel_is_the_cutoff_the_model_uses(manifest):
    from deepshift.data.atoms import FINAL_ATOM_CLOUD_RECIPE

    cloud = manifest["atom_cloud"]
    assert cloud["cutoff_angstrom"] == FINAL_ATOM_CLOUD_RECIPE.heavy_cutoff
    assert cloud["atoms_drawn"] == cloud["n_atoms"]
