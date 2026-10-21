"""Ray-trace the molecular panels of Figure 1 and the graphical abstract.

A matplotlib scatter of circles is a diagram of a molecule, not a picture of one. These
panels are ray-traced in PyMOL with ambient occlusion and depth fog, written once to
``figures/reference/renders/`` with a manifest, and read back by the figure scripts. The
figures themselves stay pure matplotlib and need no PyMOL to build.

What is rendered, and why each is chosen rather than picked:

  * ``structure``  -- Figure 1's protein, the whole chain, coloured by the secondary
    structure PyMOL assigns from the coordinates.
  * ``atom_cloud`` -- the same 8 Angstrom heavy-atom neighbourhood the model is given,
    around the same residue Figure 1 already centres on.
  * ``ring_pair`` -- the benchmark's most strongly ring-shielded proton and the ring that
    shields it, in the frame the graphical abstract's curve is about. This one needs
    $DEEPSHIFT_RUN, because the cell is chosen from the run's own numbers.

PyMOL is resolved from ``--pymol``, then ``$PYMOL``, then the path; there is no built-in
default, because a hard-coded interpreter path is one more thing to rot.

Run:  PYMOL=<path to pymol> python figures/render_structures.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import frozen
from build_graphical_abstract import chain_residues, most_shielded
from build_paper_figures import REFERENCE
from style import ACCENT, AMBER, INK

RENDERS = REFERENCE / "renders"
MANIFEST = RENDERS / "renders.json"
STRUCTURE = (REFERENCE / "1POH.pdb", "A")
CUTOFF = 8.0
SIZE = (1600, 1600)

# One palette for every render, so a panel cannot drift away from the figure it sits in.
COLORS = {"helix": ACCENT, "sheet": AMBER, "loop": INK["faint"], "accent": ACCENT,
          "faint": INK["faint"], "carbon": "#5f6368", "shell": "#949aa1", "nitrogen": "#3b6fb5",
          "oxygen": "#c0483f", "sulfur": "#c9a227", "hydrogen": "#c3c8cd"}



# Every aromatic ring a side chain can present. TRP contributes both rings of the indole,
# and the ring a proton belongs to is found by which one holds the atom it is bonded to,
# so no residue needs a special case downstream.
AROMATIC_RINGS = {
    "PHE": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "TYR": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "HIS": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "TRP": (("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"),
            ("CG", "CD1", "NE1", "CE2", "CD2")),
}



def shielding_pair(residues, cell):
    """The ring that shields the chosen proton, and the ring the proton belongs to.

    The shielding ring is the one maximising the point-dipole geometry factor
    (3cos^2(theta) - 1) / r^3, the sign and distance law of a ring current: theta is the
    angle off the ring normal and r the distance from the ring centre. Positive is
    upfield. Every coordinate, the proton included, is read from the deposited file.
    """
    def rings_of(residue):
        for names in AROMATIC_RINGS.get(residue.get_resname(), ()):
            if all(n in residue for n in names):
                xyz = np.array([residue[n].coord for n in names], dtype=float)
                centre = xyz.mean(axis=0)
                _u, _s, vt = np.linalg.svd(xyz - centre, full_matrices=False)
                yield {"xyz": xyz, "centre": centre, "normal": vt[2], "names": names,
                       "name": f"{residue.get_resname()}{residue.id[1]}"}

    owner = next(r for r in residues if r.id[1] == int(cell["resnum"]))
    if cell["atom"] not in owner:
        raise SystemExit(f"{cell['accession']} {owner.get_resname()}{owner.id[1]} has no "
                         f"{cell['atom']}; the deposited file carries no such proton")
    h = np.array(owner[cell["atom"]].coord, dtype=float)

    # The heavy atom the proton is bonded to, and so which of its residue's rings it is on.
    heavy = [a for a in owner if a.element != "H"]
    bonded = min(heavy, key=lambda a: np.linalg.norm(np.array(a.coord, float) - h))
    parent = next((r for r in rings_of(owner) if bonded.get_name() in r["names"]), None)

    best = None
    for residue in residues:
        for ring in rings_of(residue):
            if ring["name"] == f"{owner.get_resname()}{owner.id[1]}":
                continue
            v = h - ring["centre"]
            r = float(np.linalg.norm(v))
            cos = abs(float(v @ ring["normal"])) / r
            factor = (3.0 * cos ** 2 - 1.0) / r ** 3
            if best is None or factor > best["factor"]:
                best = {"face": ring, "factor": factor, "r": r,
                        "theta_deg": float(np.degrees(np.arccos(cos)))}
    if best is None or best["factor"] <= 0:
        raise SystemExit("no ring shields the chosen proton; the inset would mislead")
    best.update({"h": h, "parent": parent, "bonded": np.array(bonded.coord, dtype=float)})
    return best


def view_frame(geom):
    """The camera frame for the shielded-proton panel: across the page, and up.

    The view looks down on the shielding ring's plane from 55 degrees, the angle at which
    a ring still reads as a ring and a displacement along its normal still reads as
    height, and along the direction that puts the two ring centres across the page. The
    normal is oriented towards the proton, so the proton is above the ring, not below it.
    """
    face = geom["face"]
    normal = face["normal"]
    if np.dot(geom["h"] - face["centre"], normal) < 0:
        normal = -normal
    for candidate in (geom["parent"]["centre"] - face["centre"] if geom["parent"] else None,
                      geom["h"] - face["centre"],
                      face["xyz"][0] - face["centre"]):
        if candidate is None:
            continue
        e1 = candidate - np.dot(candidate, normal) * normal
        if np.linalg.norm(e1) > 0.5:
            break
    e1 = e1 / np.linalg.norm(e1)
    tilt = np.deg2rad(55.0)
    return e1, np.cos(tilt) * np.cross(normal, e1) + np.sin(tilt) * normal, normal


def cloud_atoms(residues, target):
    """The heavy atoms the model is handed for the target residue, as (resi, name) pairs.

    The model's own rule (``deepshift.atom_cloud``): every heavy atom of each residue with
    any heavy atom within the cutoff of the target residue. Figure 1 counts the same list.
    """
    from deepshift.atom_cloud import FINAL_ATOM_CLOUD_RECIPE, target_cloud
    from deepshift.structure import parse_structure
    assert FINAL_ATOM_CLOUD_RECIPE.heavy_cutoff == CUTOFF
    cloud = target_cloud(parse_structure(STRUCTURE[0], STRUCTURE[1], keep_nonstandard=True),
                         int(target.id[1]))
    return [[int(resi), name] for resi, name, _point in cloud]


def ring_plan(run: Path):
    """The shielded-proton panel: which structure, which residues, and the camera."""
    cell = most_shielded(run)
    geom = shielding_pair(chain_residues(cell["path"], cell["chain"]), cell)
    e1, axis_y, _normal = view_frame(geom)
    # PyMOL's view matrix is the camera basis in model space, read column by column, so
    # the rows of [e1; up; out] go in transposed.
    rot = np.vstack([e1, axis_y, np.cross(e1, axis_y)]).T.ravel().tolist()
    centre = geom["face"]["centre"].tolist()
    # zoom() keeps the rotation and resets the distance, so the placeholder camera
    # distance below only has to be sane, not correct.
    view = [*rot, 0.0, 0.0, -60.0, *centre, 40.0, 80.0, -20.0]
    residues = [int(geom["face"]["name"][3:]), int(cell["resnum"])]
    return cell, geom, {
        "name": "ring_pair", "kind": "ring", "pdb": str(cell["path"]),
        "chain": cell["chain"], "state": 1, "residues": residues,
        "proton": [int(cell["resnum"]), cell["atom"]],
        "ring_centre": centre, "view": view, "size": SIZE,
        "out": str(RENDERS / "ring_pair.png"),
    }


PANELS = ("structure", "atom_cloud", "ring_pair")


def build(run: Path, pymol: str, only=None):
    """Render every panel, or only the named ones; the others keep their manifest
    records and images untouched (re-rendering is not bit-identical)."""
    want = set(only or PANELS)
    residues = chain_residues(*STRUCTURE)
    target = residues[len(residues) // 2]

    renders = []
    if "structure" in want:
        renders.append({"name": "structure", "kind": "cartoon", "pdb": str(STRUCTURE[0]),
                        "chain": STRUCTURE[1], "size": SIZE,
                        "out": str(RENDERS / "structure.png")})
    if "atom_cloud" in want:
        renders.append({"name": "atom_cloud", "kind": "cloud", "pdb": str(STRUCTURE[0]),
                        "chain": STRUCTURE[1], "target": int(target.id[1]),
                        "atoms": cloud_atoms(residues, target), "size": SIZE,
                        "out": str(RENDERS / "atom_cloud.png")})
    if "ring_pair" in want:
        cell, geom, ring = ring_plan(run)
        renders.append(ring)
    plan = {"colors": COLORS, "measured": str(RENDERS / "measured.json"),
            "renders": renders}
    by_name = {r["name"]: r for r in renders}

    RENDERS.mkdir(parents=True, exist_ok=True)
    # The plan names absolute paths, so it is written outside the repository: a tracked
    # file that spells out someone's home directory fails tests/test_release_paths.py,
    # and rightly -- the manifest below is the record that ships.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(plan, stream, indent=1)
        plan_path = Path(stream.name)
    result = subprocess.run([pymol, "-cq", str(HERE / "_pymol_render.py"), "--",
                             str(plan_path)], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0 or not all(Path(r["out"]).exists() for r in plan["renders"]):
        raise SystemExit(f"pymol failed ({result.returncode}):\n{result.stderr[-2000:]}")

    # Crop each panel to what it actually drew. A ray-traced frame has whatever margin
    # the camera left; the card should be filled by the molecule, not by that margin.
    from PIL import Image
    for spec in plan["renders"]:
        image = Image.open(spec["out"]).convert("RGBA")
        box = image.split()[3].point(lambda a: 255 if a > 6 else 0).getbbox()
        pad = int(0.02 * max(image.size))
        image.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                    min(box[2] + pad, image.width),
                    min(box[3] + pad, image.height))).save(spec["out"])

    measured = json.loads((RENDERS / "measured.json").read_text())
    (RENDERS / "measured.json").unlink()   # folded into the manifest below
    plan_path.unlink()
    # pymol -d echoes the command it ran before its output, so take the last line.
    version = subprocess.run([pymol, "-cq", "-d", "print(cmd.get_version()[0])"],
                             capture_output=True, text=True).stdout.strip().splitlines()[-1]
    record = json.loads(MANIFEST.read_text()) if only and MANIFEST.is_file() else {}
    record.update({
        "renderer": f"PyMOL {version}, ray traced, ambient occlusion and depth fog",
        "why_prerendered": "the figures stay pure matplotlib; only this step needs PyMOL",
        "reproducibility": "ambient occlusion is sampled, so re-rendering is not "
                           "bit-identical: two runs of this script differ on ~0.01% of "
                           "pixels by at most 2 of 255."})
    if "structure" in want:
        record["structure"] = {"file": STRUCTURE[0].name, "chain": STRUCTURE[1],
                               **measured["structure"]}
    if "atom_cloud" in want:
        record["atom_cloud"] = {"target": f"{target.get_resname()}{target.id[1]}",
                                "cutoff_angstrom": CUTOFF,
                                "rule": "every heavy atom of each residue with any heavy "
                                        "atom within the cutoff of the target residue "
                                        "(deepshift.atom_cloud)",
                                "n_atoms": len(by_name["atom_cloud"]["atoms"]),
                                "atoms": by_name["atom_cloud"]["atoms"],
                                **measured["atom_cloud"]}
    if "ring_pair" in want:
        record["ring_pair"] = {
            "accession": cell["accession"], "chain": cell["chain"], "state": 1,
            "proton": f"{cell['residue']}{cell['resnum']} {cell['atom']}",
            "shielding_ring": geom["face"]["name"],
            "distance_to_ring_centre_angstrom": round(geom["r"], 3),
            "angle_off_ring_normal_deg": round(geom["theta_deg"], 1),
            "ring_ppm": round(cell["ring_ppm"], 4),
            "why": "the largest computed ring-current field in the benchmark"}
    record["images"] = {**record.get("images", {}),
                        **{r["name"]: {"file": Path(r["out"]).name,
                                       "pixels": list(Image.open(r["out"]).size)}
                           for r in plan["renders"]}}
    MANIFEST.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {MANIFEST}")
    for name, record in json.loads(MANIFEST.read_text())["images"].items():
        print(f"  {name:16} {record['file']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pymol", default=os.environ.get("PYMOL") or shutil.which("pymol"))
    parser.add_argument("--only", nargs="+", choices=PANELS,
                        help="render these panels and keep the others as recorded")
    args = parser.parse_args()
    if not args.pymol:
        raise SystemExit("no PyMOL: pass --pymol, set $PYMOL, or put pymol on the path")
    sys.exit(build(frozen.RUN_ROOT, args.pymol, args.only))
