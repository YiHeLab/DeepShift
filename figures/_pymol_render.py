"""Execute a render plan inside PyMOL. Run by ``render_structures.py``, not by hand.

PyMOL ships its own interpreter, so this file may not import ``deepshift`` or anything
else from the figure environment: everything it needs to know -- which atoms, which
residue, which view -- is decided by the planner and arrives in one JSON file. That split
is deliberate. The chemistry is chosen where the chemistry lives, and this file only draws.

    pymol -cq _pymol_render.py -- plan.json
"""

import json
import os
import pathlib
import sys

from pymol import cmd

# Ray tracing settings shared by every panel. Ambient occlusion plus depth fog is what
# gives a small panel its solidity; hard shadows at this size only add noise.
COMMON = {
    "ray_opaque_background": 0,
    "antialias": 2,
    "ray_trace_mode": 0,
    "ray_shadow": 0,
    "ambient_occlusion_mode": 1,
    "ambient_occlusion_scale": 12.0,
    "ambient_occlusion_smooth": 12,
    "ambient": 0.18,
    "direct": 0.42,
    "reflect": 0.38,
    "specular": 0.22,
    "shininess": 45,
    "spec_power": 200,
    "ray_trace_fog": 1,
    "ray_trace_fog_start": 0.58,
    "depth_cue": 1,
    "fog_start": 0.40,
    "cartoon_fancy_helices": 1,
    "cartoon_smooth_loops": 1,
    "cartoon_transparency": 0.0,
    "stick_radius": 0.115,
    "sphere_scale": 0.20,
    "orthoscopic": 0,
}


def _palette(colors):
    for name, hexcode in colors.items():
        r, g, b = (int(hexcode[i:i + 2], 16) / 255 for i in (1, 3, 5))
        cmd.set_color(f"ds_{name}", [r, g, b])


def _atom_selection(obj, chain, atoms):
    """One selection naming exactly the atoms the plan lists, and nothing else."""
    terms = " or ".join(f"(resi {resi} and name {name})" for resi, name in atoms)
    cmd.select("plan_sel", f"{obj} and chain {chain} and ({terms})")
    return "plan_sel"


def cartoon(spec, obj):
    cmd.hide("everything")
    cmd.show("cartoon", f"{obj} and polymer and chain {spec['chain']}")
    cmd.color("ds_loop", obj)
    cmd.color("ds_helix", f"{obj} and ss H")
    cmd.color("ds_sheet", f"{obj} and ss S")
    cmd.orient(f"{obj} and chain {spec['chain']}")
    # orient lays the longest axis across the page and the cards are portrait, so stand
    # the fold up: a landscape render would leave half of its card empty.
    cmd.turn("z", 90)
    cmd.zoom(f"{obj} and chain {spec['chain']}", buffer=0.2, complete=1)
    chain = f"{obj} and chain {spec['chain']} and name CA"
    return {"residues": cmd.count_atoms(chain),
            "helix": cmd.count_atoms(f"{chain} and ss H"),
            "strand": cmd.count_atoms(f"{chain} and ss S"),
            "secondary_structure": "assigned by PyMOL from the coordinates"}


def cloud(spec, obj):
    cmd.hide("everything")
    sel = _atom_selection(obj, spec["chain"], spec["atoms"])
    cmd.show("sticks", sel)
    cmd.show("spheres", sel)
    cmd.color("ds_shell", f"{sel} and elem C")
    cmd.color("ds_nitrogen", f"{sel} and elem N")
    cmd.color("ds_oxygen", f"{sel} and elem O")
    cmd.color("ds_sulfur", f"{sel} and elem S")
    cmd.select("plan_target", f"{obj} and chain {spec['chain']} and resi {spec['target']}")
    cmd.color("ds_accent", "plan_target and elem C")
    cmd.set("sphere_scale", 0.38, "plan_target")
    cmd.orient(sel)
    # complete=1 uses the exact bounding box; the fast one clips a cloud this wide, and
    # the buffer covers the sphere radii the box does not know about.
    cmd.zoom(sel, buffer=0.7, complete=1)
    return {"atoms_drawn": cmd.count_atoms(sel),
            "target_atoms": cmd.count_atoms("plan_target")}


def ring(spec, obj):
    """The shielded proton over the ring that shields it, seen from the plan's frame."""
    cmd.hide("everything")
    sel = " or ".join(f"(resi {r})" for r in spec["residues"])
    cmd.select("plan_pair", f"{obj} and chain {spec['chain']} and ({sel}) and not name O+OXT")
    cmd.show("sticks", "plan_pair")
    cmd.color("ds_carbon", "plan_pair and elem C")
    cmd.color("ds_nitrogen", "plan_pair and elem N")
    cmd.color("ds_hydrogen", "plan_pair and elem H")
    proton = f"{obj} and chain {spec['chain']} and resi {spec['proton'][0]} and name {spec['proton'][1]}"
    # The proton's own residue is context; the ring that shields it and the proton
    # itself are the subject, so the rest is faded rather than removed.
    cmd.set("stick_transparency", 0.32,
            f"plan_pair and resi {spec['proton'][0]}")
    cmd.select("plan_h", proton)
    cmd.show("spheres", "plan_h")
    cmd.color("ds_accent", "plan_h")
    cmd.set("sphere_scale", 0.30, "plan_h")
    cmd.pseudoatom("ring_centre", pos=spec["ring_centre"])
    cmd.distance("plan_axis", "plan_h", "ring_centre")
    cmd.hide("everything", "ring_centre")   # the marker is the dashed line, not a cross
    cmd.hide("labels", "plan_axis")
    cmd.color("ds_faint", "plan_axis")
    cmd.set("dash_gap", 0.28)
    cmd.set("dash_radius", 0.035)
    cmd.set_view(spec["view"])
    cmd.zoom("plan_pair or plan_h or ring_centre", buffer=0.5, complete=1)
    return {"atoms_drawn": cmd.count_atoms("plan_pair")}


KINDS = {"cartoon": cartoon, "cloud": cloud, "ring": ring}


def main(plan_path):
    plan = json.loads(pathlib.Path(plan_path).read_text())
    for key, value in COMMON.items():
        cmd.set(key, value)
    _palette(plan["colors"])
    measured = {}
    for spec in plan["renders"]:
        cmd.delete("all")
        cmd.load(spec["pdb"], "mol")
        cmd.remove("mol and not alt ''+A")
        cmd.alter("mol", "alt=''")
        if spec.get("state"):
            cmd.create("mol1", "mol", spec["state"], 1)
            cmd.delete("mol")
            cmd.set_name("mol1", "mol")
        measured[spec["name"]] = KINDS[spec["kind"]](spec, "mol") or {}
        cmd.png(spec["out"], width=spec["size"][0], height=spec["size"][1], dpi=400, ray=1)
        print(f"rendered {spec['name']} -> {spec['out']}")
    with open(plan["measured"], "w") as stream:
        json.dump(measured, stream, indent=1)


# PyMOL executes this file with its own __name__, so there is no __main__ guard: the
# plan is the last argument after the "--" separator.
try:
    main(sys.argv[-1])
except Exception:
    import traceback
    traceback.print_exc()
    os._exit(1)
