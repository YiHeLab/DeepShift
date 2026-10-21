"""The atom stage on a real structure: partner suppression, the closed recipe, the cap."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from deepshift import structure
from deepshift.data import atom_census, atoms

ALANINE = (("N", -1.2, 0.8, 0.0), ("CA", 0.0, 0.0, 0.0), ("C", 1.2, 0.8, 0.0),
           ("O", 1.2, 2.0, 0.0), ("CB", 0.0, -1.5, 0.0))
SHIFT_COLS = ["ca_shift", "cb_shift"]
MATERIALIZATION = "mz_" + "0" * 64
LABEL_SET = "ls_" + "1" * 64


def _atom_line(serial, name, residue, chain, seq, x, y, z, element):
    return (f"ATOM  {serial:>5} {name:^4} {residue:>3} {chain}{seq:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{20.0:>6.2f}          {element:>2}\n")


PARTNER_RESIDUES = 12


def write_structure(path: Path, *, target_residues=3, partner=True, dense=False):
    """Chain A along x at 3.8 A per residue; chain B parallel, 6 A away, inside the
    8 A contact shell and long enough to meet the sample-partner alignment floor.
    ``dense`` packs 130 target residues into one 8 A ball instead."""
    lines, serial = [], 1
    if dense:
        rng = np.random.default_rng(0)
        centers = rng.uniform(-2.0, 2.0, size=(130, 3))
    else:
        centers = [(3.8 * index, 0.0, 0.0) for index in range(target_residues)]
    for seq, (cx, cy, cz) in enumerate(centers, start=1):
        for name, dx, dy, dz in ALANINE:
            lines.append(_atom_line(serial, name, "ALA", "A", seq,
                                    cx + dx, cy + dy, cz + dz, name[0]))
            serial += 1
    if partner:
        for seq in range(1, PARTNER_RESIDUES + 1):
            for name, dx, dy, dz in ALANINE:
                lines.append(_atom_line(serial, name, "ALA", "B", seq,
                                        3.8 * (seq - 1) + dx, 6.0 + dy, dz, name[0]))
                serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))
    return len(centers)


def write_fold_cache(fold_dir: Path, n_residues: int):
    """One materialization of chain A: the structural arrays the atom stage binds to."""
    structural = fold_dir / "structural"
    structural.mkdir(parents=True)
    (fold_dir / "config.json").write_text(json.dumps({
        "shift_cols": SHIFT_COLS, "source_names": ["af", "nmr", "xtal"]}))
    np.save(fold_dir / "samples.npy", np.arange(n_residues, dtype=np.int64))
    rows = [str(g) for g in range(n_residues)]
    (structural / "bmrb_mapping.json").write_text(json.dumps(dict.fromkeys(rows, "7")))
    (structural / "global_to_resid.json").write_text(json.dumps({g: int(g) + 1 for g in rows}))
    (structural / "global_to_struct_resid.json").write_text(
        json.dumps({g: int(g) + 1 for g in rows}))
    (structural / "materialization_ids.json").write_text(json.dumps([MATERIALIZATION]))
    (structural / "id_maps.json").write_text(json.dumps({
        "label_set_id": dict.fromkeys(rows, LABEL_SET),
        "label_residue_id": {g: f"lr_{g}" for g in rows},
        "source": dict.fromkeys(rows, "nmr"),
        "logical_entity_id": dict.fromkeys(rows, "le_x"),
    }))
    np.save(structural / "residue_idx.npy", np.zeros(n_residues, dtype=np.int32))
    np.save(structural / "materialization_idx.npy", np.zeros(n_residues, dtype=np.int32))
    np.save(structural / "source_idx.npy", np.ones(n_residues, dtype=np.int8))
    np.save(structural / "shift_mask.npy", np.ones((n_residues, len(SHIFT_COLS)), dtype=bool))


def build_fixture(root: Path, *, partner=True, dense=False, structure_name="complex.pdb",
                  log_path=None, suppress=None, partner_in_sample=True):
    """Everything one atom-stage invocation reads. Returns the argv for ``atoms.main``."""
    store = root / "store"
    store.mkdir(parents=True, exist_ok=True)
    structure = store / structure_name
    n_residues = write_structure(structure, partner=partner, dense=dense)
    named = Path(log_path) if log_path else structure
    if named != structure:
        named.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(structure, named)
    fold_dir = root / "fold_1"
    write_fold_cache(fold_dir, n_residues if dense else 3)
    log = root / "structure_build_log_nmr.csv"
    with log.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label_set_id", "bmrb_id",
                                                    "materialization_id", "pdb_path",
                                                    "chain_id", "status"])
        writer.writeheader()
        writer.writerow({"label_set_id": LABEL_SET, "bmrb_id": "7",
                         "materialization_id": MATERIALIZATION, "pdb_path": str(named),
                         "chain_id": "A", "status": "success"})
    assembly = root / "sample_assembly.json"
    partner_sequence = "A" * PARTNER_RESIDUES
    assembly.write_text(json.dumps({"label_sets": {LABEL_SET: {
        "target_present_in_assembly": True,
        "partners": ([{"entity_id": "2", "copies": 1, "alignment_sequence": partner_sequence}]
                     if partner_in_sample else []),
    }}}))
    suppression = root / "partner_chain_suppression.json"
    suppression.write_text(json.dumps(suppress or {}))
    argv = ["--cache_dir", str(fold_dir),
            "--source_build_logs", f"nmr:{log}:experimental",
            "--pdb_search_dirs", str(store),
            "--leaky_partner_chains", str(suppression),
            "--sample_assembly", str(assembly)]
    return argv, fold_dir, log, assembly, suppression


def _atom_counts(fold_dir):
    return np.load(fold_dir / "atom" / "atom_count.npy").tolist()


def test_a_symlinked_structure_is_suppressed_under_the_build_log_name(tmp_path):
    """The suppression map and the atom stage key on the build log's basename. A build log
    that names a symlink whose target is called something else must still suppress, and
    the applied count must say so."""
    key = "1ABC.pdb\tA"
    argv, fold_dir, *_ = build_fixture(
        tmp_path / "suppressed", structure_name="complex_target.pdb",
        log_path=tmp_path / "suppressed" / "pdbs" / "1ABC.pdb", suppress={key: ["B"]})
    atoms.main(argv)
    suppressed = _atom_counts(fold_dir)
    config = json.loads((fold_dir / "atom" / "config.json").read_text())
    assert config["suppression_keys_applied"] == 1

    argv, fold_dir, *_ = build_fixture(
        tmp_path / "open", structure_name="complex_target.pdb",
        log_path=tmp_path / "open" / "pdbs" / "1ABC.pdb")
    atoms.main(argv)
    with_partner = _atom_counts(fold_dir)
    assert json.loads((fold_dir / "atom" / "config.json").read_text())[
        "suppression_keys_applied"] == 0
    assert all(a < b for a, b in zip(suppressed, with_partner))
    assert suppressed == [5 * 3] * 3


def test_a_map_that_addresses_this_fold_but_matches_nothing_refuses(tmp_path):
    _argv, fold_dir, log, *_ = build_fixture(tmp_path / "miss")
    binding = atoms._load_cache_binding(argparse.Namespace(
        cache_dir=str(fold_dir), m_atoms=512, limit=None,
        source_build_logs=[f"nmr:{log}:experimental"]))
    arrays = atoms._allocate_outputs(binding, 512)
    key = atoms.suppression_key(*binding.log[MATERIALIZATION])
    atoms._check_suppression_applied(arrays, binding, {})
    atoms._check_suppression_applied(arrays, binding, {"elsewhere.pdb\tA": ["B"]})
    with pytest.raises(SystemExit, match="none was applied"):
        atoms._check_suppression_applied(arrays, binding, {key: ["B"]})
    arrays.suppression_keys_applied.add(key)
    atoms._check_suppression_applied(arrays, binding, {key: ["B"]})


def test_the_atom_stage_refuses_to_build_without_the_assembly_map(tmp_path, capsys):
    """A cloud without the sample-composition authority would be partner-free by omission;
    the stage has no such mode."""
    for fold in ("a", "b"):
        argv, *_ = build_fixture(tmp_path / fold)
        without_map = [value for index, value in enumerate(argv)
                       if argv[index - 1] != "--sample_assembly" and value != "--sample_assembly"]
        with pytest.raises(SystemExit) as refused:
            atoms.main(without_map)
        assert refused.value.code == 2
        assert "--sample_assembly" in capsys.readouterr().err


def test_a_cloud_over_the_cap_keeps_its_nearest_atoms_and_is_counted(tmp_path):
    """130 residues inside one 8 A ball give every center 650 atoms. The census and the
    atom stage warn, the cap keeps 512 atoms per row, and the truncated rows are counted
    in the census, in atom/config.json and in atom_truncated.npy."""
    argv, fold_dir, log, assembly, suppression = build_fixture(
        tmp_path / "dense", dense=True, partner=False)
    with pytest.warns(RuntimeWarning, match="exceeds M=512: 130 of 130 centers"):
        m_atoms, stats = atom_census.run_census(
            [f"nmr:{log}:experimental"], sample_assembly=str(assembly),
            suppressed_partner_chains=str(suppression),
            search_dirs=[str(tmp_path / "dense" / "store")])
    assert (m_atoms, stats["precap_max"], stats["n_centers_over_cap"]) == (512, 650, 130)

    with pytest.warns(RuntimeWarning, match="130 of 130 rows exceed M=512"):
        atoms.main(argv)
    config = json.loads((fold_dir / "atom" / "config.json").read_text())
    assert (config["n_truncated"], config["precap_max"]) == (130, 650)
    assert np.load(fold_dir / "atom" / "atom_truncated.npy").all()
    assert set(_atom_counts(fold_dir)) == {512}
    xyz = np.load(fold_dir / "atom" / "atom_xyz.npy").astype(np.float64)
    offsets = np.load(fold_dir / "atom" / "atom_offset.npy")
    residues = structure.parse_structure(tmp_path / "dense" / "store" / "complex.pdb", "A")
    center_code = config["recipe"]["center_offset_code"]
    # After the target's own atoms, the kept neighbours are the nearest by distance to the
    # target CA; the dropped ones are the farthest.
    for row, residue_id in enumerate(sorted(residues)):
        ca = residues[residue_id]["atoms"]["CA"]
        others = np.asarray([coordinate for other, info in residues.items() if other != residue_id
                             for coordinate in info["atoms"].values()])
        every_distance = np.sort(np.linalg.norm(others - ca, axis=1))
        kept = np.sort(np.linalg.norm(xyz[row][offsets[row] != center_code], axis=1))
        assert len(kept) == 512 - len(ALANINE) and len(every_distance) > len(kept)
        assert np.allclose(kept, every_distance[:len(kept)], atol=2e-2)


def _altloc_line(serial, name, residue, chain, seq, altloc, x, y, z, occupancy, element,
                 record="ATOM"):
    return (f"{record:<6}{serial:>5} {name:^4}{altloc}{residue:>3} {chain}{seq:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{20.0:>6.2f}          "
            f"{element:>2}\n")


def test_perturber_exclusion_resolves_alternate_locations_by_occupancy(tmp_path):
    """One iron ion with two alternate positions: A at 0.30 occupancy far from residue 1,
    B at 0.70 occupancy beside it. The structure module keeps the altloc with the higher
    summed occupancy, so residue 1 sits inside the paramagnetic cutoff and is excluded."""
    from deepshift.data import exclusion, structures

    lines, serial = [], 1
    for seq, x in ((1, 0.0), (2, 40.0)):
        for name, dx, dy, dz in ALANINE:
            lines.append(_altloc_line(serial, name, "ALA", "A", seq, " ", x + dx, dy, dz, 1.0,
                                      name[0]))
            serial += 1
    lines.append(_altloc_line(serial, "FE", "FE", "X", 1, "A", 20.0, 0.0, 0.0, 0.30, "FE",
                              record="HETATM"))
    lines.append(_altloc_line(serial + 1, "FE", "FE", "X", 1, "B", 3.0, 0.0, 0.0, 0.70, "FE",
                              record="HETATM"))
    lines.append("END\n")
    pdb = tmp_path / "altloc.pdb"
    pdb.write_text("".join(lines))
    aa = structures.model_to_aa(structures.select_best_nmr_model(str(pdb), "A"))
    entities = exclusion.nonstandard_entities(str(pdb), 0)
    assert [(entity.resname, entity.heavy_atoms[0]["xyz"][0]) for entity in entities] == \
        [("FE", 3.0)]
    assert exclusion.perturber_excluded_rids(str(pdb), "A", aa) == {1}


