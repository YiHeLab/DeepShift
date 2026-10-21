"""Golden values for the offline replay: cache bytes, normalization, flags, folds.

Each golden here is a number the build must reproduce exactly on a fresh machine, so a
rebuild that silently changes rereferencing, masking, or the S_NUC fit is caught here
and not in a model comparison months later.
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from deepshift import quality_control as flags
from deepshift.atom_cloud import FINAL_ATOM_CLOUD_RECIPE
from deepshift.data import build, cache, folds, norm, reref_utils
from deepshift.data.cache_columns import parse_shift_columns, require_release_shift_columns
from deepshift.data.config import ATOM_TO_IDX
from deepshift.domain import NUCLEI, SHIFT_COLUMNS, STANDARD_RESIDUES

SHIFT_COLS = ["ca_shift", "n_shift"]


def _synthetic_id(prefix, seed):
    """A well-formed synthetic identity. Every stage refuses anything that is not
    ``<prefix><sha256>``, so a test fixture cannot use readable stand-ins."""
    return prefix + hashlib.sha256(seed.encode()).hexdigest()


def _entity_frame():
    """Two label sets, one of them with a gap row, in the entity grain the cache reads."""
    rows = []
    for tag, bmrb, source, residues in (
        ("10", "10", "af", [(1, "ALA", 52.25, 121.0, "match"),
                            (2, "GLY", np.nan, 109.0, "match")]),
        ("20", "20", "nmr", [(1, "SER", 58.5, 116.0, "gap_in_structure")]),
    ):
        for index, comp, ca, n, mismatch in residues:
            rows.append({
                "materialization_id": _synthetic_id("mz_", tag),
                "sample_id": _synthetic_id("sa_", tag),
                "label_set_id": _synthetic_id("ls_", tag),
                "logical_entity_id": _synthetic_id("le_", tag),
                "structure_instance_id": _synthetic_id("si_", tag),
                "label_residue_id": _synthetic_id("lr_", f"{tag}:{index}"),
                "struct_res_id": f"sr_{index}",
                "bmrb_id": bmrb, "source": source, "comp_index_id": index,
                "comp_id": comp, "residue_code": comp, "mismatch_type": mismatch,
                "ca_shift": ca, "n_shift": n,
            })
    return pd.DataFrame(rows)


def test_cache_rereference_and_mask_golden(tmp_path):
    """A rereference offset is added to the stored ppm; a flagged cell and a gap row are
    masked out and never zeroed away. Offsets key on label_set_id and flags on
    label_residue_id, so a sibling entity can never inherit either."""
    output = tmp_path / "fold_1"
    stats = {"ca_shift": {"mean": 55.0, "std": 3.0},
             "n_shift": {"mean": 115.0, "std": 5.0},
             "snuc": {"ca_shift": 2.0, "n_shift": 3.0}}
    cache.build_cache_for_fold_entity(
        _entity_frame(), 1, SHIFT_COLS, stats, str(output),
        reref_offsets={_synthetic_id("ls_", "10"): {"ca_shift": 0.5}},
        flagged={_synthetic_id("lr_", "10:2"): ["n_shift"]},
        source_names=["af", "nmr"], rebuild_root=str(tmp_path))

    shifts = np.load(output / "structural" / "shifts.npy")
    mask = np.load(output / "structural" / "shift_mask.npy")
    expected_shifts = np.array(
        [[52.75, 121.0], [0.0, 109.0], [58.5, 116.0]], dtype=np.float32)
    expected_mask = np.array(
        [[True, True], [False, False], [False, False]], dtype=bool)
    assert shifts.tobytes() == expected_shifts.tobytes()
    assert mask.tobytes() == expected_mask.tobytes()
    # The fence must be applied through the label-residue map, and be readable back.
    identity = json.loads((output / "structural" / "id_maps.json").read_text())
    assert identity["label_residue_id"]["1"] == _synthetic_id("lr_", "10:2")


def test_snuc_assembly_golden():
    """S_NUC is a robust (MAD) scale of the secondary shift, fitted on train folds only.
    The published constant is what makes ppm recoverable from z, so it is pinned."""
    values = np.array([51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54.0, 54.5, 55.0, 55.5])
    frame = pd.DataFrame({
        "bmrb_id": [str(i) for i in range(10)],
        "residue_id": np.arange(10),
        "residue_code": ["ALA"] * 10,
        "ca_shift": values,
    })
    stats = cache.compute_normalization_stats(frame, ["ca_shift"])
    secondary = values - 53.128                      # the ALA CA fixed center
    expected = 1.4826 * np.median(np.abs(secondary - np.median(secondary)))
    assert stats["snuc"] == {"ca_shift": expected}
    assert stats["snuc_counts"] == {"ca_shift": 10}
    assert stats["normalization"] == "secondary_snuc_train_folds_v1"


def test_snuc_is_the_scale_of_the_secondary_shift_not_of_raw_ppm():
    """The half the golden above cannot see: MAD is translation invariant, so over one
    residue type ``MAD(values - center)`` and ``MAD(values)`` are the same number and the
    word 'secondary' is unfalsifiable there -- a fit with no centring at all reproduces
    the golden. (Which is also why the golden's stated 53.128 stands while
    ``get_center('ALA', 'ca_shift')`` is the 52.5 random-coil value: the constant never
    enters the answer.)

    Two residue types whose CA centers differ by 7.4 ppm separate them at once: a raw-ppm
    fit reads the between-type spread as width and returns 5.19 instead of 1.85, which is
    a ~2.8x wrong scale on every denormalized prediction for this nucleus.
    """
    ala = np.array([51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54.0, 54.5, 55.0, 55.5])
    gly = np.array([44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5])
    frame = pd.DataFrame({
        "bmrb_id": [str(i) for i in range(20)],
        "residue_id": np.arange(20),
        "residue_code": ["ALA"] * 10 + ["GLY"] * 10,
        "ca_shift": np.concatenate([ala, gly]),
    })
    stats = cache.compute_normalization_stats(frame, ["ca_shift"])

    secondary = np.concatenate([ala - norm.get_center("ALA", "ca_shift"),
                                gly - norm.get_center("GLY", "ca_shift")])
    expected = 1.4826 * np.median(np.abs(secondary - np.median(secondary)))
    assert stats["snuc"]["ca_shift"] == pytest.approx(expected)
    assert stats["snuc_counts"] == {"ca_shift": 20}

    raw = np.concatenate([ala, gly])
    raw_scale = 1.4826 * np.median(np.abs(raw - np.median(raw)))
    assert raw_scale > 2.5 * expected, "the two fits must be distinguishable here"


def test_residue_realignment_golden():
    """The offset search that aligns a deposition's numbering to a structure's."""
    body = {1: ("A", {}), 2: ("G", {}), 3: ("S", {}), 4: ("T", {}), 5: ("V", {})}
    predictions = {item + 7: value for item, value in body.items()}
    assert reref_utils.best_shift(body, predictions) == (7, 1.0)




def test_range_flags_are_derived_from_cache_values(tmp_path):
    """The QC fence is derived from the cache's own ppm against a residue-specific
    range, never a frozen list of ids -- a frozen list silently flags nothing once a
    rebuild renumbers the rows."""
    fold = tmp_path / "cache" / "fold_1"
    structural = fold / "structural"
    structural.mkdir(parents=True)
    (fold / "config.json").write_text(json.dumps({
        "shift_storage": "ppm_float32", "shift_cols": ["ca_shift"]}))
    np.save(structural / "shifts.npy", np.array([[52.0], [99.0], [101.0]], np.float32))
    np.save(structural / "shift_mask.npy", np.array([[True], [True], [True]]))
    np.save(structural / "ambiguity.npy", np.array([[0], [0], [9]], np.int8))
    np.save(structural / "residue_idx.npy", np.array([0, 0, 0], np.int32))
    in_range = _synthetic_id("lr_", "a")
    out_of_range = _synthetic_id("lr_", "b")
    ambiguity_set = _synthetic_id("lr_", "ambiguity-set")
    (structural / "global_to_label_residue.json").write_text(
        json.dumps({"0": in_range, "1": out_of_range, "2": ambiguity_set}))
    cutoffs = {"ALA:ca_shift": {"lo": 45.0, "hi": 65.0}}
    result, report = flags.derive_p100_flags(tmp_path / "cache", cutoffs)
    assert result == {out_of_range: ["ca_shift"]}
    assert report["schema_key"] == "label_residue_id"
    assert report["instance_hits"] == report["unique_flags"] == 1
    assert report["ambiguity_instances_out_of_scope"] == 1

    cutoff_file = tmp_path / "cutoffs.json"
    cutoff_file.write_text(json.dumps(cutoffs))
    assert flags.derive_p100_flags(tmp_path / "cache", cutoff_file)[0] == result

    # A cache that is not stored in ppm cannot be range-fenced at all.
    config = json.loads((fold / "config.json").read_text())
    config["shift_storage"] = "z_scored"
    (fold / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="ppm_float32"):
        flags.derive_p100_flags(tmp_path / "cache", cutoffs)


def test_range_flags_reject_an_observed_population_without_a_cutoff(tmp_path):
    fold = tmp_path / "cache" / "fold_1"
    structural = fold / "structural"
    structural.mkdir(parents=True)
    (fold / "config.json").write_text(json.dumps({
        "shift_storage": "ppm_float32", "shift_cols": ["ca_shift"]}))
    np.save(structural / "shifts.npy", np.array([[52.0]], np.float32))
    np.save(structural / "shift_mask.npy", np.array([[True]]))
    np.save(structural / "ambiguity.npy", np.array([[0]], np.int8))
    np.save(structural / "residue_idx.npy", np.array([0], np.int32))
    (structural / "global_to_label_residue.json").write_text(
        json.dumps({"0": _synthetic_id("lr_", "a")}))
    with pytest.raises(ValueError, match="ALA:ca_shift"):
        flags.derive_p100_flags(tmp_path / "cache", {"GLY:ca_shift": {"lo": 40.0, "hi": 50.0}})


def test_structural_flags_can_use_independent_original_observations(tmp_path):
    """The structural rule reads the cache's own mask, so a cell already masked out is
    invisible to it; given an independent record of the original observations it reaches
    that cell and flags it."""
    fold = tmp_path / "cache" / "fold_0"
    structural = fold / "structural"
    atom = fold / "atom"
    structural.mkdir(parents=True)
    atom.mkdir()
    (atom / "config.json").write_text(json.dumps({
        "recipe": FINAL_ATOM_CLOUD_RECIPE.to_dict(),
    }))
    label_id = _synthetic_id("lr_", "structural-override")
    (fold / "config.json").write_text(json.dumps({
        "shift_storage": "ppm_float32",
        "shift_cols": ["cb_shift"],
        "source_names": ["af"],
        "provenance": {"fold_map_sha256": "a" * 64},
    }))
    (structural / "global_to_label_residue.json").write_text(
        json.dumps({"0": label_id})
    )
    # Oxidized by CB ppm, reduced by geometry (no neighbour SG).
    np.save(structural / "shifts.npy", np.array([[40.0]], dtype=np.float32))
    np.save(structural / "shift_mask.npy", np.array([[False]], dtype=bool))
    np.save(
        structural / "residue_idx.npy",
        np.array([STANDARD_RESIDUES.index("CYS")], dtype=np.int16),
    )
    np.save(structural / "source_idx.npy", np.array([0], dtype=np.int8))
    np.save(
        atom / "atom_type.npy",
        np.array([[ATOM_TO_IDX["SG"]]], dtype=np.int16),
    )
    np.save(
        atom / "atom_offset.npy",
        np.array([[FINAL_ATOM_CLOUD_RECIPE.center_offset_code]], dtype=np.int8),
    )
    np.save(atom / "atom_xyz.npy", np.zeros((1, 1, 3), dtype=np.float32))

    hidden, _ = flags.derive_structural_flags(tmp_path / "cache")
    restored, report = flags.derive_structural_flags(
        tmp_path / "cache",
        observation_mask_loader=lambda _fold: np.array([[True]], dtype=bool),
    )
    assert hidden == {}
    assert restored == {label_id: ["cb_shift"]}
    assert report["counts"]["disulfide"] == 1


def test_p100_and_structural_components_union_exact_cells():
    left, right = _synthetic_id("lr_", "left"), _synthetic_id("lr_", "right")
    result = flags.union_flags(
        {left: ["ca_shift"], right: ["n_shift"]},
        {right: ["ca_shift"]},
    )
    assert result == {left: ["ca_shift"], right: ["ca_shift", "n_shift"]}
    with pytest.raises(ValueError, match="unknown QC policy"):
        flags.derive_flags("p1OO")


def test_qc_policy_is_a_closed_set_of_choices():
    """The build has no selectable exclusion rule, and the fence is never a component."""
    parser = argparse.ArgumentParser()
    build.add_build_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--root", "r", "--qc-policy", "p100_structural"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--root", "r", "--also-qc-sensitivity", "p100"])
    for component in ("p100", "structural"):
        with pytest.raises(ValueError, match="unknown QC policy"):
            flags.derive_flags(component, cache="unused")
    with pytest.raises(ValueError, match="explicit fitted cutoff"):
        flags.derive_flags("p100_structural", cache="unused")


def test_p100_cache_build_requires_precomputed_flags(tmp_path):
    builder = _builder(
        tmp_path / "build",
        from_stage="cache",
        to_stage="cache",
        p100_cutoffs=str(tmp_path / "cutoffs.json"),
        flagged=None,
    )
    with pytest.raises(RuntimeError, match="--flagged"):
        builder.run()


def test_cache_science_does_not_discover_ambient_flags_or_tiny_stats(tmp_path,
                                                                    monkeypatch):
    """Production cache inputs must be visible in the command and its manifest.

    A ``$SHIFT_DATA_ROOT/qc_flags_p100.json`` picked up from the environment would
    change S_NUC for a command that names no fence, and a builder enabling the
    smoke-only 0.1 floor of its own accord would do the same. Either makes an
    apparently identical rebuild depend on ambient workstation state.
    """
    ambient = tmp_path / "ambient-flags.json"
    ambient.write_text(json.dumps({"lr_ambient": ["ca_shift"]}))
    monkeypatch.setenv("SHIFT_QC_FLAGS_P100", str(ambient))

    parser = argparse.ArgumentParser()
    cache.add_cache_arguments(parser)
    parsed = parser.parse_args(["--data_dir", str(tmp_path)])
    assert parsed.flagged is None
    assert parsed.tiny_stats_floor is False

    builder = _builder(tmp_path / "build", flagged=str(ambient))
    captured = {}

    def record(stage, module, *arguments, **kwargs):
        captured["arguments"] = [str(value) for value in arguments]
        captured["inputs"] = [str(value) for value in kwargs["inputs"]]

    builder.command = record
    builder.cache()
    assert "--flagged" in captured["arguments"]
    assert str(ambient) in captured["arguments"]
    assert "--tiny_stats_floor" not in captured["arguments"]


def test_existing_cache_requires_an_explicit_deletion_boundary(tmp_path):
    existing = tmp_path / "existing-cache"
    existing.mkdir()
    with pytest.raises(SystemExit, match="without an explicit --rebuild_root"):
        cache.assert_safe_cache_dir(existing)
    cache.assert_safe_cache_dir(existing, rebuild_root=tmp_path)


def test_fold_assignment_is_seeded_and_component_safe():
    """Whole components move together and the same inputs give the same map, whatever
    order they arrive in. (Adding a component is a different question -- see
    test_invariants.py.)"""
    members = {f"ls_{i:04d}": [f"ls_{i:04d}"] for i in range(1, 30)}
    members["ls_0001"] = ["ls_0001", "ls_0002"]        # one two-member component
    del members["ls_0002"]
    records = {f"ls_{i:04d}": {"bmrb_id": f"{17000 + i}", "n_shift_residues": 50 + i,
                               "sources": ["af"]} for i in range(1, 30)}
    first = folds.assign_labelset_folds(members, records, set())
    reversed_members = dict(reversed(list(members.items())))
    second = folds.assign_labelset_folds(reversed_members, records, set())
    assert first == second
    assert first["ls_0001"] == first["ls_0002"]        # component kept together
    assert set(first.values()) <= {0, 1, 2, 3, 4, 5, folds.HOLDOUT_FOLD}


def test_excluded_components_are_assigned_before_anything_else():
    members = {f"ls_{i:04d}": [f"ls_{i:04d}"] for i in range(1, 30)}
    records = {ls: {"bmrb_id": ls, "n_shift_residues": 60, "sources": ["af"]}
               for ls in members}
    assigned = folds.assign_labelset_folds(members, records, {"ls_0007"})
    assert assigned["ls_0007"] == folds.EXCLUDED_FOLD


def test_builder_scrubs_ambient_scientific_overrides(tmp_path, monkeypatch):
    """An ambient env var must never silently change a build's science. Only an explicit
    flag may set one, and then it is recorded in the child environment."""
    for name in build.AMBIENT_BUILD_OVERRIDES:
        monkeypatch.setenv(name, "ambient-value")
    args = argparse.Namespace(
        root=str(tmp_path / "build"), mmseqs_bin=None,
        shiftx_bin=None, shiftx_source=None, data_archive=None,
        split_seed=None, holdout_frac=None, resume=False, bmrb_ids=None,
        workers=1, folds=[1], from_stage="fetch", to_stage="fetch",
    )
    builder = build.Builder(args)
    for name in build.AMBIENT_BUILD_OVERRIDES:
        assert name not in builder.env

    args.split_seed = 123
    args.holdout_frac = 0.2
    explicit = build.Builder(args)
    assert explicit.env["SHIFT_SPLIT_SEED"] == "123"
    assert explicit.env["SHIFT_HOLDOUT_FRAC"] == "0.2"


# ---------------------------------------------------------------------------
# Declared build inputs: what is hashed must be what was asked for
# ---------------------------------------------------------------------------

def _builder(root, **overrides):
    fields = {
        "root": str(root), "mmseqs_bin": None, "shiftx_bin": None,
        "shiftx_source": None, "data_archive": None, "split_seed": None, "holdout_frac": None,
        "resume": False, "workers": 1, "folds": [1], "from_stage": "fetch", "to_stage": "fetch",
        "pdb_seqres_db": None, "pdb_method_cache": None, "pairs": None}
    return build.Builder(argparse.Namespace(**{**fields, **overrides}))


def test_repointing_a_declared_input_actually_repoints_it(tmp_path):
    """An existing build directory re-pointed at a different method/resolution cache
    stages the newly declared file. Honouring a declared input only where the
    destination is absent would keep the file already there and report success -- a
    stale-input run that looks like a clean one."""
    first, second = tmp_path / "method_a.json", tmp_path / "method_b.json"
    first.write_text(json.dumps({"1ABC": ["X-RAY", 1.5]}))
    second.write_text(json.dumps({"1ABC": ["X-RAY", 1.5], "2DEF": ["X-RAY", 2.0]}))
    seqres_a, seqres_b = tmp_path / "db_a", tmp_path / "db_b"
    for db in (seqres_a, seqres_b):
        db.mkdir()
    pairs_a, pairs_b = tmp_path / "pairs_a.csv", tmp_path / "pairs_b.csv"
    pairs_a.write_text("bmrb_id,accession\n1,1ABC\n")
    pairs_b.write_text("bmrb_id,accession\n1,1ABC\n2,2DEF\n")

    root = tmp_path / "build"
    _builder(root, pdb_method_cache=str(first), pdb_seqres_db=str(seqres_a),
             pairs=str(pairs_a))._prep_crystal_blast()
    staged = root / "data" / "crystal_blast" / "pdb_method_resolution.json"
    assert staged.resolve() == first.resolve()

    _builder(root, pdb_method_cache=str(second), pdb_seqres_db=str(seqres_b),
             pairs=str(pairs_b))._prep_crystal_blast()
    assert staged.resolve() == second.resolve()
    assert (root / "data" / "crystal_blast" / "pdb_seqres_db").resolve() == seqres_b.resolve()
    # The copied input refreshes on content, not on absence.
    assert (root / "data" / "pairs.csv").read_text() == pairs_b.read_text()


def test_builder_refuses_missing_inputs_before_launch(tmp_path, monkeypatch):
    builder = _builder(tmp_path / "build")
    launched = False

    def fake_run(*args, **kwargs):
        nonlocal launched
        launched = True

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="declared input"):
        builder.command(
            "fixture",
            "deepshift.data.fetch",
            outputs=[tmp_path / "output.json"],
            inputs=[tmp_path / "absent.json"],
        )
    assert launched is False


def test_builder_refuses_a_missing_output(tmp_path, monkeypatch):
    builder = _builder(tmp_path / "build")
    input_path = tmp_path / "input.json"
    input_path.write_text("{}\n")
    monkeypatch.setattr(build.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="declared output"):
        builder.command(
            "fixture",
            "deepshift.data.fetch",
            outputs=[tmp_path / "absent-output.json"],
            inputs=[input_path],
        )


def test_atom_vocabulary_is_the_frozen_88_token_alphabet():
    """Checkpoint embeddings index this table, so its size and order are frozen."""
    assert len(ATOM_TO_IDX) == 88
    assert ATOM_TO_IDX["C"] == 0 and ATOM_TO_IDX["CA"] == 1


# The 15 nuclei that extend the 49-nucleus set, with the residue carrying each.
ADDED_NUCLEI = {
    "TRP": ("NE1", "CZ2", "CZ3", "CE3", "CH2", "HZ2", "HZ3", "HH2"),
    "ARG": ("NE", "HH11", "HH12", "HH21", "HH22"),
    "THR": ("HG1",),
    "TYR": ("HH",),
}


def test_nucleus_vocabulary_is_the_64_predicted_nuclei():
    """The head width and the cache's label columns both come from here, and the order
    is what the cache builder's data-driven column parse produces -- pinned, because the
    two drifting apart would silently permute every trained head."""
    assert len(NUCLEI) == len(set(NUCLEI)) == 64
    assert len(SHIFT_COLUMNS) == 64
    assert tuple(parse_shift_columns(list(SHIFT_COLUMNS))) == SHIFT_COLUMNS
    # The 64 strictly extend the 49: the 15 are additions, and none of the 49 is absent.
    added = {name for names in ADDED_NUCLEI.values() for name in names}
    assert added < set(NUCLEI) and len(added) == 15
    assert len(set(NUCLEI) - added) == 49


def test_release_cache_schema_rejects_a_data_dependent_head_width():
    assert tuple(require_release_shift_columns(SHIFT_COLUMNS)) == SHIFT_COLUMNS
    with pytest.raises(ValueError, match="frozen 64-column"):
        require_release_shift_columns(SHIFT_COLUMNS[:-1])


def test_every_predicted_nucleus_has_a_center_on_the_residue_that_carries_it():
    """ppm is recovered as ``z * scale + center(aa, col)``, so a nucleus with no center
    would train against a zero center and denormalize to nonsense. The BMRB table is the
    whole csstats set and not a 49-column projection, which is why added columns need no
    new constants -- asserted here, never assumed."""
    for column in SHIFT_COLUMNS:
        assert any(norm.get_center(aa3, column) is not None
                   for aa3 in norm.BMRB_FILT), column
    for residue, names in ADDED_NUCLEI.items():
        for name in names:
            center = norm.get_center(residue, f"{name.lower()}_shift")
            assert center is not None and center > 0.0, (residue, name)
