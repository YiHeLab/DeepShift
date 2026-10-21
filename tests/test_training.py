"""Trainer contract: what a run is allowed to look at, and what it records.

These execute the real ``train()`` on a tiny synthetic five-fold cache (CPU, two epochs),
because the properties they cover are invisible to anything that does not run the loop: a
validation carve that silently withholds labelled residues from every model, and a QC
fence that matches nothing and still reports success. One run per selection mode is
shared across the assertions, so the whole file costs a handful of seconds.
"""
import argparse
import json
import re
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from deepshift import training
from deepshift.atom_cloud import FINAL_ATOM_CLOUD_RECIPE
from deepshift.data.norm import get_center
from deepshift.domain import ATOM_TYPES, SHIFT_COLUMNS, STANDARD_RESIDUES
from deepshift.loss import PROCHIRAL_PAIRS, build_loss_tensors
from deepshift.model import AtomTransformer
from deepshift.normalization import build_fixed_per_aa

CV_FOLDS = (1, 2, 3, 4, 5)
HELDOUT_FOLD = 1
ROWS_PER_FOLD = 12
ROWS_PER_ENTITY = 2
ATOM_RECIPE = FINAL_ATOM_CLOUD_RECIPE.with_vocabulary(
    len(ATOM_TYPES), FINAL_ATOM_CLOUD_RECIPE.pad_offset_index)
M_ATOMS = ATOM_RECIPE.m_atoms
ATOMS_PER_ROW = 12
FLAGGED_ROWS_PER_FOLD = 2
FLAGGED_COLUMNS = ("ca_shift", "cb_shift")


def _label_residue_id(fold, row):
    return f"lr_{fold}_{row}"


def _write_fold(fold_dir: Path, fold: int):
    """One synthetic fold in the live layout: ppm labels around the real fixed centers,
    a center residue in every atom cloud, and two rows per logical entity so an
    entity-grouped carve has something to carve."""
    rng = np.random.default_rng(fold)
    residue = rng.integers(0, len(STANDARD_RESIDUES) - 1, ROWS_PER_FOLD)
    residue[:FLAGGED_ROWS_PER_FOLD] = STANDARD_RESIDUES.index("ALA")
    snuc = {col: round(0.3 + 0.05 * i, 6) for i, col in enumerate(SHIFT_COLUMNS)}

    shifts = np.zeros((ROWS_PER_FOLD, len(SHIFT_COLUMNS)), dtype=np.float32)
    mask = np.zeros_like(shifts, dtype=bool)
    for row, code in enumerate(residue):
        aa3 = STANDARD_RESIDUES[code]
        for col_index, col in enumerate(SHIFT_COLUMNS):
            center = get_center(aa3, col)
            if center is None or rng.random() < 0.3:
                continue        # chemically absent, or simply not deposited
            shifts[row, col_index] = center + rng.normal(0.0, snuc[col])
            mask[row, col_index] = True
    # The synthetic fence below names CA/CB on the first two ALA rows. Make those
    # requested cells genuinely observed, so applying the fence is a real mask
    # transition and not a cell that was already false.
    for row in range(FLAGGED_ROWS_PER_FOLD):
        for col in FLAGGED_COLUMNS:
            col_index = SHIFT_COLUMNS.index(col)
            shifts[row, col_index] = get_center("ALA", col)
            mask[row, col_index] = True

    struct = fold_dir / "structural"
    struct.mkdir(parents=True, exist_ok=True)
    np.save(struct / "shifts.npy", shifts)
    np.save(struct / "shift_mask.npy", mask)
    np.save(struct / "residue_idx.npy", residue.astype(np.int32))
    np.save(struct / "source_idx.npy", (np.arange(ROWS_PER_FOLD) % 3).astype(np.int8))
    np.save(struct / "ambiguity.npy", np.zeros_like(shifts, dtype=np.int8))
    np.save(fold_dir / "samples.npy", np.arange(ROWS_PER_FOLD, dtype=np.int64))
    (struct / "id_maps.json").write_text(json.dumps({
        "logical_entity_id": {str(row): f"le_{fold}_{row // ROWS_PER_ENTITY}"
                              for row in range(ROWS_PER_FOLD)},
        "label_set_id": {str(row): f"ls_{fold}_{row // ROWS_PER_ENTITY}"
                         for row in range(ROWS_PER_FOLD)},
        "materialization_id": {str(row): f"mz_test_{fold}_{row // ROWS_PER_ENTITY}"
                               for row in range(ROWS_PER_FOLD)},
        "label_residue_id": {str(row): _label_residue_id(fold, row)
                             for row in range(ROWS_PER_FOLD)},
    }))
    (struct / "bmrb_mapping.json").write_text(json.dumps(
        {str(row): f"{17000 + fold}" for row in range(ROWS_PER_FOLD)}))
    (struct / "global_to_resid.json").write_text(json.dumps(
        {str(row): str(row + 1) for row in range(ROWS_PER_FOLD)}))
    (struct / "global_to_struct_resid.json").write_text(json.dumps(
        {str(row): str(row + 1) for row in range(ROWS_PER_FOLD)}))
    (struct / "materialization_ids.json").write_text(json.dumps(
        sorted({f"mz_test_{fold}_{row // ROWS_PER_ENTITY}" for row in range(ROWS_PER_FOLD)})))
    np.save(struct / "materialization_idx.npy",
            (np.arange(ROWS_PER_FOLD) // ROWS_PER_ENTITY).astype(np.int32))

    atom = fold_dir / "atom"
    atom.mkdir(parents=True, exist_ok=True)
    real = ATOMS_PER_ROW
    atom_type = np.full((ROWS_PER_FOLD, M_ATOMS), ATOM_RECIPE.pad_atom_index, dtype=np.int16)
    atom_type[:, :real] = rng.integers(0, 40, (ROWS_PER_FOLD, real))
    atom_type[:, :3] = [68, 1, 0]                     # N / CA / C of the center residue
    offset = np.full((ROWS_PER_FOLD, M_ATOMS), ATOM_RECIPE.pad_offset_index, dtype=np.int8)
    offset[:, :real] = ATOM_RECIPE.spatial_offset_code
    offset[:, :4] = AtomTransformer.CENTER_OFFSET     # the residue being predicted
    xyz = np.zeros((ROWS_PER_FOLD, M_ATOMS, 3), dtype=np.float16)
    xyz[:, :real] = rng.normal(0.0, 4.0, (ROWS_PER_FOLD, real, 3))
    np.save(atom / "atom_type.npy", atom_type)
    np.save(atom / "atom_offset.npy", offset)
    np.save(atom / "atom_xyz.npy", xyz)
    np.save(atom / "atom_count.npy", np.full(ROWS_PER_FOLD, real, dtype=np.int16))
    np.save(atom / "nucleus_atom_idx.npy",
            np.zeros((ROWS_PER_FOLD, len(SHIFT_COLUMNS)), dtype=np.int16))
    stats = {col: {"mean": 0.0, "std": 1.0} for col in SHIFT_COLUMNS}
    stats.update({
        "snuc": snuc, "snuc_counts": dict.fromkeys(SHIFT_COLUMNS, 500),
        "normalization": "secondary_snuc_train_folds_v1",
        "normalization_fit": {"policy": "cv_train_folds_only", "heldout_fold": fold,
                              "fit_folds": [f for f in CV_FOLDS if f != fold],
                              "flagged_cells_excluded": True,
                              "reref_applied_before_fit": True},
    })
    (fold_dir / "config.json").write_text(json.dumps({
        "n_shifts": len(SHIFT_COLUMNS), "shift_cols": list(SHIFT_COLUMNS),
        "shift_storage": "ppm_float32", "has_ppm_shifts": True, "entity_grained": True,
        "no_h": True, "source_names": ["af", "nmr", "xtal"], "stats": stats,
    }))
    (atom / "config.json").write_text(json.dumps({
        "recipe": ATOM_RECIPE.to_dict(),
    }))


def build_cache(root: Path):
    """A five-fold cache plus the QC flags file addressing it. Also used by the
    out-of-repo parity harness, which must see byte-identical inputs."""
    root.mkdir(parents=True, exist_ok=True)
    flags = {_label_residue_id(fold, row): list(FLAGGED_COLUMNS)
             for fold in CV_FOLDS for row in range(FLAGGED_ROWS_PER_FOLD)}
    (root / "flags.json").write_text(json.dumps(flags))
    (root / "fold_map.json").write_text("{}\n")
    for fold in CV_FOLDS:
        _write_fold(root / f"fold_{fold}", fold)
    return root


TRAIN_ARGV = ("--dim", "16", "--layers", "1", "--epochs", "2", "--batch-size", "4",
              "--ckpt-every", "2")


def run_train(root: Path, out: Path, *extra):
    """Invoke the trainer exactly as the CLI does, through its own argument parser, on
    the CPU whatever this machine carries."""
    parser = argparse.ArgumentParser()
    training.add_train_arguments(parser)
    args = parser.parse_args([
        "--cache", str(root), "--fold", str(HELDOUT_FOLD), "--out", str(out),
        "--exclude-flagged", str(root / "flags.json"), *TRAIN_ARGV, *extra])
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        return args.handler(args)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return build_cache(tmp_path_factory.mktemp("cache"))


# One run, shared by every read-only assertion below.
@pytest.fixture(scope="module")
def final_run(cache, tmp_path_factory):
    out = tmp_path_factory.mktemp("final") / "run.json"
    return run_train(cache, out)


@pytest.fixture(autouse=True)
def restore_global_torch_flags():
    """``configure_determinism`` sets process-global torch state; keep it from leaking
    into the rest of the suite."""
    saved = (torch.are_deterministic_algorithms_enabled(),
             torch.backends.cudnn.benchmark,
             torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32)
    yield
    torch.use_deterministic_algorithms(saved[0])
    torch.backends.cudnn.benchmark = saved[1]
    torch.backends.cuda.matmul.allow_tf32 = saved[2]
    torch.backends.cudnn.allow_tf32 = saved[3]


def _dataset(cache, fold):
    """A dataset over one fold, configured the way ``train()`` configures it."""
    fold_dir = cache / f"fold_{fold}"
    config = json.loads((fold_dir / "config.json").read_text())
    stats = build_fixed_per_aa(config["stats"], config["shift_cols"])
    return training.LeanCacheDataset([fold_dir], stats, config["shift_cols"])


# ---------------------------------------------------------------------------
# The delivered model: the last epoch, trained on every row of four folds
# ---------------------------------------------------------------------------

def test_a_member_trains_on_every_row_of_the_four_folds_it_does_not_hold_out(final_run):
    samples = final_run["manifest"]["samples"]
    assert samples["train_used"] == samples["train_before_cap"]
    assert samples["train_used"] == (len(CV_FOLDS) - 1) * ROWS_PER_FOLD
    assert final_run["manifest"]["folds"] == {
        "train": [f for f in CV_FOLDS if f != HELDOUT_FOLD], "heldout": [HELDOUT_FOLD]}


def test_the_delivered_model_is_the_last_epoch(final_run):
    protocol = final_run["manifest"]["protocol"]
    assert protocol["checkpoint_selection"] == "final"
    assert protocol["test_fold_used_for_selection"] is False
    assert final_run["ckpt"] == final_run["final_ckpt"]
    assert final_run["selected_epoch"] == 2
    assert not Path(final_run["ckpt"]).with_name("run.best.pt").exists()


def test_the_heldout_fold_is_scored_once_after_the_last_epoch(final_run):
    scored = [record["epoch"] for record in final_run["history"]
              if "backbone_z_mae" in record]
    assert scored == [2]


def test_prediction_dump_names_its_checkpoint(final_run):
    """The dump carries the selected checkpoint's name, epoch and policy as non-pickled
    scalar arrays."""
    checkpoint = Path(final_run["ckpt"])
    with np.load(final_run["preds_npz"], allow_pickle=False) as dump:
        assert dump["checkpoint_name"].item() == checkpoint.name
        assert int(dump["checkpoint_epoch"].item()) == final_run["selected_epoch"]
        assert dump["checkpoint_selection"].item() == final_run["selection"]


def test_determinism_is_the_default():
    parser = argparse.ArgumentParser()
    training.add_train_arguments(parser)
    args = parser.parse_args(["--cache", "c", "--fold", "1", "--out", "o",
                              "--exclude-flagged", "f"])
    assert args.deterministic is True


def test_the_qc_fence_is_not_optional():
    """``--exclude-flagged`` is required: a run must state its fence, even if empty."""
    parser = argparse.ArgumentParser()
    training.add_train_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--cache", "c", "--fold", "1", "--out", "o"])


# ---------------------------------------------------------------------------
# Cache checks: a run refuses a cache it cannot trust
# ---------------------------------------------------------------------------

def test_a_cache_fitted_on_the_wrong_folds_is_refused(tmp_path):
    """Normalization is fitted on the four training folds alone. A cache whose fit
    names a different held-out fold would leak the eval fold's scale into training, so
    the run refuses to start and never quietly rescales."""
    broken = tmp_path / "broken"
    build_cache(broken)
    config_path = broken / f"fold_{HELDOUT_FOLD}" / "config.json"
    config = json.loads(config_path.read_text())
    config["stats"]["normalization_fit"]["heldout_fold"] = HELDOUT_FOLD + 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="train-only S_NUC provenance"):
        run_train(broken, tmp_path / "broken.json")


def _restamp_atom_recipe(root, **changes):
    for fold in CV_FOLDS:
        path = root / f"fold_{fold}" / "atom" / "config.json"
        config = json.loads(path.read_text())
        config["recipe"].update(changes)
        path.write_text(json.dumps(config))


@pytest.mark.parametrize("changes", [{"m_atoms": 384}, {"partner_policy": "all"}])
def test_a_cache_built_under_another_atom_recipe_is_refused(tmp_path, changes):
    """The model is trained on the final cloud and inference rebuilds that cloud from the
    checkpoint's stamp, so a cache carrying any other recipe cannot train a member."""
    other = tmp_path / "other-recipe"
    build_cache(other)
    _restamp_atom_recipe(other, **changes)
    with pytest.raises(ValueError, match="is not the final recipe"):
        run_train(other, tmp_path / "other-recipe.json")


def test_the_manifest_records_the_recipe_the_run_trained_under(final_run):
    realized = training.realized_training_recipe(final_run["manifest"])
    assert realized == training.TrainingRecipe(epochs=2, batch_size=4, dimension=16,
                                               layers=1, checkpoint_every=2)
    assert final_run["manifest"]["protocol"]["heads"] == training.FINAL_TRAINING_RECIPE.heads
    assert (final_run["manifest"]["protocol"]["gradient_clip"]
            == training.FINAL_TRAINING_RECIPE.gradient_clip)


def _final_manifest(manifest, **model_changes):
    """A run record with every setting at the final recipe's value."""
    recipe = training.FINAL_TRAINING_RECIPE
    final = json.loads(json.dumps(manifest))
    final["protocol"].update({
        "epochs": recipe.epochs, "batch_size": recipe.batch_size,
        "learning_rate": recipe.learning_rate,
        "eta_min": recipe.learning_rate * recipe.eta_min_fraction,
        "weight_decay": recipe.weight_decay, "dropout": recipe.dropout, "seed": recipe.seed,
        "ckpt_every": recipe.checkpoint_every,
        "checkpoint_selection": recipe.checkpoint_selection,
        "determinism": {"deterministic_algorithms": recipe.deterministic}})
    final["model"].update({"dim": recipe.dimension, "layers": recipe.layers, **model_changes})
    final["loss"]["huber_delta"] = recipe.huber_delta
    return final


def test_only_a_final_recipe_manifest_passes_the_recipe_gate(final_run):
    """A paper member records the final recipe exactly, under the structure-only tag."""
    with pytest.raises(ValueError, match="epochs=2"):
        training.verify_final_training_recipe(final_run["manifest"])
    final = _final_manifest(final_run["manifest"])
    assert training.verify_final_training_recipe(final) == training.FINAL_TRAINING_RECIPE
    mistagged = _final_manifest(final_run["manifest"], tag="deepshift_other")
    with pytest.raises(ValueError, match=re.escape("model.tag")):
        training.verify_final_training_recipe(mistagged)
    ablated = _final_manifest(final_run["manifest"])
    ablated["protocol"]["data_frac"] = 0.5
    with pytest.raises(ValueError, match="data_frac"):
        training.verify_final_training_recipe(ablated)


def test_a_declared_ablation_passes_the_gate_only_when_realized(final_run):
    """A ladder point or a source-restricted member passes only by naming the switch it
    holds away from the final recipe; a declaration a paper member does not realize, an
    unknown switch, and an undeclared source restriction are all refused."""
    ladder = _final_manifest(final_run["manifest"])
    ladder["protocol"]["data_frac"] = 0.25
    assert training.verify_final_training_recipe(
        ladder, ("protocol.data_frac",)) == training.FINAL_TRAINING_RECIPE
    with pytest.raises(ValueError, match="does not realize"):
        training.verify_final_training_recipe(_final_manifest(final_run["manifest"]),
                                              ("protocol.data_frac",))
    with pytest.raises(ValueError, match="not ablation switches"):
        training.verify_final_training_recipe(ladder, ("protocol.epochs",))
    restricted = _final_manifest(final_run["manifest"])
    restricted["arguments"]["train_sources"] = ["xtal", "nmr"]
    with pytest.raises(ValueError, match=re.escape("arguments.train_sources")):
        training.verify_final_training_recipe(restricted)
    assert training.verify_final_training_recipe(
        restricted, ("arguments.train_sources",)) == training.FINAL_TRAINING_RECIPE


def test_a_cache_not_stored_in_ppm_is_refused(tmp_path):
    """Labels are stored as deposited ppm; a global-z cache would silently train
    on a different target than the checkpoint's stats can invert."""
    z_cache = tmp_path / "z_cache"
    build_cache(z_cache)
    config_path = z_cache / f"fold_{HELDOUT_FOLD}" / "config.json"
    config = json.loads(config_path.read_text())
    config["shift_storage"], config["has_ppm_shifts"] = "global_z", False
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="only ppm_float32 is supported"):
        _dataset(z_cache, HELDOUT_FOLD)


def test_a_cache_with_a_different_nucleus_vocabulary_is_refused(tmp_path):
    """The cache's column order is authoritative, its vocabulary is not: a cache built
    over some other nucleus set trains a head the released ensemble cannot average
    with, and nothing about the shapes would say so."""
    wrong = tmp_path / "wrong"
    build_cache(wrong)
    config_path = wrong / f"fold_{HELDOUT_FOLD}" / "config.json"
    config = json.loads(config_path.read_text())
    config["shift_cols"] = [c for c in config["shift_cols"] if c != "ne_shift"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="nucleus vocabulary is not the shipped one"):
        run_train(wrong, tmp_path / "wrong.json")


def test_a_shift_column_without_a_fitted_scale_is_refused(cache):
    """A per-nucleus scale is fitted per fold, so every predicted column must appear in
    the cache's own train-fold table. Falling back to a frozen table, which covers only
    the 49 nuclei, or to a unit std would normalize against the wrong width and
    denormalize to a wrong ppm without ever crashing."""
    config = json.loads((cache / f"fold_{HELDOUT_FOLD}" / "config.json").read_text())
    stats, columns = config["stats"], config["shift_cols"]
    partial = dict(stats, snuc={col: scale for col, scale in stats["snuc"].items()
                                if col != "ne_shift"})
    with pytest.raises(ValueError, match=r"no scale for \['ne_shift'\]"):
        build_fixed_per_aa(partial, columns)
    # An absent table is the same refusal, not a silent fallback for all 64 columns.
    absent = {key: value for key, value in stats.items() if key != "snuc"}
    with pytest.raises(ValueError, match="no scale for"):
        build_fixed_per_aa(absent, columns)


# ---------------------------------------------------------------------------
# The optimizer step: a finite loss is not a finite gradient
# ---------------------------------------------------------------------------

def test_optimizer_step_skips_finite_loss_with_nonfinite_gradient():
    """sqrt at zero: the loss is finite, its derivative is not. Stepping on that would
    poison every weight, so the step is skipped and the gradient dropped."""
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    stepped = training._optimizer_step(parameter.sqrt(),
                                       torch.nn.ParameterList([parameter]),
                                       optimizer, scaler=None)
    assert stepped is False
    assert parameter.item() == 0.0
    assert parameter.grad is None
    assert optimizer.state == {}


def test_optimizer_step_preserves_the_finite_path():
    """The guard must not perturb an ordinary step: same weights as unguarded
    clip-then-step, to the bit."""
    original, guarded = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    guarded.load_state_dict(original.state_dict())
    inputs, target = torch.tensor([[1.0, -2.0, 3.0]]), torch.tensor([[0.5, -0.25]])

    reference_optimizer = torch.optim.AdamW(original.parameters(), lr=1e-3)
    F.mse_loss(original(inputs), target).backward()
    torch.nn.utils.clip_grad_norm_(original.parameters(), training.GRAD_CLIP)
    reference_optimizer.step()

    optimizer = torch.optim.AdamW(guarded.parameters(), lr=1e-3)
    assert training._optimizer_step(F.mse_loss(guarded(inputs), target),
                                    guarded, optimizer, None) is True
    for expected, actual in zip(original.parameters(), guarded.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# The QC fence must never fail quietly
# ---------------------------------------------------------------------------

def test_fence_masks_exactly_the_flagged_cells(cache):
    dataset = _dataset(cache, HELDOUT_FOLD)
    flags = json.loads((cache / "flags.json").read_text())
    assert dataset.apply_flags(flags) == FLAGGED_ROWS_PER_FOLD * len(FLAGGED_COLUMNS)
    columns = [dataset.shift_cols.index(col) for col in FLAGGED_COLUMNS]
    assert not dataset[0]["shift_mask"][columns].any()


def test_empty_fence_is_a_no_op(cache):
    """An empty flags file is a caller saying "no fence" -- different from a fence that
    did not work, which is why only this case may return 0."""
    assert _dataset(cache, HELDOUT_FOLD).apply_flags({}) == 0


def test_fence_matching_no_cell_fails_loudly(cache):
    """INVARIANT: a non-empty fence that reaches nothing raises. Returning 0 is
    indistinguishable from a clean cache, so a flags file naming thousands of cells that
    address no row of this cache reports success on an unfenced run."""
    dataset = _dataset(cache, HELDOUT_FOLD)
    with pytest.raises(ValueError, match="matched no cell"):
        dataset.apply_flags({"lr_not_in_this_cache": ["ca_shift"]})
    # A {bmrb: [[residue, column], ...]} schema addresses nothing here either,
    # and must not be mistaken for a clean cache.
    with pytest.raises(ValueError, match="matched no cell"):
        dataset.apply_flags({"17001": [[1, "ca_shift"]]})
    # Neither may a real id carrying a column this cache does not have.
    with pytest.raises(ValueError, match="not shift columns"):
        dataset.apply_flags({_label_residue_id(HELDOUT_FOLD, 0): ["not_a_shift"]})


def test_one_valid_flag_cannot_hide_an_unobserved_flag(cache):
    """Every present requested cell must make a True->False mask transition.

    Counting a requested but already-false cell as applied lets a partially stale
    all-fold flag file look successful whenever one unrelated cell still matches.
    """
    dataset = _dataset(cache, HELDOUT_FOLD)
    label = _label_residue_id(HELDOUT_FOLD, 0)
    assert dataset[0]["shift_mask"][dataset.shift_cols.index("ca_shift")]
    assert not dataset[0]["shift_mask"][dataset.shift_cols.index("ne_shift")]
    with pytest.raises(ValueError, match="no observed cell"):
        dataset.apply_flags({label: ["ca_shift", "ne_shift"]})


def test_source_filter_refuses_a_fold_without_source_codes(tmp_path):
    stripped = tmp_path / "no-source"
    build_cache(stripped)
    fold_dir = stripped / f"fold_{HELDOUT_FOLD}"
    (fold_dir / "structural" / "source_idx.npy").unlink()
    config = json.loads((fold_dir / "config.json").read_text())
    stats = build_fixed_per_aa(config["stats"], config["shift_cols"])
    with pytest.raises(ValueError, match=re.escape("source_idx.npy is missing")):
        training.LeanCacheDataset(
            [fold_dir], stats, config["shift_cols"], sources=["af"]
        )


def test_length_sampler_keeps_the_partial_batch():
    sampler = training.LengthBucketBatchSampler(
        np.arange(5), batch_size=4, seed=7, mega=2
    )
    batches = list(sampler)
    assert len(sampler) == 2
    assert sorted(index for batch in batches for index in batch) == list(range(5))
    assert sorted(map(len, batches)) == [1, 4]


def test_fence_on_a_cache_without_label_residues_fails_loudly(tmp_path):
    stripped = tmp_path / "stripped"
    build_cache(stripped)
    (stripped / f"fold_{HELDOUT_FOLD}" / "structural" / "id_maps.json").unlink()
    with pytest.raises(ValueError, match="no label_residue_id map"):
        _dataset(stripped, HELDOUT_FOLD).apply_flags(
            json.loads((stripped / "flags.json").read_text()))


def test_training_reports_the_cells_the_fence_masked(final_run):
    samples = final_run["manifest"]["samples"]
    per_fold = FLAGGED_ROWS_PER_FOLD * len(FLAGGED_COLUMNS)
    assert samples["eval_flagged_cells_masked"] == per_fold
    assert samples["train_flagged_cells_masked"] == (len(CV_FOLDS) - 1) * per_fold


# ---------------------------------------------------------------------------
# The swap-invariant loss: what may exchange, and what may not
# ---------------------------------------------------------------------------

def _partners(residue3):
    """{shift column: its partner column} for one residue, read off the loss tensors."""
    _, partner = build_loss_tensors(SHIFT_COLUMNS, STANDARD_RESIDUES)
    row = partner[STANDARD_RESIDUES.index(residue3)]
    return {SHIFT_COLUMNS[i]: SHIFT_COLUMNS[int(j)]
            for i, j in enumerate(row.tolist()) if j >= 0}


def test_a_swapped_methyl_carbon_takes_its_own_protons_with_it():
    """INVARIANT: if a nucleus may swap, everything rigidly attached to it swaps too.
    LEU CD1/CD2 and VAL CG1/CG2 exchange together with their own methyl protons; a loss
    that swapped the carbons and not the protons would accept a state no molecule can
    be in."""
    leucine = _partners("LEU")
    assert leucine["cd1_shift"] == "cd2_shift"
    for a, b in (("hd11", "hd21"), ("hd12", "hd22"), ("hd13", "hd23")):
        assert leucine[f"{a}_shift"] == f"{b}_shift"
        assert leucine[f"{b}_shift"] == f"{a}_shift"
    valine = _partners("VAL")
    assert valine["cg1_shift"] == "cg2_shift"
    for a, b in (("hg11", "hg21"), ("hg12", "hg22"), ("hg13", "hg23")):
        assert valine[f"{a}_shift"] == f"{b}_shift"
        assert valine[f"{b}_shift"] == f"{a}_shift"
    # ILE reuses those very columns for its two different methyls (CD1 and CG2), which
    # are not interchangeable, so the pairing must be per-residue and absent here.
    isoleucine = _partners("ILE")
    assert "hd11_shift" not in isoleucine and "hg21_shift" not in isoleucine


def test_the_arg_guanidinium_amino_groups_exchange_but_not_within_a_group():
    """Rotation about CZ-NE exchanges the two guanidinium NH2 groups and nothing anchors
    which is NH1, so NH1's protons pair with NH2's.

    The intra-group pairing is pinned out for a chemical reason, not a bookkeeping one:
    the CZ-NH partial double bond restricts rotation, so HH11 and HH12 lie cis and trans
    to NE and are inequivalent. Pairing them would tell the loss that two distinguishable
    protons are interchangeable.
    """
    arginine = _partners("ARG")
    assert arginine["hh11_shift"] == "hh21_shift"
    assert arginine["hh12_shift"] == "hh22_shift"
    assert arginine["hh11_shift"] != "hh12_shift"
    assert arginine["hh21_shift"] != "hh22_shift"
    # ARG HH<->HE is a documented gap and not an equivalence: different nitrogens.
    assert "he_shift" not in arginine


def test_the_aromatic_ring_pairs_stay_a_documented_gap():
    """PHE/TYR ring nuclei are genuinely exchangeable under the two-fold flip, and they
    are handled by the equivalence weight (1/2 each) alone, never by a swap. Turning
    them into swap pairs changes the loss and is not a measured configuration, so the
    absence is pinned here."""
    for residue in ("PHE", "TYR"):
        partners = _partners(residue)
        assert partners == {"hb2_shift": "hb3_shift", "hb3_shift": "hb2_shift"}
        group_weight, _ = build_loss_tensors(SHIFT_COLUMNS, STANDARD_RESIDUES)
        row = group_weight[STANDARD_RESIDUES.index(residue)]
        for column in ("cd1_shift", "cd2_shift", "ce1_shift", "ce2_shift",
                       "hd1_shift", "hd2_shift", "he1_shift", "he2_shift"):
            assert row[SHIFT_COLUMNS.index(column)] == 0.5, (residue, column)


def test_no_nucleus_carries_two_prochiral_partners():
    """The partner table holds one partner per column, so a second pair naming the same
    column would silently overwrite the first. Nothing may rely on that ordering."""
    for residue, pairs in PROCHIRAL_PAIRS.items():
        named = [atom for pair in pairs for atom in pair]
        assert len(named) == len(set(named)), residue


# ---------------------------------------------------------------------------
# Determinism and the released weight-init stream
# ---------------------------------------------------------------------------

def test_determinism_is_on_by_default_and_recorded(final_run):
    determinism = final_run["manifest"]["protocol"]["determinism"]
    assert determinism["deterministic_algorithms"] is True
    assert determinism["allow_tf32"] is False
    assert determinism["cublas_workspace_config"] in training._CUBLAS_DETERMINISTIC


def test_disabled_determinism_is_recorded_not_hidden(cache, tmp_path):
    result = run_train(cache, tmp_path / "nondet.json", "--no-deterministic")
    assert result["manifest"]["protocol"]["determinism"] == {
        "deterministic_algorithms": False}


def test_two_runs_of_one_command_produce_identical_weights(cache, tmp_path, final_run):
    repeat = run_train(cache, tmp_path / "repeat.json")
    first = torch.load(final_run["final_ckpt"], map_location="cpu", weights_only=False)
    second = torch.load(repeat["final_ckpt"], map_location="cpu", weights_only=False)
    for key, value in first["model_state_dict"].items():
        assert torch.equal(value, second["model_state_dict"][key]), key


def test_a_zero_learning_rate_checkpoint_is_the_seeded_initialisation(cache, tmp_path):
    """With ``--lr 0`` no step can move a weight, so the checkpoint is exactly the model
    the seed constructs: the initialisation depends on the seed and the architecture and
    on nothing else."""
    result = run_train(cache, tmp_path / "rng.json", "--lr", "0", "--epochs", "1")
    trained = torch.load(result["final_ckpt"], map_location="cpu",
                         weights_only=False)["model_state_dict"]
    n_shifts = len(json.loads(
        (cache / f"fold_{HELDOUT_FOLD}" / "config.json").read_text())["shift_cols"])
    torch.manual_seed(0)                                 # --seed default
    model = AtomTransformer(len(ATOM_TYPES), n_shifts, dim=16, heads=8, layers=1,
                            dropout=0.3)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, trained[name]), name


def test_the_dump_resolves_prochiral_pairs_by_the_package_rule(final_run):
    """The prediction dump's resolved targets are the package resolver's, computed in
    ppm from the dump's own predictions, labels, masks and codes, so the p100 fitter, the
    evaluator and every scorer read one resolution."""
    from deepshift import evaluate
    dump = np.load(final_run["preds_npz"], allow_pickle=False)
    cols = [str(c) for c in dump["shift_cols"]]
    resolved, effective = evaluate.resolve(
        dump["pred_ppm"], dump["true_ppm"], dump["mask"], dump["ambiguity"],
        dump["residue_code"], cols)
    assert np.array_equal(effective, dump["effective_mask"])
    assert np.allclose(resolved[effective], dump["resolved_true_ppm"][effective], atol=1e-5)


def test_a_member_named_without_a_suffix_writes_its_result_beside_its_checkpoints(cache, tmp_path):
    """The documented command names a member by its stem; the result is the stem plus
    .json, and every checkpoint and the prediction dump share that stem."""
    result = run_train(cache, tmp_path / "member", "--epochs", "1")
    assert (tmp_path / "member.json").is_file()
    assert Path(result["final_ckpt"]) == tmp_path / "member.final.pt"
    assert Path(result["last_ckpt"]) == tmp_path / "member.last.pt"
    assert Path(result["preds_npz"]) == tmp_path / "member.preds.npz"
    assert not (tmp_path / "member").exists()


def test_a_resumed_run_reproduces_the_uninterrupted_run(cache, tmp_path, monkeypatch):
    """A member stopped after an epoch and continued with --resume ends in the same
    weights, with the same per-epoch losses, as one that never stopped; the result and
    every later checkpoint disclose the resumption."""
    argv = ("--ckpt-every", "1", "--epochs", "3")
    straight = run_train(cache, tmp_path / "straight.json", *argv)
    assert straight["resumed"] is None

    save = training._atomic_torch_save

    def save_then_stop(payload, path):
        save(payload, path)
        if str(path).endswith(".last.pt") and payload["epoch"] == 1:
            raise KeyboardInterrupt
    monkeypatch.setattr(training, "_atomic_torch_save", save_then_stop)
    with pytest.raises(KeyboardInterrupt):
        run_train(cache, tmp_path / "resumed.json", *argv)
    monkeypatch.setattr(training, "_atomic_torch_save", save)
    last = tmp_path / "resumed.last.pt"
    assert torch.load(last, map_location="cpu", weights_only=True)["resume"]["epoch"] == 1

    resumed = run_train(cache, tmp_path / "resumed.json", *argv, "--resume")
    assert resumed["resumed"]["from_epoch"] == 1
    assert resumed["resumed"]["times"] == 1
    assert [r["loss"] for r in resumed["history"]] == [r["loss"] for r in straight["history"]]
    weights = [torch.load(r["final_ckpt"], map_location="cpu", weights_only=True)
               for r in (straight, resumed)]
    for name, tensor in weights[0]["model_state_dict"].items():
        assert torch.equal(tensor, weights[1]["model_state_dict"][name]), name
    assert weights[1]["manifest"]["resumed"]["from_epoch"] == 1
    assert "resumed" not in weights[0]["manifest"]


def test_resume_refuses_a_checkpoint_from_another_run(cache, tmp_path):
    """The last checkpoint of a run with a different recipe is not a starting point."""
    run_train(cache, tmp_path / "member.json", "--ckpt-every", "1", "--epochs", "2")
    (tmp_path / "member.last.pt").rename(tmp_path / "other.last.pt")
    with pytest.raises(ValueError, match="training recipe differs"):
        run_train(cache, tmp_path / "other.json", "--ckpt-every", "1", "--epochs", "3",
                  "--dropout", "0.1", "--resume")


# ---------------------------------------------------------------------------
# --data-frac: the data-scaling ladder's only moving part
# ---------------------------------------------------------------------------

def test_data_frac_cuts_only_the_training_rows(cache, tmp_path):
    """A fraction changes how much data the model sees and nothing else.

    The held-out fold and the pre-fraction pool must be identical across the ladder, or
    the arms are not comparable and the curve measures two things at once.
    """
    full = run_train(cache, tmp_path / "full.json")
    quarter = run_train(cache, tmp_path / "quarter.json", "--data-frac", "0.25")

    rows = full["manifest"]["samples"]["train_used"]
    assert quarter["manifest"]["samples"]["train_used"] == round(rows * 0.25)
    assert quarter["manifest"]["samples"]["train_before_cap"] == rows
    assert quarter["manifest"]["samples"]["eval_used"] == full["manifest"]["samples"]["eval_used"]
    assert quarter["manifest"]["protocol"]["data_frac"] == 0.25
    assert full["manifest"]["protocol"]["data_frac"] == 0.0


def test_data_frac_arms_are_nested_and_frac_one_is_a_no_op(cache):
    """Sampling without replacement returns unrelated index sets at different sizes, so a
    ladder built that way can fall non-monotone from luck alone. The smaller arm has to be
    contained in the larger one."""
    dataset = _dataset(cache, HELDOUT_FOLD)
    quarter = training._nested_fraction(dataset, 0.25, 0)
    half = training._nested_fraction(dataset, 0.5, 0)
    assert set(quarter.indices) < set(half.indices)
    assert len(quarter.indices) == round(len(dataset) * 0.25)
    for frac in (0.0, 1.0, 2.0):
        assert training._nested_fraction(dataset, frac, 0) is dataset

