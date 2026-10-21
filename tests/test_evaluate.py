"""The rules the canonical scorer holds to.

Each test below pins one rule whose violation gives a wrong result that looks entirely
plausible: a uint8 mask read as an index, a correlation computed on raw ppm, an
ensemble averaged in z, a "paired" comparison over two different row sets. The
arithmetic is checked on small synthetic populations where the right answer is known by
construction; the one test that needs a GPU scores a real checkpoint over the synthetic
cache the trainer's own tests build.
"""
import argparse
import json

import numpy as np
import pytest
import torch
from test_training import HELDOUT_FOLD, build_cache

from deepshift import cohorts, evaluate, loss, normalization, statistics
from deepshift.atom_cloud import FINAL_ATOM_CLOUD_RECIPE
from deepshift.domain import ATOM_TYPES, STANDARD_RESIDUES
from deepshift.model import AtomTransformer

COLUMNS = ("ca_shift", "cb_shift")
# Residue types whose CA sit far apart in ppm; the spread between types is what makes a
# raw-ppm correlation look perfect.
TYPE_CENTER = {0: 45.0, 1: 55.0, 2: 65.0, 3: 75.0}


def _population(pred, true, *, residue, mask=None, ambiguity=None, label_set=None,
                source=None, columns=COLUMNS, keys=None, resolved=None):
    """A Predictions built directly, so a test states exactly the population it means."""
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    rows = pred.shape[0]
    mask = np.ones_like(pred, dtype=bool) if mask is None else np.asarray(mask).astype(bool)
    ambiguity = np.zeros_like(pred, dtype=np.int64) if ambiguity is None else np.asarray(ambiguity)
    label_set = np.asarray([f"ls{i}" for i in range(rows)] if label_set is None else label_set)
    return evaluate.Predictions(
        shift_cols=tuple(columns), pred_ppm=pred, true_ppm=true,
        resolved_ppm=true.copy() if resolved is None else np.asarray(resolved, dtype=float),
        mask=mask, effective_mask=mask & ~np.isin(ambiguity, loss.AMBIGUITY_SET_CODES),
        ambiguity=ambiguity, residue_code=np.asarray(residue),
        row_key=np.asarray([f"row{i}" for i in range(rows)] if keys is None else keys),
        label_set=label_set, bmrb=np.asarray([str(x) for x in label_set]),
        source=np.zeros(rows, dtype=np.int64) if source is None else np.asarray(source))


def _type_mean_predictor(seed=0, rows=2000):
    """A model that predicts the per-residue-type mean and nothing else -- the exact
    model a raw-ppm correlation cannot tell apart from a good one."""
    rng = np.random.default_rng(seed)
    residue = rng.integers(0, len(TYPE_CENTER), rows)
    centers = np.asarray([TYPE_CENTER[int(c)] for c in residue])
    true = centers + rng.normal(0.0, 1.5, rows)          # real secondary structure
    pred = centers + rng.normal(0.0, 0.01, rows)         # no secondary information at all
    return _population(np.stack([pred, pred], 1), np.stack([true, true], 1), residue=residue)


def _dump(path, preds, **overrides):
    """Write ``preds`` as a trainer-shaped npz, with fields overridable to build the
    broken dumps the reader has to reject."""
    fields = {
        "pred_ppm": preds.pred_ppm, "true_ppm": preds.true_ppm,
        "resolved_true_ppm": preds.resolved_ppm,
        "mask": preds.mask, "effective_mask": preds.effective_mask, "ambiguity": preds.ambiguity,
        "residue_code": preds.residue_code, "shift_cols": np.asarray(preds.shift_cols),
        "label_residue_id": preds.row_key,
        "materialization_id": np.asarray(["m"] * preds.n_rows),
        "label_set_id": preds.label_set, "bmrb": preds.bmrb, "source": preds.source}
    fields.update(overrides)
    np.savez(path, **{k: v for k, v in fields.items() if v is not None})
    return path


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def test_a_uint8_mask_scores_exactly_like_a_bool_mask(tmp_path):
    """INVARIANT: masks are cast, never trusted. Stored masks are often uint8, and numpy
    reads a uint8 mask as a list of row numbers and not as a selection -- which does not
    raise, it just reports fake ~0.07 ppm metrics."""
    preds = _type_mean_predictor()
    same = evaluate.read_predictions(_dump(tmp_path / "bool.npz", preds))
    cast = evaluate.read_predictions(_dump(
        tmp_path / "uint8.npz", preds, mask=preds.mask.astype(np.uint8),
        effective_mask=preds.effective_mask.astype(np.uint8)))
    assert cast.mask.dtype == bool and cast.effective_mask.dtype == bool
    # assert_equal and not ==: a thin nucleus can score a NaN correlation, and NaN is
    # not equal to itself.
    np.testing.assert_equal(evaluate.score(same, boots=0), evaluate.score(cast, boots=0))


def test_predictions_refuse_a_non_bool_mask():
    preds = _type_mean_predictor()
    with pytest.raises(TypeError, match="bool"):
        evaluate.Predictions(**{**preds.__dict__, "mask": preds.mask.astype(np.uint8)})


def test_row_selection_is_a_bool_mask_not_an_index_array():
    """``rows`` selects; it does not index. A length-N index array cast to bool keeps
    every row but row 0 and reports a table that is merely slightly wrong."""
    preds = _type_mean_predictor()
    with pytest.raises(TypeError, match="bool mask"):
        evaluate.score(preds, rows=np.arange(preds.n_rows), boots=0)


# ---------------------------------------------------------------------------
# The scored cells: effective mask, resolved targets
# ---------------------------------------------------------------------------

def _with_set_codes(rows=400, bad=40):
    """A population where ``bad`` cells of column 0 carry BMRB set code 4 -- a geminal
    set whose deposited number is an average of two sites and belongs to neither."""
    rng = np.random.default_rng(5)
    residue = rng.integers(0, len(TYPE_CENTER), rows)
    centers = np.asarray([TYPE_CENTER[int(c)] for c in residue])
    true = np.stack([centers + rng.normal(0.0, 1.5, rows)] * 2, 1)
    pred = true + rng.normal(0.0, 0.2, (rows, 2))
    ambiguity = np.zeros((rows, 2), dtype=np.int64)
    ambiguity[:bad, 0] = 4
    true[:bad, 0] += 6.0                       # the set average sits nowhere near the site
    return _population(pred, true, residue=residue, ambiguity=ambiguity)


def test_set_code_cells_are_dropped_from_the_score():
    """INVARIANT: score reads the effective mask, not the observed mask. Codes 4/5/6/9
    have no atom-specific target, and scoring them turns a 0.2 ppm nucleus into a 1.9 ppm
    one without any table looking odd."""
    preds = _with_set_codes()
    got = evaluate.score(preds, boots=0)["ca_shift"]
    assert got["n"] == 360                      # 400 observed, 40 unscorable
    assert got["rmse"] < 0.5
    raw = evaluate.score(
        evaluate.Predictions(**{**preds.__dict__, "effective_mask": preds.mask}),
        boots=0)["ca_shift"]
    assert raw["n"] == 400 and raw["rmse"] > 1.0, "the dropped cells must actually matter"


def test_a_paired_comparison_also_scores_only_the_effective_cells():
    preds = _with_set_codes()
    other = _population(preds.pred_ppm + 0.05, preds.true_ppm, residue=preds.residue_code,
                        ambiguity=preds.ambiguity, label_set=preds.label_set,
                        keys=preds.row_key)
    assert evaluate.compare(preds, other, boots=16)["ca_shift"]["n"] == 360


def test_a_paired_comparison_scores_the_resolved_targets():
    """INVARIANT: never the raw labels, on either arm. Here the deposited pair is
    swapped relative to the assignment both arms fit, so scoring ``true_ppm`` reports
    ~1 ppm of error that does not exist."""
    columns = ("hb2_shift", "hb3_shift")
    rows = 40
    residue = np.full(rows, STANDARD_RESIDUES.index("SER"))
    deposited = np.tile([3.0, 4.0], (rows, 1))
    resolved = np.tile([4.0, 3.0], (rows, 1))          # swap-resolved against the models
    baseline = _population(resolved + 0.10, deposited, residue=residue, columns=columns,
                           resolved=resolved)
    candidate = _population(resolved + 0.02, deposited, residue=residue, columns=columns,
                            resolved=resolved)
    got = evaluate.compare(baseline, candidate, boots=64)["hb2_shift"]
    assert got["rmse_baseline"] == pytest.approx(0.10)
    assert got["rmse_candidate"] == pytest.approx(0.02)
    assert got["verdict"] == "candidate"


# ---------------------------------------------------------------------------
# The secondary shift
# ---------------------------------------------------------------------------

def test_raw_ppm_correlation_is_near_one_for_a_model_that_learned_nothing():
    """INVARIANT: r and amplitude are computed on the secondary shift.

    A model that emits the per-residue-type mean has learned no chemistry, yet raw-ppm r
    reads ~1.0 because the between-type spread swamps the within-type signal. The scorer
    must not report that number.
    """
    preds = _type_mean_predictor()
    cells = preds.effective_mask[:, 0]
    raw_r = np.corrcoef(preds.pred_ppm[cells, 0], preds.resolved_ppm[cells, 0])[0, 1]
    assert raw_r > 0.99, "the synthetic model is supposed to look perfect in raw ppm"

    reported = evaluate.score(preds, boots=0)["ca_shift"]
    assert abs(reported["r"]) < 0.2               # centred: no within-type agreement left
    assert reported["amplitude"] < 0.05           # and it predicts no spread at all


def test_the_secondary_centring_leaves_ppm_error_untouched():
    """Centring is for r and amplitude only: MAE and RMSE stay the ppm error a
    spectroscopist would measure."""
    preds = _type_mean_predictor()
    got = evaluate.score(preds, boots=0)["ca_shift"]
    error = preds.pred_ppm[:, 0] - preds.resolved_ppm[:, 0]
    assert got["rmse"] == pytest.approx(float(np.sqrt((error ** 2).mean())))
    assert got["mae"] == pytest.approx(float(np.abs(error).mean()))


def test_a_singleton_residue_type_is_dropped_from_the_correlation():
    """A type with one observed cell carries no within-type information; centring it
    would plant a free (0, 0) point in both series and inflate r."""
    residue = np.asarray([0] * 8 + [1])            # residue type 1 appears once
    true = np.asarray([45.0, 46.0, 44.0, 45.5, 44.2, 46.3, 45.1, 44.8, 65.0])
    pred = true + 0.1
    preds = _population(np.stack([pred, pred], 1), np.stack([true, true], 1), residue=residue)
    got = evaluate.score(preds, boots=0)["ca_shift"]
    assert got["n"] == 9 and got["n_secondary"] == 8


# ---------------------------------------------------------------------------
# Combining models: ppm, never z
# ---------------------------------------------------------------------------

def _stats(scale):
    """Checkpoint stats whose per-nucleus scale differs between members."""
    raw = {col: {"mean": 0.0, "std": 1.0} for col in COLUMNS}
    raw["snuc"] = dict.fromkeys(COLUMNS, scale)
    return normalization.build_fixed_per_aa(raw, COLUMNS)


def test_averaging_in_z_is_a_different_answer_than_averaging_in_ppm():
    """INVARIANT: members are combined in ppm. Each CV member denormalizes with its own
    per-nucleus scale, so its z is its own unit; averaging z and decoding through one
    member's scale silently rescales the other member's prediction."""
    residue = np.asarray([0, 1, 2, 3])
    true = np.asarray([[45.0, 30.0], [55.0, 32.0], [65.0, 34.0], [75.0, 36.0]])
    first = _population(true + 0.5, true, residue=residue)
    second = _population(true - 1.5, true, residue=residue)

    combined = evaluate.ensemble([first, second])
    assert np.allclose(combined.pred_ppm, (first.pred_ppm + second.pred_ppm) / 2)

    stats_a, stats_b = _stats(1.0), _stats(3.0)
    in_z = (normalization.normalize(first.pred_ppm, residue, stats_a, COLUMNS)
            + normalization.normalize(second.pred_ppm, residue, stats_b, COLUMNS)) / 2
    through_a = normalization.denormalize(in_z, residue, stats_a, COLUMNS)
    assert not np.allclose(through_a, combined.pred_ppm)


def test_a_dump_that_stores_only_z_is_refused(tmp_path):
    preds = _type_mean_predictor()
    path = _dump(tmp_path / "z.npz", preds, pred_ppm=None, pred_z=preds.pred_ppm)
    with pytest.raises(ValueError, match="denormalize with the checkpoint's own stats"):
        evaluate.read_predictions(path)


def test_an_ensemble_must_be_the_same_rows():
    preds = _type_mean_predictor()
    other = _population(preds.pred_ppm, preds.true_ppm, residue=preds.residue_code,
                        keys=[f"other{i}" for i in range(preds.n_rows)])
    with pytest.raises(ValueError, match="do not score the same rows"):
        evaluate.ensemble([preds, other])


@pytest.mark.parametrize("field", ["label_set", "bmrb", "source"])
def test_an_ensemble_must_agree_on_the_metadata_the_bootstrap_groups_by(field):
    """row_key is materialization|label_residue and carries none of these, yet they decide
    the bootstrap's clusters and the per-source split. Members that agreed on rows and
    disagreed here would average, and the first member's metadata would silently decide
    how the result is grouped."""
    preds = _type_mean_predictor()
    original = getattr(preds, field)
    changed = (original + 1 if original.dtype.kind in "iu"
               else np.char.add(original.astype(str), "x"))
    other = evaluate.Predictions(**{**preds.__dict__, field: changed})
    with pytest.raises(ValueError, match=f"disagree on {field}"):
        evaluate.ensemble([preds, other])


def test_an_ensemble_must_read_one_cache():
    """Same rows, different deposited labels: the members are not scoring one cache,
    whatever their row keys claim."""
    preds = _type_mean_predictor()
    shifted = evaluate.Predictions(**{**preds.__dict__, "true_ppm": preds.true_ppm + 0.5})
    with pytest.raises(ValueError, match="disagree on the ground truth"):
        evaluate.ensemble([preds, shifted])


def test_the_command_line_pools_across_arguments_and_ensembles_within_one(tmp_path):
    """The two operations are one keystroke apart and mean opposite things: separate
    arguments are different rows (the held-out folds), commas are the same rows scored by
    several checkpoints. Swapping them either triple-counts residues or averages folds."""
    residue = np.zeros(12, dtype=int)
    true = np.full((12, 2), 45.0)
    fold_a1 = _population(true + 0.2, true, residue=residue,
                          keys=[f"a{i}" for i in range(12)])
    fold_a2 = _population(true - 0.4, true, residue=residue,
                          keys=[f"a{i}" for i in range(12)])   # same rows, 2nd checkpoint
    fold_b = _population(true + 0.1, true, residue=residue,
                         keys=[f"b{i}" for i in range(12)])    # different rows
    paths = [str(_dump(tmp_path / f"{n}.npz", p)) for n, p in
             (("a1", fold_a1), ("a2", fold_a2), ("b", fold_b))]
    arm = evaluate._load_arm([f"{paths[0]},{paths[1]}", paths[2]])
    assert arm.n_rows == 24                       # pooled, not averaged
    assert np.allclose(arm.pred_ppm[:12], 45.0 - 0.1)   # ensembled in ppm, not pooled


def test_pooling_the_members_of_an_ensemble_is_refused():
    """Pooling counts every residue once per member and shrinks every interval; the two
    operations look identical at the command line, so the scorer separates them."""
    preds = _type_mean_predictor()
    with pytest.raises(ValueError, match="share rows"):
        evaluate.pool([preds, preds])


# ---------------------------------------------------------------------------
# Resolved targets
# ---------------------------------------------------------------------------

def test_a_dump_without_resolved_targets_is_refused(tmp_path):
    """INVARIANT: never score the raw labels. ``true_ppm``/``mask`` and
    ``resolved_true_ppm``/``effective_mask`` differ by a prefix, and the first pair
    scores every side-chain nucleus against targets no atom carries."""
    preds = _type_mean_predictor()
    path = _dump(tmp_path / "raw.npz", preds, resolved_true_ppm=None, effective_mask=None)
    with pytest.raises(ValueError, match="resolved_true_ppm"):
        evaluate.read_predictions(path)


def test_a_dump_whose_mask_kept_unscorable_cells_is_refused(tmp_path):
    """Ambiguity set codes (4/5/6/9) have no atom-specific target. A dump whose effective
    mask still contains them was written without the resolver, so its targets are raw."""
    preds = _type_mean_predictor()
    ambiguity = np.zeros_like(preds.ambiguity)
    ambiguity[:10, 0] = 4
    path = _dump(tmp_path / "unresolved.npz", preds, ambiguity=ambiguity)
    with pytest.raises(ValueError, match="canonical ambiguity-resolved mask"):
        evaluate.read_predictions(path)


def test_stored_resolved_truth_is_recomputed_not_trusted(tmp_path):
    """A dump cannot lower its reported error by fabricating the resolved target."""
    preds = _type_mean_predictor(rows=40)
    fabricated = preds.resolved_ppm.copy()
    fabricated[preds.effective_mask] = preds.pred_ppm[preds.effective_mask]
    path = _dump(tmp_path / "fabricated.npz", preds, resolved_true_ppm=fabricated)
    with pytest.raises(ValueError, match="canonical recomputation"):
        evaluate.read_predictions(path)


def test_pickled_string_arrays_require_explicit_trust(tmp_path):
    preds = _type_mean_predictor(rows=40)
    path = _dump(
        tmp_path / "pickled_strings.npz",
        preds,
        shift_cols=np.asarray(preds.shift_cols, dtype=object),
        label_residue_id=np.asarray(preds.row_key, dtype=object),
    )
    with pytest.raises(ValueError, match="unsafe by default"):
        evaluate.read_predictions(path)
    trusted = evaluate.read_predictions(path, allow_unsafe_pickle=True)
    assert trusted.n_rows == preds.n_rows


def test_the_ensemble_re_resolves_a_swap_against_its_own_prediction():
    """A prochiral pair is relabelled to whichever assignment the scored prediction
    fits; resolving against one member would score the ensemble on another model's
    targets."""
    columns = ("hb2_shift", "hb3_shift")             # a real SER prochiral pair
    rows = 10
    residue = np.full(rows, STANDARD_RESIDUES.index("SER"))
    true = np.tile([3.0, 4.0], (rows, 1))
    swapped = _population(np.tile([4.0, 3.0], (rows, 1)), true, residue=residue, columns=columns)
    direct = _population(np.tile([3.9, 3.1], (rows, 1)), true, residue=residue, columns=columns)
    combined = evaluate.ensemble([swapped, direct])
    # Both members sit closer to the swapped labelling, so the ensemble's targets follow.
    assert np.allclose(combined.resolved_ppm, np.tile([4.0, 3.0], (rows, 1)))
    assert evaluate.score(combined, boots=0)["hb2_shift"]["rmse"] < 0.2


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def test_the_bootstrap_resamples_label_sets_not_residues():
    """INVARIANT: residues within a protein are not independent. With every row in one
    label set there is exactly one block to resample, so every draw is the same set and
    the interval collapses -- a residue-level bootstrap would report a tight interval
    around the point estimate instead."""
    preds = _type_mean_predictor()
    one_protein = evaluate.Predictions(
        **{**preds.__dict__, "label_set": np.full(preds.n_rows, "ls0")})
    got = evaluate.score(one_protein, boots=64)["ca_shift"]
    assert got["n_label_sets"] == 1
    assert got["rmse_ci"][0] == pytest.approx(got["rmse"])
    assert got["rmse_ci"][1] == pytest.approx(got["rmse"])
    # The same cells, split across proteins, do give a real interval.
    many = evaluate.score(preds, boots=256)["ca_shift"]
    assert many["rmse_ci"][0] < many["rmse"] < many["rmse_ci"][1]


def test_the_bootstrap_refuses_a_dump_with_no_label_sets():
    preds = _type_mean_predictor()
    blank = evaluate.Predictions(**{**preds.__dict__, "label_set": np.full(preds.n_rows, "")})
    with pytest.raises(ValueError, match="degenerate to one block"):
        evaluate.score(blank, boots=16)


def test_a_confidence_interval_depends_on_the_seed_and_the_nucleus_only():
    """INVARIANT: one bootstrap stream per nucleus. A single rng consumed across the
    nucleus loop makes a column's interval depend on which columns precede it, so one
    nucleus reports one CI in the whole-vocabulary table and a different one in a
    backbone-only figure -- which reads as a panel disagreeing with the table beside it.
    Bit-identity, not approx: these are the same draws or they are not."""
    preds = _type_mean_predictor()
    alone = evaluate.score(preds, columns=["cb_shift"], boots=256)["cb_shift"]
    second = evaluate.score(preds, boots=256)["cb_shift"]                 # after ca_shift
    first = evaluate.score(preds, columns=["cb_shift", "ca_shift"], boots=256)["cb_shift"]
    assert alone["rmse_ci"] == second["rmse_ci"] == first["rmse_ci"]
    # The interval has to be a real one, or the equality above is vacuous.
    assert alone["rmse_ci"][0] < alone["rmse"] < alone["rmse_ci"][1]


def test_a_paired_interval_also_depends_on_the_seed_and_the_nucleus_only():
    baseline = _type_mean_predictor(seed=1)
    candidate = _population(baseline.pred_ppm + 0.05, baseline.true_ppm,
                            residue=baseline.residue_code, label_set=baseline.label_set,
                            keys=baseline.row_key)
    alone = evaluate.compare(baseline, candidate, columns=["cb_shift"], boots=256)
    both = evaluate.compare(baseline, candidate, boots=256)
    assert alone["cb_shift"]["ci"] == both["cb_shift"]["ci"]


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def test_comparing_mismatched_row_sets_raises():
    """INVARIANT: a paired comparison is paired. Two arms that merely overlap describe
    two different cohorts, so a mismatch is an error and never a plausible table."""
    baseline = _type_mean_predictor(seed=1)
    candidate = _population(baseline.pred_ppm[:-5], baseline.true_ppm[:-5],
                            residue=baseline.residue_code[:-5])
    with pytest.raises(ValueError, match="same rows"):
        evaluate.compare(baseline, candidate, boots=16)


def test_align_then_compare_scores_the_shared_rows():
    baseline = _type_mean_predictor(seed=1)
    trimmed = _population(baseline.resolved_ppm[:-5] + 0.05, baseline.true_ppm[:-5],
                          residue=baseline.residue_code[:-5], keys=baseline.row_key[:-5],
                          label_set=baseline.label_set[:-5])
    left, right = evaluate.align(baseline, trimmed)
    got = evaluate.compare(left, right, boots=128)["ca_shift"]
    assert got["n"] == baseline.n_rows - 5
    assert got["delta_rmse"] > 0 and got["verdict"] == "candidate"   # the arm with real skill


def test_align_matches_nuclei_by_NAME_across_arms_that_order_them_differently():
    """Two arms may carry the same nuclei in different cache orders. Aligning them by
    position would score CA against CB's target and still print a table."""
    residue = np.zeros(20, dtype=int)
    ca = np.full((20, 1), 55.0)
    cb = np.full((20, 1), 30.0)
    first = _population(np.hstack([ca, cb]) + 0.1, np.hstack([ca, cb]), residue=residue,
                        columns=("ca_shift", "cb_shift"))
    second = _population(np.hstack([cb, ca]) + 0.2, np.hstack([cb, ca]), residue=residue,
                         columns=("cb_shift", "ca_shift"))
    left, right = evaluate.align(first, second)
    assert left.shift_cols == right.shift_cols
    for arm in (left, right):
        for i, name in enumerate(arm.shift_cols):
            expected = 55.0 if name == "ca_shift" else 30.0
            assert np.allclose(arm.resolved_ppm[:, i], expected), name
    assert evaluate.compare(left, right, boots=16)["ca_shift"]["rmse_baseline"] \
        == pytest.approx(0.1)


def test_align_refuses_arms_whose_row_identities_repeat():
    """A repeated key makes intersect1d keep one occurrence, silently shrinking both
    arms to a handful of rows instead of pairing the wrong residues loudly."""
    preds = _type_mean_predictor()
    collided = evaluate.Predictions(
        **{**preds.__dict__, "row_key": np.full(preds.n_rows, "dup")})
    with pytest.raises(ValueError, match="uniquely identified"):
        evaluate.align(collided, collided)


def test_comparing_arms_that_read_different_labels_raises():
    """The two arms must be scoring one cache. Disagreeing ground truth means they are
    not, whatever their row keys say."""
    baseline = _type_mean_predictor(seed=2)
    candidate = evaluate.Predictions(
        **{**baseline.__dict__, "true_ppm": baseline.true_ppm + 0.5,
           "resolved_ppm": baseline.resolved_ppm + 0.5})
    with pytest.raises(ValueError, match="disagree on the deposited labels"):
        evaluate.compare(baseline, candidate, boots=16)


def test_a_paired_difference_must_carry_an_interval():
    """A model-versus-model delta reported as a point estimate is the shape every
    over-claim here has taken."""
    preds = _type_mean_predictor(seed=3)
    with pytest.raises(ValueError, match="confidence interval"):
        evaluate.compare(preds, preds, boots=0)


# ---------------------------------------------------------------------------
# Sources, round trip, and output hygiene
# ---------------------------------------------------------------------------

def test_the_matched_cohort_is_the_label_sets_every_source_realizes():
    """A marginal per-source difference is confounded with which proteins have a crystal
    structure; the matched cohort is the same protein seen three ways."""
    residue = np.zeros(6, dtype=int)
    true = np.full((6, 2), 45.0)
    preds = _population(true + 0.1, true, residue=residue,
                        label_set=["a", "a", "a", "b", "b", "c"], source=[0, 1, 2, 0, 1, 0])
    assert list(evaluate.matched_rows(preds)) == [True, True, True, False, False, False]


def test_cell_matching_requires_the_same_label_residue_and_nucleus_in_every_source():
    residue = np.zeros(6, dtype=int)
    true = np.full((6, 2), 45.0)
    mask = np.ones((6, 2), dtype=bool)
    mask[2, 1] = False
    predictions = _population(
        true + 0.1,
        true,
        residue=residue,
        mask=mask,
        source=[0, 1, 2, 0, 1, 2],
    )
    cells, codes = cohorts.matched_cells(
        predictions,
        ["residue-a", "residue-a", "residue-a", "residue-b", "residue-b", "residue-b"],
    )
    assert np.array_equal(codes, [0, 0, 0, 1, 1, 1])
    assert np.array_equal(cells[:3], [[True, False]] * 3)
    assert np.array_equal(cells[3:], [[True, True]] * 3)


def test_evaluate_adapter_and_statistics_api_are_exactly_equal():
    predictions = _type_mean_predictor()
    spec = statistics.BootstrapSpec(unit="label_set", resamples=64, seed=31)
    np.testing.assert_equal(
        evaluate.score(predictions, boots=64, seed=31, group="label_set"),
        statistics.score(predictions, bootstrap=spec),
    )


def test_generic_bootstrap_is_reproducible_per_stream():
    spec = statistics.BootstrapSpec(unit="bmrb", resamples=8, seed=17)
    first = statistics.bootstrap_multiplicities(3, spec, stream=4)
    second = statistics.bootstrap_multiplicities(3, spec, stream=4)
    assert np.array_equal(first, second)
    assert np.array_equal(first.sum(axis=1), np.full(8, 3))


def test_a_repeated_nucleus_name_is_refused():
    """A second vocabulary that repeats a name resolves every lookup to the first column,
    so the duplicate is scored twice and its own column never at all."""
    preds = _type_mean_predictor()
    with pytest.raises(ValueError, match="repeats a name"):
        evaluate.Predictions(**{**preds.__dict__, "shift_cols": ("ca_shift", "ca_shift")})


def test_a_dump_round_trips_through_the_writer(tmp_path):
    preds = _type_mean_predictor()
    path = evaluate.write_predictions(preds, tmp_path / "round.npz")
    again = evaluate.read_predictions(path)
    np.testing.assert_equal(evaluate.score(preds, boots=32), evaluate.score(again, boots=32))


def test_results_are_refused_at_a_repository_root(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="repository root"):
        evaluate.write_predictions(_type_mean_predictor(), tmp_path / "run.npz")


def test_the_report_is_json_serializable(tmp_path):
    report = evaluate.score(_type_mean_predictor(), boots=32)
    assert json.loads(json.dumps(report))["ca_shift"]["n"] == 2000


def test_the_command_line_announces_what_BOTH_tables_dropped(tmp_path, capsys):
    """INVARIANT: a shrinking denominator is a finding, not a formatting detail. A
    nucleus under the cell floor is simply absent from the result dict and
    ``_print_comparison`` iterates that dict, so the paired table -- where every
    model-vs-model claim comes from -- announces its own loss and not only the one the
    table above it reports. Pairing intersects two effective masks, so it drops more
    than ``score`` does, never fewer."""
    rng = np.random.default_rng(3)
    rows = 300
    residue = np.zeros(rows, dtype=int)
    true = np.stack([45.0 + rng.normal(0, 1.5, rows)] * 2, 1)
    mask = np.ones((rows, 2), dtype=bool)
    mask[5:, 1] = False                     # cb_shift observed in 5 rows: under the floor
    candidate = _population(true + 0.1, true, residue=residue, mask=mask)
    baseline = _population(true + 0.3, true, residue=residue, mask=mask,
                           keys=candidate.row_key, label_set=candidate.label_set)

    parser = argparse.ArgumentParser()
    evaluate.add_evaluate_arguments(parser)
    evaluate.evaluate_command(parser.parse_args([
        "--preds", str(_dump(tmp_path / "cand.npz", candidate)),
        "--baseline", str(_dump(tmp_path / "base.npz", baseline)), "--boots", "16"]))

    printed = capsys.readouterr().out
    floor = evaluate.MIN_SCORABLE_CELLS
    assert f"not scored (under {floor} observed cells): cb_shift" in printed
    assert f"not scored (under {floor} cells both arms scored): cb_shift" in printed
    assert printed.count("ca_shift") == 2, "the nucleus that IS scorable stays in both"


# ---------------------------------------------------------------------------
# Scoring a cache fold
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return build_cache(tmp_path_factory.mktemp("evalcache"))


def _checkpoint(cache, path, seed=0, scale=1.0):
    """An untrained model stamped exactly the way ``train`` stamps one, so the scorer
    reads its stats and nucleus order from the file and not from the cache. ``scale``
    stretches the stamped per-nucleus scale, which is the thing that differs between two
    real CV members."""
    config = json.loads((cache / f"fold_{HELDOUT_FOLD}" / "config.json").read_text())
    columns = list(config["shift_cols"])
    raw = dict(config["stats"])
    raw["snuc"] = {col: value * scale for col, value in raw["snuc"].items()}
    torch.manual_seed(seed)
    model = AtomTransformer(len(ATOM_TYPES), len(columns), dim=16, layers=1)
    torch.save({"model_state_dict": model.state_dict(), "dim": 16, "layers": 1,
                "n_shifts": len(columns), "shift_cols": columns,
                "stats": normalization.build_fixed_per_aa(raw, columns),
                "recipe": FINAL_ATOM_CLOUD_RECIPE.with_vocabulary(
                    len(ATOM_TYPES), AtomTransformer.N_OFFSET_TYPES).to_dict()}, path)
    return path


def test_one_checkpoint_scores_a_fold_through_the_trainer_evaluation(cache, tmp_path):
    """The scorer hands a member to the trainer's evaluation with the partner table and
    nothing else, and the block it gets back covers every row of the fold with the
    checkpoint's own nucleus order; the CPU is enough to prove the call, and the scorer
    still refuses it for reported numbers."""
    ckpt = _checkpoint(cache, tmp_path / "one.pt")
    preds = evaluate.score_one_checkpoint(
        ckpt, cache / f"fold_{HELDOUT_FOLD}", flags={}, device="cpu", workers=0)
    config = json.loads((cache / f"fold_{HELDOUT_FOLD}" / "config.json").read_text())
    assert list(preds.shift_cols) == list(config["shift_cols"])
    assert preds.n_rows == len(np.load(cache / f"fold_{HELDOUT_FOLD}" / "samples.npy"))
    assert np.isfinite(preds.pred_ppm).all()


def test_scoring_a_cache_fold_refuses_the_cpu(cache, tmp_path):
    """INVARIANT: reported numbers never come off a CPU run."""
    ckpt = _checkpoint(cache, tmp_path / "cpu.pt")
    with pytest.raises(RuntimeError, match="requires a GPU"):
        evaluate.score_checkpoints(cache, HELDOUT_FOLD, [ckpt], flags={}, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="scoring a cache fold needs a GPU")
def test_the_cache_path_and_the_dump_path_score_the_same_population(cache, tmp_path):
    """The two entry points are one scorer: a fold scored straight from the cache and the
    same population written out and read back must give the identical table."""
    ckpts = [_checkpoint(cache, tmp_path / f"m{i}.pt", seed=i) for i in (0, 1)]
    preds = evaluate.score_checkpoints(cache, HELDOUT_FOLD, ckpts, flags={}, workers=0)
    assert preds.mask.dtype == bool and preds.n_rows > 0
    again = evaluate.read_predictions(evaluate.write_predictions(preds, tmp_path / "cache.npz"))
    assert np.array_equal(preds.row_key, again.row_key)
    assert np.array_equal(preds.effective_mask, again.effective_mask)
    assert np.allclose(preds.pred_ppm, again.pred_ppm)
    np.testing.assert_equal(evaluate.score(preds, boots=16), evaluate.score(again, boots=16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="scoring a cache fold needs a GPU")
def test_each_member_is_denormalized_through_its_own_checkpoint(cache, tmp_path):
    """INVARIANT: a model is never scored through another model's normalization. Two
    checkpoints differing only in their stamped scale must give different ppm, and the
    ensemble must be their ppm mean."""
    narrow = _checkpoint(cache, tmp_path / "narrow.pt", scale=1.0)
    wide = _checkpoint(cache, tmp_path / "wide.pt", scale=2.0)   # same weights, wider scale

    one = evaluate.score_checkpoints(cache, HELDOUT_FOLD, [narrow], flags={}, workers=0)
    other = evaluate.score_checkpoints(cache, HELDOUT_FOLD, [wide], flags={}, workers=0)
    assert not np.allclose(one.pred_ppm, other.pred_ppm)
    both = evaluate.score_checkpoints(cache, HELDOUT_FOLD, [narrow, wide], flags={}, workers=0)
    assert np.allclose(both.pred_ppm, (one.pred_ppm + other.pred_ppm) / 2)
