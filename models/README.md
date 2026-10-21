# Shipped checkpoints: the cross-validation ensemble

`deepshift_fold{1..5}.pt` are the five members of the entity-CV ensemble, trained as
described in `PROTOCOL.md` section 3. Member *f* holds out fold *f* and trains on the other
four of folds 1 to 5. `interval_calibration.json` is the prediction-interval artifact
fitted on fold 0 (`PROTOCOL.md` section 4).

These are Git LFS objects. Run `git lfs install && git lfs pull` after cloning, or the
files are pointer stubs and loading them fails.

Each member is a structure-only atom transformer: atom types and coordinates only, with no
pH or temperature input and no sequence features.

## What a file holds

`deepshift predict` loads each file with `torch.load(..., weights_only=True)` and reads:

* `model_state_dict`, `dim`, `layers`: the weights and the shape they fill. All weights
  must load into the model without a missing or extra key.
* `shift_cols`, `n_shifts`: the output columns, in order, unique, `n_shifts` of them.
* `stats`: for every output column a global `mean` and `std` (finite, `std` positive);
  `snuc`, `snuc_counts`, `per_aa` and the `normalization` marker, which turn the model's
  output back into ppm; and `normalization_fit`, which records the fit (held-out fold, fit
  folds, folds 0 and 6 always excluded, rereferencing before the fit, flagged cells
  excluded). A member's per-nucleus scale `snuc` is fitted on its four training folds alone.
* `recipe`: the atom cloud the member was trained on, with the atom vocabulary. `predict`
  builds the cloud from a PDB file with this recipe.

## How the five are averaged

Each member predicts in its own normalization, `predict` converts that prediction to ppm
with the member's own `stats`, and the five ppm values are averaged, never the model's
outputs. The spread across the five (`ensemble_sd_ppm`, with `ddof=1`) is mapped to the 90 %
half-width `ci90_halfwidth_ppm` for the six backbone nuclei by `interval_calibration.json`,
which requires exactly these five members.

`predict` rejects an ensemble whose members disagree on shift columns, atom vocabulary,
atom-cloud recipe or normalization marker. Averaging is defined only over the five members
of one training run.
