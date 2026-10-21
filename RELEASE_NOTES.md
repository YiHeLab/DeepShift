# DeepShift v1.0

Structure-only prediction of protein chemical shifts. An atom transformer reads a 3D
structure through interatomic distances alone and predicts 64 nuclei per residue: no
homology transfer, no sequence features, no retrieval.

## What is in this release

- `models/deepshift_fold{1..5}.pt` — the five-member cross-validation ensemble the paper
  reports. Each member holds out its own fold and carries its own normalization statistics.
- `models/interval_calibration.json` — the calibration behind the 90 % prediction intervals,
  fitted on the excluded fold and evaluated once on the held-out fold.
- `data/deepshift_{shifts,structures,sequences}.parquet` — the dataset. The shifts table
  holds 17,118 label sets; the 12,710 of them with a fold are the paper's dataset, over
  12,242 BMRB depositions, realized as 21,346 structure pairings. Shifts are rereferenced
  and the exclusion fence is applied; `data/README.md` is the dictionary.
- The scripts behind every figure and table in the paper, under `figures/` and `si_build/`,
  and the competitor scorers under `benchmarks/`.

## Getting predictions

`colab/DeepShift_predict.ipynb` runs in Colab with no local setup: give it a PDB id or a
file and it returns per-residue shifts with prediction intervals. Locally, `pip install .`
then:

```bash
deepshift predict --pdb figures/reference/1POH.pdb --chain A \
  --checkpoint models/deepshift_fold1.pt --checkpoint models/deepshift_fold2.pt \
  --checkpoint models/deepshift_fold3.pt --checkpoint models/deepshift_fold4.pt \
  --checkpoint models/deepshift_fold5.pt \
  --interval-calibration models/interval_calibration.json --out shifts.csv
```

## Reproducing the paper

`REPRODUCTION.md` states what reproduces from this release alone and what needs a
third-party file. `PROTOCOL.md` is the recipe for rebuilding the dataset and retraining
the five models.

## Note on third-party code

SHIFTX is not redistributed here. It is fetched from its authors' own server at build time
against a pinned checksum; see `THIRD_PARTY_NOTICES.md`.
