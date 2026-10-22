# DeepShift v1.0

Structure-only prediction of protein chemical shifts. An atom transformer reads a 3D
structure through interatomic distances alone and predicts 64 nuclei per residue: no
homology transfer, no sequence features, no retrieval.

## What is in this release

- `models/deepshift_fold{1..5}.pt` — the five-member cross-validation ensemble the paper
  reports. Each member holds out its own fold and carries its own normalization statistics.
- `models/interval_calibration.json` — the calibration behind the 90 % prediction intervals,
  fitted on fold 0 and evaluated once on fold 6.
- `data/deepshift_{shifts,structures,sequences}.parquet` — the dataset. The shifts table
  holds 17,118 label sets; the 12,710 of them with a fold are the paper's dataset, over
  12,242 BMRB depositions, realized as 21,346 structure pairings. Shifts are rereferenced
  and the exclusion fence is applied; `data/README.md` is the dictionary.
- The scripts behind every figure and table in the paper, under `figures/` and `si_build/`,
  and the competitor scorers under `benchmarks/`.
- The programs that ran the competitors and the case studies (X-EISD, IDP8, tau, the
  3DRobot decoys), under `experiments/`, with the install steps of every third-party
  predictor.
- `scripts/map_uniprot.py`, the procedure that made the frozen BMRB-to-UniProt mapping.

## Getting predictions

`colab/DeepShift_predict.ipynb` runs in Colab with no local setup
([open in Colab](https://colab.research.google.com/github/YiHeLab/DeepShift/blob/main/colab/DeepShift_predict.ipynb)):
give it a PDB id or a file and it returns per-residue shifts with prediction intervals. Locally, `pip install .`
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
the five models. The build inputs (the BMRB, UniProt and PDB snapshots of July 2026, the
coordinate files and the p100 fence), the relaxed structures the MLFF predictor was run on
and the measurements behind every figure and table are deposited at Zenodo,
[doi:10.5281/zenodo.23550867](https://doi.org/10.5281/zenodo.23550867).

## How the released files were made

The dataset and the five members were made with the development version of this code.
Preparing the release removed record-keeping (input hashes, provenance files, run stamps)
and options the paper does not report. Rebuilding the dataset from the deposited inputs
with this release gives the three Parquet tables in `data/` byte for byte
(`REPRODUCTION.md`). The training code is the code that trained the members with those
options removed; the members were not retrained with it. Training is not bit-reproducible
across GPUs (members 1 to 3 were trained on RTX 5090s, 4 and 5 on A100s), so a member
retrained with `PROTOCOL.md` is not bit-identical to the released one.

## Note on third-party code

SHIFTX is not redistributed here. It is fetched from its authors' own server at build time
against a pinned checksum; see `THIRD_PARTY_NOTICES.md`. No competitor's source is included
either: UCBShift, UCBShift 2.0 and SHIFTX2 were run with the changes in
`experiments/competitors/patches/`, applied to the upstream releases named there.
