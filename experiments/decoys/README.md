# Decoy discrimination on the 3DRobot set (Figure 4 decoy panels, Figures S9, Table S25)

Reproduces the comparison of DeepShift, UCBShift-X and the MLFF predictor on 52 targets of
the 3DRobot decoy set. Each target has 301 candidates: the experimental structure and 300
decoys with their C-alpha RMSD. A candidate's score is `mean_r`, the equal-weight mean of
the Pearson r between predicted and observed shifts for H, Hα and N, over the cells all
three arms predicted. For each arm and target the rank of the experimental structure and
the ordering of the 300 decoys by RMSD (Pearson, Spearman, Kendall) are computed; the
intervals resample targets.

Each target is predicted by one released member that never saw it (`build_cohort.py`): a
target whose label sets lie in fold k is predicted by member k, and a target in fold 0,
fold 6 or without a fold by a member chosen from its name. UCBShift-X reads the released
coordinates; MLFF reads Amber-relaxed ones, since the predictor takes relaxed input only.

## Inputs

| input | where from |
|---|---|
| 3DRobot decoy sets | Deng, Jia and Zhang, Bioinformatics 32:378 (2016); `3DRobot_set.tar.bz2` from the "3DRobot Decoy Sets" link on <https://zhanggroup.org/3DRobot/> (academic e-mail required; the page states no licence). It holds one `<target>.tar.bz2` per target; unpack the 52 of `roster.tsv` into `<3dr>/<target>/` (`native.pdb`, `list.txt` and 300 decoys) |
| `roster.tsv` | the 52 targets with a BMRB entry verified to be the same protein, and the chain scored (kept here) |
| observed shifts and folds | `../../data/deepshift_shifts.parquet` |
| DeepShift members | `../../models/deepshift_fold{1..5}.pt` |

## Third-party software

| component | version used | source and licence |
|---|---|---|
| UCBShift 1.0 | THGLab/CSpred commit `20cee31` with `../competitors/patches/ucbshift.patch` applied (first chain only, a Biopython-compatible BLOSUM62 load, environment-variable switches the X columns do not use); Python 3.8.20, scikit-learn 0.22.2, Biopython 1.74, BLAST+ 2.9.0, mTM-align, DSSP | <https://github.com/THGLab/CSpred>; UC Regents, educational, research and not-for-profit use. Models: the `models.tgz` of the repository's README (Dryad) |
| REDUCE | 4.15 (`reduce.4.15.250408`) | <https://github.com/rlabduke/reduce>; BSD-style |
| OpenMM, PDBFixer | OpenMM 8.5.2, PDBFixer 1.12.0 | <https://openmm.org>, MIT |
| MLFF predictor, MACE | as in `../tau/README.md` (`mb012/MLFF_representation` `9e4c984`, mace-torch 0.3.14, MACE-OFF23 large under the Academic Software License) | |

Only the structure-only `H_X`, `HA_X` and `N_X` columns of UCBShift are scored, so its
transfer module does not enter.

## Run

Environment variables: `UCBSHIFT_HOME` (UCBShift 1.0 checkout), `UCBSHIFT_PYTHON`,
`REDUCE_BIN`, `OPENMM_PYTHON`, `MACE_PYTHON`, `MLFF_HOME`, `MLFF_MODELS`.

```bash
DR=<3dr>      # the unpacked targets
OUT=<work directory>

python build_cohort.py --candidates $DR --out $OUT/cohort.json
python run_inference.py --cohort $OUT/cohort.json --candidates $DR --out $OUT/deepshift
python ucbshift_x.py --cohort $OUT/cohort.json --candidates $DR --out $OUT/ucbshift --workers 12
$OPENMM_PYTHON relax_decoys.py --cohort $OUT/cohort.json --candidates $DR --out $OUT/relaxed
$MACE_PYTHON   mlff_decoys.py  --cohort $OUT/cohort.json --relaxed $OUT/relaxed --out $OUT/mlff
python score_decoys.py --cohort $OUT/cohort.json --deepshift $OUT/deepshift \
    --ucbshift $OUT/ucbshift --mlff $OUT/mlff --out $OUT/pooled
```

`run_inference.py` and `mlff_decoys.py` default to `--device cuda`; the reported DeepShift
numbers were computed on a GPU (fp16 autocast), and a CPU run differs by up to about
0.01 ppm in a prediction. `relax_decoys.py` and `mlff_decoys.py` take `--shard`/`--nshards`. The relaxed candidates
the published MLFF values were computed from are deposited with the dataset
(`deepshift_mlff_relaxed_structures.tar.gz`, folder `decoys/`); pass that folder to
`mlff_decoys.py --relaxed` in place of `$OUT/relaxed` to reproduce them.
The OpenMM minimiser on OpenCL is not bit-reproducible (about 0.03 A heavy-atom RMSD
between runs). `mlff_predictor.py` is the same file as in `../tau/`.
`score_decoys.py` reads observed N shifts from the released table, which carries the
per-deposition rereferencing offset; a constant offset does not change a correlation.

## What the release's producers read

`score_decoys.py` writes three files into `$OUT/pooled`; `$OUT/cohort.json` is the fourth.

| producer | variable | files |
|---|---|---|
| `figures/fig_structure_response.py` (Figure 4 decoy panels, Figure S9) | `DEEPSHIFT_DECOY_POOLED=$OUT/pooled` | `relaxed_full_candidate_scores.parquet`, `relaxed_full_target_metrics.csv` |
| `si_build/build_decoy_ledger.py` (Table S25) | `DEEPSHIFT_DECOY_POOLED=$OUT/pooled`, `SI_DECOY_COHORT=$OUT/cohort.json` | the same two, `relaxed_full_comparison.json` (paired intervals, DeepShift prediction files), and the cohort's `single_member` and `clean_for_all_members` |
