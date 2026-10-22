# Tau conformational-sensitivity benchmark (Figure S11, panel b)

Reproduces the comparison of DeepShift, UCBShift 2.0, MLFF and SHIFTX2 on the benchmark of
Stöckelmaier and Oostenbrink (PCCP 2024, doi:10.1039/D4CP02484B): ten conformers of a
31-residue proline-rich tau fragment, five in a globular and five in a stretched state. For
every (residue, atom) site the statistic is

    S = |mean(globular) - mean(stretched)| / pooled within-state sample SD

over the five conformers of each state, and a site is state-sensitive when S >= 3.29. The
four predictors are compared on the 215 sites all four cover. The conformers are the
authors' released coordinates; DeepShift reads them as released, UCBShift 2.0 and MLFF read
copies prepared as described below.

## Inputs

| input | where from |
|---|---|
| Stöckelmaier archive | Zenodo record [13124979](https://doi.org/10.5281/zenodo.13124979), `stoeckelmaier_chemical_Shifts_revision3.zip` (1.36 GB, CC BY 4.0). Unzip; `<archive>` below is the folder `stoeckelmaier2024_chemicalShifts_toolchain_revision3` |
| DeepShift members | `../../models/deepshift_fold{1..5}.pt` (`pip install -e .` in the repository root) |

The conformers are `<archive>/PDB_files/04_freesasa/tau_apo_openmm_*.pdb`. The SHIFTX2 arm
is not run: the archive carries the authors' own SHIFTX2 predictions
(`<archive>/shiftX2/cs_{globular,stretched}.dat`), and those are what is scored.

## Third-party software

| component | version used | source and licence |
|---|---|---|
| UCBShift 2.0 | THGLab/CSpred commit `844e265`, `../competitors/patches/ucbshift2.patch` applied; Python 3.8.20, scikit-learn 0.22.2, Biopython 1.74, NumPy 1.19.5, pandas 1.1.5, BLAST+ 2.9.0, mTM-align 20180725, DSSP 2.04 | <https://github.com/THGLab/CSpred>; UC Regents, educational, research and not-for-profit use. The 141 trained models are at <https://doi.org/10.5281/zenodo.15375968> (`models.zip`, unpacked into one folder) |
| REDUCE | 4.15 (`reduce.4.15.250408`) | <https://github.com/rlabduke/reduce>; BSD-style |
| OpenMM, PDBFixer | OpenMM 8.5.2, PDBFixer 1.12.0, Python 3.11 | <https://openmm.org>, MIT; force fields `amber99sb.xml` and `amber99_obc.xml` ship with OpenMM |
| MLFF predictor | `mb012/MLFF_representation` commit `9e4c984` (MIT), with the released heads `models/all_cs_models/cs_mace` from the Dropbox link in its README (arXiv:2505.23354, Bojan et al.) | <https://github.com/mb012/MLFF_representation> |
| MACE | mace-torch 0.3.14, torch 2.9.1, e3nn 0.4.4, ase 3.27.0, matscipy 1.2.0, Python 3.12; MACE-OFF23 large, downloaded by `mace_off` on first use | <https://github.com/ACEsuit/mace>, MIT; the MACE-OFF23 weights are under the Academic Software License (non-commercial) |
| SHIFTX2 | not run; the authors' released output is read | Wishart lab, <http://www.shiftx2.ca> |

`../competitors/patches/ucbshift2.patch` (against the commit above) makes the test-mode
exclusion of transfer references use a sequence-identity cutoff, `UCB_SELF_IDENT_CUTOFF`,
default 0.90, in addition to UCBShift's coverage rule, and replaces one truthiness test on a
Biopython atom by `is not None`. Its `UCBY_*` switches serve the fold-6 benchmark and stay
unset here. Apply it from the checkout root: `patch -p1 < <repository>/experiments/competitors/patches/ucbshift2.patch`.

## Run

Environment variables: `UCBSHIFT2_HOME` (the checkout, with its `bins/`), `UCBSHIFT2_MODELS`,
`UCBSHIFT_PYTHON`, `REDUCE_BIN`, `OPENMM_PYTHON`, `MACE_PYTHON`, `MLFF_HOME`, `MLFF_MODELS`.

```bash
ARCH=<unzipped>/stoeckelmaier2024_chemicalShifts_toolchain_revision3
OUT=<work directory>

# DeepShift: all five members on every conformer (the fragment lies in fold 0, which no member trained on)
python predict_deepshift.py --benchmark-dir $ARCH --out $OUT

# UCBShift 2.0: caps and hydrogens removed, REDUCE -build, CSpred.py -t
python ucbshift2_arm.py prepare --benchmark-dir $ARCH --out $OUT
python ucbshift2_arm.py run     --benchmark-dir $ARCH --out $OUT

# MLFF: Amber99SB/OBC relaxation, then REDUCE polish and the MACE predictor
$OPENMM_PYTHON mlff_arm.py relax   --benchmark-dir $ARCH --out $OUT
$MACE_PYTHON   mlff_arm.py predict --out $OUT

# the four-predictor table
python score_tau.py --benchmark-dir $ARCH --deepshift $OUT/deepshift_long.csv \
    --ucbshift2 $OUT/ucbshift2_csv --mlff $OUT/mlff --out $OUT/score
```

`predict_deepshift.py` and `mlff_arm.py predict` default to `--device cuda`; the reported
DeepShift numbers were computed on a GPU (fp16 autocast), and a CPU run differs by up to
about 0.004 ppm in a prediction. The OpenMM minimiser on OpenCL is not bit-reproducible: two
relaxations of one conformer differ by about 0.04 A heavy-atom RMSD, which is the
difference a CPU run shows against the stored structures. The relaxed conformers the
published MLFF values were computed from are deposited with the dataset
(`deepshift_mlff_relaxed_structures.tar.gz`, folder `tau/`); copy them into `$OUT/relaxed/`
in place of `mlff_arm.py relax` to reproduce them.
`mlff_predictor.py` is the same file as in `../decoys/`.

## What the release's producers read

`score_tau.py` writes `$OUT/score/common_cells.csv`: one row per predictor and site, with
`method`, `position`, `atom`, `sensitivity` and the state means and SDs. It is the only
file read downstream.

| producer | variable | file |
|---|---|---|
| `figures/fig_downstream_applications.py` (Figure S11b; the text's 40.93, 24.19, 20.93 and 15.81 % are the shares with S >= 3.29) | `DEEPSHIFT_TAU_BENCHMARK_ROOT=$OUT/score` | `common_cells.csv` |
