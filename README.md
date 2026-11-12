# DeepShift

DeepShift predicts protein NMR chemical shifts from a three-dimensional structure with an
8 Å heavy-atom transformer. The model predicts 64 nuclei: six backbone and 58 side-chain
targets.

The code is licensed MIT (`LICENSE`); the dataset tables under `data/` are CC-BY-4.0
(`data/README.md`). Third-party components and their licenses are listed in
`THIRD_PARTY_NOTICES.md`. SHIFTX is not distributed here: the build downloads it from
upstream against a pinned SHA-256.

Predictions run in Google Colab with no local install:
[DeepShift_predict.ipynb](https://colab.research.google.com/github/YiHeLab/DeepShift/blob/main/colab/DeepShift_predict.ipynb).
The notebook reports one anonymous usage record per prediction to the authors (no
coordinates unless you opt in; `colab/README.md`); set `TELEMETRY = False` in it to send
nothing.
The dataset, the frozen inputs of its build and the measurements behind every figure and
table are archived at Zenodo, [doi:10.5281/zenodo.23550866](https://doi.org/10.5281/zenodo.23550866)
(every version; it resolves to the newest).

## Install

DeepShift requires Python 3.11 or newer. It is tested on Ubuntu (CI); macOS and Windows
are untested. No GPU is needed: prediction uses a CUDA GPU when one is visible and the CPU
otherwise.

The checkpoints and the published dataset are stored with Git LFS (about 360 MB). Install
`git-lfs` first (apt, Homebrew or conda). A clone made without it yields pointer files of
about 130 bytes, and `torch.load` fails on them with an unpickling error and no useful
message:

```bash
git lfs install
git lfs pull
```

Then:

```bash
python -m pip install .
```

On a machine without a GPU, install the CPU build of PyTorch first. The default wheel from
PyPI is the CUDA build, about 900 MB plus several GB of NVIDIA libraries; with the CPU build
the whole install took 1.5 min in our test:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

For development, and to regenerate the figures and supplementary tables:

```bash
python -m pip install -e '.[test,figures]'
python -m pytest -q
```

The wheel carries three JSON files under `deepshift/data/`: the perturber rule the atom
stage applies (`hetatm_exclusion_rule.json`) and the two BMRB lists the fold split holds
out (`ucbshift200_excluded_bmrbs.json`, `case_study_excluded_bmrbs.json`). It does not
include model checkpoints, datasets, the fitted p100 cutoff file, or an interval-calibration
artifact; the p100 cutoffs and the prediction intervals are fitted from one build: the
cutoffs are supplied explicitly and the interval artifact sits beside the checkpoints it was
fitted for, so constants from another model are not used by accident.

### System tools some paths need

The package itself needs only what `pyproject.toml` declares. Four of the reproduction
paths shell out to tools that pip cannot install, and one benchmark needs a package that
is not a runtime dependency:

| tool | needed by | how it is found |
|---|---|---|
| `blastp`, `makeblastdb` (BLAST+) | crystal discovery and the structure census | `BLASTP_BIN`, `MAKEBLASTDB_BIN` |
| `pdftotext` (poppler-utils) | the literature table, which parses competitors' own articles | on `PATH` |
| `freesasa` (pip) | `benchmarks/score_error_covariates.py`, the error-covariate table | imported at call time |
| `pymol` (conda) | `figures/render_structures.py`, the molecular-structure panels | `--pymol`, `$PYMOL`, or on `PATH` |

Two competitor-benchmark scripts also read data that is not redistributable and is located
by environment variable: `DEEPSHIFT_SIFTS_CSV` (the SIFTS chain-to-accession table) and
`DEEPSHIFT_REFDB_SHIFTS` (the RefDB shift tree from a CSpred checkout).

## Predict from a PDB

From the repository root, with the bundled structure 1POH (85 residues) and the five
ensemble members:

```bash
deepshift predict --pdb figures/reference/1POH.pdb --chain A \
  --checkpoint models/deepshift_fold1.pt --checkpoint models/deepshift_fold2.pt \
  --checkpoint models/deepshift_fold3.pt --checkpoint models/deepshift_fold4.pt \
  --checkpoint models/deepshift_fold5.pt \
  --out shifts.csv
```

This writes `shifts.csv` (85 rows) and `shifts.intervals.csv` (496 rows); the interval
file comes from `models/interval_calibration.json`, which `predict` uses by default when all
five members are given. On 20 CPU cores it takes 40 s; one member alone takes 8 s. A
396-residue protein (maltose-binding protein) takes 2 min 42 s with five members and 34 s
with one.

**Input.** `--pdb` takes a local PDB-format file; a PDB ID is not fetched. AlphaFold DB
models work as they are (`AF-<UniProt>-F1-model_v6.pdb`, chain A). Hydrogens are ignored.
For a multi-model file (an NMR ensemble), the model closest to the ensemble median is used.

**Output.** `shifts.csv` has one row per residue: `residue_id`, `residue_name`,
`atom_cloud_truncated` (true when the 8 Å cloud exceeded the 512-atom cap), then 64
`<atom>_shift` columns in ppm, blank where the residue type has no such atom.
`shifts.intervals.csv` has one row per residue and backbone nucleus: `residue_id`,
`residue_name`, `nucleus`, `ensemble_sd_ppm`, `ci90_halfwidth_ppm`. The Colab notebook
writes a different, long-format table (`colab/README.md`).

Any compatible subset of the checkpoints can be supplied with repeated `--checkpoint`;
`--cpu` forces the CPU when a GPU is visible.

The structure is parsed once. Each checkpoint is validated, run in its own normalization
frame, converted to ppm, and then ensemble-averaged. Cross-chain protein partners are
excluded by default. Name each protein chain that was present in the NMR tube with a
repeated `--sample-partner-chain CHAIN`; every other coordinate chain is excluded. For
example, add `--sample-partner-chain B` when chain B was in the measured sample.

When all five ensemble members are given, a second file (`shifts.intervals.csv`) is written
beside the predictions: one row per backbone cell with `ensemble_sd_ppm` (the spread of the
five members' predictions) and `ci90_halfwidth_ppm`. By default the interval calibration is
`interval_calibration.json` in the directory of the first `--checkpoint`;
`--interval-calibration FILE` names another, and `--no-intervals` writes no interval file.
The variance mapping and split-conformal multipliers are fitted on fold 0 only; fold 6 is
used once to report coverage. Side-chain nuclei get no interval because member
disagreement there can reflect prochiral labelling. With any member count other than five,
or with no calibration file, `predict` writes the predictions alone and prints one line
saying why.

Checkpoint loading checks the file it reads. A checkpoint must declare a unique output
schema and a finite, positive ppm transform for every output, and its weights must fill
the model exactly. PyTorch weights are loaded with `weights_only=True`. `predict` refuses
an ensemble whose members differ in shift columns, atom vocabulary, atom-cloud recipe or
normalization marker.

`models/deepshift_fold{1..5}.pt` are the five ensemble members. Each file carries its own
shift columns, normalization statistics and atom-cloud recipe; `models/README.md` says
what `predict` reads from them.

## Evaluate

The scorer reports per-nucleus RMSE and MAE. Correlation and amplitude use secondary
shifts; the scorer's bootstrap is clustered over label sets, never individual residues.

```bash
deepshift evaluate \
  --preds fold1.preds.npz fold2.preds.npz \
  --out heldout.json
```

A prediction archive carries non-pickled Unicode arrays. An archive holding pickled object
arrays is rejected unless `--allow-unsafe-legacy-pickle` is supplied explicitly, which is
how such an archive is trusted. Stored ambiguity-resolved targets are recomputed and
checked against the stored values before scoring. `deepshift evaluate --cache CACHE --fold
F --ckpt CKPT... --flags FENCE.json` scores checkpoints on a cache fold directly (this needs a GPU) and
`--save-npz` writes the scored population as a dump.

## Where the science is

The package is small; the parts a reader of the paper needs are these, in reading order.

| Module | What it holds |
|---|---|
| `src/deepshift/atom_cloud.py` | what the model sees: the contact-only heavy-atom cloud within 8 Å of each residue, capped at 512 atoms, with the sample-matched partner rule |
| `src/deepshift/model.py` | the network: radial-basis distance expansion, distance-biased attention blocks, per-nucleus heads |
| `src/deepshift/normalization.py` | ppm to model units and back: fixed residue-and-nucleus centres, one per-nucleus scale per member fitted on its own training folds |
| `src/deepshift/loss.py` | chemical equivalence and prochiral swaps in the training loss |
| `src/deepshift/training.py` | one member per held-out fold; the cache it reads and the checkpoints it writes |
| `src/deepshift/prediction.py` | inference from a PDB file with no cache: per-member ppm, then the ensemble mean, then the intervals |
| `src/deepshift/evaluate.py`, `statistics.py` | the scorer behind the accuracy comparisons: per-nucleus RMSE and MAE, secondary-shift statistics, the label-set-clustered bootstrap and paired comparisons |
| `src/deepshift/calibration.py` | prediction intervals: the variance fit and split-conformal multipliers on fold 0, evaluated once on fold 6 |
| `src/deepshift/quality_control.py` | the exclusion fence: fitted p100 accepted spans plus the structural cysteine and proline consistency rules |
| `src/deepshift/data/cluster.py`, `data/folds.py` | the entity-grained split: the sequence and structure edges that must not cross folds, fold 0 assigned by rule, and the draw of folds 1 to 6 |
| `src/deepshift/self_exclusion.py` | same-protein exclusion for the competitor benchmarks, verified from each tool's own retrieval log |
| `benchmarks/score_by_source_tier.py`, `score_paired_tier.py` | the competitor comparison the benchmark figures and tables are drawn from |

Everything else is plumbing: the data acquisition and cache stages under
`src/deepshift/data/` (fetch, assembly, discovery, materialization, rereferencing, the
cache and atom arrays), the machine-readable protocol, and one script per figure and
table.

## Reproducing the paper

Training, rebuilding the dataset from public sources, and regenerating every figure and
table the manuscript prints are covered in **`REPRODUCTION.md`**, together with what each
one needs and what cannot be reproduced from the release alone. The recipe for rebuilding
the dataset and retraining the five models, with the condition each step must meet, is
**`PROTOCOL.md`**. The programs that ran the competitors and the case studies (X-EISD,
IDP8, tau, the 3DRobot decoys) are under `experiments/`, with the steps that install each
third-party predictor in `experiments/competitors/README.md`.

## Repository layout

```text
src/deepshift/       model, inference, evaluation, training, and data stages
models/              the five ensemble checkpoints and their interval calibration (Git LFS)
data/                the released Parquet export and its data dictionary (Git LFS)
colab/               a hosted notebook that predicts from a PDB with no local install
scripts/             dataset export, the UniProt mapping, the p100 fit, the rereferencing diagnostics
figures/             one program per manuscript figure
si_build/            one program per supplementary table
benchmarks/          the competitor scoring the benchmark figures and tables are built from
experiments/         the programs that ran the competitors and the case studies
tests/               unit and invariant checks
REPRODUCTION.md      every deployed figure and table, its producer, and what it needs
PROTOCOL.md          the recipe: folds, quality control, training, calibration, dataset export
```
