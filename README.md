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
The dataset, the frozen inputs of its build and the measurements behind every figure and
table are archived at Zenodo, [doi:10.5281/zenodo.23424002](https://doi.org/10.5281/zenodo.23424002).

## Install

DeepShift requires Python 3.11 or newer.

The checkpoints and the published dataset are stored with Git LFS. A clone made without it
yields pointer files of about 130 bytes, and `torch.load` fails on them with an
unpickling error and no useful message:

```bash
git lfs install
git lfs pull
```

Then:

```bash
python -m pip install .
```

For development, and to regenerate the figures and supplementary tables:

```bash
python -m pip install -e '.[test,figures]'
python -m pytest -q
```

The wheel carries the chemistry and perturber-rule JSON read at runtime. It does not
include model checkpoints, datasets, the fitted p100 cutoff file, or an interval-calibration
artifact; the p100 cutoffs and the prediction intervals are fitted from one build and are
supplied explicitly, so constants from another model cannot be used by accident.

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

Supply one or more compatible checkpoints explicitly:

```bash
deepshift predict \
  --pdb protein.pdb \
  --chain A \
  --checkpoint models/deepshift_fold1.pt \
  --checkpoint models/deepshift_fold2.pt \
  --out shifts.csv
```

The structure is parsed once. Each checkpoint is validated, run in its own normalization
frame, converted to ppm, and then ensemble-averaged. Cross-chain protein partners are
excluded by default. Name each protein chain that was present in the NMR tube with a
repeated `--sample-partner-chain CHAIN`; every other coordinate chain is excluded. For
example, add `--sample-partner-chain B` when chain B was in the measured sample.

When all five ensemble members and their matching `--interval-calibration` artifact are
given, a second file (`shifts.intervals.csv`) is written beside the predictions: one row per
backbone cell with `ensemble_sd_ppm` (the spread of the five members' predictions) and
`ci90_halfwidth_ppm`. The variance mapping and split-conformal multipliers are fitted on
fold 0 only; fold 6 is used once to report coverage. Side-chain nuclei get no interval
because member disagreement there can reflect prochiral labelling. That file is skipped
when no artifact is supplied, and an artifact is refused for any member count other than
five.

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
shifts; uncertainty is clustered over label sets, never individual residues.

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
| `src/deepshift/evaluate.py`, `statistics.py` | the one scorer every reported number comes from: per-nucleus RMSE and MAE, secondary-shift statistics, the label-set-clustered bootstrap and paired comparisons |
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
