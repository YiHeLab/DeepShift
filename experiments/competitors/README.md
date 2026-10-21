# Competitor predictors

This directory runs the third-party predictors the paper compares against on the fold-6
benchmark and writes their outputs where `benchmarks/` reads them:
`$DEEPSHIFT_BENCH_OUT/<arm>/<structure_id>.json`, one file per row of
`$DEEPSHIFT_BENCH_OUT/fold6_tasks.tsv` (written by `python -m benchmarks.cells`), plus
`<arm>/hitlog/<structure_id>.log` for every arm that retrieves reference shifts. No
third-party source or weights are redistributed. The three predictors that were modified
(UCBShift, UCBShift 2.0, SHIFTX2) are patched with the diffs in `patches/`. Install each
tool as below, export the variables listed for it, then run the commands under
"Regenerating the arms".

The release scorers read, per structure, `status`, `predictions` (author residue number ->
`{resname, <atom>: ppm}`) and, for the retrieval arms, the hit log. Each predictor's
output is mapped to the benchmark's atom names by its adapter (`adapters/`); hydrogens are
supplied as each tool requires (REDUCE `-Trim` then `-build` where a tool needs explicit
H and does not add them).

## Layout

| file | role |
|---|---|
| `run_arm.py` | structure-only and retrieval arms over `fold6_tasks.tsv` |
| `common.py` | chain extraction, REDUCE hydrogenation, tool-path lookup |
| `adapters/*.py` | one module per predictor: runs the tool on one chain, parses its output |
| `adapters/procs15_fixed_scale.json` | per-nucleus map from ProCS15 shielding to ppm (fitted once on 1UBQ against BMRB 6457) |
| `patches/*.patch` | modifications to UCBShift, UCBShift 2.0 and SHIFTX2 |

## Predictors

Versions are those the paper's outputs were produced with.

### SPARTA+ (`sparta`)
Version 2.90, build 2017.143.12.12. https://spin.niddk.nih.gov/bax/software/SPARTA+/
(`sparta+.tar.Z`). No licence is stated; the download page forbids redistribution without
the authors' permission. Install: `zcat sparta+.tar.Z | tar xf -`, then
`ln -sf SPARTA+.static.linux9 SPARTA+/bin/SPARTA+`. Variable: `SPARTA_HOME` (the
`SPARTA+` directory, holding `bin/` and `tab/`).

### NapShift (`napshift`)
Git commit `b3c353cc48fab39954275f2592bc2f302271ce42` of https://github.com/vrettasm/NapShift
(no release tag). GPL-3.0. Install: clone and check out that commit; the models ship in the
repository (`models/`). Environment: Python 3.9.23, tensorflow 2.11.0, keras 2.11.0, numpy 1.23.5, pandas 1.5.3, scikit-learn 1.0.2, scipy 1.13.1, h5py 3.8.0,
biopython 1.85, numba 0.56.4, joblib 1.5.3, protobuf 3.19.6. Variables: `NAPSHIFT_HOME`
(the checkout), `NAPSHIFT_PYTHON` (that environment's interpreter). The input is the chain
as extracted, without added hydrogens.

### LEGOLAS (`legolas`)
Git commit `9485a137a94981523a2dd9c68fc280700413f245` of
https://github.com/roitberg-group/legolas, with its submodule `external/internal-legolas-aev`
at `86dea01bc2a7387ec1ea171760a02da9e3ff1182` (`git clone --recursive`). MIT. The models
ship in `test/ens_models/`. Environment: Python 3.11.15, torch 2.8.0+cu129, torchani 2.8.2,
numpy 2.2.6, pandas 3.0.3, biopython 1.87, mdtraj 1.11.1, ase 3.29.0, h5py 3.16.0, scipy
1.17.1 (created from the repository's `legolas_env.yaml`; installed versions differ from
its pins). Run on the CPU. Variables: `LEGOLAS_HOME` (the checkout, holding
`test/legolas.py`), `LEGOLAS_PYTHON`.

### ProCS15 (`procs15`)
PHAISTOS trunk, SVN revision 544 (`svn checkout -r 544
svn://svn.code.sf.net/p/phaistos/code/trunk phaistos`); GPL-3. Its ProCS15 module is
https://github.com/jensengroup/procs15 at commit `3d341f48eae27673810a94f915ab94e78e2af7f5`,
cloned to `phaistos/modules/procs15`; GPL-3. Add to `phaistos/CMakeLists.txt`, near line 36
(a Boost MPL limit under Boost 1.74 and gcc 11):

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFUSION_MAX_VECTOR_SIZE=40 -DBOOST_MPL_CFG_NO_PREPROCESSED_HEADERS -DBOOST_MPL_LIMIT_VECTOR_SIZE=40")

Build: `mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=/usr
-DPHAISTOS_INCLUDE_CHAIN_CA=OFF -DPHAISTOS_INCLUDE_DOCS=OFF -DPHAISTOS_INCLUDE_OMEGA_DBNS=OFF ..
&& make procs15_predictor` (gcc 11.4.0, cmake 3.22.1, system Boost 1.74.0). The binary is
`build/test/procs15_predictor`. Data: `ProCSnumpyfiles.tar.bz2` from
http://www.erda.dk/public/archives/YXJjaGl2ZS1pMHN3TXE=/procs/ProCSnumpyfiles.tar.bz2 (2.9 GB;
17 GB extracted; no licence stated). Variables: `PROCS15_BIN` (the binary), `PROCS15_DATA`
(the extracted `ProCSnumpyfiles/` folder, trailing slash). One process needs about 27 GB of
RAM and about 18 s to load the data, so `run_arm.py` runs ProCS15 in batches (`--batch`).

### NMRGNN (`nmrgnn`)
`nmrgnn` 0.7 from PyPI (https://github.com/ur-whitelab/nmrgnn); MIT. The weights ship in the
wheel. Environment: Python 3.9.23, `pip install nmrgnn==0.7 tensorflow==2.10.0 keras==2.10.0
numpy==1.23.5 pandas==2.3.3 scikit-learn==1.6.1 scipy==1.13.1 MDAnalysis==2.7.0
keras-tuner==1.0.2`. Run on the CPU. Variable: `NMRGNN_BIN` (the `nmrgnn` executable of
that environment).

### CamShift (`camshift`): PLUMED 2.10.0 and `camshift.db`
PLUMED 2.10.0 (https://github.com/plumed/plumed2; LGPL-3.0), conda-forge build
`plumed-2.10.0-nompi_h4679bf9_100` with the isdb module on: `conda create -p plumed_env -c
conda-forge plumed=2.10.0=nompi_h4679bf9_100`. The CamShift parameter file `camshift.db` is
`regtest/isdb/rt-cs2backbone/data/camshift.db` of the PLUMED source tree (md5
`90fbd6f6bdab5932461167f1470bc0c5`, identical at tag v2.10.0); it carries no separate
licence statement. Variables: `PLUMED_BIN` (the `plumed` executable), `CAMSHIFT_DB`.

### REDUCE (hydrogens for SPARTA+, LEGOLAS, NMRGNN, CamShift, UCBShift and UCBShift 2.0, MLFF)
REDUCE 4.15 (`reduce.4.15.250408`), bioconda build `reduce-4.15-h503566f_1`:
`conda create -n ucbshift -c bioconda -c conda-forge reduce=4.15=h503566f_1`.
https://github.com/rlabduke/reduce; licence BSD-4-Clause-UC per the conda metadata (the
current repository states Apache-2.0). Every deposited hydrogen is stripped (`reduce -Trim
-Quiet`) and rebuilt (`reduce -build -Quiet`). If the binary cannot find its het dictionary
(`reduce_wwPDB_het_dict.txt`) under another install prefix, set `REDUCE_HET_DICT`. Variable:
`REDUCE_BIN` (default `reduce` on `PATH`).

### UCBShift 1.0 (`ucbshift_x`, `ucbshift_full_*`)
https://github.com/THGLab/CSpred, commit `20cee31` ("Update LICENSE.txt"). Licence: UC
Regents, research, education and not-for-profit use without fee (LICENSE.txt in the
repository); commercial use needs a licence from UC Berkeley. Clone, check out `20cee31`,
and apply `patches/ucbshift.patch` from the checkout root (`patch -p1 < ucbshift.patch`).
Download the models (`models/`, 18 `.sav` files) from the Dryad link in the repository's
README. The checkout's `bins/` (BLAST+ 2.9.0+, mTM-align, mkdssp) and `refDB/` (the RefDB
BLAST database and shift tables) come with the repository. Environment (`ucbshift`):
Python 3.8.20, biopython 1.74, scikit-learn 0.22.2.post1, numpy 1.19.5, pandas 1.1.5, scipy
1.5.4, joblib 1.4.2, REDUCE 4.15. Variables: `UCBSHIFT_HOME` (the patched checkout),
`UCBSHIFT_PYTHON` (that environment's interpreter).

The patch contains: (1) `CSpred.py`, an `UCB_TIMING` block and three `perf_counter` hunks
that write feature and prediction time to the file `$UCB_TIMING.<pid>` when that variable
is set (no effect on predictions; the scripts in `experiments/speed/` do not read it); (2) `CSpred.py`,
`first_chain_only=True` in `build_input`, so only the first chain of a multi-chain file is
featurised; (3) `ucbshifty.py`, a Biopython compatibility shim (`substitution_matrices` is
absent from Biopython 1.74, so the old matrix is used); (4) `ucbshifty.py`, the
single-chain assertion replaced by a message (first chain used); (5) `ucbshifty.py`,
non-standard letters in the Needleman-Wunsch inputs become `X`; (6) `ucbshifty.py`, the
blocking hooks: `UCBY_EXCLUDE_BMRBS` (comma-separated BMRB ids whose RefDB entries are
dropped, logged `drop_bmrb`), `UCBY_SELF_IDENT_CUTOFF` (candidates at or above this
identity are dropped, logged `drop_identity`; unset means no filter) and `UCBY_HITLOG`
(path of the hit log: tab-separated target, BMRB id, score, matched length, target length,
identity, decision; kept candidates are `keep_eligible`).

### UCBShift 2.0 (`ucbshift2_*`)
https://github.com/THGLab/CSpred, tag `v2.0` (commit `844e265`; the master tree equals it
at the time of writing). Same licence as UCBShift 1.0. Apply `patches/ucbshift2.patch` from
the checkout root. The side-chain models (141 `.sav` files) are downloaded separately from
https://doi.org/10.5281/zenodo.15375968 into a directory outside the checkout. The same
`ucbshift` environment runs it. Variables: `UCBSHIFT2_HOME` (the patched checkout),
`UCBSHIFT2_MODELS` (the model directory), `UCBSHIFT_PYTHON`. The patch contains the same
three blocking hooks as above; `spartap_features.py`, `if HDatom:` becomes
`if HDatom is not None:` (a truthiness fault in the hydrogen-bond feature); and
`ucbshifty.py` reads `UCB_SELF_IDENT_CUTOFF` (default 0.90) in the CSpred exclude mode
(`-t`), which the benchmark does not use.
Reference: Ptaszek, Li, Konrat, Platzer, Head-Gordon, J. Am. Chem. Soc. 146:31733 (2024).

### SHIFTX2 (`shiftx2_*`)
SHIFTX2 v1.13 Linux standalone, http://www.shiftx2.ca/download/shiftx2-v113-linux-20180808.tgz.
No licence terms are stated in the distribution or on the download page. Unpack it, apply
`patches/shiftx2_shifty3.patch` from the directory that contains `shiftx2-linux/`
(`patch -p1`), and prepare the install as follows. The standalone is a Python 2 program:
`conda create -n shiftx2 python=2.7 numpy` (Python 2.7.15, numpy 1.16.5); Java is OpenJDK 11.
`modules/angles/get_angles` is recompiled for 64 bits and `modules/angles/reduce` is replaced
by REDUCE 4.15 (a symlink to the binary above; the bundled REDUCE 3.10 places hydrogens
differently). SHIFTY+ runs on its bundled i386 BLAST, which needs the 32-bit libc on the host.
Variables: `SHIFTX2_HOME` (the `shiftx2-linux` directory), `SHIFTX2_PYTHON` (the Python 2.7
interpreter; its directory must also supply `java`). The patch adds to `shifty3/shifty3.py`:
`SHIFTY_EXCLUDE_BMRBS` (comma-separated BMRB ids whose `bmr<id>.cs` entries are skipped),
`SHIFTY_SELF_IDEN` (hits at or above this identity percentage are skipped; the default 101
disables it) and `SHIFTY_LOG_HIT` (one `HITLOG<TAB>shifttype<TAB>entry<TAB>identity` line on
stderr per reference used), plus two trailing-space fixes. SHIFTY+ names its
temporary files by a seconds-resolution timestamp; the adapter gives every job its own
`-z` temporary directory.

### MLFF (`mlff`): MACE-OFF23 descriptors with trained shift heads, on Amber-relaxed chains
Model: Bojan et al., "Representing local protein environments with machine learning force
fields", arXiv 2505.23354 (v3). Code: https://github.com/mb012/MLFF_representation, commit
`9e4c984558f1ca638fdcf4b2f6ac787c30b46e43`; MIT. The trained head checkpoints
(`all_cs_models/cs_mace/<atom>/cs_mace_with_clip_<atom>_epoch_2000` for 11 atom types,
5.06 MB each) are not in the repository: download `force_fields_data.zip` from the Dropbox
link in its README and unpack `cs_models.zip`. Descriptors come from MACE-OFF23 large
(`MACE-OFF23_large.model`, 55,492,786 bytes, sha256
`a29e397dbf3e7a24ac50a9b0dfc919bd5a62efa346f5895a6237b0950c1d76f4`), fetched by `mace_off`
from https://github.com/ACEsuit/mace-off on first use into `~/.cache/mace/`. Its licence is
the Academic Software Licence (non-commercial use only). Environment: Python 3.12.0,
mace-torch 0.3.14 (MIT), torch 2.9.1, e3nn 0.4.4, ase 3.27.0, torch-geometric 2.8.0.post1,
matscipy 1.2.0, scikit-learn 1.9.0, numpy 2.4.0, pandas 2.3.3, matplotlib 3.10.8, tqdm 4.67.1;
another mace-torch version may break the descriptor call. Variables: `MLFF_HOME` (a
directory holding `MLFF_representation/` and `models/all_cs_models/cs_mace/`), `MACE_MODEL`
(default `~/.cache/mace/MACE-OFF23_large.model`), `REDUCE_BIN`. Run it with the interpreter
of that environment (`MACE_PYTHON` below).

Only the Amber-relaxed input is reported. The relaxation (`relax.py`) uses PDBFixer
(missing atoms, hydrogens at pH 7.0, heterogens removed) and OpenMM with `amber99sb.xml` and
`amber99_obc.xml`, hydrogen bonds constrained, heavy atoms restrained to their input
positions at 10 kcal/mol/A^2, and a local energy minimisation; ligands, ions and waters are
re-appended unchanged. A chain that fails with an OpenMM "No template found" error is
retried after removing the atoms no Amber99SB template admits (a disulfide CYS HG, a
C-terminal HXT, a surplus N-terminal backbone hydrogen, N-terminal GLU HE2 or ASP HD2).
OpenMM bonds across chain breaks, which pulls gaps shut. Environment: Python 3.11.15,
OpenMM 8.5.2 (conda-forge, LGPL-3.0-or-later; https://github.com/openmm/openmm), PDBFixer
1.12.0 (MIT; https://github.com/openmm/pdbfixer), numpy 2.4.6. The paper's relaxation ran on
CUDA; `--platform CPU` also works. The minimisation is not deterministic, so a fresh
relaxation does not reproduce the paper's chains bit for bit; MLFF predictions on a freshly
relaxed chain differed by up to 1.5 ppm (N) from the paper's on the one chain compared. The
chains the published MLFF values were computed from are deposited with the dataset
(`deepshift_mlff_relaxed_structures.tar.gz`, folder `fold6/`, one `<structure_id>.pdb` per
task); give that folder to `run_mlff.py --relaxed` to reproduce the published values.

### PLM-CS (`plmcs`)
`plm-cs` 3.4 from PyPI (https://github.com/doorpro/predict-chemical-shifts-from-protein-sequence;
Zhu, Hu, Yang, Chen, Digital Discovery 4:331-337, 2025; MIT) with its six head checkpoints
`ca/cb/c/ha/h/n-ckpt` 0.1 (about 24.7 MB each, shipped as PyPI packages). ESM-2 650M
(`esm2_t33_650M_UR50D.pt`, 2,604,537,549 bytes, plus its contact-regression file) is fetched
by `fair-esm` 2.0.0 into `$TORCH_HOME/hub/checkpoints` on first run (MIT). Environment:
`conda create -p plmcs_env python=3.11`, `pip install torch==2.5.0+cpu torchaudio==2.5.0
torchvision==0.20.0` (CPU wheels), `pip install plm-cs==3.4` (pins fair-esm 2.0.0, numpy
2.1.2, biopython 1.84, pandas 2.2.3). PLM-CS predicts from sequence alone and cannot take a
chain over 512 residues; those structures are written with `status: failed`. Its training
set (RefDB without the SHIFTX test proteins) is fixed inside the checkpoints.

## Regenerating the arms

All commands run from the repository root with `DEEPSHIFT_RUN` and `DEEPSHIFT_BENCH_OUT`
set. The first writes the task table.

```bash
python -m benchmarks.cells                       # fold6_tasks.tsv
# structure-only
for arm in sparta napshift legolas nmrgnn camshift; do
  python experiments/competitors/run_arm.py --arm $arm --all --workers 12; done
python experiments/competitors/run_arm.py --arm procs15 --all --batch 250   # one process, ~27 GB
# UCBShift 1.0 and 2.0, SHIFTX2 (retrieval arms; hit logs are written beside the outputs)
for arm in ucbshift_x ucbshift_full_leaky ucbshift_full_sameprot ucbshift_full_selfexcl \
           ucbshift2_sameprot ucbshift2_selfexcl shiftx2_leaky shiftx2_sameprot \
           shiftx2_selfexcl shiftx2_noshifty; do
  python experiments/competitors/run_arm.py --arm $arm --all --workers 6; done
# PLM-CS
$PLMCS_PYTHON experiments/competitors/plmcs.py
# MLFF: extract the chains, relax them, then run the heads (or skip relax.py and pass the
# deposited fold6/ folder as RELAXED)
python experiments/competitors/relax.py --chains CHAINS --out RELAXED
$MACE_PYTHON experiments/competitors/run_mlff.py --relaxed RELAXED
```

`run_arm.py` resumes (it skips structures with a non-empty output) and accepts
`--structures`, `--source`, `--shard i/n`. A `*_sameprot` or `*_selfexcl` arm blocks, per
structure, the BMRB ids `benchmarks/reference/competitor_same_protein.json` lists for the arm's
library (`ucbshift` for UCBShift 1.0 and 2.0, `shifty` for SHIFTX2), or, for a structure that
file does not list, the task table's `exclude_bmrbs` and its own BMRB. `*_selfexcl` arms also
drop references at or above 90 % identity (`--identity_cutoff`). `tests/test_competitor_same_protein.py`
reads the hit logs and fails if a blocked arm kept a same-protein reference. The paper's
SHIFTX2 outputs were produced with a 21,600 s per-structure limit.

## UCBShift-200 and SHIFTX2 61-protein stores

Run from `experiments/competitors/`. The UCBShift-200 inputs are read from the CSpred
checkout: `$UCBSHIFT_HOME/train_model/pdbs/test/<id>.pdb` for the ids that also have
`test_targets/<id>.csv` (200 chains; the id is the PDB id plus the chain letter, `_` for the
first chain). Both stores block, per structure, the BMRB ids in
`benchmarks/reference/ucb200_competitor_same_protein.json`; `run_ucb200.py` stops naming the
structure if the file does not list it.

```bash
python run_ucb200.py --store ucbshift --workers 12   # ucb200_ucbshift_full_sameprot_sameprotein_seq_20261002/
python run_ucb200.py --store shiftx2  --workers 12   # ucb200_shiftx2_sameprot_sameprotein_seq_20261002/ (197 chains)
python run_shiftx2_61.py --workers 12                # shiftx2_61/{asdistributed,structonly,selfexcl}/
```

The stores are written under `$DEEPSHIFT_BENCH_OUT`, with hit logs in `hitlog/`. The
UCBShift store holds one JSON per chain; the SHIFTX2 stores hold the `.cs` files SHIFTX2
writes. Variables: `UCBSHIFT_HOME`, `UCBSHIFT_PYTHON`, `REDUCE_BIN`, `SHIFTX2_HOME`,
`SHIFTX2_PYTHON`. The 61-protein roster is
`benchmarks/reference/shiftx2_test61_ucbshift_overlap.csv`; its query chains come from
`$UCBSHIFT_HOME/refDB/pdbs/<id>.pdb`, and seven chains absent there (1SNMA, 1RROA, 1T8LB,
1ODVA, 1SGZA, 1SYDA, 1KDBA) are cut from the first model of the RCSB entry, downloaded to
`<out>/rcsb/`. The `selfexcl` setting blocks the highest-identity hit and every hit at 95 %
identity or above in the as-distributed run's hit log.
