# The programs that ran the paper's experiments

`benchmarks/`, `figures/` and `si_build/` compute every reported number from finished
measurements (the reproduction bundle). The programs here made those measurements: they run
the third-party predictors and the case studies. Each directory's README names the paper
item it reproduces, the third-party software with the version used, where to get it and its
licence, the environment variables it reads, the commands in order, and the files the
release's producers then read.

| Directory | Reproduces | Third-party software |
|---|---|---|
| `competitors/` | the fold-6 competitor outputs the `benchmarks/` scorers read (Table 1, Figures 2 and 3, and every competitor column of the supplement); the UCBShift-200 stores (Tables S18, S19); SHIFTX2 on the 61-protein set (Table S20) | SPARTA+, NapShift, LEGOLAS, ProCS15, NMRGNN, CAMSHIFT (PLUMED), PLM-CS, the MACE-based MLFF predictor with its OpenMM relaxation, UCBShift, UCBShift 2.0, SHIFTX2, REDUCE |
| `speed/` | Table S27 | the predictors of `competitors/` |
| `xeisd/` | Figure 5, Figure S12 | X-EISD |
| `idp8/` | Figure S11(a), Table S28 | the PED ensembles and BMRB entries of Nagy et al. |
| `tau/` | Figure S11(b) | UCBShift 2.0, the MLFF predictor, the Stöckelmaier et al. benchmark files |
| `decoys/` | Figure 4 decoy panels, Figure S9, Table S25 | the 3DRobot decoy sets, UCBShift, the MLFF predictor |

No third-party source code or weights are included. UCBShift, UCBShift 2.0 and SHIFTX2 were
modified to block retrieval of the test protein's own depositions; those modifications are
the diffs in `competitors/patches/`, applied to the upstream releases named there.

The scripts import the `deepshift` package (`pip install -e .` at the repository root). The
DeepShift and MLFF inference steps were run on a GPU; on a CPU they give the same values to
within about 0.01 ppm. OpenMM energy minimisation is not bit-reproducible, so the relaxed
structures the MLFF predictor was run on are deposited with the dataset
(`deepshift_mlff_relaxed_structures.tar.gz`), and the reported MLFF values are reproduced
from them.
