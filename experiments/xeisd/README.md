# X-EISD ensemble selection (Figure 5, Figure S12)

This directory runs the case study in which DeepShift replaces the chemical-shift forward
model of X-EISD (Lincoff et al., *Commun. Chem.* 3, 74, 2020) on the unfolded drkN SH3
domain, with SHIFTX2 as the comparison. X-EISD selects 100 conformers from a pool by Bayesian
scoring against experimental data. Here the optimizer sees chemical shifts alone, 1,000
independent restarts per pool and forward model (seed 91), and the seven other observables
(SAXS, FRET, J-coupling, NOE, PRE, RDC, R_h) are scored afterwards on the selected
conformers. Only the chemical-shift table differs between the two arms: the ensemble, the
experimental data, the error model, the scoring and the optimizer are X-EISD's own.

| Script | Does | Writes |
|---|---|---|
| `run_shift_tables.py` | DeepShift back-calculation of the 262 experimental chemical-shift observations for each conformer of a pool | `drksh3_{ENSEMBLE,RANDOM_MIXEDROWS,RANDOM}_CS_deepshift.txt` |
| `build_mixed.py` | re-orders ENSEMBLE and RANDOM rows into X-EISD's MIXED pool layout (CPU) | `drksh3_MIXED_CS_deepshift.txt` |
| `run_paper_protocol.py` | the optimization and the eight-observable rescoring, SHIFTX2 and DeepShift arms | `<pool>/<model>/{optimize_cs,rescore_all}/{indices,results}.csv`, `run.json` |
| `analyze_pools.py` | paired DeepShift-minus-SHIFTX2 statistics over restarts, all three pools | `results.json` |
| `analyze_composition.py` | share of ENSEMBLE conformers the MIXED-pool optimization selects | `mixed_composition.json` |
| `pools.py` | shared locations and the MIXED row layout | |

## Inputs

**X-EISD** is a dependency and is not copied here. The study ran against the repository
`https://github.com/THGLab/X-EISD`, commit `a256178d8f5ba8c0571bda26534ec591bc7a0989`
(10 June 2025), with the `eisd` package unmodified. Its licence file (`LICENSE`) grants use,
copying and modification for educational, research and not-for-profit purposes without fee
(Copyright 2020 The Regents of the University of California); `setup.py` declares MIT, so
the two statements differ and the `LICENSE` text is the stricter. It requires only NumPy
and pandas, which the release pins.

```bash
git clone https://github.com/THGLab/X-EISD.git
git -C X-EISD checkout a256178d8f5ba8c0571bda26534ec591bc7a0989
git -C X-EISD lfs pull        # the data files are Git LFS objects
export XEISD_HOME=$PWD/X-EISD
```

If the LFS quota is exhausted, the repository's README names a Dryad mirror of the data
(1.14 GB) to unpack into `$XEISD_HOME/data`. The scripts read these files from
`$XEISD_HOME/data`:

* `experimental_data/` holds the experimental restraints of the drkN SH3 study, distributed
  with X-EISD: chemical shifts (`drksh3_CS_exp_mod.txt`, the 262 observations over residues
  3-57 and atoms C, CA, CB, H, HA, which fix the column order of every table), J-couplings,
  NOEs, PREs, RDCs and SAXS.
* `back_calc_data/` holds X-EISD's own back-calculated tables, including its SHIFTX2
  chemical-shift tables (`ENSEMBLE_PED/drksh3_ENSEMBLE_CS.txt`,
  `Trades/drksh3_CS_structdata.txt`, `mixpool/drksh3_mixpool_CS.txt`), which serve as the
  SHIFTX2 arm, and the seven other observables for every pool. The first column of
  `mixpool/drksh3_mixpool_CS.txt` gives the RANDOM-pool source index of each MIXED row.
* `structures/ENSEMBLE_PED/PED8AAC-pdb.tar.gz` is the 1,700-conformer ENSEMBLE pool (a Protein Ensemble
  Database deposit, archive name `PED8AAC`), read in basename order.
  `structures/Trades/all_structs_cat.pdb.bz2` is the 100,000-conformer RANDOM pool; the LFS
  object is the bz2 stream itself, and `--random-stream` names it if it is stored elsewhere.

Predictions need the five checkpoints in `models/` (Git LFS) and an installed `deepshift`
(`python -m pip install .` from the repository root). DeepShift prediction is a GPU step
(`--device cuda`, the default; fp16 autocast, as in the benchmark); the optimization and the
analysis run on a CPU.

drkN SH3 is BMRB 25500, which belongs to the case-study hold-out (fold 0) that no member
trained on, so every table row is the mean of all five members, in ppm.

## Commands

```bash
W=<work directory>
python experiments/xeisd/run_shift_tables.py --pool ensemble     --out-dir $W/tables
python experiments/xeisd/run_shift_tables.py --pool random-mixed --out-dir $W/tables
python experiments/xeisd/run_shift_tables.py --pool random       --out-dir $W/tables  # 100,000 conformers; resumable
python experiments/xeisd/build_mixed.py --tables $W/tables
python experiments/xeisd/run_paper_protocol.py --tables $W/tables --out $W/runs
mkdir -p $W/protocol && ln -s ../runs $W/protocol/runs
python experiments/xeisd/analyze_pools.py       --run-root $W/runs --out $W/protocol/results.json
python experiments/xeisd/analyze_composition.py --runs $W/runs     --out $W/protocol/mixed_composition.json
export DEEPSHIFT_XEISD_PROTOCOL=$W/protocol
```

An interrupted `run_shift_tables.py` resumes from its `.part` file. `run_paper_protocol.py`
refuses to overwrite an existing arm directory; `--pools` and `--models` run a subset, and
each arm reseeds NumPy, so a subset reproduces the same arm as the full run.

## What the producers read

`DEEPSHIFT_XEISD_PROTOCOL` is `$W/protocol`; the table directory is its sibling `$W/tables`.

| Producer | Reads | Written by |
|---|---|---|
| `figures/fig_xeisd_mixed_selection.py` (Figure 5) | `results.json`, `mixed_composition.json`, `runs/mixed/{shiftx2,deepshift}/rescore_all/{results,indices}.csv` | `analyze_pools.py`, `analyze_composition.py`, `run_paper_protocol.py` |
| `figures/fig_xeisd_composition.py` (Figure S12) | `runs/mixed/{shiftx2,deepshift}/optimize_cs/indices.csv` | `run_paper_protocol.py` |
| `si_build/emit_text_numbers.py` | `results.json`, `runs/{ensemble,mixed,random}/{deepshift,shiftx2}/rescore_all/results.csv`, `../tables/drksh3_{ENSEMBLE,MIXED,RANDOM}_CS_deepshift.txt` (row counts) | `analyze_pools.py`, `run_paper_protocol.py`, `run_shift_tables.py`, `build_mixed.py` |

The restart-level CSV files have no header; `results.csv` columns are the restart index, the
number of accepted exchanges, the eight log scores (SAXS, CS, FRET, J-coupling, NOE, PRE, RDC, R_h)
and the eight RMSEs in the same order.
