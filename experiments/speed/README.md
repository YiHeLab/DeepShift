# Inference-speed census (Table S27)

These scripts time every predictor of Table S27 on one cohort and write the census
`si_build/emit_speed.py` reads (`speed_census.json`). The third-party predictors run through
the adapters in `../competitors/adapters/` (install them and export their variables as
`../competitors/README.md` describes); DeepShift runs through the released package
(`python -m deepshift.pipeline predict` for the cold-process arms, `deepshift` batch
inference for the throughput arms, with the models in `models/`).

## Cohort and clock

The cohort is 203 crystal chains drawn from 240 chains staged in the census directory
(`chains/<id>.pdb`, `chain_manifest.json`, `cohort_fold6_xtal.json`). Each arm times its
chains on an otherwise idle machine after a two-chain warm-up that is discarded and a
page-cache prewarm (ProCS15 reads 25.7 GB of parameters cold). The census was timed on a
machine with an RTX 5090 and a 20-core Core Ultra 7 265K. The 240-chain finish rates need
the benchmark records of that 240-chain cohort (`--bench` or `DEEPSHIFT_BENCH_OUT`); the
rebuilt fold-6 task table holds 207 of the crystal tasks, so the rates are computed from the
archived records.

## Variables

`SI_SPEED_CENSUS` (the census directory; required), `DEEPSHIFT_BENCH_OUT` (summarize only),
and the predictor variables `SPARTA_HOME`, `NAPSHIFT_PYTHON`, `NAPSHIFT_HOME`,
`LEGOLAS_PYTHON`, `LEGOLAS_HOME`, `NMRGNN_BIN`, `PROCS15_BIN`, `PROCS15_DATA`,
`UCBSHIFT_HOME`, `UCBSHIFT_PYTHON`, `SHIFTX2_HOME`, `SHIFTX2_PYTHON`, `REDUCE_BIN`. The two
shell scripts read `PYTHON`.

## Command sequence

```bash
export SI_SPEED_CENSUS=/path/to/census
bash experiments/speed/run_all.sh                          # competitor arms and the serial batch arms
python experiments/speed/retime_idle_ledger.py             # idle-gated prep8 passes and DeepShift cold arms
bash experiments/speed/retime_idle.sh                      # idle-gated legolas-gpu and throughput passes
python experiments/speed/coverage240.py --members 1        # DeepShift finish rate, one member
python experiments/speed/coverage240.py --members 5        # five-member ensemble
python experiments/speed/summarize.py                      # writes speed_census.json
python si_build/emit_speed.py
```

Written under `$SI_SPEED_CENSUS`: `out/` (one record per arm and pass, `out/ledger.tsv`),
`batch/`, `work/`, `deepshift_coverage240_final*.json`, `speed_census.json`.

| script | measures |
|---|---|
| `run_speed_census.py` | one process per chain (cold) for the seven competitors and the two DeepShift cold arms; `--arm A [--subset N --repeat R]` |
| `run_speed_batch.py` | the serial batch arms: SHIFTX2, NapShift, ProCS15 |
| `run_batch_parallel_prep.py`, `run_legolas_parallel_prep.py` | parallel preparation with the predictor's own step timed separately (SPARTA+, LEGOLAS, UCBShift-X) |
| `run_speed_throughput.py` | DeepShift throughput, single model and five-member ensemble, batched on the GPU |
| `coverage240.py` | fraction of the 240 chains DeepShift completes |
| `summarize.py` | `speed_census.json` from the records |
| `retime_idle_ledger.py`, `retime_idle.sh`, `run_all.sh` | idle-gated repeats and the full queue |
| `speed_common.py` | paths, cohort and residue-count helpers |

## Differences from the timed runs

The census called the same tool commands the shipped adapters call for SPARTA+, NapShift,
LEGOLAS, NMRGNN and SHIFTX2 (`-n`). The shipped ProCS15 adapter goes through its batch entry
point with one structure and discards the predictor's stdout. The shipped UCBShift adapter
sets `CUDA_VISIBLE_DEVICES=''`, one BLAS thread, `TMPDIR` to the job directory, a
`<job>.pdb` file name and a 7200 s limit; the timed run used none of these (wall time on a
28- and a 129-residue chain was equal within 0.1 s). The models in `models/` are not the
files the census loaded (`final/fold*.final.pt`) and have different file hashes. Timings are
properties of the machine and are not expected to reproduce on other hardware.
