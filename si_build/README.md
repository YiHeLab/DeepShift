# si_build — supplementary tables

One script per table of the supplement. Each writes its LaTeX table to `out/`
(`SI_OUT` names another directory). The authors' captions are in
`reference_inputs/captions/` and are spliced in by `si_common.emit`
(`SI_CAPTIONS_DIR` names another directory).

Table S16 (`si_tables_literature`) is produced by `figures/si_tables_literature.py`, and
Table S1 (the dataset census) is typed into `SI.tex`. The inputs are finished runs and
scoring reports that are not in this repository; every root is an environment variable
with no default. Run each script from this directory: `python <script>.py`.

## Roots used by several scripts

| variable | meaning |
|---|---|
| `DEEPSHIFT_RUN` | the build: `data/`, `work/unified_cache/`, `final/`, `benchmark/` |
| `DEEPSHIFT_ARCHIVE` | finished measurements of earlier runs (`archive("...")`) |
| `SI_COMPETITOR_ROOT` | the competitor run (gate report) |
| `SI_FINAL_SCORES_DIR` | the scoring reports of the run (`score_tier2_master.json`, ...) |
| `DEEPSHIFT_BENCH_OUT` | benchmark root, default `$DEEPSHIFT_RUN/benchmark` |
| `SI_P100_CUTOFFS` | the fitted p100 cutoff file |

## Tables

| table | script | further variables |
|---|---|---|
| S2 fold composition | `gen_census_tables.py --flags <qc_flags_p100_structural.json>` | |
| S3, S4 rereferencing offsets, coverage | `si_reref_offsets.py` | `SI_REREF_DIAGNOSTICS` (output of `scripts/reref_diagnostics.py`) |
| S5 agreement with RefDB | `build_refdb_census.py --out <dir>/refdb_census.json`, then `si_refdb_agreement.py` | `DEEPSHIFT_REFDB_CORRECTIONS`, `SI_REFDB_CENSUS` |
| S6 p100 spans | `si_qc_p100.py` | |
| S7 perturber dose | `emit_perturber_dose_ppm.py` (reads the table `analyze_perturber.py` writes) | `SI_PERTURBER_REBUILD` |
| S8 leakage | `emit_leakage.py` | `SI_LEAKAGE_ARTIFACTS_DIR` |
| S9 distance to training | `emit_train_distance.py [--against NAME=FASTA]` | `SI_UCB_TRAINING_FASTA` |
| S10 nuclei | `si_nuclei.py` | |
| S11 coverage ledger | `si_competitor_ledger.py` | `DEEPSHIFT_COVERAGE_CENSUS` |
| S12-S14 backbone RMSE, MAE, MedAE | `emit_competitor_tables_final.py` | |
| S15 side chains | `emit_sidechain_final.py` | |
| S17, S21 gross error, exclusion ladder | `emit_pool_tables_final.py` | |
| S18 reproduction gate | `build_reproduction_gate.py` | `SI_GATE_UCB200_RERUN`, `SI_GATE_CSPRED_TM`, `SI_UCB200_POOL61` |
| S19 UCBShift 200-set | `emit_ucb200_common.py` | `SI_UCB200_ARTIFACTS_DIR`, `SI_UCB200_COMPETITOR_CELLS` |
| S20 published test sets | `emit_ucb200_table.py` | `SI_UCB200_ARTIFACTS_DIR`, `SI_UCB200_POOL61`, `SI_SHIFTX2_61_LOG` |
| S22 training composition | `emit_data_composition.py` | `SI_DATA_DIVERSITY_DIR`, `SI_SOURCE_TRANSFER_DIR` |
| S23 cysteine modes | `emit_cys_modes.py` | `SI_CYS_MODES_REPORT` |
| S24 error covariates | `emit_error_covariates.py` | `SI_ERROR_COVARIATES_REPORT` |
| S25 decoy ledger | `build_decoy_ledger.py` | `DEEPSHIFT_DECOY_POOLED`, `SI_DECOY_COHORT` |
| S26 interval calibration | `build_interval_calibration.py --calibration <interval_calibration.json> --evaluation <interval_evaluation_fold6_pf14.json>` | |
| S27 inference speed | `emit_speed.py` | `SI_SPEED_CENSUS` |
| S28 IDP8 roster | `build_idp8_roster.py` | `DEEPSHIFT_IDP8_ROOT` |

Other scripts:

- `emit_text_numbers.py` recomputes the numbers printed in the Methods and the supplement
  and writes `out/text_numbers.json` (`name -> {value, description}`). It reads
  `DEEPSHIFT_XEISD_PROTOCOL`, `SI_SELF_RETRIEVAL_UCBY`, `SI_SELF_RETRIEVAL_SHIFTY61`,
  `SI_GATE_CSPRED_TM`, `DEEPSHIFT_REFDB_SHIFTS`, `DEEPSHIFT_REFDB_LIBRARY` and
  `DEEPSHIFT_UCBSHIFT_LEAKAGE` besides the roots above; `--allow-missing` skips a group
  whose files are absent.
- `emit_homology_axis.py --out-dir <dir>` writes the identity axis Figure S2 plots.
- `analyze_perturber.py --preds <dir> --distances <dir> --out <dir>` writes the stratified
  error table S7 reads.
- `si_common.py` holds the roots, the LaTeX helpers and `emit`; `si_nuclei.py` and the
  other `si_*.py` files are table scripts.

## Order

`scripts/reref_diagnostics.py` before S3-S4 and `emit_text_numbers.py`;
`build_refdb_census.py` before S5; `analyze_perturber.py` before S7. The scoring reports
behind S11-S15, S17, S21 and S23-S24 are written by the `benchmarks/` scorers
(`REPRODUCTION.md` gives the commands). The remaining scripts are independent.
