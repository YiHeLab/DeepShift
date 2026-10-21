# Reproducing DeepShift

`README.md` covers using the released model. This file covers the paper: the dataset, the
five checkpoints, and every figure and table the manuscript prints. `PROTOCOL.md` is the
recipe for rebuilding the dataset and retraining the checkpoints.

## What reproduces from what

| Tier | Inputs | Result |
|---|---|---|
| A | this repository and the released checkpoints | predictions from any structure (`README.md`) |
| A+ | tier A, the deposited build inputs (the BMRB, UniProt and PDB snapshots, the BMRB-to-PDB pairs, about 40 GB of coordinate files and the p100 fence) and UCBShift's own test targets | the dataset rebuilt and the five checkpoints retrained (`PROTOCOL.md`) |
| B | tier A and the reproduction bundle | every figure and table except those in the last section |
| C | tier B and a third-party file or the deposited coordinates | the items in the last section |
| D | tier A, the third-party predictors and the case-study inputs named in `experiments/` | the competitor outputs and case-study measurements of the bundle, made again (`experiments/README.md`) |

The bundle holds the finished measurements the producers read: scored cell sets, held-out
predictions, competitor outputs and bootstrap dumps. These are outputs of runs, not
inputs to them, and cannot be derived from the Parquet tables in `data/`. The bundle
leaves out the coordinate files (deposited with the build inputs, see `PROTOCOL.md`), the atom clouds of the training cache (rebuilt by the
same command) and the checkpoints (under `models/`). Third-party files are named where a
producer needs one. Every producer runs in seconds to minutes on a laptop.

The Parquet tables are an output of the build, not an input: nothing in this repository
reads them back into a training cache.

## Rebuilding the released dataset

On 21 October 2026 the dataset was rebuilt with this release from the deposited build
inputs, in a fresh directory, and `scripts/export_dataset.py` wrote the three Parquet
tables byte-identical to those in `data/`. The fold assignment, the rereferencing offsets,
the flag file and every array of the training cache and the atom clouds were identical to
the original build's as well. With the deposit unpacked into `$IN` and the tools and
`SHIFT_UCB_TEST_TARGETS` set as in `PROTOCOL.md`:

```bash
deepshift data build --root "$DEEPSHIFT_RUN" --folds 0 1 2 3 4 5 6 \
  --snapshot "$IN/bmrb_snapshot" --uniprot-snapshot "$IN/uniprot_snapshot" \
  --pdb-seqres-db "$IN/pdb_seqres_db" --pdb-method-cache "$IN/pdb_method_resolution.json" \
  --pairs "$IN/pairs.csv" --data-archive "$IN/structures" \
  --mmseqs-bin "$MMSEQS_BIN" --blastp-bin "$BLASTP_BIN" --shiftx-bin "$SHIFTX_BIN" \
  --partner-sample-policy sample_matched \
  --flagged "$IN/p100_fit/qc_flags_p100_structural_oof64_all7.json" \
  --p100-cutoffs "$IN/p100_fit/p100_oof64_cutoffs.json" \
  --split-seed 20260701 --holdout-source-tol 0.30 --allow-atom-rejections --workers 12
python scripts/export_dataset.py --out rebuilt
for t in shifts structures sequences; do
  cmp rebuilt/deepshift_$t.parquet data/deepshift_$t.parquet
done
```

The deposited p100 fence stands in for the fit of `PROTOCOL.md` section 2, which needs the
five round-1 models. The build took five hours on a shared 20-core workstation.
Intermediate files carry an identifier formed from the build directory's name, so they
equal the original build's byte for byte only under the same name; the Parquet tables
carry no such identifier.

## Pointing the scripts at the bundle

```bash
export DEEPSHIFT_BUNDLE=/path/to/unpacked/bundle
set -a; source "$DEEPSHIFT_BUNDLE/roots.env"; set +a
```

`roots.env` sets every root variable named below to its directory in the bundle.
`DEEPSHIFT_RUN` is the finished run: `data/`, `work/unified_cache/`, `final/` and
`benchmark/`. An unset root resolves to a placeholder such as `__DEEPSHIFT_RUN_REQUIRED__`,
so a script still imports and its first read fails naming the variable. Figure scripts run
from `figures/` and write to `figures/out/` (`DEEPSHIFT_FIGURES_OUT` names another
directory); table scripts run from `si_build/` and write to `si_build/out/` (`SI_OUT`).
`figures/README.md` and `si_build/README.md` list every root.

## Benchmark scorers

Every float that compares predictors on fold 6 is scored on one frozen cell set, so a
number in Table 1 and a number in Table S12 are the same cells. `coverage_census` measures
what share of DeepShift's scorable cells each arm produced, and `master_cells` freezes, per
nucleus, the cells produced by every arm that covers at least 90 % of it. An arm below that
floor is scored on the master cells it produced, and its table rows print the count and
carry a marker. The arm roster is `benchmarks/config.py`; adding a predictor is a line
there and a rerun of everything below. The scorers read each arm's outputs from
`$DEEPSHIFT_BENCH_OUT/<arm>/<structure_id>.json`, one file per structure of the task table
`benchmarks.cells` writes there, with a `hitlog/` directory for every arm that retrieves
reference shifts. `DEEPSHIFT_ARM_DIRS` (`arm=directory`, comma-separated) names an arm run
elsewhere. The bundle carries the census (`$DEEPSHIFT_COVERAGE_CENSUS`) and the master set.

Run in this order. `<scored>` is `$SI_FINAL_SCORES_DIR`.

```bash
export DEEPSHIFT_RUN=/path/to/the/run
export DEEPSHIFT_BENCH_OUT=/path/to/the/competitor/outputs
MASTER=$DEEPSHIFT_BENCH_OUT/master_cells/master_cells_pf14.npz
CENSUS=$DEEPSHIFT_COVERAGE_CENSUS
UNFENCED=$(dirname $CENSUS)/master_cells_p100_with_anticorrelated.npz
ARMS="ucbshift_full_leaky ucbshift_full_sameprot ucbshift_x ucbshift2_sameprot \
shiftx2_leaky shiftx2_sameprot shiftx2_noshifty sparta napshift legolas procs15 nmrgnn \
camshift mlff plmcs"

python -m benchmarks.coverage_census --out-dir <census>
python -m benchmarks.master_cells --census $CENSUS --out $MASTER
python -m benchmarks.score_by_source_tier --tier 2 --metrics rmse,mae,medae \
    --master-cells $MASTER --leaky ucbshift_full_leaky shiftx2_leaky \
    --tools $ARMS --out <scored>/score_tier2_master.json
python -m benchmarks.score_paired_tier --tier 2 \
    --master-cells $MASTER --leaky ucbshift_full_leaky shiftx2_leaky \
    --tools $ARMS --out <scored>/score_tier2_master_paired.json \
    --check_marginal <scored>/score_tier2_master.json
python -m benchmarks.score_source_matched --out <scored>/source_matched.json
python -m benchmarks.score_three_source --tier 2 --master-cells $MASTER \
    --tools ucbshift_full_sameprot ucbshift_x ucbshift2_sameprot shiftx2_sameprot \
    shiftx2_noshifty sparta napshift legolas procs15 nmrgnn camshift mlff plmcs \
    --out <scored>/three_source.json
python -m benchmarks.score_by_nucleus --tier 2 --master-cells $MASTER \
    --tools ucbshift2_sameprot shiftx2_sameprot --out <scored>/score_by_nucleus_tier2.json
python -m benchmarks.score_pool_af_xtal --master-cells $MASTER \
    --leaky ucbshift_full_leaky shiftx2_leaky \
    --tools $ARMS ucbshift_full_selfexcl ucbshift2_selfexcl shiftx2_selfexcl \
    --out <scored>/pool_af_xtal.json
python -m benchmarks.score_fence_effect --master-cells $MASTER \
    --p100-cutoffs $SI_P100_CUTOFFS --check <scored>/score_tier2_master.json \
    --out <scored>/fence_effect.json
python -m benchmarks.dump_paired_cells --tools procs15 \
    --master-cells $MASTER --out <homology>/ds_procs15_cells.npz
# The comparison without the anticorrelated-channel exclusion
python -m benchmarks.master_cells --census $CENSUS --keep-anticorrelated --out $UNFENCED
python -m benchmarks.derive_anticorrelated_channels --master-cells $UNFENCED \
    --fenced $MASTER --out <rederived>/anticorrelated_channels_derived.json
python -m benchmarks.cprime_mirror_test --out <dir>/cprime_mirror_test.json
python -m benchmarks.score_by_source_tier --tier 2 --metrics rmse,mae,medae --allow-anticorrelated \
    --master-cells $UNFENCED --leaky ucbshift_full_leaky shiftx2_leaky \
    --tools $ARMS --out <unfenced>/score_tier2_master.json
python -m benchmarks.score_paired_tier --tier 2 --allow-anticorrelated \
    --master-cells $UNFENCED --leaky ucbshift_full_leaky shiftx2_leaky --tools $ARMS \
    --out <unfenced>/score_tier2_master_paired.json \
    --check_marginal <unfenced>/score_tier2_master.json
```

`score_paired_tier --check_marginal` refuses to write unless its point estimates equal the
marginal report's, cell for cell. `score_source_matched` reads the five held-out
cross-validation dumps (`--preds` names them; the default is the run's own).
`score_fence_effect` re-derives the p100 and structural flags over all seven folds of the
training cache, whose atom arrays come from the coordinates, so it runs where the cache has
been rebuilt; its report is in the bundle. Further measurements behind individual items:

```bash
python -m benchmarks.score_cys_modes --master-cells $MASTER --out <cys_report> --cells-out <cys_cells>
python -m benchmarks.dump_ring_cells --out <ring_cells>
python -m benchmarks.dump_cv_torsions --out <torsion_cells>
python -m benchmarks.score_error_covariates --out <covariate_report>
python -m benchmarks.score_data_composition --floor F --redundant R --diverse D \
    --one-per-protein P --shipped-all S --arms-report <report> \
    --out-comparison <c.json> --out-source-matched <m.json>
```

Same-protein exclusion. PLM-CS is scored with every fold-6 label set of the same protein
withheld from its training corpus (`benchmarks/reference/plmcs_same_protein.json`, 354 label
sets, written by `python -m benchmarks.build_same_protein_roster`); RefDB is not
redistributed, and without `DEEPSHIFT_REFDB_LIBRARY` the scorer reads the frozen file names
of the corpus, `benchmarks/reference/plmcs_corpus_listing.json` (2,429 BMRB accessions, no
shifts). UCBShift, UCBShift 2.0 and SHIFTX2's SHIFTY+ run with the lists in
`benchmarks/reference/competitor_same_protein.json`, written by
`python -m benchmarks.build_competitor_exclusions` (`--fetch-uniref100` refreshes
`reference/uniref100_map.json`). Same protein means the same BMRB, a shared UniProt
accession (or UniRef100 cluster), or at least 95 % identity over at least 90 % of the
shorter sequence. The UCBShift library is read from its BLAST database
(`DEEPSHIFT_UCBSHIFT_BLASTDB`, dumped with `BLASTDBCMD_BIN`); the UCBShift-200 lists are
`ucb200_competitor_same_protein.json`. `tests/test_competitor_same_protein.py` fails if a
blocked arm under `DEEPSHIFT_BENCH_OUT` used one of those references.

Every score on UCBShift's 200-protein test set leaves out the cells UCBShift's paper
removes, as `benchmarks/reference/ucb200_paper_exclusions.json` lists them: every C' of
three chains whose C' is anticorrelated with prediction, and nine ESI cells.

## Figures and tables

Run figure scripts from `figures/` and table scripts from `si_build/` (a path with `..` is relative to that directory). `<scored>` is
`$SI_FINAL_SCORES_DIR`; `$MASTER` and `$UNFENCED` are as above.

| Item | Script and command | Further inputs |
|---|---|---|
| Figure 1, Figure 2 (upper), Figures S7, S8 | `python build_paper_figures.py` | `DEEPSHIFT_RUN`; `DEEPSHIFT_SCALING_LADDER`; `DEEPSHIFT_RAMACHANDRAN_OOF` |
| Figure 2 (lower), Table 1 | `python fig_backbone_benchmark.py --marginal <scored>/score_tier2_master.json --paired <scored>/score_tier2_master_paired.json --marginal-without-exclusions <unfenced>/score_tier2_master.json --paired-without-exclusions <unfenced>/score_tier2_master_paired.json` | Table 1 is `tab_accuracy_common_rmse.tex`, with its MLFF rows from Table S12 |
| Figure 3 (upper) | `python fig_source_matched.py --scores <scored>/source_matched.json` | |
| Figure 3 (lower) | `python fig_sidechain_benchmark.py --scores <scored>/score_by_nucleus_tier2.json` | |
| Figure 4, Figure S9 | `python fig_structure_response.py` | `DEEPSHIFT_CYS_CELLS`, `DEEPSHIFT_RING_CELLS`, `DEEPSHIFT_DECOY_POOLED` |
| Figure 5 | `python fig_xeisd_mixed_selection.py` | `DEEPSHIFT_XEISD_PROTOCOL` |
| Figure S1 | `python fig_census_merged.py` | `DEEPSHIFT_RUN` |
| Figure S2 | `python ../si_build/emit_homology_axis.py --out-dir <dose>` (reads `DEEPSHIFT_BENCH_OUT`), then `python fig_homology_training_identity.py` | `DEEPSHIFT_HOMOLOGY_CELLS` (the `dump_paired_cells` output), `DEEPSHIFT_LEAKAGE_DOSE_ROOT` (`<dose>`), `DEEPSHIFT_HOMOLOGY_BENCHMARK_ROOT` |
| Figure S3 | `python fig_accuracy_by_nucleus_all.py --scores <scored>/score_by_nucleus_tier2.json` | |
| Figure S4 | `python fig_selflookup_transfer.py --master-cells $MASTER` | `DEEPSHIFT_SELFLOOKUP_CELLS`, `DEEPSHIFT_BENCH_OUT`, `DEEPSHIFT_UCBSHIFT_LEAKAGE` |
| Figure S5 | `python fig_plddt_source.py --fence <qc_flags_p100_structural.json>` | AlphaFold model files (tier C) |
| Figure S6 | `python fig_exp_source_control.py` | `DEEPSHIFT_EXP_SOURCE_CONTROL` |
| Figure S10 | `python fig_uncertainty_calibration.py` | `DEEPSHIFT_RUN` |
| Figure S11 | `python fig_downstream_applications.py` | `DEEPSHIFT_IDP8_ROOT`, `DEEPSHIFT_TAU_BENCHMARK_ROOT` |
| Figure S12 | `python fig_xeisd_composition.py` | `DEEPSHIFT_XEISD_PROTOCOL` |
| Table S2 | `python gen_census_tables.py --flags <qc_flags_p100_structural.json>` | |
| Tables S3, S4 | `python ../scripts/reref_diagnostics.py` (writes `reref_diagnostics.json` into `DEEPSHIFT_RUN/data` unless `--out` is given), then `python si_reref_offsets.py` | `SI_REREF_DIAGNOSTICS` (the diagnostics output) |
| Table S5 | `python build_refdb_census.py --out <dir>/refdb_census.json` (tier C), then `python si_refdb_agreement.py` | `SI_REFDB_CENSUS`; the census is in the bundle |
| Table S6 | `python si_qc_p100.py` | |
| Table S7 | `python analyze_perturber.py --preds <it>/work/preds --distances <it>/work/distances --out <it>/out`, then `python emit_perturber_dose_ppm.py` | `SI_PERTURBER_REBUILD` (`<it>`, below) |
| Table S8 | `python emit_leakage.py` | `SI_LEAKAGE_ARTIFACTS_DIR` (tier C) |
| Table S9 | `python emit_train_distance.py [--against NAME=FASTA]` | `SI_UCB_TRAINING_FASTA` (tier C) |
| Table S10 | `python si_nuclei.py` | |
| Table S11 | `python si_competitor_ledger.py` | `DEEPSHIFT_COVERAGE_CENSUS`, `SI_FINAL_SCORES_DIR` |
| Tables S12 to S14 | `python emit_competitor_tables_final.py` | `SI_FINAL_SCORES_DIR` |
| Table S15 | `python emit_sidechain_final.py` | `SI_FINAL_SCORES_DIR` |
| Table S16 | `python ../figures/si_tables_literature.py` | `DEEPSHIFT_PAPER_PATH`, `DEEPSHIFT_REFDB_CHAINS` (tier C) |
| Tables S17, S21 | `python emit_pool_tables_final.py` | `SI_FINAL_SCORES_DIR` |
| Table S18 | `python build_reproduction_gate.py` | `SI_COMPETITOR_ROOT`, `SI_GATE_UCB200_RERUN`, `SI_GATE_CSPRED_TM`, `SI_UCB200_POOL61` (tier C) |
| Table S19 | `python emit_ucb200_common.py` | `SI_UCB200_ARTIFACTS_DIR`, `SI_UCB200_COMPETITOR_CELLS` |
| Table S20 | `python emit_ucb200_table.py` | `SI_UCB200_ARTIFACTS_DIR`, `SI_UCB200_POOL61`, `SI_SHIFTX2_61_LOG` |
| Table S22 | `python emit_data_composition.py` | `SI_DATA_DIVERSITY_DIR`, `SI_SOURCE_TRANSFER_DIR` (else `DEEPSHIFT_ARCHIVE`) |
| Table S23 | `python emit_cys_modes.py` | `SI_CYS_MODES_REPORT` |
| Table S24 | `python emit_error_covariates.py` | `SI_ERROR_COVARIATES_REPORT` |
| Table S25 | `python build_decoy_ledger.py` | `DEEPSHIFT_DECOY_POOLED`, `SI_DECOY_COHORT` |
| Table S26 | `python build_interval_calibration.py --calibration <interval_calibration.json> --evaluation <interval_evaluation_fold6_pf14.json>` | |
| Table S27 | `python emit_speed.py` | `SI_SPEED_CENSUS` (else `DEEPSHIFT_ARCHIVE`) |
| Table S28 | `python build_idp8_roster.py` | `DEEPSHIFT_IDP8_ROOT` |

Table S1 (the dataset census) has no producer; its counts are those of the Parquet tables.
Every table's caption is the authors' text, kept in `si_build/reference_inputs/captions/`
and spliced in by `si_common.emit`. A producer does not compare its output with the
published numbers; compare a regenerated artifact with the published one directly.

### Numbers printed in the text

`python emit_text_numbers.py` (from `si_build/`) recomputes the numbers the text prints with
no float behind them and writes `out/text_numbers.json`, which maps each name to its value
and a description. It reads `DEEPSHIFT_RUN`, `DEEPSHIFT_XEISD_PROTOCOL`, `SI_FINAL_SCORES_DIR`
(the fence-effect and three-source reports), `SI_SELF_RETRIEVAL_UCBY`,
`SI_SELF_RETRIEVAL_SHIFTY61`, `SI_GATE_CSPRED_TM`, `DEEPSHIFT_REFDB_SHIFTS`,
`DEEPSHIFT_REFDB_LIBRARY`, `DEEPSHIFT_UCBSHIFT_LEAKAGE` and the released Parquet tables;
the last four are third-party or are not in the bundle, and `--allow-missing` skips a group
whose files are absent. The rereferencing numbers come from `scripts/reref_diagnostics.py`.

### The runs that made the measurements

The competitor outputs, the speed census behind Table S27 and the case-study measurements
the bundle holds were made by the programs in `experiments/`: the competitors (with the
patches that block same-protein retrieval in UCBShift, UCBShift 2.0 and SHIFTX2), the
UCBShift-200 stores of Tables S18 and S19 (run with the per-chain lists in
`benchmarks/reference/ucb200_competitor_same_protein.json`), SHIFTX2 on the 61-protein set
of Table S20, X-EISD, IDP8, tau and the 3DRobot decoys. Each needs the third-party
software its README names.

Table S7 scores residues the atom stage excludes for a nearby perturber. Its inputs come
from a shadow evaluation: the atom stage rerun on folds 0 and 6 with
`deepshift.data.exclusion.perturber_excluded_rids` returning the empty set, the five final
members evaluated on those folds with `deepshift evaluate --save-npz`, and each row's
distance to the nearest atom of each perturber category. `SI_PERTURBER_REBUILD` holds
`work/preds/paper_model_fold{0,6}.preds.npz` and `work/distances/`; both are in the bundle.

## What the release cannot ship

Five items, and part of a sixth, need something outside the bundle.

- **A third-party file.** Tables S8 and S18 read UCBShift's own split and test targets
  from a CSpred checkout, and Table S9 reads UCBShift's training sequences
  (`SI_UCB_TRAINING_FASTA`). Table S16 parses each competitor's own article and stops,
  naming the file, when one is absent from `DEEPSHIFT_PAPER_PATH`; four of those articles
  are not redistributable.
- **Coordinates.** Figure S5 reads each AlphaFold model's pLDDT from the model file, in the
  deposited coordinate archive; the producer stops when the files are missing.
- **Part of Table S5.** The table regenerates from the census in the bundle; rebuilding the
  census needs RefDB's correction reports (`DEEPSHIFT_REFDB_CORRECTIONS`).
