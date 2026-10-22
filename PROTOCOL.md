# The DeepShift protocol

The complete recipe for rebuilding the dataset and retraining the five ensemble members:
the folds, the quality-control fit, the training runs, the prediction intervals and the
dataset export, in the order they run, with the condition each step must meet. The
configuration is printed in machine-readable form by:

```bash
deepshift protocol --out final_protocol.json
```

Training runs twice. The first round of five members is trained without any exclusion and
serves only to fit the p100 exclusion spans; the second round is the released ensemble,
trained under the p100-plus-structural exclusion fence. Nothing here launches
automatically. The figures and tables are regenerated as described in `REPRODUCTION.md`.

## Stage graph

`deepshift data build` runs the stages in this order; `--from-stage` and `--to-stage` select
a part of the graph. Each stage checks that its inputs exist before it runs and its
outputs after.

    fetch → assembly → uniprot_snapshot → discover → materialize → cluster → folds
          → partner_suppression → split → reref → cache → atom_census → atoms → flags

## Inputs

The inputs of the paper's build are deposited at Zenodo
([doi:10.5281/zenodo.23550867](https://doi.org/10.5281/zenodo.23550867)) exactly as the
build read them: the BMRB macromolecule archive mirrored from
`rsync://rsync.bmrb.io/macromolecules/` on 10 July 2026 (the top-level NMR-STAR file of
17,865 entries), the BMRB-to-PDB pairs of 7 July 2026, the `pdb_seqres` BLAST database of
8 July 2026, the PDB method and resolution table of 24 July 2026, the UniProt mapping
captured on 14 July 2026, the coordinate files discovery may select (RCSB PDB entries and
AlphaFold DB version 6 models, downloaded in July 2026) and the p100 fence
(`p100_oof64_cutoffs.json` and `qc_flags_p100_structural_oof64_all7.json`). The UCBShift-200
test targets are not redistributed (`THIRD_PARTY_NOTICES.md`); they are converted from
UCBShift's own file, below.
Given these files the build calls no web service; a stage whose input is not supplied
fetches the current version from its source instead. `scripts/map_uniprot.py` is the
procedure that made the UniProt mapping, against live services.

The UCBShift-200 test targets are read from a JSON file mapping each test id to its
one-letter sequence. Convert CSpred's `test_200.pkl` once, in your own shell:

```bash
python -c 'import json, pickle, sys; json.dump(pickle.load(open(sys.argv[1], "rb")), open(sys.argv[2], "w"), sort_keys=True, indent=1)' /path/to/CSpred/test_200.pkl /path/to/ucbshift200_test_targets.json
```

Set the paths below, then install the package so that `deepshift` and the scripts import it:

```bash
export DEEPSHIFT_REPO=/path/to/this/repository
export DEEPSHIFT_RUN=/path/to/a/fresh/final-run
export SHIFT_UCB_TEST_TARGETS=/path/to/ucbshift200_test_targets.json
export BMRB_SNAPSHOT=/path/to/bmrb_snapshot
export UNIPROT_SNAPSHOT=/path/to/uniprot_snapshot
export PDB_SEQRES_DB=/path/to/pdb_seqres_db
export PDB_METHODS=/path/to/pdb_method_resolution.json
export BMRB_PDB_PAIRS=/path/to/pairs.csv
export STRUCTURE_ARCHIVE=/path/to/data_archive
export MMSEQS_BIN=/path/to/mmseqs
export BLASTP_BIN=/path/to/blastp
export SHIFTX_BIN=/path/to/shiftx

pip install -e "$DEEPSHIFT_REPO"
```

| Input | Content |
|---|---|
| `SHIFT_UCB_TEST_TARGETS` | the UCBShift-200 test targets; the folds stage refuses to run without them |
| `BMRB_SNAPSHOT` | the raw BMRB NMR-STAR files |
| `UNIPROT_SNAPSHOT` | the BMRB-to-UniProt mapping: three frozen JSON files the `uniprot_snapshot` stage merges, made by `scripts/map_uniprot.py` |
| `PDB_SEQRES_DB` | the `pdb_seqres` BLAST database (`pdb_seqres_prot.*`) |
| `PDB_METHODS` | `pdb_method_resolution.json`, the experimental method and resolution of each PDB entry |
| `BMRB_PDB_PAIRS` | `pairs.csv`, the BMRB-to-PDB pairings |
| `STRUCTURE_ARCHIVE` | the directories `pdbs/`, `crystal_pdbs/` and `alphafold/` of coordinate files that discovery may select; the build links them into its data directory |
| `MMSEQS_BIN`, `BLASTP_BIN` | the MMseqs2 and BLAST+ executables |
| `SHIFTX_BIN` | a SHIFTX executable; without `--shiftx-bin` the build downloads SHIFTX from upstream against a pinned SHA-256 (`THIRD_PARTY_NOTICES.md`) |

## 1. Correct folds and unfiltered cache

Create an explicit empty exclusion authority, then build all seven folds through the atom
stage. `sample_assembly.json` is derived from the NMR-STAR files. It records the remaining
protein sequences and copy counts of each label set after the target molecule is removed.
A coordinate partner enters only if it consumes one of those slots at at least 95 %
identity and 90 % coverage over at least 10 residues. Unknown, nonmatching and excess
lattice chains are excluded.

```bash
python -m deepshift.quality_control \
  --policy none \
  --out "$DEEPSHIFT_RUN/data/qc_flags_unfiltered_calibration.json"

deepshift data build \
  --root "$DEEPSHIFT_RUN" \
  --folds 0 1 2 3 4 5 6 \
  --snapshot "$BMRB_SNAPSHOT" \
  --uniprot-snapshot "$UNIPROT_SNAPSHOT" \
  --pdb-seqres-db "$PDB_SEQRES_DB" \
  --pdb-method-cache "$PDB_METHODS" \
  --pairs "$BMRB_PDB_PAIRS" \
  --data-archive "$STRUCTURE_ARCHIVE" \
  --mmseqs-bin "$MMSEQS_BIN" \
  --blastp-bin "$BLASTP_BIN" \
  --shiftx-bin "$SHIFTX_BIN" \
  --partner-sample-policy sample_matched \
  --flagged "$DEEPSHIFT_RUN/data/qc_flags_unfiltered_calibration.json" \
  --split-seed 20260701 \
  --holdout-source-tol 0.30 \
  --allow-atom-rejections \
  --to-stage atoms
```

A shift-bearing residue whose structure gives it no atom cloud cannot be a training row.
The atom stage drops it and writes the drop with its reason to `atom/rejections.json` in
that fold; the build stops unless `--allow-atom-rejections` accepts the listed drops.

The fold-6 holdout is one deterministic draw of whole leakage components under the split
seed. The folds stage requires the draw to hold out 9 to 11 % of label sets, residues and
depositions, and each structure source within `--holdout-source-tol` of the overall share.
The source tolerance is 30 %: crystal label sets are the smallest source and sit in the
clumpiest components, and over one hundred seeds their deviation exceeds 15 % in almost
half the draws and 30 % in one draw in twenty, while the other two sources stay under
15 %. A draw that misses a bound is answered by one deliberate change of a parameter, never
by trying seeds until one passes.

The external case-study proteins (`src/deepshift/data/case_study_excluded_bmrbs.json`: all eight proteins
of the IDP8 study, tau and drkN SH3; MEVN and RSP8 have no BMRB entry and are named by
sequence) leave training by the same direct rule as the UCBShift-200 boundary (same
deposition, same UniProt, or a homolog at identity 0.65 over coverage 0.60), plus any
record sharing a segment of at least 40 aligned residues at identity 0.90 whatever its
coverage (a 76-residue tau segment at 99 % identity covers only 59 % of K18). They are
pinned to fold 0 after the draw rather than through it, so the draw and every other
assignment are unchanged. `$DEEPSHIFT_RUN/data/fold_case_study_pin.json` lists each label set moved with
its previous fold; `$DEEPSHIFT_RUN/data/fold_case_study_boundary.json` records how each seed record was
caught, and the stage refuses any label set that satisfies the rule while staying in a
training fold. A case-study label set drawn into fold 6 is refused. The p100 fence is a
property of the corpus and is not refitted for such a rebuild.

Conditions:

- every sequence search was an exhaustive all-versus-all `easy-search` with E-value 10000
  and result caps equal to the sequence count. The training-fold families (entity sequences
  and modeled regions) ran at identity 0.80 and coverage 0.90 in coverage modes 0 and 1
  (`cluster.MMSEQS_COVERAGE_MODES`: both sequences covered, and the target alone covered).
  The fold-0 benchmark arm ran at identity 0.65 and coverage 0.60 in coverage modes 1 and 2
  (`folds.UCB_HOMOLOGY_COVERAGE_MODES`: the shorter sequence covered);
- no similarity, shared-deposition, shared-UniProt, modeled-region or co-solved edge crosses
  two of folds 1 to 6;
- `fold_balance_report.json` reports, per edge family, the number of edge groups that span
  fold 0 and a training fold, and `$DEEPSHIFT_RUN/data/fold0_training_relatives.json` lists the fold-0
  label sets in those groups. Fold 0 is the interval-calibration set;
  `$DEEPSHIFT_RUN/data/fold0_calibration_exclusions.json` is what the interval fit drops: those relatives
  together with the case-study records that landed in fold 0, because a held-out protein
  must not calibrate the uncertainty of an application it is held out of;
- the partner suppression report accounts for every co-located dataset-entity pair assigned
  to different folds; each fold's `atom/config.json` records `suppression_keys_applied`, and
  the atom stage refuses when the map addresses a structure of that fold and matched none;
- every admitted coordinate partner matches a declared sample entity and copy slot;
- the atom census chooses 512 and records `precap_max` and `n_centers_over_cap`; a cloud
  over 512 atoms keeps the 512 nearest the target residue, truncation is warned and counted,
  and the count appears in each fold's `atom/config.json` as `n_truncated`; hydrogen and
  deuterium atoms are absent;
- fold 6 was assigned but is not used to fit rereferencing, normalization, quality control
  or a model.

## 2. Fit p100 from unfiltered out-of-fold predictions

Run one member for each held-out fold 1 to 5. The commands spell out the paper settings.
Eight attention heads, the 0.05 final learning-rate fraction and gradient clipping at 1.0
are fixed in the package recipe and printed by `deepshift protocol`. Each member trains on
the four folds it does not hold out and is delivered at its final epoch; the held-out fold
is scored once, after that epoch. A member fits its own per-nucleus scale from its four
training folds only (`normalization.py`), after rereferencing and exclusion of flagged
cells, so inference denormalizes each member separately before averaging in ppm.

```bash
for fold in 1 2 3 4 5; do
  deepshift train \
    --cache "$DEEPSHIFT_RUN/work/unified_cache" \
    --fold "$fold" \
    --epochs 35 \
    --batch-size 128 \
    --lr 0.0003 \
    --eta-min 0.000015 \
    --weight-decay 0 \
    --dim 384 \
    --layers 6 \
    --dropout 0.3 \
    --seed 0 \
    --ckpt-every 5 \
    --huber-delta 1.0 \
    --deterministic \
    --exclude-flagged "$DEEPSHIFT_RUN/data/qc_flags_unfiltered_calibration.json" \
    --out "$DEEPSHIFT_RUN/refold_fold${fold}"
done

python "$DEEPSHIFT_REPO/scripts/fit_p100_oof64.py" \
  --run-root "$DEEPSHIFT_RUN" \
  --source-root "$DEEPSHIFT_REPO" \
  --k 3 \
  --out-dir "$DEEPSHIFT_RUN/p100_fit"

export FINAL_FLAGS="$DEEPSHIFT_RUN/p100_fit/qc_flags_p100_structural_oof64_all7.json"
export P100_CUTOFFS="$DEEPSHIFT_RUN/p100_fit/p100_oof64_cutoffs.json"

python -m deepshift.quality_control \
  --policy p100_structural \
  --cache "$DEEPSHIFT_RUN/work/unified_cache" \
  --cutoffs "$P100_CUTOFFS" \
  --out "$DEEPSHIFT_RUN/p100_fit/derived_check/qc_flags_p100_structural.json"
cmp "$FINAL_FLAGS" "$DEEPSHIFT_RUN/p100_fit/derived_check/qc_flags_p100_structural.json"
```

`--exclude-flagged` takes a JSON object of flagged cells; an empty object requests
unfenced training, and a nonempty file that cannot be joined to the selected cache fails
instead of matching nothing. `--resume` continues an interrupted member from its
`.last.pt`.

`fit_p100_oof64.py` resolves each prochiral value in ppm space and maps the accepted value
back to its deposited atom name. Its report separates labels rejected by the accepted-span
rule from ambiguity codes 4, 5, 6 and 9 and from invalid residue and nucleus pairs; the
latter two are loss and schema concerns and do not enter the flag file.

Conditions:

- the five round-1 members trained without exclusion, each fold 1 to 5 once, at the final
  epoch;
- the fit report is `complete`, has no missing observed residue and nucleus population and
  no scientific-sanity failure: every fitted span is finite and non-empty, and Arg
  N-epsilon carries the observed-value interval `quality_control.ARG_NE_OBSERVED_RANGE_PPM`.
  H-alpha, glycine's two H-alpha included, takes its fitted span like every other nucleus.
  The report lists each span's width beside the width of its observed population;
- the composite is the exact union of p100 and the separately derived structural Cys and
  Pro cells. The last command derives the composite a second time from the round-1 cache
  through `deepshift.quality_control`, the implementation the round-2 build repeats, and
  `cmp` must report the two files byte-identical before section 3 starts.

## 3. Final cache and final members

Rebuild normalization and the flag stage under the composite fence, then train the released
members. The atom clouds do not depend on the fence, so the build resumes at the cache
stage.

```bash
deepshift data build \
  --root "$DEEPSHIFT_RUN" \
  --folds 0 1 2 3 4 5 6 \
  --partner-sample-policy sample_matched \
  --flagged "$FINAL_FLAGS" \
  --p100-cutoffs "$P100_CUTOFFS" \
  --allow-atom-rejections \
  --from-stage cache \
  --to-stage flags

for fold in 1 2 3 4 5; do
  deepshift train \
    --cache "$DEEPSHIFT_RUN/work/unified_cache" \
    --fold "$fold" \
    --epochs 35 \
    --batch-size 128 \
    --lr 0.0003 \
    --eta-min 0.000015 \
    --weight-decay 0 \
    --dim 384 \
    --layers 6 \
    --dropout 0.3 \
    --seed 0 \
    --ckpt-every 5 \
    --huber-delta 1.0 \
    --deterministic \
    --exclude-flagged "$FINAL_FLAGS" \
    --out "$DEEPSHIFT_RUN/final/fold${fold}"
done
```

The post-cache `qc_flags_p100_structural.json` must contain exactly the supplied flag
mapping. The released files in `models/` are the five `final/fold{1..5}.final.pt` members
(`models/README.md` says what a file holds).

## 4. Prediction intervals: fold 0 fit, fold 6 evaluation

Generate one prediction dump per member so the spread is recoverable. The five fold-6
forward passes form one evaluation and feed no earlier choice.

```bash
mkdir -p "$DEEPSHIFT_RUN/final/intervals"
for fold in 1 2 3 4 5; do
  deepshift evaluate \
    --cache "$DEEPSHIFT_RUN/work/unified_cache" \
    --fold 0 \
    --ckpt "$DEEPSHIFT_RUN/final/fold${fold}.final.pt" \
    --flags "$FINAL_FLAGS" \
    --save-npz "$DEEPSHIFT_RUN/final/intervals/fold0_member${fold}.npz" \
    --out "$DEEPSHIFT_RUN/final/intervals/fold0_member${fold}.json"
done

deepshift intervals fit \
  --member "$DEEPSHIFT_RUN/final/intervals/"fold0_member{1,2,3,4,5}.npz \
  --exclude-label-sets "$DEEPSHIFT_RUN/data/fold0_calibration_exclusions.json" \
  --bins 10 \
  --level 0.68 0.90 0.95 \
  --out "$DEEPSHIFT_RUN/final/interval_calibration.json"

for fold in 1 2 3 4 5; do
  deepshift evaluate \
    --cache "$DEEPSHIFT_RUN/work/unified_cache" \
    --fold 6 \
    --ckpt "$DEEPSHIFT_RUN/final/fold${fold}.final.pt" \
    --flags "$FINAL_FLAGS" \
    --save-npz "$DEEPSHIFT_RUN/final/intervals/fold6_member${fold}.npz" \
    --out "$DEEPSHIFT_RUN/final/intervals/fold6_member${fold}.json"
done

deepshift intervals evaluate \
  --member "$DEEPSHIFT_RUN/final/intervals/"fold6_member{1,2,3,4,5}.npz \
  --calibration "$DEEPSHIFT_RUN/final/interval_calibration.json" \
  --exclude-channels benchmarks/reference/anticorrelated_channels.json \
  --out "$DEEPSHIFT_RUN/final/interval_evaluation_fold6_pf14.json"
```

The interval fit accepts only fold 0 and fits on fold 0 minus the label sets in
`$DEEPSHIFT_RUN/data/fold0_calibration_exclusions.json`: the 415 fold-0 label sets with a relative in a
training fold at the cross-validation axis (section 1), plus the case-study proteins, 461
in all. `$DEEPSHIFT_RUN/data/fold0_training_relatives.json` holds only the first 415, and fitting on it
gives a different calibration from the one shipped. The calibration artifact records how
many label sets and cells were excluded. The evaluation applies the frozen artifact to fold
6 and drops the fourteen anticorrelated channels every fold-6 benchmark drops.
`models/interval_calibration.json` is this artifact; `deepshift predict` writes an
interval only when given the five members and this file.

## 5. Dataset export

```bash
python "$DEEPSHIFT_REPO/scripts/export_dataset.py"
```

Reads `$DEEPSHIFT_RUN/data`: `chemical_shifts.csv`, `entity_catalog.csv`, the three
`structure_build_log_*.csv`, `realization_ledger.csv`, `reref_offsets.json`, the fold map,
and the build's `qc_flags_p100_structural.json` with its `.report.json` (the report must
declare the `p100_structural` policy); and the curated catalogue `data/deposition_notes.tsv`
of this repository. It writes the three Parquet tables and `_export_report.json` into
`data/`; `data/README.md` is the dictionary. `--source`, `--qc-flags`, `--out`,
`--reref-offsets` and `--fold-map` name any of these explicitly.

## 6. Figures and tables

Regenerating the figures, tables and printed numbers of the paper is described in
`REPRODUCTION.md`.
