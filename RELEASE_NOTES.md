# DeepShift v1.1.3

v1.1.3 (11 November 2026) makes the prediction intervals the default of `deepshift predict`
and corrects the reproduction instructions. The models, the dataset and every prediction
are those of v1.0.

- `deepshift predict` writes the 90 % interval file by default: with all five members it
  uses `interval_calibration.json` from the directory of the first `--checkpoint`, so the
  five-model command no longer needs `--interval-calibration`, which still names another
  file. `--no-intervals` turns the default off. With fewer members, or with no calibration
  file, it predicts without intervals and prints one line saying why
  (`src/deepshift/prediction.py`).
- `REPRODUCTION.md`: commands that failed as written against the bundle are corrected. It
  names the path of Figure 3's source-matched report, what `<unfenced>` and the flag file
  are, the `--allow-missing` that `emit_text_numbers.py` needs and the two groups it then
  skips, the held-out dumps `score_source_matched` needs and the bundle lacks, and a copy
  for Table S7's `analyze_perturber.py` output, which would otherwise overwrite the
  bundle's; Table S8 regenerates from the bundle alone.
- Clearer stops: `figures/fig_leakage_audit.py` names a missing text-number group and the
  variable it needs instead of raising a `KeyError`, `si_build/build_reproduction_gate.py`
  names `SI_GATE_CSPRED_TM` when UCBShift's test targets are absent, and
  `si_build/emit_train_distance.py` checks its inputs before it downloads MMseqs2.
- Four `experiments/` READMEs name the Zenodo archives that exist:
  `deepshift_mlff_relaxed_fold6_tau.tar.gz` (the fold-6 chains and the tau conformers) and
  `deepshift_mlff_relaxed_decoys.tar.gz` (the 3DRobot decoys).
- Table S15: the footnote starts below the table instead of beside it
  (`si_build/emit_sidechain_final.py`).
- At Zenodo, the reproduction bundle no longer contains workstation names and now includes
  UCBShift's test-set definition (`ucbshift_test_targets/ucbshift200_test_targets.json`),
  under the terms quoted in `THIRD_PARTY_NOTICES.md`; `PROTOCOL.md` says it can be used in
  place of converting `test_200.pkl`.
- The Colab notebook installs the package from tag v1.1.3 and downloads the models from the
  v1.0 release page, as before.

# DeepShift v1.1.2

v1.1.2 (11 November 2026) changes how Table 1 marks significance, removes two
supplementary figures and fixes the Colab notebook. The models, the dataset and every
prediction are those of v1.0.

- Table 1: an asterisk now marks a lead that is significant after Bonferroni correction
  over the table's 18 (source, nucleus) cells, in place of a 95 % paired interval that
  excludes zero. 16 of the 18 leads are marked (`figures/fig_backbone_benchmark.py`).
- Supplementary Figures 11 and 12 are removed, with their scripts
  `figures/fig_downstream_applications.py` and `figures/fig_xeisd_composition.py`: they
  repeated Figure 5.
- Figure 5: panel (a) labels the UCBShift value the text reports, and panel (d) drops the
  C-alpha asterisk (`figures/fig_applications.py`). Figure 3 drops the carets under the
  nuclei no competitor predicts (`figures/fig_sidechain_benchmark.py`).
- Two table footnotes are shorter: Table S2's (`si_build/gen_census_tables.py`) and
  Table S9's (`si_build/emit_train_distance.py`).
- Colab notebook: with fewer than five members it predicts without an interval instead of
  stopping; usage is reported only inside Colab, and the notebook and `colab/README.md`
  list exactly the fields the record holds; the receiver (`colab/telemetry/Code.gs`)
  stores every field as text, refuses a malformed record and splits a shared structure
  across rows. The notebook installs the package from tag v1.1.2 and downloads the five
  models and the interval calibration from the v1.0 release page, as before.
- At Zenodo, the reproduction bundle is one archive, `reproduction_bundle.tar.gz`, which
  holds everything the separate `reproduction_bundle_v1.1_additions.tar.gz` added; that
  archive is gone. The bundle discloses the measured UCBShift test-set shifts of the
  scored cells, under the terms quoted in `THIRD_PARTY_NOTICES.md`.
- Documentation, comments and error messages that disagreed with the code are corrected.

# DeepShift v1.1.1

v1.1.1 (11 November 2026) changes one table. The models, the dataset and every prediction are
those of v1.0.

- Table S7 (prediction error near a perturber): the bold cells are now significant at 0.05
  after Bonferroni correction over the eligible cells of the six printed nuclei (241 cells),
  in place of Benjamini-Hochberg over every eligible cell. 45 cells are bold, where 86 were.
  The RMSE values and the Welch t-test p-values are unchanged
  (`si_build/analyze_perturber.py`, `si_build/emit_perturber_dose_ppm.py`).
- At Zenodo, `reproduction_bundle_v1.1_additions.tar.gz` also holds the table's
  `stratified_errors_experimental.csv` with the corrected p-values.

# DeepShift v1.1

v1.1 (9 November 2026) adds the code and the measurements behind the figures of the Nature
Methods version of the manuscript. The models, the dataset and every prediction are those of
v1.0.

- `benchmarks/assignment_accuracy.py`: resonance-assignment accuracy on the fold-6 benchmark
  (Figure 5a), from a `benchmarks.dump_paired_cells` dump.
- `figures/fig_pipeline.py` (Figure 1), `figures/fig_applications.py` (Figure 5; it holds the
  X-EISD panels of the former `fig_xeisd_mixed_selection.py`), `figures/fig_leakage_audit.py`
  (Extended Data Fig. 2) and `figures/journal_style.py`, the page they are built on. The
  architecture schematic of `build_paper_figures.py` is now Extended Data Fig. 1.
- `benchmarks/self_retrieval.py`: the UCBShift-Y copy count, shared by Extended Data Fig. 2 and
  `si_build/emit_text_numbers.py`, whose printed numbers are unchanged.
- The README gives the supported platforms, the CPU-only install, a quick start on the bundled
  structure, and the input and output formats.
- At Zenodo, `reproduction_bundle_v1.1_additions.tar.gz` (unpack it over the bundle) holds the
  assignment-accuracy outputs and the two inputs of Extended Data Fig. 2 the bundle lacked.

# DeepShift v1.0

Structure-only prediction of protein chemical shifts. An atom transformer reads a 3D
structure through interatomic distances alone and predicts 64 nuclei per residue: no
homology transfer, no sequence features, no retrieval.

## What is in this release

- `models/deepshift_fold{1..5}.pt` — the five-member cross-validation ensemble the paper
  reports. Each member holds out its own fold and carries its own normalization statistics.
- `models/interval_calibration.json` — the calibration behind the 90 % prediction intervals,
  fitted on fold 0 and evaluated once on fold 6.
- `data/deepshift_{shifts,structures,sequences}.parquet` — the dataset. The shifts table
  holds 17,118 label sets; the 12,710 of them with a fold are the paper's dataset, over
  12,242 BMRB depositions, realized as 21,346 structure pairings. Shifts are rereferenced
  and the exclusion fence is applied; `data/README.md` is the dictionary.
- The scripts behind every figure and table in the paper, under `figures/` and `si_build/`,
  and the competitor scorers under `benchmarks/`.
- The programs that ran the competitors and the case studies (X-EISD, IDP8, tau, the
  3DRobot decoys), under `experiments/`, with the install steps of every third-party
  predictor.
- `scripts/map_uniprot.py`, the procedure that made the frozen BMRB-to-UniProt mapping.

## Getting predictions

`colab/DeepShift_predict.ipynb` runs in Colab with no local setup
([open in Colab](https://colab.research.google.com/github/YiHeLab/DeepShift/blob/main/colab/DeepShift_predict.ipynb)):
give it a PDB id or a file and it returns per-residue shifts with prediction intervals. Locally, `pip install .`
then:

```bash
deepshift predict --pdb figures/reference/1POH.pdb --chain A \
  --checkpoint models/deepshift_fold1.pt --checkpoint models/deepshift_fold2.pt \
  --checkpoint models/deepshift_fold3.pt --checkpoint models/deepshift_fold4.pt \
  --checkpoint models/deepshift_fold5.pt \
  --interval-calibration models/interval_calibration.json --out shifts.csv
```

## Reproducing the paper

`REPRODUCTION.md` states what reproduces from this release alone and what needs a
third-party file. `PROTOCOL.md` is the recipe for rebuilding the dataset and retraining
the five models. The build inputs (the BMRB, UniProt and PDB snapshots of July 2026, the
coordinate files and the p100 fence), the relaxed structures the MLFF predictor was run on
and the measurements behind every figure and table are deposited at Zenodo:
[doi:10.5281/zenodo.23550867](https://doi.org/10.5281/zenodo.23550867) is the v1.0 version
of that record, and [doi:10.5281/zenodo.23550866](https://doi.org/10.5281/zenodo.23550866)
names all its versions and resolves to the latest.

## How the released files were made

The dataset and the five members were made with the development version of this code.
Preparing the release removed record-keeping (input hashes, provenance files, run stamps)
and options the paper does not report. Rebuilding the dataset from the deposited inputs
with this release gives the three Parquet tables in `data/` byte for byte
(`REPRODUCTION.md`). The training code is the code that trained the members with those
options removed; the members were not retrained with it. Training is not bit-reproducible
across GPUs (members 1 to 3 were trained on RTX 5090s, 4 and 5 on A100s), so a member
retrained with `PROTOCOL.md` is not bit-identical to the released one.

## Note on third-party code

SHIFTX is not redistributed here. It is fetched from its authors' own server at build time
against a pinned checksum; see `THIRD_PARTY_NOTICES.md`. No competitor's source is included
either: UCBShift, UCBShift 2.0 and SHIFTX2 were run with the changes in
`experiments/competitors/patches/`, applied to the upstream releases named there.
