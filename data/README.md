# DeepShift dataset

The shippable, queryable snapshot of the DeepShift training data: observed protein NMR
chemical shifts paired with their structures and cross-validation folds.

`scripts/export_dataset.py` writes it from one build, as `PROTOCOL.md` §5 runs it:

```bash
python scripts/export_dataset.py
```

It reads the build's data directory through `$DEEPSHIFT_RUN` — the observed shifts, the
entity catalog, the three structure build logs, the realization ledger
(`realization_ledger.csv`), the rereference offsets, the fold map, and the build's own
`qc_flags_p100_structural.json` with its report, which must declare the `p100_structural`
policy — and the curated catalogue `deposition_notes.tsv` beside this file. It writes three
parquet files (pyarrow / snappy) and `_export_report.json`, which records the exclusion
policy, the row counts, the number of cells each fence removed, and the SHA-256 of the
three parquet files.

The parquet files are Git LFS objects. A clone without LFS leaves pointer files in their
place instead — 132 bytes for `deepshift_sequences.parquet` and `deepshift_structures.parquet`,
134 for `deepshift_shifts.parquet` (its recorded size is two digits longer) — and pyarrow
fails on them with an unhelpful error, so fetch them first:

```bash
git lfs install
git lfs pull
```

## deepshift_shifts.parquet — main dataset

One row per label residue (a residue in a deduplicated BMRB shift assignment).

**Shifts are rereferenced and the exclusion fence is applied:**

- The `*_shift` columns hold observed chemical shifts in ppm with the per-deposition
  rereference offset already **added**. An offset corrects a whole reference channel: the
  aliphatic ¹³C offset goes to `ca`/`cb` and every side-chain carbon, the ¹⁵N offset to
  every nitrogen column (`n` and the side-chain nitrogens), and C′ keeps its own. The proton columns carry the observed value as
  deposited. Cα and Cβ share one constant when both have enough shifts to fit it; where
  only one does, the build leaves the other uncorrected. That is 47 cells in 22 label sets
  (31 Cβ, 16 Cα; none in the held-out fold 6), listed in `cacb_unshared_offsets.tsv` with
  the offset the fitted nucleus received; the shipped models were trained on these values.
- Cells the `p100_structural` fence excludes — the fitted p100 accepted spans, the arginine
  Nε observed-value rule, and the structural cysteine/proline consistency filter,
  unioned — are **removed** (set to NaN), not masked. `_export_report.json` records how
  many cells that was (`excluded_cells`, 39,327). The fence is evaluated on the residues
  the training cache holds, those matched to a realized structure; a residue no structure
  realizes keeps its deposited values unchecked. Among them are 151 arginine Nε values
  outside the band (149 in label sets with no fold, 2 in BMRB 6236). On a realized residue
  the fence tests the cells the cache observes (`shift_mask`) whose ambiguity code is not
  4, 5, 6 or 9: a set-ambiguity cell never trains or scores, and a cell the cache does not
  observe is neither trained on nor scored, so either can keep an out-of-range value here.
- An amide proton deposited on a proline, which has none, is removed the same way
  (`proline_amide_blanked_cells`, 20 cells).

Columns:

- `label_set_id` — deduplicated shift-assignment (deposition) identity.
- `logical_entity_id` — logical polymer entity (join key to sequences).
- `label_residue_id` — unique residue-within-label-set identity.
- `bmrb_id`, `entity_id` — source BMRB entry and entity.
- `comp_index_id`, `comp_id`, `residue_id`, `residue_code` — residue position,
  three-letter comp id, numeric residue id, one-letter code.
- `fold` — CV fold 0–6 (Int64; null where the label set has no assignment).
  **Folds 1–5 = cross-validation, fold 6 = held-out test fold, fold 0 = excluded from
  training** (see *Folds* below: fold 0 is not separated from the other six).
- `qc_excluded_nuclei` — the nuclei the QC fence and the proline rule removed from this
  row, comma separated, empty when they removed none. A fenced cell and a cell the
  deposition never reported are both null in the `*_shift` columns, so without this you
  cannot tell them apart. Across the release it names 39,347 cells on 21,200 rows: the
  39,327 fenced cells and the 20 proline amide protons `_export_report.json` records.
- `curated_excluded_nuclei` — the nuclei blanked because `deposition_notes.tsv` records
  this deposition as carrying wrong values for them, comma separated, empty when none. It
  names only the cells this catalogue itself removed: 100 cells on 99 rows, across the
  `defect`, `recoverable` and `suspect` entries (`curated_blanked_cells`). A catalogued
  cell the QC fence had already removed is named in `qc_excluded_nuclei` alone, so no cell
  is named twice and a cell named in neither column was never deposited. The two
  fences answer different questions and so have their own columns: the p100 fence is a
  statistical range rule fitted from out-of-fold predictions, and cannot reach a cell it
  has no prediction for; the catalogue is a record of depositions read back from the BMRB
  archive and judged wrong by hand.
- one `*_shift` per nucleus — rereferenced, fenced observed ppm (NaN = unobserved or
  excluded; the two columns above say which, and empty in both means unobserved).
- one `*_ambiguity_code` per nucleus — the BMRB ambiguity code for that shift.

The nucleus vocabulary is the package's own (`deepshift.domain.SHIFT_COLUMNS`); the export
refuses a source whose header does not carry exactly it.

## Folds

Folds 1–6 are drawn over whole leakage components: no same-deposition, same-UniProt,
sequence-similarity, modeled-region or co-solved edge crosses two of them (`PROTOCOL.md`
§1). **Fold 0 is not separated from folds 1–6.** It holds the label sets excluded from
training by a direct rule rather than by the draw: the UCBShift-200 test proteins, and the
case-study proteins pinned there after the draw (all eight IDP8 proteins, tau and the drkN
SH3 domain). Fold 0 therefore shares proteins with the training folds by design; it is
the interval-calibration set, fitted on fold 0 minus the 461 label sets in the build's
`fold0_calibration_exclusions.json`: the 415 with a relative in a training fold and the
case-study records, which overlap them in part.

Three proteins appear in both fold 0 and fold 6, by UniProt accession or PDB entry:

| protein | fold 0 (BMRB) | fold 6 (BMRB) |
|---|---|---|
| UniProt Q9P212 (AlphaFold model) | 6635 | 6624 |
| PDB 2L29, chain A | 17127 | 17128, 19117, 19153 |
| PDB 7RNO | 51044 (chains A and B), 51045 (chain B) | 51046 (chain A) |

Those are five fold-0 label sets, and none of them is in the calibration exclusions. Two
of them put backbone cells into the interval fit: BMRB 6635 (1,163 cells) and the 2L29
chain-A label set of 17127 (787), 1,950 cells in all; the three 7RNO label sets contribute
none, and its chain B is a co-solved partner, not the protein fold 6 holds. So two label
sets that calibrate the prediction intervals share a protein with the fold that evaluates
them. No BMRB deposition is in both folds.

## deepshift_structures.parquet — BMRB ↔ structure pairs

Deduplicated successful structure builds. Absolute coordinate paths are not shipped; use
`accession` + `chain_id` + `source` to fetch the structure from its own archive.

- `bmrb_id`, `label_set_id` — the shift side.
- `source` — `af` (AlphaFold), `nmr` (any NMR method the PDB records: solution or
  solid-state; 102 of the 9,947 `nmr` rows are solid-state only), or `xtal` (crystal).
- `accession` — UniProt accession (AF) or PDB id (nmr/xtal).
- `chain_id` — chain used.
- `identity`, `query_coverage` — how well the coordinate's sequence matches the deposited
  one, as fractions, from the build's realization ledger.
- `resolution` — X-ray resolution in Angstroms. Null for AlphaFold rows and for almost
  every NMR row, which have none; ten `nmr` rows carry a value between 4.0 and
  26.5 Angstroms, depositions solved by a method that reports one and filed in the NMR
  lane.

Those last three are what the paper's dataset census reports as its identity, coverage and
resolution rows, so that table can be reproduced from this file alone.

## deepshift_sequences.parquet — per logical entity

- `logical_entity_id` — join key to shifts.
- `bmrb_id`, `entity_id` — source.
- `polymer_length` — length of the polymer.
- `alignment_sequence` — the aligned residue sequence.

## deposition_notes.tsv — known-bad archive records

Individual BMRB records whose deposited values are not chemical shifts, found by two
detectors and then read back from the BMRB archive in every case. The `status` column
separates them: `recoverable` where the correct values are present in the archive under
another tag, `defect` where the values are physically impossible for that nucleus and the
cause is identified, `suspect` where the cause is not established, and `anticorrelated`
described below.

The first detector is a **range scan**: any (deposition, nucleus) channel lying entirely
outside that nucleus's corpus range. The second catches what a range
scan cannot. A channel can be physically plausible value by value and still be wrong,
because only its ORDER is corrupt: the series runs backwards against the structure. Such a
channel is found by correlating the deposited values against predictions from independent
predictors — ProCS15, SPARTA+, UCBShift-X and SHIFTX2, with DeepShift excluded from the
judges so the test cannot be tuned to this package's own errors — and flagging channels where all
four judges correlate negatively over at least 20 observations. Fourteen channels qualify,
2,003 cells: twelve C-prime, one amide H and one alpha hydrogen. Each was read back from
`api.bmrb.io`, and thirteen of the fourteen match this package's stored values at r = 1.000
over the residues the archive and the structure share.
That shows the ingest is faithful up to the constant rereferencing offset; it is not by
itself proof that the deposited values are experimentally wrong. The case for that rests on
the same proteins' other nuclei predicting normally while C-prime does not, and on what a
reflection does. Mirroring each of the twelve C-prime channels about its own mean brings
six of them to 1.00–1.28 ppm against the four judges, inside the range the judges reach on
the 587 other fold-6 C-prime channels with at least 20 cells they predicted (median 1.07 ppm,
10th–90th percentile 0.90–1.33 ppm),
which is the signature of a series deposited inverted about a centre point; five more improve by
0.70–1.34 ppm, to 1.76–2.48 ppm (`benchmarks/reference/cprime_mirror_test.json`, written by
`benchmarks/cprime_mirror_test.py`). Those five are mirrored as well once a second defect is removed: two have the
C-prime list displaced by one residue (27191, 5752), one carries a 2–4 ppm referencing
offset (7300), and two have a few values aliased by one spectral width (6342, 52587). So
eleven of the twelve C-prime channels are mirror images of a physical series. The twelfth,
BMRB 5683, moves by 0.04 ppm when reflected and carries no residue-type or secondary-structure information, cause unidentified.
For eleven of the twelve (all but 5683, at +0.07) the deposited C-prime secondary shift
correlates negatively with the same deposition's own Cα secondary shift, against a median of
+0.58 over the 579 unflagged fold-6 C-prime channels with at least 20 residues carrying both,
a test that uses no predictor (`benchmarks/reference/cprime_judge_free.json`). The full evidence, per
channel, is in `benchmarks/reference/anticorrelated_channels.json`.

UCBShift reported the same class of defect in their own test set and removed those cells
for the same reason (Li, Bennett, Liu, Martin and Head-Gordon, *Chem. Sci.* 2020, **11**,
3180, Methods and ESI Fig. S1). Cells in an `anticorrelated` channel are excluded from
every fold-6 predictor comparison; they remain in the dataset tables, flagged here, because
the deposited values are what the archive contains. That is the one status this export
keeps: `defect`, `recoverable` and `suspect` cells are blanked and named in
`curated_excluded_nuclei`, because for those the archive's own record is wrong rather
than merely disputed.

**BMRB 7016 is recoverable and should be recovered.** `Val` and `Val_err` are transposed in
the archive: every amide proton is deposited as 0.02 ppm and every nitrogen as 0.20 — the
canonical uncertainties — while the real shifts sit in the error column. Swapping the two
tags restores the proton and nitrogen ranges the record should carry. The ingest reads the
record as deposited.

The others are not repairable from the record: sibling entries carrying carbon-magnitude
values under a proton atom name, entries carrying aliphatic values on aromatic or methyl
carbon names, and entries out of range with no cause established. They are listed so they
are not rediscovered as new.

## Licence and attribution

The tables in this directory are released under the Creative Commons Attribution 4.0
International licence (CC-BY-4.0); the code in this repository is MIT (`LICENSE`). They
are derived from three public archives, whose terms continue to apply to what came from
them:

- **BMRB** — every chemical shift, ambiguity code and sequence. BMRB data are freely
  accessible under its aims and policies (https://bmrb.io/bmrb/aims_and_policies.shtml),
  which ask that publications using them acknowledge BMRB. Its citation request
  (https://bmrb.io/bmrb/citation.shtml) is Hoch et al. (2023) "Biological Magnetic
  Resonance Data Bank", *Nucleic Acids Research* **51**:D368–D376,
  doi:10.1093/nar/gkac1050. Each entry also names its depositors and primary publication;
  `bmrb_id` leads to both.
- **RCSB PDB** — the PDB identifiers and chain names in `deepshift_structures.parquet`.
  PDB data are released under CC0 1.0; cite Berman et al. (2000) "The Protein Data Bank",
  *Nucleic Acids Research* **28**:235–242.
- **AlphaFold Protein Structure Database** — the UniProt accessions of the AlphaFold
  rows. Its models are CC-BY-4.0; cite Jumper et al. (2021) *Nature* **596**:583–589 and
  Varadi et al. (2024) *Nucleic Acids Research* **52**:D368–D375.

No coordinates are in these tables: `deepshift_structures.parquet` records `accession` and
`chain_id` so each structure can be fetched from its own archive. Two coordinate files do
ship elsewhere in the repository, both from the RCSB PDB under CC0: `figures/reference/1POH.pdb`
(the structure the figures draw and the prediction check runs on) and the three-residue
parser fixture `tests/fixtures/2trx_residues_1_3.pdb`.
