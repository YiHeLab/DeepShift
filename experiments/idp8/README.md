# IDP8 ensembles (Figure S11 panel a, Table S28)

This directory repeats the protocol of Nagy et al. (*Appl. Spectrosc.* 78, 897, 2024,
doi:10.1177/00037028241239977) with DeepShift in place of SPARTA+. They back-calculate
C-alpha, C-beta and C' shifts for every conformer of each deposited IDP8 ensemble, average
over conformers with uniform weights, and report the root-mean-square deviation (RMSD) from
the measured BMRB shifts (their Table III). Here the same ensembles, the same BMRB
references, the same statistic and no re-referencing are used, and only the predictor
changes. Nine of the 14 IDP8 ensembles can be scored, because the measured shifts of the
other five (measles virus nucleoprotein tail MEVN-A/B/C and RS-repeat peptide RSP8-B/C) were
supplied to Nagy et al. privately and are not in the BMRB.

| Script | Does | Writes |
|---|---|---|
| `fetch.py` | downloads the nine PED ensembles and six BMRB entries | `ensembles/*.pdb`, `shifts/*.str`, `manifest.json` |
| `run_inference.py` | DeepShift prediction for every conformer, five-member mean | `<label>.parquet` per ensemble |
| `score.py` | residue alignment by sequence, ensemble average, RMSD against BMRB | one JSON row per ensemble |

The SPARTA+ column of Figure S11 and Table S28 is not recomputed. It is Nagy et al.'s
published Table III, transcribed into the `PUBLISHED` table of `score.py` (None where the
paper reports no value); `si_build/build_idp8_roster.py` checks its own transcription of the
same table against it. SPARTA+ is not run by anything in this directory.

## Inputs

* **Ensembles**, from the Protein Ensemble Database (`https://deposition.proteinensemble.org`):
  PED00024 (asyn-A), PED00160 (sic1-A), PED00192 (tk18-A), PED00228 (cbpn-B), PED00235
  (cbpn-C), PED00229 (p53t-B), PED00236 (p53t-C), PED00230 (actr-B), PED00237 (actr-C), each
  entry's ensemble `e001`. Groups B and C were refined against C-alpha chemical shifts by
  their authors, so the C-alpha comparison on them is partly circular; C-beta and C' were
  held out.
* **Measured shifts**, from the BMRB (`https://bmrb.io`): 19257 (alpha-synuclein, an in-cell
  measurement), 16657 (Sic1), 19253 (tau K18), 15397 (ACTR), 16363 (CBP NCBD) and 17760
  (p53 transactivation domain). The text of Nagy et al. gives 17660 for p53, which is an
  unrelated protein (LmrA-NBD); PED00229 and PED00236 cross-reference 17760.
* **Checkpoints**: the five files in `models/` (Git LFS) and an installed `deepshift`
  (`python -m pip install .` from the repository root). `run_inference.py` is a GPU step
  (`--device cuda`, the default; fp16 autocast, as in the benchmark). `fetch.py` needs
  network access; `score.py` runs on a CPU and needs `pynmrstar` and Biopython, both pinned
  by the release.

All nine proteins belong to the case-study hold-out (fold 0) or to no fold, so no member
trained on them, and every conformer is predicted by all five members.

## Commands

```bash
D=<idp8 data directory>
W=<work directory>
python experiments/idp8/fetch.py --out $D
python experiments/idp8/run_inference.py --data $D --out $W/predictions     # resumable
mkdir -p $W/protocol
python experiments/idp8/score.py --data $D --predictions $W/predictions \
    --out $W/protocol/scores_audited.json
cp experiments/idp8/score.py $W/protocol/
export DEEPSHIFT_IDP8_ROOT=$W/protocol
```

## What the producers read

`DEEPSHIFT_IDP8_ROOT` is the directory holding both files.

| Producer | Reads | Written by |
|---|---|---|
| `figures/fig_downstream_applications.py` (Figure S11 panel a) | `scores_audited.json` | `score.py` |
| `si_build/build_idp8_roster.py` (Table S28) | `scores_audited.json` and `score.py` (its `PUBLISHED` table) | `score.py` |

Each row of `scores_audited.json` carries, per nucleus, the number of matched residues, the
RMSD, the mean absolute error, the mean signed error, the published RMSD and their
difference. The figure and the table use the RMSD and the published value only.
