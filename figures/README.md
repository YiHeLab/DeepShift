# Figures

One script per figure or table of the paper. Each reads finished results from disk and
writes `<stem>.pdf` and `<stem>.png` (or `<stem>.tex` for a table) to `figures/out/`, or
to `$DEEPSHIFT_FIGURES_OUT` when that is set.

- `style.py`: rcParams, palette and mark helpers shared by the figures.
- `frozen.py`: where the results live (run root, directories under it, environment
  variables) and the helper that writes a figure.
- `reference/`: small inputs kept in the repository (the Figure 1 structure, the arginine
  N-epsilon fence, the ray-traced renders).

All statistics come from `deepshift.evaluate`; the scripts plot what it returns.

## Which script makes what

```
SCRIPT                             OUTPUT                            PAPER
build_paper_figures.py             fig_01_architecture               Figure 1
                                   fig_02_accuracy                   Figure 2 (upper)
                                   fig_07_scaling                    Figure S7
                                   fig_08_ramachandran               Figure S8
build_graphical_abstract.py        graphical_abstract                Graphical abstract
fig_accuracy_by_nucleus_all.py     fig_accuracy_by_nucleus_all       Figure S3
fig_backbone_benchmark.py          fig_03_benchmark                  Figure 2 (lower), Table 1
fig_census_merged.py               fig_census_merged                 Figure S1
fig_downstream_applications.py     fig_downstream_applications       Figure S11
fig_exp_source_control.py          fig_exp_source_control            Figure S6
fig_homology_training_identity.py  fig_homology_training_identity    Figure S2
fig_plddt_source.py                fig_plddt_source_advantage        Figure S5
fig_selflookup_transfer.py         fig_04_selflookup                 Figure S4
fig_sidechain_benchmark.py         fig_05_sidechain                  Figure 3 (lower)
fig_source_matched.py              fig_06_source                     Figure 3 (upper)
fig_structure_response.py          fig_04_structure                  Figure 4
                                   fig_si_decoy_r_vs_rmsd            Figure S9
fig_uncertainty_calibration.py     fig_uncertainty_calibration_fold6 Figure S10
fig_xeisd_composition.py           fig_xeisd_composition             Figure S12
fig_xeisd_mixed_selection.py       fig_05_xeisd_mixed                Figure 5
si_tables_literature.py            si_tables_literature (.tex)       Table S16
render_structures.py               reference/renders/                ray-traced panels of Figure 1
```

`fig_backbone_benchmark.py` also writes `tab_accuracy_common_rmse.tex`, the body of
Table 1. `fig_structure_source.py` draws nothing; it is the matched-cohort module that
`fig_plddt_source.py` imports. Several scripts also print the numbers they computed.

## Run

```bash
cd figures
DEEPSHIFT_RUN=/path/to/the/run python fig_accuracy_by_nucleus_all.py
```

No `PYTHONPATH` is needed; `frozen.py` puts `../src` on the path. A script that draws from
a scoring report takes the report as an argument (`--help` lists them). An unset
variable resolves to a placeholder path such as `__DEEPSHIFT_RUN_REQUIRED__`, so a script
still imports and the first read fails naming the variable it needs.

## Environment variables

| Variable | What it holds |
|---|---|
| `DEEPSHIFT_RUN` | the finished run: `final/` members and their prediction dumps, `data/` |
| `DEEPSHIFT_FIGURES_OUT` | output directory; `figures/out` when unset |
| `DEEPSHIFT_HOMOLOGY_BENCHMARK_ROOT` | competitor benchmark root: fold-6 task table, ProCS15 outputs, fenced master cells (`fig_homology_training_identity`) |
| `DEEPSHIFT_HOMOLOGY_CELLS`, `DEEPSHIFT_LEAKAGE_DOSE_ROOT` | per-cell dump and homology axis (`fig_homology_training_identity`) |
| `DEEPSHIFT_RING_CELLS`, `DEEPSHIFT_CYS_CELLS`, `DEEPSHIFT_DECOY_POOLED` | ring-current cells, cysteine cells and pooled decoy scores (`fig_structure_response`) |
| `DEEPSHIFT_SELFLOOKUP_CELLS`, `DEEPSHIFT_UCBSHIFT_LEAKAGE`, `DEEPSHIFT_BENCH_OUT` | panels (b) and (a) of `fig_selflookup_transfer`, and the competitor outputs behind (b) |
| `DEEPSHIFT_RAMACHANDRAN_OOF`, `DEEPSHIFT_SCALING_LADDER` | out-of-fold torsion cells (Figure S8) and the data-fraction ladder (Figure S7) |
| `DEEPSHIFT_EXP_SOURCE_CONTROL` | crystal+NMR-only retraining control on fold 6 (Figure S6) |
| `DEEPSHIFT_XEISD_PROTOCOL` | X-EISD protocol run directory (Figures 5 and S12) |
| `DEEPSHIFT_IDP8_ROOT`, `DEEPSHIFT_TAU_BENCHMARK_ROOT` | the two downstream applications (Figure S11) |
| `DEEPSHIFT_PAPER_PATH`, `DEEPSHIFT_REFDB_CHAINS` | competitor articles and RefDB chains read by `si_tables_literature.py` |
| `PYMOL` | PyMOL executable for `render_structures.py` |

`REPRODUCTION.md` at the repository root says which archive supplies each of these.

Set `SOURCE_DATE_EPOCH` before rendering to keep the PDF `/CreationDate` fixed.

## Notes

- Prediction dumps must store string columns as Unicode, not pickled objects;
  `evaluate.read_predictions` refuses anything else.
- Never average ppm across nuclei: facet by isotope, one ppm axis per panel.
- Score `resolved_ppm` against `effective_mask`, not the raw `true_ppm`/`mask`.
- Uncertainty is a bootstrap over label sets, not residues.
- `build_paper_figures.py` has its own `house_style()`; the other scripts use `style.py`.
