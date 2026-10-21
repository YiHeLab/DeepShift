# DeepShift on Colab

`DeepShift_predict.ipynb` predicts per-residue chemical shifts for one chain of a PDB
structure: upload a file or type a PDB ID, pick the chain, get the shifts as CSV.

The notebook is a wrapper around the installed package. It resolves the checkpoints, the
structure and the chain, calls
`deepshift.prediction.predict_pdb` once, and writes what that call returns:

* `<stem>_<chain>_deepshift.csv` — one row per (residue, nucleus) the residue type has,
  with `pred_ppm`; when all five members and the interval calibration were loaded, the
  backbone nuclei also carry the across-model spread `ensemble_sd_ppm` and the 90 %
  interval half-width `ci90_halfwidth_ppm`;
* `<stem>_<chain>_deepshift_backbone.csv` — the six backbone nuclei, one row per residue.

Which chains enter the atom cloud, which (residue, nucleus) cells exist, how members are
averaged in ppm and when an interval is defined are the package's rules; the notebook
reshapes what the call returns. `python -m deepshift.pipeline predict` makes the same
call.

## The checkpoints

The five fold models are `models/deepshift_fold{1..5}.pt`. Section 2 predicts with every
`.pt` file in `CHECKPOINT_DIR` — the repo's `models/` when it holds checkpoints and a
download directory otherwise — and downloads the five members when that folder holds none.
`deepshift.model.load_checkpoint` reads each file's nuclei, atom vocabulary, atom-cloud
recipe and normalization off the file, and `predict_pdb` refuses an ensemble whose members
disagree on any of them, so a damaged file or a checkpoint from another model stops the run
instead of producing numbers. Fewer than five checkpoints is allowed: the notebook says so,
predicts with the members it has, and reports no spread and no interval.

`interval_calibration.json` ships beside the five members and is what turns the spread into
a 90 % interval. The notebook looks for it in
`CHECKPOINT_DIR`, downloads it with the members, and says so when it has none: without it
`predict_pdb` returns no interval table, and with it and fewer than five members it
refuses rather than widening a half-width fitted on five.

If the download does not resolve, put the files somewhere the session can read and point
`DEEPSHIFT_CKPT_DIR` at that folder, or name the artifact with `DEEPSHIFT_CALIBRATION`.

## Installing the package in a Colab session

```bash
pip install -q biopython
pip install -q --no-deps <package spec>
```

* `--no-deps` keeps Colab's own CUDA build of torch and its numpy, pandas, scipy and
  pyarrow; `pyproject.toml` pins exact versions of all five, and honouring those pins in a
  Colab session replaces a working GPU torch with a multi-gigabyte download.
* `biopython` is installed on its own because `--no-deps` skips it, and
  `deepshift.data.alignment` imports `Bio.Align` at module scope.
* `GIT_LFS_SKIP_SMUDGE=1` is set for the install so pip does not pull the LFS training
  artifacts that inference never reads.

## Running the cells outside Colab

Every input has an environment-variable override, so the extracted cells run headless:

```bash
PYTHONPATH=src \
DEEPSHIFT_CKPT_DIR=models \
DEEPSHIFT_PDB=figures/reference/1POH.pdb \
DEEPSHIFT_CHAIN=A \
DEEPSHIFT_OUT_DIR=deepshift_out \
python -c "import json;exec(''.join(''.join(c['source'])+chr(10)*2 for c in json.load(open('colab/DeepShift_predict.ipynb'))['cells'] if c['cell_type']=='code'))"
```

| variable | meaning |
|---|---|
| `DEEPSHIFT_CKPT_DIR` | folder holding `deepshift_fold{1..5}.pt` (default: the repo's `models/` if present, else `deepshift_checkpoints`) |
| `DEEPSHIFT_CKPT_URL` | base URL to download them from; empty means local files only |
| `DEEPSHIFT_CALIBRATION` | the interval calibration artifact (default: `interval_calibration.json` in `DEEPSHIFT_CKPT_DIR`) |
| `DEEPSHIFT_PDB` | local `.pdb` path, or a 4-character PDB ID to fetch from RCSB |
| `DEEPSHIFT_CHAIN` | chain to predict |
| `DEEPSHIFT_OUT_DIR` | where the CSVs are written (default: the working directory) |
| `DEEPSHIFT_SPEC` | pip spec for the package |

`tests/test_colab_notebook.py` runs exactly that extraction on
`figures/reference/1POH.pdb`, with a checkpoint it trains for the purpose, and asserts the
CSV it writes carries the same ppm, cell for cell, as `deepshift predict` on that
checkpoint.

## Notes

* The notebook keeps no hidden state: run it top to bottom in a fresh runtime.
* `display` is an IPython builtin, not a Python one; section 1 falls back to `print` so a
  headless run reaches the end.
* A chain id is required, and it has to be one the structure cell lists: that cell
  reports every chain in the file with its amino-acid residue count.
* The upload cell opens Colab's file picker as soon as it runs; an uploaded file wins over
  the `pdb` field, and `DEEPSHIFT_PDB` wins over both. The input is PDB format.
* The nucleus set comes from the checkpoint, so a differently trained ensemble needs no
  edit here; the (residue, nucleus) presence rule lives in `src/deepshift/prediction.py`.

## Usage telemetry

The notebook reports one anonymous record per prediction (time, version, RCSB id or
"upload", chain, residue count, run time, success) to a receiver the maintainers run; it
never sends coordinates, file names, accounts or addresses unless the user sets
`SHARE_STRUCTURE`. The record is printed before it is sent, `TELEMETRY = False` sends
nothing, and a failed report never interrupts a prediction. The receiver is
`telemetry/Code.gs`, a Google Apps Script web app bound to a Google Sheet; deploy it from
the lab account (instructions at the top of the file) and put the `/exec` URL in the
notebook's `TELEMETRY_URL` (or the `DEEPSHIFT_TELEMETRY_URL` environment variable).
The URL is empty until the receiver is deployed, and an empty URL sends nothing.
