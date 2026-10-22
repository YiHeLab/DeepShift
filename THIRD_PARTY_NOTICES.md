# Third-party notices

`LICENSE` (MIT) covers the code authored for DeepShift. The dataset tables under `data/`
are released under CC-BY-4.0, as `data/README.md` states. Neither covers the third-party
components listed here. Each entry says what the component is, whether this
repository *redistributes* it or merely *fetches* it at build time, and under what terms.

| component | how it reaches you | license |
|---|---|---|
| SHIFTX 1.1 | downloaded by `src/deepshift/data/external.py` | none stated by the publisher — see below |
| MMseqs2 | downloaded by `src/deepshift/data/external.py` | MIT |
| RefDB | supplied by you to three reproduction paths; not redistributed | see below |
| CSpred / UCBShift test targets | supplied by you as JSON, not redistributed | UC Regents, academic / not-for-profit |
| BMRB, RCSB PDB, AlphaFold DB | fetched by the acquire phase; derived tables and two PDB files ship | see below |
| UCBShift 1.0 and 2.0, SHIFTX2 | modified as diffs in `experiments/competitors/patches/`; upstream source not included | UC Regents, educational / research / not-for-profit (UCBShift); none stated (SHIFTX2) |
| competitor predictors and X-EISD | run by `experiments/`; installed by you, not redistributed | per tool, in `experiments/competitors/README.md` and the other `experiments/*/README.md` |

---

## SHIFTX 1.1 — downloaded, not redistributed

**Not redistributed.** `src/deepshift/data/external.py::ensure_shiftx` downloads the authors' source
distribution from `https://shiftx.wishartlab.com/download/shiftx.tar.gz` at build time
(SHA-256 pinned: `a08bc301b119154675eb05d9c5c69b7f582e8c2fa0cc81c5040a6ac013ac1965`),
compiles it unmodified, and `src/deepshift/data/reref.py` runs the resulting binary to compute
per-deposition reference offsets. The server's TLS certificate does not validate, so the
pinned hash is what authenticates the artifact.
`SHIFTX_SOURCE` points the build at an already-downloaded source tree; `SHIFTX_BIN` at an
already-compiled binary.

Upstream is SHIFTX, from David Wishart's group at the University of Alberta:

> Stephen Neal, Alex M. Nip, Haiyan Zhang, David S. Wishart (2003). "Rapid and accurate
> calculation of protein 1H, 13C and 15N chemical shifts." *Journal of Biomolecular NMR*
> **26**:215–240.

The distribution's own `FEATURES` file records that it derives from the `Cys_oct2003`
version with the hydrogen-reading path removed so that NMR structures are treated like
X-ray ones; that modification is upstream's, not ours.

Why it is not vendored: the tree states no license terms (`README` says to contact the
authors), and `Cspline.c` plus part of `Hydrogens.c` carry Numerical Recipes copyright
notices whose license does not permit source redistribution. Downloading from the authors'
own server at build time redistributes nothing. The binary built from the downloaded
tarball reproduces the distribution's own output on its shipped `2TRX.pdb` example
byte-for-byte.

## MMseqs2 — downloaded, MIT

**Not redistributed.** `src/deepshift/data/external.py::ensure_mmseqs` downloads the static Linux build
from `https://mmseqs.com/archive/4fe20e6bb0aa219a35f6190322a91ab19cb6898d/` into
`<root>/tools`, verifying pinned SHA-256s for both archive and binary.

MIT License, Copyright © 2024 The MMseqs2 Development Team. MIT-compatible; the license
text ships with the download as `mmseqs/LICENSE.md`.

> Steinegger, M. & Söding, J. (2017). "MMseqs2 enables sensitive protein sequence searching
> for the analysis of massive data sets." *Nature Biotechnology* **35**:1026–1028.

## RefDB — supplied by you, not redistributed

**Not redistributed, and not read by the dataset build.** The rereferencing method in
`src/deepshift/data/reref.py` and `src/deepshift/data/reref_utils.py` follows the procedure this paper describes, and
the code comments cite its pages where a rule comes from it. Three reproduction paths read
RefDB files you supply: the RefDB agreement table reads RefDB's own correction reports
(`$DEEPSHIFT_REFDB_CORRECTIONS`), the competitor benchmark reads the RefDB shift tree of a
CSpred checkout (`$DEEPSHIFT_REFDB_SHIFTS`), and the literature table counts the RefDB
chains of the same checkout (`$DEEPSHIFT_REFDB_CHAINS`).

> Zhang, H., Neal, S. & Wishart, D. S. (2003). "RefDB: a database of uniformly referenced
> protein chemical shifts." *Journal of Biomolecular NMR* **25**:173–195.

## CSpred / UCBShift test targets — user-supplied, academic-use only

**Not redistributed.** The `folds` stage needs the UCBShift-200 test targets (the JSON
conversion of CSpred's `test_200.pkl`, `PROTOCOL.md` gives the command) as the
exclusion authority and fails closed if `$SHIFT_UCB_TEST_TARGETS` is unset. CSpred is
Copyright © 2020 The Regents of the University of California, licensed for educational,
research and not-for-profit use only — which is why it stays an input you provide rather
than a file in this repository.

`src/deepshift/data/ucbshift200_excluded_bmrbs.json` is a list of BMRB accession numbers
derived from that benchmark. It is a set of database identifiers, not CSpred code or data.

> Li, J., Bennett, K. C., Liu, Y., Martin, M. V. & Head-Gordon, T. (2020). "Accurate
> prediction of chemical shifts for aqueous protein structure on 'Real World' data."
> *Chemical Science* **11**:3180–3191.

## Patches to UCBShift, UCBShift 2.0 and SHIFTX2 — diffs only

`experiments/competitors/patches/` holds the changes made to three competitors to block
retrieval of the test protein's own depositions: `ucbshift.patch` (THGLab/CSpred commit
`20cee31`), `ucbshift2.patch` (CSpred `v2.0`, commit `844e265`) and
`shiftx2_shifty3.patch` (SHIFTX2 1.13, `shifty3/shifty3.py`). They carry lines of the
upstream files as context; no other upstream source or weights are included. SHIFTX2
states no licence terms. CSpred's terms, which apply to the two CSpred patches:

> Copyright ©2020 The Regents of the University of California (Regents). All Rights
> Reserved. Permission to use, copy, modify, and distribute this software and its
> documentation for educational, research, and not-for-profit purposes, without fee and
> without a signed licensing agreement, is hereby granted, provided that the above
> copyright notice, this paragraph and the following paragraphs appear in all copies,
> modifications, and distributions. Contact The Office of Technology Licensing, UC
> Berkeley, 2150 Shattuck Avenue, Suite 408, Berkeley, CA 94704-1362, otl@berkeley.edu.
>
> IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
> INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF
> THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
> OF SUCH DAMAGE.
>
> REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND
> ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS
> NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

## Programs run by `experiments/` — installed by you, not redistributed

The competitor predictors (SPARTA+, NapShift, LEGOLAS, ProCS15, NMRGNN, PLUMED with
CamShift, PLM-CS, the MACE-based MLFF predictor, UCBShift, UCBShift 2.0, SHIFTX2), REDUCE,
OpenMM and PDBFixer, and X-EISD are run as external programs. Each `experiments/*/README.md`
names the version used, where to obtain it and its licence.

## Scientific databases

The acquire phase reads three public archives. The tables in `data/` are derived from them
and released under CC-BY-4.0; the archives' own terms still apply to what came from them.

- **BMRB** (`bmrb.io`) — the chemical shift depositions the labels come from. Freely
  accessible under BMRB's aims and policies, which ask that publications acknowledge it:
  Hoch et al. (2023) *Nucleic Acids Research* **51**:D368–D376 (doi:10.1093/nar/gkac1050).
- **RCSB PDB** (`rcsb.org`) — experimental structures. PDB archive data are released under
  CC0 1.0 (wwPDB usage policy); cite Berman et al. (2000) *Nucleic Acids Research*
  **28**:235–242.
- **AlphaFold Protein Structure Database** (`alphafold.ebi.ac.uk`) — predicted structures,
  licensed CC-BY 4.0. Attribution: Jumper et al. (2021) *Nature* **596**:583–589 and
  Varadi et al. (2024) *Nucleic Acids Research* **52**:D368–D375.

The shipped tables in `data/` contain BMRB-derived chemical shifts and sequences plus
UniProt and PDB accession codes, and no coordinates: `deepshift_structures.parquet` records
`accession` + `chain_id` so you can fetch each structure yourself. Two RCSB PDB coordinate
files do ship, under CC0: `figures/reference/1POH.pdb`, the entry the figures draw and the
release's prediction check runs on, and `tests/fixtures/2trx_residues_1_3.pdb`, a
three-residue excerpt of entry 2TRX used as a parser fixture.
