"""Where the fold-6 competitor benchmark's inputs live.

Every scientific input is resolved from the run root named by ``DEEPSHIFT_RUN``: the
fold-6 cache the scored cells come from, the fold map, edge tables, build logs and
UniProt ledger under ``data/``, the fitted exclusion fence under ``p100_fit/``, and the
five fold-6 member prediction dumps ``deepshift evaluate --save-npz`` wrote under
``final/intervals/``.

Two corpora are named separately because the run does not build them: the SIFTS
PDB-chain-to-UniProt table (``DEEPSHIFT_SIFTS_CSV``) and the RefDB shift files
(``DEEPSHIFT_REFDB_SHIFTS``). Competitor outputs and the benchmark's own tables live
under ``DEEPSHIFT_BENCH_OUT``, which defaults to ``benchmark/`` inside the run root.

A missing variable yields a visibly invalid sentinel, so import and ``--help`` work and
the first read fails naming the variable it needs.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

from deepshift.calibration import EVALUATION_FOLD, MEMBER_FOLDS
from deepshift.data.uniprot_snapshot import LEDGER_COLUMNS

#: The run root every scientific input is resolved from.
RUN_VAR = "DEEPSHIFT_RUN"
OUT_VAR = "DEEPSHIFT_BENCH_OUT"

#: Authorities inside the run root, by the names the build writes them under.
FOLD_MAP_NAME = "fold_assignments_labelset_identity90.json"
FLAGS_NAME = "p100_fit/qc_flags_p100_structural_oof64_all7.json"
MEMBER_NAME = "final/intervals/fold%d_member%d.npz"
UNIPROT_LEDGER_NAME = "uniprot_mapping_candidates.csv"

#: Label-set edge tables whose groups define same-UniProt and same-deposition linkage.
EDGE_FILES = ("labelset_edges_uniprot.csv", "labelset_edges_same_bmrb.csv")

#: The competitor arms this run scores, each the name of a directory of per-structure
#: outputs under ``DEEPSHIFT_BENCH_OUT``. Adding a predictor -- PLM-CS, when its
#: predictions are staged -- is a line here and a re-run of the census and the master
#: cells; nothing else names the roster. ``procs15_striph`` and ``sparta_orig`` are
#: diagnostic re-runs of an arm already listed, so they are not reported arms and are
#: not here.
BENCHMARK_ARMS = (
    "ucbshift_full_leaky", "ucbshift_full_sameprot", "ucbshift_full_selfexcl",
    "ucbshift_x", "ucbshift2_sameprot", "ucbshift2_selfexcl",
    "shiftx2_leaky", "shiftx2_sameprot", "shiftx2_selfexcl", "shiftx2_noshifty",
    "sparta", "napshift", "legolas", "procs15", "nmrgnn", "camshift", "mlff", "plmcs",
)

#: Arms scored on the master cells they emitted but never allowed to shrink the master
#: set, whatever their coverage. PLM-CS cannot predict a chain over 512 residues, so its
#: 8.9 % of missing fold-6 structures are exactly the longest proteins; letting it join at
#: 91 % coverage would remove those proteins from every predictor's common set. PLM-CS is
#: reported on what it has and does not exclude cells from any other arm.
MASTER_MARKED_ARMS = ("plmcs",)

#: Arms whose own training corpus overlaps this benchmark, and the roster that names it.
#: A structure of the same protein as any entry in the arm's training set is dropped from
#: that arm alone before scoring, so the arm is reported on the part of the benchmark it
#: did not train on. Every other arm keeps the whole master set, exactly as for an arm
#: below the coverage floor, and the count dropped is written into the report for the
#: caption.
#:
#: Only PLM-CS is listed. It is the one arm whose reported lead over DeepShift on any
#: nucleus came from this overlap: on solution-NMR amide 1H it led DeepShift over all
#: cells, and trails it once its training depositions are out (Table S12).
#: The other trained arms (NMRGNN, NapShift, LEGOLAS, MLFF) lead DeepShift nowhere at
#: either cohort, and rosters for two of them are not published in a machine-readable form.
ARM_TRAINING_ROSTERS = {"plmcs": "refdb_library"}

#: The frozen names of each corpus's files (``reference/``), read when the corpus itself is
#: absent -- RefDB is third-party and not redistributed.
ARM_CORPUS_LISTINGS = {"plmcs": "plmcs_corpus_listing.json"}

#: The fold-6 label sets of the same protein as some entry of that corpus, frozen under
#: ``benchmarks/reference/``: same BMRB, a shared UniProt accession, or >=95 % identity
#: over >=90 % of the shorter sequence. A leak is the same protein, not only the same
#: deposition, so this file -- not the BMRB listing alone -- decides what is withheld.
ARM_SAME_PROTEIN_ROSTERS = {"plmcs": "plmcs_same_protein.json"}

#: The DeepShift arm's name in every scoring report; its cells are the universe.
DEEPSHIFT_ARM = "deepshift"


def arm_exclusion_tier(arm: str) -> int | None:
    """The self-exclusion level an arm is verified at, read from the name it is staged
    under: ``None`` for ``*_leaky`` (scored with no exclusion enforced, and reported as
    such), 3 for ``*_selfexcl`` (same-protein depositions plus >=90% identity blocked),
    and 2 -- the reported same-protein level -- for every other arm. An arm that
    retrieves nothing has nothing to verify at any level.
    """
    if arm.endswith("_leaky"):
        return None
    if arm.endswith("_selfexcl"):
        return 3
    return 2


def _root(var: str) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else Path(f"__{var}_REQUIRED__")


@dataclass(frozen=True)
class Build:
    """One run root, and the fold-6 evaluation packet inside it."""

    run: Path
    out_root: Path

    @property
    def data(self) -> Path: return self.run / "data"
    @property
    def cache(self) -> Path: return self.run / f"work/unified_cache/fold_{EVALUATION_FOLD}"
    @property
    def structural(self) -> Path: return self.cache / "structural"
    @property
    def atom(self) -> Path: return self.cache / "atom"
    @property
    def fold_map(self) -> Path: return self.data / FOLD_MAP_NAME
    @property
    def flags(self) -> Path: return self.run / FLAGS_NAME
    @property
    def uniprot_ledger(self) -> Path: return self.data / UNIPROT_LEDGER_NAME

    @property
    def members(self) -> tuple:
        """The five fold-6 member dumps, in model-fold order."""
        return tuple(self.run / (MEMBER_NAME % (EVALUATION_FOLD, fold))
                     for fold in MEMBER_FOLDS)

    @property
    def pdb_dirs(self) -> dict:
        return {"af": self.data / "alphafold",
                "nmr": self.data / "pdbs",
                "xtal": self.data / "crystal_pdbs"}

    @property
    def sifts(self) -> Path:
        return _root("DEEPSHIFT_SIFTS_CSV")

    @property
    def refdb_shifts(self) -> Path:
        return _root("DEEPSHIFT_REFDB_SHIFTS")

    @property
    def refdb_library(self) -> Path:
        """The released RefDB snapshot as SHIFTX2 ships it, one ``bmr<id>.cs`` per entry.

        This is the corpus PLM-CS states it trained on, and the transfer library both
        UCBShift versions search, so it is the roster :data:`ARM_TRAINING_ROSTERS` reads.
        It is the released snapshot, not our reprocessed copy under
        ``DEEPSHIFT_REFDB_SHIFTS``, which is a subset of it.
        """
        return _root("DEEPSHIFT_REFDB_LIBRARY")


def current() -> Build:
    run = _root(RUN_VAR)
    out_root = Path(os.environ.get(OUT_VAR) or run / "benchmark")
    return Build(run=run, out_root=out_root)


def missing(build: Build, *, need_pred: bool = False) -> list:
    """Everything the benchmark will read, checked before it reads any of it.

    ``need_pred`` adds the fold-6 member dumps, which the scorers need and the
    competitor driver does not.
    """
    gaps = []
    if need_pred:
        gaps += [f"member={path}" for path in build.members if not path.exists()]
    for name in ("run", "cache", "structural", "atom", "data", "fold_map", "flags",
                 "uniprot_ledger"):
        path = getattr(build, name)
        if not path.exists():
            gaps.append(f"{name}={path}")
    for src, directory in build.pdb_dirs.items():
        if not directory.is_dir():
            gaps.append(f"pdb_dirs[{src}]={directory}")
    for edge in EDGE_FILES:
        if not (build.data / edge).exists():
            gaps.append(str(build.data / edge))
    for src in ("af", "nmr", "xtal"):
        path = build.data / f"structure_build_log_{src}.csv"
        if not path.exists():
            gaps.append(str(path))
            continue
        with open(path) as fh:
            header = csv.DictReader(fh).fieldnames or []
        gaps += [f"{path}: no column {c}" for c in ("materialization_id", "pdb_path", "chain_id")
                 if c not in header]
    for name in ("id_maps.json", "global_to_struct_resid.json", "shift_mask.npy"):
        if not (build.structural / name).exists():
            gaps.append(str(build.structural / name))
    if not (build.atom / "atom_count.npy").exists():
        gaps.append(str(build.atom / "atom_count.npy"))
    if build.uniprot_ledger.exists():
        with open(build.uniprot_ledger) as fh:
            header = csv.DictReader(fh).fieldnames or []
        gaps += [f"{build.uniprot_ledger}: no column {c}" for c in LEDGER_COLUMNS
                 if c not in header]
    if not build.sifts.exists():
        gaps.append(f"sifts={build.sifts}")
    return gaps
