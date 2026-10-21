"""One-command construction of DeepShift's entity-grained training cache.

Entity stage graph:

    fetch -> assembly -> uniprot_snapshot -> discover -> materialize -> cluster -> folds
          -> partner_suppression -> split -> reref -> cache -> atom_census -> atoms -> flags

``--uniprot-mode`` accepts only ``frozen``: the BMRB-to-UniProt mapping the fold stage
reads (``deepshift.data.uniprot.load_merged_mapping``) is a static merge of two frozen
JSON files under ``<data_dir>/alphafold``, called unconditionally and carrying no network
or BLAST logic. Every stage checks that its declared inputs exist before it runs and its
declared outputs exist after; ``--from-stage``/``--to-stage`` run part of the graph.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..quality_control import FINAL_QC_POLICY
from .paths import BLASTP_BIN, UCB_TEST_TARGETS


def _path_is_nonempty(path: Path) -> bool:
    """True for a nonempty regular file or a directory containing a file."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(child.is_file() for child in path.rglob("*"))
    return False


def _point_at(source, destination: Path) -> None:
    """Symlink ``destination`` -> ``source``, always re-pointing a link that resolves
    elsewhere -- a stale link to a different method/resolution cache must not silently
    survive. Same rule as ``_materialize_exclusion``."""
    source = Path(source).resolve()
    if destination.is_symlink() or destination.exists():
        if destination.exists() and destination.resolve() == source:
            return
        destination.unlink()
    os.symlink(source, destination)


def _copy_input(source, destination: Path) -> None:
    """Copy a declared input into the build."""
    source = Path(source)
    if destination.exists() and destination.samefile(source):
        return
    shutil.copyfile(source, destination)


STRUCTURE_ARCHIVE_DIRS = ("pdbs", "crystal_pdbs", "alphafold")


def link_structure_archive(archive: Path, data_dir: Path) -> dict:
    """Link every coordinate file of the frozen structure archive into the build's own
    ``pdbs``, ``crystal_pdbs`` and ``alphafold`` directories, where discovery and
    materialization look for them. Returns the file count per directory.

    A link already pointing at the archive's file is left alone; one pointing elsewhere
    is re-pointed; a regular file in the way is an error, since the build must read the
    archive and nothing else."""
    archive = Path(archive).resolve()
    counts = {}
    for name in STRUCTURE_ARCHIVE_DIRS:
        source_dir = archive / name
        if not source_dir.is_dir():
            raise RuntimeError(f"structure archive has no {name}/ directory: {archive}")
        target_dir = data_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        counts[name] = 0
        for source in sorted(source_dir.iterdir()):
            if not source.is_file():
                continue
            link = target_dir / source.name
            if link.is_symlink():
                if link.resolve() == source:
                    counts[name] += 1
                    continue
                link.unlink()
            elif link.exists():
                raise RuntimeError(f"{link} is a file, not a link into the archive")
            link.symlink_to(source)
            counts[name] += 1
    return counts


def uniprot_map_for_discovery(ledger: Path, universe) -> dict:
    """BMRB -> [accessions] from the frozen candidate ledger, with an explicit empty list
    for every deposition in ``universe`` the snapshot holds no candidate for.

    Discovery counts a deposition as queried for AlphaFold when it is a key of this map:
    every deposition in the build was looked up in the frozen snapshot, so every one is
    a key, and an empty list is the recorded answer "no mapping"."""
    by_bmrb = {}
    with open(ledger, newline="") as fh:
        for row in csv.DictReader(fh):
            b, acc = str(row["bmrb_id"]), row["accession"]
            pri = 0 if row.get("mapping_source") == "baseline" else 1
            by_bmrb.setdefault(b, {}).setdefault(acc, pri)
    af_map = {b: [a for a, _ in sorted(accs.items(), key=lambda kv: (kv[1], kv[0]))]
              for b, accs in by_bmrb.items()}
    for b in universe:
        af_map.setdefault(str(b), [])
    return af_map


def _materialize_exclusion(packaged: Path, dest: Path, what: str) -> None:
    """Refresh a build-local exclusion list from its non-empty package authority."""
    if not packaged.is_file():
        raise RuntimeError(f"packaged {what} exclusion is missing: {packaged}")
    payload = json.loads(packaged.read_text())
    bmrbs = payload.get("bmrbs", []) if isinstance(payload, dict) else payload
    if not bmrbs:
        raise RuntimeError(f"packaged {what} exclusion is empty: {packaged}")
    # Always refresh: a present build-local file may be stale, truncated, or empty.
    shutil.copyfile(packaged, dest)


# Entity stage graph. Names are Builder method names.
STAGES = (
    "fetch", "assembly", "uniprot_snapshot", "discover", "materialize", "cluster",
    "folds", "partner_suppression", "split", "reref", "cache", "atom_census", "atoms",
    "flags",
)

AMBIENT_BUILD_OVERRIDES = (
    "MMSEQS_BIN", "BLASTP_BIN", "SHIFTX_BIN", "SHIFTX_SOURCE",
    "AF_BUILDLOG", "SHIFT_DATA_ARCHIVE", "SHIFT_SPLIT_SEED", "SHIFT_HOLDOUT_FRAC",
)

@dataclass(frozen=True)
class BuildPaths:
    root: Path

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def folds_dir(self) -> Path:
        return self.work / "unified_folds"

    @property
    def cache(self) -> Path:
        return self.work / "unified_cache"

    @property
    def tools(self) -> Path:
        return self.root / "tools"


class Builder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.paths = BuildPaths(Path(args.root).resolve())
        self.build_id = self.paths.root.name
        self.paths.data.mkdir(parents=True, exist_ok=True)
        self.paths.work.mkdir(parents=True, exist_ok=True)

        self._mmseqs_bin = getattr(args, "mmseqs_bin", None)
        self.blastp_bin = getattr(args, "blastp_bin", None) or BLASTP_BIN

        self.env = os.environ.copy()
        for name in AMBIENT_BUILD_OVERRIDES:
            self.env.pop(name, None)
        # MMseqs/BLAST binaries pass to stages as explicit CLI args (--mmseqs_bin/
        # --blastp_bin), never via env, so ambient MMSEQS_BIN/BLASTP_BIN stays scrubbed.
        explicit = {
            "SHIFTX_BIN": getattr(args, "shiftx_bin", None),
            "SHIFTX_SOURCE": getattr(args, "shiftx_source", None),
            "SHIFT_DATA_ARCHIVE": getattr(args, "data_archive", None),
            "SHIFT_SPLIT_SEED": getattr(args, "split_seed", None),
            "SHIFT_HOLDOUT_FRAC": getattr(args, "holdout_frac", None),
        }
        self.env.update({k: str(v) for k, v in explicit.items() if v is not None})
        self.env.update({
            "SHIFT_DATA_ROOT": str(self.paths.data),
            "SHIFT_PDB_DIR": str(self.paths.data / "pdbs"),
            "SHIFT_ALPHAFOLD_DIR": str(self.paths.data / "alphafold"),
            "SHIFT_BIG_STORAGE": str(self.paths.work),
            "BIG_STORAGE": str(self.paths.work),
            "SHIFT_TOOL_DIR": str(self.paths.tools),
            "SHIFT_CHECKPOINT_DIR": str(self.paths.data / "checkpoints"),
            "CHEMSHIFTS": str(self.paths.data / "chemical_shifts.csv"),
            "PDB_SEQRES_DB": str(self.paths.data / "crystal_blast" /
                                 "pdb_seqres_db" / "pdb_seqres_prot"),
            "CUDA_VISIBLE_DEVICES": "",
        })

    # ---- input resolution ------------------------------------------------
    def _in(self, name, default_under_inputs):
        val = getattr(self.args, name, None)
        return Path(val) if val else (self.paths.inputs / default_under_inputs)

    def command(self, stage, module, *arguments, outputs=(), inputs=()):
        argv = [sys.executable, "-m", module, *(str(x) for x in arguments)]
        missing_inputs = [str(path) for path in inputs if not _path_is_nonempty(path)]
        if missing_inputs:
            raise RuntimeError(
                f"[{stage}] declared input(s) are missing or empty: "
                + ", ".join(missing_inputs)
            )
        print(f"[{stage}] {' '.join(argv)}", flush=True)
        subprocess.run(argv, env=self.env, check=True)
        missing_outputs = [str(path) for path in outputs if not _path_is_nonempty(path)]
        if missing_outputs:
            raise RuntimeError(
                f"[{stage}] command exited successfully but declared output(s) "
                "are missing or empty: " + ", ".join(missing_outputs)
            )

    # ---- convenience -----------------------------------------------------
    @property
    def mmseqs_bin(self):
        """The MMseqs2 executable: --mmseqs-bin, else MMSEQS_BIN/PATH, else a SHA-256-pinned
        copy provisioned on the spot. Resolved on first use rather than in __init__, so a run of
        stages that never cluster anything cannot trigger a download.
        """
        if not self._mmseqs_bin:
            from .sequence_clustering import resolve_mmseqs
            self._mmseqs_bin = resolve_mmseqs()
        return self._mmseqs_bin

    @property
    def cache_folds(self):
        """Requested fold universe, including fold 0 and the fold-6 holdout.

        QC applies to evaluation as well as training, so an ``all7`` authority has to
        cover folds 0 and 6.
        """
        requested = sorted(set(self.args.folds))
        invalid = [fold for fold in requested if fold not in range(0, 7)]
        if invalid:
            raise SystemExit(f"--folds contains invalid fold ids: {invalid}")
        if not requested:
            raise SystemExit("--folds must name at least one fold")
        return requested

    def _api_ids_file(self) -> Path:
        """Materialize the entity-mode api-ids JSON (subset if --limit-bmrbs)."""
        out = self.paths.data / "entity_api_ids.json"
        snapshot = self._in("snapshot", "bmrb_snapshot")
        if getattr(self.args, "limit_bmrbs", None):
            ids = [l.strip() for l in Path(self.args.limit_bmrbs).read_text().splitlines()
                   if l.strip() and not l.startswith("#")]
        else:
            import re
            ids = sorted({m.group(1) for name in os.listdir(snapshot)
                          if (m := re.fullmatch(r"bmr(\d+)", name))})
        out.write_text(json.dumps(ids))
        return out

    # ================= STAGES =================
    def fetch(self):
        snapshot = self._in("snapshot", "bmrb_snapshot")
        ids_file = self._api_ids_file()
        args = ["--out", self.paths.data, "--snapshot", snapshot,
                "--api-ids-file", ids_file,
                "--coverage-floor", getattr(self.args, "coverage_floor", 0.80)]
        if getattr(self.args, "limit_bmrbs", None):
            args += ["--limit-bmrbs", self.args.limit_bmrbs]
        self.command("fetch", "deepshift.data.fetch", *args,
                     outputs=[self.paths.data / "chemical_shifts.csv",
                              self.paths.data / "entity_catalog.csv",
                              self.paths.data / "label_set_selection.csv",
                              self.paths.data / "fetch_manifest.json"],
                     inputs=[ids_file, snapshot])

    def assembly(self):
        """Classify protein copy count directly from the frozen NMR-STAR snapshot."""
        snapshot = self._in("snapshot", "bmrb_snapshot")
        out = self.paths.data / "sample_assembly.json"
        self.command(
            "assembly", "deepshift.data.assembly",
            "--snapshot", snapshot,
            "--entity-catalog", self.paths.data / "entity_catalog.csv",
            "--out", out,
            "--workers", self.args.workers,
            outputs=[out],
            inputs=[snapshot, self.paths.data / "entity_catalog.csv"],
        )

    def uniprot_snapshot(self):
        """Frozen mode: consolidate the pinned snapshot -> candidate ledger and a
        BMRB->UniProt AF map for discovery. The live EBI BLAST is not called."""
        if getattr(self.args, "uniprot_mode", "frozen") != "frozen":
            raise SystemExit("the entity build requires --uniprot-mode frozen")
        snap = self._in("uniprot_snapshot_dir", "uniprot_snapshot")
        ledger = getattr(self.args, "uniprot_candidates", None)
        ledger = Path(ledger) if ledger else (self.paths.data / "uniprot_mapping_candidates.csv")
        manifest = self.paths.data / "uniprot_mapping_manifest.json"
        if not ledger.is_file():
            self.command("uniprot_snapshot", "deepshift.data.uniprot_snapshot",
                         "--snapshot-dir", snap, "--out-ledger", ledger,
                         "--out-manifest", manifest,
                         outputs=[ledger, manifest], inputs=[snap])
        # BMRB -> [accessions] AF map (baseline first), consumed by discover; one key per
        # deposition of this build, empty where the snapshot holds no candidate.
        af_dir = self.paths.data / "alphafold"
        af_dir.mkdir(parents=True, exist_ok=True)
        universe = json.loads((self.paths.data / "entity_api_ids.json").read_text())
        af_map = uniprot_map_for_discovery(ledger, universe)
        (af_dir / "bmrb_uniprot_mapping.json").write_text(json.dumps(af_map))

    def _prep_crystal_blast(self):
        """Stage discovery's declared inputs into the build, re-pointing every one of
        them when the caller names a different file."""
        cb = self.paths.data / "crystal_blast"
        cb.mkdir(parents=True, exist_ok=True)
        _point_at(self._in("pdb_seqres_db", "pdb_seqres_db"), cb / "pdb_seqres_db")
        _point_at(self._in("pdb_method_cache", "pdb_method_resolution.json"),
                  cb / "pdb_method_resolution.json")
        _copy_input(self._in("pairs", "pairs.csv"), self.paths.data / "pairs.csv")

    def _structure_archive(self) -> Path:
        """The frozen structure archive, linked into the build's coordinate directories."""
        archive = getattr(self.args, "data_archive", None)
        if not archive:
            raise SystemExit("--data-archive must name the structure archive holding "
                             + ", ".join(f"{d}/" for d in STRUCTURE_ARCHIVE_DIRS))
        archive = Path(archive).resolve()
        counts = link_structure_archive(archive, self.paths.data)
        print(f"structure archive {archive}: "
              + ", ".join(f"{name} {n:,}" for name, n in counts.items()), flush=True)
        return archive

    def discover(self):
        self._prep_crystal_blast()
        self._structure_archive()
        cb = self.paths.data / "crystal_blast"
        self.command("discover", "deepshift.data.discover",
                     "--data_dir", self.paths.data, "--out_dir", self.paths.data,
                     "--blastp_bin", self.blastp_bin,
                     "--threads", self.args.workers,
                     "--max_target", getattr(self.args, "discovery_max_targets", 5000),
                     "--no_remote_metadata",
                     outputs=[self.paths.data / "discovery_selection_af.csv",
                              self.paths.data / "discovery_selection_nmr.csv",
                              self.paths.data / "discovery_selection_xtal.csv",
                              self.paths.data / "candidate_ledger.csv",
                              # per-BMRB x per-source coverage-accounting gate --
                              # discovery fails closed if a valid BMRB was never queried.
                              self.paths.data / "coverage_accounting.csv"],
                     inputs=[self.paths.data / "chemical_shifts.csv",
                             self.paths.data / "entity_catalog.csv",
                             self.paths.data / "alphafold" / "bmrb_uniprot_mapping.json",
                             self.paths.data / "pairs.csv",
                             cb / "pdb_seqres_db",
                             cb / "pdb_method_resolution.json",
                             Path(self.blastp_bin)])

    def materialize(self):
        archive = self._structure_archive()
        self.command("materialize", "deepshift.data.materialize",
                     "--shifts-file", self.paths.data / "chemical_shifts.csv",
                     "--selection-dir", self.paths.data,
                     "--out-dir", self.paths.data,
                     "--pdb-dir", self.paths.data / "pdbs",
                     "--alphafold-dir", self.paths.data / "alphafold",
                     "--crystal-dir", self.paths.data / "crystal_pdbs",
                     "--build-id", self.build_id,
                     "--min-query-coverage", getattr(self.args, "min_query_coverage", 0.90),
                     "--min-identity", getattr(self.args, "min_identity", 0.95),
                     outputs=[self.paths.data / "structure_data_af.csv",
                              self.paths.data / "structure_data_nmr.csv",
                              self.paths.data / "structure_data_xtal.csv",
                              self.paths.data / "structure_build_log_af.csv",
                              self.paths.data / "structure_build_log_nmr.csv",
                              self.paths.data / "structure_build_log_xtal.csv",
                              self.paths.data / "build_log_af.csv",
                              self.paths.data / "build_log_nmr.csv",
                              self.paths.data / "build_log_xtal.csv",
                              self.paths.data / "selection_af.csv",
                              self.paths.data / "selection_nmr.csv",
                              self.paths.data / "selection_xtal.csv",
                              self.paths.data / "realization_ledger.csv",
                              self.paths.data / "label_set_dedup.csv",
                              self.paths.data / "pdbs",
                              self.paths.data / "alphafold",
                              self.paths.data / "crystal_pdbs"],
                     inputs=[self.paths.data / "chemical_shifts.csv",
                             self.paths.data / "entity_catalog.csv",
                             self.paths.data / "candidate_ledger.csv",
                             *(archive / name for name in STRUCTURE_ARCHIVE_DIRS)])

    def cluster(self):
        self.command("cluster", "deepshift.data.cluster",
                     "--data_dir", self.paths.data, "--out_dir", self.paths.data,
                     "--mmseqs_bin", self.mmseqs_bin,
                     outputs=[self.paths.data / "entity_edges_exact_token.csv",
                              self.paths.data / "entity_edges_alignment.csv",
                              self.paths.data / "entity_edges_mmseqs_cov0.csv",
                              self.paths.data / "entity_edges_mmseqs_containment.csv",
                              self.paths.data / "entity_edges_provenance.json"],
                     inputs=[self.paths.data / "structure_data_af.csv",
                             self.paths.data / "structure_data_nmr.csv",
                             self.paths.data / "structure_data_xtal.csv",
                             self.paths.data / "entity_catalog.csv",
                             self.paths.data / "build_log_af.csv",
                             self.paths.data / "build_log_nmr.csv",
                             self.paths.data / "build_log_xtal.csv",
                             Path(self.mmseqs_bin)])

    def folds(self):
        packaged = Path(__file__).with_name("ucbshift200_excluded_bmrbs.json")
        dest = self.paths.data / packaged.name
        _materialize_exclusion(packaged, dest, "UCBShift-200")
        case_study = Path(__file__).with_name("case_study_excluded_bmrbs.json")
        case_dest = self.paths.data / case_study.name
        _materialize_exclusion(case_study, case_dest, "case-study")
        # The direct rule over the 200 UCBShift test proteins is the exclusion authority;
        # the packaged BMRB list closes same-BMRB ownership. The test-target JSON is passed
        # explicitly.
        if not UCB_TEST_TARGETS or not Path(UCB_TEST_TARGETS).is_file():
            raise RuntimeError(
                f"UCBShift-200 test targets missing: "
                f"{UCB_TEST_TARGETS or '$SHIFT_UCB_TEST_TARGETS is unset'}. Point "
                f"SHIFT_UCB_TEST_TARGETS at the JSON conversion of CSpred's test_200.pkl.")
        test_targets = Path(UCB_TEST_TARGETS)
        self.command("folds", "deepshift.data.folds",
                     "--data_dir", self.paths.data, "--edge_dir", self.paths.data,
                     "--ledger", self.paths.data / "candidate_ledger.csv",
                     "--excluded_bmrbs", dest, "--test_targets", test_targets,
                     "--case_study_bmrbs", case_dest,
                     "--mmseqs_bin", self.mmseqs_bin,
                     "--balance_tol", getattr(self.args, "balance_tol", 0.15),
                     "--holdout_source_tol", getattr(self.args, "holdout_source_tol", 0.15),
                     outputs=[self.paths.data / "fold_assignments_labelset_identity90.json",
                              self.paths.data / "fold_assignments_labelset_derived_bmrb.json",
                              self.paths.data / "quarantine_short_entities.csv",
                              self.paths.data / "labelset_edges_provenance.json",
                              self.paths.data / "fold_balance_report.json",
                              self.paths.data / "fold0_training_relatives.json",
                              self.paths.data / "fold_case_study_pin.json",
                              self.paths.data / "fold_case_study_boundary.json",
                              self.paths.data / "fold0_calibration_exclusions.json",
                              *[self.paths.data / f"labelset_edges_{f}.csv"
                                for f in ("same_bmrb", "uniprot", "exact_token",
                                          "alignment", "mmseqs_cov0",
                                          "mmseqs_containment", "modeled_region")]],
                     inputs=[self.paths.data / "candidate_ledger.csv",
                             self.paths.data / "entity_edges_provenance.json",
                             self.paths.data / "structure_data_af.csv",
                             self.paths.data / "structure_data_nmr.csv",
                             self.paths.data / "structure_data_xtal.csv",
                             self.paths.data / "entity_catalog.csv",
                             self.paths.data / "build_log_af.csv",
                             self.paths.data / "build_log_nmr.csv",
                             self.paths.data / "build_log_xtal.csv",
                             self.paths.data / "entity_edges_exact_token.csv",
                             self.paths.data / "entity_edges_alignment.csv",
                             self.paths.data / "entity_edges_mmseqs_cov0.csv",
                             self.paths.data / "entity_edges_mmseqs_containment.csv",
                             dest,
                             case_dest,
                             test_targets,
                             self.paths.data / "alphafold" / "bmrb_uniprot_mapping.json",
                             Path(self.mmseqs_bin)])

    def partner_suppression(self):
        """Bind cross-fold partner suppression to the newly generated fold map."""
        d = self.paths.data
        out = d / "partner_chain_suppression.json"
        report = d / "partner_chain_suppression.report.json"
        build_logs = [d / f"structure_build_log_{source}.csv"
                      for source in ("af", "nmr", "xtal")]
        fold_map = d / "fold_assignments_labelset_identity90.json"
        self.command(
            "partner_suppression", "deepshift.data.partner_suppression",
            "--build-logs", *build_logs,
            "--fold-map", fold_map,
            "--out", out,
            "--report", report,
            outputs=[out, report],
            inputs=[fold_map, *build_logs],
        )

    def split(self):
        d = self.paths.data
        self.command("split", "deepshift.data.split",
                     "--source_master", f"af:{d / 'structure_data_af.csv'}",
                     "--source_master", f"nmr:{d / 'structure_data_nmr.csv'}",
                     "--source_master", f"xtal:{d / 'structure_data_xtal.csv'}",
                     "--fold_map", d / "fold_assignments_labelset_identity90.json",
                     "--out_dir", self.paths.folds_dir, "--dataset", "unified",
                     outputs=[self.paths.folds_dir],
                     inputs=[d / "fold_assignments_labelset_identity90.json",
                             d / "structure_data_af.csv",
                             d / "structure_data_nmr.csv",
                             d / "structure_data_xtal.csv"])

    def reref(self):
        d = self.paths.data
        # One offset per label set and reference channel, fitted on that label set's
        # own materialized structure with SHIFTX.
        args = ["--shifts_file", d / "chemical_shifts.csv",
                "--out", d / "reref_offsets.json",
                "--workers", self.args.workers]
        for source in ("af", "nmr", "xtal"):
            args += ["--structure-master", f"{source}:{d / f'structure_data_{source}.csv'}",
                     "--build-log", f"{source}:{d / f'structure_build_log_{source}.csv'}"]
        # The SHIFTX executable the stage runs is an input. The path is the one
        # deepshift.data.reref resolves (SHIFTX_BIN, else the SHIFT_TOOL_DIR build).
        tool_dir = self.env.get("SHIFT_TOOL_DIR", str(self.paths.tools))
        tool_paths = [self.env.get("SHIFTX_BIN") or os.path.join(tool_dir, "shiftx", "shiftx")]
        if self.env.get("SHIFTX_SOURCE"):
            tool_paths.append(self.env["SHIFTX_SOURCE"])
        self.command("reref", "deepshift.data.reref", *args,
                     outputs=[d / "reref_offsets.json",
                              d / "reref_provenance.json"],
                     inputs=[d / "chemical_shifts.csv",
                             d / "entity_catalog.csv",
                             d / "structure_data_af.csv",
                             d / "structure_data_nmr.csv",
                             d / "structure_data_xtal.csv",
                             d / "structure_build_log_af.csv",
                             d / "structure_build_log_nmr.csv",
                             d / "structure_build_log_xtal.csv",
                             *(Path(p) for p in dict.fromkeys(tool_paths))])

    def _source_build_log_specs(self):
        d = self.paths.data
        return [f"af:{d / 'structure_build_log_af.csv'}:alphafold",
                f"nmr:{d / 'structure_build_log_nmr.csv'}:experimental",
                f"xtal:{d / 'structure_build_log_xtal.csv'}:crystal"]

    def cache(self):
        d = self.paths.data
        folds = [str(f) for f in self.cache_folds]
        cache_outputs = []
        for fold in self.cache_folds:
            fold_dir = self.paths.cache / f"fold_{fold}"
            cache_outputs += [fold_dir / "config.json", fold_dir / "samples.npy",
                              fold_dir / "structural"]
        flagged_inputs = (
            [Path(self.args.flagged)]
            if getattr(self.args, "flagged", None)
            else []
        )
        self.command("cache", "deepshift.data.cache",
                     "--data_dir", self.paths.folds_dir, "--output_dir", self.paths.cache,
                     "--rebuild_root", self.paths.root, "--folds", *folds,
                     "--source_build_logs", *self._source_build_log_specs(),
                     "--source_names", "af", "nmr", "xtal",
                     "--reref_offsets", d / "reref_offsets.json",
                     "--no_h",
                     # p100 -> S_NUC: the robust per-nucleus scale fit excludes the
                     # frozen p100(_structural) flagged cells (physically implausible +
                     # structure/shift-mismatch), matching what the CV trainer excludes.
                     *(["--flagged", str(self.args.flagged)]
                       if getattr(self.args, "flagged", None) else []),
                     *(["--tiny_stats_floor"]
                       if getattr(self.args, "tiny_stats_floor", False) else []),
                     outputs=cache_outputs,
                     inputs=[d / "reref_offsets.json",
                             self.paths.folds_dir,
                             d / "structure_build_log_af.csv",
                             d / "structure_build_log_nmr.csv",
                             d / "structure_build_log_xtal.csv",
                             *flagged_inputs])

    def _partner_authorities(self):
        """The sample-composition map and the cross-fold suppression map every cloud
        is built under."""
        sample_assembly = Path(getattr(self.args, "sample_assembly", None)
                               or self.paths.data / "sample_assembly.json")
        suppress_map = Path(getattr(self.args, "leaky_partner_chains", None)
                            or self.paths.data / "partner_chain_suppression.json")
        return sample_assembly, suppress_map

    def atom_census(self):
        out = self.paths.data / "atom_census.json"
        sample_assembly, suppress_map = self._partner_authorities()
        self.command("atom_census", "deepshift.data.atom_census",
                     "--source_build_logs", *self._source_build_log_specs(),
                     "--pdb_search_dirs", self.paths.data / "pdbs",
                     self.paths.data / "alphafold", self.paths.data / "crystal_pdbs",
                     "--out", out,
                     "--sample_assembly", sample_assembly,
                     "--leaky_partner_chains", suppress_map,
                     outputs=[out],
                     inputs=[self.paths.data / "structure_build_log_af.csv",
                             self.paths.data / "structure_build_log_nmr.csv",
                             self.paths.data / "structure_build_log_xtal.csv",
                             self.paths.data / "pdbs",
                             self.paths.data / "alphafold",
                             self.paths.data / "crystal_pdbs",
                             sample_assembly, suppress_map])

    def atoms(self):
        census = self.paths.data / "atom_census.json"
        if not census.is_file():
            raise RuntimeError("atom_census.json is required before atom construction")
        m_atoms = int(json.loads(census.read_text())["chosen_m_atoms"])
        if m_atoms != 512:
            raise RuntimeError(f"final atom recipe requires M=512; census declared {m_atoms}")
        # The atom stage refuses any materialization or residue that could not build a
        # cloud. --allow-atom-rejections accepts the ledgered drops instead (recorded in
        # each fold's atom/rejections.json).
        rej = ["--allow_atom_rejections"] if getattr(
            self.args, "allow_atom_rejections", False) else []
        policy = getattr(self.args, "partner_sample_policy", "sample_matched")
        if policy != "sample_matched":
            raise RuntimeError("the final build supports only sample_matched partners")
        sample_assembly, suppress_map = self._partner_authorities()
        partner_args = ["--partner_sample_policy", policy,
                        "--sample_assembly", str(sample_assembly),
                        "--leaky_partner_chains", str(suppress_map)]
        for fold in self.cache_folds:
            fold_dir = self.paths.cache / f"fold_{fold}"
            self.command(
                f"atoms-{fold}", "deepshift.data.atoms",
                "--cache_dir", fold_dir,
                "--m_atoms", m_atoms,
                *rej, *partner_args,
                "--pdb_search_dirs", self.paths.data / "pdbs",
                self.paths.data / "alphafold", self.paths.data / "crystal_pdbs",
                "--source_build_logs", *self._source_build_log_specs(),
                outputs=[fold_dir / "atom"],
                inputs=[census, fold_dir / "structural",
                        self.paths.data / "structure_build_log_af.csv",
                        self.paths.data / "structure_build_log_nmr.csv",
                        self.paths.data / "structure_build_log_xtal.csv",
                        self.paths.data / "pdbs",
                        self.paths.data / "alphafold",
                        self.paths.data / "crystal_pdbs",
                        suppress_map, sample_assembly])

    def flags(self):
        policy = FINAL_QC_POLICY
        exclusion_out = self.paths.data / f"qc_flags_{policy}.json"
        exclusion_report = exclusion_out.with_suffix(".report.json")
        cutoff_args = (
            ["--cutoffs", str(self.args.p100_cutoffs)]
            if policy.startswith("p100")
            else []
        )
        self.command("flags-exclusion", "deepshift.quality_control",
                     "--policy", policy, "--cache", self.paths.cache,
                     *cutoff_args,
                     "--out", exclusion_out,
                     "--report", exclusion_report,
                     outputs=[exclusion_out, exclusion_report],
                     inputs=[self.paths.cache]
                     + ([Path(self.args.p100_cutoffs)]
                        if policy.startswith("p100") else []))
        if getattr(self.args, "flagged", None):
            supplied = json.loads(Path(self.args.flagged).read_text())
            derived = json.loads(exclusion_out.read_text())
            if supplied != derived:
                raise RuntimeError(
                    "the pre-cache --flagged authority differs from the "
                    f"post-cache {policy} derivation"
                )

    def run(self):
        first, last = (STAGES.index(self.args.from_stage),
                       STAGES.index(self.args.to_stage))
        # --from-stage must not come after --to-stage: an inverted range selects no
        # stages at all.
        if first > last:
            raise SystemExit(
                f"--from-stage {self.args.from_stage} is after --to-stage "
                f"{self.args.to_stage}; the stage graph is {' -> '.join(STAGES)}")
        selected = STAGES[first:last + 1]
        if "flags" in selected:
            cutoff = getattr(self.args, "p100_cutoffs", None)
            if not cutoff or not _path_is_nonempty(Path(cutoff)):
                raise RuntimeError(
                    "p100 policy requires --p100-cutoffs pointing to a nonempty "
                    "held-out-CV cutoff authority"
                )
        if "cache" in selected:
            flagged = getattr(self.args, "flagged", None)
            if not flagged or not _path_is_nonempty(Path(flagged)):
                raise RuntimeError(
                    "p100 cache normalization requires --flagged pointing to the "
                    "precomputed all-fold p100/p100_structural authority"
                )
        for stage in selected:
            getattr(self, stage)()


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True, help="fresh build directory")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(7)))
    parser.add_argument("--workers", type=int, default=8)
    # Immutable declared inputs (default under <root>/inputs).
    parser.add_argument("--snapshot", help="BMRB raw snapshot dir")
    parser.add_argument("--pairs", help="pairs.csv")
    parser.add_argument("--pdb-seqres-db", dest="pdb_seqres_db",
                        help="pdb_seqres BLAST DB dir (contains pdb_seqres_prot.*)")
    parser.add_argument("--pdb-method-cache", dest="pdb_method_cache",
                        help="pdb_method_resolution.json")
    parser.add_argument("--uniprot-snapshot", dest="uniprot_snapshot_dir",
                        help="pinned UniProt mapping snapshot dir")
    parser.add_argument("--uniprot-candidates", dest="uniprot_candidates",
                        help="pre-consolidated uniprot_mapping_candidates.csv")
    parser.add_argument("--uniprot-mode", choices=("frozen",), default="frozen",
                        help="only frozen is supported: BMRB-to-UniProt mapping is "
                             "always the static merge of the two pinned snapshot files")
    parser.add_argument("--limit-bmrbs", dest="limit_bmrbs",
                        help="file with one BMRB id per line (smoke subset)")
    parser.add_argument("--coverage-floor", type=float, default=0.80)
    parser.add_argument("--balance-tol", dest="balance_tol", type=float, default=0.15,
                        help="max per-fold count deviation; relax for tiny smoke "
                             "universes where 15%% is mathematically impossible")
    parser.add_argument("--holdout-source-tol", dest="holdout_source_tol", type=float,
                        default=0.15,
                        help="max relative deviation of each structure source's fold-6 "
                             "share from the overall holdout share")
    parser.add_argument("--tiny-stats-floor", dest="tiny_stats_floor", action="store_true",
                        help="floor every fitted nucleus scale at 0.1 ppm instead of "
                             "refusing a narrow one; for tiny smoke universes only, "
                             "recorded in the cache config")
    parser.add_argument("--discovery-max-targets", type=int, default=5000)
    parser.add_argument("--min-query-coverage", type=float, default=0.90)
    parser.add_argument("--min-identity", type=float, default=0.95)
    parser.add_argument("--allow-atom-rejections", dest="allow_atom_rejections",
                        action="store_true",
                        help="accept the atom stage's logged, ledgered "
                             "materialization/residue drops instead of "
                             "failing closed")
    parser.add_argument("--flagged", default=None,
                        help="frozen p100(_structural) flags JSON fed to the cache S_NUC fit "
                             "so normalization EXCLUDES physically-implausible/flagged cells")
    parser.add_argument(
        "--p100-cutoffs",
        dest="p100_cutoffs",
        help="held-out-CV p100 cutoff JSON; required for a p100* QC policy",
    )
    parser.add_argument("--leaky-partner-chains", dest="leaky_partner_chains", default=None,
                        help="override the generated cross-fold partner-chain suppress map")
    parser.add_argument("--partner-sample-policy", dest="partner_sample_policy",
                        choices=["sample_matched"], default="sample_matched",
                        help="include protein partners only when the NMR sample held them")
    parser.add_argument("--sample-assembly", dest="sample_assembly", default=None,
                        help="override the generated label-set-grained sample_assembly.json")
    parser.add_argument("--mmseqs-bin", help="MMseqs2 executable")
    parser.add_argument("--blastp-bin", help="blastp executable")
    parser.add_argument("--shiftx-bin")
    parser.add_argument("--shiftx-source")
    parser.add_argument("--data-archive")
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--holdout-frac", type=float)
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.set_defaults(handler=lambda args: Builder(args).run())
