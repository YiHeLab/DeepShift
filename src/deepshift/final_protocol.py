"""The machine-readable form of the DeepShift protocol."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .atom_cloud import FINAL_ATOM_CLOUD_RECIPE
from .calibration import (
    CALIBRATION_FOLD,
    DEFAULT_LEVELS,
    DEFAULT_N_BINS,
    EVALUATION_FOLD,
    MEMBER_FOLDS,
)
from .data.cluster import MMSEQS_COVERAGE, MMSEQS_COVERAGE_MODES, MMSEQS_IDENTITY
from .data.folds import (
    UCB_HOMOLOGY_COVERAGE,
    UCB_HOMOLOGY_COVERAGE_MODES,
    UCB_HOMOLOGY_IDENTITY,
)
from .data.sequence_clustering import EVALUE
from .data.structures import (
    SAMPLE_PARTNER_MIN_ALIGNED_LENGTH,
    SAMPLE_PARTNER_MIN_IDENTITY,
    SAMPLE_PARTNER_MIN_QUERY_COVERAGE,
)
from .domain import SHIFT_COLUMNS
from .quality_control import ARG_NE_OBSERVED_RANGE_PPM, FINAL_QC_POLICY
from .training import FINAL_TRAINING_RECIPE

SCHEMA = "deepshift.final-rebuild-protocol.v2"

# Each gate names the code that refuses when it is not met. A gate that only prints is
# not a gate, so every entry here must point at a raise, an exit code, or a test.
GATES = [
    {
        "gate": "every sequence search is an exhaustive all-versus-all easy-search at the "
                "declared identity, coverage, E-value and result cap, and no edge of any "
                "family crosses two of folds 1-6",
        "enforced_by": ["data.sequence_clustering.run_all_vs_all",
                        "data.folds.verify_no_labelset_leakage",
                        "data.folds.entity_main"],
    },
    {
        "gate": "fold-0 label sets with a training-fold relative at the cross-validation "
                "axis are counted per family and listed for the interval fit to exclude",
        "enforced_by": ["data.folds.fold0_training_relatives",
                        "calibration.fit_command"],
    },
    {
        "gate": "every admitted protein partner consumes a sequence- and copy-matched slot "
                "declared in that NMR sample",
        "enforced_by": ["data.structures.sample_matched_partner_residues_for"],
    },
    {
        "gate": "structure basenames are unique before the partner-suppression map is "
                "accepted, and a map that addresses a fold's structures must match at "
                "least one",
        "enforced_by": ["data.partner_suppression.build_suppression_map",
                        "data.atoms._check_suppression_applied"],
    },
    {
        "gate": "the atom cloud is the final recipe: 512 atoms, the nearest kept when a "
                "residue has more, truncation warned and counted, zero atom-build "
                "rejections, no hydrogen or deuterium",
        "enforced_by": ["data.atoms.atom_cloud_recipe", "data.atoms.choose_m_atoms",
                        "data.atoms.finalize_atom_rejections",
                        "atom_cloud._gather_rows", "training.load_atom_recipe"],
    },
    {
        "gate": "the five p100 calibration checkpoints declare unfiltered_calibration, "
                "zero excluded cells, final-epoch selection and one of folds 1-5 each",
        "enforced_by": ["scripts/fit_p100_oof64.py:_validate_calibration_checkpoint_manifest"],
    },
    {
        "gate": "the p100 fit is complete, every fitted span is finite and non-empty, and "
                "Arg N-epsilon carries the observed-value interval",
        "enforced_by": ["scripts/fit_p100_oof64.py:scientific_sanity"],
    },
    {
        "gate": "the composite fence re-derived from the cache equals the fitted file",
        "enforced_by": ["data.build.Builder.flags", "quality_control.derive_flags"],
    },
    {
        "gate": "every final checkpoint records the final training recipe",
        "enforced_by": ["training.verify_final_training_recipe"],
    },
    {
        "gate": "the interval fit reads fold 0 only and fold 6 is evaluated once, with the "
                "five fitted members",
        "enforced_by": ["calibration.calibration_population",
                        "calibration.IntervalCalibration",
                        "calibration.evaluate_interval_calibration"],
    },
    {
        "gate": "fold 6 is never used for model selection, QC fitting, normalization, or "
                "interval fitting",
        "enforced_by": ["training.train", "data.cache.compute_normalization_stats",
                        "scripts/fit_p100_oof64.py:CALIBRATION_FOLDS",
                        "calibration.CALIBRATION_FOLD"],
    },
]


def protocol() -> dict:
    """Return the closed scientific configuration and its pass/fail gates."""
    return {
        "schema": SCHEMA,
        "folds": {
            "calibration": CALIBRATION_FOLD,
            "model_members": list(MEMBER_FOLDS),
            "reserved_evaluation": EVALUATION_FOLD,
            "sequence_search": {
                "program": "MMseqs2 easy-search all-versus-all",
                "evalue": EVALUE,
                "result_cap": "number of sequences",
                "training_fold_families": {
                    "identity_min": MMSEQS_IDENTITY,
                    "coverage_min": MMSEQS_COVERAGE,
                    "coverage_modes": list(MMSEQS_COVERAGE_MODES),
                },
                "benchmark_exclusion_arm": {
                    "identity_min": UCB_HOMOLOGY_IDENTITY,
                    "coverage_min": UCB_HOMOLOGY_COVERAGE,
                    "coverage_modes": list(UCB_HOMOLOGY_COVERAGE_MODES),
                },
            },
        },
        "targets": {
            "nuclei": list(SHIFT_COLUMNS),
            "invalid_pairs_are_schema_not_qc": True,
            "ambiguity_sets_are_loss_and_evaluation_not_qc": True,
        },
        "quality_control": {
            "policy": FINAL_QC_POLICY,
            "shift_value_rule": "p100 fitted accepted spans for every residue and nucleus",
            "structural_rule": "Cys disulfide and Pro cis/trans consistency",
            "combination": "union",
            "arg_ne_observed_range_ppm": list(ARG_NE_OBSERVED_RANGE_PPM),
        },
        "atom_cloud": {
            **FINAL_ATOM_CLOUD_RECIPE.to_dict(),
            "over_cap_rule": "keep the atoms nearest the target residue; warn and count",
        },
        "sample_matched_partners": {
            "authority": "frozen NMR-STAR entity assembly, per label set",
            "identity_min": SAMPLE_PARTNER_MIN_IDENTITY,
            "query_coverage_min": SAMPLE_PARTNER_MIN_QUERY_COVERAGE,
            "aligned_residues_min": SAMPLE_PARTNER_MIN_ALIGNED_LENGTH,
            "declared_copy_count_is_a_hard_cap": True,
            "unknown_or_extra_coordinate_chains": "excluded",
            "cross_fold_dataset_partner_suppression": True,
        },
        "training": {
            **asdict(FINAL_TRAINING_RECIPE),
            "optimizer": "AdamW",
            "scheduler": "cosine annealing",
            "loss_reduction": "0.60 backbone + 0.40 side chain",
            "normalization": "fold-specific training folds only; final QC excluded",
            "ensemble": "mean in ppm after member-specific denormalization",
            "rounds": [
                {"name": "p100_calibration", "exclusion": "none",
                 "members": list(MEMBER_FOLDS), "use": "fit p100 only"},
                {"name": "final", "exclusion": FINAL_QC_POLICY,
                 "members": list(MEMBER_FOLDS), "use": "the paper models"},
            ],
        },
        "prediction_intervals": {
            "member_spread_ddof": 1,
            "variance_mapping": "RMSE^2 = floor^2 + gain * spread^2",
            "equal_count_bins": DEFAULT_N_BINS,
            "levels": list(DEFAULT_LEVELS),
            "variance_fit_fold": CALIBRATION_FOLD,
            "split_conformal_fit_fold": CALIBRATION_FOLD,
            "fit_excludes": "fold-0 label sets with a training-fold relative, and the "
                            "case-study records (data/fold0_calibration_exclusions.json)",
            "evaluate_once_fold": EVALUATION_FOLD,
            "backbone_only": True,
        },
        "gates": GATES,
    }


def protocol_command(args):
    payload = protocol()
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        print(f"wrote {destination}")
    else:
        print(text, end="")
    return payload


def add_protocol_arguments(parser):
    parser.add_argument("--out")
    parser.set_defaults(handler=protocol_command)
