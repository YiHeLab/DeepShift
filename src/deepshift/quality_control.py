"""Final label-exclusion policy: p100 plus structural Cys/Pro consistency."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import domain as identity
from .atom_cloud import AtomCloudRecipe
from .chemistry import CHEMICALLY_IMPOSSIBLE_PAIRS, STANDARD_RESIDUES
from .geometry_context import cysteine_state, proline_state
from .loss import AMBIGUITY_SET_CODES

ARG_NE_OBSERVED_RANGE_PPM = (78.976, 90.212)
FINAL_QC_POLICY = "p100_structural"
# ``none`` writes the empty fence the unfiltered calibration members train under, whose
# out-of-fold predictions fit the p100 spans (Methods, "Masking and outlier control").
POLICIES = ("none", FINAL_QC_POLICY)


@dataclass(frozen=True)
class P100Policy:
    """Fitted accepted spans, with the observed-value Arg Nε interval."""

    cutoffs: Mapping[str, Mapping[str, float]]
    arg_ne_range_ppm: tuple[float, float] = ARG_NE_OBSERVED_RANGE_PPM

    def accepted_range(self, residue: str, column: str) -> tuple[float, float] | None:
        pair = f"{residue}:{column}"
        if (residue, column) in CHEMICALLY_IMPOSSIBLE_PAIRS:
            return None
        if residue == "ARG" and column == "ne_shift":
            return self.arg_ne_range_ppm
        entry = self.cutoffs.get(pair)
        if not entry:
            return None
        low, high = entry.get("lo"), entry.get("hi")
        if low is None or high is None:
            return None
        low, high = float(low), float(high)
        if not (np.isfinite(low) and np.isfinite(high) and low <= high):
            raise ValueError(f"invalid p100 interval for {pair}: lo={low}, hi={high}")
        return low, high

    def excludes(self, residue: str, column: str, observed_ppm: float) -> bool:
        """Whether an observed value falls outside its inclusive accepted interval."""
        accepted = self.accepted_range(residue, column)
        if accepted is None:
            raise KeyError(f"no p100 interval for {residue}:{column}")
        low, high = accepted
        return not low <= float(observed_ppm) <= high


def load_p100_cutoffs(path: str | Path) -> dict:
    """Load the fitted p100 authority's residue/nucleus intervals."""
    payload = json.loads(Path(path).read_text())
    cutoffs = {key: value for key, value in payload.items() if ":" in key}
    if not cutoffs:
        raise ValueError(f"p100 authority has no residue/nucleus intervals: {path}")
    return cutoffs


def flags_by_label_residue(cells) -> dict[str, list[str]]:
    """Group exact ``(label_residue_id, shift_column)`` cell identities."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for label_residue_id, shift_column in cells:
        identity._require_id(label_residue_id, "lr_")
        grouped[label_residue_id].add(str(shift_column))
    return {key: sorted(columns) for key, columns in sorted(grouped.items())}


def write_flags(path, flags) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(flags, indent=1, sort_keys=True) + "\n")


def _fold_directories(cache: Path) -> list[Path]:
    folds = sorted(
        (path for path in cache.glob("fold_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not folds:
        raise ValueError(f"no fold_* directories under {cache}")
    return folds


def _shift_columns(fold: Path, expected):
    config = json.loads((fold / "config.json").read_text())
    if config.get("shift_storage") != "ppm_float32" and not config.get("has_ppm_shifts"):
        raise ValueError(f"{fold}: p100 requires ppm_float32 deposited shifts")
    columns = tuple(config["shift_cols"])
    if expected is not None and columns != expected:
        raise ValueError(f"{fold}: shift columns disagree between folds")
    return columns


def derive_p100_flags(
    cache: str | Path,
    cutoffs: Mapping[str, Mapping[str, float]] | str | Path,
) -> tuple[dict, dict]:
    """Apply the p100 spans and the Arg Nε interval to every observed deposited value."""
    if isinstance(cutoffs, (str, Path)):
        cutoff_values = load_p100_cutoffs(cutoffs)
    else:
        cutoff_values = {key: value for key, value in cutoffs.items() if ":" in key}
    policy = P100Policy(cutoff_values)
    cache = Path(cache)
    label_cells: list[tuple[str, str]] = []
    instance_hits = 0
    columns = None
    by_column: dict[str, int] = defaultdict(int)
    ambiguity_instances_out_of_scope = 0
    folds_read: list[str] = []
    missing_cutoffs: set[str] = set()

    for fold in _fold_directories(cache):
        folds_read.append(fold.name)
        columns = _shift_columns(fold, columns)
        structural = fold / "structural"
        shifts = np.load(structural / "shifts.npy", mmap_mode="r")
        mask = np.load(structural / "shift_mask.npy", mmap_mode="r")
        ambiguity = np.load(structural / "ambiguity.npy", mmap_mode="r")
        residue_index = np.load(structural / "residue_idx.npy", mmap_mode="r")
        amino_acid = (residue_index if residue_index.ndim == 1
                      else residue_index[:, residue_index.shape[1] // 2])
        if (shifts.shape != mask.shape or ambiguity.shape != mask.shape
                or shifts.shape[1] != len(columns)):
            raise ValueError(f"{fold}: shifts/mask/ambiguity/config shapes disagree")
        label_map_path = structural / "global_to_label_residue.json"
        if not label_map_path.is_file():
            raise ValueError(f"{fold}: p100 requires global_to_label_residue.json")
        label_map = json.loads(label_map_path.read_text())

        for column_index, column in enumerate(columns):
            observed = np.asarray(mask[:, column_index], dtype=bool)
            set_code = observed & np.isin(
                np.asarray(ambiguity[:, column_index]), AMBIGUITY_SET_CODES
            )
            ambiguity_instances_out_of_scope += int(set_code.sum())
            rows = np.flatnonzero(observed & ~set_code)
            if not len(rows):
                continue
            values = np.asarray(shifts[rows, column_index], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{fold}: observed {column} shifts are non-finite")
            codes = np.asarray(amino_acid[rows])
            bad = np.zeros(len(rows), dtype=bool)
            for code in np.unique(codes):
                if not 0 <= int(code) < len(STANDARD_RESIDUES):
                    continue
                residue = STANDARD_RESIDUES[int(code)]
                pair = f"{residue}:{column}"
                selected = codes == code
                if (residue, column) in CHEMICALLY_IMPOSSIBLE_PAIRS:
                    continue
                accepted = policy.accepted_range(residue, column)
                if accepted is None:
                    missing_cutoffs.add(pair)
                    continue
                low, high = accepted
                bad |= selected & ((values < low) | (values > high))
            for row in rows[bad]:
                label_residue_id = label_map.get(str(int(row)))
                if label_residue_id is not None:
                    label_cells.append((label_residue_id, column))
                instance_hits += 1
                by_column[column] += 1

    if missing_cutoffs:
        raise ValueError(
            "p100 authority does not cover observed residue/nucleus populations: "
            f"{sorted(missing_cutoffs)}"
        )
    result = flags_by_label_residue(label_cells)
    report = {
        "policy": "p100",
        "schema_key": "label_residue_id",
        "rule": "fitted accepted spans with the observed Arg Nepsilon interval",
        "arg_ne_observed_range_ppm": list(ARG_NE_OBSERVED_RANGE_PPM),
        "instance_hits": instance_hits,
        "unique_flags": sum(len(columns) for columns in result.values()),
        "proteins": len(result),
        "instance_hits_by_nucleus": dict(sorted(by_column.items())),
        "ambiguity_set_codes_out_of_scope": list(AMBIGUITY_SET_CODES),
        "ambiguity_instances_out_of_scope": ambiguity_instances_out_of_scope,
        "folds_read": folds_read,
        "cells_without_label_residue": instance_hits - len(label_cells),
    }
    return result, report


def derive_structural_flags(cache, *, observation_mask_loader=None):
    """Flag all cells on residues with a Cys or Pro structure/shift conflict."""
    cache = Path(cache)
    cys_code = STANDARD_RESIDUES.index("CYS")
    pro_code = STANDARD_RESIDUES.index("PRO")
    label_cells: list[tuple[str, str]] = []
    columns = None
    counts: dict[str, int] = defaultdict(int)
    examples: list[dict] = []
    folds_read: list[str] = []

    for fold in _fold_directories(cache):
        folds_read.append(fold.name)
        columns = _shift_columns(fold, columns)
        column_index = {column: index for index, column in enumerate(columns)}
        if "cb_shift" not in column_index:
            continue
        cb_index = column_index["cb_shift"]
        cg_index = column_index.get("cg_shift")
        structural, atom = fold / "structural", fold / "atom"
        for name in ("atom_type.npy", "atom_offset.npy", "atom_xyz.npy"):
            if not (atom / name).is_file():
                raise ValueError(f"{fold}: structural QC needs {name}")
        label_map = json.loads(
            (structural / "global_to_label_residue.json").read_text()
        )
        fold_config = json.loads((fold / "config.json").read_text())
        source_names = fold_config.get("source_names") or []
        atom_config = json.loads((atom / "config.json").read_text())
        recipe = AtomCloudRecipe.from_mapping(atom_config["recipe"])
        amino_acid = np.load(structural / "residue_idx.npy", mmap_mode="r")
        amino_acid = (amino_acid if amino_acid.ndim == 1
                      else amino_acid[:, amino_acid.shape[1] // 2])
        source = np.load(structural / "source_idx.npy", mmap_mode="r")
        shifts = np.load(structural / "shifts.npy", mmap_mode="r")
        stored_mask = np.asarray(
            np.load(structural / "shift_mask.npy", mmap_mode="r"), dtype=bool
        )
        mask = (stored_mask if observation_mask_loader is None
                else np.asarray(observation_mask_loader(fold), dtype=bool))
        if mask.shape != stored_mask.shape:
            raise ValueError(f"{fold}: structural QC observation-mask shape differs")
        if np.any(mask & ~np.isfinite(np.asarray(shifts))):
            raise ValueError(f"{fold}: structural QC mask selects non-finite shifts")
        atom_type = np.asarray(np.load(atom / "atom_type.npy"))
        atom_offset = np.asarray(np.load(atom / "atom_offset.npy"))
        atom_xyz = np.asarray(np.load(atom / "atom_xyz.npy"))

        candidates = np.flatnonzero(
            ((amino_acid == cys_code) | (amino_acid == pro_code)) & mask[:, cb_index]
        )
        for row in candidates:
            label_residue_id = label_map.get(str(int(row)))
            if label_residue_id is None:
                continue
            if amino_acid[row] == cys_code:
                state = cysteine_state(
                    atom_type[row], atom_offset[row], atom_xyz[row],
                    shifts[row, cb_index], recipe=recipe,
                )
                kind = "disulfide"
            else:
                if cg_index is None or not mask[row, cg_index]:
                    continue
                state = proline_state(
                    atom_type[row], atom_offset[row], atom_xyz[row],
                    shifts[row, cb_index], shifts[row, cg_index], recipe=recipe,
                )
                kind = "cis_trans_pro"
            if not state.conflicts:
                continue
            source_name = (source_names[int(source[row])]
                           if int(source[row]) < len(source_names) else "?")
            counts[kind] += 1
            counts[f"{kind}:{source_name}"] += 1
            counts[f"source:{source_name}"] += 1
            for index, column in enumerate(columns):
                if mask[row, index]:
                    label_cells.append((label_residue_id, column))
            if len(examples) < 12:
                examples.append({
                    "fold": fold.name,
                    "kind": kind,
                    "source": source_name,
                    "shift_state": state.shift_state,
                    "geometry_state": state.geometry_state,
                })

    flags = flags_by_label_residue(label_cells)
    report = {
        "policy": "structural",
        "rule": "Cys disulfide and Pro cis/trans structure-label consistency",
        "residues_flagged": counts["disulfide"] + counts["cis_trans_pro"],
        "cells_flagged": sum(len(columns) for columns in flags.values()),
        "proteins": len(flags),
        "folds_read": folds_read,
        "counts": dict(sorted(counts.items())),
        "examples": examples,
    }
    return flags, report


def union_flags(*flag_dicts):
    """Union exact label-residue cell maps."""
    merged: dict[str, set[str]] = defaultdict(set)
    for flag_dict in flag_dicts:
        for label_residue_id, columns in (flag_dict or {}).items():
            merged[label_residue_id].update(columns)
    return {key: sorted(columns) for key, columns in sorted(merged.items())}


def derive_flags(policy, cache=None, cutoffs=None) -> tuple[dict, dict]:
    """Derive the empty calibration fence or the final p100-plus-structural union."""
    if policy not in POLICIES:
        raise ValueError(f"unknown QC policy {policy!r}; expected one of {POLICIES}")
    if policy == "none":
        return {}, {"policy": "none", "unique_flags": 0, "proteins": 0}
    if not cutoffs:
        raise ValueError("p100 requires an explicit fitted cutoff authority")
    p100_flags, p100_report = derive_p100_flags(cache, cutoffs)
    structural_flags, structural_report = derive_structural_flags(cache)
    flags = union_flags(p100_flags, structural_flags)
    return flags, {
        "policy": FINAL_QC_POLICY,
        "rule": "union of p100 and separately named structural consistency flags",
        "unique_flags": sum(len(columns) for columns in flags.values()),
        "proteins": len(flags),
        "p100": p100_report,
        "structural": structural_report,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, default=FINAL_QC_POLICY)
    parser.add_argument("--cache")
    parser.add_argument("--cutoffs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.policy != "none" and not args.cache:
        parser.error(f"--policy {args.policy} requires --cache")
    flags, report = derive_flags(args.policy, args.cache, args.cutoffs)
    reports = [report, *(value for value in report.values() if isinstance(value, dict))]
    for part in reports:
        if part.get("cells_without_label_residue", 0):
            raise RuntimeError(
                f"{part['policy']} produced {part['cells_without_label_residue']} "
                "unaddressable cells"
            )
    write_flags(args.out, flags)
    report_path = (Path(args.report) if args.report
                   else Path(args.out).with_suffix(".report.json"))
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"policy={args.policy} wrote {report.get('unique_flags', 0)} flags "
        f"across {report.get('proteins', len(flags))} label residues"
    )


if __name__ == "__main__":
    main()
