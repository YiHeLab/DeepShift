"""The retrieval exclusion competitor benchmarks are scored under."""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ExclusionLevel(StrEnum):
    """Reference candidates excluded from a predictor's retrieval database."""

    OWN_DEPOSITION = "own_deposition"
    SAME_PROTEIN = "same_protein"
    BROAD_HOMOLOGY = "broad_homology"


DECLARED_MODES = {
    ExclusionLevel.OWN_DEPOSITION: frozenset({"bmrb_only"}),
    ExclusionLevel.SAME_PROTEIN: frozenset({"same_protein_bmrb_uniprot"}),
    ExclusionLevel.BROAD_HOMOLOGY: frozenset({"bmrb_list+identity_cutoff"}),
}

RETRIEVAL_ARMS = frozenset(
    {
        "ucbshift_full_leaky",
        "ucbshift_full_selfexcl",
        "shiftx2_leaky",
        "shiftx2_selfexcl",
        "ucbshift2",
        "ucbshift_full_bmrbonly",
        "shiftx2_bmrbonly",
        "ucbshift_full_sameprot",
        "ucbshift2_sameprot",
        "ucbshift2_selfexcl",
        "shiftx2_sameprot",
        "ucbshift2_bmrbonly",
    }
)

#: The exclusion the paper reports for every retrieval arm it scores: a candidate is a
#: leak if it is any deposition of the query's own protein -- the query's own BMRB entry,
#: an entry sharing its UniProt accession, or a reference keyed to its own PDB entry.
#: The shipped competitor stores also block the same protein by sequence: any reference at
#: >=95% identity over >=90% of the shorter sequence, or in the same UniRef100 cluster
#: (benchmarks/build_competitor_exclusions.py). Homologs below that stay eligible.
REPORTED_EXCLUSION = ExclusionLevel.SAME_PROTEIN
#: How a table names that exclusion beside a predictor.
REPORTED_EXCLUSION_PHRASE = "same-protein depositions blocked"
_REPORTED_ARM_SUFFIX = "sameprot"


#: Competitor arms staged outside the benchmark root, as ``arm=directory`` pairs separated
#: by commas (a re-run of one predictor). Each is read from its own directory, and the
#: reports record that directory.
ARM_DIRS_VAR = "DEEPSHIFT_ARM_DIRS"


def arm_overrides() -> dict:
    """``{arm: directory}`` from :data:`ARM_DIRS_VAR`, each directory resolved and present."""
    out = {}
    for item in filter(None, (part.strip() for part in os.environ.get(ARM_DIRS_VAR, "").split(","))):
        arm, sep, directory = item.partition("=")
        path = Path(directory).expanduser().resolve()
        if not (sep and arm and path.is_dir()):
            raise SystemExit(f"{ARM_DIRS_VAR}: {item!r} is not arm=<existing directory>")
        out[arm.strip()] = path
    return out


def arm_directory(output_root, arm: str) -> Path:
    """Where one competitor arm's per-structure outputs and ``hitlog/`` are."""
    return arm_overrides().get(arm) or Path(output_root) / arm


def reported_arm(predictor: str) -> str:
    """The key a scored cell dump gives ``predictor`` under the reported exclusion."""
    return f"{predictor}_{_REPORTED_ARM_SUFFIX}"


@dataclass(frozen=True)
class KeptCandidate:
    """One retrieval candidate retained by a competitor."""

    bmrb: str
    identity: float
    pdb_chain: str = ""


def _candidate_from_line(line: str) -> KeptCandidate | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 7 and parts[6] in {
        "keep_eligible",
        "drop_bmrb",
        "drop_identity",
    }:
        if parts[6] != "keep_eligible":
            return None
        target = parts[0]
        return KeptCandidate(
            bmrb=parts[1],
            identity=float(parts[5]),
            pdb_chain=target.split(".", 1)[1] if "." in target else "",
        )
    if line.startswith("HITLOG"):
        if len(parts) < 3:
            raise ValueError("HITLOG record has fewer than three fields")
        bmrb = "".join(character for character in parts[2].split(".")[0] if character.isdigit())
        identity = float(parts[3]) / 100.0 if len(parts) > 3 else 0.0
        return KeptCandidate(bmrb=bmrb, identity=identity)
    return None


def _violation(candidate=None, *, codes, reasons):
    result = {
        "candidate": None if candidate is None else candidate.bmrb,
        "reason_codes": list(codes),
        "reasons": list(reasons),
    }
    if candidate is not None:
        result["identity"] = round(candidate.identity, 3)
    return result


def verify_self_exclusion(
    tool: str,
    structure_id: str,
    task: dict,
    authority,
    *,
    output_root: str | Path,
    retrieval: bool,
    level: ExclusionLevel,
):
    """Return ``(passed, violations)`` for one competitor retrieval log.

    A log is inspected whenever one exists, whatever the arm is called. ``retrieval``
    says whether a log is mandatory: a predictor that retrieves reference shifts must
    leave one, and its absence fails closed; a structure-only predictor leaves none. An
    empty log records that the predictor found no reference to use; a log whose lines
    are all unrecognised is malformed and fails closed.
    """
    log = arm_directory(output_root, tool) / "hitlog" / f"{structure_id}.log"
    if not log.is_file():
        if not retrieval:
            return True, []
        violation = _violation(
            codes=["missing_hitlog"],
            reasons=["missing_hitlog: cannot verify self-exclusion"],
        )
        return False, [violation]

    own_bmrb = str(task["bmrb_id"])
    if level is ExclusionLevel.OWN_DEPOSITION:
        excluded_bmrbs = {own_bmrb}
        query_uniprots = set()
    else:
        excluded_bmrbs = {
            value for value in str(task["exclude_bmrbs"]).split(",") if value
        }
        query_uniprots = authority.query_uniprots(
            task["bmrb_id"], task["source"], task["accession"], task["chain"]
        )

    violations = []
    recognized = 0
    line_number = 0
    with log.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                candidate = _candidate_from_line(line)
            except (IndexError, TypeError, ValueError) as exc:
                violations.append(
                    _violation(
                        codes=["malformed_hitlog"],
                        reasons=[f"malformed_hitlog line {line_number}: {exc}"],
                    )
                )
                continue
            if candidate is None:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 7 and parts[6] in {"drop_bmrb", "drop_identity"}:
                    recognized += 1
                continue
            recognized += 1
            codes, reasons = [], []
            if candidate.bmrb in excluded_bmrbs:
                codes.append("same_protein" if level is not ExclusionLevel.OWN_DEPOSITION else "own_deposition")
                reasons.append("same_bmrb/linked")
            if (
                level is ExclusionLevel.BROAD_HOMOLOGY
                and candidate.identity >= 0.90
            ):
                codes.append("high_identity")
                reasons.append(f"identity {candidate.identity:.2f}>=0.90")
            if candidate.pdb_chain and level is not ExclusionLevel.OWN_DEPOSITION:
                candidate_uniprots = authority.pdb_uniprot.get(
                    (candidate.pdb_chain[:4].upper(), candidate.pdb_chain[4:5]), set()
                ) | authority.pdb_uniprot.get(
                    (candidate.pdb_chain[:4].upper(), "_"), set()
                )
                shared = sorted(candidate_uniprots & query_uniprots)
                if shared:
                    codes.append("same_uniprot")
                    reasons.append(f"same_uniprot {shared}")
            if reasons:
                violations.append(_violation(candidate, codes=codes, reasons=reasons))

    if line_number and recognized == 0 and not violations:
        violations.append(
            _violation(
                codes=["malformed_hitlog"],
                reasons=["malformed_hitlog: no recognized candidate records"],
            )
        )
    return not violations, violations

