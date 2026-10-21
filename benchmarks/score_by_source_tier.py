#!/usr/bin/env python3
"""The fold-6 competitor benchmark at one named self-exclusion tier.

`--tier` selects the exclusion level a retrieval arm is held to, and nothing else: this
module hands score_by_source.main the tiered verifier, so the common-cell rule, residual
convention, bootstrap and schema are that scorer's.

Tier 1 blocks the query's own BMRB deposition, tier 2 every deposition of the query's own
protein, tier 3 those plus any candidate at >= 90% identity. At every tier a retrieval
arm with no hit log is dropped, whatever the tier admits, and each arm must declare in
its own outputs the exclusion mode the tier it is scored at names.

Usage (the same-protein tier the manuscript's Table 1 reports):
  python -m benchmarks.score_by_source_tier --tier 2 \
      --tools ucbshift_full_sameprot ucbshift_x shiftx2_sameprot sparta \
      --boot 2000 --out <out>/score_tier2.json
"""
import argparse
import csv
import json
from pathlib import Path

from deepshift.self_exclusion import (
    DECLARED_MODES,
    ExclusionLevel,
    arm_directory,
    verify_self_exclusion,
)
from deepshift.self_exclusion import (
    RETRIEVAL_ARMS as HOMOLOGY_TOOLS,
)
from deepshift.statistics import DEFAULT_SEED

from . import cells as common
from . import config
from . import score_by_source as sbs

# self_exclusion.mode each arm must declare in its own output JSONs to be scored at a tier.
# Tier 2 blocks every deposition of the query's own protein -- own BMRB, the same-UniProt
# BMRB family, references keyed to its own PDB entry, and by sequence any reference at >=95%
# identity over >=90% of the shorter sequence or in the same UniRef100 cluster -- leaving
# homologs below that eligible. It sits between Tier 1 (own deposition only) and Tier 3
# (which also blocks >=90% genuine homologs, a stricter rule than the leak).
LEVEL_BY_TIER = {
    1: ExclusionLevel.OWN_DEPOSITION,
    2: ExclusionLevel.SAME_PROTEIN,
    3: ExclusionLevel.BROAD_HOMOLOGY,
}
TIER_MODES = {tier: DECLARED_MODES[level] for tier, level in LEVEL_BY_TIER.items()}

# What a kept retrieval candidate may be at each tier, as the report of every scorer
# states it.
TIER_PREDICATE = {
    1: "a kept candidate leaks iff its BMRB is the query's own deposition; a retrieval "
       "arm with no hit log fails closed",
    2: "a kept candidate leaks iff it is any deposition of the query's own protein: the "
       "query's BMRB, its same-UniProt component, or a reference keyed to a PDB entry "
       "sharing the query's UniProt. A homolog of another protein is eligible at every "
       "identity, and a retrieval arm with no hit log fails closed",
    3: "a kept candidate leaks iff it is same-BMRB/linked, or at identity >= 0.90, or "
       "same-UniProt; a retrieval arm with no hit log fails closed",
}


def verify_tiered(tool, sid, task, exc, tier):
    """Verify one benchmark arm under one named exclusion level."""
    try:
        level = LEVEL_BY_TIER[tier]
    except KeyError as exc_key:
        raise ValueError(f'unknown tier {tier!r}') from exc_key
    return verify_self_exclusion(
        tool,
        sid,
        task,
        exc,
        output_root=common.OUT_ROOT,
        retrieval=tool in HOMOLOGY_TOOLS,
        level=level,
    )


def verify_at_own_level(tool, sid, task, exc):
    """Verify one arm at the exclusion level its own name declares.

    ``config.arm_exclusion_tier`` reads the level off the staged arm name, so a census
    or a pool that scores every arm at once holds each of them to the level its table
    row reports, without a caller listing them.
    """
    tier = config.arm_exclusion_tier(tool)
    if tier is None:
        return True, 'leaky arm: no exclusion enforced'
    return verify_tiered(tool, sid, task, exc, tier)


def check_arm_policy(tools, leaky, tier, sids):
    """Every enforced homology arm must declare the tier's exclusion mode in its own output
    JSONs, catching a mis-tiered arm before a number is produced."""
    bad = {}
    for tool in tools:
        if tool in leaky or tool not in HOMOLOGY_TOOLS:
            continue
        modes = set()
        for sid in sids:
            p = arm_directory(common.OUT_ROOT, tool) / (sid + '.json')
            if p.exists():
                with open(p) as fh:
                    d = json.load(fh)
                if d.get('status') == 'ok':
                    modes.add(d.get('self_exclusion', {}).get('mode'))
        print(f'  policy {tool:26s} declared_modes={sorted(str(m) for m in modes)}')
        if not modes <= TIER_MODES[tier]:
            bad[tool] = sorted(str(m) for m in modes)
    if bad:
        raise SystemExit(f'ARM/TIER MISMATCH: tier {tier} admits {sorted(TIER_MODES[tier])}, '
                         f'arms declare {bad}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', type=int, choices=sorted(LEVEL_BY_TIER), required=True)
    ap.add_argument('--tools', nargs='+', required=True)
    ap.add_argument('--leaky', nargs='*', default=[])
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--metrics', default=','.join(sbs.DEFAULT_METRICS),
                    help='forwarded to score_by_source.py; add medae for the median')
    ap.add_argument('--master-cells', dest='master_cells', default=None,
                    help='forwarded to score_by_source.py: the frozen master cell set')
    ap.add_argument('--allow-anticorrelated', action='store_true',
                    help='forwarded to score_by_source.py: score a master set that still '
                         'carries the anticorrelated channels (the without-exclusions check)')
    ap.add_argument('--out', required=True)                # never default onto a shipped file
    ns = ap.parse_args()

    if Path(ns.out).exists():
        raise SystemExit(f'refusing to overwrite existing {ns.out}')

    with open(common.OUT_ROOT / 'fold6_tasks.tsv') as fh:
        tasks = {t['structure_id']: t for t in csv.DictReader(fh, delimiter='\t')}
    print(f'TIER {ns.tier}  ({len(tasks)} structure instances, OUT_ROOT={common.OUT_ROOT})')
    check_arm_policy(ns.tools, ns.leaky, ns.tier, tasks)

    # The tier sets one thing: which self-exclusion predicate the scorer applies. It
    # takes that predicate as an argument, so nothing here rebinds a name or rewrites
    # argv to reach inside it.
    fwd = ['--tools', *ns.tools, '--boot', str(ns.boot), '--seed', str(ns.seed),
           '--metrics', ns.metrics, '--out', ns.out]
    if ns.master_cells:
        fwd += ['--master-cells', ns.master_cells]
    if ns.allow_anticorrelated:
        fwd += ['--allow-anticorrelated']
    if ns.leaky:
        fwd += ['--leaky', *ns.leaky]
    sbs.main(fwd, verify=lambda tool, sid, task, exc: verify_tiered(tool, sid, task, exc,
                                                                   ns.tier))


if __name__ == '__main__':
    main()
