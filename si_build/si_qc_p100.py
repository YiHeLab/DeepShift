#!/usr/bin/env python3
"""The p100 exclusion ranges, as an explicit per-(residue, nucleus) interval table.

For each (residue, nucleus) the interval is the span of deposited values over which the
calibration models made at least one prediction within k times that column's held-out MAE;
Arg N-epsilon carries the observed-value interval. Every row is read through
``deepshift.quality_control.P100Policy`` from the fitted cutoff file named by
``SI_P100_CUTOFFS``, and the table counts, per observed population of the build's cache,
how many deposited cells the interval removes. A cell carrying an ambiguity-set code
(``deepshift.loss.AMBIGUITY_SET_CODES``) is observed but never tested by the fence
(``quality_control``), so it is never counted as removed.

Usage: DEEPSHIFT_RUN=... SI_P100_CUTOFFS=... python3 si_qc_p100.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from deepshift.loss import AMBIGUITY_SET_CODES
from deepshift.quality_control import P100Policy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import si_common as C

CACHE = C.CACHE
OUT = C.OUT

GREEK = {'a':r'\alpha','b':r'\beta','g':r'\gamma','d':r'\delta','e':r'\varepsilon',
         'z':r'\zeta','h':r'\eta'}

def pretty(col: str) -> str:
    s = col[:-6]
    if s == 'c':
        return "C$'$"
    if s in ('h','n','ca','cb'):
        return {'h':'H$^{\\mathrm{N}}$','n':'N','ca':r'C$\alpha$','cb':r'C$\beta$'}[s]
    el = s[0].upper()
    rest = s[1:]
    if rest and rest[0] in GREEK:
        return f"{el}${GREEK[rest[0]]}${rest[1:]}" if len(rest) > 1 else f"{el}${GREEK[rest[0]]}$"
    return el + rest


def measure(policy: P100Policy, cache: Path):
    """Count the build's observed cells per (residue, nucleus) and how many the rule
    removes; returns ``(spans, observed, removed)``.

    A pair appears in ``spans`` exactly when the policy states an accepted interval for
    it, so the published table is the policy's own row set restricted to what the cache
    observes. Pairs the policy declines -- chemically impossible ones, and any the fitted
    authority does not cover -- are counted in ``observed`` and published nowhere.
    """
    cfg = json.loads((cache / 'fold_1/config.json').read_text())
    columns = {c: i for i, c in enumerate(cfg['shift_cols'])}

    obs, removed, spans = {}, {}, {}
    for fold in sorted(int(d.name.split('_')[-1]) for d in cache.glob('fold_*') if d.is_dir()):
        sd = cache / f'fold_{fold}' / 'structural'
        sh = np.load(sd / 'shifts.npy', mmap_mode='r')
        mk = np.load(sd / 'shift_mask.npy', mmap_mode='r')
        amb = np.load(sd / 'ambiguity.npy', mmap_mode='r')
        idm = json.loads((sd / 'id_maps.json').read_text())
        aa = np.array([idm['comp_id'][str(i)] for i in range(sh.shape[0])])
        for col, j in columns.items():
            m = np.asarray(mk[:, j]).astype(bool)
            if not m.any():
                continue
            vals = np.asarray(sh[:, j])
            tested = m & ~np.isin(np.asarray(amb[:, j]), AMBIGUITY_SET_CODES)
            for r in np.unique(aa[m]):
                sel = m & (aa == r)
                n = int(sel.sum())
                if not n:
                    continue
                key = f'{r}:{col}'
                obs[key] = obs.get(key, 0) + n
                accepted = policy.accepted_range(str(r), col)
                if accepted is None:
                    continue
                spans[key] = accepted
                lo, hi = accepted
                v = vals[sel & tested]
                removed[key] = removed.get(key, 0) + int(((v < lo) | (v > hi)).sum())
    return spans, obs, removed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spans, obs, removed = measure(C.p100_policy(), CACHE)

    covered = sum(n for k, n in obs.items() if k in spans)
    total = sum(obs.values())
    drop = sum(removed.values())
    rows = sorted(spans, key=lambda k: (-removed.get(k, 0), -obs[k]))

    hdr = (r'Residue & Nucleus & Accepted lo & Accepted hi & Observed & Would remove \\')
    L = []
    for k in rows:
        r, col = k.split(':')
        lo, hi = spans[k]
        L.append(f'{r} & {pretty(col)} & {lo:.2f} & {hi:.2f} & {obs[k]:,} & {removed.get(k,0):,} \\\\')
    L.append(r'\bottomrule')
    tex = C.table('% The facts this caption needs are in the comment beneath the table.',
                  'tab:qc_p100', '\n'.join(L), longtable='llrrrr', head=hdr,
                  caption='__CAPTION__')

    C.emit('si_qc_p100', tex, script='si_qc_p100.py')
    print(f'{len(rows)} pairs | covered {covered:,}/{total:,} | would remove {drop:,} '
          f'({drop/covered*100:.3f}%)')


if __name__ == '__main__':
    main()
