#!/usr/bin/env python3
"""Fetch the Nagy IDP8 ensembles and their BMRB reference shifts.

Nagy et al., Appl Spectrosc 2024 (doi:10.1177/00037028241239977) publish per-nucleus
CA/CB/C' RMSDs against BMRB shifts for 14 IDP8 ensembles, back-calculated with SPARTA+
v2.6. Nine of the 14 can be scored from public shift deposits; the measles virus
nucleoprotein tail and RS-repeat shifts were supplied privately to Nagy et al. and are not
deposited, so those five ensembles (mevn-A/B/C, rsp8-B/C) are not fetched.

Writes ``<out>/ensembles/<label>_<PED id>_<ensemble>.pdb`` (multi-model PDB),
``<out>/shifts/<protein>_bmrb<id>.str`` (NMR-STAR) and ``<out>/manifest.json``, which
``run_inference.py`` and ``score.py`` read. Manifest paths are relative to ``<out>``.

The PED endpoint serves a gzipped tar whose single member is the multi-model PDB, with no
Content-Encoding header, so the gzip magic is sniffed and the archive unwrapped.

The paper's text gives BMRB 17660 for the p53 transactivation domain, which is an unrelated
protein (LmrA-NBD); the PED cross-reference of PED00229 and PED00236 gives 17760.

    python fetch.py --out <idp8 data directory>
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

PED_URL = ("https://deposition.proteinensemble.org/v1/entries/{ped}/ensembles/{ens}"
           "/ensemble-pdb")
BMRB_URL = "https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr{bmrb}/bmr{bmrb}_3.str"

# (ped, ensemble, label, protein, group). Group A ensembles were never refined against
# chemical shifts; B and C were refined against CA only, so CA is partly circular there
# while CB and C' were held out by the authors themselves.
ENSEMBLES = [
    ("PED00024", "e001", "asyn-A", "asyn", "A"),
    ("PED00160", "e001", "sic1-A", "sic1", "A"),
    ("PED00192", "e001", "tk18-A", "tk18", "A"),
    ("PED00228", "e001", "cbpn-B", "cbpn", "B"),
    ("PED00235", "e001", "cbpn-C", "cbpn", "C"),
    ("PED00229", "e001", "p53t-B", "p53t", "B"),
    ("PED00236", "e001", "p53t-C", "p53t", "C"),
    ("PED00230", "e001", "actr-B", "actr", "B"),
    ("PED00237", "e001", "actr-C", "actr", "C"),
]

# protein -> BMRB entry. asyn (19257) is an in-cell measurement.
SHIFTS = {
    "asyn": "19257",
    "sic1": "16657",
    "tk18": "19253",
    "actr": "15397",
    "cbpn": "16363",
    "p53t": "17760",
}


def get(url: str, dest: Path) -> int:
    """Download to ``dest`` as PDB or NMR-STAR text, unwrapping gzip and tar."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read()
    if not body:
        raise RuntimeError(f"empty body from {url}")

    if body[:2] == b"\x1f\x8b":
        raw = gzip.decompress(body)
        # Concatenate every member in name order rather than silently picking one, so a
        # multi-member archive cannot lose models.
        if raw[257:262] == b"ustar" or b"MODEL" not in raw[:4096]:
            with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
                members = sorted((m for m in archive.getmembers() if m.isfile()),
                                 key=lambda m: m.name)
                if not members:
                    raise RuntimeError(f"empty tar from {url}")
                raw = b"".join(archive.extractfile(m).read() for m in members)
        body = raw

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return len(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = {"ensembles": [], "shifts": [], "failures": []}

    for ped, ens, label, protein, group in ENSEMBLES:
        relative = Path("ensembles") / f"{label}_{ped}_{ens}.pdb"
        try:
            size = get(PED_URL.format(ped=ped, ens=ens), args.out / relative)
            text = (args.out / relative).read_text()
            models = text.count("\nMODEL ") + text.startswith("MODEL ")
            manifest["ensembles"].append(
                {"label": label, "ped": ped, "ensemble": ens, "protein": protein,
                 "group": group, "path": str(relative), "bytes": size, "models": models})
            print(f"{label:10s} {ped} {size / 1e6:7.2f} MB  {models:4d} models", flush=True)
        except Exception as exc:
            manifest["failures"].append(
                {"label": label, "ped": ped, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            print(f"{label:10s} {ped} FAILED {type(exc).__name__}: {str(exc)[:120]}",
                  flush=True)

    for protein, bmrb in SHIFTS.items():
        relative = Path("shifts") / f"{protein}_bmrb{bmrb}.str"
        try:
            size = get(BMRB_URL.format(bmrb=bmrb), args.out / relative)
            manifest["shifts"].append(
                {"protein": protein, "bmrb": bmrb, "path": str(relative), "bytes": size})
            print(f"{protein:10s} BMRB {bmrb} {size / 1e6:7.2f} MB", flush=True)
        except Exception as exc:
            manifest["failures"].append(
                {"protein": protein, "bmrb": bmrb,
                 "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            print(f"{protein:10s} BMRB {bmrb} FAILED {type(exc).__name__}: {str(exc)[:120]}",
                  flush=True)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{len(manifest['ensembles'])}/{len(ENSEMBLES)} ensembles, "
          f"{len(manifest['shifts'])}/{len(SHIFTS)} shift sets, "
          f"{len(manifest['failures'])} failures -> {args.out}")
    sys.exit(1 if manifest["failures"] else 0)


if __name__ == "__main__":
    main()
