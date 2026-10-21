#!/usr/bin/env python3
"""IDP8 ensemble roster.

All 14 published IDP8 ensembles, so the nine-row denominator behind the
downstream-applications panel is visible and not implied. The roster and the
published per-nucleus accuracies come from Nagy et al. 2024; the DeepShift column
is read from the scorer's audited output, the file Figure S11 draws. Nothing is re-run
and no ensemble is re-scored.
"""

from __future__ import annotations

import json

import si_common

# The audited scores, read from the same root the downstream-applications figure reads
# them from, and built into the same per-nucleus blocks its panel (a) records.
IDP8 = si_common.root("DEEPSHIFT_IDP8_ROOT")
AUDITED = IDP8 / "scores_audited.json"
SCORER = IDP8 / "score.py"

OUT = si_common.OUT
OUT_SNIPPET = OUT / "SNIPPET_idp8_roster.tex"

# Nagy et al. 2024, Table II (roster) and Table III (published shift accuracy),
# transcribed from the paper text on disk. published = (CA, CB, C'), ppm RMSD.
ROSTER = [
    ("A", "asyn-A", "asyn", "PED:00024", (0.36, 0.66, 0.66)),
    ("A", "mevn-A", "mevn", "PED:00020", (0.28, 0.37, 0.40)),
    ("A", "sic1-A", "sic1", "PED:00160", (1.30, 0.49, None)),
    ("A", "tk18-A", "tk18", "PED:00192", (0.56, 1.01, 0.52)),
    ("B", "mevn-B", "mevn", "PED:00233", (0.34, 0.38, 0.52)),
    ("B", "actr-B", "actr", "PED:00230", (0.35, 0.35, 0.46)),
    ("B", "cbpn-B", "cbpn", "PED:00228", (0.69, 0.38, 0.62)),
    ("B", "p53t-B", "p53t", "PED:00229", (0.36, 0.34, 0.53)),
    ("B", "rsp8-B", "rsp8", "PED:00231", (0.41, None, 0.60)),
    ("C", "mevn-C", "mevn", "PED:00234", (0.34, 0.32, 0.37)),
    ("C", "actr-C", "actr", "PED:00237", (0.28, 0.32, 0.39)),
    ("C", "cbpn-C", "cbpn", "PED:00235", (0.67, 0.40, 0.62)),
    ("C", "p53t-C", "p53t", "PED:00236", (0.28, 0.32, 0.47)),
    ("C", "rsp8-C", "rsp8", "PED:00238", (0.57, None, 0.64)),
]

# Where the measured shifts live. The paper names BMRB entries for six proteins and
# says the other two sets were supplied privately by the groups that measured them.
SHIFT_SOURCE = {
    "asyn": ("19257", True),
    "sic1": ("16657", True),
    "tk18": ("19253", True),
    "actr": ("15397", True),
    "cbpn": ("16363", True),
    "p53t": ("17760", True),
    "mevn": (None, False),
    "rsp8": (None, False),
}

PROTEIN_NAMES = {
    "asyn": "alpha-synuclein",
    "mevn": "measles virus nucleoprotein tail domain",
    "sic1": "yeast Sic1 N-terminal targeting domain",
    "tk18": "human tau K18 fragment",
    "actr": "ACTR activation domain 1",
    "cbpn": "CBP nuclear coactivator binding domain",
    "p53t": "p53 N-terminal transactivation domain",
    "rsp8": "RS-repeat peptide",
}

NUCLEI = [("CA", "C$\\alpha$"), ("CB", "C$\\beta$"), ("C", "C$'$")]


def fmt(value, places: int) -> str:
    return "---" if value is None else f"{value:.{places}f}"


def scored_blocks(audited_rows) -> dict:
    """Per nucleus, every ensemble with both a DeepShift and a published RMSD: the rule
    figures/fig_downstream_applications.py draws panel (a) by."""
    blocks = {}
    for nucleus, _ in NUCLEI:
        cells = {r["label"]: {"published_spartaplus_rmsd_ppm": float(r[f"{nucleus}_published"]),
                              "deepshift_rmsd_ppm": float(r[f"{nucleus}_rmsd"])}
                 for r in audited_rows
                 if r.get(f"{nucleus}_rmsd") is not None
                 and r.get(f"{nucleus}_published") is not None
                 and r[f"{nucleus}_rmsd"] == r[f"{nucleus}_rmsd"]}       # not NaN
        blocks[nucleus] = {
            "n_ensembles": len(cells),
            "deepshift_wins": sum(c["deepshift_rmsd_ppm"] < c["published_spartaplus_rmsd_ppm"]
                                  for c in cells.values()),
            "per_ensemble": cells}
    return blocks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idp8 = {"nuclei": scored_blocks(json.loads(AUDITED.read_text()))}
    # The table's mean row is the plain mean over the scored ensembles' cells.
    for block in idp8["nuclei"].values():
        cells = list(block["per_ensemble"].values())
        block["deepshift_mean_rmsd_ppm"] = sum(c["deepshift_rmsd_ppm"] for c in cells) / len(cells)
        block["published_spartaplus_mean_rmsd_ppm"] = (
            sum(c["published_spartaplus_rmsd_ppm"] for c in cells) / len(cells))
    with open(AUDITED) as fh:
        audited = {row["label"]: row for row in json.load(fh)}

    # Independent check 1: the published values transcribed from the paper must match
    # the published table the scorer used.
    scorer_text = SCORER.read_text()
    published_from_scorer = {}
    capture = False
    for line in scorer_text.splitlines():
        if line.startswith("PUBLISHED"):
            capture = True
            body = ""
        if capture:
            body += line
            if line.rstrip().endswith("}"):
                capture = False
                published_from_scorer = eval(body.split("=", 1)[1].strip())
                break
    mismatches = [
        label
        for _, label, _, _, published in ROSTER
        if published_from_scorer.get(label) != published
    ]
    assert not mismatches, mismatches
    assert len(published_from_scorer) == 14, len(published_from_scorer)

    # Check 2: every DeepShift value is the scorer's own audited output, and every
    # published value the scorer's table, row by row.
    mismatched = []
    for nucleus, _ in NUCLEI:
        for label, cell in idp8["nuclei"][nucleus]["per_ensemble"].items():
            mine = audited[label].get(f"{nucleus}_rmsd")
            if mine is None or abs(mine - cell["deepshift_rmsd_ppm"]) > 1e-12:
                mismatched.append((nucleus, label))
            if abs(cell["published_spartaplus_rmsd_ppm"] - dict(
                zip(("CA", "CB", "C"), published_from_scorer[label])
            )[nucleus]) > 1e-12:
                mismatched.append((nucleus, label, "published"))
    assert not mismatched, mismatched

    scored = {
        nucleus: idp8["nuclei"][nucleus]["per_ensemble"] for nucleus, _ in NUCLEI
    }
    rows = []
    for group, label, protein, ped, published in ROSTER:
        bmrb, public = SHIFT_SOURCE[protein]
        row = {
            "ensemble": label,
            "protein": protein,
            "protein_name": PROTEIN_NAMES[protein],
            "group": group,
            "ped_code": ped,
            "shifts_publicly_deposited": public,
            "bmrb_entry": bmrb,
            "scored_by_deepshift": label in scored["CA"],
        }
        for nucleus, _ in NUCLEI:
            index = {"CA": 0, "CB": 1, "C": 2}[nucleus]
            row[f"{nucleus}_published_spartaplus_rmsd_ppm"] = published[index]
            cell = scored[nucleus].get(label)
            row[f"{nucleus}_deepshift_rmsd_ppm"] = (
                cell["deepshift_rmsd_ppm"] if cell else None
            )
        rows.append(row)

    assert sum(r["shifts_publicly_deposited"] for r in rows) == 9
    assert sum(r["scored_by_deepshift"] for r in rows) == 9
    for r in rows:
        assert r["shifts_publicly_deposited"] == r["scored_by_deepshift"], r["ensemble"]

    # ---- table -------------------------------------------------------------
    banner = "\n".join([
        "% NUMBERS: the roster, the group labels and the published accuracies are",
        "%   transcribed from Nagy et al., Applied Spectroscopy 78(9):897-911 (2024),",
        "%   Tables II and III, from the copy on disk. The DeepShift column is read",
        "%   from the scorer's audited output; no ensemble was re-scored.",
    ])
    lines = [
        "\\begin{table}[H]\\centering\\small",
        "\\caption{}\\label{tab:idp8_roster}",
        "\\setlength{\\tabcolsep}{4.5pt}",
        "\\begin{tabular}{llcrrrrrr}",
        "\\toprule",
        " & & & \\multicolumn{2}{c}{C$\\alpha$} & \\multicolumn{2}{c}{C$\\beta$} & "
        "\\multicolumn{2}{c}{C$'$} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        "Ensemble & Group & BMRB entry & DeepShift & SPARTA+ & DeepShift & SPARTA+ & "
        "DeepShift & SPARTA+ \\\\",
        " & & & \\multicolumn{6}{c}{root-mean-square deviation, ppm} \\\\",
        "\\midrule",
    ]
    for row in rows:
        shifts = row["bmrb_entry"] or "none"
        cells = []
        for nucleus, _ in NUCLEI:
            cells.append(fmt(row[f"{nucleus}_deepshift_rmsd_ppm"], 3))
            cells.append(fmt(row[f"{nucleus}_published_spartaplus_rmsd_ppm"], 2))
        lines.append(
            f"{row['ensemble']} & {row['group']} & {shifts} & " + " & ".join(cells) + " \\\\"
        )
    lines.append("\\midrule")
    mean_cells = []
    for nucleus, _ in NUCLEI:
        block = idp8["nuclei"][nucleus]
        mean_cells.append(f"{block['deepshift_mean_rmsd_ppm']:.3f}")
        mean_cells.append(f"{block['published_spartaplus_mean_rmsd_ppm']:.2f}")
    lines.append(
        "\\multicolumn{3}{l}{Mean, scored rows} & " + " & ".join(mean_cells) + " \\\\"
    )
    counts = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{idp8['nuclei'][n]['deepshift_wins']} of "
        f"{idp8['nuclei'][n]['n_ensembles']}}}"
        for n, _ in NUCLEI
    )
    lines.append("\\multicolumn{3}{l}{DeepShift lower in} & " + counts + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    si_common.emit("si_idp8_roster", "\n".join(lines), script="build_idp8_roster.py",
                   banner=banner)

    # ---- splice snippet ----------------------------------------------------
    names = "; ".join(f"{code} is {name}" for code, name in PROTEIN_NAMES.items())
    cb = idp8["nuclei"]["CB"]
    snippet = f"""% Splice block for SI_v3.tex -- paste where the IDP8 roster belongs, next to the
% downstream-applications figure.
% The \\capneeded argument is a bare fact list, not caption prose. The caption in
% si_idp8_roster.tex is empty and needs a short title from the author.

\\input{{si/tables/si_idp8_roster}}
\\cohort{{All 14 published IDP8 ensembles; ppm root-mean-square deviation from the
measured backbone shifts; raw residuals, nothing mean-centred; the published column is
the source paper's own Table III and the DeepShift column is the scorer's audited
output, unrecomputed.}}
\\capneeded{{Facts to assert: the published data set contains 14 structural ensembles of
eight disordered proteins, grouped A, B and C by how they were built. Nine of the 14
have measured backbone shifts in the BMRB and can be scored from public data; the
other five are the three measles virus nucleoprotein tail ensembles and the two
RS-repeat peptide ensembles, whose shifts were supplied privately to the original
authors and never deposited. That is the whole reason the accuracy panel has a
nine-row denominator. A dash means the quantity does not exist for that row: in a
DeepShift column because the shifts are not public, or because that deposition carries
no shifts for the nucleus, which is why sic1-A has no carbonyl carbon entry on either
side; in a SPARTA+ column because the paper reports no value there. On the nine scored
rows DeepShift's mean C$\\beta$ deviation is
{cb['deepshift_mean_rmsd_ppm']:.3f} ppm against the published
{cb['published_spartaplus_mean_rmsd_ppm']:.3f} ppm, lower on
{cb['deepshift_wins']} of {cb['n_ensembles']} rows. Ensemble codes: {names}. Note for
the record that the source paper's text cites BMRB 17660 for the p53 transactivation
domain, which resolves to an unrelated protein; the entry used here and printed in the
table is 17760.}}
"""
    OUT_SNIPPET.write_text(snippet)

    print(f"rows {len(rows)}  public {sum(r['shifts_publicly_deposited'] for r in rows)}")
    print(f"wrote {OUT_SNIPPET}")


if __name__ == "__main__":
    main()
