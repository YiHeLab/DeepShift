#!/usr/bin/env python3
"""Build the SI interval table from the final fold-0 fit and fold-6 evaluation.

This is a presentation adapter. The scientific fit is performed by
``deepshift intervals fit`` and the one reserved evaluation by
``deepshift intervals evaluate``. This script cannot refit, reverse the fold direction,
or pool the two folds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import si_common

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from deepshift.calibration import (
    BACKBONE_NUCLEI,
    EVALUATION_SCHEMA,
    load_calibration,
)

HOUSE_ORDER = ("HA", "H", "CA", "CB", "C", "N")

BANNER = """% The variance mapping and every conformal multiplier were fitted on fold 0 only.
% Fold 6 was opened once to evaluate the frozen artifact. No fold-6 cell entered either fit.
% Backbone nuclei only; side-chain ensemble spread can include prochiral label assignment."""


def load_inputs(calibration_path: Path, evaluation_path: Path):
    calibration = load_calibration(calibration_path)
    evaluation = json.loads(evaluation_path.read_text())
    if evaluation.get("schema") != EVALUATION_SCHEMA:
        raise ValueError("evaluation artifact has the wrong schema")
    if evaluation.get("calibration_fold") != 0 or evaluation.get("evaluation_fold") != 6:
        raise ValueError("interval table requires fit fold 0 -> evaluation fold 6")
    if set(evaluation.get("results", {})) != set(BACKBONE_NUCLEI):
        raise ValueError("evaluation does not contain all six backbone nuclei")
    return calibration, evaluation


def emit_tex(calibration, evaluation):
    rows = []
    for nucleus in HOUSE_ORDER:
        fit = calibration.constants[nucleus]
        measured = evaluation["results"][nucleus]
        levels = measured["levels"]
        rows.append(
            f"{si_common.tex_nucleus(nucleus)} & "
            f"{si_common.tex_thousands(measured['n_evaluation_cells'])} & "
            f"{fit['floor_ppm']:.3f} & {fit['gain']:.3f} & "
            f"{100 * levels['0.68']['realized_coverage']:.1f} & "
            f"{100 * levels['0.90']['realized_coverage']:.1f} & "
            f"{100 * levels['0.95']['realized_coverage']:.1f} & "
            f"{levels['0.90']['median_half_width_ppm']:.3f} \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[H]\centering\small
\caption{{Prediction intervals fitted on fold 0 and evaluated once on fold 6.}}
\label{{tab:interval_calibration}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lrrrrrrr@{{}}}}
\toprule
 & & \multicolumn{{2}}{{c}}{{Variance mapping}} &
 \multicolumn{{3}}{{c}}{{Coverage on fold 6 (\%)}} & \\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-7}}
Nucleus & Cells & Floor (ppm) & Gain & 68\% & 90\% & 95\% &
 Median 90\% half-width (ppm) \\
\midrule
{body}
\bottomrule
\end{{tabular*}}
\end{{table}}
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args(argv)
    calibration, evaluation = load_inputs(args.calibration, args.evaluation)
    si_common.emit(
        "si_interval_calibration",
        emit_tex(calibration, evaluation),
        script=Path(__file__).name,
        banner=BANNER,
        placeholder_caption=False,
    )

if __name__ == "__main__":
    main()
