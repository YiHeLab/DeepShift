"""The ``deepshift`` command-line entry point.

  deepshift predict          predict shifts from a PDB (cache-free ensemble)
  deepshift train            train an atom-transformer fold
  deepshift evaluate         score held-out predictions per nucleus
  deepshift intervals fit|evaluate   fit on fold 0, then evaluate once on fold 6
  deepshift data build       build the training cache (offline replay over frozen snapshots)
  deepshift protocol         print the frozen protocol configuration and gates

Individual build stages are also runnable directly, e.g. ``python -m deepshift.data.fetch``,
``... .reref``, ``... .folds``, the same programs ``data build`` orchestrates.
"""
from __future__ import annotations

import argparse

from .calibration import add_interval_arguments
from .data.build import add_build_arguments
from .evaluate import add_evaluate_arguments
from .final_protocol import add_protocol_arguments
from .prediction import add_predict_arguments
from .training import add_train_arguments


def build_parser() -> argparse.ArgumentParser:
    """The complete command surface, so a command can be parsed without being run."""
    parser = argparse.ArgumentParser(prog="deepshift")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="build or verify the training dataset")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    add_build_arguments(data_sub.add_parser(
        "build", help="build the training cache (offline replay over frozen snapshots)"))

    add_train_arguments(sub.add_parser("train", help="train an atom-transformer fold"))
    add_predict_arguments(sub.add_parser("predict", help="predict shifts from a PDB (cache-free)"))
    add_evaluate_arguments(sub.add_parser(
        "evaluate", help="score held-out predictions per nucleus (dumps, or a cache fold)"))
    intervals = sub.add_parser("intervals", help="fit or evaluate prediction intervals")
    add_interval_arguments(intervals.add_subparsers(dest="interval_command", required=True))
    add_protocol_arguments(sub.add_parser(
        "protocol", help="print the frozen protocol configuration and gates"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Handlers return rich objects for programmatic use, but console-script entry
    # points pass this return value to ``sys.exit``, so ``main`` returns the exit status.
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
