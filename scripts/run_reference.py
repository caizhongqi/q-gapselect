#!/usr/bin/env python3
"""Run the multi-seed Q-GapSelect analytic-reference regression pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from qgapselect.reference_experiments import (
    REFERENCE_BACKEND,
    REFERENCE_CLAIM_STATUS,
    load_reference_config,
    run_reference_experiments,
    write_reference_report,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "reference.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "reference_results.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="reference JSON config (default: configs/reference.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON artifact path (default: artifacts/reference_results.json)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        help=(
            "explicitly override preregistered repetitions for a local "
            "diagnostic run"
        ),
    )
    parser.add_argument("--seed", type=int, help="override the master seed")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="run only this named scenario; may be supplied more than once",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_reference_config(
            args.config,
            trials_override=args.trials,
            seed_override=args.seed,
            scenario_names=args.scenarios,
        )
        report = run_reference_experiments(config)
        output = write_reference_report(report, args.output)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"reference experiment configuration error: {error}") from error

    sys.stdout.write(
        f"wrote {len(report['raw_records'])} trial records to {output}\n"
        f"backend={REFERENCE_BACKEND}\n"
        f"claim_status={REFERENCE_CLAIM_STATUS}\n"
        "This artifact is a simulator regression, not coherent batch execution "
        "or evidence of quantum acceleration.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
