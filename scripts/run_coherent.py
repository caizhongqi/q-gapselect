#!/usr/bin/env python3
"""Run exact-state coherent Q-GapSelect experiments and executable baselines."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from qgapselect.coherent_experiments import (
    COHERENT_BACKEND,
    COHERENT_CLAIM_STATUS,
    load_coherent_config,
    run_coherent_experiments,
    write_coherent_report,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "coherent.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "coherent_results.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="coherent JSON config (default: configs/coherent.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path (default: artifacts/coherent_results.json)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="override the configured repetitions per fixed scenario",
    )
    parser.add_argument("--seed", type=int, help="override the master seed")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="run only this scenario; may be supplied more than once",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_coherent_config(
            args.config,
            trials_override=args.trials,
            seed_override=args.seed,
            scenario_names=args.scenarios,
        )
        report = run_coherent_experiments(config)
        output = write_coherent_report(report, args.output)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"coherent experiment failed: {error}") from error

    records = report["raw_execution_records"]
    sys.stdout.write(
        f"wrote {len(records)} execution records to {output}\n"
        f"backend={COHERENT_BACKEND}\n"
        f"claim_status={COHERENT_CLAIM_STATUS}\n"
        "Resources come from executed exact-state circuit IR or explicitly labelled "
        "baselines; candidate theory is stored separately.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
