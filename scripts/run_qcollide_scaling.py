#!/usr/bin/env python3
"""Run the frozen multi-scale Q-COLLIDE F0--F4 calibration campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.scaling import compact_summary, run_scaling_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qcollide_scaling.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qcollide_scaling_diagnostic.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = run_scaling_campaign(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(compact_summary(artifact), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
