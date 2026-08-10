#!/usr/bin/env python3
"""Run one or more real-image Q-COLLIDE CNN/Transformer components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.vision import (
    compact_real_vision_summary,
    run_real_vision_campaign,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qcollide_real_vision_digits.json"),
    )
    parser.add_argument("--architecture", choices=("cnn", "tiny_vit"))
    parser.add_argument("--model-seed", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qcollide_real_vision_diagnostic.json"),
    )
    parser.add_argument("--summary", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.architecture is not None:
        config["architectures"] = [args.architecture]
    if args.model_seed is not None:
        if args.model_seed < 0:
            raise ValueError("--model-seed must be non-negative")
        config["model_seeds"] = [args.model_seed]
    artifact = run_real_vision_campaign(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(
                compact_real_vision_summary(artifact),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "architectures": config["architectures"],
                "model_seeds": config["model_seeds"],
                "output": str(args.output),
                "row_count": len(artifact["rows"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
