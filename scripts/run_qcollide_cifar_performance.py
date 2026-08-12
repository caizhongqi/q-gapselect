#!/usr/bin/env python3
"""Replay one CIFAR model and record full checkpoint classification accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.performance_matched_cifar import (
    run_cifar_performance_component,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qcollide_cifar10_topology_pilot.json"),
    )
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = run_cifar_performance_component(
        config,
        architecture=args.architecture,
        model_seed=args.model_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "architecture": args.architecture,
                "model_seed": args.model_seed,
                "checkpoints": len(artifact["rows"]),
                "final_accuracy": artifact["rows"][-1]["evaluation_accuracy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
