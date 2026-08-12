#!/usr/bin/env python3
"""Run one calibration-selected CIFAR repair v2 component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.cifar_repair_calibrated import (
    run_cifar_calibrated_repair_component,
)
from qgapselect.qcollide.cifar_repair_data import install_prepared_cifar10_adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    install_prepared_cifar10_adapter()
    artifact = run_cifar_calibrated_repair_component(
        config,
        architecture=args.architecture,
        model_seed=args.model_seed,
    )
    artifact["experiment_config"] = str(args.config)
    artifact["dataset_backend"] = str(config["dataset_backend"])
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
                "visible_rank": artifact["operating_point"]["visible_rank"],
                "epsilon_multiplier": artifact["operating_point"]["epsilon_multiplier"],
                "heldout_capacity_fraction": artifact["heldout_baseline_profile"]["nominal"][
                    "capacity_fraction"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
