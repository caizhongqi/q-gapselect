#!/usr/bin/env python3
"""Run one CIFAR-100 broad-pool single-run main-v5 component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.cifar100_prepared import install_prepared_cifar100_adapter
from qgapselect.qcollide.cifar_single_run_broad_pool import (
    run_cifar_single_run_broad_pool_component,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    install_prepared_cifar100_adapter()
    artifact = run_cifar_single_run_broad_pool_component(
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
                "selected_checkpoint_epoch": artifact["selected_checkpoint_epoch"],
                "anchor_classes": artifact["selection"]["anchor_coverage"][
                    "unique_fine_classes"
                ],
                "candidate_classes": artifact["selection"]["candidate_coverage"][
                    "unique_fine_classes"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
