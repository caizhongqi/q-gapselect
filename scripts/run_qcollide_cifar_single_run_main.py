#!/usr/bin/env python3
"""Train once, select a matched CIFAR checkpoint, and measure topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.cifar100_prepared import install_prepared_cifar100_adapter
from qgapselect.qcollide.cifar_single_run_selected import (
    run_cifar_single_run_selected_component,
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
    artifact = run_cifar_single_run_selected_component(
        config,
        architecture=args.architecture,
        model_seed=args.model_seed,
    )
    artifact["experiment_config"] = str(args.config)
    artifact["dataset_backend"] = "prepared:huggingface:uoft-cs/cifar100"
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
                "selected_performance": artifact["selected_performance"],
                "dataset_backend": artifact["dataset_backend"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
