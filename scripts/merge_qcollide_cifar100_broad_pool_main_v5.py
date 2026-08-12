#!/usr/bin/env python3
"""Merge CIFAR-100 broad-pool main-v5 components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.cifar_broad_pool_results import (
    merge_broad_pool_cifar_components,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_directory.rglob("component.json"))
    ]
    merged = merge_broad_pool_cifar_components(
        artifacts,
        dataset=str(config["dataset"]),
        architectures=[str(value) for value in config["architectures"]],
        model_seeds=[int(value) for value in config["model_seeds"]],
        target_accuracy=float(config["performance_match_target"]),
        maximum_accuracy_mismatch=float(config["performance_match_tolerance"]),
        visible_ranks=[int(value) for value in config["main_visible_ranks"]],
        minimum_unique_fine_classes=int(config["minimum_unique_fine_classes"]),
    )
    merged["experiment_config"] = str(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gates": merged["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
