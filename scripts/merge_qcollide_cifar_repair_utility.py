#!/usr/bin/env python3
"""Merge prospective CIFAR collision-repair components into the utility table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.cifar_repair_results import merge_cifar_repair_components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_directory.rglob("component.json"))
    ]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    merged = merge_cifar_repair_components(
        artifacts,
        architectures=[str(value) for value in config["architectures"]],
        model_seeds=[int(value) for value in config["model_seeds"]],
        closure_ranks=[int(value) for value in config["closure_ranks"]],
        primary_closure_rank=int(config["primary_closure_rank"]),
        maximum_accuracy_loss=float(config["maximum_accuracy_loss"]),
    )
    merged["experiment_config"] = str(args.config)
    merged["dataset_backend"] = str(config["dataset_backend"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gates": merged["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
