#!/usr/bin/env python3
"""Merge calibration-selected CIFAR repair v2 components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.cifar_repair_calibrated_results import (
    merge_calibrated_repair_components,
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
    merged = merge_calibrated_repair_components(
        artifacts,
        architectures=[str(value) for value in config["architectures"]],
        model_seeds=[int(value) for value in config["model_seeds"]],
        primary_closure_rank=int(config["primary_closure_rank"]),
        primary_energy_budget_fraction=float(config["primary_energy_budget_fraction"]),
        maximum_accuracy_loss=float(config["maximum_accuracy_loss"]),
        maximum_energy_mismatch=float(config["maximum_energy_mismatch"]),
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
