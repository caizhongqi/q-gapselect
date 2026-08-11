#!/usr/bin/env python3
"""Merge checkpoint-local CIFAR components into the final matched main table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.cifar_selected_results import merge_selected_cifar_components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_directory.rglob("*.json"))
    ]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = merge_selected_cifar_components(
        artifacts,
        dataset=str(config["dataset"]),
        architectures=[str(value) for value in config["architectures"]],
        model_seeds=[int(value) for value in config["model_seeds"]],
        target_accuracy=float(config["performance_match_target"]),
        maximum_accuracy_mismatch=float(config["performance_match_tolerance"]),
        visible_ranks=[int(value) for value in config["main_visible_ranks"]],
    )
    result["experiment_config"] = str(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
