#!/usr/bin/env python3
"""Build a performance-matched CIFAR functional-collision main table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.cifar_main_results import performance_matched_cifar_main_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--performance", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    topology = json.loads(args.topology.read_text(encoding="utf-8"))
    performance = json.loads(args.performance.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = performance_matched_cifar_main_table(
        topology,
        performance,
        target_accuracy=float(config["performance_match_target"]),
        maximum_accuracy_mismatch=float(config["performance_match_tolerance"]),
        visible_ranks=[int(value) for value in config["main_visible_ranks"]],
    )
    artifact["experiment_config"] = str(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gates": artifact["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
