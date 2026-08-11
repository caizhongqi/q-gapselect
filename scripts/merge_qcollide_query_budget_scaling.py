#!/usr/bin/env python3
"""Merge query-budget collision-capacity scaling v2 components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.query_budget_results import merge_query_budget_components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_directory.rglob("*.json"))
    ]
    merged = merge_query_budget_components(
        artifacts,
        matching_n_values=[int(value) for value in config["matching_n_values"]],
        overlap_n_values=[int(value) for value in config["overlap_n_values"]],
        matching_densities=[float(value) for value in config["matching_densities"]],
    )
    merged["experiment_config"] = str(args.config)
    merged["oracle"] = str(config["oracle"])
    merged["additive_epsilon"] = float(config["additive_epsilon"])
    merged["relative_epsilon"] = float(config["relative_epsilon"])
    merged["target_success_probability"] = float(config["target_success_probability"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gates": merged["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
