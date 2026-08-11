#!/usr/bin/env python3
"""Run one fixed-N query-budget collision-capacity scaling component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.query_budget_scaling import evaluate_query_budget_component


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", choices=("matching", "overlap"), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    allowed_n = (
        config["matching_n_values"]
        if args.panel == "matching"
        else config["overlap_n_values"]
    )
    if args.n not in [int(value) for value in allowed_n]:
        raise ValueError(f"n={args.n} is not configured for panel={args.panel}")

    artifact = evaluate_query_budget_component(
        panel=args.panel,
        n=args.n,
        matching_densities=tuple(float(value) for value in config["matching_densities"]),
        budget_fractions=tuple(float(value) for value in config["budget_fractions"]),
        graph_seeds=tuple(int(value) for value in config["graph_seeds"]),
        sampling_trials=int(config["sampling_trials"]),
        additive_epsilon=float(config["additive_epsilon"]),
        relative_epsilon=float(config["relative_epsilon"]),
        success_probability=float(config["target_success_probability"]),
        seed=int(config["master_seed"]),
    )
    artifact["experiment_config"] = str(args.config)
    artifact["oracle"] = str(config["oracle"])
    artifact["query_cost"] = str(config["query_cost"])
    artifact["pairwise_collision_comparison_after_endpoint_query"] = str(
        config["pairwise_collision_comparison_after_endpoint_query"]
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
                "panel": args.panel,
                "n": args.n,
                "cell_count": len(artifact["cells"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
