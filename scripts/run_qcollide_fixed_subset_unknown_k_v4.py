#!/usr/bin/env python3
"""Run one N-slice of the unknown-K fixed-subset scale scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.fixed_subset_unknown_k import scan_unknown_capacity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    n_values = {int(value) for value in config["n_values"]}
    if args.n not in n_values:
        raise ValueError(f"n={args.n} is not registered")
    rows = []
    for denominator in [int(value) for value in config["capacity_density_denominators"]]:
        matching_size = max(1, args.n // denominator)
        for seed in [int(value) for value in config["replicate_seeds"]]:
            result = scan_unknown_capacity(
                args.n,
                matching_size,
                scan_trials=int(config["scan_trials_per_scale"]),
                estimation_trials=int(config["estimation_trials"]),
                seed=seed + 1009 * args.n + 17 * denominator,
                survival_low=float(config["survival_low"]),
                survival_high=float(config["survival_high"]),
            )
            result["capacity_density_denominator"] = denominator
            result["replicate_seed"] = seed
            rows.append(result)
    artifact = {
        "artifact_type": "qcollide_fixed_subset_unknown_k_component",
        "schema_version": 1,
        "n": args.n,
        "rows": rows,
        "experiment_config": str(args.config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "row_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
