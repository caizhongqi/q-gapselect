#!/usr/bin/env python3
"""Merge unknown-K fixed-subset scale-scan components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    components = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_directory.rglob("component.json"))
    ]
    expected_n = {int(value) for value in config["n_values"]}
    observed_n = {int(component["n"]) for component in components}
    if observed_n != expected_n:
        raise ValueError(f"incomplete N grid: expected={sorted(expected_n)}, got={sorted(observed_n)}")
    rows = [row for component in components for row in component["rows"]]
    errors = np.asarray([float(row["relative_capacity_error"]) for row in rows], dtype=float)
    scale_ratios = np.asarray([float(row["scale_to_capacity_ratio"]) for row in rows], dtype=float)
    stages = np.asarray([len(row["observations"]) for row in rows], dtype=float)
    target = float(config["target_relative_error"])
    success_fraction = float(np.mean(errors <= target))
    artifact = {
        "artifact_type": "qcollide_fixed_subset_unknown_k_main_table",
        "schema_version": 1,
        "row_count": len(rows),
        "rows": rows,
        "summary": {
            "mean_relative_capacity_error": float(errors.mean()),
            "median_relative_capacity_error": float(np.median(errors)),
            "p95_relative_capacity_error": float(np.quantile(errors, 0.95)),
            "maximum_relative_capacity_error": float(errors.max()),
            "success_fraction_at_target_error": success_fraction,
            "median_selected_scale_to_capacity_ratio": float(np.median(scale_ratios)),
            "p95_selected_scale_to_capacity_ratio": float(np.quantile(scale_ratios, 0.95)),
            "mean_scan_stage_count": float(stages.mean()),
            "maximum_scan_stage_count": int(stages.max()),
        },
        "gates": {
            "complete_n_grid": True,
            "target_relative_error": target,
            "minimum_success_fraction": float(config["minimum_success_fraction"]),
            "empirical_success_gate": success_fraction
            >= float(config["minimum_success_fraction"]),
        },
        "claim_boundary": config["claim_boundary"],
        "experiment_config": str(args.config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gates": artifact["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
