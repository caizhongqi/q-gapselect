#!/usr/bin/env python3
"""Merge fixed-subset packed-capacity validation components."""

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
    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_directory.rglob("component.json"))
    ]
    expected_n = sorted(int(value) for value in config["n_values"])
    observed_n = sorted(int(artifact["n"]) for artifact in artifacts)
    if observed_n != expected_n:
        raise ValueError(
            f"fixed-subset component grid mismatch: expected={expected_n}, observed={observed_n}"
        )
    if {str(artifact["artifact_type"]) for artifact in artifacts} != {
        "qcollide_fixed_subset_capacity_validation_component_v3"
    }:
        raise ValueError("unexpected fixed-subset component artifact type")

    rows = [row for artifact in artifacts for row in artifact["rows"]]
    epsilon = float(config["relative_epsilon"])
    exact_inversion_success = all(
        int(row["exact_inversion_capacity"]) == int(row["true_capacity"])
        for row in rows
    )
    theorem_precision_success = all(
        float(row["worst_relative_error_at_registered_probability_precision"])
        <= epsilon + 1e-12
        for row in rows
    )
    bounds_cover = all(
        float(row["bonferroni_lower"])
        <= float(row["exact_survival_probability"]) + 1e-12
        and float(row["exact_survival_probability"])
        <= float(row["union_upper"]) + 1e-12
        for row in rows
    )
    monte_carlo_relative = np.asarray(
        [float(row["monte_carlo_relative_capacity_error"]) for row in rows],
        dtype=float,
    )
    monte_carlo_probability = np.asarray(
        [float(row["monte_carlo_probability_error"]) for row in rows],
        dtype=float,
    )
    artifact = {
        "artifact_type": "qcollide_fixed_subset_capacity_validation_main_v3",
        "schema_version": 1,
        "row_count": len(rows),
        "n_values": expected_n,
        "relative_epsilon": epsilon,
        "rows": sorted(
            rows,
            key=lambda row: (
                int(row["n"]),
                int(row["capacity_scale"]),
                float(row["capacity_multiplier"]),
            ),
        ),
        "summary": {
            "mean_monte_carlo_probability_error": float(
                monte_carlo_probability.mean()
            ),
            "maximum_monte_carlo_probability_error": float(
                monte_carlo_probability.max()
            ),
            "mean_monte_carlo_relative_capacity_error": float(
                monte_carlo_relative.mean()
            ),
            "maximum_monte_carlo_relative_capacity_error": float(
                monte_carlo_relative.max()
            ),
            "monte_carlo_cells_within_registered_relative_epsilon_fraction": float(
                np.mean(monte_carlo_relative <= epsilon)
            ),
        },
        "gates": {
            "complete_n_grid": True,
            "all_exact_inversions_recover_capacity": exact_inversion_success,
            "all_bonferroni_bounds_cover_exact_survival": bounds_cover,
            "all_registered_probability_perturbations_respect_relative_error": (
                theorem_precision_success
            ),
        },
        "claim_boundary": {
            **config["claim_boundary"],
            "monte_carlo_is_quantum_execution": False,
            "monte_carlo_validates_probability_and_inversion_not_query_speed": True,
            "complexity_exponent_is_not_fitted_from_generated_proxy_values": True,
        },
    }
    if not exact_inversion_success or not bounds_cover or not theorem_precision_success:
        raise RuntimeError(f"fixed-subset theorem-validation gate failed: {artifact['gates']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gates": artifact["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
