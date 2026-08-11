#!/usr/bin/env python3
"""Validate fixed-subset packed collision-capacity inversion on one N slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.fixed_subset_capacity import (
    choose_stable_subset_size,
    fixed_subset_quantum_query_proxy,
    fixed_subset_survival_bounds,
    fixed_subset_survival_probability,
    invert_fixed_subset_survival,
    simulate_fixed_subset_survival,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.n not in [int(value) for value in config["n_values"]]:
        raise ValueError("n is not registered")
    epsilon = float(config["relative_epsilon"])
    trials = int(config["monte_carlo_trials"])
    seed = int(config["fixture_seed"])
    rows = []
    for denominator_index, denominator in enumerate(config["capacity_denominators"]):
        kappa = max(1, args.n // int(denominator))
        scale = choose_stable_subset_size(args.n, kappa)
        subset_size = int(scale["subset_size"])
        probability_precision = 3.0 * epsilon / 128.0
        for multiplier_index, multiplier in enumerate(config["capacity_scale_multipliers"]):
            capacity = min(
                2 * kappa,
                max(kappa, int(round(kappa * float(multiplier)))),
            )
            exact = fixed_subset_survival_probability(
                args.n,
                capacity,
                subset_size,
            )
            bounds = fixed_subset_survival_bounds(
                args.n,
                capacity,
                subset_size,
            )
            inverted_exact = invert_fixed_subset_survival(
                args.n,
                subset_size,
                exact,
                minimum_capacity=kappa,
                maximum_capacity=2 * kappa,
            )
            low_probability = max(0.0, exact - probability_precision)
            high_probability = min(1.0, exact + probability_precision)
            inverted_low = invert_fixed_subset_survival(
                args.n,
                subset_size,
                low_probability,
                minimum_capacity=kappa,
                maximum_capacity=2 * kappa,
            )
            inverted_high = invert_fixed_subset_survival(
                args.n,
                subset_size,
                high_probability,
                minimum_capacity=kappa,
                maximum_capacity=2 * kappa,
            )
            worst_capacity_error = max(
                abs(int(inverted_low["estimated_capacity"]) - capacity),
                abs(int(inverted_high["estimated_capacity"]) - capacity),
            )
            monte_carlo = simulate_fixed_subset_survival(
                args.n,
                capacity,
                subset_size,
                trials=trials,
                seed=(
                    seed
                    + args.n * 1009
                    + denominator_index * 97
                    + multiplier_index * 17
                ),
            )
            inverted_monte_carlo = invert_fixed_subset_survival(
                args.n,
                subset_size,
                monte_carlo,
                minimum_capacity=kappa,
                maximum_capacity=2 * kappa,
            )
            estimated_capacity = int(inverted_monte_carlo["estimated_capacity"])
            query_proxy = fixed_subset_quantum_query_proxy(
                args.n,
                kappa,
                relative_epsilon=epsilon,
                include_geometric_scale_scan=False,
            )
            query_proxy_scan = fixed_subset_quantum_query_proxy(
                args.n,
                kappa,
                relative_epsilon=epsilon,
                include_geometric_scale_scan=True,
            )
            rows.append(
                {
                    "n": args.n,
                    "capacity_scale": kappa,
                    "true_capacity": capacity,
                    "capacity_multiplier": float(multiplier),
                    "subset_size": subset_size,
                    "single_pair_survival_probability": float(
                        scale["single_pair_survival_probability"]
                    ),
                    "exact_survival_probability": exact,
                    "bonferroni_lower": bounds["bonferroni_lower"],
                    "union_upper": bounds["union_upper"],
                    "registered_probability_precision": probability_precision,
                    "exact_inversion_capacity": int(
                        inverted_exact["estimated_capacity"]
                    ),
                    "worst_capacity_error_at_registered_probability_precision": int(
                        worst_capacity_error
                    ),
                    "worst_relative_error_at_registered_probability_precision": (
                        worst_capacity_error / capacity
                    ),
                    "monte_carlo_survival_probability": monte_carlo,
                    "monte_carlo_probability_error": abs(monte_carlo - exact),
                    "monte_carlo_estimated_capacity": estimated_capacity,
                    "monte_carlo_relative_capacity_error": abs(
                        estimated_capacity - capacity
                    )
                    / capacity,
                    "constant_scale_quantum_query_proxy": float(
                        query_proxy["total_query_proxy"]
                    ),
                    "geometric_scan_quantum_query_proxy": float(
                        query_proxy_scan["total_query_proxy"]
                    ),
                    "inner_pair_detection_scale": float(
                        query_proxy["inner_pair_detection_scale"]
                    ),
                }
            )

    artifact = {
        "artifact_type": "qcollide_fixed_subset_capacity_validation_component_v3",
        "schema_version": 1,
        "n": args.n,
        "relative_epsilon": epsilon,
        "monte_carlo_trials": trials,
        "rows": rows,
        "claim_boundary": config["claim_boundary"],
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
