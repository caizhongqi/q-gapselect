"""F0--F2 pair-oracle, packed-claw, and random-range scaling panels."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .costs import (
    classical_packed_cost,
    pair_grover_cost,
    product_johnson_cost,
)
from .fixtures import packed_claw, pair_oracle_negative, random_range
from .graph import packing_statistics
from .scaling_common import (
    derived_seed,
    fit_log_linear,
    float_values,
    int_values,
    positive_int,
    power_count,
)


def run_f0(config: Mapping[str, object], master_seed: int) -> dict[str, object]:
    n_values = int_values(config.get("n_values"), "f0.n_values")
    exponents = float_values(config.get("marked_exponents"), "f0.marked_exponents")
    repetitions = positive_int(config.get("repetitions"), "f0.repetitions")
    rows: list[dict[str, object]] = []
    for n in n_values:
        for exponent in exponents:
            marked = power_count(n, exponent, maximum=n * n)
            for repetition in range(repetitions):
                instance = pair_oracle_negative(
                    n=n,
                    marked_pairs=marked,
                    seed=derived_seed(master_seed, "F0", n, exponent, repetition),
                )
                stats = packing_statistics(instance)
                rows.append(
                    {
                        "n": n,
                        "marked_exponent": exponent,
                        "marked_pairs": marked,
                        "edge_count": stats.edge_count,
                        "matching_size_diagnostic": stats.matching_size,
                        "pair_grover_cost": pair_grover_cost(n, n, marked),
                        "structured_claw_admissible": False,
                        "fingerprint": instance.metadata["fingerprint"],
                    }
                )
    return {
        "fixture": "F0",
        "semantics": "arbitrary pair-oracle negative control",
        "rows": rows,
        "fits_by_marked_exponent": [
            {
                "marked_exponent": exponent,
                "expected_n_coefficient": 1.0 - exponent / 2.0,
                "fit": fit_log_linear(
                    [row for row in rows if row["marked_exponent"] == exponent],
                    response="pair_grover_cost",
                    predictors=("n",),
                ),
            }
            for exponent in exponents
        ],
    }


def run_f1(config: Mapping[str, object], master_seed: int) -> dict[str, object]:
    n_values = int_values(config.get("n_values"), "f1.n_values")
    exponents = float_values(config.get("matching_exponents"), "f1.matching_exponents")
    repetitions = positive_int(config.get("repetitions"), "f1.repetitions")
    rows: list[dict[str, object]] = []
    for n in n_values:
        for exponent in exponents:
            matching_size = power_count(n, exponent, maximum=n)
            for repetition in range(repetitions):
                instance = packed_claw(
                    n=n,
                    matching_size=matching_size,
                    seed=derived_seed(master_seed, "F1", n, exponent, repetition),
                )
                stats = packing_statistics(instance)
                quantum = product_johnson_cost(n, stats.matching_size)
                rows.append(
                    {
                        "n": n,
                        "matching_exponent": exponent,
                        "matching_size": stats.matching_size,
                        "edge_count": stats.edge_count,
                        "independence_ratio": stats.independence_ratio,
                        "classical_packed_cost": classical_packed_cost(n, stats.matching_size),
                        "product_johnson_total": quantum.total_cost,
                        "product_johnson_proxy": quantum.asymptotic_proxy,
                        "setup_size": quantum.setup_size,
                        "fingerprint": instance.metadata["fingerprint"],
                    }
                )
    return {
        "fixture": "F1",
        "semantics": "exact vertex-disjoint endpoint-local claws",
        "rows": rows,
        "joint_fits": {
            "classical": fit_log_linear(
                rows,
                response="classical_packed_cost",
                predictors=("n", "matching_size"),
            ),
            "quantum_proxy": fit_log_linear(
                rows,
                response="product_johnson_proxy",
                predictors=("n", "matching_size"),
            ),
            "quantum_integer_optimized": fit_log_linear(
                rows,
                response="product_johnson_total",
                predictors=("n", "matching_size"),
            ),
        },
        "expected_coefficients": {
            "classical": {"n": 1.0, "matching_size": -0.5},
            "quantum": {"n": 2.0 / 3.0, "matching_size": -1.0 / 3.0},
        },
    }


def run_f2(config: Mapping[str, object], master_seed: int) -> dict[str, object]:
    n_values = int_values(config.get("n_values"), "f2.n_values")
    target_exponents = float_values(
        config.get("target_collision_exponents"),
        "f2.target_collision_exponents",
    )
    repetitions = positive_int(config.get("repetitions"), "f2.repetitions")
    rows: list[dict[str, object]] = []
    for n in n_values:
        for exponent in target_exponents:
            target_collisions = power_count(n, exponent, maximum=n * n)
            range_size = max(1, int(round(n * n / target_collisions)))
            for repetition in range(repetitions):
                instance = random_range(
                    n=n,
                    range_size=range_size,
                    seed=derived_seed(master_seed, "F2", n, exponent, repetition),
                )
                stats = packing_statistics(instance)
                rows.append(
                    {
                        "n": n,
                        "target_collision_exponent": exponent,
                        "target_collision_count": target_collisions,
                        "range_size": range_size,
                        "edge_count": stats.edge_count,
                        "matching_size": stats.matching_size,
                        "independence_ratio": stats.independence_ratio,
                        "max_left_degree": stats.max_left_degree,
                        "max_right_degree": stats.max_right_degree,
                        "degree_gini": stats.degree_gini,
                        "fingerprint": instance.metadata["fingerprint"],
                    }
                )
    summaries: list[dict[str, object]] = []
    for n in n_values:
        for exponent in target_exponents:
            cell = [
                row
                for row in rows
                if row["n"] == n and row["target_collision_exponent"] == exponent
            ]
            edge_counts = np.asarray([row["edge_count"] for row in cell], dtype=float)
            matching_sizes = np.asarray([row["matching_size"] for row in cell], dtype=float)
            target = float(cell[0]["target_collision_count"])
            summaries.append(
                {
                    "n": n,
                    "target_collision_exponent": exponent,
                    "target_collision_count": int(target),
                    "mean_edge_count": float(edge_counts.mean()),
                    "mean_matching_size": float(matching_sizes.mean()),
                    "edge_to_target_ratio": float(edge_counts.mean() / target),
                    "matching_to_target_ratio": float(matching_sizes.mean() / target),
                    "nonzero_matching_fraction": float(np.mean(matching_sizes > 0.0)),
                }
            )
    return {
        "fixture": "F2",
        "semantics": "random range occupancy and packing concentration",
        "rows": rows,
        "cell_summaries": summaries,
    }
