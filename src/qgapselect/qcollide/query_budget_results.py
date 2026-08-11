"""Aggregation and claim gates for query-budget collision-capacity scaling v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import log
from typing import Any

import numpy as np

from .query_budget_scaling import ceil_query_proxy


def _fit_log_exponent(rows: Sequence[Mapping[str, Any]], query_key: str) -> float | None:
    points: list[tuple[float, float]] = []
    for row in rows:
        query_value = row.get(query_key)
        if query_value is None:
            continue
        n = int(row["n"])
        if int(query_value) >= 2 * n:
            continue
        effective = float(row["qccs_packed_query_proxy"]["effective_pair_scale"])
        if effective > 0.0 and int(query_value) > 0:
            points.append((log(effective), log(int(query_value))))
    if len(points) < 2:
        return None
    x = np.asarray([item[0] for item in points], dtype=float)
    y = np.asarray([item[1] for item in points], dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def _full_budget_row(cell: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = list(cell["budget_rows"])
    if not rows:
        raise ValueError("query-budget cell has no budget rows")
    row = max(rows, key=lambda item: float(item["endpoint_query_fraction"]))
    if abs(float(row["endpoint_query_fraction"]) - 1.0) > 1e-12:
        raise ValueError("each query-budget cell must include full endpoint reconstruction")
    return row


def merge_query_budget_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    matching_n_values: Sequence[int],
    overlap_n_values: Sequence[int],
    matching_densities: Sequence[float],
) -> dict[str, object]:
    """Merge fixed-grid components and separate proved/empirical/proxy statements."""

    if not artifacts:
        raise ValueError("artifacts must be non-empty")
    if {str(item["artifact_type"]) for item in artifacts} != {
        "qcollide_query_budget_scaling_component_v2"
    }:
        raise ValueError("unexpected query-budget artifact type")

    expected = {
        *(('matching', int(n)) for n in matching_n_values),
        *(('overlap', int(n)) for n in overlap_n_values),
    }
    observed = {(str(item["panel"]), int(item["n"])) for item in artifacts}
    if observed != expected:
        raise ValueError(
            f"query-budget component grid mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )

    density_reference = tuple(float(value) for value in matching_densities)
    for artifact in artifacts:
        observed_densities = tuple(float(value) for value in artifact["matching_densities"])
        if observed_densities != density_reference:
            raise ValueError("matching-density grid changed between components")

    cells = [
        dict(cell)
        for artifact in sorted(artifacts, key=lambda item: (str(item["panel"]), int(item["n"])))
        for cell in artifact["cells"]
    ]
    matching_cells = [row for row in cells if row["family"] == "matching"]
    overlap_cells = [row for row in cells if row["panel"] == "overlap"]

    matching_main_table: list[dict[str, object]] = []
    for cell in matching_cells:
        relative_queries = cell["minimum_matching_scaled_queries_relative"]
        additive_queries = cell["minimum_matching_scaled_queries_additive"]
        proxy_value = float(cell["qccs_packed_query_proxy"]["epsilon_adjusted_query_proxy"])
        matching_main_table.append(
            {
                "n": int(cell["n"]),
                "matching_size": int(cell["matching_size"]),
                "matching_density": float(cell["matching_density"]),
                "effective_pair_scale": float(
                    cell["qccs_packed_query_proxy"]["effective_pair_scale"]
                ),
                "empirical_classical_queries_relative": (
                    int(relative_queries) if relative_queries is not None else None
                ),
                "empirical_classical_queries_additive": (
                    int(additive_queries) if additive_queries is not None else None
                ),
                "full_reconstruction_queries": 2 * int(cell["n"]),
                "qccs_packed_epsilon_adjusted_proxy": proxy_value,
                "qccs_packed_epsilon_adjusted_proxy_ceil": ceil_query_proxy(proxy_value),
                "classical_to_qccs_proxy_ratio_relative": (
                    float(relative_queries) / proxy_value
                    if relative_queries is not None
                    else None
                ),
            }
        )

    overlap_robustness_table: list[dict[str, object]] = []
    for cell in overlap_cells:
        full = _full_budget_row(cell)
        overlap_robustness_table.append(
            {
                "family": str(cell["family"]),
                "n": int(cell["n"]),
                "matching_size": int(cell["matching_size"]),
                "matching_density": float(cell["matching_density"]),
                "edge_count": int(cell["edge_count"]),
                "maximum_degree": int(cell["maximum_degree"]),
                "minimum_matching_scaled_queries_relative": cell[
                    "minimum_matching_scaled_queries_relative"
                ],
                "minimum_matching_scaled_queries_additive": cell[
                    "minimum_matching_scaled_queries_additive"
                ],
                "minimum_edge_proxy_queries_relative": cell[
                    "minimum_edge_proxy_queries_relative"
                ],
                "full_budget_matching_relative_success_rate": float(
                    full["matching_scaled_relative_success_rate"]
                ),
                "full_budget_edge_proxy_relative_success_rate": float(
                    full["edge_proxy_relative_success_rate"]
                ),
                "full_budget_edge_proxy_mean_relative_error": float(
                    full["edge_proxy_mean_relative_error"]
                ),
            }
        )

    matching_full_success = [
        float(_full_budget_row(cell)["matching_scaled_relative_success_rate"])
        for cell in matching_cells
    ]
    overlap_full_matching_success = [
        float(_full_budget_row(cell)["matching_scaled_relative_success_rate"])
        for cell in overlap_cells
    ]
    overlap_edge_failures = [
        float(_full_budget_row(cell)["edge_proxy_relative_success_rate"]) == 0.0
        for cell in overlap_cells
    ]
    proxy_ratios = [
        float(row["classical_to_qccs_proxy_ratio_relative"])
        for row in matching_main_table
        if row["classical_to_qccs_proxy_ratio_relative"] is not None
    ]

    relative_exponent = _fit_log_exponent(
        matching_cells,
        "minimum_matching_scaled_queries_relative",
    )
    additive_exponent = _fit_log_exponent(
        matching_cells,
        "minimum_matching_scaled_queries_additive",
    )
    return {
        "artifact_type": "qcollide_query_budget_scaling_main_table_v2",
        "schema_version": 2,
        "component_count": len(artifacts),
        "cell_count": len(cells),
        "matching_main_table": sorted(
            matching_main_table,
            key=lambda row: (int(row["n"]), float(row["matching_density"])),
        ),
        "overlap_robustness_table": sorted(
            overlap_robustness_table,
            key=lambda row: (
                str(row["family"]),
                int(row["n"]),
                float(row["matching_density"]),
            ),
        ),
        "summary": {
            "matching_empirical_relative_query_exponent_vs_effective_pair_scale": (
                relative_exponent
            ),
            "matching_empirical_additive_query_exponent_vs_effective_pair_scale": (
                additive_exponent
            ),
            "qccs_packed_proxy_exponent_vs_effective_pair_scale": 1.0 / 3.0,
            "minimum_matching_classical_to_qccs_proxy_ratio": (
                min(proxy_ratios) if proxy_ratios else None
            ),
            "mean_matching_classical_to_qccs_proxy_ratio": (
                float(np.mean(proxy_ratios)) if proxy_ratios else None
            ),
            "maximum_matching_classical_to_qccs_proxy_ratio": (
                max(proxy_ratios) if proxy_ratios else None
            ),
        },
        "gates": {
            "complete_component_grid": True,
            "matching_full_reconstruction_relative_success": bool(
                all(abs(value - 1.0) < 1e-12 for value in matching_full_success)
            ),
            "overlap_full_reconstruction_relative_success": bool(
                all(abs(value - 1.0) < 1e-12 for value in overlap_full_matching_success)
            ),
            "overlap_edge_count_negative_control_failure_fraction": float(
                np.mean(overlap_edge_failures)
            )
            if overlap_edge_failures
            else None,
        },
        "claim_boundary": {
            "matching_classical_query_counts_are_measured": True,
            "qccs_values_are_analytic_proxy_not_runtime": True,
            "qccs_epsilon_factor_is_a_closed_theorem": False,
            "overlap_quantum_advantage_claimed": False,
            "hardware_quantum_advantage_claimed": False,
            "full_reconstruction_recovers_exact_capacity": True,
            "edge_count_is_capacity": False,
        },
    }


__all__ = ["merge_query_budget_components"]
