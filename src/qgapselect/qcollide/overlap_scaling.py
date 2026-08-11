"""Cross-scale summaries for overlap-aware collision-capacity experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt
from typing import Any

import numpy as np


def analytic_packed_query_scales(
    n_left: int,
    n_right: int,
    matching_size: int,
) -> dict[str, float | bool | None]:
    """Return positive-capacity classical/quantum endpoint-query scaling proxies."""

    if n_left < 0 or n_right < 0 or matching_size < 0:
        raise ValueError("domain sizes and matching_size must be non-negative")
    if matching_size > min(n_left, n_right):
        raise ValueError("matching_size exceeds the bipartite domain cap")
    if n_left == 0 or n_right == 0 or matching_size == 0:
        return {
            "defined_for_positive_capacity": False,
            "effective_pair_scale": None,
            "classical_query_scale": None,
            "quantum_query_scale": None,
            "classical_to_quantum_ratio": None,
        }
    effective = (n_left * n_right) / matching_size
    classical = sqrt(effective)
    quantum = effective ** (1.0 / 3.0)
    return {
        "defined_for_positive_capacity": True,
        "effective_pair_scale": float(effective),
        "classical_query_scale": float(classical),
        "quantum_query_scale": float(quantum),
        "classical_to_quantum_ratio": float(classical / quantum),
    }


def _relative_width(lower: float, upper: float, truth: int) -> float | None:
    if truth <= 0:
        return None
    return float((upper - lower) / truth)


def _paired_family_records(
    component: Mapping[str, object],
) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    grouped: dict[str, dict[bool, Mapping[str, Any]]] = {}
    for raw in component["records"]:
        record = raw
        family = str(record["family"])
        grouped.setdefault(family, {})[bool(record["degree_aware"])] = record
    output: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for family, records in grouped.items():
        if set(records) != {False, True}:
            raise ValueError(f"family {family!r} lacks uniform/degree-aware records")
        output[family] = (records[False], records[True])
    return output


def merge_overlap_scaling_components(
    artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Merge scaling components while separating empirical and analytic quantities."""

    if not artifacts:
        raise ValueError("artifacts must be non-empty")
    if {str(item["artifact_type"]) for item in artifacts} != {
        "qcollide_overlap_capacity_benchmark"
    }:
        raise ValueError("unexpected overlap scaling artifact type")

    cells: list[dict[str, object]] = []
    deterministic_coverage: list[bool] = []
    survival_coverage: list[bool] = []
    for component in sorted(
        artifacts,
        key=lambda item: (int(item["n"]), int(item["requested_matching_size"])),
    ):
        n = int(component["n"])
        requested = int(component["requested_matching_size"])
        requested_density = requested / n
        for family, (uniform, degree_aware) in sorted(
            _paired_family_records(component).items()
        ):
            truth = int(uniform["true_matching_size"])
            if truth != int(degree_aware["true_matching_size"]):
                raise ValueError("uniform and degree-aware records disagree on ground truth")
            deterministic = uniform["deterministic_capacity_envelope"]
            lower = float(deterministic["lower"])
            upper = float(deterministic["upper"])
            deterministic_contains = lower <= truth <= upper
            deterministic_coverage.append(deterministic_contains)

            certificate = uniform.get("uniform_survival_capacity_certificate")
            certificate_payload: dict[str, float | bool | None]
            if certificate is None:
                certificate_payload = {
                    "capacity_lower": None,
                    "capacity_upper": None,
                    "relative_width": None,
                    "contains_truth": False,
                }
            else:
                capacity_lower = float(certificate["capacity_lower"])
                capacity_upper = float(certificate["capacity_upper"])
                contains_truth = capacity_lower <= truth <= capacity_upper
                survival_coverage.append(contains_truth)
                certificate_payload = {
                    "capacity_lower": capacity_lower,
                    "capacity_upper": capacity_upper,
                    "relative_width": _relative_width(
                        capacity_lower,
                        capacity_upper,
                        truth,
                    ),
                    "contains_truth": contains_truth,
                }

            scales = analytic_packed_query_scales(n, n, truth)
            cells.append(
                {
                    "n": n,
                    "requested_matching_size": requested,
                    "requested_matching_density": requested_density,
                    "family": family,
                    "edge_count": int(uniform["edge_count"]),
                    "maximum_degree": int(uniform["maximum_degree"]),
                    "true_matching_size": truth,
                    "true_matching_density": truth / n,
                    "retention_probability": float(uniform["target_q"]),
                    "uniform_survival_probability": float(
                        uniform["survival_probability"]
                    ),
                    "uniform_packed_inversion_estimate": float(
                        uniform["packed_inversion_estimate"]
                    ),
                    "uniform_packed_inversion_absolute_error": float(
                        uniform["absolute_error_packed_inversion"]
                    ),
                    "uniform_scaled_matching_estimate": float(
                        uniform["scaled_matching_estimate"]
                    ),
                    "uniform_scaled_matching_absolute_error": float(
                        uniform["absolute_error_scaled_matching"]
                    ),
                    "degree_aware_packed_inversion_estimate": float(
                        degree_aware["packed_inversion_estimate"]
                    ),
                    "degree_aware_packed_inversion_absolute_error": float(
                        degree_aware["absolute_error_packed_inversion"]
                    ),
                    "degree_aware_scaled_matching_estimate": float(
                        degree_aware["scaled_matching_estimate"]
                    ),
                    "degree_aware_scaled_matching_absolute_error": float(
                        degree_aware["absolute_error_scaled_matching"]
                    ),
                    "deterministic_capacity_lower": lower,
                    "deterministic_capacity_upper": upper,
                    "deterministic_relative_width": _relative_width(lower, upper, truth),
                    "deterministic_contains_truth": deterministic_contains,
                    "survival_capacity_certificate": certificate_payload,
                    "analytic_query_scales": scales,
                }
            )

    positive = [
        cell
        for cell in cells
        if int(cell["true_matching_size"]) > 0
        and bool(cell["analytic_query_scales"]["defined_for_positive_capacity"])
    ]
    ratios = [
        float(cell["analytic_query_scales"]["classical_to_quantum_ratio"])
        for cell in positive
    ]
    matching_errors = [
        float(cell["uniform_packed_inversion_absolute_error"])
        / int(cell["true_matching_size"])
        for cell in cells
        if cell["family"] == "matching" and int(cell["true_matching_size"]) > 0
    ]
    sparse_uniform_errors = [
        float(cell["uniform_packed_inversion_absolute_error"])
        for cell in cells
        if cell["family"] == "sparse_overlap"
    ]
    sparse_degree_errors = [
        float(cell["degree_aware_packed_inversion_absolute_error"])
        for cell in cells
        if cell["family"] == "sparse_overlap"
    ]
    return {
        "artifact_type": "qcollide_overlap_scaling_main_table",
        "schema_version": 1,
        "component_count": len(artifacts),
        "cell_count": len(cells),
        "cells": cells,
        "summary": {
            "n_values": sorted({int(cell["n"]) for cell in cells}),
            "requested_matching_densities": sorted(
                {float(cell["requested_matching_density"]) for cell in cells}
            ),
            "families": sorted({str(cell["family"]) for cell in cells}),
            "mean_positive_capacity_classical_to_quantum_proxy_ratio": (
                float(np.mean(ratios)) if ratios else None
            ),
            "minimum_positive_capacity_classical_to_quantum_proxy_ratio": (
                min(ratios) if ratios else None
            ),
            "mean_matching_packed_inversion_relative_error": (
                float(np.mean(matching_errors)) if matching_errors else None
            ),
            "mean_sparse_uniform_packed_error": (
                float(np.mean(sparse_uniform_errors))
                if sparse_uniform_errors
                else None
            ),
            "mean_sparse_degree_aware_packed_error": (
                float(np.mean(sparse_degree_errors))
                if sparse_degree_errors
                else None
            ),
        },
        "gates": {
            "all_deterministic_envelopes_cover_truth": bool(all(deterministic_coverage)),
            "survival_certificate_coverage_fraction": (
                float(np.mean(survival_coverage)) if survival_coverage else None
            ),
            "positive_capacity_cell_count": len(positive),
            "analytic_classical_effective_scale_exponent": 0.5,
            "analytic_quantum_effective_scale_exponent": 1.0 / 3.0,
        },
        "claim_boundary": {
            "analytic_query_scales_are_runtime_measurements": False,
            "hardware_quantum_advantage_claimed": False,
            "uniform_survival_capacity_interval_proved": True,
            "degree_aware_point_estimator_proved_general": False,
            "matching_ground_truth_is_exact": True,
        },
    }


__all__ = ["analytic_packed_query_scales", "merge_overlap_scaling_components"]
