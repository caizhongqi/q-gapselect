"""Shared validation, summaries, and cost accounting for vision experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .costs import classical_packed_cost, product_johnson_cost
from .vision_attack import local_geometry


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def architectures(value: object) -> tuple[str, ...]:
    result = tuple(str(item) for item in sequence(value, "architectures"))
    if not result or any(item not in {"cnn", "tiny_vit"} for item in result):
        raise ValueError("architectures must contain only 'cnn' and 'tiny_vit'")
    return result


def integer_values(
    value: object,
    name: str,
    *,
    allow_zero: bool = False,
) -> tuple[int, ...]:
    parser = nonnegative_int if allow_zero else positive_int
    result = tuple(
        parser(item, f"{name}[{index}]")
        for index, item in enumerate(sequence(value, name))
    )
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def mean_geometry(model: torch.nn.Module, panel, calibration) -> tuple[float, float]:
    openness: list[float] = []
    dimensions: list[int] = []
    for index in range(panel.size):
        _, value, dimension = local_geometry(model, panel, index, calibration)
        openness.append(value)
        dimensions.append(dimension)
    return float(np.mean(openness)), float(np.mean(dimensions))


def packed_costs(anchor_count: int, matching_size: int) -> tuple[float | None, float | None]:
    if matching_size <= 0:
        return None, None
    return (
        float(classical_packed_cost(anchor_count, matching_size)),
        float(product_johnson_cost(anchor_count, matching_size).total_cost),
    )


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["architecture"]),
            int(row["visible_rank"]),
            str(row["intervention"]),
            int(row["closure_rank"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    topology_fields = {
        "mean_collision_active_vertex_fraction": "collision_active_vertex_fraction",
        "mean_collision_beta0_active": "collision_beta0_active",
        "mean_collision_beta1_active": "collision_beta1_active",
        "mean_collision_component_entropy": "collision_component_entropy",
        "mean_collision_displacement_entropy_rank": (
            "collision_displacement_entropy_rank"
        ),
        "mean_collision_displacement_rank_95": "collision_displacement_rank_95",
        "mean_collision_displacement_stable_rank": (
            "collision_displacement_stable_rank"
        ),
        "mean_collision_effective_component_count": (
            "collision_effective_component_count"
        ),
        "mean_collision_independence_ratio": "collision_independence_ratio",
        "mean_collision_largest_component_edge_fraction": (
            "collision_largest_component_edge_fraction"
        ),
        "mean_collision_normalized_component_entropy": (
            "collision_normalized_component_entropy"
        ),
    }
    for key, cell in sorted(groups.items()):
        architecture, visible_rank, intervention, closure_rank = key
        packing = np.asarray([float(row["packing_fraction"]) for row in cell])
        record: dict[str, Any] = {
            "architecture": architecture,
            "visible_rank": visible_rank,
            "intervention": intervention,
            "closure_rank": closure_rank,
            "mean_effective_closure_rank": float(
                np.mean([row["effective_closure_rank"] for row in cell])
            ),
            "model_count": len(cell),
            "mean_packing_fraction": float(packing.mean()),
            "standard_error": (
                float(packing.std(ddof=1) / np.sqrt(len(cell)))
                if len(cell) > 1
                else 0.0
            ),
            "mean_candidate_fraction": float(
                np.mean([row["candidate_fraction"] for row in cell])
            ),
            "mean_openness": float(np.mean([row["mean_openness"] for row in cell])),
            "mean_tunnel_dimension": float(
                np.mean([row["mean_tunnel_dimension"] for row in cell])
            ),
            "mean_benign_control_acceptance": float(
                np.mean([row["benign_control_acceptance"] for row in cell])
            ),
            "mean_evaluation_accuracy": float(
                np.mean([row["evaluation_accuracy"] for row in cell])
            ),
            "mean_residual_energy_fraction": float(
                np.mean([row["residual_energy_fraction"] for row in cell])
            ),
        }
        for output_name, row_name in topology_fields.items():
            if all(row_name in row and row[row_name] is not None for row in cell):
                record[output_name] = float(
                    np.mean([float(row[row_name]) for row in cell])
                )
        output.append(record)
    return output


def adaptive_rank_summary(
    summary: Sequence[Mapping[str, Any]],
    spectra: Sequence[Mapping[str, Any]],
    *,
    packing_target: float,
) -> list[dict[str, Any]]:
    architectures_present = sorted({str(row["architecture"]) for row in summary})
    visible_ranks = sorted({int(row["visible_rank"]) for row in summary})
    target_tolerance = 1e-12 * max(1.0, abs(packing_target))
    output: list[dict[str, Any]] = []
    for architecture in architectures_present:
        for visible_rank in visible_ranks:
            baseline = [
                row
                for row in summary
                if row["architecture"] == architecture
                and row["visible_rank"] == visible_rank
                and row["intervention"] == "baseline"
            ][0]
            targeted = [
                row
                for row in summary
                if row["architecture"] == architecture
                and row["visible_rank"] == visible_rank
                and row["intervention"] == "targeted"
            ]
            eligible = [
                int(row["closure_rank"])
                for row in targeted
                if float(row["mean_packing_fraction"])
                <= packing_target + target_tolerance
            ]
            if (
                float(baseline["mean_packing_fraction"])
                <= packing_target + target_tolerance
            ):
                eligible.append(0)
            static = [
                row
                for row in spectra
                if row["architecture"] == architecture
                and row["visible_rank"] == visible_rank
            ]
            output.append(
                {
                    "architecture": architecture,
                    "visible_rank": visible_rank,
                    "mean_static_gradient_rank_95": float(
                        np.mean([row["rank_95"] for row in static])
                    ),
                    "mean_static_displacement_rank_95": float(
                        np.mean([row["displacement_rank_95"] for row in static])
                    ),
                    "adaptive_closure_rank": min(eligible) if eligible else None,
                    "packing_target": packing_target,
                }
            )
    return output


__all__ = [
    "adaptive_rank_summary",
    "architectures",
    "integer_values",
    "mean_geometry",
    "number",
    "packed_costs",
    "positive_int",
    "sequence",
    "summarize_rows",
]
