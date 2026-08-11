"""Merge Cora functional-collision atlas components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _standard_error(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0


def _nominal_point(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nominal = float(row["nominal_epsilon"])
    return min(
        row["filtration_points"],
        key=lambda point: abs(float(point["control_epsilon"]) - nominal),
    )


def merge_graph_atlas_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    configured_architectures: Sequence[str],
    configured_seeds: Sequence[int],
) -> dict[str, object]:
    """Validate architecture×seed coverage and aggregate nominal topology cells."""

    if not artifacts:
        raise ValueError("artifacts must be non-empty")
    expected = {
        (str(architecture), int(seed))
        for architecture in configured_architectures
        for seed in configured_seeds
    }
    observed = {
        (str(artifact["architecture"]), int(artifact["model_seed"]))
        for artifact in artifacts
    }
    if observed != expected:
        raise ValueError(
            f"graph atlas component grid mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    if {str(artifact["artifact_type"]) for artifact in artifacts} != {
        "qcollide_graph_functional_collision_atlas_component"
    }:
        raise ValueError("unexpected graph atlas component type")

    raw_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        diagnostics = artifact["model_diagnostics"]
        for row in artifact["rows"]:
            point = _nominal_point(row)
            denominator = max(1, min(int(point["n_left"]), int(point["n_right"])))
            raw_rows.append(
                {
                    "architecture": str(artifact["architecture"]),
                    "model_seed": int(artifact["model_seed"]),
                    "visible_rank": int(row["visible_rank"]),
                    "evaluation_accuracy": float(diagnostics["evaluation_accuracy"]),
                    "control_head_calibration_r2": float(
                        diagnostics["control_head_calibration_r2"]
                    ),
                    "parameter_count": int(diagnostics["parameter_count"]),
                    "capacity_fraction": float(point["capacity_fraction"]),
                    "basin_density": float(point["beta0_active"]) / denominator,
                    "within_basin_multiplicity": (
                        float(point["matching_size"]) / int(point["beta0_active"])
                        if int(point["beta0_active"]) > 0
                        else 0.0
                    ),
                    "cycle_density": float(point["beta1_active"])
                    / max(1, int(point["edge_count"])),
                    "component_entropy": float(point["normalized_component_edge_entropy"]),
                    "displacement_entropy_rank": float(point["displacement_entropy_rank"] or 0.0),
                    "capacity_auc": float(row["filtration_summary"]["capacity_auc"]),
                    "capacity_robustness_ratio": float(
                        row["filtration_summary"]["capacity_robustness_ratio"]
                    ),
                    "persistent_basin_lifetime": float(
                        row["basin_persistence"]["normalized_total_lifetime"]
                    ),
                    "max_overlap_degree": int(point["max_overlap_degree"]),
                }
            )

    metric_names = (
        "evaluation_accuracy",
        "control_head_calibration_r2",
        "capacity_fraction",
        "basin_density",
        "within_basin_multiplicity",
        "cycle_density",
        "component_entropy",
        "displacement_entropy_rank",
        "capacity_auc",
        "capacity_robustness_ratio",
        "persistent_basin_lifetime",
        "max_overlap_degree",
    )
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        groups.setdefault((str(row["architecture"]), int(row["visible_rank"])), []).append(row)

    cells: list[dict[str, object]] = []
    for (architecture, visible_rank), observations in sorted(groups.items()):
        cell: dict[str, object] = {
            "architecture": architecture,
            "visible_rank": visible_rank,
            "seed_count": len(observations),
            "parameter_count_mean": _mean(
                [float(row["parameter_count"]) for row in observations]
            ),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in observations]
            cell[f"mean_{metric}"] = _mean(values)
            cell[f"se_{metric}"] = _standard_error(values)
        cells.append(cell)

    nonzero = [float(row["capacity_fraction"]) > 0.0 for row in raw_rows]
    finite = [
        float(value)
        for row in raw_rows
        for key, value in row.items()
        if key != "architecture"
    ]
    return {
        "artifact_type": "qcollide_graph_functional_collision_atlas_merged",
        "schema_version": 1,
        "domain": "graph_learning",
        "dataset": "cora",
        "architectures": [str(value) for value in configured_architectures],
        "model_seeds": [int(value) for value in configured_seeds],
        "component_count": len(artifacts),
        "raw_rows": raw_rows,
        "atlas_cells": cells,
        "gates": {
            "complete_architecture_seed_grid": True,
            "architecture_count": len(configured_architectures),
            "seed_count": len(configured_seeds),
            "nonzero_capacity_row_fraction": float(np.mean(nonzero)),
            "all_rows_finite": bool(all(np.isfinite(value) for value in finite)),
        },
        "claim_boundary": {
            "seeds_are_independent_model_training_seeds": True,
            "cora_is_transductive_node_classification": True,
            "cross_domain_universality_proved": False,
            "quantum_execution": False,
        },
    }


__all__ = ["merge_graph_atlas_components"]
