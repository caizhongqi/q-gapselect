"""Merge and summarize independent CIFAR functional-collision topology components."""

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


def _summarize_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    final_epoch: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if int(row["checkpoint_epoch"]) != final_epoch:
            continue
        key = (str(row["architecture"]), int(row["visible_rank"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (architecture, visible_rank), cell in sorted(groups.items()):
        nominal = [_nominal_point(row) for row in cell]
        summaries = [row["filtration_summary"] for row in cell]
        persistence = [row["basin_persistence"] for row in cell]
        output.append(
            {
                "architecture": architecture,
                "visible_rank": visible_rank,
                "model_count": len(cell),
                "mean_hidden_openness": _mean(
                    [float(row["mean_hidden_openness"]) for row in cell]
                ),
                "mean_nominal_capacity_fraction": _mean(
                    [float(point["capacity_fraction"]) for point in nominal]
                ),
                "capacity_standard_error": _standard_error(
                    [float(point["capacity_fraction"]) for point in nominal]
                ),
                "mean_nominal_basin_density": _mean(
                    [
                        float(point["beta0_active"])
                        / max(1, min(int(point["n_left"]), int(point["n_right"])))
                        for point in nominal
                    ]
                ),
                "mean_nominal_cycle_density": _mean(
                    [
                        float(point["beta1_active"]) / max(1, int(point["edge_count"]))
                        for point in nominal
                    ]
                ),
                "mean_nominal_component_entropy": _mean(
                    [float(point["normalized_component_edge_entropy"]) for point in nominal]
                ),
                "mean_nominal_displacement_entropy_rank": _mean(
                    [float(point["displacement_entropy_rank"] or 0.0) for point in nominal]
                ),
                "mean_capacity_auc": _mean(
                    [float(summary["capacity_auc"]) for summary in summaries]
                ),
                "mean_capacity_robustness_ratio": _mean(
                    [float(summary["capacity_robustness_ratio"]) for summary in summaries]
                ),
                "mean_basin_density_auc": _mean(
                    [float(summary["basin_density_auc"]) for summary in summaries]
                ),
                "mean_cycle_density_auc": _mean(
                    [float(summary["cycle_density_auc"]) for summary in summaries]
                ),
                "mean_normalized_basin_lifetime": _mean(
                    [
                        float(item["normalized_total_lifetime"])
                        for item in persistence
                    ]
                ),
                "mean_effective_persistent_basin_count": _mean(
                    [
                        float(item["effective_persistent_basin_count"])
                        for item in persistence
                    ]
                ),
            }
        )
    return output


def _summarize_dynamics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["architecture"]),
            int(row["checkpoint_epoch"]),
            int(row["visible_rank"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (architecture, epoch, visible_rank), cell in sorted(groups.items()):
        nominal = [_nominal_point(row) for row in cell]
        output.append(
            {
                "architecture": architecture,
                "checkpoint_epoch": epoch,
                "visible_rank": visible_rank,
                "model_count": len(cell),
                "mean_hidden_openness": _mean(
                    [float(row["mean_hidden_openness"]) for row in cell]
                ),
                "mean_capacity_fraction": _mean(
                    [float(point["capacity_fraction"]) for point in nominal]
                ),
                "mean_basin_density": _mean(
                    [
                        float(point["beta0_active"])
                        / max(1, min(int(point["n_left"]), int(point["n_right"])))
                        for point in nominal
                    ]
                ),
                "mean_displacement_entropy_rank": _mean(
                    [float(point["displacement_entropy_rank"] or 0.0) for point in nominal]
                ),
            }
        )
    return output


def _architecture_fingerprints(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    architectures = sorted({str(cell["architecture"]) for cell in cells})
    output: list[dict[str, Any]] = []
    for architecture in architectures:
        subset = [cell for cell in cells if cell["architecture"] == architecture]
        output.append(
            {
                "architecture": architecture,
                "rank_count": len(subset),
                "mean_capacity_auc": _mean(
                    [float(cell["mean_capacity_auc"]) for cell in subset]
                ),
                "mean_basin_density_auc": _mean(
                    [float(cell["mean_basin_density_auc"]) for cell in subset]
                ),
                "mean_cycle_density_auc": _mean(
                    [float(cell["mean_cycle_density_auc"]) for cell in subset]
                ),
                "mean_persistent_basin_lifetime": _mean(
                    [float(cell["mean_normalized_basin_lifetime"]) for cell in subset]
                ),
                "mean_displacement_entropy_rank": _mean(
                    [
                        float(cell["mean_nominal_displacement_entropy_rank"])
                        for cell in subset
                    ]
                ),
                "mean_hidden_openness": _mean(
                    [float(cell["mean_hidden_openness"]) for cell in subset]
                ),
            }
        )
    return output


def merge_cifar_topology_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Merge component artifacts without averaging pre-averaged observations."""

    if not artifacts:
        raise ValueError("artifacts must be non-empty")
    artifact_types = {str(item["artifact_type"]) for item in artifacts}
    schema_versions = {int(item["schema_version"]) for item in artifacts}
    master_seeds = {int(item["master_seed"]) for item in artifacts}
    if artifact_types != {"qcollide_cifar_functional_collision_topology_component"}:
        raise ValueError("unexpected component artifact type")
    if len(schema_versions) != 1 or len(master_seeds) != 1:
        raise ValueError("component artifacts must share schema and master seed")
    rows = [row for artifact in artifacts for row in artifact["rows"]]
    training = [artifact["training"] for artifact in artifacts]
    epochs = int(config["epochs"])
    cells = _summarize_cells(rows, final_epoch=epochs)
    dynamics = _summarize_dynamics(rows)
    fingerprints = _architecture_fingerprints(cells)
    parameters = [int(item["parameter_count"]) for item in training]
    accuracies = [float(item["evaluation_accuracy"]) for item in training]
    monotone_rows = [
        bool(row["filtration_summary"]["edge_count_monotone"])
        and bool(row["filtration_summary"]["capacity_monotone"])
        and bool(row["filtration_summary"]["cycle_rank_monotone"])
        for row in rows
    ]
    return {
        "artifact_type": "qcollide_cifar_functional_collision_topology_merged",
        "schema_version": next(iter(schema_versions)),
        "master_seed": next(iter(master_seeds)),
        "claim_scope": {
            "dataset": str(config["dataset"]),
            "architectures": sorted({str(item["architecture"]) for item in artifacts}),
            "natural_cross_class_collision_graph": True,
            "adaptive_attack_claimed": False,
            "persistent_control_threshold_filtration": True,
            "training_checkpoint_dynamics": True,
            "quantum_topology_estimation_claimed": False,
            "coherent_quantum_execution": False,
        },
        "config": dict(config),
        "component_count": len(artifacts),
        "training": training,
        "rows": rows,
        "final_cells": cells,
        "training_dynamics": dynamics,
        "architecture_fingerprints": fingerprints,
        "gates": {
            "minimum_evaluation_accuracy": min(accuracies),
            "mean_evaluation_accuracy": _mean(accuracies),
            "maximum_parameter_ratio": max(parameters) / min(parameters),
            "all_filtrations_monotone": bool(all(monotone_rows)),
            "all_architectures_present": sorted(
                {str(item["architecture"]) for item in artifacts}
            )
            == sorted(str(value) for value in config["architectures"]),
            "nonzero_persistent_capacity_cells": int(
                np.count_nonzero(
                    [float(cell["mean_capacity_auc"]) > 0.0 for cell in cells]
                )
            ),
        },
    }


def compact_cifar_topology_summary(artifact: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "artifact_type",
        "schema_version",
        "master_seed",
        "claim_scope",
        "training",
        "final_cells",
        "training_dynamics",
        "architecture_fingerprints",
        "gates",
    )
    return {key: artifact[key] for key in keys}


__all__ = [
    "compact_cifar_topology_summary",
    "merge_cifar_topology_components",
]
