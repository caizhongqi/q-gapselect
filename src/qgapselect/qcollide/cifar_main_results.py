"""Performance-matched main-table construction for CIFAR collision topology."""

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


def performance_matched_cifar_main_table(
    topology: Mapping[str, object],
    performance: Mapping[str, object],
    *,
    target_accuracy: float,
    maximum_accuracy_mismatch: float,
    visible_ranks: Sequence[int],
) -> dict[str, object]:
    """Join topology to calibration-accuracy matched checkpoints without interpolation."""

    if not 0.0 < target_accuracy < 1.0:
        raise ValueError("target_accuracy must lie in (0,1)")
    if not 0.0 <= maximum_accuracy_mismatch < 1.0:
        raise ValueError("maximum_accuracy_mismatch must lie in [0,1)")
    ranks = tuple(int(rank) for rank in visible_ranks)
    if not ranks or any(rank <= 0 for rank in ranks):
        raise ValueError("visible_ranks must contain positive integers")

    topology_type = str(topology.get("artifact_type"))
    performance_type = str(performance.get("artifact_type"))
    if topology_type != "qcollide_cifar_functional_collision_topology_merged":
        raise ValueError("unexpected topology artifact type")
    if performance_type != "qcollide_cifar_checkpoint_performance_merged":
        raise ValueError("unexpected performance artifact type")

    topology_rows = list(topology["rows"])
    performance_rows = list(performance["rows"])
    architectures = sorted({str(row["architecture"]) for row in performance_rows})
    seeds = sorted({int(row["model_seed"]) for row in performance_rows})

    performance_by_model: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in performance_rows:
        key = (str(row["architecture"]), int(row["model_seed"]))
        performance_by_model.setdefault(key, []).append(row)

    topology_by_key: dict[tuple[str, int, int, int], Mapping[str, Any]] = {}
    for row in topology_rows:
        key = (
            str(row["architecture"]),
            int(row["model_seed"]),
            int(row["checkpoint_epoch"]),
            int(row["visible_rank"]),
        )
        topology_by_key[key] = row

    selections: list[dict[str, object]] = []
    cells: dict[tuple[str, int], list[dict[str, float]]] = {}
    missing_topology: list[tuple[str, int, int, int]] = []

    for architecture in architectures:
        architecture_seeds = sorted(
            seed for name, seed in performance_by_model if name == architecture
        )
        for seed in architecture_seeds:
            candidates = sorted(
                performance_by_model[(architecture, seed)],
                key=lambda row: (
                    abs(float(row["calibration_accuracy"]) - target_accuracy),
                    int(row["checkpoint_epoch"]),
                ),
            )
            chosen = candidates[0]
            checkpoint = int(chosen["checkpoint_epoch"])
            calibration_accuracy = float(chosen["calibration_accuracy"])
            evaluation_accuracy = float(chosen["evaluation_accuracy"])
            mismatch = abs(calibration_accuracy - target_accuracy)
            selections.append(
                {
                    "architecture": architecture,
                    "model_seed": seed,
                    "checkpoint_epoch": checkpoint,
                    "calibration_accuracy": calibration_accuracy,
                    "evaluation_accuracy": evaluation_accuracy,
                    "absolute_calibration_mismatch": mismatch,
                    "within_tolerance": mismatch <= maximum_accuracy_mismatch,
                }
            )
            for rank in ranks:
                key = (architecture, seed, checkpoint, rank)
                row = topology_by_key.get(key)
                if row is None:
                    missing_topology.append(key)
                    continue
                point = _nominal_point(row)
                denominator = max(
                    1,
                    min(int(point["n_left"]), int(point["n_right"])),
                )
                cells.setdefault((architecture, rank), []).append(
                    {
                        "capacity_fraction": float(point["capacity_fraction"]),
                        "basin_density": float(point["beta0_active"]) / denominator,
                        "cycle_density": float(point["beta1_active"])
                        / max(1, int(point["edge_count"])),
                        "component_entropy": float(
                            point["normalized_component_edge_entropy"]
                        ),
                        "displacement_entropy_rank": float(
                            point["displacement_entropy_rank"] or 0.0
                        ),
                        "capacity_auc": float(row["filtration_summary"]["capacity_auc"]),
                        "capacity_robustness_ratio": float(
                            row["filtration_summary"]["capacity_robustness_ratio"]
                        ),
                        "persistent_basin_lifetime": float(
                            row["basin_persistence"]["normalized_total_lifetime"]
                        ),
                        "evaluation_accuracy": evaluation_accuracy,
                        "calibration_accuracy": calibration_accuracy,
                    }
                )

    main_cells: list[dict[str, object]] = []
    for (architecture, rank), observations in sorted(cells.items()):

        def metric(name: str) -> tuple[float, float]:
            values = [float(item[name]) for item in observations]
            return _mean(values), _standard_error(values)

        capacity, capacity_se = metric("capacity_fraction")
        basin, basin_se = metric("basin_density")
        cycle, cycle_se = metric("cycle_density")
        entropy, entropy_se = metric("component_entropy")
        displacement, displacement_se = metric("displacement_entropy_rank")
        auc, auc_se = metric("capacity_auc")
        robustness, robustness_se = metric("capacity_robustness_ratio")
        lifetime, lifetime_se = metric("persistent_basin_lifetime")
        evaluation, evaluation_se = metric("evaluation_accuracy")
        calibration, calibration_se = metric("calibration_accuracy")
        main_cells.append(
            {
                "architecture": architecture,
                "visible_rank": rank,
                "model_count": len(observations),
                "mean_calibration_accuracy": calibration,
                "calibration_accuracy_standard_error": calibration_se,
                "mean_evaluation_accuracy": evaluation,
                "evaluation_accuracy_standard_error": evaluation_se,
                "mean_capacity_fraction": capacity,
                "capacity_standard_error": capacity_se,
                "mean_basin_density": basin,
                "basin_density_standard_error": basin_se,
                "mean_cycle_density": cycle,
                "cycle_density_standard_error": cycle_se,
                "mean_component_entropy": entropy,
                "component_entropy_standard_error": entropy_se,
                "mean_displacement_entropy_rank": displacement,
                "displacement_entropy_rank_standard_error": displacement_se,
                "mean_capacity_auc": auc,
                "capacity_auc_standard_error": auc_se,
                "mean_capacity_robustness_ratio": robustness,
                "capacity_robustness_ratio_standard_error": robustness_se,
                "mean_persistent_basin_lifetime": lifetime,
                "persistent_basin_lifetime_standard_error": lifetime_se,
            }
        )

    mismatches = [
        float(item["absolute_calibration_mismatch"])
        for item in selections
    ]
    return {
        "artifact_type": "qcollide_cifar_performance_matched_main_table",
        "schema_version": 1,
        "dataset": topology["claim_scope"]["dataset"],
        "target_accuracy": target_accuracy,
        "maximum_accuracy_mismatch": maximum_accuracy_mismatch,
        "visible_ranks": list(ranks),
        "architectures": architectures,
        "model_seeds": seeds,
        "checkpoint_selections": selections,
        "main_cells": main_cells,
        "gates": {
            "all_model_checkpoints_within_tolerance": all(
                bool(item["within_tolerance"]) for item in selections
            ),
            "maximum_observed_accuracy_mismatch": max(mismatches, default=0.0),
            "missing_topology_cell_count": len(missing_topology),
            "complete_topology_join": not missing_topology,
            "selected_model_count": len(selections),
        },
        "claim_boundary": {
            "checkpoint_interpolation_used": False,
            "architecture_causality_claimed": False,
            "final_checkpoint_only_claimed": False,
            "quantum_values_are": "not evaluated in this artifact",
        },
    }


__all__ = ["performance_matched_cifar_main_table"]
