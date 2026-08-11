"""Aggregate control-validated Cora atlas v2 components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _se(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0


def _nominal_point(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nominal = float(row["nominal_epsilon"])
    return min(
        row["filtration_points"],
        key=lambda point: abs(float(point["control_epsilon"]) - nominal),
    )


def merge_controlled_graph_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    configured_architectures: Sequence[str],
    configured_seeds: Sequence[int],
) -> dict[str, object]:
    """Fail closed unless the full v2 grid and control-quality gates are satisfied."""

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
            f"controlled graph grid mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    if {str(artifact["artifact_type"]) for artifact in artifacts} != {
        "qcollide_controlled_graph_functional_collision_component"
    }:
        raise ValueError("unexpected controlled graph component type")

    component_gates = [
        {
            "architecture": str(artifact["architecture"]),
            "model_seed": int(artifact["model_seed"]),
            "control_r2": float(artifact["model_diagnostics"]["control_head_calibration_r2"]),
            "evaluation_accuracy": float(artifact["model_diagnostics"]["evaluation_accuracy"]),
            "control_r2_pass": bool(artifact["gates"]["control_r2_pass"]),
            "evaluation_accuracy_pass": bool(artifact["gates"]["evaluation_accuracy_pass"]),
        }
        for artifact in artifacts
    ]
    raw_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        diagnostics = artifact["model_diagnostics"]
        for row in artifact["rows"]:
            point = _nominal_point(row)
            denominator = max(1, min(int(point["n_left"]), int(point["n_right"])))
            beta0 = int(point["beta0_active"])
            raw_rows.append(
                {
                    "architecture": str(artifact["architecture"]),
                    "model_seed": int(artifact["model_seed"]),
                    "visible_rank": int(row["visible_rank"]),
                    "evaluation_accuracy": float(diagnostics["evaluation_accuracy"]),
                    "control_head_calibration_r2": float(
                        diagnostics["control_head_calibration_r2"]
                    ),
                    "capacity_fraction": float(point["capacity_fraction"]),
                    "basin_density": float(beta0) / denominator,
                    "within_basin_multiplicity": (
                        float(point["matching_size"]) / beta0 if beta0 > 0 else 0.0
                    ),
                    "cycle_density": float(point["beta1_active"])
                    / max(1, int(point["edge_count"])),
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

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        groups.setdefault((str(row["architecture"]), int(row["visible_rank"])), []).append(row)
    metrics = (
        "evaluation_accuracy",
        "control_head_calibration_r2",
        "capacity_fraction",
        "basin_density",
        "within_basin_multiplicity",
        "cycle_density",
        "capacity_auc",
        "capacity_robustness_ratio",
        "persistent_basin_lifetime",
        "max_overlap_degree",
    )
    cells: list[dict[str, object]] = []
    for (architecture, rank), rows in sorted(groups.items()):
        cell: dict[str, object] = {
            "architecture": architecture,
            "visible_rank": rank,
            "seed_count": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            cell[f"mean_{metric}"] = _mean(values)
            cell[f"se_{metric}"] = _se(values)
        cells.append(cell)

    all_control = all(row["control_r2_pass"] for row in component_gates)
    all_accuracy = all(row["evaluation_accuracy_pass"] for row in component_gates)
    return {
        "artifact_type": "qcollide_controlled_graph_atlas_v2_merged",
        "schema_version": 1,
        "domain": "graph_learning",
        "dataset": "cora",
        "architectures": [str(value) for value in configured_architectures],
        "model_seeds": [int(value) for value in configured_seeds],
        "component_count": len(artifacts),
        "component_gates": component_gates,
        "raw_rows": raw_rows,
        "atlas_cells": cells,
        "gates": {
            "complete_architecture_seed_grid": True,
            "all_control_r2_gates_pass": all_control,
            "all_evaluation_accuracy_gates_pass": all_accuracy,
            "eligible_for_universal_atlas_main_claim": all_control and all_accuracy,
            "nonzero_capacity_row_fraction": float(
                np.mean([float(row["capacity_fraction"]) > 0.0 for row in raw_rows])
            ),
        },
        "claim_boundary": {
            "main_claim_requires_all_quality_gates": True,
            "control_loss_changes_training_objective": True,
            "architecture_causality_claimed": False,
            "quantum_execution": False,
        },
    }


__all__ = ["merge_controlled_graph_components"]
