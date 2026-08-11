"""Merge ETTm1 time-series collision atlas components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _nominal_point(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nominal = float(row["nominal_epsilon"])
    return min(
        row["filtration_points"],
        key=lambda point: abs(float(point["control_epsilon"]) - nominal),
    )


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _se(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0


def merge_time_atlas_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    configured_architectures: Sequence[str],
    configured_seeds: Sequence[int],
) -> dict[str, object]:
    """Validate full architecture×seed coverage and aggregate time-series cells."""

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
            f"time atlas component grid mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    if {str(artifact["artifact_type"]) for artifact in artifacts} != {
        "qcollide_time_functional_collision_atlas_component"
    }:
        raise ValueError("unexpected time atlas component type")

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
                    "forecast_mse": float(diagnostics["forecast_mse"]),
                    "forecast_mae": float(diagnostics["forecast_mae"]),
                    "control_head_calibration_r2": float(
                        diagnostics["control_head_calibration_r2"]
                    ),
                    "parameter_count": int(diagnostics["parameter_count"]),
                    "capacity_fraction": float(point["capacity_fraction"]),
                    "basin_density": float(beta0) / denominator,
                    "within_basin_multiplicity": (
                        float(point["matching_size"]) / beta0 if beta0 > 0 else 0.0
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
                }
            )

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        groups.setdefault((str(row["architecture"]), int(row["visible_rank"])), []).append(row)
    metric_names = (
        "forecast_mse",
        "forecast_mae",
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
    )
    cells: list[dict[str, object]] = []
    for (architecture, visible_rank), rows in sorted(groups.items()):
        cell: dict[str, object] = {
            "architecture": architecture,
            "visible_rank": visible_rank,
            "seed_count": len(rows),
            "parameter_count_mean": _mean([float(row["parameter_count"]) for row in rows]),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            cell[f"mean_{metric}"] = _mean(values)
            cell[f"se_{metric}"] = _se(values)
        cells.append(cell)

    finite = [
        float(value)
        for row in raw_rows
        for key, value in row.items()
        if key != "architecture"
    ]
    return {
        "artifact_type": "qcollide_time_functional_collision_atlas_merged",
        "schema_version": 1,
        "domain": "time_series",
        "dataset": "ETTm1",
        "architectures": [str(value) for value in configured_architectures],
        "model_seeds": [int(value) for value in configured_seeds],
        "component_count": len(artifacts),
        "raw_rows": raw_rows,
        "atlas_cells": cells,
        "gates": {
            "complete_architecture_seed_grid": True,
            "all_rows_finite": bool(all(np.isfinite(value) for value in finite)),
            "nonzero_capacity_row_fraction": float(
                np.mean([float(row["capacity_fraction"]) > 0.0 for row in raw_rows])
            ),
        },
        "claim_boundary": {
            "compact_architecture_reimplementations": True,
            "performance_matching_not_yet_applied": True,
            "cross_domain_universality_proved": False,
        },
    }


__all__ = ["merge_time_atlas_components"]
