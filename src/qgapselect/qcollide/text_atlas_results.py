"""Merge and summarize cross-model text functional-collision atlas components."""

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


def merge_text_atlas_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    configured_models: Sequence[str],
    configured_seeds: Sequence[int],
) -> dict[str, object]:
    """Validate model × fixture-seed coverage and produce model/rank atlas cells."""

    if not artifacts:
        raise ValueError("artifacts must be non-empty")
    expected = {
        (str(model), int(seed))
        for model in configured_models
        for seed in configured_seeds
    }
    observed = {
        (str(artifact["model"]), int(artifact["fixture_seed"]))
        for artifact in artifacts
    }
    if observed != expected:
        raise ValueError(
            f"text atlas component grid mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    if {str(artifact["artifact_type"]) for artifact in artifacts} != {
        "qcollide_text_functional_collision_atlas_component"
    }:
        raise ValueError("unexpected text atlas component type")

    raw_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        diagnostics = artifact["model_diagnostics"]
        for row in artifact["rows"]:
            point = _nominal_point(row)
            denominator = max(
                1,
                min(int(point["n_left"]), int(point["n_right"])),
            )
            raw_rows.append(
                {
                    "model": str(artifact["model"]),
                    "fixture_seed": int(artifact["fixture_seed"]),
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
                    "component_entropy": float(
                        point["normalized_component_edge_entropy"]
                    ),
                    "displacement_entropy_rank": float(
                        point["displacement_entropy_rank"] or 0.0
                    ),
                    "maximum_degree": int(point["maximum_degree"]),
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
        groups.setdefault((str(row["model"]), int(row["visible_rank"])), []).append(row)
    cells: list[dict[str, object]] = []
    metric_names = (
        "evaluation_accuracy",
        "control_head_calibration_r2",
        "capacity_fraction",
        "basin_density",
        "within_basin_multiplicity",
        "cycle_density",
        "component_entropy",
        "displacement_entropy_rank",
        "maximum_degree",
        "capacity_auc",
        "capacity_robustness_ratio",
        "persistent_basin_lifetime",
    )
    for (model, visible_rank), observations in sorted(groups.items()):
        cell: dict[str, object] = {
            "model": model,
            "visible_rank": visible_rank,
            "fixture_count": len(observations),
            "parameter_count": int(observations[0]["parameter_count"]),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in observations]
            cell[f"mean_{metric}"] = _mean(values)
            cell[f"se_{metric}"] = _standard_error(values)
        cells.append(cell)

    nonzero = [float(row["capacity_fraction"]) > 0.0 for row in raw_rows]
    return {
        "artifact_type": "qcollide_text_functional_collision_atlas_merged",
        "schema_version": 1,
        "domain": "language_understanding",
        "dataset": "20newsgroups",
        "models": list(configured_models),
        "fixture_seeds": [int(value) for value in configured_seeds],
        "component_count": len(artifacts),
        "raw_rows": raw_rows,
        "atlas_cells": cells,
        "gates": {
            "complete_model_seed_grid": True,
            "model_count": len(configured_models),
            "fixture_seed_count": len(configured_seeds),
            "nonzero_capacity_row_fraction": float(np.mean(nonzero)),
            "all_rows_finite": bool(
                all(
                    np.isfinite(float(value))
                    for row in raw_rows
                    for key, value in row.items()
                    if key not in {"model"}
                )
            ),
        },
        "claim_boundary": {
            "fixture_seeds_are_independent_model_training_seeds": False,
            "encoders_are_frozen_pretrained_models": True,
            "classifier_is_linear_probe": True,
            "generated_attack_text": False,
            "cross_domain_universality_proved": False,
        },
    }


__all__ = ["merge_text_atlas_components"]
