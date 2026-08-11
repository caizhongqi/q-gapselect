"""Sensitivity analyses for performance-matched CIFAR collision tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt
from typing import Any

import numpy as np

_METRICS = (
    "calibration_accuracy",
    "evaluation_accuracy",
    "capacity_fraction",
    "basin_density",
    "cycle_density",
    "component_entropy",
    "displacement_entropy_rank",
    "capacity_auc",
    "capacity_robustness_ratio",
    "persistent_basin_lifetime",
)


def _mean_and_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("values must be non-empty")
    mean = float(array.mean())
    se = float(array.std(ddof=1) / sqrt(array.size)) if array.size > 1 else 0.0
    return mean, se


def strict_matched_subset_sensitivity(main_table: Mapping[str, object]) -> dict[str, object]:
    """Recompute cells using only checkpoints that passed the frozen accuracy gate."""

    if str(main_table.get("artifact_type")) != "qcollide_cifar_performance_matched_main_table":
        raise ValueError("unexpected CIFAR main-table artifact type")
    selections = list(main_table["checkpoint_selections"])
    observations = list(main_table["matched_observations"])
    allowed = {
        (str(row["architecture"]), int(row["model_seed"]))
        for row in selections
        if bool(row["within_tolerance"])
    }
    excluded = [
        {
            "architecture": str(row["architecture"]),
            "model_seed": int(row["model_seed"]),
            "checkpoint_epoch": int(row["checkpoint_epoch"]),
            "absolute_calibration_mismatch": float(row["absolute_calibration_mismatch"]),
        }
        for row in selections
        if not bool(row["within_tolerance"])
    ]
    retained = [
        row
        for row in observations
        if (str(row["architecture"]), int(row["model_seed"])) in allowed
    ]
    if not retained:
        raise ValueError("no observations passed the frozen matching gate")

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in retained:
        groups.setdefault((str(row["architecture"]), int(row["visible_rank"])), []).append(row)

    cells: list[dict[str, object]] = []
    for (architecture, rank), rows in sorted(groups.items()):
        cell: dict[str, object] = {
            "architecture": architecture,
            "visible_rank": rank,
            "model_count": len(rows),
        }
        for metric in _METRICS:
            mean, se = _mean_and_se([float(row[metric]) for row in rows])
            cell[f"mean_{metric}"] = mean
            cell[f"se_{metric}"] = se
        cells.append(cell)

    architecture_counts = {
        architecture: len({int(row["model_seed"]) for row in retained if row["architecture"] == architecture})
        for architecture in sorted({str(row["architecture"]) for row in retained})
    }
    return {
        "artifact_type": "qcollide_cifar_strict_matched_subset_sensitivity",
        "schema_version": 1,
        "dataset": str(main_table["dataset"]),
        "target_accuracy": float(main_table["target_accuracy"]),
        "frozen_maximum_accuracy_mismatch": float(main_table["maximum_accuracy_mismatch"]),
        "selected_model_count_original": len(selections),
        "selected_model_count_retained": len(allowed),
        "excluded_models": excluded,
        "architecture_model_counts": architecture_counts,
        "retained_observation_count": len(retained),
        "visible_ranks": [int(value) for value in main_table["visible_ranks"]],
        "main_cells": cells,
        "gates": {
            "posthoc_tolerance_relaxation_used": False,
            "all_retained_models_within_frozen_tolerance": True,
            "at_least_four_models_per_architecture": min(architecture_counts.values()) >= 4,
        },
        "claim_boundary": {
            "sensitivity_analysis_only": True,
            "excluded_models_selected_only_by_preregistered_accuracy_gate": True,
            "architecture_causality_claimed": False,
        },
    }


__all__ = ["strict_matched_subset_sensitivity"]
