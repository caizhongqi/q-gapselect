"""Aggregate paired learned-vs-random text control-subspace falsification results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _se(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0


def merge_text_random_projection_null_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    configured_models: Sequence[str],
    configured_seeds: Sequence[int],
) -> dict[str, object]:
    """Validate full coverage and aggregate paired learned-minus-random effects."""

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
            f"text null grid mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    if {str(artifact["artifact_type"]) for artifact in artifacts} != {
        "qcollide_text_random_projection_null_component"
    }:
        raise ValueError("unexpected text null component type")

    raw_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        diagnostics = artifact["model_diagnostics"]
        for row in artifact["rows"]:
            summary = row["random_projection_summary"]
            capacity = summary["capacity_fraction"]
            auc = summary["capacity_auc"]
            persistence = summary["persistent_basin_lifetime"]
            raw_rows.append(
                {
                    "model": str(artifact["model"]),
                    "fixture_seed": int(artifact["fixture_seed"]),
                    "visible_rank": int(row["visible_rank"]),
                    "evaluation_accuracy": float(diagnostics["evaluation_accuracy"]),
                    "control_head_calibration_r2": float(
                        diagnostics["control_head_calibration_r2"]
                    ),
                    "learned_capacity_fraction": float(capacity["learned_value"]),
                    "random_capacity_fraction_mean": float(capacity["random_mean"]),
                    "capacity_delta": float(capacity["learned_minus_random_mean"]),
                    "learned_capacity_percentile": float(
                        capacity["learned_percentile_among_random"]
                    ),
                    "learned_capacity_auc": float(auc["learned_value"]),
                    "random_capacity_auc_mean": float(auc["random_mean"]),
                    "capacity_auc_delta": float(auc["learned_minus_random_mean"]),
                    "learned_persistence": float(persistence["learned_value"]),
                    "random_persistence_mean": float(persistence["random_mean"]),
                    "persistence_delta": float(
                        persistence["learned_minus_random_mean"]
                    ),
                }
            )

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        groups.setdefault((str(row["model"]), int(row["visible_rank"])), []).append(row)
    cells: list[dict[str, object]] = []
    numeric_metrics = (
        "evaluation_accuracy",
        "control_head_calibration_r2",
        "learned_capacity_fraction",
        "random_capacity_fraction_mean",
        "capacity_delta",
        "learned_capacity_percentile",
        "learned_capacity_auc",
        "random_capacity_auc_mean",
        "capacity_auc_delta",
        "learned_persistence",
        "random_persistence_mean",
        "persistence_delta",
    )
    for (model, rank), rows in sorted(groups.items()):
        cell: dict[str, object] = {
            "model": model,
            "visible_rank": rank,
            "fixture_count": len(rows),
            "capacity_delta_positive_fixture_fraction": float(
                np.mean([float(row["capacity_delta"]) > 0.0 for row in rows])
            ),
        }
        for metric in numeric_metrics:
            values = [float(row[metric]) for row in rows]
            cell[f"mean_{metric}"] = _mean(values)
            cell[f"se_{metric}"] = _se(values)
        cells.append(cell)

    deltas = [float(row["capacity_delta"]) for row in raw_rows]
    percentiles = [float(row["learned_capacity_percentile"]) for row in raw_rows]
    return {
        "artifact_type": "qcollide_text_random_projection_null_merged",
        "schema_version": 1,
        "domain": "language_understanding",
        "dataset": "20newsgroups",
        "component_count": len(artifacts),
        "models": [str(value) for value in configured_models],
        "fixture_seeds": [int(value) for value in configured_seeds],
        "raw_rows": raw_rows,
        "cells": cells,
        "global_summary": {
            "mean_learned_minus_random_capacity": _mean(deltas),
            "positive_capacity_delta_row_fraction": float(
                np.mean([value > 0.0 for value in deltas])
            ),
            "mean_learned_capacity_percentile": _mean(percentiles),
        },
        "gates": {
            "complete_model_seed_grid": True,
            "all_rows_finite": bool(
                all(
                    np.isfinite(float(value))
                    for row in raw_rows
                    for key, value in row.items()
                    if key != "model"
                )
            ),
        },
        "claim_boundary": {
            "random_projection_null_only": True,
            "does_not_rule_out_all_alternative_controls": True,
            "fixture_seeds_are_not_independent_pretraining_seeds": True,
        },
    }


__all__ = ["merge_text_random_projection_null_components"]
