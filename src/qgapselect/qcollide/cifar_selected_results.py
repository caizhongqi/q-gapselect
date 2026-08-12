"""Merge checkpoint-local CIFAR topology components into the final matched table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .cifar_main_results import performance_matched_cifar_main_table


def merge_selected_cifar_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    dataset: str,
    architectures: Sequence[str],
    model_seeds: Sequence[int],
    target_accuracy: float,
    maximum_accuracy_mismatch: float,
    visible_ranks: Sequence[int],
) -> dict[str, object]:
    """Fail closed on the full architecture × seed grid and build Table 1."""

    expected = {
        (str(architecture), int(seed))
        for architecture in architectures
        for seed in model_seeds
    }
    indexed: dict[tuple[str, int], Mapping[str, object]] = {}
    for artifact in artifacts:
        if str(artifact.get("artifact_type")) != (
            "qcollide_cifar_selected_checkpoint_topology_component"
        ):
            raise ValueError("unexpected selected CIFAR artifact type")
        key = (str(artifact["architecture"]), int(artifact["model_seed"]))
        if key in indexed:
            raise ValueError(f"duplicate selected CIFAR cell {key}")
        indexed[key] = artifact
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)
    if missing or extra:
        raise ValueError(
            f"selected CIFAR grid is incomplete: missing={missing}, extra={extra}"
        )

    topology_rows: list[Mapping[str, Any]] = []
    performance_rows: list[dict[str, object]] = []
    selected_cells: list[dict[str, object]] = []
    for key in sorted(expected):
        artifact = indexed[key]
        if str(artifact["claim_scope"]["dataset"]) != dataset:
            raise ValueError("selected CIFAR component dataset mismatch")
        performance = artifact["selected_performance"]
        if not bool(performance["within_tolerance"]):
            raise ValueError(f"performance-matching gate failed for {key}")
        checkpoint = int(artifact["selected_checkpoint_epoch"])
        topology_rows.extend(artifact["rows"])
        performance_rows.append(
            {
                "architecture": key[0],
                "model_seed": key[1],
                "checkpoint_epoch": checkpoint,
                "calibration_accuracy": float(performance["calibration_accuracy"]),
                "evaluation_accuracy": float(performance["evaluation_accuracy"]),
            }
        )
        selected_cells.append(
            {
                "architecture": key[0],
                "model_seed": key[1],
                "checkpoint_epoch": checkpoint,
                "calibration_accuracy": float(performance["calibration_accuracy"]),
                "evaluation_accuracy": float(performance["evaluation_accuracy"]),
                "absolute_calibration_mismatch": float(
                    performance["absolute_calibration_mismatch"]
                ),
                "selection_checkpoint_epoch": int(
                    artifact["selection"]["selection_checkpoint_epoch"]
                ),
            }
        )

    topology = {
        "artifact_type": "qcollide_cifar_functional_collision_topology_merged",
        "schema_version": 1,
        "claim_scope": {
            "dataset": dataset,
            "selected_checkpoint_only": True,
            "checkpoint_local_anchor_selection": True,
        },
        "rows": topology_rows,
    }
    performance = {
        "artifact_type": "qcollide_cifar_checkpoint_performance_merged",
        "schema_version": 1,
        "rows": performance_rows,
    }
    main_table = performance_matched_cifar_main_table(
        topology,
        performance,
        target_accuracy=target_accuracy,
        maximum_accuracy_mismatch=maximum_accuracy_mismatch,
        visible_ranks=visible_ranks,
    )
    return {
        "artifact_type": "qcollide_cifar_selected_checkpoint_main_bundle",
        "schema_version": 1,
        "dataset": dataset,
        "selected_cells": selected_cells,
        "selected_topology": topology,
        "selected_performance": performance,
        "main_table": main_table,
        "gates": {
            "complete_component_grid": True,
            "all_selected_checkpoints_within_tolerance": bool(
                main_table["gates"]["all_model_checkpoints_within_tolerance"]
            ),
            "complete_topology_join": bool(
                main_table["gates"]["complete_topology_join"]
            ),
            "selected_model_count": int(main_table["gates"]["selected_model_count"]),
        },
        "claim_boundary": {
            "final_epoch_anchors_reused": False,
            "checkpoint_interpolation_used": False,
            "performance_target_changed_after_dense_replay": False,
            "performance_tolerance_changed_after_dense_replay": False,
            "architecture_causality_claimed": False,
        },
    }


__all__ = ["merge_selected_cifar_components"]
