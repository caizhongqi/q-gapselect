"""Merge CIFAR-100 broad-pool single-run components into the matched main table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .cifar_main_results import performance_matched_cifar_main_table


def merge_broad_pool_cifar_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    dataset: str,
    architectures: Sequence[str],
    model_seeds: Sequence[int],
    target_accuracy: float,
    maximum_accuracy_mismatch: float,
    visible_ranks: Sequence[int],
    minimum_unique_fine_classes: int,
) -> dict[str, object]:
    expected = {
        (str(architecture), int(seed))
        for architecture in architectures
        for seed in model_seeds
    }
    indexed: dict[tuple[str, int], Mapping[str, object]] = {}
    for artifact in artifacts:
        if str(artifact.get("artifact_type")) != "qcollide_cifar_broad_pool_selected_component":
            raise ValueError("unexpected broad-pool CIFAR artifact type")
        key = (str(artifact["architecture"]), int(artifact["model_seed"]))
        if key in indexed:
            raise ValueError(f"duplicate broad-pool CIFAR cell {key}")
        indexed[key] = artifact
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)
    if missing or extra:
        raise ValueError(f"broad-pool CIFAR grid incomplete: missing={missing}, extra={extra}")

    topology_rows: list[Mapping[str, Any]] = []
    performance_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for key in sorted(expected):
        artifact = indexed[key]
        performance = artifact["selected_performance"]
        if not bool(performance["within_tolerance"]):
            raise ValueError(f"performance gate failed for {key}")
        anchor = artifact["selection"]["anchor_coverage"]
        candidate = artifact["selection"]["candidate_coverage"]
        anchor_classes = int(anchor["unique_fine_classes"])
        candidate_classes = int(candidate["unique_fine_classes"])
        if min(anchor_classes, candidate_classes) < int(minimum_unique_fine_classes):
            raise ValueError(f"fine-class coverage gate failed for {key}")
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
        coverage_rows.append(
            {
                "architecture": key[0],
                "model_seed": key[1],
                "checkpoint_epoch": checkpoint,
                "anchor_unique_fine_classes": anchor_classes,
                "candidate_unique_fine_classes": candidate_classes,
                "anchor_pool_size": int(anchor["pool_size"]),
                "candidate_pool_size": int(candidate["pool_size"]),
                "calibration_accuracy": float(performance["calibration_accuracy"]),
                "evaluation_accuracy": float(performance["evaluation_accuracy"]),
            }
        )

    topology = {
        "artifact_type": "qcollide_cifar_functional_collision_topology_merged",
        "schema_version": 1,
        "claim_scope": {
            "dataset": dataset,
            "selected_checkpoint_only": True,
            "fixed_broad_coverage_pools": True,
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
        "artifact_type": "qcollide_cifar100_broad_pool_main_bundle",
        "schema_version": 1,
        "dataset": dataset,
        "coverage_rows": coverage_rows,
        "selected_topology": topology,
        "selected_performance": performance,
        "main_table": main_table,
        "gates": {
            "complete_component_grid": True,
            "all_performance_matched": bool(
                main_table["gates"]["all_model_checkpoints_within_tolerance"]
            ),
            "all_class_coverage_gates_pass": True,
            "minimum_observed_anchor_class_coverage": min(
                int(row["anchor_unique_fine_classes"]) for row in coverage_rows
            ),
            "minimum_observed_candidate_class_coverage": min(
                int(row["candidate_unique_fine_classes"]) for row in coverage_rows
            ),
            "complete_topology_join": bool(main_table["gates"]["complete_topology_join"]),
            "selected_model_count": int(main_table["gates"]["selected_model_count"]),
        },
        "claim_boundary": {
            "all_100_fine_classes_required": False,
            "pool_size_fixed_across_architectures": True,
            "coverage_gate_fixed_before_v5_results": True,
            "architecture_causality_claimed": False,
        },
    }


__all__ = ["merge_broad_pool_cifar_components"]
