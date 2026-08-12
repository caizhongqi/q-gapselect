"""Merge deterministic CIFAR checkpoint-performance replay artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def merge_cifar_performance_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    configured_architectures: Sequence[str],
    configured_seeds: Sequence[int],
) -> dict[str, object]:
    """Validate and merge architecture-by-seed checkpoint performance rows."""

    if not artifacts:
        raise ValueError("artifacts must be non-empty")
    expected = {
        (architecture, int(seed))
        for architecture in configured_architectures
        for seed in configured_seeds
    }
    observed = {
        (str(artifact["architecture"]), int(artifact["model_seed"]))
        for artifact in artifacts
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"component grid mismatch: missing={missing}, extra={extra}")
    artifact_types = {str(artifact["artifact_type"]) for artifact in artifacts}
    schema_versions = {int(artifact["schema_version"]) for artifact in artifacts}
    master_seeds = {int(artifact["master_seed"]) for artifact in artifacts}
    if artifact_types != {"qcollide_cifar_checkpoint_performance_component"}:
        raise ValueError("unexpected performance component artifact type")
    if len(schema_versions) != 1 or len(master_seeds) != 1:
        raise ValueError("components must share schema version and master seed")

    rows = sorted(
        [row for artifact in artifacts for row in artifact["rows"]],
        key=lambda row: (
            str(row["architecture"]),
            int(row["model_seed"]),
            int(row["checkpoint_epoch"]),
        ),
    )
    training = sorted(
        [artifact["training"] for artifact in artifacts],
        key=lambda row: (str(row["architecture"]), int(row["model_seed"])),
    )
    checkpoints = sorted({int(row["checkpoint_epoch"]) for row in rows})
    summaries: list[dict[str, object]] = []
    for architecture in sorted(configured_architectures):
        for checkpoint in checkpoints:
            cell = [
                row
                for row in rows
                if row["architecture"] == architecture
                and int(row["checkpoint_epoch"]) == checkpoint
            ]
            evaluation = np.asarray(
                [float(row["evaluation_accuracy"]) for row in cell],
                dtype=float,
            )
            calibration = np.asarray(
                [float(row["calibration_accuracy"]) for row in cell],
                dtype=float,
            )
            summaries.append(
                {
                    "architecture": architecture,
                    "checkpoint_epoch": checkpoint,
                    "model_count": len(cell),
                    "mean_evaluation_accuracy": float(evaluation.mean()),
                    "evaluation_standard_error": (
                        float(evaluation.std(ddof=1) / np.sqrt(len(evaluation)))
                        if len(evaluation) > 1
                        else 0.0
                    ),
                    "mean_calibration_accuracy": float(calibration.mean()),
                    "calibration_standard_error": (
                        float(calibration.std(ddof=1) / np.sqrt(len(calibration)))
                        if len(calibration) > 1
                        else 0.0
                    ),
                }
            )
    consistency_errors = np.asarray(
        [float(artifact["final_accuracy_consistency_error"]) for artifact in artifacts],
        dtype=float,
    )
    return {
        "artifact_type": "qcollide_cifar_checkpoint_performance_merged",
        "schema_version": next(iter(schema_versions)),
        "master_seed": next(iter(master_seeds)),
        "claim_scope": {
            "deterministic_checkpoint_replay": True,
            "performance_matching_support": True,
            "topology_measured_in_this_artifact": False,
            "architecture_causality_claimed": False,
        },
        "component_count": len(artifacts),
        "training": training,
        "rows": rows,
        "checkpoint_summary": summaries,
        "gates": {
            "complete_component_grid": True,
            "maximum_final_accuracy_consistency_error": float(
                consistency_errors.max(initial=0.0)
            ),
            "checkpoint_count": len(checkpoints),
        },
    }


__all__ = ["merge_cifar_performance_components"]
