"""Aggregation for frozen pretrained-vision functional-collision experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _nominal_capacity(row: Mapping[str, object]) -> float:
    nominal = float(row["nominal_epsilon"])
    point = min(
        row["filtration_points"],
        key=lambda item: abs(float(item["control_epsilon"]) - nominal),
    )
    return float(point["capacity_fraction"])


def merge_pretrained_vision_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    architectures: Sequence[str],
    fixture_seeds: Sequence[int],
) -> dict[str, object]:
    """Merge a complete pretrained-vision architecture × fixture grid."""

    expected = {
        (str(architecture), int(seed))
        for architecture in architectures
        for seed in fixture_seeds
    }
    indexed: dict[tuple[str, int], Mapping[str, object]] = {}
    for artifact in artifacts:
        if str(artifact.get("artifact_type")) != (
            "qcollide_pretrained_vision_functional_collision_component"
        ):
            raise ValueError("unexpected pretrained vision artifact type")
        key = (str(artifact["architecture"]), int(artifact["fixture_seed"]))
        if key in indexed:
            raise ValueError(f"duplicate pretrained vision cell {key}")
        indexed[key] = artifact
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)

    rows: list[dict[str, object]] = []
    for architecture in [str(value) for value in architectures]:
        cells = [
            indexed[(architecture, int(seed))]
            for seed in fixture_seeds
            if (architecture, int(seed)) in indexed
        ]
        if not cells:
            continue
        control_r2 = [float(cell["model_diagnostics"]["control_head_calibration_r2"]) for cell in cells]
        accuracy = [float(cell["model_diagnostics"]["evaluation_accuracy"]) for cell in cells]
        rank_rows: list[dict[str, object]] = []
        visible_ranks = sorted(
            {int(row["visible_rank"]) for cell in cells for row in cell["rows"]}
        )
        for rank in visible_ranks:
            values = [
                _nominal_capacity(
                    next(row for row in cell["rows"] if int(row["visible_rank"]) == rank)
                )
                for cell in cells
            ]
            auc_values = [
                float(
                    next(
                        row
                        for row in cell["rows"]
                        if int(row["visible_rank"]) == rank
                    )["filtration_summary"]["capacity_auc"]
                )
                for cell in cells
            ]
            rank_rows.append(
                {
                    "visible_rank": rank,
                    "mean_capacity_fraction": float(np.mean(values)),
                    "capacity_fraction_standard_error": (
                        float(np.std(values, ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else 0.0
                    ),
                    "mean_capacity_auc": float(np.mean(auc_values)),
                    "nonzero_capacity_seed_fraction": float(
                        np.mean(np.asarray(values, dtype=float) > 0.0)
                    ),
                }
            )
        rows.append(
            {
                "architecture": architecture,
                "fixture_count": len(cells),
                "mean_evaluation_accuracy": float(np.mean(accuracy)),
                "minimum_evaluation_accuracy": min(accuracy),
                "mean_control_r2": float(np.mean(control_r2)),
                "minimum_control_r2": min(control_r2),
                "all_accuracy_gates_pass": all(
                    bool(cell["gates"]["evaluation_accuracy_pass"]) for cell in cells
                ),
                "all_control_r2_gates_pass": all(
                    bool(cell["gates"]["control_r2_pass"]) for cell in cells
                ),
                "rank_rows": rank_rows,
            }
        )

    all_quality = bool(rows) and all(
        bool(row["all_accuracy_gates_pass"]) and bool(row["all_control_r2_gates_pass"])
        for row in rows
    )
    return {
        "artifact_type": "qcollide_pretrained_vision_atlas_merged",
        "schema_version": 1,
        "architectures": [str(value) for value in architectures],
        "fixture_seeds": [int(value) for value in fixture_seeds],
        "component_count": len(indexed),
        "architecture_rows": rows,
        "gates": {
            "complete_component_grid": not missing and not extra,
            "missing_cell_count": len(missing),
            "extra_cell_count": len(extra),
            "all_architectures_pass_quality_gates": all_quality,
        },
        "claim_boundary": {
            "frozen_imagenet_pretrained_encoders": True,
            "common_cifar10_linear_probe": True,
            "architecture_causality_claimed": False,
            "performance_matching_claimed": False,
            "universal_ordering_claimed": False,
        },
    }


__all__ = ["merge_pretrained_vision_components"]
