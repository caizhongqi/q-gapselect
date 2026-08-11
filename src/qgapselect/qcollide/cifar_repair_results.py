"""Aggregate prospective CIFAR collision-repair experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt

import numpy as np

from .main_statistics import exact_paired_sign_flip_test


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _se(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1) / sqrt(len(array))) if len(array) > 1 else 0.0


def merge_cifar_repair_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    architectures: Sequence[str],
    model_seeds: Sequence[int],
    closure_ranks: Sequence[int],
    primary_closure_rank: int,
    maximum_accuracy_loss: float,
) -> dict[str, object]:
    """Require the full registered grid and compare targeted to matched random repair."""

    expected = {
        (str(architecture), int(seed))
        for architecture in architectures
        for seed in model_seeds
    }
    observed = {
        (str(artifact["architecture"]), int(artifact["model_seed"]))
        for artifact in artifacts
    }
    if observed != expected:
        raise ValueError(
            f"repair component grid mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    if {str(artifact["artifact_type"]) for artifact in artifacts} != {
        "qcollide_cifar_prospective_repair_component"
    }:
        raise ValueError("unexpected repair component type")

    registered_ranks = tuple(int(value) for value in closure_ranks)
    if primary_closure_rank not in registered_ranks:
        raise ValueError("primary_closure_rank must be registered")

    raw_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for artifact in sorted(
        artifacts,
        key=lambda item: (str(item["architecture"]), int(item["model_seed"])),
    ):
        baseline = float(
            artifact["heldout_baseline_profile"]["nominal"]["capacity_fraction"]
        )
        component_rows.append(
            {
                "architecture": str(artifact["architecture"]),
                "model_seed": int(artifact["model_seed"]),
                "baseline_capacity_fraction": baseline,
                "baseline_evaluation_accuracy": float(
                    artifact["baseline_evaluation_accuracy"]
                ),
                "design_collision_edge_count": int(
                    artifact["dangerous_spectrum"]["collision_edge_count"]
                ),
                "design_rank_90": int(artifact["dangerous_spectrum"]["rank_90"]),
                "design_rank_95": int(artifact["dangerous_spectrum"]["rank_95"]),
            }
        )
        seen = set()
        for row in artifact["rows"]:
            key = (str(row["intervention"]), int(row["closure_rank"]))
            if key in seen:
                raise ValueError(f"duplicate repair row {key}")
            seen.add(key)
            raw_rows.append(
                {
                    "architecture": str(artifact["architecture"]),
                    "model_seed": int(artifact["model_seed"]),
                    "baseline_capacity_fraction": baseline,
                    **{
                        field: row[field]
                        for field in (
                            "intervention",
                            "closure_rank",
                            "effective_rank",
                            "evaluation_accuracy",
                            "accuracy_change",
                            "accuracy_loss",
                            "capacity_fraction",
                            "capacity_reduction",
                            "capacity_auc",
                            "capacity_auc_reduction",
                            "basin_density",
                            "control_drift_rms",
                            "removed_support_energy",
                            "accuracy_within_budget",
                        )
                    },
                }
            )
        expected_rows = {
            (intervention, rank)
            for intervention in ("targeted", "random_energy_matched", "variance_null")
            for rank in registered_ranks
        }
        if seen != expected_rows:
            raise ValueError(
                f"repair row grid mismatch for {artifact['architecture']}, "
                f"seed={artifact['model_seed']}"
            )

    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in raw_rows:
        groups.setdefault(
            (str(row["intervention"]), int(row["closure_rank"])),
            [],
        ).append(row)

    summary_cells: list[dict[str, object]] = []
    for (intervention, rank), rows in sorted(groups.items()):
        reductions = [float(row["capacity_reduction"]) for row in rows]
        auc_reductions = [float(row["capacity_auc_reduction"]) for row in rows]
        accuracy_losses = [float(row["accuracy_loss"]) for row in rows]
        control_drifts = [float(row["control_drift_rms"]) for row in rows]
        summary_cells.append(
            {
                "intervention": intervention,
                "closure_rank": rank,
                "model_count": len(rows),
                "mean_capacity_reduction": _mean(reductions),
                "capacity_reduction_standard_error": _se(reductions),
                "mean_capacity_auc_reduction": _mean(auc_reductions),
                "capacity_auc_reduction_standard_error": _se(auc_reductions),
                "mean_accuracy_loss": _mean(accuracy_losses),
                "accuracy_loss_standard_error": _se(accuracy_losses),
                "mean_control_drift_rms": _mean(control_drifts),
                "accuracy_budget_pass_fraction": _mean(
                    [float(bool(row["accuracy_within_budget"])) for row in rows]
                ),
                "positive_capacity_reduction_fraction": _mean(
                    [float(value > 0.0) for value in reductions]
                ),
            }
        )

    primary_targeted: list[float] = []
    primary_random: list[float] = []
    primary_pairs: list[dict[str, object]] = []
    for architecture in architectures:
        for seed in model_seeds:
            pair_rows = [
                row
                for row in raw_rows
                if row["architecture"] == architecture
                and int(row["model_seed"]) == int(seed)
                and int(row["closure_rank"]) == primary_closure_rank
            ]
            by_name = {str(row["intervention"]): row for row in pair_rows}
            targeted = float(by_name["targeted"]["capacity_reduction"])
            random = float(by_name["random_energy_matched"]["capacity_reduction"])
            primary_targeted.append(targeted)
            primary_random.append(random)
            primary_pairs.append(
                {
                    "architecture": str(architecture),
                    "model_seed": int(seed),
                    "targeted_capacity_reduction": targeted,
                    "random_capacity_reduction": random,
                    "targeted_advantage": targeted - random,
                    "targeted_accuracy_loss": float(
                        by_name["targeted"]["accuracy_loss"]
                    ),
                    "random_accuracy_loss": float(
                        by_name["random_energy_matched"]["accuracy_loss"]
                    ),
                    "targeted_accuracy_within_budget": bool(
                        by_name["targeted"]["accuracy_within_budget"]
                    ),
                }
            )

    exact_test = exact_paired_sign_flip_test(
        primary_targeted,
        primary_random,
        alternative="greater",
    )
    targeted_wins = sum(
        float(row["targeted_advantage"]) > 0.0 for row in primary_pairs
    )
    targeted_budget_passes = sum(
        bool(row["targeted_accuracy_within_budget"]) for row in primary_pairs
    )
    mean_advantage = _mean(
        [float(row["targeted_advantage"]) for row in primary_pairs]
    )

    return {
        "artifact_type": "qcollide_cifar_prospective_repair_utility_table",
        "schema_version": 1,
        "dataset": "CIFAR-10",
        "architectures": [str(value) for value in architectures],
        "model_seeds": [int(value) for value in model_seeds],
        "closure_ranks": list(registered_ranks),
        "primary_closure_rank": int(primary_closure_rank),
        "maximum_accuracy_loss": float(maximum_accuracy_loss),
        "component_rows": component_rows,
        "raw_rows": raw_rows,
        "summary_cells": summary_cells,
        "primary_pairs": primary_pairs,
        "primary_exact_paired_sign_flip_test": exact_test,
        "gates": {
            "complete_component_grid": True,
            "primary_pair_count": len(primary_pairs),
            "primary_targeted_win_count": int(targeted_wins),
            "primary_targeted_accuracy_budget_pass_count": int(
                targeted_budget_passes
            ),
            "mean_primary_targeted_advantage": mean_advantage,
            "targeted_beats_random_on_mean_capacity_reduction": mean_advantage > 0.0,
            "all_targeted_primary_rows_within_accuracy_budget": (
                targeted_budget_passes == len(primary_pairs)
            ),
        },
        "claim_boundary": {
            "basis_learned_from_calibration_only": True,
            "repair_evaluated_on_disjoint_heldout_collision_fixture": True,
            "post_intervention_threshold_recalibration": False,
            "paired_model_instances_share_dataset": True,
            "cross_dataset_repair_generalization_claimed": False,
            "model_weight_retraining_claimed": False,
            "hidden_representation_intervention_claimed": True,
        },
    }


__all__ = ["merge_cifar_repair_components"]
