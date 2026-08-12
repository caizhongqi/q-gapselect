"""Merge and test calibration-selected CIFAR repair v2 components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .main_statistics import exact_paired_sign_flip_test


def merge_calibrated_repair_components(
    artifacts: Sequence[Mapping[str, object]],
    *,
    architectures: Sequence[str],
    model_seeds: Sequence[int],
    primary_closure_rank: int,
    primary_energy_budget_fraction: float,
    maximum_accuracy_loss: float,
    maximum_energy_mismatch: float,
) -> dict[str, object]:
    expected = {(str(a), int(s)) for a in architectures for s in model_seeds}
    by_key: dict[tuple[str, int], Mapping[str, object]] = {}
    for artifact in artifacts:
        if str(artifact.get("artifact_type")) != "qcollide_cifar_calibrated_repair_component":
            raise ValueError("unexpected calibrated-repair artifact type")
        key = (str(artifact["architecture"]), int(artifact["model_seed"]))
        if key in by_key:
            raise ValueError(f"duplicate calibrated-repair component {key}")
        by_key[key] = artifact
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise ValueError(f"incomplete calibrated-repair grid; missing={missing}, extra={extra}")

    primary_pairs: list[dict[str, object]] = []
    targeted_values: list[float] = []
    random_values: list[float] = []
    all_rows: list[dict[str, object]] = []
    operating_rows: list[dict[str, object]] = []
    for architecture, seed in sorted(expected):
        artifact = by_key[(architecture, seed)]
        operating = artifact["operating_point"]
        operating_rows.append(
            {
                "architecture": architecture,
                "model_seed": seed,
                "visible_rank": int(operating["visible_rank"]),
                "epsilon_multiplier": float(operating["epsilon_multiplier"]),
                "design_capacity_fraction": float(operating["design_capacity_fraction"]),
                "heldout_capacity_fraction": float(
                    artifact["heldout_baseline_profile"]["nominal"]["capacity_fraction"]
                ),
                "baseline_evaluation_accuracy": float(artifact["baseline_evaluation_accuracy"]),
            }
        )
        rows = list(artifact["rows"])
        for row in rows:
            all_rows.append({"architecture": architecture, "model_seed": seed, **row})
        selected = [
            row
            for row in rows
            if int(row["closure_rank"]) == int(primary_closure_rank)
            and abs(
                float(row["energy_budget_fraction"])
                - float(primary_energy_budget_fraction)
            )
            <= 1e-12
        ]
        by_name = {str(row["intervention"]): row for row in selected}
        if {"targeted_soft", "random_high_energy_soft"} - set(by_name):
            raise ValueError(f"missing primary intervention rows for {(architecture, seed)}")
        targeted = by_name["targeted_soft"]
        random = by_name["random_high_energy_soft"]
        targeted_reduction = float(targeted["capacity_reduction"])
        random_reduction = float(random["capacity_reduction"])
        targeted_values.append(targeted_reduction)
        random_values.append(random_reduction)
        primary_pairs.append(
            {
                "architecture": architecture,
                "model_seed": seed,
                "targeted_capacity_reduction": targeted_reduction,
                "random_capacity_reduction": random_reduction,
                "targeted_advantage": targeted_reduction - random_reduction,
                "targeted_accuracy_loss": float(targeted["accuracy_loss"]),
                "random_accuracy_loss": float(random["accuracy_loss"]),
                "targeted_accuracy_within_budget": bool(
                    float(targeted["accuracy_loss"]) <= float(maximum_accuracy_loss)
                ),
                "targeted_energy_mismatch": float(targeted["relative_energy_mismatch"]),
                "random_energy_mismatch": float(random["relative_energy_mismatch"]),
            }
        )

    paired = exact_paired_sign_flip_test(
        targeted_values,
        random_values,
        alternative="greater",
    )
    max_energy_mismatch = max(
        float(row["relative_energy_mismatch"]) for row in all_rows
    )
    return {
        "artifact_type": "qcollide_cifar_calibrated_repair_utility_table",
        "schema_version": 2,
        "architectures": list(architectures),
        "model_seeds": [int(value) for value in model_seeds],
        "primary_closure_rank": int(primary_closure_rank),
        "primary_energy_budget_fraction": float(primary_energy_budget_fraction),
        "operating_points": operating_rows,
        "primary_pairs": primary_pairs,
        "primary_exact_paired_sign_flip_test": paired,
        "raw_rows": all_rows,
        "gates": {
            "complete_component_grid": True,
            "primary_pair_count": len(primary_pairs),
            "all_registered_rows_energy_matched": max_energy_mismatch
            <= float(maximum_energy_mismatch),
            "maximum_observed_energy_mismatch": max_energy_mismatch,
            "primary_targeted_accuracy_budget_pass_count": sum(
                bool(row["targeted_accuracy_within_budget"]) for row in primary_pairs
            ),
            "primary_targeted_win_count": sum(
                float(row["targeted_advantage"]) > 0.0 for row in primary_pairs
            ),
            "mean_primary_targeted_advantage": float(paired["mean_difference"]),
            "primary_exact_p_value": float(paired["exact_p_value"]),
        },
        "claim_boundary": {
            "operating_point_selected_from_calibration_only": True,
            "energy_matching_is_soft_and_exact_on_support": True,
            "model_weight_retraining_claimed": False,
            "cross_dataset_repair_generalization_claimed": False,
        },
    }


__all__ = ["merge_calibrated_repair_components"]
