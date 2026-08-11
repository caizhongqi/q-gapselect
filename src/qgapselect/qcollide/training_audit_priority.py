"""Practical utility: prioritize limited training audits using early collision capacity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil
from typing import Any


def _nominal_point(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nominal = float(row["nominal_epsilon"])
    return min(
        row["filtration_points"],
        key=lambda point: abs(float(point["control_epsilon"]) - nominal),
    )


def _capture_fraction(
    records: Sequence[tuple[tuple[str, int], float, float]],
    *,
    budget_fraction: float,
) -> tuple[int, float, list[tuple[str, int]]]:
    if not 0.0 < budget_fraction <= 1.0:
        raise ValueError("budget_fraction must lie in (0,1]")
    count = max(1, int(ceil(budget_fraction * len(records))))
    ordered = sorted(records, key=lambda item: (-item[1], item[0]))
    selected = ordered[:count]
    total_risk = sum(float(item[2]) for item in records)
    captured = sum(float(item[2]) for item in selected)
    fraction = captured / total_risk if total_risk > 0.0 else 0.0
    return count, float(fraction), [item[0] for item in selected]


def build_training_audit_priority_table(
    topology: Mapping[str, object],
    performance: Mapping[str, object],
    *,
    visible_ranks: Sequence[int] = (1, 8, 32),
    early_epochs: Sequence[int] = (3, 6, 9),
    audit_fractions: Sequence[float] = (0.2, 0.4, 0.6),
    final_epoch: int = 24,
) -> dict[str, object]:
    """Compare early-topology vs early-accuracy audit prioritization.

    The operational endpoint is the fraction of final collision capacity captured
    by auditing only the top fixed fraction of model runs. All registered ranks,
    epochs and audit budgets are reported; no best cell is selected.
    """

    if str(topology.get("artifact_type")) != (
        "qcollide_cifar_functional_collision_topology_merged"
    ):
        raise ValueError("unexpected topology artifact type")
    if str(performance.get("artifact_type")) != (
        "qcollide_cifar_checkpoint_performance_merged"
    ):
        raise ValueError("unexpected performance artifact type")

    performance_by_key = {
        (
            str(row["architecture"]),
            int(row["model_seed"]),
            int(row["checkpoint_epoch"]),
        ): row
        for row in performance["rows"]
    }
    topology_by_key = {
        (
            str(row["architecture"]),
            int(row["model_seed"]),
            int(row["checkpoint_epoch"]),
            int(row["visible_rank"]),
        ): row
        for row in topology["rows"]
    }
    models = sorted({(key[0], key[1]) for key in performance_by_key})
    if not models:
        raise ValueError("no model runs found")

    rows: list[dict[str, object]] = []
    for rank in [int(value) for value in visible_ranks]:
        final_risk: dict[tuple[str, int], float] = {}
        for model in models:
            row = topology_by_key.get((model[0], model[1], final_epoch, rank))
            if row is None:
                raise ValueError(f"missing final topology cell for {model}, rank={rank}")
            final_risk[model] = float(_nominal_point(row)["capacity_fraction"])

        for epoch in [int(value) for value in early_epochs]:
            topology_records: list[tuple[tuple[str, int], float, float]] = []
            accuracy_records: list[tuple[tuple[str, int], float, float]] = []
            for model in models:
                topology_row = topology_by_key.get((model[0], model[1], epoch, rank))
                performance_row = performance_by_key.get((model[0], model[1], epoch))
                if topology_row is None or performance_row is None:
                    raise ValueError(
                        f"missing early cell for {model}, epoch={epoch}, rank={rank}"
                    )
                early_capacity = float(_nominal_point(topology_row)["capacity_fraction"])
                early_accuracy = float(performance_row["calibration_accuracy"])
                risk = final_risk[model]
                topology_records.append((model, early_capacity, risk))
                accuracy_records.append((model, early_accuracy, risk))

            for fraction in [float(value) for value in audit_fractions]:
                count, topology_capture, topology_selected = _capture_fraction(
                    topology_records,
                    budget_fraction=fraction,
                )
                _, accuracy_capture, accuracy_selected = _capture_fraction(
                    accuracy_records,
                    budget_fraction=fraction,
                )
                random_expectation = count / len(models)
                rows.append(
                    {
                        "visible_rank": rank,
                        "early_epoch": epoch,
                        "final_epoch": final_epoch,
                        "audit_fraction": fraction,
                        "audit_count": count,
                        "model_count": len(models),
                        "final_capacity_mass": float(sum(final_risk.values())),
                        "early_collision_capacity_capture": topology_capture,
                        "early_accuracy_capture": accuracy_capture,
                        "random_expected_capture": float(random_expectation),
                        "collision_lift_over_accuracy": topology_capture - accuracy_capture,
                        "collision_lift_over_random": topology_capture - random_expectation,
                        "collision_selected_models": [
                            {"architecture": model[0], "model_seed": model[1]}
                            for model in topology_selected
                        ],
                        "accuracy_selected_models": [
                            {"architecture": model[0], "model_seed": model[1]}
                            for model in accuracy_selected
                        ],
                    }
                )

    low_budget_high_rank = [
        row
        for row in rows
        if int(row["visible_rank"]) in {8, 32}
        and abs(float(row["audit_fraction"]) - 0.2) <= 1e-12
    ]
    low_budget_high_rank_wins = sum(
        float(row["collision_lift_over_accuracy"]) > 0.0
        for row in low_budget_high_rank
    )
    return {
        "artifact_type": "qcollide_training_audit_priority_utility_table",
        "schema_version": 1,
        "dataset": str(topology["claim_scope"]["dataset"]),
        "visible_ranks": [int(value) for value in visible_ranks],
        "early_epochs": [int(value) for value in early_epochs],
        "audit_fractions": [float(value) for value in audit_fractions],
        "final_epoch": final_epoch,
        "rows": rows,
        "summary": {
            "cell_count": len(rows),
            "low_budget_high_rank_cell_count": len(low_budget_high_rank),
            "low_budget_high_rank_cells_where_collision_beats_accuracy": int(
                low_budget_high_rank_wins
            ),
            "mean_low_budget_high_rank_lift_over_accuracy": (
                float(
                    sum(
                        float(row["collision_lift_over_accuracy"])
                        for row in low_budget_high_rank
                    )
                    / len(low_budget_high_rank)
                )
                if low_budget_high_rank
                else None
            ),
        },
        "claim_boundary": {
            "practical_use": "limited_training_audit_prioritization",
            "analysis_is_exploratory": True,
            "all_registered_ranks_epochs_and_budgets_reported": True,
            "best_cell_selected_post_hoc": False,
            "same_training_runs_used_for_priority_and_final_risk": True,
            "external_replication_claimed": False,
            "causal_early_warning_claimed": False,
        },
    }


__all__ = ["build_training_audit_priority_table"]
