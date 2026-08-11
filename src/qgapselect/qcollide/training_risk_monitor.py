"""Training-time utility: early collision topology as a final-capacity risk signal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        mean_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = mean_rank
        start = end
    return ranks


def _correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if len(left) < 2 or np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    return _correlation(_rankdata(np.asarray(x, dtype=float)), _rankdata(np.asarray(y, dtype=float)))


def _partial_architecture_accuracy(
    x: Sequence[float],
    y: Sequence[float],
    architectures: Sequence[str],
    accuracy: Sequence[float],
) -> float | None:
    names = sorted(set(str(value) for value in architectures))
    if len(names) < 2:
        return None
    architecture_columns = [
        [float(str(value) == name) for name in names[1:]]
        for value in architectures
    ]
    controls = np.column_stack(
        [
            np.ones(len(architectures), dtype=float),
            np.asarray(architecture_columns, dtype=float),
            np.asarray(accuracy, dtype=float),
        ]
    )
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    left_residual = left - controls @ np.linalg.lstsq(controls, left, rcond=None)[0]
    right_residual = right - controls @ np.linalg.lstsq(controls, right, rcond=None)[0]
    return _correlation(left_residual, right_residual)


def _nominal_point(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nominal = float(row["nominal_epsilon"])
    return min(
        row["filtration_points"],
        key=lambda point: abs(float(point["control_epsilon"]) - nominal),
    )


def build_training_risk_monitor_table(
    topology: Mapping[str, object],
    performance: Mapping[str, object],
    *,
    visible_ranks: Sequence[int] = (1, 8, 32),
    early_epochs: Sequence[int] = (3, 6, 9),
    final_epoch: int = 24,
) -> dict[str, object]:
    """Evaluate all frozen early epoch × rank cells without selecting a best cell."""

    if str(topology.get("artifact_type")) != "qcollide_cifar_functional_collision_topology_merged":
        raise ValueError("unexpected topology artifact type")
    if str(performance.get("artifact_type")) != "qcollide_cifar_checkpoint_performance_merged":
        raise ValueError("unexpected performance artifact type")

    performance_by_key = {
        (str(row["architecture"]), int(row["model_seed"]), int(row["checkpoint_epoch"])): row
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
    rows: list[dict[str, object]] = []
    for rank in [int(value) for value in visible_ranks]:
        final_capacity: dict[tuple[str, int], float] = {}
        for model in models:
            row = topology_by_key.get((model[0], model[1], final_epoch, rank))
            if row is None:
                raise ValueError(f"missing final topology cell for {model}, rank={rank}")
            final_capacity[model] = float(_nominal_point(row)["capacity_fraction"])

        for epoch in [int(value) for value in early_epochs]:
            architecture_values: list[str] = []
            early_accuracy: list[float] = []
            final_values: list[float] = []
            metrics: dict[str, list[float]] = {
                "capacity_fraction": [],
                "basin_density": [],
                "capacity_auc": [],
                "capacity_robustness_ratio": [],
                "persistent_basin_lifetime": [],
            }
            for model in models:
                topology_row = topology_by_key.get((model[0], model[1], epoch, rank))
                performance_row = performance_by_key.get((model[0], model[1], epoch))
                if topology_row is None or performance_row is None:
                    raise ValueError(f"missing early cell for {model}, epoch={epoch}, rank={rank}")
                point = _nominal_point(topology_row)
                denominator = max(1, min(int(point["n_left"]), int(point["n_right"])))
                architecture_values.append(model[0])
                early_accuracy.append(float(performance_row["calibration_accuracy"]))
                final_values.append(final_capacity[model])
                metrics["capacity_fraction"].append(float(point["capacity_fraction"]))
                metrics["basin_density"].append(float(point["beta0_active"]) / denominator)
                metrics["capacity_auc"].append(float(topology_row["filtration_summary"]["capacity_auc"]))
                metrics["capacity_robustness_ratio"].append(
                    float(topology_row["filtration_summary"]["capacity_robustness_ratio"])
                )
                metrics["persistent_basin_lifetime"].append(
                    float(topology_row["basin_persistence"]["normalized_total_lifetime"])
                )

            accuracy_spearman = _spearman(early_accuracy, final_values)
            for metric, values in metrics.items():
                metric_spearman = _spearman(values, final_values)
                rows.append(
                    {
                        "visible_rank": rank,
                        "early_epoch": epoch,
                        "final_epoch": final_epoch,
                        "model_count": len(models),
                        "metric": metric,
                        "metric_to_final_capacity_spearman": metric_spearman,
                        "early_accuracy_to_final_capacity_spearman": accuracy_spearman,
                        "spearman_gain_over_early_accuracy": (
                            metric_spearman - accuracy_spearman
                            if metric_spearman is not None and accuracy_spearman is not None
                            else None
                        ),
                        "partial_pearson_controlling_architecture_and_early_accuracy": (
                            _partial_architecture_accuracy(
                                values,
                                final_values,
                                architecture_values,
                                early_accuracy,
                            )
                        ),
                    }
                )

    capacity_rows = [row for row in rows if row["metric"] == "capacity_fraction"]
    high_rank = [row for row in capacity_rows if int(row["visible_rank"]) in {8, 32}]
    return {
        "artifact_type": "qcollide_training_risk_monitor_utility_table",
        "schema_version": 1,
        "dataset": str(topology["claim_scope"]["dataset"]),
        "visible_ranks": [int(value) for value in visible_ranks],
        "early_epochs": [int(value) for value in early_epochs],
        "final_epoch": final_epoch,
        "rows": rows,
        "summary": {
            "cell_count": len(rows),
            "capacity_monitor_cell_count": len(capacity_rows),
            "high_rank_capacity_cells_exceeding_early_accuracy_spearman": int(
                sum(
                    float(row["spearman_gain_over_early_accuracy"] or 0.0) > 0.0
                    for row in high_rank
                )
            ),
            "high_rank_capacity_cell_count": len(high_rank),
            "minimum_high_rank_capacity_partial_correlation": min(
                float(row["partial_pearson_controlling_architecture_and_early_accuracy"])
                for row in high_rank
                if row["partial_pearson_controlling_architecture_and_early_accuracy"] is not None
            ),
        },
        "claim_boundary": {
            "practical_use": "training_time_structural_risk_monitoring",
            "analysis_is_exploratory": True,
            "all_frozen_early_epoch_rank_cells_reported": True,
            "best_epoch_or_rank_selected_post_hoc": False,
            "architecture_fixed_effects_controlled_in_partial_correlation": True,
            "early_calibration_accuracy_controlled_in_partial_correlation": True,
            "causal_early_warning_claimed": False,
            "out_of_distribution_prediction_claimed": False,
        },
    }


__all__ = ["build_training_risk_monitor_table"]
