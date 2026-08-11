from __future__ import annotations

from qgapselect.qcollide.training_risk_monitor import build_training_risk_monitor_table


def _topology_row(architecture: str, seed: int, epoch: int, capacity: float):
    return {
        "architecture": architecture,
        "model_seed": seed,
        "checkpoint_epoch": epoch,
        "visible_rank": 8,
        "nominal_epsilon": 1.0,
        "filtration_points": [
            {
                "control_epsilon": 1.0,
                "capacity_fraction": capacity,
                "beta0_active": int(round(capacity * 10)),
                "n_left": 10,
                "n_right": 10,
            }
        ],
        "filtration_summary": {
            "capacity_auc": capacity,
            "capacity_robustness_ratio": capacity,
        },
        "basin_persistence": {"normalized_total_lifetime": 1.0 - capacity},
    }


def test_training_risk_monitor_reports_all_requested_cells() -> None:
    model_values = [
        ("a", 0, 0.20, 0.25, 0.40),
        ("a", 1, 0.25, 0.30, 0.45),
        ("b", 0, 0.50, 0.35, 0.70),
        ("b", 1, 0.55, 0.40, 0.75),
    ]
    topology_rows = []
    performance_rows = []
    for architecture, seed, early_capacity, early_accuracy, final_capacity in model_values:
        topology_rows.extend(
            [
                _topology_row(architecture, seed, 3, early_capacity),
                _topology_row(architecture, seed, 24, final_capacity),
            ]
        )
        performance_rows.extend(
            [
                {
                    "architecture": architecture,
                    "model_seed": seed,
                    "checkpoint_epoch": 3,
                    "calibration_accuracy": early_accuracy,
                    "evaluation_accuracy": early_accuracy,
                },
                {
                    "architecture": architecture,
                    "model_seed": seed,
                    "checkpoint_epoch": 24,
                    "calibration_accuracy": 0.8,
                    "evaluation_accuracy": 0.8,
                },
            ]
        )
    topology = {
        "artifact_type": "qcollide_cifar_functional_collision_topology_merged",
        "claim_scope": {"dataset": "cifar10"},
        "rows": topology_rows,
    }
    performance = {
        "artifact_type": "qcollide_cifar_checkpoint_performance_merged",
        "rows": performance_rows,
    }
    result = build_training_risk_monitor_table(
        topology,
        performance,
        visible_ranks=(8,),
        early_epochs=(3,),
        final_epoch=24,
    )
    assert len(result["rows"]) == 5
    capacity = next(row for row in result["rows"] if row["metric"] == "capacity_fraction")
    assert capacity["metric_to_final_capacity_spearman"] > 0.99
    assert result["claim_boundary"]["causal_early_warning_claimed"] is False
    assert result["claim_boundary"]["best_epoch_or_rank_selected_post_hoc"] is False
