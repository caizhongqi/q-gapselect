from qgapselect.qcollide.training_audit_priority import build_training_audit_priority_table


def _topology_row(architecture, seed, epoch, rank, capacity):
    return {
        "architecture": architecture,
        "model_seed": seed,
        "checkpoint_epoch": epoch,
        "visible_rank": rank,
        "nominal_epsilon": 1.0,
        "filtration_points": [
            {
                "control_epsilon": 1.0,
                "capacity_fraction": capacity,
            }
        ],
    }


def _fixtures():
    topology = {
        "artifact_type": "qcollide_cifar_functional_collision_topology_merged",
        "claim_scope": {"dataset": "cifar10"},
        "rows": [],
    }
    performance = {
        "artifact_type": "qcollide_cifar_checkpoint_performance_merged",
        "rows": [],
    }
    models = [
        ("a", 0, 0.90, 0.30),
        ("a", 1, 0.80, 0.40),
        ("b", 0, 0.20, 0.95),
        ("b", 1, 0.10, 0.85),
    ]
    for architecture, seed, early_capacity, early_accuracy in models:
        for epoch in (3, 6, 9):
            performance["rows"].append(
                {
                    "architecture": architecture,
                    "model_seed": seed,
                    "checkpoint_epoch": epoch,
                    "calibration_accuracy": early_accuracy,
                }
            )
            for rank in (1, 8, 32):
                topology["rows"].append(
                    _topology_row(architecture, seed, epoch, rank, early_capacity)
                )
        performance["rows"].append(
            {
                "architecture": architecture,
                "model_seed": seed,
                "checkpoint_epoch": 24,
                "calibration_accuracy": 0.9,
            }
        )
        final_capacity = 1.0 if architecture == "a" else 0.0
        for rank in (1, 8, 32):
            topology["rows"].append(
                _topology_row(architecture, seed, 24, rank, final_capacity)
            )
    return topology, performance


def test_collision_priority_captures_more_final_risk_than_accuracy():
    topology, performance = _fixtures()
    result = build_training_audit_priority_table(topology, performance)
    target = next(
        row
        for row in result["rows"]
        if row["visible_rank"] == 8
        and row["early_epoch"] == 3
        and abs(row["audit_fraction"] - 0.5) < 1e-12
    )
    assert target["early_collision_capacity_capture"] == 1.0
    assert target["early_accuracy_capture"] == 0.0
    assert target["collision_lift_over_accuracy"] == 1.0


def test_all_registered_grid_cells_are_reported():
    topology, performance = _fixtures()
    result = build_training_audit_priority_table(topology, performance)
    assert len(result["rows"]) == 27
    assert result["claim_boundary"]["best_cell_selected_post_hoc"] is False
