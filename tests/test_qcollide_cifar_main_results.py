from qgapselect.qcollide.cifar_main_results import performance_matched_cifar_main_table


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
                "n_left": 10,
                "n_right": 10,
                "capacity_fraction": capacity,
                "beta0_active": 2,
                "beta1_active": 1,
                "edge_count": 4,
                "normalized_component_edge_entropy": 0.5,
                "displacement_entropy_rank": 2.0,
            }
        ],
        "filtration_summary": {
            "capacity_auc": capacity,
            "capacity_robustness_ratio": 0.8,
        },
        "basin_persistence": {"normalized_total_lifetime": 0.4},
    }


def _fixtures(include_all_topology=True):
    topology = {
        "artifact_type": "qcollide_cifar_functional_collision_topology_merged",
        "claim_scope": {"dataset": "cifar10"},
        "rows": [],
    }
    performance = {
        "artifact_type": "qcollide_cifar_checkpoint_performance_merged",
        "rows": [],
    }
    for architecture in ("resnet18", "vit_tiny"):
        for seed in (0, 1):
            performance["rows"].extend(
                [
                    {
                        "architecture": architecture,
                        "model_seed": seed,
                        "checkpoint_epoch": 3,
                        "calibration_accuracy": 0.50 + 0.01 * seed,
                        "evaluation_accuracy": 0.49 + 0.01 * seed,
                    },
                    {
                        "architecture": architecture,
                        "model_seed": seed,
                        "checkpoint_epoch": 6,
                        "calibration_accuracy": 0.59 + 0.01 * seed,
                        "evaluation_accuracy": 0.58 + 0.01 * seed,
                    },
                ]
            )
            for epoch in (3, 6):
                for rank in (1, 8, 32):
                    if not include_all_topology and architecture == "vit_tiny" and seed == 1 and rank == 32:
                        continue
                    topology["rows"].append(
                        _topology_row(architecture, seed, epoch, rank, 0.2 + 0.01 * seed)
                    )
    return topology, performance


def test_performance_matched_table_selects_observed_checkpoints():
    topology, performance = _fixtures()
    result = performance_matched_cifar_main_table(
        topology,
        performance,
        target_accuracy=0.58,
        maximum_accuracy_mismatch=0.03,
        visible_ranks=(1, 8, 32),
    )
    assert result["gates"]["all_model_checkpoints_within_tolerance"] is True
    assert result["gates"]["complete_topology_join"] is True
    assert result["gates"]["selected_model_count"] == 4
    assert len(result["main_cells"]) == 6
    assert {item["checkpoint_epoch"] for item in result["checkpoint_selections"]} == {6}


def test_performance_matched_table_fails_closed_on_missing_topology():
    topology, performance = _fixtures(include_all_topology=False)
    result = performance_matched_cifar_main_table(
        topology,
        performance,
        target_accuracy=0.58,
        maximum_accuracy_mismatch=0.03,
        visible_ranks=(1, 8, 32),
    )
    assert result["gates"]["complete_topology_join"] is False
    assert result["gates"]["missing_topology_cell_count"] == 1
