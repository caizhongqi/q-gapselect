from qgapselect.qcollide.cifar_selected_results import merge_selected_cifar_components


def _row(architecture, seed, checkpoint, rank, capacity):
    return {
        "architecture": architecture,
        "model_seed": seed,
        "checkpoint_epoch": checkpoint,
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


def _component(architecture, seed, checkpoint):
    return {
        "artifact_type": "qcollide_cifar_selected_checkpoint_topology_component",
        "architecture": architecture,
        "model_seed": seed,
        "selected_checkpoint_epoch": checkpoint,
        "claim_scope": {"dataset": "cifar100"},
        "selected_performance": {
            "calibration_accuracy": 0.30,
            "evaluation_accuracy": 0.31,
            "absolute_calibration_mismatch": 0.0,
            "within_tolerance": True,
        },
        "selection": {"selection_checkpoint_epoch": checkpoint},
        "rows": [
            _row(architecture, seed, checkpoint, rank, 0.2 + 0.01 * seed)
            for rank in (1, 8, 32)
        ],
    }


def test_merge_builds_complete_checkpoint_local_main_table():
    artifacts = [
        _component(architecture, seed, 7 + seed)
        for architecture in ("resnet18", "vit_tiny")
        for seed in (0, 1)
    ]
    result = merge_selected_cifar_components(
        artifacts,
        dataset="cifar100",
        architectures=("resnet18", "vit_tiny"),
        model_seeds=(0, 1),
        target_accuracy=0.30,
        maximum_accuracy_mismatch=0.05,
        visible_ranks=(1, 8, 32),
    )
    assert result["gates"]["complete_component_grid"] is True
    assert result["gates"]["all_selected_checkpoints_within_tolerance"] is True
    assert result["gates"]["complete_topology_join"] is True
    assert result["gates"]["selected_model_count"] == 4
    assert len(result["selected_topology"]["rows"]) == 12
    assert len(result["main_table"]["main_cells"]) == 6
    assert result["claim_boundary"]["final_epoch_anchors_reused"] is False


def test_merge_fails_closed_on_missing_component():
    artifacts = [_component("resnet18", 0, 7)]
    try:
        merge_selected_cifar_components(
            artifacts,
            dataset="cifar100",
            architectures=("resnet18",),
            model_seeds=(0, 1),
            target_accuracy=0.30,
            maximum_accuracy_mismatch=0.05,
            visible_ranks=(1, 8, 32),
        )
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("missing selected CIFAR component should fail closed")
