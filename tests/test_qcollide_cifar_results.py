from qgapselect.qcollide.cifar_results import merge_cifar_topology_components


def _row(architecture: str, seed: int, rank: int, epoch: int) -> dict[str, object]:
    metrics = {
        "n_left": 10,
        "n_right": 10,
        "edge_count": 4,
        "active_left_count": 4,
        "active_right_count": 4,
        "active_vertex_fraction": 0.4,
        "beta0_active": 4,
        "beta1_active": 0,
        "component_edge_sizes": [1, 1, 1, 1],
        "component_vertex_sizes": [2, 2, 2, 2],
        "component_edge_entropy": 1.0,
        "normalized_component_edge_entropy": 1.0,
        "effective_component_count": 4.0,
        "largest_component_edge_fraction": 0.25,
        "matching_size": 4,
        "capacity_fraction": 0.4,
        "independence_ratio": 1.0,
        "displacement_rank_95": 2,
        "displacement_entropy_rank": 1.8,
        "displacement_stable_rank": 1.5,
        "displacement_total_energy": 2.0,
    }
    return {
        "architecture": architecture,
        "model_seed": seed,
        "checkpoint_epoch": epoch,
        "visible_rank": rank,
        "mean_hidden_openness": 0.7,
        "hidden_tunnel_dimension": 127,
        "nominal_epsilon": 1.0,
        "payload_delta": 0.1,
        "behavior_gamma": 0.1,
        "control_epsilon_multipliers": [0.5, 1.0, 2.0],
        "filtration_summary": {
            "epsilon_min": 0.5,
            "epsilon_max": 2.0,
            "nominal_epsilon": 1.0,
            "point_count": 3,
            "edge_count_monotone": True,
            "capacity_monotone": True,
            "cycle_rank_monotone": True,
            "capacity_auc": 0.3,
            "basin_density_auc": 0.3,
            "cycle_density_auc": 0.0,
            "component_entropy_auc": 1.0,
            "displacement_entropy_rank_auc": 1.8,
            "capacity_onset_epsilon": 0.5,
            "half_max_capacity_epsilon": 1.0,
            "cycle_onset_epsilon": None,
            "peak_beta0": 4,
            "peak_beta0_epsilon": 1.0,
            "maximum_capacity_fraction": 0.4,
            "nominal_capacity_fraction": 0.4,
            "capacity_robustness_ratio": 0.75,
        },
        "basin_persistence": {
            "epsilon_max": 2.0,
            "interval_count": 4,
            "finite_interval_count": 0,
            "essential_interval_count": 4,
            "zero_lifetime_count": 0,
            "total_lifetime": 4.0,
            "normalized_total_lifetime": 0.2,
            "mean_lifetime": 1.0,
            "maximum_lifetime": 1.0,
            "persistence_entropy": 1.0,
            "effective_persistent_basin_count": 4.0,
            "half_window_persistent_basin_count": 4,
            "intervals": [],
        },
        "filtration_points": [
            {"control_epsilon": 0.5, **metrics},
            {"control_epsilon": 1.0, **metrics},
            {"control_epsilon": 2.0, **metrics},
        ],
        "valid_pair_fraction": 0.9,
    }


def _component(architecture: str) -> dict[str, object]:
    return {
        "artifact_type": "qcollide_cifar_functional_collision_topology_component",
        "schema_version": 1,
        "master_seed": 17,
        "architecture": architecture,
        "model_seed": 0,
        "claim_scope": {},
        "training": {
            "architecture": architecture,
            "parameter_count": 100,
            "evaluation_accuracy": 0.8,
        },
        "rows": [
            _row(architecture, 0, 1, 0),
            _row(architecture, 0, 1, 1),
        ],
    }


def test_merge_preserves_raw_rows_and_builds_fingerprints() -> None:
    architectures = ["resnet18", "vit_tiny", "mlp_mixer"]
    artifact = merge_cifar_topology_components(
        [_component(architecture) for architecture in architectures],
        config={"epochs": 1, "dataset": "cifar10", "architectures": architectures},
    )
    assert len(artifact["rows"]) == 6
    assert len(artifact["final_cells"]) == 3
    assert len(artifact["architecture_fingerprints"]) == 3
    assert artifact["gates"]["all_filtrations_monotone"] is True
    assert artifact["gates"]["all_architectures_present"] is True
