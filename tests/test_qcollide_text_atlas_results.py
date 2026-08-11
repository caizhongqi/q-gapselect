from qgapselect.qcollide.text_atlas_results import merge_text_atlas_components


def _component(model, seed, capacity):
    return {
        "artifact_type": "qcollide_text_functional_collision_atlas_component",
        "model": model,
        "fixture_seed": seed,
        "model_diagnostics": {
            "evaluation_accuracy": 0.75,
            "control_head_calibration_r2": 0.40,
            "parameter_count": 100,
        },
        "rows": [
            {
                "visible_rank": 1,
                "nominal_epsilon": 1.0,
                "filtration_points": [
                    {
                        "control_epsilon": 1.0,
                        "n_left": 10,
                        "n_right": 10,
                        "edge_count": 5,
                        "beta0_active": 2,
                        "beta1_active": 1,
                        "matching_size": int(round(10 * capacity)),
                        "capacity_fraction": capacity,
                        "normalized_component_edge_entropy": 0.5,
                        "displacement_entropy_rank": 2.0,
                    }
                ],
                "filtration_summary": {
                    "capacity_auc": capacity,
                    "capacity_robustness_ratio": 0.8,
                },
                "basin_persistence": {"normalized_total_lifetime": 0.3},
            }
        ],
    }


def test_text_atlas_merger_requires_complete_grid_and_keeps_seed_rows():
    models = ["a", "b"]
    seeds = [1, 2]
    artifacts = [
        _component("a", 1, 0.2),
        _component("a", 2, 0.3),
        _component("b", 1, 0.1),
        _component("b", 2, 0.2),
    ]
    merged = merge_text_atlas_components(
        artifacts,
        configured_models=models,
        configured_seeds=seeds,
    )
    assert merged["gates"]["complete_model_seed_grid"] is True
    assert merged["component_count"] == 4
    assert len(merged["raw_rows"]) == 4
    assert len(merged["atlas_cells"]) == 2
    assert merged["gates"]["all_rows_finite"] is True
