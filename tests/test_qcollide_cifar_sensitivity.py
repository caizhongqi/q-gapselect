from __future__ import annotations

import numpy as np

from qgapselect.qcollide.cifar_sensitivity import strict_matched_subset_sensitivity


def test_strict_subset_drops_only_failed_gate_models():
    table = {
        "artifact_type": "qcollide_cifar_performance_matched_main_table",
        "dataset": "cifar10",
        "target_accuracy": 0.58,
        "maximum_accuracy_mismatch": 0.04,
        "visible_ranks": [1],
        "checkpoint_selections": [
            {
                "architecture": "a",
                "model_seed": 0,
                "checkpoint_epoch": 1,
                "absolute_calibration_mismatch": 0.01,
                "within_tolerance": True,
            },
            {
                "architecture": "a",
                "model_seed": 1,
                "checkpoint_epoch": 1,
                "absolute_calibration_mismatch": 0.05,
                "within_tolerance": False,
            },
        ],
        "matched_observations": [
            {
                "architecture": "a",
                "model_seed": seed,
                "visible_rank": 1,
                "calibration_accuracy": 0.58,
                "evaluation_accuracy": 0.57,
                "capacity_fraction": capacity,
                "basin_density": 0.2,
                "cycle_density": 0.1,
                "component_entropy": 0.3,
                "displacement_entropy_rank": 2.0,
                "capacity_auc": 0.4,
                "capacity_robustness_ratio": 0.5,
                "persistent_basin_lifetime": 0.6,
            }
            for seed, capacity in [(0, 0.25), (1, 0.75)]
        ],
    }
    result = strict_matched_subset_sensitivity(table)
    assert result["selected_model_count_retained"] == 1
    assert result["excluded_models"][0]["model_seed"] == 1
    assert np.isclose(result["main_cells"][0]["mean_capacity_fraction"], 0.25)
    assert result["gates"]["posthoc_tolerance_relaxation_used"] is False
