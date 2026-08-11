from __future__ import annotations

import numpy as np

from qgapselect.qcollide.time_atlas import time_control_descriptors
from qgapselect.qcollide.time_atlas_results import merge_time_atlas_components


def test_time_control_descriptors_are_finite_and_shape_stable():
    time = np.linspace(0.0, 4.0 * np.pi, 96)
    windows = np.stack(
        [
            np.column_stack([np.sin(time + shift), np.cos(time + shift), time / time.max()])
            for shift in (0.0, 0.2, 0.4)
        ]
    )
    descriptors = time_control_descriptors(windows, target_index=0)
    assert descriptors.shape == (3, 8)
    assert np.all(np.isfinite(descriptors))


def _component(architecture: str, seed: int, capacity: float) -> dict[str, object]:
    point = {
        "control_epsilon": 1.0,
        "n_left": 4,
        "n_right": 4,
        "capacity_fraction": capacity,
        "beta0_active": 2,
        "beta1_active": 1,
        "edge_count": 4,
        "matching_size": int(round(4 * capacity)),
        "normalized_component_edge_entropy": 0.5,
        "displacement_entropy_rank": 2.0,
    }
    return {
        "artifact_type": "qcollide_time_functional_collision_atlas_component",
        "architecture": architecture,
        "model_seed": seed,
        "model_diagnostics": {
            "forecast_mse": 0.4,
            "forecast_mae": 0.5,
            "control_head_calibration_r2": 0.3,
            "parameter_count": 100,
        },
        "rows": [
            {
                "visible_rank": 1,
                "nominal_epsilon": 1.0,
                "filtration_points": [point],
                "filtration_summary": {
                    "capacity_auc": 0.3,
                    "capacity_robustness_ratio": 0.7,
                },
                "basin_persistence": {"normalized_total_lifetime": 0.2},
            }
        ],
    }


def test_time_merge_requires_full_grid_and_aggregates():
    artifacts = [
        _component("lstm", 0, 0.25),
        _component("lstm", 1, 0.50),
        _component("tcn", 0, 0.75),
        _component("tcn", 1, 1.00),
    ]
    merged = merge_time_atlas_components(
        artifacts,
        configured_architectures=["lstm", "tcn"],
        configured_seeds=[0, 1],
    )
    assert merged["gates"]["complete_architecture_seed_grid"] is True
    assert merged["gates"]["all_rows_finite"] is True
    cells = {(row["architecture"], row["visible_rank"]): row for row in merged["atlas_cells"]}
    assert np.isclose(cells[("lstm", 1)]["mean_capacity_fraction"], 0.375)
    assert np.isclose(cells[("tcn", 1)]["mean_capacity_fraction"], 0.875)
