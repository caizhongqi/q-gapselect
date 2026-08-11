from __future__ import annotations

import numpy as np

from qgapselect.qcollide.graph_atlas import graph_control_descriptors
from qgapselect.qcollide.graph_atlas_results import merge_graph_atlas_components


def test_graph_control_descriptors_are_finite_and_label_agnostic_shape():
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    edge_index = np.array(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ]
    )
    descriptors = graph_control_descriptors(features, edge_index)
    assert descriptors.shape == (4, 6)
    assert np.all(np.isfinite(descriptors))
    assert np.all(descriptors[:, 0] > 0.0)


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
        "max_overlap_degree": 2,
    }
    return {
        "artifact_type": "qcollide_graph_functional_collision_atlas_component",
        "architecture": architecture,
        "model_seed": seed,
        "model_diagnostics": {
            "evaluation_accuracy": 0.8,
            "control_head_calibration_r2": 0.4,
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


def test_merge_graph_atlas_requires_complete_grid_and_aggregates():
    artifacts = [
        _component("gcn", 0, 0.25),
        _component("gcn", 1, 0.50),
        _component("gat", 0, 0.75),
        _component("gat", 1, 1.00),
    ]
    merged = merge_graph_atlas_components(
        artifacts,
        configured_architectures=["gcn", "gat"],
        configured_seeds=[0, 1],
    )
    assert merged["gates"]["complete_architecture_seed_grid"] is True
    assert merged["gates"]["all_rows_finite"] is True
    cells = {(row["architecture"], row["visible_rank"]): row for row in merged["atlas_cells"]}
    assert np.isclose(cells[("gcn", 1)]["mean_capacity_fraction"], 0.375)
    assert np.isclose(cells[("gat", 1)]["mean_capacity_fraction"], 0.875)
