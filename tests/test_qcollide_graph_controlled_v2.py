from __future__ import annotations

import numpy as np

from qgapselect.qcollide.graph_controlled_atlas import multioutput_r2
from qgapselect.qcollide.graph_controlled_results import merge_controlled_graph_components


def test_multioutput_r2_is_one_for_exact_predictions():
    expected = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
    mean, scores = multioutput_r2(expected, expected.copy())
    assert np.isclose(mean, 1.0)
    assert np.allclose(scores, 1.0)


def _component(architecture: str, seed: int, control_r2: float) -> dict[str, object]:
    point = {
        "control_epsilon": 1.0,
        "n_left": 4,
        "n_right": 4,
        "capacity_fraction": 0.5,
        "beta0_active": 2,
        "beta1_active": 1,
        "edge_count": 4,
        "matching_size": 2,
        "max_overlap_degree": 2,
    }
    return {
        "artifact_type": "qcollide_controlled_graph_functional_collision_component",
        "architecture": architecture,
        "model_seed": seed,
        "model_diagnostics": {
            "evaluation_accuracy": 0.75,
            "control_head_calibration_r2": control_r2,
        },
        "gates": {
            "control_r2_pass": control_r2 >= 0.2,
            "evaluation_accuracy_pass": True,
        },
        "rows": [
            {
                "visible_rank": 1,
                "nominal_epsilon": 1.0,
                "filtration_points": [point],
                "filtration_summary": {
                    "capacity_auc": 0.4,
                    "capacity_robustness_ratio": 0.6,
                },
                "basin_persistence": {"normalized_total_lifetime": 0.3},
            }
        ],
    }


def test_controlled_merge_fails_main_claim_gate_if_one_control_gate_fails():
    artifacts = [
        _component("gcn", 0, 0.3),
        _component("gcn", 1, 0.1),
        _component("gat", 0, 0.4),
        _component("gat", 1, 0.5),
    ]
    merged = merge_controlled_graph_components(
        artifacts,
        configured_architectures=["gcn", "gat"],
        configured_seeds=[0, 1],
    )
    assert merged["gates"]["complete_architecture_seed_grid"] is True
    assert merged["gates"]["all_control_r2_gates_pass"] is False
    assert merged["gates"]["eligible_for_universal_atlas_main_claim"] is False
