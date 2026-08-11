from __future__ import annotations

import numpy as np

from qgapselect.qcollide.pretrained_vision_atlas import low_frequency_rgb_controls
from qgapselect.qcollide.pretrained_vision_results import merge_pretrained_vision_components


def test_low_frequency_rgb_controls_has_expected_shape():
    images = np.arange(2 * 32 * 32 * 3, dtype=np.float32).reshape(2, 32, 32, 3)
    controls = low_frequency_rgb_controls(images, 4)
    assert controls.shape == (2, 48)
    assert np.all(np.isfinite(controls))


def _component(architecture, seed, capacity):
    return {
        "artifact_type": "qcollide_pretrained_vision_functional_collision_component",
        "architecture": architecture,
        "fixture_seed": seed,
        "model_diagnostics": {
            "evaluation_accuracy": 0.8,
            "control_head_calibration_r2": 0.2,
        },
        "gates": {
            "evaluation_accuracy_pass": True,
            "control_r2_pass": True,
        },
        "rows": [
            {
                "visible_rank": 8,
                "nominal_epsilon": 1.0,
                "filtration_summary": {"capacity_auc": capacity},
                "filtration_points": [
                    {"control_epsilon": 1.0, "capacity_fraction": capacity}
                ],
            }
        ],
    }


def test_merge_requires_complete_quality_gated_grid():
    artifacts = [
        _component("resnet50", 1, 0.2),
        _component("resnet50", 2, 0.4),
        _component("vit_b_16", 1, 0.3),
        _component("vit_b_16", 2, 0.5),
    ]
    result = merge_pretrained_vision_components(
        artifacts,
        architectures=("resnet50", "vit_b_16"),
        fixture_seeds=(1, 2),
    )
    assert result["gates"]["complete_component_grid"] is True
    assert result["gates"]["all_architectures_pass_quality_gates"] is True
    rows = {row["architecture"]: row for row in result["architecture_rows"]}
    rank_row = rows["resnet50"]["rank_rows"][0]
    assert abs(rank_row["mean_capacity_fraction"] - 0.3) < 1e-12
