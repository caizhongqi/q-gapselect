from __future__ import annotations

import numpy as np

from qgapselect.qcollide.audio_atlas import audio_control_descriptors
from qgapselect.qcollide.audio_atlas_results import merge_audio_atlas_components


def test_audio_control_descriptors_are_finite_and_label_free_shape():
    time = np.linspace(0.0, 1.0, 16000, endpoint=False)
    waveforms = np.stack(
        [
            np.sin(2.0 * np.pi * frequency * time)
            for frequency in (220.0, 440.0, 880.0)
        ]
    )
    descriptors = audio_control_descriptors(waveforms)
    assert descriptors.shape == (3, 8)
    assert np.all(np.isfinite(descriptors))


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
        "normalized_component_edge_entropy": 0.5,
        "displacement_entropy_rank": 2.0,
    }
    return {
        "artifact_type": "qcollide_audio_functional_collision_atlas_component",
        "architecture": architecture,
        "fixture_seed": seed,
        "model_diagnostics": {
            "evaluation_accuracy": 0.75,
            "control_head_calibration_r2": control_r2,
            "parameter_count": 100,
        },
        "gates": {
            "evaluation_accuracy_pass": True,
            "control_r2_pass": control_r2 >= 0.1,
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


def test_audio_merge_preserves_fail_closed_quality_gate():
    artifacts = [
        _component("audio_cnn", 0, 0.2),
        _component("audio_cnn", 1, 0.2),
        _component("wav2vec2", 0, 0.2),
        _component("wav2vec2", 1, 0.05),
    ]
    merged = merge_audio_atlas_components(
        artifacts,
        configured_architectures=["audio_cnn", "wav2vec2"],
        configured_seeds=[0, 1],
    )
    assert merged["gates"]["complete_architecture_fixture_grid"] is True
    assert merged["gates"]["all_control_r2_gates_pass"] is False
    assert merged["gates"]["eligible_for_universal_atlas_main_claim"] is False
