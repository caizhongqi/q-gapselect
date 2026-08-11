from __future__ import annotations

import numpy as np

from qgapselect.qcollide.text_null_controls import random_orthogonal_projection
from qgapselect.qcollide.text_null_results import merge_text_random_projection_null_components


def test_random_projection_has_orthonormal_rows_and_is_reproducible():
    first = random_orthogonal_projection(12, 4, seed=17)
    second = random_orthogonal_projection(12, 4, seed=17)
    assert first.shape == (4, 12)
    assert np.allclose(first @ first.T, np.eye(4), atol=1e-10)
    assert np.allclose(first, second)


def _component(model: str, seed: int, delta: float) -> dict[str, object]:
    metric = {
        "random_mean": 0.25,
        "random_std": 0.1,
        "random_min": 0.1,
        "random_max": 0.4,
        "learned_value": 0.25 + delta,
        "learned_minus_random_mean": delta,
        "learned_percentile_among_random": 0.75,
    }
    return {
        "artifact_type": "qcollide_text_random_projection_null_component",
        "model": model,
        "fixture_seed": seed,
        "model_diagnostics": {
            "evaluation_accuracy": 0.6,
            "control_head_calibration_r2": 0.5,
        },
        "rows": [
            {
                "visible_rank": 1,
                "random_projection_summary": {
                    "capacity_fraction": metric,
                    "capacity_auc": metric,
                    "persistent_basin_lifetime": metric,
                },
            }
        ],
    }


def test_null_merge_is_paired_and_complete():
    artifacts = [
        _component("a", 0, 0.1),
        _component("a", 1, -0.1),
        _component("b", 0, 0.2),
        _component("b", 1, 0.0),
    ]
    merged = merge_text_random_projection_null_components(
        artifacts,
        configured_models=["a", "b"],
        configured_seeds=[0, 1],
    )
    assert merged["gates"]["complete_model_seed_grid"] is True
    assert merged["gates"]["all_rows_finite"] is True
    assert np.isclose(merged["global_summary"]["mean_learned_minus_random_capacity"], 0.05)
    assert np.isclose(merged["global_summary"]["positive_capacity_delta_row_fraction"], 0.5)
