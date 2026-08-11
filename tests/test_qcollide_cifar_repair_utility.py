from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("sklearn")

from qgapselect.qcollide.cifar_repair_results import merge_cifar_repair_components
from qgapselect.qcollide.cifar_repair_utility import (
    _apply_closure,
    _energy_matched_random_basis,
)


def test_control_null_closure_preserves_control_coordinate():
    rng = np.random.default_rng(7)
    hidden = rng.normal(size=(50, 5))
    center = hidden.mean(axis=0)
    projection = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]])
    basis = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    modified = _apply_closure(
        hidden,
        center=center,
        basis=basis,
        rank=2,
    )
    np.testing.assert_allclose(
        modified @ projection.T,
        hidden @ projection.T,
        atol=1e-12,
        rtol=0.0,
    )


def test_energy_matched_random_basis_matches_support_energy():
    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(5000, 8))
    center = hidden.mean(axis=0)
    projection = np.zeros((1, 8), dtype=float)
    projection[0, 0] = 1.0
    target = np.zeros((8, 3), dtype=float)
    target[1, 0] = 1.0
    target[2, 1] = 1.0
    target[3, 2] = 1.0
    basis, diagnostics = _energy_matched_random_basis(
        projection=projection,
        hidden_support=hidden,
        center=center,
        target_basis=target,
        rank=2,
        candidates=128,
        seed=17,
    )
    assert basis.shape[1] >= 2
    assert diagnostics["target_removed_energy"] > 0.0
    assert diagnostics["random_removed_energy"] > 0.0
    assert diagnostics["relative_energy_mismatch"] < 0.20


def _repair_component(architecture: str, seed: int) -> dict[str, object]:
    rows = []
    for closure_rank in (1, 2, 4):
        for intervention, reduction, accuracy_loss in (
            ("targeted", 0.08 * closure_rank, 0.01),
            ("random_energy_matched", 0.02 * closure_rank, 0.01),
            ("variance_null", 0.03 * closure_rank, 0.015),
        ):
            rows.append(
                {
                    "intervention": intervention,
                    "closure_rank": closure_rank,
                    "effective_rank": closure_rank,
                    "evaluation_accuracy": 0.80 - accuracy_loss,
                    "accuracy_change": -accuracy_loss,
                    "accuracy_loss": accuracy_loss,
                    "capacity_fraction": 0.50 - reduction,
                    "capacity_reduction": reduction,
                    "capacity_auc": 0.60 - reduction,
                    "capacity_auc_reduction": reduction,
                    "basin_density": 0.20,
                    "control_drift_rms": 1e-12,
                    "removed_support_energy": float(closure_rank),
                    "accuracy_within_budget": accuracy_loss <= 0.03,
                }
            )
    return {
        "artifact_type": "qcollide_cifar_prospective_repair_component",
        "architecture": architecture,
        "model_seed": seed,
        "heldout_baseline_profile": {
            "nominal": {"capacity_fraction": 0.50},
        },
        "baseline_evaluation_accuracy": 0.80,
        "dangerous_spectrum": {
            "collision_edge_count": 20,
            "rank_90": 3,
            "rank_95": 4,
        },
        "rows": rows,
    }


def test_repair_merge_uses_complete_paired_model_grid():
    architectures = ("resnet18", "vit_tiny", "mlp_mixer")
    seeds = (0, 1, 2)
    artifacts = [
        _repair_component(architecture, seed)
        for architecture in architectures
        for seed in seeds
    ]
    merged = merge_cifar_repair_components(
        artifacts,
        architectures=architectures,
        model_seeds=seeds,
        closure_ranks=(1, 2, 4),
        primary_closure_rank=2,
        maximum_accuracy_loss=0.03,
    )
    assert merged["gates"]["complete_component_grid"] is True
    assert merged["gates"]["primary_pair_count"] == 9
    assert merged["gates"]["primary_targeted_win_count"] == 9
    assert merged["gates"]["all_targeted_primary_rows_within_accuracy_budget"] is True
    assert merged["gates"]["mean_primary_targeted_advantage"] > 0.0
    assert merged["primary_exact_paired_sign_flip_test"]["exact_p_value"] == 1.0 / 512.0
