import numpy as np

from qgapselect.qcollide.cifar_repair_calibrated import (
    apply_soft_closure,
    soft_strength_for_energy,
)
from qgapselect.qcollide.cifar_repair_calibrated_results import (
    merge_calibrated_repair_components,
)


def test_soft_closure_matches_registered_support_energy():
    hidden = np.asarray(
        [
            [2.0, 0.0, 1.0],
            [-2.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
        ]
    )
    center = hidden.mean(axis=0)
    basis = np.eye(3)[:, :1]
    centered = hidden - center
    full_energy = float(np.mean((centered[:, 0]) ** 2))
    budget = 0.25 * full_energy
    strength = soft_strength_for_energy(full_energy, budget)
    modified = apply_soft_closure(
        hidden,
        center=center,
        basis=basis,
        rank=1,
        strength=strength,
    )
    actual = float(np.mean(np.sum((modified - hidden) ** 2, axis=1)))
    assert abs(actual - budget) < 1e-12
    assert abs(strength - 0.5) < 1e-12


def _component(architecture: str, seed: int, targeted: float, random: float):
    base = {
        "artifact_type": "qcollide_cifar_calibrated_repair_component",
        "architecture": architecture,
        "model_seed": seed,
        "baseline_evaluation_accuracy": 0.8,
        "operating_point": {
            "visible_rank": 2,
            "epsilon_multiplier": 0.5,
            "design_capacity_fraction": 0.5,
        },
        "heldout_baseline_profile": {"nominal": {"capacity_fraction": 0.45}},
    }
    rows = []
    for name, reduction in (
        ("targeted_soft", targeted),
        ("random_high_energy_soft", random),
        ("variance_soft", random),
    ):
        rows.append(
            {
                "intervention": name,
                "closure_rank": 2,
                "energy_budget_fraction": 0.25,
                "capacity_reduction": reduction,
                "accuracy_loss": 0.01,
                "relative_energy_mismatch": 1e-12,
            }
        )
    return {**base, "rows": rows}


def test_calibrated_repair_merge_uses_nine_matched_model_instances():
    architectures = ["resnet18", "vit_tiny", "mlp_mixer"]
    seeds = [0, 1, 2]
    artifacts = [
        _component(architecture, seed, 0.10, 0.02)
        for architecture in architectures
        for seed in seeds
    ]
    merged = merge_calibrated_repair_components(
        artifacts,
        architectures=architectures,
        model_seeds=seeds,
        primary_closure_rank=2,
        primary_energy_budget_fraction=0.25,
        maximum_accuracy_loss=0.03,
        maximum_energy_mismatch=1e-8,
    )
    assert merged["gates"]["complete_component_grid"] is True
    assert merged["gates"]["primary_pair_count"] == 9
    assert merged["gates"]["all_registered_rows_energy_matched"] is True
    assert merged["gates"]["primary_targeted_accuracy_budget_pass_count"] == 9
    assert merged["gates"]["primary_targeted_win_count"] == 9
    assert merged["primary_exact_paired_sign_flip_test"]["exact_p_value"] == 1.0 / 512.0
