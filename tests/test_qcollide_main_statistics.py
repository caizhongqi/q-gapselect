from qgapselect.qcollide.main_statistics import (
    basin_capacity_decomposition,
    exact_paired_sign_flip_test,
    paired_architecture_contrast,
)


def test_exact_sign_flip_all_positive_five_pairs_has_one_sided_p_one_over_32():
    result = exact_paired_sign_flip_test(
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        alternative="greater",
    )
    assert result["exact_p_value"] == 1.0 / 32.0
    assert result["mean_difference"] == 1.0


def _artifact():
    rows = []
    for seed in range(5):
        rows.extend(
            [
                {
                    "architecture": "vit_tiny",
                    "model_seed": seed,
                    "visible_rank": 8,
                    "capacity_fraction": 0.40 + seed * 0.01,
                    "basin_density": 0.20 + seed * 0.005,
                },
                {
                    "architecture": "resnet18",
                    "model_seed": seed,
                    "visible_rank": 8,
                    "capacity_fraction": 0.30 + seed * 0.01,
                    "basin_density": 0.15 + seed * 0.005,
                },
            ]
        )
    return {
        "artifact_type": "qcollide_cifar_performance_matched_main_table",
        "dataset": "cifar10",
        "visible_ranks": [8],
        "matched_observations": rows,
    }


def test_paired_architecture_contrast_uses_matched_seed_replicates():
    result = paired_architecture_contrast(
        _artifact(),
        left_architecture="vit_tiny",
        right_architecture="resnet18",
        visible_rank=8,
        metric="capacity_fraction",
        alternative="greater",
    )
    assert result["pair_count"] == 5
    assert result["exact_p_value"] == 1.0 / 32.0
    assert result["model_seeds"] == [0, 1, 2, 3, 4]


def test_basin_decomposition_closes_capacity_gap_exactly():
    result = basin_capacity_decomposition(
        _artifact(),
        left_architecture="vit_tiny",
        right_architecture="resnet18",
        visible_rank=8,
    )
    for row in result["per_seed"]:
        reconstructed = (
            row["basin_density_contribution"]
            + row["within_basin_multiplicity_contribution"]
        )
        assert abs(reconstructed - row["capacity_difference"]) < 1e-12
