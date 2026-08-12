from qgapselect.qcollide.fixed_subset_capacity import fixed_subset_survival_probability
from qgapselect.qcollide.fixed_subset_unknown_k import (
    geometric_capacity_scales,
    invert_survival_binary,
    scan_unknown_capacity,
)


def test_geometric_scales_descend_to_one():
    assert geometric_capacity_scales(16) == (16, 8, 4, 2, 1)


def test_binary_inversion_recovers_exact_capacity_probability():
    n = 256
    capacity = 24
    subset_size = 32
    survival = fixed_subset_survival_probability(n, capacity, subset_size)
    result = invert_survival_binary(n, subset_size, survival)
    assert result["estimated_capacity"] == capacity
    assert result["absolute_probability_residual"] < 1e-12


def test_unknown_capacity_scan_recovers_sparse_matching_fixture():
    result = scan_unknown_capacity(
        512,
        32,
        scan_trials=4096,
        estimation_trials=16384,
        seed=20260812,
        survival_low=0.05,
        survival_high=0.30,
    )
    assert result["relative_capacity_error"] <= 0.25
    assert len(result["observations"]) >= 1
