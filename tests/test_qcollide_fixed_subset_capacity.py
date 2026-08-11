from __future__ import annotations

import itertools

from qgapselect.qcollide.fixed_subset_capacity import (
    choose_stable_subset_size,
    fixed_subset_survival_bounds,
    fixed_subset_survival_probability,
    invert_fixed_subset_survival,
    simulate_fixed_subset_survival,
)


def _brute_survival(n: int, matching_size: int, subset_size: int) -> float:
    subsets = list(itertools.combinations(range(n), subset_size))
    marked_pairs = tuple((index, index) for index in range(matching_size))
    successful = 0
    total = 0
    for left in subsets:
        left_set = set(left)
        for right in subsets:
            right_set = set(right)
            total += 1
            successful += int(
                any(
                    left_index in left_set and right_index in right_set
                    for left_index, right_index in marked_pairs
                )
            )
    return successful / total


def test_exact_fixed_subset_probability_matches_bruteforce():
    for n in range(3, 7):
        for matching_size in range(1, n + 1):
            for subset_size in range(1, n):
                expected = _brute_survival(n, matching_size, subset_size)
                observed = fixed_subset_survival_probability(
                    n,
                    matching_size,
                    subset_size,
                )
                assert abs(observed - expected) < 1e-12


def test_bonferroni_bounds_contain_exact_survival():
    exact = fixed_subset_survival_probability(128, 12, 20)
    bounds = fixed_subset_survival_bounds(128, 12, 20)
    assert bounds["bonferroni_lower"] <= exact <= bounds["union_upper"]


def test_stable_subset_window_and_exact_inversion():
    n = 4096
    kappa = 64
    true_capacity = 91
    scale = choose_stable_subset_size(n, kappa)
    subset_size = int(scale["subset_size"])
    probability = fixed_subset_survival_probability(
        n,
        true_capacity,
        subset_size,
    )
    inverted = invert_fixed_subset_survival(
        n,
        subset_size,
        probability,
        minimum_capacity=kappa,
        maximum_capacity=2 * kappa,
    )
    assert inverted["estimated_capacity"] == true_capacity


def test_monte_carlo_matches_exact_probability():
    exact = fixed_subset_survival_probability(1024, 32, 52)
    simulated = simulate_fixed_subset_survival(
        1024,
        32,
        52,
        trials=30000,
        seed=20260811,
    )
    assert abs(simulated - exact) < 0.02
