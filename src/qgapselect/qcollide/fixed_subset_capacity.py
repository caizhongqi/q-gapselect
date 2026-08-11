"""Fixed-subset survival sketch for packed/disjoint collision capacity.

Unlike the earlier Bernoulli-thinning contour, this construction never assumes
that a randomly retained endpoint set has been compacted into a free indexed
list. It samples fixed-size subsets of the original indexed domains.
"""

from __future__ import annotations

from math import exp, lgamma

import numpy as np


def _log_combination(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0)


def fixed_subset_survival_probability(n: int, matching_size: int, subset_size: int) -> float:
    """Exact survival probability for K disjoint pairs and two uniform r-subsets.

    The left and right subsets are chosen independently and uniformly without
    replacement. A collision survives when at least one of the K paired
    endpoints is selected on both sides.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= matching_size <= n:
        raise ValueError("matching_size must lie in [0,n]")
    if not 0 <= subset_size <= n:
        raise ValueError("subset_size must lie in [0,n]")
    if matching_size == 0 or subset_size == 0:
        return 0.0
    if subset_size == n:
        return 1.0

    log_denominator = _log_combination(n, subset_size)
    t_min = max(0, subset_size - (n - matching_size))
    t_max = min(matching_size, subset_size)
    no_collision = 0.0
    for selected_matched_left in range(t_min, t_max + 1):
        log_left_probability = (
            _log_combination(matching_size, selected_matched_left)
            + _log_combination(
                n - matching_size,
                subset_size - selected_matched_left,
            )
            - log_denominator
        )
        if n - selected_matched_left < subset_size:
            continue
        log_right_avoid = (
            _log_combination(n - selected_matched_left, subset_size)
            - log_denominator
        )
        no_collision += exp(log_left_probability + log_right_avoid)
    return float(min(1.0, max(0.0, 1.0 - no_collision)))


def fixed_subset_survival_bounds(
    n: int,
    matching_size: int,
    subset_size: int,
) -> dict[str, float]:
    """First/second-order union bounds for the fixed-subset survival event."""

    if n <= 0 or not 0 <= matching_size <= n or not 0 <= subset_size <= n:
        raise ValueError("invalid fixed-subset parameters")
    p = (subset_size / n) ** 2
    if n <= 1:
        pair_intersection = p * p
    else:
        endpoint_pair = subset_size * max(0, subset_size - 1) / (n * (n - 1))
        pair_intersection = endpoint_pair * endpoint_pair
    upper = min(1.0, matching_size * p)
    lower = max(
        0.0,
        matching_size * p
        - matching_size * max(0, matching_size - 1) * pair_intersection / 2.0,
    )
    return {
        "single_pair_survival_probability": float(p),
        "two_pair_joint_probability": float(pair_intersection),
        "bonferroni_lower": float(lower),
        "union_upper": float(upper),
    }


def choose_stable_subset_size(n: int, capacity_scale: int) -> dict[str, float | int]:
    """Choose r so p=(r/N)^2 is near 1/(12*kappa).

    The formal stability lemma uses the safe window
    1/(16*kappa) <= p <= 1/(8*kappa). If rounding cannot place r in that
    window, the function fails closed rather than pretending the scale theorem
    applies.
    """

    if n <= 0 or capacity_scale <= 0 or capacity_scale > n:
        raise ValueError("require 1 <= capacity_scale <= n")
    lower_r = int(np.ceil(n / np.sqrt(16.0 * capacity_scale)))
    upper_r = int(np.floor(n / np.sqrt(8.0 * capacity_scale)))
    if lower_r > upper_r or upper_r < 1:
        raise RuntimeError("domain is too small for the registered stable subset window")
    target = n / np.sqrt(12.0 * capacity_scale)
    subset_size = min(upper_r, max(lower_r, int(round(target))))
    p = (subset_size / n) ** 2
    lower_p = 1.0 / (16.0 * capacity_scale)
    upper_p = 1.0 / (8.0 * capacity_scale)
    if p < lower_p - 1e-15 or p > upper_p + 1e-15:
        raise RuntimeError("rounded subset size left the registered stable window")
    return {
        "subset_size": subset_size,
        "single_pair_survival_probability": float(p),
        "capacity_scale": capacity_scale,
        "lower_probability_bound": lower_p,
        "upper_probability_bound": upper_p,
    }


def invert_fixed_subset_survival(
    n: int,
    subset_size: int,
    observed_survival: float,
    *,
    minimum_capacity: int,
    maximum_capacity: int,
) -> dict[str, object]:
    """Invert the exact monotone survival curve over an integer capacity range."""

    if not 0.0 <= observed_survival <= 1.0:
        raise ValueError("observed_survival must lie in [0,1]")
    if not 0 <= minimum_capacity <= maximum_capacity <= n:
        raise ValueError("invalid capacity interval")
    curve = []
    for capacity in range(minimum_capacity, maximum_capacity + 1):
        probability = fixed_subset_survival_probability(n, capacity, subset_size)
        curve.append((capacity, probability))
    selected_capacity, selected_probability = min(
        curve,
        key=lambda item: (abs(item[1] - observed_survival), item[0]),
    )
    return {
        "estimated_capacity": int(selected_capacity),
        "estimated_survival_probability": float(selected_probability),
        "absolute_probability_residual": float(
            abs(selected_probability - observed_survival)
        ),
        "capacity_interval": [minimum_capacity, maximum_capacity],
    }


def simulate_fixed_subset_survival(
    n: int,
    matching_size: int,
    subset_size: int,
    *,
    trials: int,
    seed: int,
) -> float:
    """Monte Carlo survival without materializing endpoint subsets.

    First sample how many matched-left endpoints enter the left subset. Then
    sample whether the right subset contains at least one of their partners.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= matching_size <= n or not 0 <= subset_size <= n:
        raise ValueError("invalid fixed-subset parameters")
    if matching_size == 0 or subset_size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(trials):
        left_hits = int(
            rng.hypergeometric(
                matching_size,
                n - matching_size,
                subset_size,
            )
        )
        if left_hits == 0:
            continue
        right_hits = int(
            rng.hypergeometric(
                left_hits,
                n - left_hits,
                subset_size,
            )
        )
        successes += int(right_hits > 0)
    return successes / trials


def fixed_subset_quantum_query_proxy(
    n: int,
    capacity_scale: int,
    *,
    relative_epsilon: float,
    include_geometric_scale_scan: bool = False,
) -> dict[str, float | int | bool]:
    """Analytic record-query contour for the candidate capacity estimator.

    One coherent subset-survival predicate is assigned the known pair-detection
    query scale r^(2/3). Probability estimation contributes 1/epsilon. This is
    an analytic query contour, not a hardware runtime and not yet a proved gate-
    complexity theorem for arbitrary functional collision predicates.
    """

    if not 0.0 < relative_epsilon < 1.0:
        raise ValueError("relative_epsilon must lie in (0,1)")
    scale = choose_stable_subset_size(n, capacity_scale)
    subset_size = int(scale["subset_size"])
    detector_scale = subset_size ** (2.0 / 3.0)
    probability_precision = 3.0 * relative_epsilon / 128.0
    outer_calls = 1.0 / probability_precision
    base_proxy = detector_scale * outer_calls
    scale_count = int(np.ceil(np.log2(n))) + 1
    total = base_proxy * scale_count if include_geometric_scale_scan else base_proxy
    return {
        "subset_size": subset_size,
        "inner_pair_detection_scale": float(detector_scale),
        "registered_probability_precision": float(probability_precision),
        "outer_probability_estimation_call_proxy": float(outer_calls),
        "constant_factor_scale_query_proxy": float(base_proxy),
        "geometric_scale_count": scale_count,
        "include_geometric_scale_scan": bool(include_geometric_scale_scan),
        "total_query_proxy": float(total),
    }


__all__ = [
    "choose_stable_subset_size",
    "fixed_subset_quantum_query_proxy",
    "fixed_subset_survival_bounds",
    "fixed_subset_survival_probability",
    "invert_fixed_subset_survival",
    "simulate_fixed_subset_survival",
]
