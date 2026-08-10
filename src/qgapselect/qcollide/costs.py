"""Transparent analytic cost models used by the frozen calibration suite."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil, inf, sqrt

import numpy as np


@dataclass(frozen=True)
class JohnsonWalkCost:
    setup_size: int
    setup_cost: float
    walk_cost: float
    total_cost: float
    asymptotic_proxy: float


def pair_grover_cost(n_left: int, n_right: int, marked_pairs: int) -> float:
    if min(n_left, n_right, marked_pairs) <= 0:
        raise ValueError("domain sizes and marked-pair count must be positive")
    if marked_pairs > n_left * n_right:
        raise ValueError("marked-pair count exceeds pair domain")
    return sqrt((n_left * n_right) / marked_pairs)


def classical_packed_cost(n: int, matching_size: int) -> float:
    if n <= 0 or not 1 <= matching_size <= n:
        raise ValueError("require n > 0 and 1 <= matching_size <= n")
    return n / sqrt(matching_size)


def product_johnson_cost(n: int, matching_size: int) -> JohnsonWalkCost:
    """Minimize r + n/sqrt(nu*r) over integer subset sizes."""

    if n <= 0 or not 1 <= matching_size <= n:
        raise ValueError("require n > 0 and 1 <= matching_size <= n")
    continuous = (n * n / matching_size) ** (1.0 / 3.0)
    candidates = {
        1,
        n,
        max(1, min(n, int(continuous))),
        max(1, min(n, int(ceil(continuous)))),
    }
    best: JohnsonWalkCost | None = None
    for r in sorted(candidates):
        walk = n / sqrt(matching_size * r)
        total = r + walk
        candidate = JohnsonWalkCost(
            setup_size=r,
            setup_cost=float(r),
            walk_cost=float(walk),
            total_cost=float(total),
            asymptotic_proxy=float(continuous),
        )
        if best is None or candidate.total_cost < best.total_cost:
            best = candidate
    assert best is not None
    return best


def prefix_survival_rates(
    left_prefixes: Iterable[tuple[tuple[int, ...], ...]],
    right_prefixes: Iterable[tuple[tuple[int, ...], ...]],
) -> tuple[float, ...]:
    """Return s_l = P[first l prefixes agree], including s_0 = 1."""

    left = tuple(left_prefixes)
    right = tuple(right_prefixes)
    if not left or not right:
        raise ValueError("prefix domains must be non-empty")
    levels = len(left[0])
    if levels == 0:
        raise ValueError("at least one prefix level is required")
    if any(len(item) != levels for item in left + right):
        raise ValueError("all endpoints must expose the same number of levels")
    total = len(left) * len(right)
    rates = [1.0]
    for level in range(1, levels + 1):
        left_counts = Counter(item[:level] for item in left)
        right_counts = Counter(item[:level] for item in right)
        matches = sum(count * right_counts.get(prefix, 0) for prefix, count in left_counts.items())
        rates.append(matches / total)
    return tuple(rates)


def prefix_rms_cost(stage_costs: tuple[float, ...], survival_rates: tuple[float, ...]) -> float:
    """Compute sqrt(sum s_{l-1}(C_l^2-C_{l-1}^2))."""

    if not stage_costs or any(cost <= 0.0 for cost in stage_costs):
        raise ValueError("stage costs must be strictly positive")
    if len(survival_rates) != len(stage_costs) + 1:
        raise ValueError("survival_rates must contain s_0 through s_L")
    if abs(survival_rates[0] - 1.0) > 1e-12:
        raise ValueError("s_0 must equal one")
    if any(not 0.0 <= value <= 1.0 for value in survival_rates):
        raise ValueError("survival rates must lie in [0,1]")
    if any(a < b for a, b in zip(survival_rates, survival_rates[1:], strict=False)):
        raise ValueError("survival rates must be non-increasing")

    cumulative = np.cumsum(np.asarray(stage_costs, dtype=float))
    previous_square = 0.0
    second_moment = 0.0
    for level, cumulative_cost in enumerate(cumulative, start=1):
        square = float(cumulative_cost * cumulative_cost)
        second_moment += survival_rates[level - 1] * (square - previous_square)
        previous_square = square
    return sqrt(second_moment)


def weighted_qcollide_cost(
    n: int,
    matching_size: int,
    rms_endpoint_cost: float,
) -> float:
    if rms_endpoint_cost <= 0.0:
        raise ValueError("rms_endpoint_cost must be positive")
    return rms_endpoint_cost * product_johnson_cost(n, matching_size).total_cost


def forbidden_claw_cost_for_pair_oracle(*, endpoint_local: bool) -> float:
    """A hard guard preventing a claw claim on an arbitrary pair oracle."""

    if not endpoint_local:
        return inf
    return 0.0
