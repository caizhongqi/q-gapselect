"""Geometric scale localization for the fixed-subset capacity estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fixed_subset_capacity import (
    choose_stable_subset_size,
    fixed_subset_survival_probability,
    simulate_fixed_subset_survival,
)


@dataclass(frozen=True)
class ScaleObservation:
    capacity_scale: int
    subset_size: int
    observed_survival: float


def geometric_capacity_scales(n: int) -> tuple[int, ...]:
    if n <= 0:
        raise ValueError("n must be positive")
    scales = [n]
    while scales[-1] > 1:
        next_scale = max(1, scales[-1] // 2)
        if next_scale == scales[-1]:
            break
        scales.append(next_scale)
    return tuple(scales)


def invert_survival_binary(
    n: int,
    subset_size: int,
    observed_survival: float,
    *,
    minimum_capacity: int = 0,
    maximum_capacity: int | None = None,
) -> dict[str, float | int]:
    """Invert the monotone exact survival curve with logarithmic evaluations."""

    if not 0.0 <= observed_survival <= 1.0:
        raise ValueError("observed_survival must lie in [0,1]")
    maximum = n if maximum_capacity is None else int(maximum_capacity)
    minimum = int(minimum_capacity)
    if not 0 <= minimum <= maximum <= n:
        raise ValueError("invalid capacity interval")
    lo = minimum
    hi = maximum
    while lo < hi:
        mid = (lo + hi) // 2
        probability = fixed_subset_survival_probability(n, mid, subset_size)
        if probability < observed_survival:
            lo = mid + 1
        else:
            hi = mid
    candidates = {lo}
    if lo > minimum:
        candidates.add(lo - 1)
    if lo < maximum:
        candidates.add(lo + 1)
    selected = min(
        candidates,
        key=lambda capacity: (
            abs(
                fixed_subset_survival_probability(n, capacity, subset_size)
                - observed_survival
            ),
            capacity,
        ),
    )
    probability = fixed_subset_survival_probability(n, selected, subset_size)
    return {
        "estimated_capacity": int(selected),
        "estimated_survival_probability": float(probability),
        "absolute_probability_residual": float(abs(probability - observed_survival)),
    }


def scan_unknown_capacity(
    n: int,
    matching_size: int,
    *,
    scan_trials: int,
    estimation_trials: int,
    seed: int,
    survival_low: float = 0.05,
    survival_high: float = 0.30,
) -> dict[str, object]:
    """Locate a stable scale with constant precision, then estimate K there."""

    if not 1 <= matching_size <= n:
        raise ValueError("matching_size must lie in [1,n]")
    if scan_trials <= 0 or estimation_trials <= 0:
        raise ValueError("trial counts must be positive")
    if not 0.0 < survival_low < survival_high < 1.0:
        raise ValueError("invalid survival window")
    rng = np.random.default_rng(seed)
    observations: list[ScaleObservation] = []
    selected: ScaleObservation | None = None
    target = 0.5 * (survival_low + survival_high)
    for scale in geometric_capacity_scales(n):
        try:
            subset = choose_stable_subset_size(n, scale)
        except RuntimeError:
            continue
        subset_size = int(subset["subset_size"])
        observed = simulate_fixed_subset_survival(
            n,
            matching_size,
            subset_size,
            trials=scan_trials,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        current = ScaleObservation(scale, subset_size, observed)
        observations.append(current)
        if observed >= survival_low:
            if observed <= survival_high or len(observations) == 1:
                selected = current
            else:
                previous = observations[-2]
                selected = min(
                    (previous, current),
                    key=lambda row: abs(row.observed_survival - target),
                )
            break
    if selected is None:
        selected = observations[-1]
    final_observed = simulate_fixed_subset_survival(
        n,
        matching_size,
        selected.subset_size,
        trials=estimation_trials,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    lower = max(1, selected.capacity_scale // 8)
    upper = min(n, max(lower, selected.capacity_scale * 8))
    inversion = invert_survival_binary(
        n,
        selected.subset_size,
        final_observed,
        minimum_capacity=lower,
        maximum_capacity=upper,
    )
    estimate = int(inversion["estimated_capacity"])
    return {
        "n": n,
        "true_capacity": matching_size,
        "scan_trials_per_scale": scan_trials,
        "estimation_trials": estimation_trials,
        "survival_window": [survival_low, survival_high],
        "observations": [
            {
                "capacity_scale": row.capacity_scale,
                "subset_size": row.subset_size,
                "observed_survival": row.observed_survival,
            }
            for row in observations
        ],
        "selected_scale": selected.capacity_scale,
        "selected_subset_size": selected.subset_size,
        "selected_scan_survival": selected.observed_survival,
        "final_observed_survival": final_observed,
        "estimated_capacity": estimate,
        "relative_capacity_error": abs(estimate - matching_size) / matching_size,
        "scale_to_capacity_ratio": selected.capacity_scale / matching_size,
        "inversion_interval": [lower, upper],
        "inversion": inversion,
    }


__all__ = [
    "geometric_capacity_scales",
    "invert_survival_binary",
    "scan_unknown_capacity",
]
