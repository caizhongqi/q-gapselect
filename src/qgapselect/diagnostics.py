"""Trusted construction diagnostics for the canonical Bernoulli oracle.

These helpers never accept an oracle object.  They require an explicit mean and
therefore cannot recover hidden instance data from the algorithm-facing oracle.
They are intended for unit tests and circuit specification checks, not selection.
"""

from __future__ import annotations

import math


def _validate_mean(mean: float) -> float:
    value = float(mean)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("a Bernoulli mean must be finite and in [0, 1]")
    return value


def reward_rotation_matrix(
    mean: float,
    *,
    inverse: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the fully specified ``Ry`` reward block for an explicit mean."""

    value = _validate_mean(mean)
    sine = math.sqrt(value)
    cosine = math.sqrt(1.0 - value)
    if inverse:
        return ((cosine, sine), (-sine, cosine))
    return ((cosine, -sine), (sine, cosine))


def controlled_reward_rotation_matrix(
    mean: float,
    *,
    inverse: bool = False,
) -> tuple[tuple[float, ...], ...]:
    r"""Return the control-on-one block diagonal ``I_2 \oplus Ry``."""

    rotation = reward_rotation_matrix(mean, inverse=inverse)
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, rotation[0][0], rotation[0][1]),
        (0.0, 0.0, rotation[1][0], rotation[1][1]),
    )


def amplified_success_probability(mean: float, grover_power: int = 0) -> float:
    """Return ``sin^2((2m+1) asin(sqrt(mean)))`` for explicit test data."""

    value = _validate_mean(mean)
    if grover_power < 0:
        raise ValueError("grover_power cannot be negative")
    theta = math.asin(math.sqrt(value))
    return math.sin((2 * grover_power + 1) * theta) ** 2
