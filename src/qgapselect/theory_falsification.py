"""Analytic falsification families for the conjectural orientation functional.

Nothing in this module is an executed quantum algorithm or a proved query
bound.  The records compare a declared candidate arithmetic functional with
simple same-gap proxies so that the proposed separation can be tested before
proof work.  The subsequent same-interface composition audit invalidated this
explicit family as a strict novelty witness; this module is retained to make
that negative result reproducible.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .complexity import candidate_layer_profile

CLAIM_STATUS = "analytic_candidate_separation_no_algorithm_or_theorem"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class OrientationSeparationRecord:
    """One point in the explicit ``n=3m, k=m`` falsification family."""

    m: int
    n: int
    k: int
    gamma: float
    beta: float
    far_offset: float
    selected_candidate_proxy: float
    rejected_candidate_proxy: float
    orientation_candidate_proxy: float
    chosen_orientation: str
    independent_all_arm_proxy: float
    worst_gap_marked_extraction_proxy: float
    independent_over_candidate: float
    marked_extraction_over_candidate: float
    generic_composition_audit_status: str = "failed_explicit_family"
    claim_status: str = CLAIM_STATUS


def orientation_separation_record(
    m: int,
    *,
    gamma_exponent: float = 2.0,
    beta: float = math.pi / 4.0,
    far_offset: float = math.pi / 8.0,
) -> OrientationSeparationRecord:
    """Evaluate the candidate and simple proxies on one separation instance.

    There are ``m`` selected angles at ``beta + gamma``, one rejected angle at
    ``beta - gamma``, and ``2m-1`` rejected angles at
    ``beta - far_offset``.  Arm identities are irrelevant to these symmetric
    arithmetic functionals; an eventual query lower bound must randomize them.
    """

    m = _integer(m, "m")
    exponent = _finite(gamma_exponent, "gamma_exponent")
    beta = _finite(beta, "beta")
    far_offset = _finite(far_offset, "far_offset")
    if m < 2:
        raise ValueError("m must be at least two")
    if exponent <= 0.0:
        raise ValueError("gamma_exponent must be positive")
    if not 0.0 < beta < math.pi / 2.0:
        raise ValueError("beta must lie in (0, pi/2)")
    if not 0.0 < far_offset < beta:
        raise ValueError("far_offset must lie in (0, beta)")
    gamma = m ** (-exponent)
    if beta + gamma >= math.pi / 2.0 or beta - gamma <= beta - far_offset:
        raise ValueError("m and angular parameters do not define the promised order")

    n = 3 * m
    angles = (
        (beta + gamma,) * m
        + (beta - gamma,)
        + (beta - far_offset,) * (2 * m - 1)
    )
    means = tuple(math.sin(angle) ** 2 for angle in angles)
    layer = candidate_layer_profile(means, m)
    representations = {
        item.representation: item.value for item in layer.representations
    }
    independent = n / gamma
    marked = math.sqrt(n * m) / gamma
    return OrientationSeparationRecord(
        m=m,
        n=n,
        k=m,
        gamma=gamma,
        beta=beta,
        far_offset=far_offset,
        selected_candidate_proxy=representations["selected"],
        rejected_candidate_proxy=representations["rejected_complement"],
        orientation_candidate_proxy=layer.value,
        chosen_orientation=layer.representation,
        independent_all_arm_proxy=independent,
        worst_gap_marked_extraction_proxy=marked,
        independent_over_candidate=independent / layer.value,
        marked_extraction_over_candidate=marked / layer.value,
    )


def orientation_separation_sweep(
    m_values: Sequence[int],
    *,
    gamma_exponent: float = 2.0,
    beta: float = math.pi / 4.0,
    far_offset: float = math.pi / 8.0,
) -> tuple[OrientationSeparationRecord, ...]:
    """Evaluate a strictly increasing sequence of separation sizes."""

    try:
        values = tuple(_integer(value, "m") for value in m_values)
    except TypeError as error:
        raise TypeError("m_values must be a sequence of integers") from error
    if not values:
        raise ValueError("m_values cannot be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError("m_values must be strictly increasing and unique")
    return tuple(
        orientation_separation_record(
            m,
            gamma_exponent=gamma_exponent,
            beta=beta,
            far_offset=far_offset,
        )
        for m in values
    )


def descriptive_loglog_slope(
    records: Sequence[OrientationSeparationRecord], field: str
) -> float:
    """Return a descriptive slope; this is not an asymptotic proof."""

    rows = tuple(records)
    allowed = {
        "orientation_candidate_proxy",
        "selected_candidate_proxy",
        "rejected_candidate_proxy",
        "independent_all_arm_proxy",
        "worst_gap_marked_extraction_proxy",
    }
    if field not in allowed:
        raise ValueError(f"field must be one of {sorted(allowed)}")
    if len(rows) < 2:
        raise ValueError("at least two records are required")
    if any(not isinstance(row, OrientationSeparationRecord) for row in rows):
        raise TypeError("records must contain OrientationSeparationRecord values")
    x = np.log(np.asarray([row.m for row in rows], dtype=np.float64))
    y = np.log(np.asarray([getattr(row, field) for row in rows], dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])
