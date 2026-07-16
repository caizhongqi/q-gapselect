"""Same-interface composition falsification for the orientation candidate.

This module evaluates arithmetic query *proxies*.  It neither executes a
quantum algorithm nor proves an upper or lower bound.  Its purpose is to stop
a novelty claim as soon as a known outer composition or an explicit
same-output-relation construction matches the proposed scaling.

The strongest currently encoded counterexample is the ``n=3m, k=m`` family
from :mod:`qgapselect.theory_falsification`.  A constant-precision coarse pass
separates the far rejected arms.  The remaining ``m+1`` arms contain exactly
one rejected arm, so strong-oracle best-arm identification applied after a
sign reversal recovers that exceptional arm.  The resulting proxy has the
same ``sqrt(m) / gamma`` leading scale as the orientation candidate.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .complexity import candidate_layer_profile

CLAIM_STATUS = "composition_falsification_proxies_no_algorithm_or_theorem"


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
class CompositionAuditRecord:
    """One auditable point on the explicit orientation-separation family."""

    m: int
    n: int
    k: int
    gamma: float
    near_arm_count: int
    far_arm_count: int
    output_count: int
    chosen_orientation: str
    orientation_candidate_proxy: float
    direct_layered_multi_output_proxy: float
    variable_time_rms_proxy: float
    coarse_partition_plus_bai_proxy: float
    independent_all_arm_proxy: float
    worst_gap_marked_extraction_proxy: float
    exceptional_search_lower_proxy: float
    candidate_over_direct_layered: float
    coarse_bai_over_candidate: float
    lower_over_candidate: float
    outer_composition_matches_candidate: bool
    explicit_family_separation_survives: bool
    matching_upper_bound_status: str
    matching_lower_bound_status: str
    novelty_gate: str
    claim_status: str = CLAIM_STATUS


def orientation_family_composition_record(
    m: int,
    *,
    gamma_exponent: float = 2.0,
    beta: float = math.pi / 4.0,
    far_offset: float = math.pi / 8.0,
) -> CompositionAuditRecord:
    """Audit generic compositions on the declared ``n=3m, k=m`` family.

    The quantities omit logarithmic and constant factors and therefore must
    only be compared at the scaling level.  ``direct_layered_multi_output``
    charges known all-marked extraction separately at every declared layer;
    it does not solve the still-open unknown-boundary transducer.  The coarse
    plus BAI proxy has the same final output relation on this explicit family
    and therefore invalidates that family as a strict-separation witness.
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
    chosen = layer.chosen

    direct_layered = sum(
        math.sqrt(term.active_count * max(term.newly_certified_outputs, 1))
        / term.epsilon
        for term in chosen.terms
    )
    angular_gaps = tuple(
        beta + gamma - angle if angle < beta else angle - (beta - gamma)
        for angle in angles
    )
    variable_time_rms = math.sqrt(sum(1.0 / gap**2 for gap in angular_gaps))
    near_arm_count = m + 1
    far_arm_count = 2 * m - 1
    # A constant-precision pass identifies the far arms.  Sign-reversed BAI
    # among the remaining m+1 arms finds the unique rejected arm.
    coarse_bai = n / far_offset + math.sqrt(near_arm_count) / gamma
    independent = n / gamma
    worst_gap_marked = math.sqrt(n * len(chosen.output_indices)) / gamma
    exceptional_lower = math.sqrt(near_arm_count) / gamma

    candidate = layer.value
    direct_ratio = candidate / direct_layered
    coarse_ratio = coarse_bai / candidate
    lower_ratio = exceptional_lower / candidate
    # The declared M+1 and max(M,1) layer charges differ by at most sqrt(2).
    outer_matches = direct_ratio <= math.sqrt(2.0) + 1e-12
    # A bounded same-relation ratio on the full configured sweep is checked by
    # the suite.  Pointwise, this field records that the proposed witness has
    # already been met by the explicit coarse+BAI construction.
    survives = not (outer_matches and coarse_ratio <= 4.0)
    return CompositionAuditRecord(
        m=m,
        n=n,
        k=m,
        gamma=gamma,
        near_arm_count=near_arm_count,
        far_arm_count=far_arm_count,
        output_count=len(chosen.output_indices),
        chosen_orientation=chosen.representation,
        orientation_candidate_proxy=candidate,
        direct_layered_multi_output_proxy=direct_layered,
        variable_time_rms_proxy=variable_time_rms,
        coarse_partition_plus_bai_proxy=coarse_bai,
        independent_all_arm_proxy=independent,
        worst_gap_marked_extraction_proxy=worst_gap_marked,
        exceptional_search_lower_proxy=exceptional_lower,
        candidate_over_direct_layered=direct_ratio,
        coarse_bai_over_candidate=coarse_ratio,
        lower_over_candidate=lower_ratio,
        outer_composition_matches_candidate=outer_matches,
        explicit_family_separation_survives=survives,
        matching_upper_bound_status=(
            "explicit_family_matched_by_coarse_partition_plus_strong_oracle_bai"
        ),
        matching_lower_bound_status=(
            "exceptional_search_scaling_proxy_only_all_algorithms_proof_open"
        ),
        novelty_gate=(
            "failed_explicit_family" if not survives else "requires_further_audit"
        ),
    )


def composition_audit_sweep(
    m_values: Sequence[int],
    *,
    gamma_exponent: float = 2.0,
    beta: float = math.pi / 4.0,
    far_offset: float = math.pi / 8.0,
) -> tuple[CompositionAuditRecord, ...]:
    """Evaluate a strictly increasing family for composition collapse."""

    try:
        values = tuple(_integer(value, "m") for value in m_values)
    except TypeError as error:
        raise TypeError("m_values must be a sequence of integers") from error
    if not values:
        raise ValueError("m_values cannot be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError("m_values must be strictly increasing and unique")
    return tuple(
        orientation_family_composition_record(
            m,
            gamma_exponent=gamma_exponent,
            beta=beta,
            far_offset=far_offset,
        )
        for m in values
    )


def composition_loglog_slope(
    records: Sequence[CompositionAuditRecord], field: str
) -> float:
    """Return a descriptive finite-family slope, never a theorem."""

    rows = tuple(records)
    allowed = {
        "orientation_candidate_proxy",
        "direct_layered_multi_output_proxy",
        "variable_time_rms_proxy",
        "coarse_partition_plus_bai_proxy",
        "independent_all_arm_proxy",
        "worst_gap_marked_extraction_proxy",
        "exceptional_search_lower_proxy",
    }
    if field not in allowed:
        raise ValueError(f"field must be one of {sorted(allowed)}")
    if len(rows) < 2:
        raise ValueError("at least two records are required")
    if any(not isinstance(row, CompositionAuditRecord) for row in rows):
        raise TypeError("records must contain CompositionAuditRecord values")
    x = np.log(np.asarray([row.m for row in rows], dtype=np.float64))
    y = np.log(np.asarray([getattr(row, field) for row in rows], dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])
