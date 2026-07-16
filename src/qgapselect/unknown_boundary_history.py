"""Unknown-boundary coherent activity-history candidate core.

This module starts the next quantum-algorithm candidate after the orientation
proxy was falsified by same-interface composition baselines.  It is purposely
written as an audit harness, not as a theorem implementation.

The candidate problem is stricter than the earlier known-boundary layer
functional:

* the Top-k boundary is not supplied as a classical threshold;
* active sets at each precision level are not supplied as QRAM lists;
* a valid algorithm must generate activity predicates coherently from the
  charged oracle interface; and
* direct multi-output extraction is compared against rebuilt-history,
  variable-time, Grover-activity, independent-scan, and coarse+BAI proxies.

Every number below is a constant/log-factor-free query proxy.  A record whose
``novelty_gate`` remains open still needs a reversible transducer construction
and a matching adversary lower bound before it can become a paper theorem.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

CLAIM_STATUS = "open_unknown_boundary_history_candidate_no_theorem"
NOVELTY_OPEN = "open_no_encoded_baseline_match_requires_unitary_and_lower_bound"
NOVELTY_FAILED = "failed_encoded_baseline_match"


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


def _positive_exponent(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _power_count(m: int, exponent: float, name: str) -> int:
    value = int(math.ceil(float(m) ** exponent))
    if value <= 0:
        raise ValueError(f"{name} produced a non-positive count")
    return value


@dataclass(frozen=True, slots=True)
class HistoryScale:
    """One precision/activity level in the unknown-boundary family."""

    level: int
    epsilon: float
    active_count: int
    output_births: int
    inactive_count: int
    activity_predicate_cost: float
    extraction_weight: float


@dataclass(frozen=True, slots=True)
class UnknownBoundaryHistoryRecord:
    """One auditable point for the new no-free-QRAM candidate core."""

    m: int
    n: int
    k: int
    level_count: int
    gamma: float
    active_base_count: int
    output_births_per_level: int
    total_outputs: int
    epsilon_growth_exponent: float
    activity_decay_exponent: float
    baseline_match_tolerance: float
    layers: tuple[HistoryScale, ...]
    boundary_localization_proxy: float
    coherent_activity_history_proxy: float
    direct_multi_output_quadrature_proxy: float
    candidate_total_proxy: float
    known_boundary_free_history_layered_proxy: float
    rebuild_history_scan_layered_proxy: float
    variable_time_rebuild_rms_proxy: float
    grover_activity_layered_proxy: float
    independent_all_arm_proxy: float
    coarse_partition_bai_proxy: float
    adversary_lower_target_proxy: float
    min_encoded_valid_baseline_proxy: float
    min_encoded_valid_baseline_name: str
    min_valid_baseline_over_candidate: float
    free_history_over_candidate: float
    lower_target_over_candidate: float
    encoded_baseline_match_found: bool
    novelty_gate: str
    no_free_qram_assumption: str
    matching_upper_bound_status: str
    matching_lower_bound_status: str
    claim_status: str = CLAIM_STATUS


def unknown_boundary_history_record(
    m: int,
    *,
    n_exponent: float = 3.0,
    level_exponent: float = 1.0,
    active_exponent: float = 1.0,
    gamma_exponent: float = 2.0,
    output_births_per_level: int = 1,
    epsilon_growth_exponent: float = 0.25,
    activity_decay_exponent: float = 0.0,
    predicate_cost_exponent: float = 0.0,
    baseline_match_tolerance: float = 1.2,
) -> UnknownBoundaryHistoryRecord:
    """Return one no-free-QRAM unknown-boundary history audit point.

    The default family has ``n=m^3`` arms, ``m`` precision epochs, ``m`` hidden
    active arms per epoch, one newly required output per epoch, and minimum
    angular precision ``gamma=m^-2``.  The inactive sea is intentionally large:
    it makes rebuilt activity histories expensive, while the candidate is only
    allowed to ignore the inactive sea if a coherent activity-history
    transducer is later constructed and proved.
    """

    m = _integer(m, "m")
    if m < 2:
        raise ValueError("m must be at least two")
    n_exponent = _positive_exponent(n_exponent, "n_exponent")
    level_exponent = _positive_exponent(level_exponent, "level_exponent")
    active_exponent = _positive_exponent(active_exponent, "active_exponent")
    gamma_exponent = _positive_exponent(gamma_exponent, "gamma_exponent")
    epsilon_growth_exponent = _nonnegative(
        epsilon_growth_exponent, "epsilon_growth_exponent"
    )
    activity_decay_exponent = _nonnegative(
        activity_decay_exponent, "activity_decay_exponent"
    )
    predicate_cost_exponent = _nonnegative(
        predicate_cost_exponent, "predicate_cost_exponent"
    )
    output_births = _integer(output_births_per_level, "output_births_per_level")
    if output_births <= 0:
        raise ValueError("output_births_per_level must be positive")
    tolerance = _finite(baseline_match_tolerance, "baseline_match_tolerance")
    if tolerance < 1.0:
        raise ValueError("baseline_match_tolerance must be at least one")

    n = _power_count(m, n_exponent, "n_exponent")
    level_count = _power_count(m, level_exponent, "level_exponent")
    active_base = _power_count(m, active_exponent, "active_exponent")
    if active_base >= n:
        raise ValueError("active_base_count must be strictly below n")
    total_outputs = level_count * output_births
    if total_outputs >= n:
        raise ValueError("total output count must be strictly below n")
    if output_births > active_base:
        raise ValueError("output_births_per_level cannot exceed active_base_count")

    gamma = float(m) ** (-gamma_exponent)
    layers: list[HistoryScale] = []
    extraction_sum = 0.0
    free_layered = 0.0
    rebuild_layered = 0.0
    rebuild_rms_sum = 0.0
    grover_layered = 0.0
    history_sum = 0.0
    for level in range(level_count):
        scale = float(level + 1)
        epsilon = gamma * (scale**epsilon_growth_exponent)
        active_count = max(
            output_births,
            int(math.ceil(active_base / (scale**activity_decay_exponent))),
        )
        if active_count >= n:
            raise ValueError("activity parameters produced active_count >= n")
        inactive_count = n - active_count
        predicate_cost = scale**predicate_cost_exponent
        extraction_weight = active_count * (output_births + 1) / (epsilon**2)
        extraction_sum += extraction_weight
        free_layered += math.sqrt(active_count * (output_births + 1)) / epsilon
        rebuild_layered += n / epsilon
        rebuild_rms_sum += n / (epsilon**2)
        grover_layered += math.sqrt(n * active_count) / epsilon
        history_sum += active_count * (predicate_cost**2)
        layers.append(
            HistoryScale(
                level=level,
                epsilon=epsilon,
                active_count=active_count,
                output_births=output_births,
                inactive_count=inactive_count,
                activity_predicate_cost=predicate_cost,
                extraction_weight=extraction_weight,
            )
        )

    min_epsilon = min(layer.epsilon for layer in layers)
    boundary = math.sqrt(n) / min_epsilon
    coherent_history = math.sqrt(history_sum)
    direct_quadrature = math.sqrt(extraction_sum)
    candidate = boundary + coherent_history + direct_quadrature
    variable_time_rebuild = math.sqrt(rebuild_rms_sum)
    independent = n / min_epsilon
    coarse_bai = math.sqrt(n * total_outputs) / min_epsilon
    lower_target = max(boundary, direct_quadrature)
    valid_baselines = {
        "variable_time_rebuild_rms": variable_time_rebuild,
        "grover_activity_layered": grover_layered,
        "independent_all_arm_scan": independent,
        "coarse_partition_plus_bai": coarse_bai,
    }
    min_name, min_value = min(valid_baselines.items(), key=lambda item: item[1])
    baseline_ratio = min_value / candidate
    match = baseline_ratio <= tolerance
    return UnknownBoundaryHistoryRecord(
        m=m,
        n=n,
        k=total_outputs,
        level_count=level_count,
        gamma=gamma,
        active_base_count=active_base,
        output_births_per_level=output_births,
        total_outputs=total_outputs,
        epsilon_growth_exponent=epsilon_growth_exponent,
        activity_decay_exponent=activity_decay_exponent,
        baseline_match_tolerance=tolerance,
        layers=tuple(layers),
        boundary_localization_proxy=boundary,
        coherent_activity_history_proxy=coherent_history,
        direct_multi_output_quadrature_proxy=direct_quadrature,
        candidate_total_proxy=candidate,
        known_boundary_free_history_layered_proxy=free_layered,
        rebuild_history_scan_layered_proxy=rebuild_layered,
        variable_time_rebuild_rms_proxy=variable_time_rebuild,
        grover_activity_layered_proxy=grover_layered,
        independent_all_arm_proxy=independent,
        coarse_partition_bai_proxy=coarse_bai,
        adversary_lower_target_proxy=lower_target,
        min_encoded_valid_baseline_proxy=min_value,
        min_encoded_valid_baseline_name=min_name,
        min_valid_baseline_over_candidate=baseline_ratio,
        free_history_over_candidate=free_layered / candidate,
        lower_target_over_candidate=lower_target / candidate,
        encoded_baseline_match_found=match,
        novelty_gate=NOVELTY_FAILED if match else NOVELTY_OPEN,
        no_free_qram_assumption=(
            "active precision histories are not supplied as QRAM lists; every "
            "activity predicate must be generated coherently from charged "
            "oracle access"
        ),
        matching_upper_bound_status=(
            "candidate_transducer_cost_only_reversible_construction_open"
        ),
        matching_lower_bound_status=(
            "adversary_or_polynomial_method_lower_bound_open"
        ),
    )


def unknown_boundary_history_sweep(
    m_values: Sequence[int],
    *,
    n_exponent: float = 3.0,
    level_exponent: float = 1.0,
    active_exponent: float = 1.0,
    gamma_exponent: float = 2.0,
    output_births_per_level: int = 1,
    epsilon_growth_exponent: float = 0.25,
    activity_decay_exponent: float = 0.0,
    predicate_cost_exponent: float = 0.0,
    baseline_match_tolerance: float = 1.2,
) -> tuple[UnknownBoundaryHistoryRecord, ...]:
    """Evaluate a strictly increasing unknown-boundary history family."""

    try:
        values = tuple(_integer(value, "m") for value in m_values)
    except TypeError as error:
        raise TypeError("m_values must be a sequence of integers") from error
    if not values:
        raise ValueError("m_values cannot be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError("m_values must be strictly increasing and unique")
    return tuple(
        unknown_boundary_history_record(
            value,
            n_exponent=n_exponent,
            level_exponent=level_exponent,
            active_exponent=active_exponent,
            gamma_exponent=gamma_exponent,
            output_births_per_level=output_births_per_level,
            epsilon_growth_exponent=epsilon_growth_exponent,
            activity_decay_exponent=activity_decay_exponent,
            predicate_cost_exponent=predicate_cost_exponent,
            baseline_match_tolerance=baseline_match_tolerance,
        )
        for value in values
    )


def unknown_boundary_history_loglog_slope(
    records: Sequence[UnknownBoundaryHistoryRecord], field: str
) -> float:
    """Return a descriptive finite-family slope, never a theorem."""

    rows = tuple(records)
    allowed = {
        "boundary_localization_proxy",
        "coherent_activity_history_proxy",
        "direct_multi_output_quadrature_proxy",
        "candidate_total_proxy",
        "known_boundary_free_history_layered_proxy",
        "rebuild_history_scan_layered_proxy",
        "variable_time_rebuild_rms_proxy",
        "grover_activity_layered_proxy",
        "independent_all_arm_proxy",
        "coarse_partition_bai_proxy",
        "adversary_lower_target_proxy",
        "min_encoded_valid_baseline_proxy",
    }
    if field not in allowed:
        raise ValueError(f"field must be one of {sorted(allowed)}")
    if len(rows) < 2:
        raise ValueError("at least two records are required")
    if any(not isinstance(row, UnknownBoundaryHistoryRecord) for row in rows):
        raise TypeError("records must contain UnknownBoundaryHistoryRecord values")
    x = np.log(np.asarray([row.m for row in rows], dtype=np.float64))
    y = np.log(np.asarray([getattr(row, field) for row in rows], dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])
