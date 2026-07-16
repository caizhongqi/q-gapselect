"""Variable-time charged activity-history alignment audit.

The charged finite-phase prototype proves that activity predicates need not be
supplied as rows.  This module moves that evidence back onto the main
unknown-boundary line: it charges a finite-QPE cost ``c_r`` for every precision
level and asks whether the desired active-history quadrature still separates
from rebuilt-history baselines under the same charged costs.

The formulas are audit proxies, not theorems.  A record whose gate remains open
is a proof target for P0-U08; a record whose gate fails is a rejected family.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .unknown_boundary_history import (
    UnknownBoundaryHistoryRecord,
    unknown_boundary_history_record,
)

CLAIM_STATUS = "variable_time_charged_history_alignment_no_theorem"
NOVELTY_OPEN = "open_charged_variable_time_gap_requires_upper_and_lower_bound"
NOVELTY_FAILED = "failed_charged_baseline_match"


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


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class VariableTimeChargedLevel:
    """One charged precision epoch in the mainline alignment audit."""

    level: int
    epsilon: float
    precision_bits: int
    qpe_query_units: int
    active_count: int
    output_births: int
    inactive_count: int
    active_history_weight: float
    direct_extraction_weight: float
    rebuild_rms_weight: float


@dataclass(frozen=True, slots=True)
class VariableTimeChargedHistoryRecord:
    """One charged variable-time alignment point."""

    m: int
    n: int
    k: int
    level_count: int
    gamma: float
    precision_multiplier: float
    baseline_match_tolerance: float
    layers: tuple[VariableTimeChargedLevel, ...]
    boundary_localization_proxy: float
    charged_active_history_rms_proxy: float
    charged_direct_extraction_proxy: float
    charged_candidate_total_proxy: float
    serial_rebuild_scan_proxy: float
    variable_time_rebuild_rms_proxy: float
    grover_activity_layered_proxy: float
    independent_all_arm_proxy: float
    coarse_partition_bai_proxy: float
    adversary_lower_target_proxy: float
    min_valid_baseline_proxy: float
    min_valid_baseline_name: str
    min_valid_baseline_over_candidate: float
    lower_target_over_candidate: float
    charged_serial_over_candidate: float
    encoded_baseline_match_found: bool
    novelty_gate: str
    alignment_status: str
    claim_status: str = CLAIM_STATUS


def precision_bits_for_epsilon(
    epsilon: float,
    *,
    precision_multiplier: float = 2.0,
    min_precision_bits: int = 1,
    max_precision_bits: int | None = None,
) -> int:
    """Return bits giving QPE grid step at most ``epsilon / multiplier``."""

    epsilon = _positive(epsilon, "epsilon")
    multiplier = _positive(precision_multiplier, "precision_multiplier")
    min_bits = _integer(min_precision_bits, "min_precision_bits")
    if min_bits < 1:
        raise ValueError("min_precision_bits must be at least one")
    if max_precision_bits is not None:
        max_bits = _integer(max_precision_bits, "max_precision_bits")
        if max_bits < min_bits:
            raise ValueError("max_precision_bits must be at least min_precision_bits")
    else:
        max_bits = None
    bits = max(min_bits, int(math.ceil(math.log2(multiplier / epsilon))))
    if max_bits is not None:
        bits = min(bits, max_bits)
    return bits


def variable_time_charged_history_record(
    history: UnknownBoundaryHistoryRecord | int,
    *,
    precision_multiplier: float = 2.0,
    min_precision_bits: int = 1,
    max_precision_bits: int | None = None,
    baseline_match_tolerance: float = 1.2,
    **history_kwargs: object,
) -> VariableTimeChargedHistoryRecord:
    """Charge the unknown-boundary history target with finite-QPE level costs."""

    if isinstance(history, UnknownBoundaryHistoryRecord):
        if history_kwargs:
            raise ValueError("history_kwargs are only allowed when history is an m value")
        base = history
    else:
        base = unknown_boundary_history_record(_integer(history, "history"), **history_kwargs)
    tolerance = _positive(baseline_match_tolerance, "baseline_match_tolerance")
    if tolerance < 1.0:
        raise ValueError("baseline_match_tolerance must be at least one")

    layers: list[VariableTimeChargedLevel] = []
    active_history_sum = 0.0
    direct_extraction_sum = 0.0
    rebuild_rms_sum = 0.0
    serial_rebuild = 0.0
    grover_activity = 0.0
    max_cost = 0.0
    for layer in base.layers:
        bits = precision_bits_for_epsilon(
            layer.epsilon,
            precision_multiplier=precision_multiplier,
            min_precision_bits=min_precision_bits,
            max_precision_bits=max_precision_bits,
        )
        qpe_cost = float((1 << bits) - 1)
        active_weight = layer.active_count * (qpe_cost**2)
        extraction_weight = layer.active_count * (layer.output_births + 1) * (
            qpe_cost**2
        )
        rebuild_weight = base.n * (qpe_cost**2)
        active_history_sum += active_weight
        direct_extraction_sum += extraction_weight
        rebuild_rms_sum += rebuild_weight
        serial_rebuild += base.n * qpe_cost
        grover_activity += math.sqrt(base.n * layer.active_count) * qpe_cost
        max_cost = max(max_cost, qpe_cost)
        layers.append(
            VariableTimeChargedLevel(
                level=layer.level,
                epsilon=layer.epsilon,
                precision_bits=bits,
                qpe_query_units=int(qpe_cost),
                active_count=layer.active_count,
                output_births=layer.output_births,
                inactive_count=layer.inactive_count,
                active_history_weight=active_weight,
                direct_extraction_weight=extraction_weight,
                rebuild_rms_weight=rebuild_weight,
            )
        )

    active_history = math.sqrt(active_history_sum)
    direct_extraction = math.sqrt(direct_extraction_sum)
    boundary = base.boundary_localization_proxy
    candidate = boundary + active_history + direct_extraction
    variable_time_rebuild = math.sqrt(rebuild_rms_sum)
    independent = base.n * max_cost
    coarse_bai = math.sqrt(base.n * base.k) * max_cost
    lower_target = max(boundary, active_history, direct_extraction)
    valid_baselines = {
        "serial_rebuild_scan": serial_rebuild,
        "variable_time_rebuild_rms": variable_time_rebuild,
        "grover_activity_layered": grover_activity,
        "independent_all_arm_scan": independent,
        "coarse_partition_plus_bai": coarse_bai,
    }
    min_name, min_value = min(valid_baselines.items(), key=lambda item: item[1])
    ratio = min_value / candidate
    match = ratio <= tolerance
    return VariableTimeChargedHistoryRecord(
        m=base.m,
        n=base.n,
        k=base.k,
        level_count=base.level_count,
        gamma=base.gamma,
        precision_multiplier=precision_multiplier,
        baseline_match_tolerance=tolerance,
        layers=tuple(layers),
        boundary_localization_proxy=boundary,
        charged_active_history_rms_proxy=active_history,
        charged_direct_extraction_proxy=direct_extraction,
        charged_candidate_total_proxy=candidate,
        serial_rebuild_scan_proxy=serial_rebuild,
        variable_time_rebuild_rms_proxy=variable_time_rebuild,
        grover_activity_layered_proxy=grover_activity,
        independent_all_arm_proxy=independent,
        coarse_partition_bai_proxy=coarse_bai,
        adversary_lower_target_proxy=lower_target,
        min_valid_baseline_proxy=min_value,
        min_valid_baseline_name=min_name,
        min_valid_baseline_over_candidate=ratio,
        lower_target_over_candidate=lower_target / candidate,
        charged_serial_over_candidate=serial_rebuild / candidate,
        encoded_baseline_match_found=match,
        novelty_gate=NOVELTY_FAILED if match else NOVELTY_OPEN,
        alignment_status=(
            "charged_costs_still_separate_from_encoded_baselines"
            if not match
            else "charged_costs_matched_by_encoded_baseline"
        ),
    )


def variable_time_charged_history_sweep(
    m_values: Sequence[int],
    *,
    precision_multiplier: float = 2.0,
    min_precision_bits: int = 1,
    max_precision_bits: int | None = None,
    baseline_match_tolerance: float = 1.2,
    **history_kwargs: object,
) -> tuple[VariableTimeChargedHistoryRecord, ...]:
    """Evaluate a strictly increasing charged mainline sweep."""

    try:
        values = tuple(_integer(value, "m") for value in m_values)
    except TypeError as error:
        raise TypeError("m_values must be a sequence of integers") from error
    if not values:
        raise ValueError("m_values cannot be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError("m_values must be strictly increasing and unique")
    return tuple(
        variable_time_charged_history_record(
            value,
            precision_multiplier=precision_multiplier,
            min_precision_bits=min_precision_bits,
            max_precision_bits=max_precision_bits,
            baseline_match_tolerance=baseline_match_tolerance,
            **history_kwargs,
        )
        for value in values
    )


def variable_time_charged_history_loglog_slope(
    records: Sequence[VariableTimeChargedHistoryRecord], field: str
) -> float:
    """Return a finite-family descriptive slope."""

    rows = tuple(records)
    allowed = {
        "boundary_localization_proxy",
        "charged_active_history_rms_proxy",
        "charged_direct_extraction_proxy",
        "charged_candidate_total_proxy",
        "serial_rebuild_scan_proxy",
        "variable_time_rebuild_rms_proxy",
        "grover_activity_layered_proxy",
        "independent_all_arm_proxy",
        "coarse_partition_bai_proxy",
        "adversary_lower_target_proxy",
        "min_valid_baseline_proxy",
    }
    if field not in allowed:
        raise ValueError(f"field must be one of {sorted(allowed)}")
    if len(rows) < 2:
        raise ValueError("at least two records are required")
    if any(not isinstance(row, VariableTimeChargedHistoryRecord) for row in rows):
        raise TypeError("records must contain VariableTimeChargedHistoryRecord values")
    x = np.log(np.asarray([row.m for row in rows], dtype=np.float64))
    y = np.log(np.asarray([getattr(row, field) for row in rows], dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])
