"""Composition-frontier audit for the unknown-boundary charged candidate.

This module is the P1-COMP counterpart of the stopping-unitary theorem
scaffold.  It instantiates known-composition-style baselines under the same
charged finite-QPE proxy interface and records whether any encoded baseline
matches the candidate within a declared tolerance.

The result is still an audit, not a novelty proof.  Passing the frontier only
means the currently encoded baselines did not kill the candidate family.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .variable_time_charged_history import (
    VariableTimeChargedHistoryRecord,
    variable_time_charged_history_record,
)

CLAIM_STATUS = "composition_frontier_audit_no_novelty_theorem"
NOVELTY_OPEN = "open_no_encoded_composition_match"
NOVELTY_FAILED = "failed_encoded_composition_match"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True, slots=True)
class CompositionBaseline:
    """One encoded composition baseline."""

    name: str
    proxy: float
    valid_same_interface: bool
    assumption: str
    limitation: str
    over_candidate: float
    matches_candidate: bool


@dataclass(frozen=True, slots=True)
class CompositionFrontierRecord:
    """One charged candidate composition-frontier audit point."""

    m: int
    n: int
    k: int
    level_count: int
    gamma: float
    candidate_proxy: float
    baseline_match_tolerance: float
    baselines: tuple[CompositionBaseline, ...]
    strongest_valid_baseline_name: str
    strongest_valid_baseline_proxy: float
    strongest_valid_baseline_over_candidate: float
    encoded_match_found: bool
    novelty_gate: str
    proof_obligation: str
    claim_status: str = CLAIM_STATUS


def composition_frontier_record(
    candidate: VariableTimeChargedHistoryRecord | int,
    *,
    baseline_match_tolerance: float = 1.2,
    **candidate_kwargs: object,
) -> CompositionFrontierRecord:
    """Instantiate the composition frontier around one charged candidate."""

    if isinstance(candidate, VariableTimeChargedHistoryRecord):
        if candidate_kwargs:
            raise ValueError("candidate_kwargs are only allowed when candidate is m")
        record = candidate
    else:
        record = variable_time_charged_history_record(
            _integer(candidate, "candidate"),
            baseline_match_tolerance=baseline_match_tolerance,
            **candidate_kwargs,
        )
    tolerance = _positive(baseline_match_tolerance, "baseline_match_tolerance")
    if tolerance < 1.0:
        raise ValueError("baseline_match_tolerance must be at least one")
    candidate_proxy = record.charged_candidate_total_proxy

    raw_baselines = (
        (
            "loop_variable_time_rebuild",
            record.variable_time_rebuild_rms_proxy,
            True,
            "Rebuilds charged history through variable-time RMS composition.",
            "Valid same-interface comparator; it pays inactive-sea rebuild cost.",
        ),
        (
            "all_marked_extraction_generated_predicate",
            record.grover_activity_layered_proxy,
            True,
            "Uses generated activity predicates and all-marked extraction per level.",
            "Does not get free active lists; must charge predicate generation.",
        ),
        (
            "coarse_partition_plus_qbai",
            record.coarse_partition_bai_proxy,
            True,
            "Coarse partition followed by quantum best-arm identification proxy.",
            "Only valid when the coarse partition exposes a same-output relation.",
        ),
        (
            "independent_all_arm_qpe",
            record.independent_all_arm_proxy,
            True,
            "Runs charged finite-QPE precision on every arm independently.",
            "Baseline is intentionally strong but loses active-history adaptivity.",
        ),
        (
            "serial_rebuild_scan",
            record.serial_rebuild_scan_proxy,
            True,
            "Scans every arm at every charged level.",
            "Valid but usually much larger than variable-time rebuild RMS.",
        ),
        (
            "free_history_qram_layered",
            record.charged_direct_extraction_proxy,
            False,
            "Assumes active histories are already materialized.",
            "Invalid under no-free-QRAM interface; tracked as a forbidden collapse.",
        ),
    )
    baselines = tuple(
        CompositionBaseline(
            name=name,
            proxy=proxy,
            valid_same_interface=valid,
            assumption=assumption,
            limitation=limitation,
            over_candidate=proxy / candidate_proxy,
            matches_candidate=valid and proxy / candidate_proxy <= tolerance,
        )
        for name, proxy, valid, assumption, limitation in raw_baselines
    )
    valid_baselines = tuple(item for item in baselines if item.valid_same_interface)
    strongest = min(valid_baselines, key=lambda item: item.proxy)
    match = strongest.over_candidate <= tolerance
    return CompositionFrontierRecord(
        m=record.m,
        n=record.n,
        k=record.k,
        level_count=record.level_count,
        gamma=record.gamma,
        candidate_proxy=candidate_proxy,
        baseline_match_tolerance=tolerance,
        baselines=baselines,
        strongest_valid_baseline_name=strongest.name,
        strongest_valid_baseline_proxy=strongest.proxy,
        strongest_valid_baseline_over_candidate=strongest.over_candidate,
        encoded_match_found=match,
        novelty_gate=NOVELTY_FAILED if match else NOVELTY_OPEN,
        proof_obligation=(
            "Instantiate published loop composition, exact/approximate k-minimum, "
            "QBAI, and variable-time search theorems with formal assumptions."
        ),
    )


def composition_frontier_sweep(
    m_values: Sequence[int],
    *,
    baseline_match_tolerance: float = 1.2,
    **candidate_kwargs: object,
) -> tuple[CompositionFrontierRecord, ...]:
    """Evaluate a strictly increasing composition-frontier sweep."""

    try:
        values = tuple(_integer(value, "m") for value in m_values)
    except TypeError as error:
        raise TypeError("m_values must be a sequence of integers") from error
    if not values:
        raise ValueError("m_values cannot be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError("m_values must be strictly increasing and unique")
    return tuple(
        composition_frontier_record(
            value,
            baseline_match_tolerance=baseline_match_tolerance,
            **candidate_kwargs,
        )
        for value in values
    )


def composition_frontier_loglog_slope(
    records: Sequence[CompositionFrontierRecord], field: str
) -> float:
    """Return a descriptive finite-family slope."""

    rows = tuple(records)
    allowed = {
        "candidate_proxy",
        "strongest_valid_baseline_proxy",
    }
    if field not in allowed:
        raise ValueError(f"field must be one of {sorted(allowed)}")
    if len(rows) < 2:
        raise ValueError("at least two records are required")
    if any(not isinstance(row, CompositionFrontierRecord) for row in rows):
        raise TypeError("records must contain CompositionFrontierRecord values")
    x = np.log(np.asarray([row.m for row in rows], dtype=np.float64))
    y = np.log(np.asarray([getattr(row, field) for row in rows], dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])
