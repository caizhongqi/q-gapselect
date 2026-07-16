"""Lower-bound proof program for the unknown-boundary candidate.

The module records the L-07 proof target as executable symbolic scaffolding:
what hard-family blocks exist, what target proxy they should imply, and which
parts still require an adversary or polynomial-method proof.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .variable_time_charged_history import (
    VariableTimeChargedHistoryRecord,
    variable_time_charged_history_record,
)

CLAIM_STATUS = "lower_bound_program_scaffold_no_adversary_proof"
READINESS = "symbolic_program_started_proof_obligations_remain"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


@dataclass(frozen=True, slots=True)
class LowerBoundBlock:
    """One intended hard-family component."""

    block_id: str
    description: str
    proxy: float
    target_role: str
    proof_status: str
    missing_argument: str


@dataclass(frozen=True, slots=True)
class LowerBoundProgramRecord:
    """One symbolic lower-bound proof-program point."""

    m: int
    n: int
    k: int
    level_count: int
    gamma: float
    candidate_proxy: float
    adversary_target_proxy: float
    target_over_candidate: float
    blocks: tuple[LowerBoundBlock, ...]
    strongest_block_id: str
    strongest_block_proxy: float
    proof_obligation_count: int
    local_facts_count: int
    lower_bound_claim_status: str
    claim_status: str = CLAIM_STATUS


def lower_bound_program_record(
    candidate: VariableTimeChargedHistoryRecord | int,
    **candidate_kwargs: object,
) -> LowerBoundProgramRecord:
    """Create one lower-bound proof-program record."""

    if isinstance(candidate, VariableTimeChargedHistoryRecord):
        if candidate_kwargs:
            raise ValueError("candidate_kwargs are only allowed when candidate is m")
        record = candidate
    else:
        record = variable_time_charged_history_record(
            _integer(candidate, "candidate"),
            **candidate_kwargs,
        )
    blocks = (
        LowerBoundBlock(
            block_id="LB-B01",
            description="Boundary localization by angular discrimination near the Top-k cut.",
            proxy=record.boundary_localization_proxy,
            target_role="local angular boundary barrier",
            proof_status="local_fact_available",
            missing_argument=(
                "Lift local two-state discrimination to the unknown-boundary "
                "multi-output relation."
            ),
        ),
        LowerBoundBlock(
            block_id="LB-B02",
            description="Direct active-history recovery barrier over charged levels.",
            proxy=record.charged_active_history_rms_proxy,
            target_role="history direct-sum component",
            proof_status="proof_obligation",
            missing_argument=(
                "Construct adversary matrix or polynomial relation forcing "
                "coherent active-history discovery."
            ),
        ),
        LowerBoundBlock(
            block_id="LB-B03",
            description="Direct multi-output extraction barrier from active history.",
            proxy=record.charged_direct_extraction_proxy,
            target_role="output-sensitive extraction component",
            proof_status="proof_obligation",
            missing_argument=(
                "Show all adaptive quantum algorithms must pay the charged "
                "multi-output relation cost, not only estimate-then-sort."
            ),
        ),
        LowerBoundBlock(
            block_id="LB-B04",
            description="Composition-frontier exclusion barrier.",
            proxy=record.variable_time_rebuild_rms_proxy,
            target_role="prior-composition separation requirement",
            proof_status="proof_obligation",
            missing_argument=(
                "Prove known loop, k-minimum, QBAI, and variable-time theorems "
                "do not imply a matching same-interface upper bound."
            ),
        ),
    )
    strongest = max(blocks, key=lambda block: block.proxy)
    proof_obligations = sum(block.proof_status == "proof_obligation" for block in blocks)
    local_facts = sum(block.proof_status == "local_fact_available" for block in blocks)
    return LowerBoundProgramRecord(
        m=record.m,
        n=record.n,
        k=record.k,
        level_count=record.level_count,
        gamma=record.gamma,
        candidate_proxy=record.charged_candidate_total_proxy,
        adversary_target_proxy=record.adversary_lower_target_proxy,
        target_over_candidate=(
            record.adversary_lower_target_proxy / record.charged_candidate_total_proxy
        ),
        blocks=blocks,
        strongest_block_id=strongest.block_id,
        strongest_block_proxy=strongest.proxy,
        proof_obligation_count=proof_obligations,
        local_facts_count=local_facts,
        lower_bound_claim_status=READINESS,
    )


def lower_bound_program_sweep(
    m_values: Sequence[int],
    **candidate_kwargs: object,
) -> tuple[LowerBoundProgramRecord, ...]:
    """Evaluate a strictly increasing lower-bound proof-program sweep."""

    try:
        values = tuple(_integer(value, "m") for value in m_values)
    except TypeError as error:
        raise TypeError("m_values must be a sequence of integers") from error
    if not values:
        raise ValueError("m_values cannot be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError("m_values must be strictly increasing and unique")
    return tuple(
        lower_bound_program_record(value, **candidate_kwargs)
        for value in values
    )


def lower_bound_program_loglog_slope(
    records: Sequence[LowerBoundProgramRecord], field: str
) -> float:
    """Return a descriptive finite-family slope."""

    rows = tuple(records)
    allowed = {
        "candidate_proxy",
        "adversary_target_proxy",
        "strongest_block_proxy",
    }
    if field not in allowed:
        raise ValueError(f"field must be one of {sorted(allowed)}")
    if len(rows) < 2:
        raise ValueError("at least two records are required")
    if any(not isinstance(row, LowerBoundProgramRecord) for row in rows):
        raise TypeError("records must contain LowerBoundProgramRecord values")
    x = np.log(np.asarray([row.m for row in rows], dtype=np.float64))
    y = np.log(np.asarray([getattr(row, field) for row in rows], dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])
