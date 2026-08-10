"""Cost-profile lower bounds for prefix-conditional packed claw finding.

The bounds in this module are valid for the explicit composed-oracle model
stated in ``docs/qcollide_weighted_prefix_lower_bound.md``. They are not a
claim that every variable-time implementation is characterized by the same
profile.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageRestrictionBound:
    stage: int
    cumulative_endpoint_cost: float
    left_survivors: int
    right_survivors: int
    packed_solutions: int
    lower_bound: float


@dataclass(frozen=True)
class PrefixRestrictionProfile:
    stages: tuple[StageRestrictionBound, ...]
    maximum_lower_bound: float
    maximizing_stage: int | None


def homogeneous_packed_claw_lower_bound(
    *,
    left_domain: int,
    right_domain: int,
    packed_solutions: int,
    endpoint_cost: float,
) -> float:
    """Return C * ((N_A N_B) / nu)^(1/3) in the composed-oracle model."""

    if left_domain <= 0 or right_domain <= 0:
        raise ValueError("domain sizes must be positive")
    if not 1 <= packed_solutions <= min(left_domain, right_domain):
        raise ValueError("packed_solutions must lie in [1, min(N_A,N_B)]")
    if endpoint_cost <= 0.0:
        raise ValueError("endpoint_cost must be positive")
    return endpoint_cost * ((left_domain * right_domain) / packed_solutions) ** (1.0 / 3.0)


def prefix_stage_restriction_profile(
    *,
    incremental_costs: tuple[float, ...],
    left_survivors: tuple[int, ...],
    right_survivors: tuple[int, ...],
    packed_solutions: tuple[int, ...],
) -> PrefixRestrictionProfile:
    """Compute the maximum lower bound obtained by restricting to one stage.

    Stage ``l`` is interpreted as a homogeneous composed packed-claw instance
    on the endpoints surviving through that stage. A zero packed-solution
    count contributes a zero bound because it cannot contain a positive search
    instance.
    """

    levels = len(incremental_costs)
    if levels == 0:
        raise ValueError("at least one stage is required")
    if not (
        len(left_survivors)
        == len(right_survivors)
        == len(packed_solutions)
        == levels
    ):
        raise ValueError("all stage arrays must have equal length")
    if any(cost <= 0.0 for cost in incremental_costs):
        raise ValueError("incremental costs must be positive")
    if any(value <= 0 for value in left_survivors + right_survivors):
        raise ValueError("survivor counts must be positive")
    if any(a < b for a, b in zip(left_survivors, left_survivors[1:], strict=False)):
        raise ValueError("left survivor counts must be non-increasing")
    if any(a < b for a, b in zip(right_survivors, right_survivors[1:], strict=False)):
        raise ValueError("right survivor counts must be non-increasing")

    cumulative = 0.0
    rows: list[StageRestrictionBound] = []
    for stage, (increment, left, right, solutions) in enumerate(
        zip(
            incremental_costs,
            left_survivors,
            right_survivors,
            packed_solutions,
            strict=True,
        ),
        start=1,
    ):
        cumulative += increment
        if solutions < 0 or solutions > min(left, right):
            raise ValueError("invalid packed solution count at a stage")
        bound = (
            0.0
            if solutions == 0
            else homogeneous_packed_claw_lower_bound(
                left_domain=left,
                right_domain=right,
                packed_solutions=solutions,
                endpoint_cost=cumulative,
            )
        )
        rows.append(
            StageRestrictionBound(
                stage=stage,
                cumulative_endpoint_cost=cumulative,
                left_survivors=left,
                right_survivors=right,
                packed_solutions=solutions,
                lower_bound=bound,
            )
        )
    positive = [row for row in rows if row.lower_bound > 0.0]
    maximizing = max(positive, key=lambda row: row.lower_bound) if positive else None
    return PrefixRestrictionProfile(
        stages=tuple(rows),
        maximum_lower_bound=max((row.lower_bound for row in rows), default=0.0),
        maximizing_stage=maximizing.stage if maximizing is not None else None,
    )


__all__ = [
    "PrefixRestrictionProfile",
    "StageRestrictionBound",
    "homogeneous_packed_claw_lower_bound",
    "prefix_stage_restriction_profile",
]
