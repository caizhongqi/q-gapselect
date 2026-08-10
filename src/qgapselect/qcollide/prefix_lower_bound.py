"""Audited lower-bound profiles for prefix-conditional packed claw finding.

The module distinguishes three levels of validity:

* unconditional adversary composition and ordinary single-claw bounds;
* a distributional random-range packed-claw theorem proved by reduction to
  random-oracle collision finding;
* a general worst-case multi-solution expression that remains conditional on a
  unit-cost outer-relation adversary premise.

None of these statements is silently promoted to the still-missing fully
heterogeneous variable-time matching theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


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


@dataclass(frozen=True)
class RandomRangeClawBound:
    """Distributional scale for two independent random endpoint functions."""

    left_domain: int
    right_domain: int
    range_size: int
    endpoint_cost: float
    expected_cross_claws: float
    expected_isolated_cross_claws: float
    domain_condition_satisfied: bool
    isolated_mass_condition_satisfied: bool
    premises_satisfied: bool
    quantum_omega_scale: float
    classical_omega_scale: float


@dataclass(frozen=True)
class RandomRangeStageBound:
    stage: int
    cumulative_endpoint_cost: float
    bound: RandomRangeClawBound


@dataclass(frozen=True)
class RandomRangePrefixProfile:
    stages: tuple[RandomRangeStageBound, ...]
    maximum_quantum_omega_scale: float
    maximizing_stage: int | None


def homogeneous_packed_claw_lower_bound(
    *,
    left_domain: int,
    right_domain: int,
    packed_solutions: int,
    endpoint_cost: float,
) -> float:
    """Return the conditional composed profile C*((N_A*N_B)/nu)^(1/3).

    For ``packed_solutions == 1`` this specializes to the established
    homogeneous single-claw composition bound. For more than one solution, the
    caller must separately justify the corresponding unit-cost outer-relation
    adversary lower bound.
    """

    if left_domain <= 0 or right_domain <= 0:
        raise ValueError("domain sizes must be positive")
    if not 1 <= packed_solutions <= min(left_domain, right_domain):
        raise ValueError("packed_solutions must lie in [1, min(N_A,N_B)]")
    if endpoint_cost <= 0.0:
        raise ValueError("endpoint_cost must be positive")
    return endpoint_cost * ((left_domain * right_domain) / packed_solutions) ** (1.0 / 3.0)


def random_range_claw_distributional_bound(
    *,
    left_domain: int,
    right_domain: int,
    range_size: int,
    endpoint_cost: float = 1.0,
    minimum_expected_isolated_claws: float = 0.25,
) -> RandomRangeClawBound:
    """Certify the random-range average-case claw lower-bound scale.

    Let ``f:[N_A]->[R]`` and ``g:[N_B]->[R]`` be independent uniform random
    functions. Their disjoint union is itself a random function. Therefore an
    algorithm that finds a cross-domain claw with constant probability also
    finds an ordinary random-function collision, which requires
    ``Omega(R^(1/3))`` quantum queries. Uniform endpoint cost multiplies this
    scale through costed adversary composition.

    The cross-domain problem must be non-vacuous. The returned premise flag
    checks both the random-collision domain regime ``N_A+N_B >= sqrt(R)`` and a
    constant expected mass of isolated cross claws. An isolated output label
    has exactly one preimage on each side, so all such labels form a
    vertex-disjoint matching.
    """

    if left_domain <= 0 or right_domain <= 0:
        raise ValueError("domain sizes must be positive")
    if range_size < 2:
        raise ValueError("range_size must be at least two")
    if endpoint_cost <= 0.0:
        raise ValueError("endpoint_cost must be positive")
    if minimum_expected_isolated_claws <= 0.0:
        raise ValueError("minimum_expected_isolated_claws must be positive")

    expected_cross = (left_domain * right_domain) / range_size
    isolation_factor = (1.0 - 1.0 / range_size) ** (
        left_domain + right_domain - 2
    )
    expected_isolated = expected_cross * isolation_factor
    domain_condition = left_domain + right_domain >= sqrt(range_size)
    isolated_condition = expected_isolated >= minimum_expected_isolated_claws
    premises = domain_condition and isolated_condition
    quantum_scale = endpoint_cost * range_size ** (1.0 / 3.0) if premises else 0.0
    classical_scale = endpoint_cost * sqrt(range_size) if premises else 0.0
    return RandomRangeClawBound(
        left_domain=left_domain,
        right_domain=right_domain,
        range_size=range_size,
        endpoint_cost=endpoint_cost,
        expected_cross_claws=expected_cross,
        expected_isolated_cross_claws=expected_isolated,
        domain_condition_satisfied=domain_condition,
        isolated_mass_condition_satisfied=isolated_condition,
        premises_satisfied=premises,
        quantum_omega_scale=quantum_scale,
        classical_omega_scale=classical_scale,
    )


def random_range_prefix_restriction_profile(
    *,
    incremental_costs: tuple[float, ...],
    left_survivors: tuple[int, ...],
    right_survivors: tuple[int, ...],
    range_sizes: tuple[int, ...],
    minimum_expected_isolated_claws: float = 0.25,
) -> RandomRangePrefixProfile:
    """Return a rigorous distributional stage-restriction lower profile.

    For each stage, all endpoints outside the stated survivor sets are fixed to
    public early rejects. The remaining endpoint labels are independent random
    functions into the stated range. Since the full promise family contains
    every such stage-restricted subfamily, its worst-case complexity is at
    least the maximum certified random-range stage scale.
    """

    levels = len(incremental_costs)
    if levels == 0:
        raise ValueError("at least one stage is required")
    if not (
        len(left_survivors)
        == len(right_survivors)
        == len(range_sizes)
        == levels
    ):
        raise ValueError("all stage arrays must have equal length")
    if any(cost <= 0.0 for cost in incremental_costs):
        raise ValueError("incremental costs must be positive")
    if any(value <= 0 for value in left_survivors + right_survivors):
        raise ValueError("survivor counts must be positive")
    if any(value < 2 for value in range_sizes):
        raise ValueError("range sizes must be at least two")
    if any(a < b for a, b in zip(left_survivors, left_survivors[1:], strict=False)):
        raise ValueError("left survivor counts must be non-increasing")
    if any(a < b for a, b in zip(right_survivors, right_survivors[1:], strict=False)):
        raise ValueError("right survivor counts must be non-increasing")

    cumulative = 0.0
    stages: list[RandomRangeStageBound] = []
    for stage, (increment, left, right, range_size) in enumerate(
        zip(
            incremental_costs,
            left_survivors,
            right_survivors,
            range_sizes,
            strict=True,
        ),
        start=1,
    ):
        cumulative += increment
        bound = random_range_claw_distributional_bound(
            left_domain=left,
            right_domain=right,
            range_size=range_size,
            endpoint_cost=cumulative,
            minimum_expected_isolated_claws=minimum_expected_isolated_claws,
        )
        stages.append(
            RandomRangeStageBound(
                stage=stage,
                cumulative_endpoint_cost=cumulative,
                bound=bound,
            )
        )
    certified = [row for row in stages if row.bound.premises_satisfied]
    maximizing = (
        max(certified, key=lambda row: row.bound.quantum_omega_scale)
        if certified
        else None
    )
    return RandomRangePrefixProfile(
        stages=tuple(stages),
        maximum_quantum_omega_scale=(
            maximizing.bound.quantum_omega_scale if maximizing is not None else 0.0
        ),
        maximizing_stage=maximizing.stage if maximizing is not None else None,
    )


def prefix_stage_restriction_profile(
    *,
    incremental_costs: tuple[float, ...],
    left_survivors: tuple[int, ...],
    right_survivors: tuple[int, ...],
    packed_solutions: tuple[int, ...],
) -> PrefixRestrictionProfile:
    """Compute a conditional general stage-restriction hardness profile.

    Each nonzero stage value assumes that the restricted unit-cost outer
    relation has adversary value at least
    ``((N_A,l*N_B,l)/nu_l)^(1/3)``. A zero packed-solution count contributes
    zero because it cannot contain a positive search instance.
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
    "RandomRangeClawBound",
    "RandomRangePrefixProfile",
    "RandomRangeStageBound",
    "StageRestrictionBound",
    "homogeneous_packed_claw_lower_bound",
    "prefix_stage_restriction_profile",
    "random_range_claw_distributional_bound",
    "random_range_prefix_restriction_profile",
]
