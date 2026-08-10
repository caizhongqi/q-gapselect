"""Tight L2 profiles for an explicit disjoint-stage claw family."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class StageSelectorComponent:
    stage: int
    cumulative_endpoint_cost: float
    outer_adversary_value: float
    composed_stage_hardness: float
    balanced_domain: int | None


@dataclass(frozen=True)
class StageSelectorProfile:
    components: tuple[StageSelectorComponent, ...]
    l2_lower_bound: float
    maximum_stage_lower_bound: float
    aggregation_gain_over_maximum: float
    exact_one_selector_promise: bool
    disjoint_stage_blocks: bool
    outer_values_certified: bool


@dataclass(frozen=True)
class StageSelectorComplexityScale:
    """Normalized scale of a matching lower/upper theorem."""

    lower_profile: StageSelectorProfile
    normalized_l2_scale: float
    lower_bound_notation: str
    upper_bound_notation: str
    tight_up_to_polylogarithmic_factors: bool
    coherent_stage_subroutines_required: bool


def conditional_stage_selector_profile(
    *,
    incremental_costs: tuple[float, ...],
    outer_adversary_values: tuple[float, ...],
    outer_values_certified: bool = False,
) -> StageSelectorProfile:
    """Aggregate stage hardness through a costed exact-one selector.

    Stage ``l`` is a public, disjoint endpoint block whose records cost the
    cumulative prefix cost ``C_l``. Exactly one stage block is a yes-instance.
    Costed exact-one-OR adversary composition gives the L2 aggregation
    ``sqrt(sum_l (C_l A_l)^2)``. The returned profile is rigorous only when the
    supplied outer adversary values have independently been certified for the
    stage decision promises.
    """

    levels = len(incremental_costs)
    if levels == 0:
        raise ValueError("at least one stage is required")
    if len(outer_adversary_values) != levels:
        raise ValueError("cost and adversary arrays must have equal length")
    if any(cost <= 0.0 for cost in incremental_costs):
        raise ValueError("incremental costs must be positive")
    if any(value <= 0.0 for value in outer_adversary_values):
        raise ValueError("outer adversary values must be positive")

    cumulative = 0.0
    components: list[StageSelectorComponent] = []
    for stage, (increment, outer_value) in enumerate(
        zip(incremental_costs, outer_adversary_values, strict=True),
        start=1,
    ):
        cumulative += increment
        components.append(
            StageSelectorComponent(
                stage=stage,
                cumulative_endpoint_cost=cumulative,
                outer_adversary_value=outer_value,
                composed_stage_hardness=cumulative * outer_value,
                balanced_domain=None,
            )
        )
    hardnesses = tuple(row.composed_stage_hardness for row in components)
    maximum = max(hardnesses)
    l2 = sqrt(sum(value * value for value in hardnesses))
    return StageSelectorProfile(
        components=tuple(components),
        l2_lower_bound=l2,
        maximum_stage_lower_bound=maximum,
        aggregation_gain_over_maximum=l2 / maximum,
        exact_one_selector_promise=True,
        disjoint_stage_blocks=True,
        outer_values_certified=outer_values_certified,
    )


def disjoint_single_claw_stage_selector_profile(
    *,
    incremental_costs: tuple[float, ...],
    balanced_domains: tuple[int, ...],
) -> StageSelectorProfile:
    """Return the rigorous lower profile for ordinary single claws.

    Block ``l`` contains two balanced domains of size ``n_l`` and obeys the
    ordinary no-claw versus single-claw decision promise. Its unit-cost
    adversary value is ``Theta(n_l^(2/3))``. With endpoint-generation cost
    ``C_l`` and an exact-one marked-stage selector, composition yields

    ``Omega(sqrt(sum_l C_l^2 n_l^(4/3)))``.
    """

    if len(balanced_domains) != len(incremental_costs):
        raise ValueError("cost and domain arrays must have equal length")
    if any(domain <= 0 for domain in balanced_domains):
        raise ValueError("balanced domain sizes must be positive")
    profile = conditional_stage_selector_profile(
        incremental_costs=incremental_costs,
        outer_adversary_values=tuple(
            domain ** (2.0 / 3.0) for domain in balanced_domains
        ),
        outer_values_certified=True,
    )
    components = tuple(
        StageSelectorComponent(
            stage=row.stage,
            cumulative_endpoint_cost=row.cumulative_endpoint_cost,
            outer_adversary_value=row.outer_adversary_value,
            composed_stage_hardness=row.composed_stage_hardness,
            balanced_domain=balanced_domains[row.stage - 1],
        )
        for row in profile.components
    )
    return StageSelectorProfile(
        components=components,
        l2_lower_bound=profile.l2_lower_bound,
        maximum_stage_lower_bound=profile.maximum_stage_lower_bound,
        aggregation_gain_over_maximum=profile.aggregation_gain_over_maximum,
        exact_one_selector_promise=True,
        disjoint_stage_blocks=True,
        outer_values_certified=True,
    )


def disjoint_single_claw_stage_selector_complexity_scale(
    *,
    incremental_costs: tuple[float, ...],
    balanced_domains: tuple[int, ...],
) -> StageSelectorComplexityScale:
    """Return the matching heterogeneous complexity scale.

    The lower bound is the costed exact-one adversary profile. For the upper
    bound, run the optimal coherent claw subroutine for each stage and compose
    the unequal stage runtimes with variable-time exact-one search. This gives
    the same L2 scale up to polylogarithmic implementation factors.
    """

    profile = disjoint_single_claw_stage_selector_profile(
        incremental_costs=incremental_costs,
        balanced_domains=balanced_domains,
    )
    return StageSelectorComplexityScale(
        lower_profile=profile,
        normalized_l2_scale=profile.l2_lower_bound,
        lower_bound_notation="Omega(L2_stage_hardness)",
        upper_bound_notation="O_tilde(L2_stage_hardness)",
        tight_up_to_polylogarithmic_factors=True,
        coherent_stage_subroutines_required=True,
    )


__all__ = [
    "StageSelectorComponent",
    "StageSelectorComplexityScale",
    "StageSelectorProfile",
    "conditional_stage_selector_profile",
    "disjoint_single_claw_stage_selector_complexity_scale",
    "disjoint_single_claw_stage_selector_profile",
]
