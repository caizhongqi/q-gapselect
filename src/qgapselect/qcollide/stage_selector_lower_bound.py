"""Tight mixed-norm profiles for an explicit disjoint-stage claw family."""

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
class CollisionWeightedCostNorm:
    domain_mass_fractions: tuple[float, ...]
    collision_weighted_norm: float
    endpoint_rms_cost: float
    endpoint_rms_over_collision_norm: float


@dataclass(frozen=True)
class StageSelectorComplexityScale:
    """Normalized scale of a matching lower/upper theorem."""

    lower_profile: StageSelectorProfile
    normalized_l2_scale: float
    total_balanced_domain: int
    cost_norm: CollisionWeightedCostNorm
    mixed_norm_identity_error: float
    lower_bound_notation: str
    upper_bound_notation: str
    tight_up_to_polylogarithmic_factors: bool
    coherent_stage_subroutines_required: bool


def collision_weighted_cost_norm(
    *,
    cumulative_costs: tuple[float, ...],
    domain_masses: tuple[int, ...],
) -> CollisionWeightedCostNorm:
    """Compare the claw mixed norm with ordinary endpoint RMS cost.

    For total balanced domain ``N`` and mass fraction ``p_l=n_l/N``, the
    collision-weighted norm is

    ``sqrt(sum_l C_l^2 p_l^(4/3))``.

    The ordinary endpoint RMS is ``sqrt(sum_l C_l^2 p_l)``. The former is no
    larger, and it is the exact cost factor in the partitioned single-claw law.
    """

    if not cumulative_costs:
        raise ValueError("at least one cost is required")
    if len(cumulative_costs) != len(domain_masses):
        raise ValueError("cost and domain arrays must have equal length")
    if any(cost <= 0.0 for cost in cumulative_costs):
        raise ValueError("cumulative costs must be positive")
    if any(mass <= 0 for mass in domain_masses):
        raise ValueError("domain masses must be positive")
    total = sum(domain_masses)
    fractions = tuple(mass / total for mass in domain_masses)
    collision_norm = sqrt(
        sum(
            cost * cost * fraction ** (4.0 / 3.0)
            for cost, fraction in zip(cumulative_costs, fractions, strict=True)
        )
    )
    endpoint_rms = sqrt(
        sum(
            cost * cost * fraction
            for cost, fraction in zip(cumulative_costs, fractions, strict=True)
        )
    )
    return CollisionWeightedCostNorm(
        domain_mass_fractions=fractions,
        collision_weighted_norm=collision_norm,
        endpoint_rms_cost=endpoint_rms,
        endpoint_rms_over_collision_norm=endpoint_rms / collision_norm,
    )


def conditional_stage_selector_profile(
    *,
    incremental_costs: tuple[float, ...],
    outer_adversary_values: tuple[float, ...],
    outer_values_certified: bool = False,
) -> StageSelectorProfile:
    """Aggregate stage hardness through a costed exact-one selector."""

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
    """Return the rigorous lower profile for ordinary single claws."""

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
    """Return the matching partitioned heterogeneous complexity law.

    The lower bound is costed exact-one adversary composition. The upper bound
    coherently combines optimal stage claw subroutines with variable-time
    exact-one search. In mixed-norm form, for ``N=sum_l n_l`` and
    ``p_l=n_l/N``, the scale is

    ``N^(2/3) sqrt(sum_l C_l^2 p_l^(4/3))``.
    """

    profile = disjoint_single_claw_stage_selector_profile(
        incremental_costs=incremental_costs,
        balanced_domains=balanced_domains,
    )
    cumulative_costs = tuple(
        component.cumulative_endpoint_cost for component in profile.components
    )
    cost_norm = collision_weighted_cost_norm(
        cumulative_costs=cumulative_costs,
        domain_masses=balanced_domains,
    )
    total_domain = sum(balanced_domains)
    reconstructed = total_domain ** (2.0 / 3.0) * cost_norm.collision_weighted_norm
    return StageSelectorComplexityScale(
        lower_profile=profile,
        normalized_l2_scale=profile.l2_lower_bound,
        total_balanced_domain=total_domain,
        cost_norm=cost_norm,
        mixed_norm_identity_error=abs(reconstructed - profile.l2_lower_bound),
        lower_bound_notation="Omega(N^(2/3) * collision_weighted_cost_norm)",
        upper_bound_notation="O_tilde(N^(2/3) * collision_weighted_cost_norm)",
        tight_up_to_polylogarithmic_factors=True,
        coherent_stage_subroutines_required=True,
    )


__all__ = [
    "CollisionWeightedCostNorm",
    "StageSelectorComponent",
    "StageSelectorComplexityScale",
    "StageSelectorProfile",
    "collision_weighted_cost_norm",
    "conditional_stage_selector_profile",
    "disjoint_single_claw_stage_selector_complexity_scale",
    "disjoint_single_claw_stage_selector_profile",
]
