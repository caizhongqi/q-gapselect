"""Provable capacity envelopes for overlapping bipartite collision graphs."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf, log, sqrt

from .overlap_capacity import OverlapGraph


@dataclass(frozen=True)
class CapacityEnvelope:
    edge_count: int
    maximum_degree: int
    lower_bound: int
    upper_bound: int
    true_matching_size: int
    lower_ratio: float
    upper_ratio: float


@dataclass(frozen=True)
class SurvivalProbabilityEnvelope:
    matching_size: int
    maximum_degree: int
    retention_probability: float
    lower_survival: float
    upper_survival: float


@dataclass(frozen=True)
class SurvivalCapacityInterval:
    observed_survival: float
    trials: int
    failure_probability: float
    survival_lower: float
    survival_upper: float
    capacity_lower: float
    capacity_upper: float


def bipartite_capacity_envelope(graph: OverlapGraph) -> CapacityEnvelope:
    """Return deterministic matching-capacity bounds from edge count and max degree.

    A bipartite graph of maximum degree Delta admits a proper edge colouring
    with Delta colours. Each colour class is a matching, so one class contains
    at least ceil(|E| / Delta) edges. Therefore

        ceil(|E| / Delta) <= nu(G) <= min(|A|, |B|, |E|).

    The lower bound is exact for a star and the upper bound is exact for a
    disjoint matching. This envelope is intentionally coarse but applies to
    arbitrary bipartite overlap without assuming independent witnesses.
    """

    edges = graph.edge_count
    degree = graph.maximum_degree
    truth = graph.matching_size
    if edges == 0:
        lower = upper = 0
    else:
        if degree <= 0:
            raise RuntimeError("nonempty graph must have positive maximum degree")
        lower = ceil(edges / degree)
        upper = min(graph.n_left, graph.n_right, edges)
    return CapacityEnvelope(
        edge_count=edges,
        maximum_degree=degree,
        lower_bound=lower,
        upper_bound=upper,
        true_matching_size=truth,
        lower_ratio=(lower / truth) if truth else 1.0,
        upper_ratio=(upper / truth) if truth else 1.0,
    )


def survival_probability_envelope(
    matching_size: int,
    maximum_degree: int,
    retention_probability: float,
) -> SurvivalProbabilityEnvelope:
    """Bound collision survival after uniform independent vertex retention.

    Let ``K`` be the maximum matching size and ``Delta`` the maximum degree.
    Retain every left and right vertex independently with probability ``q`` and
    let ``S_G(q)`` be the probability that the induced graph contains an edge.

    A fixed maximum matching contains ``K`` vertex-disjoint edges, whose
    survival events are independent. Therefore

        S_G(q) >= 1 - (1-q^2)^K.

    By Konig's theorem, a bipartite graph with matching number ``K`` has a
    vertex cover of size ``K``. Every cover vertex is incident to at most
    ``Delta`` edges, hence ``|E| <= K Delta``. A union bound over edge-survival
    events gives

        S_G(q) <= min(1, K Delta q^2).

    No forest, independent-witness, or bounded-cycle assumption is used.
    """

    if matching_size < 0:
        raise ValueError("matching_size must be non-negative")
    if maximum_degree < 0:
        raise ValueError("maximum_degree must be non-negative")
    if matching_size > 0 and maximum_degree <= 0:
        raise ValueError("a positive matching requires positive maximum degree")
    q = float(retention_probability)
    if not 0.0 <= q <= 1.0:
        raise ValueError("retention_probability must lie in [0,1]")
    lower = 1.0 - (1.0 - q * q) ** matching_size
    upper = min(1.0, matching_size * maximum_degree * q * q)
    return SurvivalProbabilityEnvelope(
        matching_size=matching_size,
        maximum_degree=maximum_degree,
        retention_probability=q,
        lower_survival=float(lower),
        upper_survival=float(upper),
    )


def capacity_bounds_from_population_survival(
    survival_probability: float,
    *,
    retention_probability: float,
    maximum_degree: int,
    domain_cap: int | None = None,
) -> tuple[float, float]:
    """Invert the survival envelope into a matching-capacity interval.

    For population survival probability ``S`` and ``0 < q < 1``,

        S / (Delta q^2) <= K
        K <= log(1-S) / log(1-q^2).

    The upper inversion is the exact packed-matching inverse, but for an
    overlapping graph it is used only as an upper bound. ``domain_cap`` may be
    supplied to enforce the trivial bound ``K <= min(|A|,|B|)`` and to keep the
    interval finite when a confidence bound reaches survival one.
    """

    survival = float(survival_probability)
    q = float(retention_probability)
    if not 0.0 <= survival <= 1.0:
        raise ValueError("survival_probability must lie in [0,1]")
    if not 0.0 < q < 1.0:
        raise ValueError("retention_probability must lie in (0,1)")
    if maximum_degree <= 0:
        if survival == 0.0:
            return 0.0, 0.0
        raise ValueError("positive survival requires positive maximum_degree")
    if domain_cap is not None and domain_cap < 0:
        raise ValueError("domain_cap must be non-negative")

    lower = survival / (maximum_degree * q * q)
    if survival >= 1.0:
        upper = inf
    elif survival <= 0.0:
        upper = 0.0
    else:
        upper = log(1.0 - survival) / log(1.0 - q * q)
    if domain_cap is not None:
        lower = min(lower, float(domain_cap))
        upper = min(upper, float(domain_cap))
    return float(lower), float(upper)


def survival_capacity_confidence_interval(
    successes: int,
    trials: int,
    *,
    retention_probability: float,
    maximum_degree: int,
    failure_probability: float,
    domain_cap: int | None = None,
) -> SurvivalCapacityInterval:
    """Give a finite-sample high-probability capacity certificate.

    Hoeffding's inequality yields a simultaneous population-survival interval
    ``[S_low,S_high]`` with failure probability at most ``delta``. Monotonicity
    of the population inversion then gives

        K >= S_low / (Delta q^2),
        K <= packed_inverse(S_high,q).

    This certificate remains valid for arbitrary bipartite overlap under
    uniform independent vertex retention. It may be loose when ``Delta`` is
    large; that looseness is an explicit overlap penalty, not hidden bias.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0,trials]")
    delta = float(failure_probability)
    if not 0.0 < delta < 1.0:
        raise ValueError("failure_probability must lie in (0,1)")
    observed = successes / trials
    radius = sqrt(log(2.0 / delta) / (2.0 * trials))
    survival_lower = max(0.0, observed - radius)
    survival_upper = min(1.0, observed + radius)
    capacity_lower, _ = capacity_bounds_from_population_survival(
        survival_lower,
        retention_probability=retention_probability,
        maximum_degree=maximum_degree,
        domain_cap=domain_cap,
    )
    _, capacity_upper = capacity_bounds_from_population_survival(
        survival_upper,
        retention_probability=retention_probability,
        maximum_degree=maximum_degree,
        domain_cap=domain_cap,
    )
    return SurvivalCapacityInterval(
        observed_survival=float(observed),
        trials=trials,
        failure_probability=delta,
        survival_lower=float(survival_lower),
        survival_upper=float(survival_upper),
        capacity_lower=float(capacity_lower),
        capacity_upper=float(capacity_upper),
    )


__all__ = [
    "CapacityEnvelope",
    "SurvivalCapacityInterval",
    "SurvivalProbabilityEnvelope",
    "bipartite_capacity_envelope",
    "capacity_bounds_from_population_survival",
    "survival_capacity_confidence_interval",
    "survival_probability_envelope",
]
