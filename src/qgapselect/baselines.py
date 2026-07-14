"""Transparent complexity baselines for Q-GapSelect experiments.

The quantum entries in this module are analytic *proxies*.  They are useful
for falsifying a proposed scaling law, but they are not substitutes for an
implemented algorithm's ``QueryLedger`` or a proved finite-query theorem.
Every returned estimate therefore carries an explicit claim status.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import ceil, inf, isfinite, log, sqrt

from .complexity import candidate_layer_bound, topk_gap_profile


@dataclass(frozen=True, slots=True)
class BaselineEstimate:
    """A named estimate whose epistemic status travels with the number."""

    method: str
    value: float
    unit: str
    claim_status: str
    description: str


def uniform_ae_kmin_proxy(means: Sequence[float], k: int) -> float:
    r"""Uniform angular estimation followed by approximate k-minimum finding.

    The constant/log-factor-free proxy is

    ``sqrt(n * (min(k, n-k) + 1)) / gamma_min``, where ``gamma`` is
    the canonical Bernoulli rotation-angle gap.

    It intentionally uses the smallest angular boundary gap globally; unlike the
    candidate layer expression, this baseline cannot exploit heterogeneous
    certification scales.
    """

    profile = topk_gap_profile(means, k)
    return (
        sqrt(profile.n * (profile.smaller_side_size + 1))
        / profile.minimum_angular_gap
    )


def repeated_qbai_proxy(means: Sequence[float], k: int) -> float:
    r"""Optimistic sequential-QBAI proxy for the better representation.

    At each round, all still-valid desired outputs are treated as acceptable;
    competitors on that same side are not charged.  This is deliberately
    favorable to the baseline and avoids imposing a needless total ordering
    on tied Top-k members.  The round cost is the standard BAI-shaped
    ``sqrt(sum_j gap_j**-2)`` against the incorrect side.

    The rounds are summed because every required output must be emitted.  A
    direct-sum theorem for this cost is *not* asserted here.
    """

    profile = topk_gap_profile(means, k)

    def orientation_cost(indices: Sequence[int], sign: float) -> float:
        desired = tuple(sign * profile.angles[index] for index in indices)
        desired_indices = frozenset(indices)
        incorrect = tuple(
            sign * profile.angles[index]
            for index in range(profile.n)
            if index not in desired_indices
        )
        total = 0.0
        for score in sorted(desired, reverse=True):
            differences = tuple(score - competitor for competitor in incorrect)
            if any(difference <= 0.0 for difference in differences):
                raise ValueError("the requested output side is not strictly separated")
            total += sqrt(sum(difference**-2 for difference in differences))
        return total

    return min(
        orientation_cost(profile.selected_indices, 1.0),
        orientation_cost(profile.rejected_indices, -1.0),
    )


def classical_uniform_proxy(means: Sequence[float], k: int) -> float:
    r"""Weak Hoeffding proxy ``n/Delta_min**2`` for regression only."""

    profile = topk_gap_profile(means, k)
    return profile.n / (profile.minimum_mean_gap**2)


def _bernoulli_kl(first: float, second: float) -> float:
    def term(probability: float, reference: float) -> float:
        if probability == 0.0:
            return 0.0
        if reference == 0.0:
            return inf
        return probability * log(probability / reference)

    return term(first, second) + term(1.0 - first, 1.0 - second)


def bernoulli_jensen_shannon(first: float, second: float) -> float:
    """Symmetric finite information distance between two Bernoulli laws."""

    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in (first, second)):
        raise ValueError("Bernoulli parameters must lie in [0, 1]")
    midpoint = 0.5 * (first + second)
    return 0.5 * (
        _bernoulli_kl(first, midpoint) + _bernoulli_kl(second, midpoint)
    )


def classical_information_proxy(means: Sequence[float], k: int) -> float:
    r"""Instance-wise inverse-information proxy using Bernoulli JS distance.

    This comparator has the correct endpoint scale: distinguishing means
    ``eta`` and ``2*eta`` costs order ``1/eta``, not the weak Hoeffding
    ``1/eta**2``.  It is an analytic information proxy, not a finite-sample
    implementation or an activated instance-optimal theorem.
    """

    profile = topk_gap_profile(means, k)
    selected = frozenset(profile.selected_indices)
    selected_boundary = profile.means[profile.selected_indices[-1]]
    rejected_boundary = profile.means[profile.rejected_indices[0]]
    information = tuple(
        bernoulli_jensen_shannon(
            mean,
            rejected_boundary if index in selected else selected_boundary,
        )
        for index, mean in enumerate(profile.means)
    )
    if any(value <= 0.0 for value in information):
        raise ValueError("strict Top-k instances must have positive boundary information")
    return sum(1.0 / value for value in information)


def classical_uniform_hoeffding_upper(
    means: Sequence[float], k: int, *, delta: float = 0.05
) -> int:
    """A finite classical sample upper bound from Hoeffding plus a union bound.

    Sampling every arm ``m`` times so that all empirical errors are at most
    ``Delta_min/4`` is sufficient to recover a strict Top-k set.  This bound is
    intentionally simple rather than instance optimal.
    """

    if not isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    profile = topk_gap_profile(means, k)
    error = profile.minimum_mean_gap / 4.0
    samples_per_arm = ceil(log(2.0 * profile.n / delta) / (2.0 * error**2))
    return profile.n * samples_per_arm


def candidate_layer_estimate(means: Sequence[float], k: int) -> float:
    """Expose the proposed expression alongside the comparison baselines."""

    return candidate_layer_bound(means, k)


def baseline_estimates(
    means: Sequence[float],
    k: int,
    *,
    include_finite_classical_bound: bool = False,
    delta: float = 0.05,
) -> tuple[BaselineEstimate, ...]:
    """Evaluate all preregistered comparison functionals for one instance."""

    estimates = [
        BaselineEstimate(
            method="candidate_layer",
            value=candidate_layer_estimate(means, k),
            unit="normalized_query_proxy",
            claim_status="conjectural_query_proxy",
            description="dyadic sum sqrt(N_r(M_r+1))/epsilon_r",
        ),
        BaselineEstimate(
            method="prior_uniform_ae_kmin",
            value=uniform_ae_kmin_proxy(means, k),
            unit="normalized_query_proxy",
            claim_status="analytic_baseline_proxy",
            description=(
                "uniform angular estimation plus a Gao-style approximate k-min routine"
            ),
        ),
        BaselineEstimate(
            method="repeated_qbai",
            value=repeated_qbai_proxy(means, k),
            unit="normalized_query_proxy",
            claim_status="optimistic_analytic_baseline_proxy",
            description="sequential recovery using the cheaper certified representation",
        ),
        BaselineEstimate(
            method="classical_information",
            value=classical_information_proxy(means, k),
            unit="normalized_sample_proxy",
            claim_status="analytic_information_proxy",
            description="sum of inverse Bernoulli Jensen-Shannon boundary information",
        ),
        BaselineEstimate(
            method="classical_uniform",
            value=classical_uniform_proxy(means, k),
            unit="normalized_sample_proxy",
            claim_status="analytic_baseline_proxy",
            description="weak Hoeffding n / Delta_min^2 without confidence logs",
        ),
    ]
    if include_finite_classical_bound:
        estimates.append(
            BaselineEstimate(
                method="classical_uniform_hoeffding_upper",
                value=float(classical_uniform_hoeffding_upper(means, k, delta=delta)),
                unit="bernoulli_samples",
                claim_status="rigorous_finite_upper_bound",
                description="uniform allocation with Hoeffding and a union bound",
            )
        )
    return tuple(estimates)


def partition_baseline_estimates(
    groups: Iterable[tuple[Sequence[float], int]],
) -> tuple[BaselineEstimate, ...]:
    """Add per-group proxies for a required-output partition direct sum."""

    materialized = tuple(groups)
    if not materialized:
        raise ValueError("at least one partition group is required")

    methods: tuple[
        tuple[str, Callable[[Sequence[float], int], float], str, str], ...
    ] = (
        (
            "candidate_layer",
            candidate_layer_estimate,
            "normalized_query_proxy",
            "conjectural_direct_sum_proxy",
        ),
        (
            "prior_uniform_ae_kmin",
            uniform_ae_kmin_proxy,
            "normalized_query_proxy",
            "analytic_direct_sum_baseline_proxy",
        ),
        (
            "repeated_qbai",
            repeated_qbai_proxy,
            "normalized_query_proxy",
            "optimistic_analytic_direct_sum_baseline_proxy",
        ),
        (
            "classical_information",
            classical_information_proxy,
            "normalized_sample_proxy",
            "analytic_information_direct_sum_proxy",
        ),
        (
            "classical_uniform",
            classical_uniform_proxy,
            "normalized_sample_proxy",
            "analytic_direct_sum_baseline_proxy",
        ),
    )
    return tuple(
        BaselineEstimate(
            method=method,
            value=sum(function(means, k) for means, k in materialized),
            unit=unit,
            claim_status=status,
            description="sum of independently required partition-group costs",
        )
        for method, function, unit, status in methods
    )
