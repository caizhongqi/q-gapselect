"""Instance descriptors and *candidate* complexity functionals.

This module deliberately separates arithmetic from theorem claims.  In
particular, :func:`candidate_layer_bound` evaluates the expression proposed in
the research plan; the function name does not mean that the expression has
already been proved to be an upper or lower query bound.

All values returned here are dimensionless, constant/log-factor-free query
proxies.  Actual oracle calls made by an implementation must be taken from its
``QueryLedger`` instead.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import asin, isfinite, pi, sqrt


@dataclass(frozen=True, slots=True)
class GapProfile:
    """Boundary-gap description of a strict Top-k instance.

    The selected and rejected representations are both retained.  Choosing a
    representation solely by cardinality is unsound on heterogeneous profiles:
    a larger side whose elements have coarse gaps can be cheaper to enumerate.
    """

    means: tuple[float, ...]
    angles: tuple[float, ...]
    k: int
    selected_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    mean_gaps: tuple[float, ...]
    angular_gaps: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.means)

    @property
    def smaller_side_indices(self) -> tuple[int, ...]:
        """Cardinality-minimal representation, not a complexity optimum."""

        if len(self.selected_indices) <= len(self.rejected_indices):
            return self.selected_indices
        return self.rejected_indices

    @property
    def smaller_side_is_complement(self) -> bool:
        return len(self.rejected_indices) < len(self.selected_indices)

    @property
    def smaller_side_size(self) -> int:
        return min(len(self.selected_indices), len(self.rejected_indices))

    @property
    def minimum_mean_gap(self) -> float:
        return min(self.mean_gaps)

    @property
    def minimum_angular_gap(self) -> float:
        return min(self.angular_gaps)


@dataclass(frozen=True, slots=True)
class LayerTerm:
    """One dyadic angular-precision term in the proposed layer functional."""

    level: int
    epsilon: float
    active_count: int
    newly_certified_outputs: int
    value: float


@dataclass(frozen=True, slots=True)
class OrientationLayerComplexity:
    """Candidate charge for one verifiable output representation."""

    representation: str
    output_indices: tuple[int, ...]
    terms: tuple[LayerTerm, ...]

    @property
    def value(self) -> float:
        return sum(term.value for term in self.terms)


@dataclass(frozen=True, slots=True)
class LayerComplexity:
    """Both representation charges and the orientation-optimized candidate."""

    representations: tuple[OrientationLayerComplexity, ...]
    certification_levels: tuple[int, ...]
    theorem_status: str = "conjectural_query_proxy"

    @property
    def chosen(self) -> OrientationLayerComplexity:
        return min(
            self.representations,
            key=lambda item: (
                item.value,
                len(item.output_indices),
                item.representation != "selected",
            ),
        )

    @property
    def value(self) -> float:
        return self.chosen.value

    @property
    def terms(self) -> tuple[LayerTerm, ...]:
        return self.chosen.terms

    @property
    def representation(self) -> str:
        return self.chosen.representation

    @property
    def output_indices(self) -> tuple[int, ...]:
        return self.chosen.output_indices

    @property
    def alternative_value(self) -> float:
        return max(item.value for item in self.representations)


def topk_gap_profile(means: Sequence[float], k: int) -> GapProfile:
    """Return the exact boundary gaps for a strict Top-k instance.

    Raises:
        ValueError: if the instance is malformed or the k/(k+1) boundary is
            tied.  Ties make exact fixed-confidence identification undefined
            without an additional tie-breaking promise.
    """

    values = tuple(float(value) for value in means)
    n = len(values)
    if n < 2:
        raise ValueError("a Top-k instance needs at least two arms")
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < n; got k={k}, n={n}")
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Bernoulli means must be finite and lie in [0, 1]")

    order = tuple(sorted(range(n), key=lambda index: (-values[index], index)))
    selected = order[:k]
    rejected = order[k:]
    selected_boundary = values[selected[-1]]
    rejected_boundary = values[rejected[0]]
    if selected_boundary <= rejected_boundary:
        raise ValueError(
            "the Top-k boundary must be strict: mu_(k) must exceed mu_(k+1)"
        )

    selected_set = frozenset(selected)
    mean_gaps = tuple(
        value - rejected_boundary
        if index in selected_set
        else selected_boundary - value
        for index, value in enumerate(values)
    )
    angles = tuple(asin(sqrt(value)) for value in values)
    selected_angular_boundary = angles[selected[-1]]
    rejected_angular_boundary = angles[rejected[0]]
    angular_gaps = tuple(
        angle - rejected_angular_boundary
        if index in selected_set
        else selected_angular_boundary - angle
        for index, angle in enumerate(angles)
    )
    if any(gap <= 0.0 for gap in mean_gaps + angular_gaps):
        raise ValueError("all mean and angular boundary gaps must be positive")

    return GapProfile(
        means=values,
        angles=angles,
        k=k,
        selected_indices=selected,
        rejected_indices=rejected,
        mean_gaps=mean_gaps,
        angular_gaps=angular_gaps,
    )


def dyadic_certification_level(gap: float, start_epsilon: float = pi / 2.0) -> int:
    """First dyadic scale below a positive discrimination (angular) gap."""

    gap = float(gap)
    start_epsilon = float(start_epsilon)
    if not isfinite(gap) or gap <= 0.0:
        raise ValueError("gap must be positive and finite")
    if not isfinite(start_epsilon) or start_epsilon <= 0.0:
        raise ValueError("start_epsilon must be positive and finite")

    level = 0
    epsilon = start_epsilon
    while epsilon > gap:
        epsilon *= 0.5
        level += 1
    return level


def candidate_layer_profile(
    means: Sequence[float],
    k: int,
    *,
    start_epsilon: float = pi / 2.0,
) -> LayerComplexity:
    r"""Evaluate the proposed dyadic layer functional.

    The evaluated expression is

    .. math::

       \sum_r \frac{\sqrt{N_r(M_r+1)}}{\epsilon_r},

    where every scale and certification gap is measured in Bernoulli rotation
    angle, ``N_r`` counts arms not certifiable before scale ``r``, and ``M_r``
    counts represented output elements first certifiable at that scale.  The
    selected set and rejected-set complement are evaluated independently, and
    the candidate is their minimum.  A certifying implementation can dovetail
    both representations and stop at the first complete certificate.

    This is a research hypothesis, not an activated theorem.  Keeping the
    per-layer terms makes counterexamples and off-by-one definitions visible.
    """

    profile = topk_gap_profile(means, k)
    levels = tuple(
        dyadic_certification_level(gap, start_epsilon)
        for gap in profile.angular_gaps
    )
    max_level = max(levels)
    representations: list[OrientationLayerComplexity] = []
    for representation, indices in (
        ("selected", profile.selected_indices),
        ("rejected_complement", profile.rejected_indices),
    ):
        output = frozenset(indices)
        terms: list[LayerTerm] = []
        for level in range(max_level + 1):
            epsilon = start_epsilon * (2.0**-level)
            active_count = sum(item_level >= level for item_level in levels)
            new_outputs = sum(
                index in output and item_level == level
                for index, item_level in enumerate(levels)
            )
            value = sqrt(active_count * (new_outputs + 1)) / epsilon
            terms.append(
                LayerTerm(
                    level=level,
                    epsilon=epsilon,
                    active_count=active_count,
                    newly_certified_outputs=new_outputs,
                    value=value,
                )
            )
        representations.append(
            OrientationLayerComplexity(
                representation=representation,
                output_indices=indices,
                terms=tuple(terms),
            )
        )
    return LayerComplexity(
        representations=tuple(representations),
        certification_levels=levels,
    )


def candidate_layer_bound(
    means: Sequence[float],
    k: int,
    *,
    start_epsilon: float = pi / 2.0,
) -> float:
    """Return the conjectural layer expression as a scalar.

    ``bound`` is retained to match the paper notation.  Callers must label the
    returned number ``conjectural_query_proxy`` until the claimed theorem and
    matching lower bound are proved.
    """

    return candidate_layer_profile(
        means, k, start_epsilon=start_epsilon
    ).value


def candidate_layer_proxy(
    means: Sequence[float],
    k: int,
    *,
    start_epsilon: float = pi / 2.0,
) -> float:
    """Unambiguous alias for :func:`candidate_layer_bound`."""

    return candidate_layer_bound(means, k, start_epsilon=start_epsilon)


def equal_gap_reference(n: int, k: int, discrimination_gap: float) -> float:
    r"""Return the constant-free equal-angular-gap Top-k reference.

    The expression ``sqrt(k*(n-k))/gap`` is used only as an asymptotic sanity
    reference.  This function does not claim a particular finite-instance
    query constant.
    """

    if n < 2 or not 1 <= k < n:
        raise ValueError("expected n >= 2 and 1 <= k < n")
    if not isfinite(discrimination_gap) or discrimination_gap <= 0.0:
        raise ValueError("discrimination_gap must be positive and finite")
    return sqrt(k * (n - k)) / discrimination_gap


def partition_direct_sum(
    groups: Iterable[tuple[Sequence[float], int]],
    *,
    start_epsilon: float = pi / 2.0,
) -> float:
    """Sum candidate layer proxies for independent partition groups.

    The additive aggregation is intentional: a valid direct-sum theorem must
    show whether independent required outputs really force this behavior.
    This helper evaluates that candidate; it does not prove the theorem.
    """

    materialized = tuple(groups)
    if not materialized:
        raise ValueError("at least one partition group is required")
    return sum(
        candidate_layer_bound(means, k, start_epsilon=start_epsilon)
        for means, k in materialized
    )
