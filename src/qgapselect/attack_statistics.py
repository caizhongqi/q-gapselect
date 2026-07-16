"""Task-clustered inference helpers for Q-GapAttack experiments.

Rows from repeated completions are not independent experimental units.  These
helpers therefore expose paired tests and cluster resampling explicitly rather
than accepting a flat count of generations as a sample size.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist, fmean

import numpy as np


def _probability(value: object, name: str, *, open_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    if not math.isfinite(result) or not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class ProportionInterval:
    successes: int
    total: int
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str = "wilson_score"

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "successes": self.successes,
            "total": self.total,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
        }


def wilson_score_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = 0.95,
) -> ProportionInterval:
    """Return a Wilson interval without treating a zero denominator as data."""

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer")
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie between zero and total")
    confidence = _probability(
        confidence_level, "confidence_level", open_interval=True
    )
    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    estimate = successes / total
    denominator = 1.0 + z * z / total
    centre = (estimate + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return ProportionInterval(
        successes=successes,
        total=total,
        estimate=estimate,
        lower=0.0 if successes == 0 else max(0.0, centre - half_width),
        upper=1.0 if successes == total else min(1.0, centre + half_width),
        confidence_level=confidence,
    )


def _exact_binomial_lower_tail(n: int, observed: int) -> float:
    if observed < 0:
        return 0.0
    if observed >= n:
        return 1.0
    log_terms = [
        math.lgamma(n + 1)
        - math.lgamma(index + 1)
        - math.lgamma(n - index + 1)
        - n * math.log(2.0)
        for index in range(observed + 1)
    ]
    maximum = max(log_terms)
    return min(1.0, math.exp(maximum) * sum(math.exp(item - maximum) for item in log_terms))


@dataclass(frozen=True, slots=True)
class PairedBinaryComparison:
    sample_size: int
    method_successes: int
    baseline_successes: int
    both_success: int
    method_only_success: int
    baseline_only_success: int
    neither_success: int
    absolute_rate_difference: float
    matched_odds_ratio_haldane: float
    exact_mcnemar_p_value: float
    method_interval: ProportionInterval
    baseline_interval: ProportionInterval
    test: str = "exact_two_sided_mcnemar"

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_size": self.sample_size,
            "method_successes": self.method_successes,
            "baseline_successes": self.baseline_successes,
            "both_success": self.both_success,
            "method_only_success": self.method_only_success,
            "baseline_only_success": self.baseline_only_success,
            "neither_success": self.neither_success,
            "absolute_rate_difference": self.absolute_rate_difference,
            "matched_odds_ratio_haldane": self.matched_odds_ratio_haldane,
            "exact_mcnemar_p_value": self.exact_mcnemar_p_value,
            "method_interval": self.method_interval.as_dict(),
            "baseline_interval": self.baseline_interval.as_dict(),
            "test": self.test,
        }


def exact_mcnemar_comparison(
    method_success: Sequence[bool],
    baseline_success: Sequence[bool],
    *,
    confidence_level: float = 0.95,
) -> PairedBinaryComparison:
    """Compare two methods on the same preregistered task-level units."""

    method = tuple(method_success)
    baseline = tuple(baseline_success)
    if not method or len(method) != len(baseline):
        raise ValueError("paired success arrays must have the same positive length")
    if any(not isinstance(item, bool) for item in (*method, *baseline)):
        raise TypeError("paired success arrays must contain booleans")

    both = sum(first and second for first, second in zip(method, baseline, strict=True))
    method_only = sum(
        first and not second for first, second in zip(method, baseline, strict=True)
    )
    baseline_only = sum(
        second and not first for first, second in zip(method, baseline, strict=True)
    )
    neither = len(method) - both - method_only - baseline_only
    discordant = method_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = min(
            1.0,
            2.0 * _exact_binomial_lower_tail(discordant, min(method_only, baseline_only)),
        )
    method_count = both + method_only
    baseline_count = both + baseline_only
    return PairedBinaryComparison(
        sample_size=len(method),
        method_successes=method_count,
        baseline_successes=baseline_count,
        both_success=both,
        method_only_success=method_only,
        baseline_only_success=baseline_only,
        neither_success=neither,
        absolute_rate_difference=(method_count - baseline_count) / len(method),
        matched_odds_ratio_haldane=(method_only + 0.5) / (baseline_only + 0.5),
        exact_mcnemar_p_value=p_value,
        method_interval=wilson_score_interval(
            method_count, len(method), confidence_level=confidence_level
        ),
        baseline_interval=wilson_score_interval(
            baseline_count, len(method), confidence_level=confidence_level
        ),
    )


@dataclass(frozen=True, slots=True)
class ClusterBootstrapDifference:
    sample_size: int
    cluster_count: int
    stratum_count: int
    method_mean: float
    baseline_mean: float
    point_difference: float
    lower: float
    upper: float
    confidence_level: float
    repetitions: int
    seed: int
    method: str = "stratified_cluster_percentile_bootstrap"

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "sample_size": self.sample_size,
            "cluster_count": self.cluster_count,
            "stratum_count": self.stratum_count,
            "method_mean": self.method_mean,
            "baseline_mean": self.baseline_mean,
            "point_difference": self.point_difference,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "repetitions": self.repetitions,
            "seed": self.seed,
            "method": self.method,
        }


def stratified_cluster_bootstrap_difference(
    method_values: Sequence[float],
    baseline_values: Sequence[float],
    cluster_ids: Sequence[Hashable],
    *,
    strata: Sequence[Hashable] | None = None,
    repetitions: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> ClusterBootstrapDifference:
    """Bootstrap a paired mean difference by resampling whole task clusters.

    All rows for one task/repository cluster move together.  Optional strata can
    preserve a preregistered CWE, language, or benchmark composition, provided
    each cluster belongs to exactly one stratum.
    """

    method = tuple(float(item) for item in method_values)
    baseline = tuple(float(item) for item in baseline_values)
    clusters = tuple(cluster_ids)
    if not method or not (len(method) == len(baseline) == len(clusters)):
        raise ValueError("values and cluster_ids must have the same positive length")
    if any(not math.isfinite(item) for item in (*method, *baseline)):
        raise ValueError("bootstrap values must be finite")
    if any(not isinstance(item, Hashable) for item in clusters):
        raise TypeError("cluster_ids must be hashable")
    repeats = _positive_integer(repetitions, "repetitions")
    if repeats < 1000:
        raise ValueError("repetitions must be at least 1000")
    confidence = _probability(
        confidence_level, "confidence_level", open_interval=True
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if strata is None:
        stratum_values: tuple[Hashable, ...] = ("all",) * len(method)
    else:
        stratum_values = tuple(strata)
        if len(stratum_values) != len(method):
            raise ValueError("strata must match the value-array length")
        if any(not isinstance(item, Hashable) for item in stratum_values):
            raise TypeError("strata must be hashable")

    cluster_rows: dict[Hashable, list[int]] = defaultdict(list)
    cluster_stratum: dict[Hashable, Hashable] = {}
    for index, (cluster, stratum) in enumerate(
        zip(clusters, stratum_values, strict=True)
    ):
        cluster_rows[cluster].append(index)
        previous = cluster_stratum.setdefault(cluster, stratum)
        if previous != stratum:
            raise ValueError("each cluster must belong to exactly one stratum")
    clusters_by_stratum: dict[Hashable, list[Hashable]] = defaultdict(list)
    for cluster, stratum in cluster_stratum.items():
        clusters_by_stratum[stratum].append(cluster)

    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled_rows: list[int] = []
        for candidates in clusters_by_stratum.values():
            draws = rng.integers(0, len(candidates), size=len(candidates))
            for draw in draws:
                cluster = candidates[int(draw)]
                sampled_rows.extend(cluster_rows[cluster])
        differences[repeat] = fmean(method[index] - baseline[index] for index in sampled_rows)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(differences, [tail, 1.0 - tail])
    return ClusterBootstrapDifference(
        sample_size=len(method),
        cluster_count=len(cluster_rows),
        stratum_count=len(clusters_by_stratum),
        method_mean=fmean(method),
        baseline_mean=fmean(baseline),
        point_difference=fmean(
            first - second
            for first, second in zip(method, baseline, strict=True)
        ),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence,
        repetitions=repeats,
        seed=seed,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm family-wise adjusted p-values, preserving input keys."""

    if not p_values:
        raise ValueError("p_values cannot be empty")
    checked = {
        str(name): _probability(value, f"p_values[{name!r}]")
        for name, value in p_values.items()
    }
    if len(checked) != len(p_values):
        raise ValueError("p-value names must be unique after string conversion")
    ordered = sorted(checked.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in checked}
