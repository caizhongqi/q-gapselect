"""Paired inference for frozen, non-commercial algorithm experiments.

The fixture pair is the experimental unit throughout this module.  Success
comparisons use the full paired 2x2 table, while optional resource differences
are summarized over *all* fixture pairs.  In particular, resource use is never
conditioned on both methods succeeding; doing so would select an outcome-
dependent subset and can reverse the apparent resource comparison.

These utilities provide finite-sample descriptive inference.  They do not, by
themselves, establish superiority, a quantum implementation, or a quantum
advantage.
"""

from __future__ import annotations

import hashlib
import math
import operator
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

CLAIM_SCOPE = (
    "frozen_fixture_level_paired_statistics_"
    "no_superiority_or_quantum_advantage_claim"
)
QUANTUM_REFERENCE_ADAPTER_CLAIM_SCOPE = (
    "frozen_quantum_reference_gap_aided_paired_adapter_"
    "no_superiority_or_quantum_advantage_claim"
)
QGAPSELECT_METHOD_ID = "qgapselect"
GAP_AIDED_BASELINE_METHOD_ID = "independent_iae_topk"


class FrozenQuantumReferenceRunLike(Protocol):
    """Structural input required from a frozen quantum-reference run.

    A protocol keeps this statistics-only module independent of the benchmark
    runner and therefore avoids an import cycle.
    """

    family_id: str
    instance_id: str
    repetition: int
    method_id: str
    certified_exact_recovery: bool
    coherent_queries: int


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _probability(value: object, name: str, *, open_interval: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    interval = "(0, 1)" if open_interval else "[0, 1]"
    if not math.isfinite(result) or not valid:
        raise ValueError(f"{name} must be finite and lie in {interval}")
    return result


def _resource(value: object | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_claim_scope(value: object) -> str:
    if value != CLAIM_SCOPE:
        raise ValueError(f"claim_scope must equal {CLAIM_SCOPE!r}")
    return CLAIM_SCOPE


@dataclass(frozen=True, slots=True)
class PairedFixtureOutcome:
    """One fixture evaluated by a method and its paired baseline.

    Resources are optional, but they must be supplied as a pair.  A collection
    passed to :func:`analyze_paired_fixtures` must either supply resources for
    every fixture or for none of them.
    """

    method_success: bool
    baseline_success: bool
    method_resource: float | None = None
    baseline_resource: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method_success, bool):
            raise TypeError("method_success must be bool")
        if not isinstance(self.baseline_success, bool):
            raise TypeError("baseline_success must be bool")
        method_resource = _resource(self.method_resource, "method_resource")
        baseline_resource = _resource(self.baseline_resource, "baseline_resource")
        if (method_resource is None) != (baseline_resource is None):
            raise ValueError("method_resource and baseline_resource must be supplied together")
        object.__setattr__(self, "method_resource", method_resource)
        object.__setattr__(self, "baseline_resource", baseline_resource)

    def as_dict(self) -> dict[str, bool | float | None]:
        return {
            "method_success": self.method_success,
            "baseline_success": self.baseline_success,
            "method_resource": self.method_resource,
            "baseline_resource": self.baseline_resource,
        }


@dataclass(frozen=True, slots=True)
class PairedBinaryStatistics:
    """Full paired 2x2 table and its exact McNemar comparison."""

    pair_count: int
    both_success: int
    method_only_success: int
    baseline_only_success: int
    neither_success: int
    method_successes: int
    baseline_successes: int
    method_success_rate: float
    baseline_success_rate: float
    risk_difference: float
    exact_two_sided_mcnemar_p_value: float
    test: str = "exact_two_sided_mcnemar"
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        pair_count = _integer(self.pair_count, "pair_count", minimum=1)
        counts = (
            _integer(self.both_success, "both_success"),
            _integer(self.method_only_success, "method_only_success"),
            _integer(self.baseline_only_success, "baseline_only_success"),
            _integer(self.neither_success, "neither_success"),
        )
        if sum(counts) != pair_count:
            raise ValueError("paired 2x2 counts must sum to pair_count")
        method_successes = _integer(self.method_successes, "method_successes")
        baseline_successes = _integer(self.baseline_successes, "baseline_successes")
        if method_successes != counts[0] + counts[1]:
            raise ValueError("method_successes is inconsistent with the paired table")
        if baseline_successes != counts[0] + counts[2]:
            raise ValueError("baseline_successes is inconsistent with the paired table")
        method_rate = _probability(self.method_success_rate, "method_success_rate")
        baseline_rate = _probability(self.baseline_success_rate, "baseline_success_rate")
        if not math.isclose(method_rate, method_successes / pair_count, abs_tol=1e-15):
            raise ValueError("method_success_rate is inconsistent with method_successes")
        if not math.isclose(baseline_rate, baseline_successes / pair_count, abs_tol=1e-15):
            raise ValueError("baseline_success_rate is inconsistent with baseline_successes")
        risk_difference = float(self.risk_difference)
        if not math.isfinite(risk_difference) or not -1.0 <= risk_difference <= 1.0:
            raise ValueError("risk_difference must be finite and lie in [-1, 1]")
        if not math.isclose(risk_difference, method_rate - baseline_rate, abs_tol=1e-15):
            raise ValueError("risk_difference is inconsistent with the success rates")
        p_value = _probability(
            self.exact_two_sided_mcnemar_p_value,
            "exact_two_sided_mcnemar_p_value",
        )
        expected_p_value = exact_two_sided_mcnemar_p_value(counts[1], counts[2])
        if not math.isclose(p_value, expected_p_value, abs_tol=1e-15):
            raise ValueError("exact_two_sided_mcnemar_p_value is inconsistent with the table")
        if self.test != "exact_two_sided_mcnemar":
            raise ValueError("test must equal 'exact_two_sided_mcnemar'")
        _validate_claim_scope(self.claim_scope)

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "pair_count": self.pair_count,
            "both_success": self.both_success,
            "method_only_success": self.method_only_success,
            "baseline_only_success": self.baseline_only_success,
            "neither_success": self.neither_success,
            "method_successes": self.method_successes,
            "baseline_successes": self.baseline_successes,
            "method_success_rate": self.method_success_rate,
            "baseline_success_rate": self.baseline_success_rate,
            "risk_difference": self.risk_difference,
            "exact_two_sided_mcnemar_p_value": self.exact_two_sided_mcnemar_p_value,
            "test": self.test,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class FixturePairedBootstrapInterval:
    """Percentile interval from resampling whole fixture pairs."""

    pair_count: int
    point_risk_difference: float
    lower: float
    upper: float
    confidence_level: float
    repetitions: int
    seed: int
    resampling_unit: str = "fixture_pair"
    method: str = "paired_fixture_percentile_bootstrap"
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _integer(self.pair_count, "pair_count", minimum=1)
        for name in ("point_risk_difference", "lower", "upper"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [-1, 1]")
        if self.lower > self.upper:
            raise ValueError("lower cannot exceed upper")
        _probability(self.confidence_level, "confidence_level", open_interval=True)
        _integer(self.repetitions, "repetitions", minimum=1)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.resampling_unit != "fixture_pair":
            raise ValueError("resampling_unit must equal 'fixture_pair'")
        if self.method != "paired_fixture_percentile_bootstrap":
            raise ValueError("method must equal 'paired_fixture_percentile_bootstrap'")
        _validate_claim_scope(self.claim_scope)

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "pair_count": self.pair_count,
            "point_risk_difference": self.point_risk_difference,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "repetitions": self.repetitions,
            "seed": self.seed,
            "resampling_unit": self.resampling_unit,
            "method": self.method,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class UnconditionalPairedResourceStatistics:
    """Resource differences over every preregistered fixture pair."""

    pair_count: int
    method_mean_resource: float
    baseline_mean_resource: float
    mean_paired_difference: float
    min_paired_difference: float
    max_paired_difference: float
    conditioning: str = "all_fixture_pairs_unconditional"
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _integer(self.pair_count, "pair_count", minimum=1)
        for name in (
            "method_mean_resource",
            "baseline_mean_resource",
            "mean_paired_difference",
            "min_paired_difference",
            "max_paired_difference",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.method_mean_resource < 0.0 or self.baseline_mean_resource < 0.0:
            raise ValueError("mean resources must be non-negative")
        if self.min_paired_difference > self.max_paired_difference:
            raise ValueError("min_paired_difference cannot exceed max_paired_difference")
        if not (
            self.min_paired_difference
            <= self.mean_paired_difference
            <= self.max_paired_difference
        ):
            raise ValueError("mean_paired_difference must lie within the observed range")
        expected = self.method_mean_resource - self.baseline_mean_resource
        if not math.isclose(self.mean_paired_difference, expected, abs_tol=1e-12):
            raise ValueError("mean_paired_difference is inconsistent with mean resources")
        if self.conditioning != "all_fixture_pairs_unconditional":
            raise ValueError("resource statistics must be unconditional over all fixture pairs")
        _validate_claim_scope(self.claim_scope)

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "pair_count": self.pair_count,
            "method_mean_resource": self.method_mean_resource,
            "baseline_mean_resource": self.baseline_mean_resource,
            "mean_paired_difference": self.mean_paired_difference,
            "min_paired_difference": self.min_paired_difference,
            "max_paired_difference": self.max_paired_difference,
            "conditioning": self.conditioning,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class HolmAdjustedPValue:
    """One hypothesis in a Holm family, returned in input order."""

    hypothesis_id: str
    raw_p_value: float
    adjusted_p_value: float
    rank: int
    family_size: int
    rejected_at_alpha: bool
    alpha: float
    method: str = "holm_fwer"
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis_id, str) or not self.hypothesis_id.strip():
            raise TypeError("hypothesis_id must be a non-empty string")
        _probability(self.raw_p_value, "raw_p_value")
        _probability(self.adjusted_p_value, "adjusted_p_value")
        if self.adjusted_p_value < self.raw_p_value:
            raise ValueError("adjusted_p_value cannot be smaller than raw_p_value")
        family_size = _integer(self.family_size, "family_size", minimum=1)
        rank = _integer(self.rank, "rank", minimum=1)
        if rank > family_size:
            raise ValueError("rank cannot exceed family_size")
        alpha = _probability(self.alpha, "alpha", open_interval=True)
        if not isinstance(self.rejected_at_alpha, bool):
            raise TypeError("rejected_at_alpha must be bool")
        if self.rejected_at_alpha != (self.adjusted_p_value <= alpha):
            raise ValueError("rejected_at_alpha is inconsistent with adjusted_p_value")
        if self.method != "holm_fwer":
            raise ValueError("method must equal 'holm_fwer'")
        _validate_claim_scope(self.claim_scope)

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "raw_p_value": self.raw_p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "rank": self.rank,
            "family_size": self.family_size,
            "rejected_at_alpha": self.rejected_at_alpha,
            "alpha": self.alpha,
            "method": self.method,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class HolmFWERCorrection:
    """A complete Holm family with adjusted values in original input order."""

    adjustments: tuple[HolmAdjustedPValue, ...]
    alpha: float
    method: str = "holm_fwer"
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.adjustments, tuple) or not self.adjustments:
            raise TypeError("adjustments must be a non-empty tuple")
        if any(not isinstance(item, HolmAdjustedPValue) for item in self.adjustments):
            raise TypeError("adjustments must contain HolmAdjustedPValue records")
        alpha = _probability(self.alpha, "alpha", open_interval=True)
        if any(item.alpha != alpha for item in self.adjustments):
            raise ValueError("every adjustment must use the family alpha")
        if len({item.hypothesis_id for item in self.adjustments}) != len(self.adjustments):
            raise ValueError("hypothesis identifiers must be unique")
        if any(item.family_size != len(self.adjustments) for item in self.adjustments):
            raise ValueError("each adjustment must report the complete family size")
        by_rank = sorted(self.adjustments, key=lambda item: item.rank)
        if [item.rank for item in by_rank] != list(range(1, len(by_rank) + 1)):
            raise ValueError("Holm ranks must be exactly 1 through family_size")
        if any(
            first.raw_p_value > second.raw_p_value
            for first, second in zip(by_rank, by_rank[1:], strict=False)
        ):
            raise ValueError("raw p-values must be non-decreasing in Holm rank order")
        if any(
            first.adjusted_p_value > second.adjusted_p_value
            for first, second in zip(by_rank, by_rank[1:], strict=False)
        ):
            raise ValueError("adjusted p-values must be non-decreasing in Holm rank order")
        if self.method != "holm_fwer":
            raise ValueError("method must equal 'holm_fwer'")
        _validate_claim_scope(self.claim_scope)

    def as_dict(self) -> dict[str, object]:
        return {
            "adjustments": [item.as_dict() for item in self.adjustments],
            "alpha": self.alpha,
            "method": self.method,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class FrozenPairedAnalysis:
    """Paired success inference plus optional all-pair resource summary."""

    binary: PairedBinaryStatistics
    bootstrap: FixturePairedBootstrapInterval
    resources: UnconditionalPairedResourceStatistics | None
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.binary, PairedBinaryStatistics):
            raise TypeError("binary must be PairedBinaryStatistics")
        if not isinstance(self.bootstrap, FixturePairedBootstrapInterval):
            raise TypeError("bootstrap must be FixturePairedBootstrapInterval")
        if self.resources is not None and not isinstance(
            self.resources, UnconditionalPairedResourceStatistics
        ):
            raise TypeError("resources must be UnconditionalPairedResourceStatistics or None")
        if self.binary.pair_count != self.bootstrap.pair_count:
            raise ValueError("binary and bootstrap pair counts must agree")
        if self.resources is not None and self.resources.pair_count != self.binary.pair_count:
            raise ValueError("resource summary must include every fixture pair")
        _validate_claim_scope(self.claim_scope)

    def as_dict(self) -> dict[str, object]:
        return {
            "binary": self.binary.as_dict(),
            "bootstrap": self.bootstrap.as_dict(),
            "resources": None if self.resources is None else self.resources.as_dict(),
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class FrozenQuantumFamilyPairedAnalysis:
    """One family comparison against the gap-aided independent-IAE baseline."""

    family_id: str
    analysis: FrozenPairedAnalysis
    bootstrap_seed: int
    method_id: str = QGAPSELECT_METHOD_ID
    baseline_method_id: str = GAP_AIDED_BASELINE_METHOD_ID
    comparison_information_matched: bool = False
    baseline_is_gap_aided: bool = True
    baseline_information: str = "k_and_public_gap_floor"
    resource_conditioning: str = "all_fixture_pairs_unconditional"
    claim_scope: str = QUANTUM_REFERENCE_ADAPTER_CLAIM_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, str) or not self.family_id.strip():
            raise TypeError("family_id must be a non-empty string")
        if not isinstance(self.analysis, FrozenPairedAnalysis):
            raise TypeError("analysis must be FrozenPairedAnalysis")
        if self.analysis.resources is None:
            raise ValueError("the adapter analysis must include all-pair coherent resources")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(self.bootstrap_seed, int):
            raise TypeError("bootstrap_seed must be an integer")
        if self.analysis.bootstrap.seed != self.bootstrap_seed:
            raise ValueError("bootstrap_seed must equal the nested bootstrap seed")
        if self.method_id != QGAPSELECT_METHOD_ID:
            raise ValueError(f"method_id must equal {QGAPSELECT_METHOD_ID!r}")
        if self.baseline_method_id != GAP_AIDED_BASELINE_METHOD_ID:
            raise ValueError(
                f"baseline_method_id must equal {GAP_AIDED_BASELINE_METHOD_ID!r}"
            )
        if self.comparison_information_matched is not False:
            raise ValueError("comparison_information_matched must be false")
        if self.baseline_is_gap_aided is not True:
            raise ValueError("baseline_is_gap_aided must be true")
        if self.baseline_information != "k_and_public_gap_floor":
            raise ValueError("baseline_information must identify the public gap-floor input")
        if self.resource_conditioning != "all_fixture_pairs_unconditional":
            raise ValueError("coherent-query resources must include all fixture pairs")
        if self.claim_scope != QUANTUM_REFERENCE_ADAPTER_CLAIM_SCOPE:
            raise ValueError(
                "claim_scope must prohibit superiority and quantum-advantage claims"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "analysis": self.analysis.as_dict(),
            "bootstrap_seed": self.bootstrap_seed,
            "method_id": self.method_id,
            "baseline_method_id": self.baseline_method_id,
            "comparison_information_matched": self.comparison_information_matched,
            "baseline_is_gap_aided": self.baseline_is_gap_aided,
            "baseline_information": self.baseline_information,
            "resource_conditioning": self.resource_conditioning,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class FrozenQuantumPairedAdapterResult:
    """Family analyses and their across-family Holm FWER correction."""

    family_analyses: tuple[FrozenQuantumFamilyPairedAnalysis, ...]
    holm_fwer: HolmFWERCorrection
    master_seed: int
    bootstrap_repetitions: int
    confidence_level: float
    holm_alpha: float
    method_id: str = QGAPSELECT_METHOD_ID
    baseline_method_id: str = GAP_AIDED_BASELINE_METHOD_ID
    comparison_information_matched: bool = False
    baseline_is_gap_aided: bool = True
    baseline_information: str = "k_and_public_gap_floor"
    claim_scope: str = QUANTUM_REFERENCE_ADAPTER_CLAIM_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.family_analyses, tuple) or not self.family_analyses:
            raise TypeError("family_analyses must be a non-empty tuple")
        if any(
            not isinstance(item, FrozenQuantumFamilyPairedAnalysis)
            for item in self.family_analyses
        ):
            raise TypeError(
                "family_analyses must contain FrozenQuantumFamilyPairedAnalysis records"
            )
        family_ids = tuple(item.family_id for item in self.family_analyses)
        if len(set(family_ids)) != len(family_ids):
            raise ValueError("family_analyses must contain unique family IDs")
        if family_ids != tuple(sorted(family_ids)):
            raise ValueError("family_analyses must be sorted by family_id")
        if not isinstance(self.holm_fwer, HolmFWERCorrection):
            raise TypeError("holm_fwer must be HolmFWERCorrection")
        if tuple(item.hypothesis_id for item in self.holm_fwer.adjustments) != family_ids:
            raise ValueError("Holm hypothesis IDs must match the ordered family IDs")
        if isinstance(self.master_seed, bool) or not isinstance(self.master_seed, int):
            raise TypeError("master_seed must be an integer")
        repeats = _integer(
            self.bootstrap_repetitions,
            "bootstrap_repetitions",
            minimum=1,
        )
        confidence = _probability(
            self.confidence_level,
            "confidence_level",
            open_interval=True,
        )
        alpha = _probability(self.holm_alpha, "holm_alpha", open_interval=True)
        if self.holm_fwer.alpha != alpha:
            raise ValueError("holm_alpha must equal the nested Holm family alpha")
        if any(
            item.analysis.bootstrap.repetitions != repeats
            or item.analysis.bootstrap.confidence_level != confidence
            for item in self.family_analyses
        ):
            raise ValueError("family bootstrap settings must match the adapter settings")
        if self.method_id != QGAPSELECT_METHOD_ID:
            raise ValueError(f"method_id must equal {QGAPSELECT_METHOD_ID!r}")
        if self.baseline_method_id != GAP_AIDED_BASELINE_METHOD_ID:
            raise ValueError(
                f"baseline_method_id must equal {GAP_AIDED_BASELINE_METHOD_ID!r}"
            )
        if self.comparison_information_matched is not False:
            raise ValueError("comparison_information_matched must be false")
        if self.baseline_is_gap_aided is not True:
            raise ValueError("baseline_is_gap_aided must be true")
        if self.baseline_information != "k_and_public_gap_floor":
            raise ValueError("baseline_information must identify the public gap-floor input")
        if self.claim_scope != QUANTUM_REFERENCE_ADAPTER_CLAIM_SCOPE:
            raise ValueError(
                "claim_scope must prohibit superiority and quantum-advantage claims"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "family_analyses": [item.as_dict() for item in self.family_analyses],
            "holm_fwer": self.holm_fwer.as_dict(),
            "master_seed": self.master_seed,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "confidence_level": self.confidence_level,
            "holm_alpha": self.holm_alpha,
            "method_id": self.method_id,
            "baseline_method_id": self.baseline_method_id,
            "comparison_information_matched": self.comparison_information_matched,
            "baseline_is_gap_aided": self.baseline_is_gap_aided,
            "baseline_information": self.baseline_information,
            "claim_scope": self.claim_scope,
        }


def _exact_symmetric_binomial_lower_tail(trials: int, observed: int) -> float:
    """Return P[Binom(trials, 1/2) <= observed] using log-sum-exp.

    ``lgamma`` avoids construction of enormous binomial coefficients.  The
    recurrence accumulates the tail relative to its largest (last) term, which
    remains stable both at the endpoints and near the centre.
    """

    if observed < 0:
        return 0.0
    if observed >= trials:
        return 1.0
    log_last = (
        math.lgamma(trials + 1)
        - math.lgamma(observed + 1)
        - math.lgamma(trials - observed + 1)
        - trials * math.log(2.0)
    )
    relative_sum = 1.0
    relative_term = 1.0
    for index in range(observed, 0, -1):
        relative_term *= index / (trials - index + 1)
        relative_sum += relative_term
    log_probability = log_last + math.log(relative_sum)
    if log_probability < math.log(math.nextafter(0.0, 1.0)):
        return 0.0
    return min(1.0, math.exp(log_probability))


def exact_two_sided_mcnemar_p_value(
    method_only_success: int,
    baseline_only_success: int,
) -> float:
    """Return the conventional doubled exact binomial McNemar p-value."""

    method_only = _integer(method_only_success, "method_only_success")
    baseline_only = _integer(baseline_only_success, "baseline_only_success")
    discordant = method_only + baseline_only
    if discordant == 0:
        return 1.0
    lower_tail = _exact_symmetric_binomial_lower_tail(
        discordant, min(method_only, baseline_only)
    )
    return min(1.0, 2.0 * lower_tail)


def paired_binary_statistics(
    outcomes: Sequence[PairedFixtureOutcome],
) -> PairedBinaryStatistics:
    """Build the full paired table without discarding agreements."""

    pairs = _validated_outcomes(outcomes)
    both = sum(pair.method_success and pair.baseline_success for pair in pairs)
    method_only = sum(pair.method_success and not pair.baseline_success for pair in pairs)
    baseline_only = sum(pair.baseline_success and not pair.method_success for pair in pairs)
    neither = len(pairs) - both - method_only - baseline_only
    method_successes = both + method_only
    baseline_successes = both + baseline_only
    return PairedBinaryStatistics(
        pair_count=len(pairs),
        both_success=both,
        method_only_success=method_only,
        baseline_only_success=baseline_only,
        neither_success=neither,
        method_successes=method_successes,
        baseline_successes=baseline_successes,
        method_success_rate=method_successes / len(pairs),
        baseline_success_rate=baseline_successes / len(pairs),
        risk_difference=(method_successes - baseline_successes) / len(pairs),
        exact_two_sided_mcnemar_p_value=exact_two_sided_mcnemar_p_value(
            method_only, baseline_only
        ),
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def paired_bootstrap_risk_difference(
    outcomes: Sequence[PairedFixtureOutcome],
    *,
    repetitions: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> FixturePairedBootstrapInterval:
    """Bootstrap risk difference by resampling complete fixture pairs."""

    pairs = _validated_outcomes(outcomes)
    repeats = _integer(repetitions, "repetitions", minimum=1)
    confidence = _probability(
        confidence_level, "confidence_level", open_interval=True
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    differences = tuple(
        int(pair.method_success) - int(pair.baseline_success) for pair in pairs
    )
    point = math.fsum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrap_values = [
        math.fsum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(repeats)
    ]
    bootstrap_values.sort()
    tail = (1.0 - confidence) / 2.0
    return FixturePairedBootstrapInterval(
        pair_count=len(pairs),
        point_risk_difference=point,
        lower=_percentile(bootstrap_values, tail),
        upper=_percentile(bootstrap_values, 1.0 - tail),
        confidence_level=confidence,
        repetitions=repeats,
        seed=seed,
    )


def unconditional_paired_resource_statistics(
    outcomes: Sequence[PairedFixtureOutcome],
) -> UnconditionalPairedResourceStatistics:
    """Summarize paired resource differences over every supplied fixture."""

    pairs = _validated_outcomes(outcomes)
    if any(pair.method_resource is None for pair in pairs):
        raise ValueError("resources must be supplied for every fixture pair")
    method_resources = tuple(float(pair.method_resource) for pair in pairs)
    baseline_resources = tuple(float(pair.baseline_resource) for pair in pairs)
    differences = tuple(
        method - baseline
        for method, baseline in zip(method_resources, baseline_resources, strict=True)
    )
    return UnconditionalPairedResourceStatistics(
        pair_count=len(pairs),
        method_mean_resource=math.fsum(method_resources) / len(pairs),
        baseline_mean_resource=math.fsum(baseline_resources) / len(pairs),
        mean_paired_difference=math.fsum(differences) / len(pairs),
        min_paired_difference=min(differences),
        max_paired_difference=max(differences),
    )


def analyze_paired_fixtures(
    outcomes: Sequence[PairedFixtureOutcome],
    *,
    bootstrap_repetitions: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> FrozenPairedAnalysis:
    """Run the preregistered paired analysis on frozen fixture outcomes."""

    pairs = _validated_outcomes(outcomes)
    resource_presence = tuple(pair.method_resource is not None for pair in pairs)
    if any(resource_presence) and not all(resource_presence):
        raise ValueError("resources must be present for all fixture pairs or for none")
    return FrozenPairedAnalysis(
        binary=paired_binary_statistics(pairs),
        bootstrap=paired_bootstrap_risk_difference(
            pairs,
            repetitions=bootstrap_repetitions,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        ),
        resources=(
            unconditional_paired_resource_statistics(pairs)
            if all(resource_presence)
            else None
        ),
    )


def holm_fwer_adjusted_p_values(
    p_values: Mapping[str, float] | Sequence[float],
    *,
    alpha: float = 0.05,
) -> HolmFWERCorrection:
    """Return Holm adjusted p-values with strong family-wise error control."""

    family_alpha = _probability(alpha, "alpha", open_interval=True)
    if isinstance(p_values, Mapping):
        items = tuple(p_values.items())
        if any(not isinstance(key, str) or not key.strip() for key, _ in items):
            raise TypeError("mapping hypothesis identifiers must be non-empty strings")
    elif isinstance(p_values, Sequence) and not isinstance(p_values, (str, bytes)):
        items = tuple((f"h{index}", value) for index, value in enumerate(p_values))
    else:
        raise TypeError("p_values must be a mapping or a non-string sequence")
    if not items:
        raise ValueError("p_values must be non-empty")
    if len({key for key, _ in items}) != len(items):
        raise ValueError("hypothesis identifiers must be unique")
    probabilities = tuple(_probability(value, f"p_values[{key!r}]") for key, value in items)
    order = sorted(range(len(items)), key=lambda index: (probabilities[index], index))
    adjusted_by_index: dict[int, float] = {}
    rank_by_index: dict[int, int] = {}
    running_max = 0.0
    family_size = len(items)
    for zero_rank, index in enumerate(order):
        adjusted = min(1.0, (family_size - zero_rank) * probabilities[index])
        running_max = max(running_max, adjusted)
        adjusted_by_index[index] = running_max
        rank_by_index[index] = zero_rank + 1
    adjustments = tuple(
        HolmAdjustedPValue(
            hypothesis_id=key,
            raw_p_value=probabilities[index],
            adjusted_p_value=adjusted_by_index[index],
            rank=rank_by_index[index],
            family_size=family_size,
            rejected_at_alpha=adjusted_by_index[index] <= family_alpha,
            alpha=family_alpha,
        )
        for index, (key, _) in enumerate(items)
    )
    return HolmFWERCorrection(adjustments=adjustments, alpha=family_alpha)


def _validated_outcomes(
    outcomes: Sequence[PairedFixtureOutcome],
) -> tuple[PairedFixtureOutcome, ...]:
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise TypeError("outcomes must be a sequence of PairedFixtureOutcome records")
    pairs = tuple(outcomes)
    if not pairs:
        raise ValueError("outcomes must contain at least one fixture pair")
    if any(not isinstance(pair, PairedFixtureOutcome) for pair in pairs):
        raise TypeError("outcomes must contain only PairedFixtureOutcome records")
    return pairs


def _adapter_bootstrap_seed(master_seed: int, family_id: str) -> int:
    material = (
        f"qgapselect-frozen-paired-adapter-v1\0{master_seed}\0{family_id}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _validated_run_field(run: object, field_name: str) -> object:
    try:
        return getattr(run, field_name)
    except AttributeError as error:
        raise TypeError(
            f"run records must provide the FrozenQuantumReferenceRun-like field "
            f"{field_name!r}"
        ) from error


def analyze_frozen_quantum_reference_pairs(
    runs: Sequence[FrozenQuantumReferenceRunLike],
    *,
    master_seed: int = 0,
    bootstrap_repetitions: int = 10_000,
    confidence_level: float = 0.95,
    holm_alpha: float = 0.05,
) -> FrozenQuantumPairedAdapterResult:
    """Pair QGapSelect with the gap-aided independent-IAE reference.

    Panels are keyed by ``(family_id, instance_id, repetition)``.  Every panel
    must contain exactly one run for each target method.  Other unique method
    records may coexist in the input (for example, a known-threshold control),
    but they are not part of this comparison.

    The independent-IAE baseline receives a public gap-floor-derived precision,
    so ``comparison_information_matched`` is deliberately false.  Consequently
    neither a small p-value nor a resource difference is a superiority or
    quantum-advantage claim.
    """

    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence):
        raise TypeError("runs must be a sequence of FrozenQuantumReferenceRun-like records")
    records = tuple(runs)
    if not records:
        raise ValueError("runs must be non-empty")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    repeats = _integer(
        bootstrap_repetitions,
        "bootstrap_repetitions",
        minimum=1,
    )
    confidence = _probability(
        confidence_level,
        "confidence_level",
        open_interval=True,
    )
    alpha = _probability(holm_alpha, "holm_alpha", open_interval=True)

    panels: dict[tuple[str, str, int], dict[str, tuple[bool, int]]] = {}
    for run in records:
        family_id = _validated_run_field(run, "family_id")
        instance_id = _validated_run_field(run, "instance_id")
        repetition = _validated_run_field(run, "repetition")
        method_id = _validated_run_field(run, "method_id")
        success = _validated_run_field(run, "certified_exact_recovery")
        coherent_queries = _validated_run_field(run, "coherent_queries")
        if not isinstance(family_id, str) or not family_id.strip():
            raise TypeError("run family_id must be a non-empty string")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise TypeError("run instance_id must be a non-empty string")
        repetition = _integer(repetition, "run repetition")
        if not isinstance(method_id, str) or not method_id.strip():
            raise TypeError("run method_id must be a non-empty string")
        if not isinstance(success, bool):
            raise TypeError("run certified_exact_recovery must be bool")
        queries = _integer(coherent_queries, "run coherent_queries")

        panel_key = (family_id, instance_id, repetition)
        panel = panels.setdefault(panel_key, {})
        if method_id in panel:
            raise ValueError(
                "duplicate method record in panel "
                f"{panel_key!r}: method_id={method_id!r}"
            )
        panel[method_id] = (success, queries)

    required_methods = {QGAPSELECT_METHOD_ID, GAP_AIDED_BASELINE_METHOD_ID}
    for panel_key, panel in panels.items():
        missing = required_methods - set(panel)
        if missing:
            raise ValueError(
                f"incomplete comparison panel {panel_key!r}; missing {sorted(missing)!r}"
            )

    outcomes_by_family: dict[str, list[tuple[str, int, PairedFixtureOutcome]]] = {}
    for (family_id, instance_id, repetition), panel in panels.items():
        method_success, method_queries = panel[QGAPSELECT_METHOD_ID]
        baseline_success, baseline_queries = panel[GAP_AIDED_BASELINE_METHOD_ID]
        outcome = PairedFixtureOutcome(
            method_success=method_success,
            baseline_success=baseline_success,
            method_resource=method_queries,
            baseline_resource=baseline_queries,
        )
        outcomes_by_family.setdefault(family_id, []).append(
            (instance_id, repetition, outcome)
        )

    family_analyses: list[FrozenQuantumFamilyPairedAnalysis] = []
    for family_id in sorted(outcomes_by_family):
        ordered_rows = sorted(
            outcomes_by_family[family_id],
            key=lambda row: (row[0], row[1]),
        )
        family_seed = _adapter_bootstrap_seed(master_seed, family_id)
        analysis = analyze_paired_fixtures(
            tuple(row[2] for row in ordered_rows),
            bootstrap_repetitions=repeats,
            confidence_level=confidence,
            bootstrap_seed=family_seed,
        )
        family_analyses.append(
            FrozenQuantumFamilyPairedAnalysis(
                family_id=family_id,
                analysis=analysis,
                bootstrap_seed=family_seed,
            )
        )

    family_tuple = tuple(family_analyses)
    holm = holm_fwer_adjusted_p_values(
        {
            item.family_id: item.analysis.binary.exact_two_sided_mcnemar_p_value
            for item in family_tuple
        },
        alpha=alpha,
    )
    return FrozenQuantumPairedAdapterResult(
        family_analyses=family_tuple,
        holm_fwer=holm,
        master_seed=master_seed,
        bootstrap_repetitions=repeats,
        confidence_level=confidence,
        holm_alpha=alpha,
    )


__all__ = [
    "CLAIM_SCOPE",
    "GAP_AIDED_BASELINE_METHOD_ID",
    "QGAPSELECT_METHOD_ID",
    "QUANTUM_REFERENCE_ADAPTER_CLAIM_SCOPE",
    "FixturePairedBootstrapInterval",
    "FrozenPairedAnalysis",
    "FrozenQuantumFamilyPairedAnalysis",
    "FrozenQuantumPairedAdapterResult",
    "FrozenQuantumReferenceRunLike",
    "HolmAdjustedPValue",
    "HolmFWERCorrection",
    "PairedBinaryStatistics",
    "PairedFixtureOutcome",
    "UnconditionalPairedResourceStatistics",
    "analyze_paired_fixtures",
    "analyze_frozen_quantum_reference_pairs",
    "exact_two_sided_mcnemar_p_value",
    "holm_fwer_adjusted_p_values",
    "paired_binary_statistics",
    "paired_bootstrap_risk_difference",
    "unconditional_paired_resource_statistics",
]
