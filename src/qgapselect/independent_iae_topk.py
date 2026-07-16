"""Independent analytic-IAE Top-k reference over a frozen coherent oracle.

Every arm is estimated independently.  The returned empirical ranking is
always available for diagnostic plots, but it is certified only under strict
simultaneous interval separation.  This module analytically samples Grover
measurement laws; it makes no circuit, hardware, or quantum-advantage claim.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .estimators import AnalyticIterativeAmplitudeEstimator
from .models import IAEConfig
from .oracles import QueryLedger, QuerySnapshot

CLAIM_SCOPE = "analytic_independent_iae_topk_reference_no_hardware_claim"
BACKEND = "analytic_iterative_amplitude_estimation_measurement_law"


@runtime_checkable
class IndependentIAEOracleProtocol(Protocol):
    """Exact algorithm-side capability used by this reference baseline."""

    @property
    def n_arms(self) -> int: ...

    def query_snapshot(self) -> QuerySnapshot: ...

    def run_grover_experiment(
        self,
        arm: int,
        grover_power: int,
        shots: int,
        *,
        controlled: bool = False,
        tag: str | None = None,
    ) -> int: ...


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


def _open_probability(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _angular_precision(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("target_angular_precision must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("target_angular_precision must be a real number") from error
    if not math.isfinite(result) or not 0.0 < result < math.pi / 2.0:
        raise ValueError("target_angular_precision must lie in (0, pi/2)")
    return result


def _immutable_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


@dataclass(frozen=True, slots=True)
class IndependentIAEArmRecord:
    """One independent arm estimate and its actually charged resources."""

    arm: int
    estimate: float
    interval: tuple[float, float]
    angular_interval: tuple[float, float]
    target_precision_met: bool
    estimator_calls: int
    grover_experiments: int
    measurement_shots: int
    query_counts: Mapping[str, int]
    numerical_warning: str | None

    @property
    def oracle_calls(self) -> int:
        return int(self.query_counts.get("total", 0))


@dataclass(frozen=True, slots=True)
class IndependentIAETopKResult:
    """Ranking, strict interval certificate, and complete logical-query audit."""

    selected: tuple[int, ...]
    ranking: tuple[int, ...]
    estimates: Mapping[int, float]
    intervals: Mapping[int, tuple[float, float]]
    angular_intervals: Mapping[int, tuple[float, float]]
    certified: bool
    status: str
    unresolved_reason: str | None
    k: int
    confidence: float
    per_arm_confidence: float
    target_angular_precision: float
    certificate_boundary: tuple[float, float] | None
    trace: tuple[IndependentIAEArmRecord, ...]
    query_counts: Mapping[str, int]
    per_arm_calls: Mapping[int, int]
    per_arm_estimator_calls: Mapping[int, int]
    per_arm_query_counts: Mapping[int, Mapping[str, int]]
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE
    hardware_claimable: bool = False

    @property
    def outputs(self) -> tuple[int, ...]:
        return self.selected

    @property
    def heuristic_output_only(self) -> bool:
        return not self.certified

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("total", 0))

    def as_dict(self) -> dict[str, object]:
        return {
            "selected": list(self.selected),
            "ranking": list(self.ranking),
            "estimates": {str(key): value for key, value in self.estimates.items()},
            "intervals": {str(key): list(value) for key, value in self.intervals.items()},
            "angular_intervals": {
                str(key): list(value) for key, value in self.angular_intervals.items()
            },
            "certified": self.certified,
            "status": self.status,
            "unresolved_reason": self.unresolved_reason,
            "k": self.k,
            "confidence": self.confidence,
            "per_arm_confidence": self.per_arm_confidence,
            "target_angular_precision": self.target_angular_precision,
            "certificate_boundary": (
                list(self.certificate_boundary) if self.certificate_boundary is not None else None
            ),
            "query_counts": dict(self.query_counts),
            "per_arm_calls": {str(key): value for key, value in self.per_arm_calls.items()},
            "per_arm_estimator_calls": {
                str(key): value for key, value in self.per_arm_estimator_calls.items()
            },
            "per_arm_query_counts": {
                str(key): dict(value) for key, value in self.per_arm_query_counts.items()
            },
            "backend": self.backend,
            "claim_scope": self.claim_scope,
            "hardware_claimable": self.hardware_claimable,
        }


class IndependentIAETopKReference:
    """Run one independent analytic IAE instance for every arm."""

    def __init__(
        self,
        oracle: IndependentIAEOracleProtocol,
        k: int,
        *,
        config: IAEConfig,
        confidence: float,
        target_angular_precision: float,
    ) -> None:
        if not isinstance(oracle, IndependentIAEOracleProtocol):
            raise TypeError("oracle must expose n_arms, query_snapshot, and run_grover_experiment")
        if not isinstance(config, IAEConfig):
            raise TypeError("config must be an IAEConfig")
        n_arms = _integer(oracle.n_arms, "oracle.n_arms", minimum=1)
        top_k = _integer(k, "k", minimum=1)
        if top_k > n_arms:
            raise ValueError("k cannot exceed oracle.n_arms")
        self.oracle = oracle
        self.n_arms = n_arms
        self.k = top_k
        self.config = config
        self.confidence = _open_probability(confidence, "confidence")
        self.target_angular_precision = _angular_precision(target_angular_precision)

    def run(self) -> IndependentIAETopKResult:
        before = self.oracle.query_snapshot()
        per_arm_confidence = self.confidence / self.n_arms
        estimator = AnalyticIterativeAmplitudeEstimator(self.config)
        trace: list[IndependentIAEArmRecord] = []

        for arm in range(self.n_arms):
            estimate = estimator.estimate(
                self.oracle,
                arm,
                confidence=per_arm_confidence,
                target_angular_precision=self.target_angular_precision,
                tag=f"independent_iae_topk_arm_{arm}",
            )
            angular = estimate.angular_interval
            trace.append(
                IndependentIAEArmRecord(
                    arm=arm,
                    estimate=estimate.estimate,
                    interval=(estimate.interval.lower, estimate.interval.upper),
                    angular_interval=(angular.lower, angular.upper),
                    target_precision_met=(angular.width <= 2.0 * self.target_angular_precision),
                    estimator_calls=1,
                    grover_experiments=len(estimate.observations),
                    measurement_shots=sum(
                        observation.shots for observation in estimate.observations
                    ),
                    query_counts=_immutable_counts(estimate.executed_query_counts),
                    numerical_warning=estimate.numerical_warning,
                )
            )

        ranking = tuple(
            sorted(
                range(self.n_arms),
                key=lambda arm: (-trace[arm].estimate, arm),
            )
        )
        selected = ranking[: self.k]
        outside = ranking[self.k :]
        if outside:
            minimum_selected_lower = min(trace[arm].interval[0] for arm in selected)
            maximum_outside_upper = max(trace[arm].interval[1] for arm in outside)
            certified = minimum_selected_lower > maximum_outside_upper
            certificate_boundary: tuple[float, float] | None = (
                minimum_selected_lower,
                maximum_outside_upper,
            )
        else:
            # Selecting every arm is vacuously exact; no boundary competitor
            # exists, although estimates are still executed for comparability.
            certified = True
            certificate_boundary = None

        if certified and outside:
            status = "certified_strict_interval_separation"
            unresolved_reason = None
        elif certified:
            status = "certified_all_arms_selected"
            unresolved_reason = None
        else:
            status = "unresolved_heuristic_ranking_only"
            unresolved_reason = (
                "selected lower intervals do not all strictly exceed the "
                "maximum outside upper interval"
            )

        query_counts = _immutable_counts(
            QueryLedger.difference(self.oracle.query_snapshot(), before)
        )
        per_arm_query_counts = MappingProxyType(
            {record.arm: record.query_counts for record in trace}
        )
        per_arm_calls = MappingProxyType({record.arm: record.oracle_calls for record in trace})
        per_arm_estimator_calls = MappingProxyType(
            {record.arm: record.estimator_calls for record in trace}
        )
        if sum(per_arm_calls.values()) != int(query_counts.get("total", 0)):
            raise RuntimeError("per-arm and total IAE query ledgers differ")

        return IndependentIAETopKResult(
            selected=selected,
            ranking=ranking,
            estimates=MappingProxyType({record.arm: record.estimate for record in trace}),
            intervals=MappingProxyType({record.arm: record.interval for record in trace}),
            angular_intervals=MappingProxyType(
                {record.arm: record.angular_interval for record in trace}
            ),
            certified=certified,
            status=status,
            unresolved_reason=unresolved_reason,
            k=self.k,
            confidence=self.confidence,
            per_arm_confidence=per_arm_confidence,
            target_angular_precision=self.target_angular_precision,
            certificate_boundary=certificate_boundary,
            trace=tuple(trace),
            query_counts=query_counts,
            per_arm_calls=per_arm_calls,
            per_arm_estimator_calls=per_arm_estimator_calls,
            per_arm_query_counts=per_arm_query_counts,
        )


def run_independent_iae_topk(
    oracle: IndependentIAEOracleProtocol,
    k: int,
    *,
    config: IAEConfig,
    confidence: float,
    target_angular_precision: float,
) -> IndependentIAETopKResult:
    """Functional entry point for the independent analytic-IAE reference."""

    return IndependentIAETopKReference(
        oracle,
        k,
        config=config,
        confidence=confidence,
        target_angular_precision=target_angular_precision,
    ).run()


__all__ = [
    "BACKEND",
    "CLAIM_SCOPE",
    "IndependentIAEArmRecord",
    "IndependentIAEOracleProtocol",
    "IndependentIAETopKReference",
    "IndependentIAETopKResult",
    "run_independent_iae_topk",
]
