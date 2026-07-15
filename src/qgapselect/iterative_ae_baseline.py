"""Executed analytic iterative-amplitude-estimation threshold baseline.

The direct QPE implementation must be compared with a phase-register-free
amplitude-estimation procedure.  This module scans arms independently using
``AnalyticIterativeAmplitudeEstimator`` and classifies only when its angular
confidence interval lies strictly on one side of the public threshold.

The backend samples the exact measurement law of each Grover experiment and
charges logical ``A``/``A_dagger`` calls.  It does not allocate a circuit or a
statevector, so its gate, qubit, wall-time, and hardware costs are deliberately
not mixed with the direct-QPE ledger.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from .estimators import AnalyticIterativeAmplitudeEstimator
from .models import IAEConfig
from .oracles import CanonicalBernoulliOracleSimulator, QueryLedger

BACKEND = "analytic_iterative_ae_measurement_law"
CLAIM_STATUS = "executed_analytic_iae_baseline_no_hardware_or_advantage_claim"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _probability(value: object, name: str, *, closed: bool) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    valid = 0.0 <= result <= 1.0 if closed else 0.0 < result < 1.0
    if not math.isfinite(result) or not valid:
        interval = "[0, 1]" if closed else "(0, 1)"
        raise ValueError(f"{name} must be finite and lie in {interval}")
    return result


def _relation(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("relation must be a string")
    if value not in {"above", "below"}:
        raise ValueError("relation must be 'above' or 'below'")
    return value


def _immutable(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


@dataclass(frozen=True, slots=True)
class IterativeAEArmRecord:
    """One independent analytic-IAE angular classification."""

    ordinal: int
    index: int
    status: str
    accepted: bool
    estimate: float
    mean_interval: tuple[float, float]
    angular_interval: tuple[float, float]
    rounds: int
    measurement_shots: int
    query_counts: Mapping[str, int]
    numerical_warning: str | None


@dataclass(frozen=True, slots=True)
class IterativeAEResources:
    """Logical-query resources for the analytic measurement-law backend."""

    query_counts: Mapping[str, int]
    arms_examined: int
    grover_experiments: int
    measurement_shots: int
    maximum_grover_power: int
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("total", 0))


@dataclass(frozen=True, slots=True)
class IterativeAEThresholdResult:
    """Independent threshold-scan result with no absence certification."""

    method: str
    outputs: tuple[int, ...]
    expected_count: int
    relation: str
    threshold: float
    threshold_angle: float
    complete: bool
    verified: bool
    status: str
    failure_reason: str | None
    confidence: float
    per_arm_confidence: float
    trace: tuple[IterativeAEArmRecord, ...]
    resources: IterativeAEResources
    absence_certified: bool = False
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS

    @property
    def found_indices(self) -> tuple[int, ...]:
        return self.outputs


class IterativeAEThresholdScan:
    """Scan arms with phase-register-free analytic IAE confidence intervals."""

    def __init__(
        self,
        oracle: CanonicalBernoulliOracleSimulator,
        threshold: float,
        expected_count: int,
        *,
        relation: str = "above",
        confidence: float = 0.05,
        target_angular_precision: float = 0.02,
        config: IAEConfig | None = None,
        seed: int | None = None,
    ) -> None:
        if not isinstance(oracle, CanonicalBernoulliOracleSimulator):
            raise TypeError("oracle must be a CanonicalBernoulliOracleSimulator")
        self.oracle = oracle
        self.threshold = _probability(threshold, "threshold", closed=True)
        self.expected_count = _integer(expected_count, "expected_count")
        self.relation = _relation(relation)
        self.confidence = _probability(confidence, "confidence", closed=False)
        if isinstance(target_angular_precision, bool):
            raise TypeError("target_angular_precision must be a real number, not bool")
        self.target_angular_precision = float(target_angular_precision)
        if (
            not math.isfinite(self.target_angular_precision)
            or not 0.0 < self.target_angular_precision < math.pi / 2.0
        ):
            raise ValueError("target_angular_precision must lie in (0, pi/2)")
        if not 0 <= self.expected_count <= oracle.n_arms:
            raise ValueError("expected_count must lie in [0, number of arms]")
        if config is None:
            config = IAEConfig()
        if not isinstance(config, IAEConfig):
            raise TypeError("config must be an IAEConfig")
        self.config = replace(
            config,
            target_angular_precision=self.target_angular_precision,
        )
        if seed is not None:
            seed = _integer(seed, "seed")
        self.seed = seed

    def run(self) -> IterativeAEThresholdResult:
        before = self.oracle.query_snapshot()
        per_arm_confidence = self.confidence / max(1, self.oracle.n_arms)
        order = np.arange(self.oracle.n_arms, dtype=np.int64)
        if self.seed is not None and self.oracle.n_arms > 1:
            order = np.random.default_rng(self.seed).permutation(order)
        threshold_angle = math.asin(math.sqrt(self.threshold))
        estimator = AnalyticIterativeAmplitudeEstimator(self.config)
        outputs: list[int] = []
        trace: list[IterativeAEArmRecord] = []

        for raw_index in order:
            if len(outputs) == self.expected_count:
                break
            index = int(raw_index)
            arm_before = self.oracle.query_snapshot()
            estimate = estimator.estimate(
                self.oracle,
                index,
                confidence=per_arm_confidence,
                target_angular_precision=self.target_angular_precision,
                tag="iterative_ae_threshold_scan",
            )
            angular = estimate.angular_interval
            if self.relation == "above":
                accepted = angular.lower > threshold_angle
                rejected = angular.upper < threshold_angle
            else:
                accepted = angular.upper < threshold_angle
                rejected = angular.lower > threshold_angle
            status = "accepted" if accepted else "rejected" if rejected else "unresolved"
            if accepted:
                outputs.append(index)
            delta = QueryLedger.difference(
                self.oracle.query_snapshot(),
                arm_before,
            )
            trace.append(
                IterativeAEArmRecord(
                    ordinal=len(trace) + 1,
                    index=index,
                    status=status,
                    accepted=accepted,
                    estimate=estimate.estimate,
                    mean_interval=(estimate.interval.lower, estimate.interval.upper),
                    angular_interval=(angular.lower, angular.upper),
                    rounds=len(estimate.observations),
                    measurement_shots=sum(
                        observation.shots for observation in estimate.observations
                    ),
                    query_counts=_immutable(delta),
                    numerical_warning=estimate.numerical_warning,
                )
            )

        complete = len(outputs) == self.expected_count
        if complete:
            status = "complete_analytic_iae_angular_intervals"
            failure_reason = None
        else:
            status = "scan_exhausted_without_target"
            failure_reason = "finite_scan_does_not_certify_absence"
        query_counts = QueryLedger.difference(self.oracle.query_snapshot(), before)
        # All per-round shot counts are identical under IAEConfig; retain the
        # actual total from records and the configured maximum power separately.
        resources = IterativeAEResources(
            query_counts=_immutable(query_counts),
            arms_examined=len(trace),
            grover_experiments=sum(record.rounds for record in trace),
            measurement_shots=sum(record.measurement_shots for record in trace),
            maximum_grover_power=self.config.max_grover_power,
        )
        return IterativeAEThresholdResult(
            method="analytic_iterative_ae_threshold_scan",
            outputs=tuple(outputs),
            expected_count=self.expected_count,
            relation=self.relation,
            threshold=self.threshold,
            threshold_angle=threshold_angle,
            complete=complete,
            verified=complete and all(
                record.accepted for record in trace if record.index in outputs
            ),
            status=status,
            failure_reason=failure_reason,
            confidence=self.confidence,
            per_arm_confidence=per_arm_confidence,
            trace=tuple(trace),
            resources=resources,
        )
