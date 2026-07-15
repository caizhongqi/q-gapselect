"""Executed threshold-discovery baselines for direct-QPE experiments.

Both baselines in this module obtain arm information only through public,
charged oracle operations.  ``IndependentQPEThresholdScan`` reruns the direct
QPE classifier independently on every examined arm.  ``ClassicalThresholdScan``
uses measured one-query reward experiments and simultaneous Hoeffding bounds.

These are fixed-budget discovery procedures, not absence tests.  Exhausting an
arm or query budget therefore never certifies that no qualifying arm exists.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .coherent import CanonicalRyStatevectorOracle
from .direct_phase import DirectAmplitudeThresholdFlag
from .oracles import QueryKind, QueryLedger, QuerySnapshot

CLAIM_STATUS = "executed_charged_threshold_baseline_no_complexity_claim"


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


def _optional_nonnegative_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, name)
    if result < 0:
        raise ValueError(f"{name} cannot be negative")
    return result


def _relation(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("relation must be a string")
    if value not in {"above", "below"}:
        raise ValueError("relation must be 'above' or 'below'")
    return value


def _excluded_indices(
    values: Sequence[int],
    *,
    n_arms: int,
) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise TypeError("excluded_indices must be a sequence of integers") from error
    result = tuple(_integer(index, "excluded index") for index in raw)
    if len(set(result)) != len(result):
        raise ValueError("excluded_indices must be unique")
    if any(not 0 <= index < n_arms for index in result):
        raise IndexError("an excluded index is outside the valid arm range")
    return result


def _immutable_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _empty_query_counts() -> dict[str, int]:
    counts = {kind.value: 0 for kind in QueryKind}
    counts.update(coherent_total=0, classical_total=0, total=0)
    return counts


def _query_delta(oracle: CanonicalRyStatevectorOracle, before: QuerySnapshot) -> dict[str, int]:
    return QueryLedger.difference(oracle.query_snapshot(), before)


def _ordered_eligible_indices(
    n_arms: int,
    excluded: tuple[int, ...],
    seed: int | None,
) -> tuple[int, ...]:
    eligible = np.asarray(
        [index for index in range(n_arms) if index not in excluded],
        dtype=np.int64,
    )
    if seed is not None and eligible.size > 1:
        eligible = np.random.default_rng(seed).permutation(eligible)
    return tuple(int(index) for index in eligible)


@dataclass(frozen=True, slots=True)
class ThresholdBaselineResources:
    """Measured resources shared by the independent-QPE and classical scans."""

    query_counts: Mapping[str, int]
    gate_counts: Mapping[str, int]
    arms_examined: int
    verifier_calls: int
    measurement_shots: int
    qpe_calls: int
    controlled_qpe_grover_iterations: int
    depth: int
    phase_qubits: int | None
    phase_bins: int | None
    access_mode: str
    claim_status: str = CLAIM_STATUS

    @property
    def coherent_oracle_queries(self) -> int:
        return int(self.query_counts.get("coherent_total", 0))

    @property
    def classical_oracle_queries(self) -> int:
        return int(self.query_counts.get("classical_total", 0))

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("total", 0))


@dataclass(frozen=True, slots=True)
class ThresholdBaselineArmRecord:
    """One actually executed per-arm classification."""

    ordinal: int
    index: int
    status: str
    accepted: bool
    successes: int
    shots: int
    estimate: float
    interval: tuple[float, float]
    decision_boundary: float
    failure_budget: float
    interval_domain: str
    query_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ThresholdBaselineResult:
    """Immutable output and audit trace of an executed threshold baseline."""

    method: str
    outputs: tuple[int, ...]
    expected_count: int
    relation: str
    threshold: float
    excluded_indices: tuple[int, ...]
    complete: bool
    verified: bool
    status: str
    failure_reason: str | None
    search_failure_budget: float
    per_arm_failure_budget: float
    scan_order: tuple[int, ...]
    trace: tuple[ThresholdBaselineArmRecord, ...]
    resources: ThresholdBaselineResources
    absence_certified: bool = False
    claim_status: str = CLAIM_STATUS

    @property
    def found_indices(self) -> tuple[int, ...]:
        return self.outputs

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return self.outputs


class IndependentQPEThresholdScan:
    """Scan arms with independent, freshly measured direct-QPE verifiers.

    The whole-search ``confidence`` budget is divided by the number of
    eligible arms before any measurements are taken.  Consequently a union
    bound covers every verifier that this scan can execute, including rejected
    and unresolved arms rather than only reported outputs.
    """

    def __init__(
        self,
        oracle: CanonicalRyStatevectorOracle,
        threshold: float,
        expected_count: int,
        *,
        phase_qubits: int = 5,
        relation: str = "above",
        excluded_indices: Sequence[int] = (),
        verification_shots: int = 128,
        confidence: float = 0.05,
        max_verifications: int | None = None,
        max_oracle_queries: int | None = None,
        seed: int | None = None,
    ) -> None:
        if not isinstance(oracle, CanonicalRyStatevectorOracle):
            raise TypeError("oracle must be a CanonicalRyStatevectorOracle")
        self.oracle = oracle
        self.threshold = _probability(threshold, "threshold", closed=True)
        self.expected_count = _integer(expected_count, "expected_count")
        self.phase_qubits = _integer(phase_qubits, "phase_qubits")
        self.relation = _relation(relation)
        self.verification_shots = _integer(verification_shots, "verification_shots")
        self.confidence = _probability(confidence, "confidence", closed=False)
        self.max_verifications = _optional_nonnegative_integer(
            max_verifications,
            "max_verifications",
        )
        self.max_oracle_queries = _optional_nonnegative_integer(
            max_oracle_queries,
            "max_oracle_queries",
        )
        if self.expected_count < 0:
            raise ValueError("expected_count cannot be negative")
        if self.phase_qubits <= 0:
            raise ValueError("phase_qubits must be positive")
        if self.phase_qubits > 12:
            raise ValueError("phase_qubits exceeds the small-state limit of 12")
        if self.verification_shots <= 0:
            raise ValueError("verification_shots must be positive")
        if seed is not None:
            seed = _integer(seed, "seed")
        self.seed = seed
        self.excluded_indices = _excluded_indices(
            excluded_indices,
            n_arms=oracle.n_arms,
        )

    @property
    def queries_per_verification(self) -> int:
        return self.verification_shots * (2 * (1 << self.phase_qubits) - 1)

    def run(self) -> ThresholdBaselineResult:
        before = self.oracle.query_snapshot()
        order = _ordered_eligible_indices(
            self.oracle.n_arms,
            self.excluded_indices,
            self.seed,
        )
        per_arm_failure = self.confidence / max(1, len(order))
        outputs: list[int] = []
        trace: list[ThresholdBaselineArmRecord] = []
        gate_counts: Counter[str] = Counter()
        qpe_calls = 0
        controlled_iterations = 0
        depth = 0
        rng = np.random.default_rng(self.seed)
        status: str | None = None
        failure_reason: str | None = None

        if self.expected_count > len(order):
            status = "target_exceeds_eligible_indices"
            failure_reason = "requested_target_exceeds_scan_domain"

        for index in order:
            if status is not None or len(outputs) == self.expected_count:
                break
            if self.max_verifications is not None and len(trace) >= self.max_verifications:
                status = "verification_budget_exhausted"
                failure_reason = "budget_exhaustion_does_not_certify_absence"
                break
            spent = _query_delta(self.oracle, before)["total"]
            if (
                self.max_oracle_queries is not None
                and spent + self.queries_per_verification > self.max_oracle_queries
            ):
                status = "query_budget_exhausted"
                failure_reason = "budget_exhaustion_does_not_certify_absence"
                break

            flag = DirectAmplitudeThresholdFlag(
                self.oracle,
                self.threshold,
                phase_qubits=self.phase_qubits,
                relation=self.relation,
                excluded_indices=self.excluded_indices,
            )
            arm_before = self.oracle.query_snapshot()
            verification_seed = int(rng.integers(0, 2**63, dtype=np.int64))
            verification = flag.verify_index(
                index,
                shots=self.verification_shots,
                confidence=per_arm_failure,
                seed=verification_seed,
                tag="independent_qpe_threshold_scan",
            )
            arm_queries = _query_delta(self.oracle, arm_before)
            resources = verification.resources
            gate_counts.update(resources.gate_counts)
            qpe_calls += resources.qpe_calls
            controlled_iterations += resources.controlled_grover_iterations
            depth += resources.depth
            accepted = verification.accepted
            if accepted:
                outputs.append(index)
            trace.append(
                ThresholdBaselineArmRecord(
                    ordinal=len(trace) + 1,
                    index=index,
                    status=verification.status,
                    accepted=accepted,
                    successes=verification.successes,
                    shots=verification.shots,
                    estimate=verification.estimate,
                    interval=verification.interval,
                    decision_boundary=verification.decision_cutoff,
                    failure_budget=per_arm_failure,
                    interval_domain="qpe_predicate_acceptance_probability",
                    query_counts=_immutable_counts(arm_queries),
                )
            )

        complete = len(outputs) == self.expected_count
        verified = complete and all(
            record.accepted for record in trace if record.index in outputs
        )
        if complete:
            status = "complete_fixed_confidence_qpe_predicate"
            failure_reason = None
        elif status is None:
            status = "scan_exhausted_without_target"
            failure_reason = "finite_scan_does_not_certify_absence"

        query_counts = _query_delta(self.oracle, before)
        resources = ThresholdBaselineResources(
            query_counts=_immutable_counts(query_counts),
            gate_counts=_immutable_counts(gate_counts),
            arms_examined=len(trace),
            verifier_calls=len(trace),
            measurement_shots=sum(record.shots for record in trace),
            qpe_calls=qpe_calls,
            controlled_qpe_grover_iterations=controlled_iterations,
            depth=depth,
            phase_qubits=self.phase_qubits,
            phase_bins=1 << self.phase_qubits,
            access_mode="coherent_controlled_qpe_from_canonical_rotation",
        )
        return ThresholdBaselineResult(
            method="independent_qpe_threshold_scan",
            outputs=tuple(outputs),
            expected_count=self.expected_count,
            relation=self.relation,
            threshold=self.threshold,
            excluded_indices=self.excluded_indices,
            complete=complete,
            verified=verified,
            status=status,
            failure_reason=failure_reason,
            search_failure_budget=self.confidence,
            per_arm_failure_budget=per_arm_failure,
            scan_order=order,
            trace=tuple(trace),
            resources=resources,
        )


class ClassicalThresholdScan:
    """Sequential fixed-shot scan with simultaneous reward-mean intervals."""

    def __init__(
        self,
        oracle: CanonicalRyStatevectorOracle,
        threshold: float,
        expected_count: int,
        *,
        relation: str = "above",
        excluded_indices: Sequence[int] = (),
        shots_per_arm: int = 128,
        confidence: float = 0.05,
        max_arms: int | None = None,
        max_oracle_queries: int | None = None,
        seed: int | None = None,
    ) -> None:
        if not isinstance(oracle, CanonicalRyStatevectorOracle):
            raise TypeError("oracle must be a CanonicalRyStatevectorOracle")
        self.oracle = oracle
        self.threshold = _probability(threshold, "threshold", closed=True)
        self.expected_count = _integer(expected_count, "expected_count")
        self.relation = _relation(relation)
        self.shots_per_arm = _integer(shots_per_arm, "shots_per_arm")
        self.confidence = _probability(confidence, "confidence", closed=False)
        self.max_arms = _optional_nonnegative_integer(max_arms, "max_arms")
        self.max_oracle_queries = _optional_nonnegative_integer(
            max_oracle_queries,
            "max_oracle_queries",
        )
        if self.expected_count < 0:
            raise ValueError("expected_count cannot be negative")
        if self.shots_per_arm <= 0:
            raise ValueError("shots_per_arm must be positive")
        if seed is not None:
            seed = _integer(seed, "seed")
        self.seed = seed
        self.excluded_indices = _excluded_indices(
            excluded_indices,
            n_arms=oracle.n_arms,
        )

    def run(self) -> ThresholdBaselineResult:
        before = self.oracle.query_snapshot()
        order = _ordered_eligible_indices(
            self.oracle.n_arms,
            self.excluded_indices,
            self.seed,
        )
        per_arm_failure = self.confidence / max(1, len(order))
        radius = math.sqrt(
            math.log(2.0 / per_arm_failure) / (2.0 * self.shots_per_arm)
        )
        outputs: list[int] = []
        trace: list[ThresholdBaselineArmRecord] = []
        status: str | None = None
        failure_reason: str | None = None

        if self.expected_count > len(order):
            status = "target_exceeds_eligible_indices"
            failure_reason = "requested_target_exceeds_scan_domain"

        for index in order:
            if status is not None or len(outputs) == self.expected_count:
                break
            if self.max_arms is not None and len(trace) >= self.max_arms:
                status = "arm_budget_exhausted"
                failure_reason = "budget_exhaustion_does_not_certify_absence"
                break
            spent = _query_delta(self.oracle, before)["total"]
            if (
                self.max_oracle_queries is not None
                and spent + self.shots_per_arm > self.max_oracle_queries
            ):
                status = "query_budget_exhausted"
                failure_reason = "budget_exhaustion_does_not_certify_absence"
                break

            arm_before = self.oracle.query_snapshot()
            successes = self.oracle.reward_experiment(
                index,
                self.shots_per_arm,
                tag="classical_threshold_scan",
            )
            arm_queries = _query_delta(self.oracle, arm_before)
            estimate = successes / self.shots_per_arm
            interval = (
                max(0.0, estimate - radius),
                min(1.0, estimate + radius),
            )
            if self.relation == "above":
                accepted = interval[0] > self.threshold
                rejected = interval[1] < self.threshold
            else:
                accepted = interval[1] < self.threshold
                rejected = interval[0] > self.threshold
            arm_status = "accepted" if accepted else "rejected" if rejected else "unresolved"
            if accepted:
                outputs.append(index)
            trace.append(
                ThresholdBaselineArmRecord(
                    ordinal=len(trace) + 1,
                    index=index,
                    status=arm_status,
                    accepted=accepted,
                    successes=successes,
                    shots=self.shots_per_arm,
                    estimate=estimate,
                    interval=interval,
                    decision_boundary=self.threshold,
                    failure_budget=per_arm_failure,
                    interval_domain="reward_mean",
                    query_counts=_immutable_counts(arm_queries),
                )
            )

        complete = len(outputs) == self.expected_count
        verified = complete and all(
            record.accepted for record in trace if record.index in outputs
        )
        if complete:
            status = "complete_simultaneous_hoeffding"
            failure_reason = None
        elif status is None:
            status = "scan_exhausted_without_target"
            failure_reason = "finite_scan_does_not_certify_absence"

        query_counts = _query_delta(self.oracle, before)
        measurement_shots = sum(record.shots for record in trace)
        resources = ThresholdBaselineResources(
            query_counts=_immutable_counts(query_counts),
            gate_counts=_immutable_counts(
                {
                    "reward_oracle_forward": measurement_shots,
                    "reward_measurement": measurement_shots,
                }
            ),
            arms_examined=len(trace),
            verifier_calls=0,
            measurement_shots=measurement_shots,
            qpe_calls=0,
            controlled_qpe_grover_iterations=0,
            depth=2 * measurement_shots,
            phase_qubits=None,
            phase_bins=None,
            access_mode="basis_state_forward_oracle_and_reward_measurement",
        )
        return ThresholdBaselineResult(
            method="classical_threshold_scan",
            outputs=tuple(outputs),
            expected_count=self.expected_count,
            relation=self.relation,
            threshold=self.threshold,
            excluded_indices=self.excluded_indices,
            complete=complete,
            verified=verified,
            status=status,
            failure_reason=failure_reason,
            search_failure_budget=self.confidence,
            per_arm_failure_budget=per_arm_failure,
            scan_order=order,
            trace=tuple(trace),
            resources=resources,
        )
