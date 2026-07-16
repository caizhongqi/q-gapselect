"""Same-interface, hard-cap composition controls for the S3 experiment.

The executable methods in this module deliberately accept one narrow runtime
interface: ``n``, ``k``, ``delta``, a canonical Bernoulli-oracle handle, and an
atomic logical-query cap.  They never accept a gap, threshold, fixture family,
ground truth, hidden partition, or stopping-time schedule.

All methods analytically sample the exact reward-measurement law of Grover
experiments.  This is a simulator of logical queries, not a QPE circuit, a
coherent variable-time search, a hardware implementation, or a faithful
reproduction of a cited literature algorithm.  Those boundaries are carried
in every result and the strong-composition registry remains fail closed.
"""

from __future__ import annotations

import math
import operator
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

from .oracles import QueryKind, QueryLedger, QuerySnapshot

INFORMATION_REGIME = "n_k_delta_canonical_oracle_atomic_query_cap_only"
BACKEND = "analytic_sampling_of_exact_grover_reward_measurement_law"
FIDELITY_STATUS = "executable_composition_control_not_full_literature_reproduction"
CLAIM_SCOPE = "same_interface_s3_control_no_hardware_theorem_or_advantage_claim"
OUTPUT_EXACT = "EXACT_TOP_K"
OUTPUT_INCONCLUSIVE = "INCONCLUSIVE"


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


def _immutable_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    counts = {str(key): int(value) for key, value in values.items()}
    if any(value < 0 for value in counts.values()):
        raise ValueError("query counts cannot be negative")
    return MappingProxyType(counts)


def _mask(indices: Sequence[int]) -> int:
    result = 0
    for arm in indices:
        result |= 1 << arm
    return result


@runtime_checkable
class CanonicalS3OracleProtocol(Protocol):
    """The complete algorithm-visible oracle capability for S3."""

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


class AtomicQueryCapExceeded(RuntimeError):
    """Raised before an oracle experiment that would exceed the public cap."""

    def __init__(self, *, cap: int, spent: int, requested: int) -> None:
        super().__init__(
            f"atomic query cap {cap} would be exceeded: spent={spent}, requested={requested}"
        )
        self.cap = cap
        self.spent = spent
        self.requested = requested


class _HardCapOracle:
    """Non-reading adapter that enforces the cap before every experiment."""

    def __init__(self, oracle: CanonicalS3OracleProtocol, cap: int) -> None:
        self._oracle = oracle
        self.cap = cap
        self._before = oracle.query_snapshot()

    @property
    def n_arms(self) -> int:
        return self._oracle.n_arms

    @property
    def spent(self) -> int:
        return QueryLedger.difference(self._oracle.query_snapshot(), self._before)["total"]

    def run_grover_experiment(
        self,
        arm: int,
        grover_power: int,
        shots: int,
        *,
        controlled: bool,
        tag: str,
    ) -> int:
        power = _integer(grover_power, "grover_power")
        shot_count = _integer(shots, "shots", minimum=1)
        requested = shot_count * (2 * power + 1)
        spent = self.spent
        if spent + requested > self.cap:
            raise AtomicQueryCapExceeded(
                cap=self.cap,
                spent=spent,
                requested=requested,
            )
        return self._oracle.run_grover_experiment(
            arm,
            power,
            shot_count,
            controlled=controlled,
            tag=tag,
        )


@dataclass(frozen=True, slots=True)
class S3ExecutionConfig:
    """Fixture-independent compile schedule shared by the S3 controls."""

    phase_powers: tuple[int, ...] = (0, 1, 3, 7, 15, 31, 63)
    fixed_shots_per_power: int = 32
    unknown_time_shots_per_level: int = 64
    grid_points: int = 4097

    def __post_init__(self) -> None:
        powers = tuple(_integer(value, "phase power") for value in self.phase_powers)
        if not powers or powers != tuple(sorted(set(powers))):
            raise ValueError("phase_powers must be nonempty, unique, and increasing")
        if any(((power + 1) & power) != 0 for power in powers):
            raise ValueError("each phase power must have the form 2**r - 1")
        object.__setattr__(self, "phase_powers", powers)
        _integer(self.fixed_shots_per_power, "fixed_shots_per_power", minimum=1)
        _integer(
            self.unknown_time_shots_per_level,
            "unknown_time_shots_per_level",
            minimum=1,
        )
        grid_points = _integer(self.grid_points, "grid_points", minimum=257)
        if grid_points % 2 == 0:
            raise ValueError("grid_points must be odd")


@dataclass(frozen=True, slots=True)
class _Observation:
    grover_power: int
    successes: int
    shots: int

    @property
    def frequency(self) -> int:
        return 2 * self.grover_power + 1


@dataclass(frozen=True, slots=True)
class _PhaseInterval:
    estimate: float
    lower: float
    upper: float
    numerical_warning: str | None


def _fit_phase_interval(
    observations: Sequence[_Observation],
    *,
    confidence: float,
    grid_points: int,
) -> _PhaseInterval:
    """Return a simultaneous-Hoeffding outer hull on the amplitude angle."""

    rows = tuple(observations)
    if not rows:
        return _PhaseInterval(0.5 * math.pi / 2.0, 0.0, math.pi / 2.0, "no data")
    alpha = _open_probability(confidence, "confidence")
    step = (math.pi / 2.0) / (grid_points - 1)
    grid = np.arange(grid_points, dtype=np.float64) * step
    frequencies = np.asarray([row.frequency for row in rows], dtype=np.float64)
    shots = np.asarray([row.shots for row in rows], dtype=np.float64)
    successes = np.asarray([row.successes for row in rows], dtype=np.float64)
    probabilities = np.sin(frequencies[:, np.newaxis] * grid[np.newaxis, :]) ** 2
    empirical = (successes / shots)[:, np.newaxis]
    radii = np.sqrt(np.log(2.0 * len(rows) / alpha) / (2.0 * shots))[:, np.newaxis]
    padding = (frequencies * step / 2.0)[:, np.newaxis]
    consistent = np.all(np.abs(probabilities - empirical) <= radii + padding, axis=0)
    indices = np.flatnonzero(consistent)
    if indices.size == 0:
        return _PhaseInterval(
            0.5 * math.pi / 2.0,
            0.0,
            math.pi / 2.0,
            "empty numerical confidence set; returned vacuous interval",
        )

    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    likelihood = np.sum(
        successes[:, np.newaxis] * np.log(clipped)
        + (shots - successes)[:, np.newaxis] * np.log1p(-clipped),
        axis=0,
    )
    mle_index = int(indices[int(np.argmax(likelihood[indices]))])
    lower = max(0.0, float(grid[int(indices[0])]) - step / 2.0)
    upper = min(math.pi / 2.0, float(grid[int(indices[-1])]) + step / 2.0)
    return _PhaseInterval(float(grid[mle_index]), lower, upper, None)


@dataclass(frozen=True, slots=True)
class S3StageRecord:
    """One public-schedule stage and its exact ledger delta."""

    stage_id: str
    phase_powers: tuple[int, ...]
    active_count_before: int
    completed_arm_count: int
    newly_accepted: tuple[int, ...]
    newly_rejected: tuple[int, ...]
    status: str
    query_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "phase_powers": list(self.phase_powers),
            "active_count_before": self.active_count_before,
            "completed_arm_count": self.completed_arm_count,
            "newly_accepted": list(self.newly_accepted),
            "newly_rejected": list(self.newly_rejected),
            "status": self.status,
            "query_counts": dict(self.query_counts),
        }


@dataclass(frozen=True, slots=True)
class S3BaselineResult:
    """Fail-closed result under the canonical five-field runtime interface."""

    method_id: str
    output_relation: str
    output_indices: tuple[int, ...] | None
    output_mask: int | None
    certified: bool
    status: str
    n: int
    k: int
    delta: float
    atomic_query_cap: int
    exact_canonical_query_count: int
    query_counts: Mapping[str, int]
    per_arm_query_counts: Mapping[int, int]
    hard_cap_respected: bool
    stages: tuple[S3StageRecord, ...]
    variable_time_l2_query_proxy: float | None
    information_regime: str = INFORMATION_REGIME
    fidelity_status: str = FIDELITY_STATUS
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE
    qpe_circuit_executed: bool = False
    coherent_variable_time_search_executed: bool = False
    official_literature_reproduction: bool = False
    registry_coverage_activated: bool = False
    hardware_claimable: bool = False
    quantum_advantage_claimable: bool = False
    attempt_must_remain_in_denominator: bool = True

    @property
    def inconclusive(self) -> bool:
        return self.output_relation == OUTPUT_INCONCLUSIVE

    @property
    def budget_valid(self) -> bool:
        return self.hard_cap_respected and (
            self.exact_canonical_query_count <= self.atomic_query_cap
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "output_relation": self.output_relation,
            "output_indices": (None if self.output_indices is None else list(self.output_indices)),
            "output_mask": self.output_mask,
            "certified": self.certified,
            "status": self.status,
            "n": self.n,
            "k": self.k,
            "delta": self.delta,
            "atomic_query_cap": self.atomic_query_cap,
            "exact_canonical_query_count": self.exact_canonical_query_count,
            "query_counts": dict(self.query_counts),
            "per_arm_query_counts": {
                str(arm): count for arm, count in self.per_arm_query_counts.items()
            },
            "hard_cap_respected": self.hard_cap_respected,
            "budget_valid": self.budget_valid,
            "stages": [stage.as_dict() for stage in self.stages],
            "variable_time_l2_query_proxy": self.variable_time_l2_query_proxy,
            "information_regime": self.information_regime,
            "fidelity_status": self.fidelity_status,
            "backend": self.backend,
            "claim_scope": self.claim_scope,
            "qpe_circuit_executed": self.qpe_circuit_executed,
            "coherent_variable_time_search_executed": (self.coherent_variable_time_search_executed),
            "official_literature_reproduction": self.official_literature_reproduction,
            "registry_coverage_activated": self.registry_coverage_activated,
            "hardware_claimable": self.hardware_claimable,
            "quantum_advantage_claimable": self.quantum_advantage_claimable,
            "attempt_must_remain_in_denominator": (self.attempt_must_remain_in_denominator),
        }


@dataclass(frozen=True, slots=True)
class S3AttemptScore:
    """Trusted-harness score; truth is never retained as an algorithm input."""

    method_id: str
    atomic_query_cap: int
    strict_instance: bool
    included_in_all_attempt_denominator: bool
    all_attempt_success: bool
    certified_exact_success: bool
    fail_closed_success: bool
    inconclusive: bool
    incorrect_certificate: bool
    budget_valid: bool
    exact_canonical_query_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "atomic_query_cap": self.atomic_query_cap,
            "strict_instance": self.strict_instance,
            "included_in_all_attempt_denominator": (self.included_in_all_attempt_denominator),
            "all_attempt_success": self.all_attempt_success,
            "certified_exact_success": self.certified_exact_success,
            "fail_closed_success": self.fail_closed_success,
            "inconclusive": self.inconclusive,
            "incorrect_certificate": self.incorrect_certificate,
            "budget_valid": self.budget_valid,
            "exact_canonical_query_count": self.exact_canonical_query_count,
        }


@dataclass(frozen=True, slots=True)
class S3AttemptAggregate:
    """Unfiltered aggregate: every launched attempt is in the denominator."""

    method_id: str
    atomic_query_cap: int
    all_attempt_count: int
    all_attempt_success_count: int
    all_attempt_success_rate: float
    certified_exact_count: int
    certified_exact_rate_all_attempts: float
    fail_closed_success_count: int
    inconclusive_count: int
    inconclusive_rate_all_attempts: float
    incorrect_certificate_count: int
    budget_violation_count: int
    mean_exact_canonical_queries: float
    median_exact_canonical_queries: float

    def as_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "method_id",
                "atomic_query_cap",
                "all_attempt_count",
                "all_attempt_success_count",
                "all_attempt_success_rate",
                "certified_exact_count",
                "certified_exact_rate_all_attempts",
                "fail_closed_success_count",
                "inconclusive_count",
                "inconclusive_rate_all_attempts",
                "incorrect_certificate_count",
                "budget_violation_count",
                "mean_exact_canonical_queries",
                "median_exact_canonical_queries",
            )
        }


def _validate_runtime(
    *,
    n: int,
    k: int,
    delta: float,
    oracle: CanonicalS3OracleProtocol,
    atomic_query_cap: int,
) -> tuple[int, int, float, int, QuerySnapshot, _HardCapOracle]:
    if not isinstance(oracle, CanonicalS3OracleProtocol):
        raise TypeError("oracle must implement CanonicalS3OracleProtocol")
    arm_count = _integer(n, "n", minimum=1)
    if _integer(oracle.n_arms, "oracle.n_arms", minimum=1) != arm_count:
        raise ValueError("n must exactly equal oracle.n_arms")
    top_k = _integer(k, "k", minimum=1)
    if top_k > arm_count:
        raise ValueError("k cannot exceed n")
    failure = _open_probability(delta, "delta")
    cap = _integer(atomic_query_cap, "atomic_query_cap")
    before = oracle.query_snapshot()
    if before.total != 0:
        raise ValueError("S3 methods require a fresh zero-ledger oracle")
    return arm_count, top_k, failure, cap, before, _HardCapOracle(oracle, cap)


def _stage_counts(oracle: CanonicalS3OracleProtocol, before: QuerySnapshot) -> Mapping[str, int]:
    return _immutable_counts(QueryLedger.difference(oracle.query_snapshot(), before))


def _build_result(
    *,
    method_id: str,
    oracle: CanonicalS3OracleProtocol,
    before: QuerySnapshot,
    n: int,
    k: int,
    delta: float,
    cap: int,
    certified_indices: Sequence[int] | None,
    status: str,
    stages: Sequence[S3StageRecord],
    variable_time_proxy: bool,
) -> S3BaselineResult:
    counts = _immutable_counts(QueryLedger.difference(oracle.query_snapshot(), before))
    per_arm: dict[int, int] = {}
    after = oracle.query_snapshot()
    for arm in range(n):
        after_counts = after.by_arm.get(arm, {})
        before_counts = before.by_arm.get(arm, {})
        per_arm[arm] = sum(
            int(after_counts.get(kind.value, 0) - before_counts.get(kind.value, 0))
            for kind in QueryKind
        )
    total = int(counts["total"])
    if total != sum(per_arm.values()):
        raise RuntimeError("aggregate and per-arm canonical ledgers disagree")
    if total != sum(int(stage.query_counts["total"]) for stage in stages):
        raise RuntimeError("stage and aggregate canonical ledgers disagree")
    if total > cap:
        raise RuntimeError("atomic query cap was violated")

    if certified_indices is None:
        output_indices = None
        output_mask = None
        output_relation = OUTPUT_INCONCLUSIVE
        certified = False
    else:
        output_indices = tuple(sorted(_integer(arm, "output arm") for arm in certified_indices))
        if len(output_indices) != k or len(set(output_indices)) != k:
            raise RuntimeError("a certified output must contain exactly k unique arms")
        output_mask = _mask(output_indices)
        output_relation = OUTPUT_EXACT
        certified = True

    proxy = (
        math.sqrt(sum(value * value for value in per_arm.values())) if variable_time_proxy else None
    )
    return S3BaselineResult(
        method_id=method_id,
        output_relation=output_relation,
        output_indices=output_indices,
        output_mask=output_mask,
        certified=certified,
        status=status,
        n=n,
        k=k,
        delta=delta,
        atomic_query_cap=cap,
        exact_canonical_query_count=total,
        query_counts=counts,
        per_arm_query_counts=MappingProxyType(per_arm),
        hard_cap_respected=True,
        stages=tuple(stages),
        variable_time_l2_query_proxy=proxy,
    )


class S3BaselineProtocol(Protocol):
    method_id: str

    def run(
        self,
        *,
        n: int,
        k: int,
        delta: float,
        oracle: CanonicalS3OracleProtocol,
        atomic_query_cap: int,
    ) -> S3BaselineResult: ...


class FixedPrecisionGlobalTopKBAI:
    """One fixed phase schedule followed by a global Top-k interval test.

    This control avoids the unnecessary requirement that arms tied *inside*
    the optimal set be ordered.  It is an executable global BAI-like interval
    procedure, not a reproduction of coherent QBAI or approximate k-minimum.
    """

    method_id = "fixed_precision_global_topk_bai"

    def __init__(self, config: S3ExecutionConfig | None = None) -> None:
        self.config = config if config is not None else S3ExecutionConfig()

    def run(
        self,
        *,
        n: int,
        k: int,
        delta: float,
        oracle: CanonicalS3OracleProtocol,
        atomic_query_cap: int,
    ) -> S3BaselineResult:
        n, k, delta, cap, before, budgeted = _validate_runtime(
            n=n,
            k=k,
            delta=delta,
            oracle=oracle,
            atomic_query_cap=atomic_query_cap,
        )
        if k == n:
            return _build_result(
                method_id=self.method_id,
                oracle=oracle,
                before=before,
                n=n,
                k=k,
                delta=delta,
                cap=cap,
                certified_indices=tuple(range(n)),
                status="CERTIFIED_BY_CARDINALITY",
                stages=(),
                variable_time_proxy=False,
            )

        stage_before = oracle.query_snapshot()
        estimates: dict[int, _PhaseInterval] = {}
        completed = 0
        try:
            for arm in range(n):
                observations: list[_Observation] = []
                for power in self.config.phase_powers:
                    successes = budgeted.run_grover_experiment(
                        arm,
                        power,
                        self.config.fixed_shots_per_power,
                        controlled=True,
                        tag=f"{self.method_id}:arm_{arm}:power_{power}",
                    )
                    observations.append(
                        _Observation(
                            grover_power=power,
                            successes=successes,
                            shots=self.config.fixed_shots_per_power,
                        )
                    )
                estimates[arm] = _fit_phase_interval(
                    observations,
                    confidence=delta / n,
                    grid_points=self.config.grid_points,
                )
                completed += 1
        except AtomicQueryCapExceeded:
            stage = S3StageRecord(
                stage_id="global_topk",
                phase_powers=self.config.phase_powers,
                active_count_before=n,
                completed_arm_count=completed,
                newly_accepted=(),
                newly_rejected=(),
                status="INCONCLUSIVE_QUERY_CAP",
                query_counts=_stage_counts(oracle, stage_before),
            )
            return _build_result(
                method_id=self.method_id,
                oracle=oracle,
                before=before,
                n=n,
                k=k,
                delta=delta,
                cap=cap,
                certified_indices=None,
                status="INCONCLUSIVE_QUERY_CAP",
                stages=(stage,),
                variable_time_proxy=False,
            )

        ranking = tuple(sorted(range(n), key=lambda arm: (-estimates[arm].estimate, arm)))
        selected = ranking[:k]
        outside = ranking[k:]
        selected_lower = min(estimates[arm].lower for arm in selected)
        outside_upper = max(estimates[arm].upper for arm in outside)
        certified = selected_lower > outside_upper
        status = (
            "CERTIFIED_EXACT_TOP_K"
            if certified
            else "INCONCLUSIVE_FIXED_PRECISION_BOUNDARY_OVERLAP"
        )
        stage = S3StageRecord(
            stage_id="global_topk",
            phase_powers=self.config.phase_powers,
            active_count_before=n,
            completed_arm_count=completed,
            newly_accepted=tuple(sorted(selected)) if certified else (),
            newly_rejected=tuple(sorted(outside)) if certified else (),
            status=status,
            query_counts=_stage_counts(oracle, stage_before),
        )
        return _build_result(
            method_id=self.method_id,
            oracle=oracle,
            before=before,
            n=n,
            k=k,
            delta=delta,
            cap=cap,
            certified_indices=selected if certified else None,
            status=status,
            stages=(stage,),
            variable_time_proxy=False,
        )


class RepeatedFixedPrecisionPhaseBAI:
    """Repeated one-best-arm composition using one fixed phase schedule.

    The controlled Grover experiments are sampled independently.  Consequently
    this is a legal fixed-precision phase-interrogation/BAI control, not an
    executed coherent QPE register and not the Wang et al. QBAI algorithm.
    """

    method_id = "repeated_fixed_precision_phase_bai"

    def __init__(self, config: S3ExecutionConfig | None = None) -> None:
        self.config = config if config is not None else S3ExecutionConfig()

    def run(
        self,
        *,
        n: int,
        k: int,
        delta: float,
        oracle: CanonicalS3OracleProtocol,
        atomic_query_cap: int,
    ) -> S3BaselineResult:
        n, k, delta, cap, before, budgeted = _validate_runtime(
            n=n,
            k=k,
            delta=delta,
            oracle=oracle,
            atomic_query_cap=atomic_query_cap,
        )
        remaining = list(range(n))
        accepted: list[int] = []
        stages: list[S3StageRecord] = []
        local_confidence = delta / (n * k)

        for output_position in range(k):
            stage_before = oracle.query_snapshot()
            active_before = len(remaining)
            if len(remaining) == 1:
                winner = remaining[0]
                accepted.append(winner)
                remaining.remove(winner)
                stages.append(
                    S3StageRecord(
                        stage_id=f"winner_{output_position}",
                        phase_powers=(),
                        active_count_before=1,
                        completed_arm_count=0,
                        newly_accepted=(winner,),
                        newly_rejected=(),
                        status="CERTIFIED_BY_CARDINALITY",
                        query_counts=_stage_counts(oracle, stage_before),
                    )
                )
                continue

            estimates: dict[int, _PhaseInterval] = {}
            completed = 0
            try:
                for arm in remaining:
                    observations: list[_Observation] = []
                    for power in self.config.phase_powers:
                        successes = budgeted.run_grover_experiment(
                            arm,
                            power,
                            self.config.fixed_shots_per_power,
                            controlled=True,
                            tag=(
                                f"{self.method_id}:winner_{output_position}:arm_{arm}:power_{power}"
                            ),
                        )
                        observations.append(
                            _Observation(
                                grover_power=power,
                                successes=successes,
                                shots=self.config.fixed_shots_per_power,
                            )
                        )
                    estimates[arm] = _fit_phase_interval(
                        observations,
                        confidence=local_confidence,
                        grid_points=self.config.grid_points,
                    )
                    completed += 1
            except AtomicQueryCapExceeded:
                stages.append(
                    S3StageRecord(
                        stage_id=f"winner_{output_position}",
                        phase_powers=self.config.phase_powers,
                        active_count_before=active_before,
                        completed_arm_count=completed,
                        newly_accepted=(),
                        newly_rejected=(),
                        status="INCONCLUSIVE_QUERY_CAP",
                        query_counts=_stage_counts(oracle, stage_before),
                    )
                )
                return _build_result(
                    method_id=self.method_id,
                    oracle=oracle,
                    before=before,
                    n=n,
                    k=k,
                    delta=delta,
                    cap=cap,
                    certified_indices=None,
                    status="INCONCLUSIVE_QUERY_CAP",
                    stages=stages,
                    variable_time_proxy=False,
                )

            winner = max(remaining, key=lambda arm: (estimates[arm].estimate, -arm))
            other_upper = max(estimates[arm].upper for arm in remaining if arm != winner)
            separated = estimates[winner].lower > other_upper
            if not separated:
                stages.append(
                    S3StageRecord(
                        stage_id=f"winner_{output_position}",
                        phase_powers=self.config.phase_powers,
                        active_count_before=active_before,
                        completed_arm_count=completed,
                        newly_accepted=(),
                        newly_rejected=(),
                        status="INCONCLUSIVE_FIXED_PRECISION_OVERLAP",
                        query_counts=_stage_counts(oracle, stage_before),
                    )
                )
                return _build_result(
                    method_id=self.method_id,
                    oracle=oracle,
                    before=before,
                    n=n,
                    k=k,
                    delta=delta,
                    cap=cap,
                    certified_indices=None,
                    status="INCONCLUSIVE_FIXED_PRECISION_OVERLAP",
                    stages=stages,
                    variable_time_proxy=False,
                )

            accepted.append(winner)
            remaining.remove(winner)
            stages.append(
                S3StageRecord(
                    stage_id=f"winner_{output_position}",
                    phase_powers=self.config.phase_powers,
                    active_count_before=active_before,
                    completed_arm_count=completed,
                    newly_accepted=(winner,),
                    newly_rejected=(),
                    status="CERTIFIED_ONE_BEST_ARM",
                    query_counts=_stage_counts(oracle, stage_before),
                )
            )

        return _build_result(
            method_id=self.method_id,
            oracle=oracle,
            before=before,
            n=n,
            k=k,
            delta=delta,
            cap=cap,
            certified_indices=accepted,
            status="CERTIFIED_EXACT_TOP_K",
            stages=stages,
            variable_time_proxy=False,
        )


def _classify_active(
    intervals: Mapping[int, _PhaseInterval], remaining_quota: int
) -> tuple[set[int], set[int]]:
    arms = tuple(intervals)
    accepted: set[int] = set()
    rejected: set[int] = set()
    for arm in arms:
        interval = intervals[arm]
        possible_above = sum(
            intervals[other].upper >= interval.lower for other in arms if other != arm
        )
        if possible_above < remaining_quota:
            accepted.add(arm)
            continue
        certainly_above = sum(
            intervals[other].lower > interval.upper for other in arms if other != arm
        )
        if certainly_above >= remaining_quota:
            rejected.add(arm)
    return accepted, rejected


class PublicCapUnknownTimeSearchComposition:
    """Public geometric schedule with data-dependent arm stopping.

    The implementation serially executes every active arm at each public
    level.  Its L2 field is a post-run composition proxy only.  No coherent
    unknown-time search or variable-time amplitude amplification is executed.
    """

    method_id = "public_cap_unknown_time_search_composition"

    def __init__(self, config: S3ExecutionConfig | None = None) -> None:
        self.config = config if config is not None else S3ExecutionConfig()

    def run(
        self,
        *,
        n: int,
        k: int,
        delta: float,
        oracle: CanonicalS3OracleProtocol,
        atomic_query_cap: int,
    ) -> S3BaselineResult:
        n, k, delta, cap, before, budgeted = _validate_runtime(
            n=n,
            k=k,
            delta=delta,
            oracle=oracle,
            atomic_query_cap=atomic_query_cap,
        )
        if k == n:
            return _build_result(
                method_id=self.method_id,
                oracle=oracle,
                before=before,
                n=n,
                k=k,
                delta=delta,
                cap=cap,
                certified_indices=tuple(range(n)),
                status="CERTIFIED_BY_CARDINALITY",
                stages=(),
                variable_time_proxy=True,
            )

        active = set(range(n))
        accepted: set[int] = set()
        observations: dict[int, list[_Observation]] = {arm: [] for arm in range(n)}
        stages: list[S3StageRecord] = []
        local_confidence = delta / (n * len(self.config.phase_powers))

        for level, power in enumerate(self.config.phase_powers):
            stage_before = oracle.query_snapshot()
            active_before = tuple(sorted(active))
            completed = 0
            try:
                for arm in active_before:
                    successes = budgeted.run_grover_experiment(
                        arm,
                        power,
                        self.config.unknown_time_shots_per_level,
                        controlled=False,
                        tag=f"{self.method_id}:level_{level}:arm_{arm}:power_{power}",
                    )
                    observations[arm].append(
                        _Observation(
                            grover_power=power,
                            successes=successes,
                            shots=self.config.unknown_time_shots_per_level,
                        )
                    )
                    completed += 1
            except AtomicQueryCapExceeded:
                stages.append(
                    S3StageRecord(
                        stage_id=f"level_{level}",
                        phase_powers=(power,),
                        active_count_before=len(active_before),
                        completed_arm_count=completed,
                        newly_accepted=(),
                        newly_rejected=(),
                        status="INCONCLUSIVE_QUERY_CAP",
                        query_counts=_stage_counts(oracle, stage_before),
                    )
                )
                return _build_result(
                    method_id=self.method_id,
                    oracle=oracle,
                    before=before,
                    n=n,
                    k=k,
                    delta=delta,
                    cap=cap,
                    certified_indices=None,
                    status="INCONCLUSIVE_QUERY_CAP",
                    stages=stages,
                    variable_time_proxy=True,
                )

            intervals = {
                arm: _fit_phase_interval(
                    observations[arm],
                    confidence=local_confidence,
                    grid_points=self.config.grid_points,
                )
                for arm in active_before
            }
            remaining_quota = k - len(accepted)
            newly_accepted, newly_rejected = _classify_active(intervals, remaining_quota)
            newly_rejected.difference_update(newly_accepted)
            accepted.update(newly_accepted)
            active.difference_update(newly_accepted | newly_rejected)
            stages.append(
                S3StageRecord(
                    stage_id=f"level_{level}",
                    phase_powers=(power,),
                    active_count_before=len(active_before),
                    completed_arm_count=completed,
                    newly_accepted=tuple(sorted(newly_accepted)),
                    newly_rejected=tuple(sorted(newly_rejected)),
                    status="PUBLIC_LEVEL_COMPLETE",
                    query_counts=_stage_counts(oracle, stage_before),
                )
            )

            remaining_quota = k - len(accepted)
            if remaining_quota == 0:
                return _build_result(
                    method_id=self.method_id,
                    oracle=oracle,
                    before=before,
                    n=n,
                    k=k,
                    delta=delta,
                    cap=cap,
                    certified_indices=tuple(accepted),
                    status="CERTIFIED_EXACT_TOP_K",
                    stages=stages,
                    variable_time_proxy=True,
                )
            if remaining_quota == len(active):
                accepted.update(active)
                return _build_result(
                    method_id=self.method_id,
                    oracle=oracle,
                    before=before,
                    n=n,
                    k=k,
                    delta=delta,
                    cap=cap,
                    certified_indices=tuple(accepted),
                    status="CERTIFIED_BY_REMAINING_CARDINALITY",
                    stages=stages,
                    variable_time_proxy=True,
                )
            if remaining_quota < 0 or remaining_quota > len(active):
                return _build_result(
                    method_id=self.method_id,
                    oracle=oracle,
                    before=before,
                    n=n,
                    k=k,
                    delta=delta,
                    cap=cap,
                    certified_indices=None,
                    status="INCONCLUSIVE_INTERNAL_CLASSIFICATION_CONFLICT",
                    stages=stages,
                    variable_time_proxy=True,
                )

        return _build_result(
            method_id=self.method_id,
            oracle=oracle,
            before=before,
            n=n,
            k=k,
            delta=delta,
            cap=cap,
            certified_indices=None,
            status="INCONCLUSIVE_PUBLIC_PRECISION_SCHEDULE_EXHAUSTED",
            stages=stages,
            variable_time_proxy=True,
        )


def score_s3_attempt(
    result: S3BaselineResult,
    expected_top_k: Sequence[int] | None,
) -> S3AttemptScore:
    """Score after execution; ``expected_top_k`` belongs to the trusted harness."""

    if not isinstance(result, S3BaselineResult):
        raise TypeError("result must be an S3BaselineResult")
    if expected_top_k is None:
        truth: tuple[int, ...] | None = None
    else:
        truth = tuple(sorted(_integer(arm, "expected arm") for arm in expected_top_k))
        if len(truth) != result.k or len(set(truth)) != result.k:
            raise ValueError("expected_top_k must contain exactly k unique arms")
        if any(arm >= result.n for arm in truth):
            raise ValueError("expected_top_k contains an arm outside the oracle")

    strict = truth is not None
    certified_exact = strict and result.certified and result.output_indices == truth
    fail_closed = (not strict) and result.inconclusive
    incorrect_certificate = result.certified and (not strict or result.output_indices != truth)
    return S3AttemptScore(
        method_id=result.method_id,
        atomic_query_cap=result.atomic_query_cap,
        strict_instance=strict,
        included_in_all_attempt_denominator=True,
        all_attempt_success=certified_exact or fail_closed,
        certified_exact_success=certified_exact,
        fail_closed_success=fail_closed,
        inconclusive=result.inconclusive,
        incorrect_certificate=incorrect_certificate,
        budget_valid=result.budget_valid,
        exact_canonical_query_count=result.exact_canonical_query_count,
    )


def aggregate_s3_attempts(scores: Sequence[S3AttemptScore]) -> S3AttemptAggregate:
    """Aggregate every score without success-conditioned row filtering."""

    rows = tuple(scores)
    if not rows:
        raise ValueError("scores cannot be empty")
    if any(not isinstance(row, S3AttemptScore) for row in rows):
        raise TypeError("scores must contain S3AttemptScore values")
    if any(not row.included_in_all_attempt_denominator for row in rows):
        raise ValueError("every S3 score must remain in the all-attempt denominator")
    method_ids = {row.method_id for row in rows}
    caps = {row.atomic_query_cap for row in rows}
    if len(method_ids) != 1 or len(caps) != 1:
        raise ValueError("aggregate rows must share method_id and atomic_query_cap")
    attempts = len(rows)
    successes = sum(row.all_attempt_success for row in rows)
    exact = sum(row.certified_exact_success for row in rows)
    inconclusive = sum(row.inconclusive for row in rows)
    queries = tuple(row.exact_canonical_query_count for row in rows)
    return S3AttemptAggregate(
        method_id=rows[0].method_id,
        atomic_query_cap=rows[0].atomic_query_cap,
        all_attempt_count=attempts,
        all_attempt_success_count=successes,
        all_attempt_success_rate=successes / attempts,
        certified_exact_count=exact,
        certified_exact_rate_all_attempts=exact / attempts,
        fail_closed_success_count=sum(row.fail_closed_success for row in rows),
        inconclusive_count=inconclusive,
        inconclusive_rate_all_attempts=inconclusive / attempts,
        incorrect_certificate_count=sum(row.incorrect_certificate for row in rows),
        budget_violation_count=sum(not row.budget_valid for row in rows),
        mean_exact_canonical_queries=statistics.fmean(queries),
        median_exact_canonical_queries=float(statistics.median(queries)),
    )


__all__ = [
    "BACKEND",
    "CLAIM_SCOPE",
    "FIDELITY_STATUS",
    "INFORMATION_REGIME",
    "OUTPUT_EXACT",
    "OUTPUT_INCONCLUSIVE",
    "AtomicQueryCapExceeded",
    "CanonicalS3OracleProtocol",
    "FixedPrecisionGlobalTopKBAI",
    "PublicCapUnknownTimeSearchComposition",
    "RepeatedFixedPrecisionPhaseBAI",
    "S3AttemptAggregate",
    "S3AttemptScore",
    "S3BaselineProtocol",
    "S3BaselineResult",
    "S3ExecutionConfig",
    "S3StageRecord",
    "aggregate_s3_attempts",
    "score_s3_attempt",
]
