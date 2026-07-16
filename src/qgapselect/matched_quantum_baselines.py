"""Information- and query-cap-matched Layer-C Top-k references.

This module provides executable *reference implementations* for falsification
experiments.  Every method consumes the same narrow Layer-C capability
(``n_arms``, ``query_snapshot`` and ``run_grover_experiment``), the public
cardinality ``k``, one total failure budget and one hard logical-query cap.
The cap is enforced before an oracle experiment is issued, so an overshooting
method cannot borrow queries and later be truncated in post-processing.

The design is informed by, but is **not** an official reproduction of:

* Wang, You, Li and Childs, "Quantum exploration algorithms for multi-armed
  bandits", AAAI 2021, https://arxiv.org/abs/2007.07049 (fixed-confidence
  quantum BAI and variable-time composition);
* Ambainis, "Variable time amplitude amplification ...",
  https://arxiv.org/abs/1010.4458 (known stopping-time VTAA);
* Suzuki et al., "Amplitude estimation without phase estimation",
  https://arxiv.org/abs/1904.10246 (iterative/likelihood AE motivation); and
* Gao, Ji and Wang, "Quantum Approximate k-Minimum Finding",
  https://arxiv.org/abs/2412.16586 (direct multi-output comparison target).

In particular, the code below executes independent analytic IAE measurements.
``known_time`` and ``unknown_time`` expose an RMS *composition proxy* computed
from actually charged per-arm queries; they do not execute coherent
variable-time amplitude amplification.  ``coarse_partition_bai`` is a valid
same-output-relation composition of the local interval procedure, not the
Wang--You--Li--Childs algorithm.  These boundaries are attached to every
result so a benchmark cannot silently relabel a stand-in as a paper baseline.
"""

from __future__ import annotations

import hashlib
import math
import operator
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .estimators import AnalyticIterativeAmplitudeEstimator
from .models import IAEConfig
from .oracles import QueryKind, QueryLedger, QuerySnapshot

CLAIM_SCOPE = "layer_c_matched_reference_no_hardware_or_advantage_claim"
BACKEND = "independent_analytic_iae_with_strict_query_cap"
REFERENCE_STATUS = "paper_informed_stand_in_not_official_reproduction"
REFERENCE_SOURCES: Mapping[str, str] = MappingProxyType(
    {
        "quantum_bai": "https://arxiv.org/abs/2007.07049",
        "variable_time_amplitude_amplification": "https://arxiv.org/abs/1010.4458",
        "amplitude_estimation_without_qpe": "https://arxiv.org/abs/1904.10246",
        "approximate_k_minimum": "https://arxiv.org/abs/2412.16586",
    }
)
METHOD_ASSUMPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "oracle": "blind canonical Layer-C Bernoulli amplitude access",
        "confidence": "simultaneous interval union bound over all possible arm-level calls",
        "query_cost": "one A or A_dagger use is one logical query",
        "budget": "the next full Grover experiment is rejected before cap overshoot",
        "variable_time": "reported RMS value is a composition proxy, not executed VTAA",
        "paper_fidelity": "all methods are paper-informed stand-ins, not official reproductions",
    }
)


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
    result = MappingProxyType({str(key): int(value) for key, value in values.items()})
    if any(value < 0 for value in result.values()):
        raise ValueError("query counts cannot be negative")
    return result


def _empty_counts() -> Mapping[str, int]:
    values = {kind.value: 0 for kind in QueryKind}
    values.update(coherent_total=0, classical_total=0, total=0)
    return _immutable_counts(values)


@runtime_checkable
class LayerCOracleProtocol(Protocol):
    """Only the blind algorithm-side capability shared by all methods."""

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


class QueryCapExceeded(RuntimeError):
    """Raised before the first Layer-C experiment that would exceed the cap."""

    def __init__(self, *, cap: int, spent: int, requested: int) -> None:
        super().__init__(
            f"query cap {cap} would be exceeded: spent={spent}, requested={requested}"
        )
        self.cap = cap
        self.spent = spent
        self.requested = requested


class _BudgetedOracle:
    """Non-reading proxy that atomically enforces a total logical-query cap."""

    def __init__(self, oracle: LayerCOracleProtocol, query_cap: int) -> None:
        if not isinstance(oracle, LayerCOracleProtocol):
            raise TypeError("oracle must implement the Layer-C oracle protocol")
        self._oracle = oracle
        self._query_cap = _integer(query_cap, "query_cap")
        self._before = oracle.query_snapshot()

    @property
    def n_arms(self) -> int:
        return self._oracle.n_arms

    def query_snapshot(self) -> QuerySnapshot:
        return self._oracle.query_snapshot()

    @property
    def spent(self) -> int:
        return QueryLedger.difference(self.query_snapshot(), self._before)["total"]

    def run_grover_experiment(
        self,
        arm: int,
        grover_power: int,
        shots: int,
        *,
        controlled: bool = False,
        tag: str | None = None,
    ) -> int:
        power = _integer(grover_power, "grover_power")
        shot_count = _integer(shots, "shots", minimum=1)
        requested = shot_count * (2 * power + 1)
        spent = self.spent
        if spent + requested > self._query_cap:
            raise QueryCapExceeded(
                cap=self._query_cap,
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


class _SubsetOracle:
    """Map a local subproblem to global arms without creating a second ledger."""

    def __init__(self, oracle: _BudgetedOracle, arms: Sequence[int]) -> None:
        self._oracle = oracle
        self._arms = tuple(_integer(arm, "arm") for arm in arms)
        if not self._arms:
            raise ValueError("a subproblem must contain at least one arm")
        if len(set(self._arms)) != len(self._arms):
            raise ValueError("subproblem arms must be unique")
        if any(not 0 <= arm < oracle.n_arms for arm in self._arms):
            raise IndexError("a subproblem arm is outside the oracle domain")

    @property
    def n_arms(self) -> int:
        return len(self._arms)

    def query_snapshot(self) -> QuerySnapshot:
        return self._oracle.query_snapshot()

    def run_grover_experiment(
        self,
        arm: int,
        grover_power: int,
        shots: int,
        *,
        controlled: bool = False,
        tag: str | None = None,
    ) -> int:
        local = _integer(arm, "arm")
        if not 0 <= local < len(self._arms):
            raise IndexError("local arm is outside the subproblem")
        return self._oracle.run_grover_experiment(
            self._arms[local],
            grover_power,
            shots,
            controlled=controlled,
            tag=tag,
        )


@dataclass(frozen=True, slots=True)
class MatchedBaselineConfig:
    """Common public schedule; no gap, threshold, or latent mean is accepted."""

    initial_angular_precision: float = math.pi / 16.0
    precision_decay: float = 0.5
    max_levels: int = 8
    iae: IAEConfig = field(
        default_factory=lambda: IAEConfig(
            target_angular_precision=math.pi / 64.0,
            confidence=0.05,
            shots_per_round=64,
            max_rounds=6,
            max_grover_power=31,
            grid_points=2049,
        )
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.initial_angular_precision < math.pi / 2.0:
            raise ValueError("initial_angular_precision must lie in (0, pi/2)")
        if not 0.0 < self.precision_decay < 1.0:
            raise ValueError("precision_decay must lie in (0, 1)")
        _integer(self.max_levels, "max_levels", minimum=1)
        if not isinstance(self.iae, IAEConfig):
            raise TypeError("iae must be an IAEConfig")


@dataclass(frozen=True, slots=True)
class MatchedBaselineStage:
    """One auditable subproblem in a baseline composition."""

    stage_id: str
    arms: tuple[int, ...]
    requested_k: int
    selected: tuple[int, ...]
    certified: bool
    status: str
    levels_executed: int
    query_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class MatchedBaselineResult:
    """Strict relation certificate and the complete fresh-ledger audit."""

    method_id: str
    information_regime: str
    selected: tuple[int, ...]
    certified: bool
    status: str
    timeout: bool
    k: int
    failure_budget: float
    query_cap: int
    query_counts: Mapping[str, int]
    per_arm_queries: Mapping[int, int]
    stages: tuple[MatchedBaselineStage, ...]
    variable_time_rms_proxy: float | None
    serial_query_total: int
    backend: str = BACKEND
    reference_status: str = REFERENCE_STATUS
    claim_scope: str = CLAIM_SCOPE
    official_reproduction: bool = False
    hardware_claimable: bool = False

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("total", 0))

    @property
    def budget_valid(self) -> bool:
        return self.oracle_queries <= self.query_cap

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "information_regime": self.information_regime,
            "selected": list(self.selected),
            "certified": self.certified,
            "status": self.status,
            "timeout": self.timeout,
            "k": self.k,
            "failure_budget": self.failure_budget,
            "query_cap": self.query_cap,
            "query_counts": dict(self.query_counts),
            "per_arm_queries": {
                str(arm): count for arm, count in self.per_arm_queries.items()
            },
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "arms": list(stage.arms),
                    "requested_k": stage.requested_k,
                    "selected": list(stage.selected),
                    "certified": stage.certified,
                    "status": stage.status,
                    "levels_executed": stage.levels_executed,
                    "query_counts": dict(stage.query_counts),
                }
                for stage in self.stages
            ],
            "variable_time_rms_proxy": self.variable_time_rms_proxy,
            "serial_query_total": self.serial_query_total,
            "backend": self.backend,
            "reference_status": self.reference_status,
            "claim_scope": self.claim_scope,
            "official_reproduction": self.official_reproduction,
            "hardware_claimable": self.hardware_claimable,
        }


@dataclass(frozen=True, slots=True)
class FixedCapEvaluation:
    """Post-run scoring; every trial remains in the denominator."""

    method_id: str
    query_cap: int
    expected_top_k: tuple[int, ...]
    result: MatchedBaselineResult
    output_exact: bool
    certified_exact: bool
    budget_valid: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "query_cap": self.query_cap,
            "expected_top_k": list(self.expected_top_k),
            "result": self.result.as_dict(),
            "output_exact": self.output_exact,
            "certified_exact": self.certified_exact,
            "budget_valid": self.budget_valid,
        }


@dataclass(frozen=True, slots=True)
class FixedCapAggregate:
    """All-attempt aggregate, deliberately without both-success filtering."""

    method_id: str
    query_cap: int
    attempts: int
    certified_exact_count: int
    certified_exact_rate: float
    timeout_count: int
    uncertified_count: int
    budget_violation_count: int
    mean_queries: float
    median_queries: float

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "query_cap": self.query_cap,
            "attempts": self.attempts,
            "certified_exact_count": self.certified_exact_count,
            "certified_exact_rate": self.certified_exact_rate,
            "timeout_count": self.timeout_count,
            "uncertified_count": self.uncertified_count,
            "budget_violation_count": self.budget_violation_count,
            "mean_queries": self.mean_queries,
            "median_queries": self.median_queries,
        }


@dataclass(frozen=True, slots=True)
class _AdaptiveOutcome:
    selected: tuple[int, ...]
    certified: bool
    status: str
    timeout: bool
    levels_executed: int


def _classify(
    intervals: Mapping[int, tuple[float, float]], quota: int
) -> tuple[set[int], set[int]]:
    arms = tuple(intervals)
    accepted: set[int] = set()
    rejected: set[int] = set()
    for arm in arms:
        lower, upper = intervals[arm]
        possible_above = sum(
            intervals[other][1] >= lower for other in arms if other != arm
        )
        if possible_above < quota:
            accepted.add(arm)
            continue
        certainly_above = sum(
            intervals[other][0] > upper for other in arms if other != arm
        )
        if certainly_above >= quota:
            rejected.add(arm)
    return accepted, rejected


def _adaptive_topk(
    oracle: _SubsetOracle,
    k: int,
    *,
    config: MatchedBaselineConfig,
    failure_budget: float,
    tag: str,
    arm_level_limits: Sequence[int] | None = None,
) -> _AdaptiveOutcome:
    n = oracle.n_arms
    quota_total = _integer(k, "k", minimum=1)
    if quota_total > n:
        raise ValueError("k cannot exceed the subproblem size")
    known_stopping_schedule = arm_level_limits is not None
    if known_stopping_schedule:
        assert arm_level_limits is not None
        limits = tuple(_integer(value, "arm level limit", minimum=1) for value in arm_level_limits)
        if len(limits) != n:
            raise ValueError("arm_level_limits must align with the subproblem")
    else:
        limits = (config.max_levels,) * n

    accepted: set[int] = set()
    active: set[int] = set(range(n))
    if quota_total == n:
        return _AdaptiveOutcome(tuple(range(n)), True, "certified_by_cardinality", False, 0)

    levels_executed = 0
    try:
        for level in range(1, config.max_levels + 1):
            remaining_quota = quota_total - len(accepted)
            if remaining_quota == 0:
                return _AdaptiveOutcome(
                    tuple(sorted(accepted)), True, "certified_strict_interval_separation", False,
                    levels_executed,
                )
            if remaining_quota == len(active):
                accepted.update(active)
                return _AdaptiveOutcome(
                    tuple(sorted(accepted)), True, "certified_by_remaining_cardinality", False,
                    levels_executed,
                )

            expired = (
                {arm for arm in active if limits[arm] < level}
                if known_stopping_schedule
                else set()
            )
            if expired:
                return _AdaptiveOutcome(
                    tuple(sorted(accepted)), False, "known_stopping_schedule_exhausted", False,
                    levels_executed,
                )

            precision = config.initial_angular_precision * (
                config.precision_decay ** (level - 1)
            )
            # Sum_{r>=1} 6/(pi^2 r^2) = 1; the extra n factor covers every
            # arm at every level without assuming which arms eliminate early.
            local_failure = (
                failure_budget * 6.0 / (math.pi**2 * level**2 * n)
            )
            iae_config = replace(
                config.iae,
                confidence=local_failure,
                target_angular_precision=max(precision, 1e-10),
            )
            estimator = AnalyticIterativeAmplitudeEstimator(iae_config)
            intervals: dict[int, tuple[float, float]] = {}
            for arm in sorted(active):
                estimate = estimator.estimate(
                    oracle,
                    arm,
                    confidence=local_failure,
                    target_angular_precision=max(precision, 1e-10),
                    tag=f"{tag}:level_{level}",
                )
                intervals[arm] = (
                    estimate.angular_interval.lower,
                    estimate.angular_interval.upper,
                )
            levels_executed = level
            newly_accepted, newly_rejected = _classify(intervals, remaining_quota)
            accepted.update(newly_accepted)
            active.difference_update(newly_accepted | newly_rejected)

            expiring = (
                {arm for arm in active if limits[arm] == level}
                if known_stopping_schedule
                else set()
            )
            if expiring:
                return _AdaptiveOutcome(
                    tuple(sorted(accepted)), False, "known_stopping_schedule_exhausted", False,
                    levels_executed,
                )
    except QueryCapExceeded:
        return _AdaptiveOutcome(
            tuple(sorted(accepted)), False, "query_cap_exhausted", True, levels_executed
        )

    remaining_quota = quota_total - len(accepted)
    if remaining_quota == len(active):
        accepted.update(active)
        return _AdaptiveOutcome(
            tuple(sorted(accepted)), True, "certified_by_remaining_cardinality", False,
            levels_executed,
        )
    return _AdaptiveOutcome(
        tuple(sorted(accepted)), False, "precision_schedule_exhausted", False, levels_executed
    )


def _per_arm_queries(after: QuerySnapshot, before: QuerySnapshot, n_arms: int) -> Mapping[int, int]:
    totals: dict[int, int] = {}
    for arm in range(n_arms):
        after_counts = after.by_arm.get(arm, {})
        before_counts = before.by_arm.get(arm, {})
        totals[arm] = sum(
            int(after_counts.get(kind.value, 0) - before_counts.get(kind.value, 0))
            for kind in QueryKind
        )
        if totals[arm] < 0:
            raise RuntimeError("oracle ledger moved backwards")
    return MappingProxyType(totals)


def _result(
    *,
    method_id: str,
    information_regime: str,
    selected: Sequence[int],
    certified: bool,
    status: str,
    timeout: bool,
    k: int,
    failure_budget: float,
    query_cap: int,
    oracle: LayerCOracleProtocol,
    before: QuerySnapshot,
    stages: Sequence[MatchedBaselineStage],
    report_variable_time_proxy: bool,
) -> MatchedBaselineResult:
    after = oracle.query_snapshot()
    counts = _immutable_counts(QueryLedger.difference(after, before))
    per_arm = _per_arm_queries(after, before, oracle.n_arms)
    total = int(counts["total"])
    if sum(per_arm.values()) != total:
        raise RuntimeError("per-arm and aggregate Layer-C ledgers disagree")
    if total > query_cap:
        raise RuntimeError("hard query cap was violated")
    rms = (
        math.sqrt(sum(value * value for value in per_arm.values()))
        if report_variable_time_proxy
        else None
    )
    return MatchedBaselineResult(
        method_id=method_id,
        information_regime=information_regime,
        selected=tuple(sorted(_integer(arm, "selected arm") for arm in selected)),
        certified=bool(certified),
        status=status,
        timeout=bool(timeout),
        k=k,
        failure_budget=failure_budget,
        query_cap=query_cap,
        query_counts=counts,
        per_arm_queries=per_arm,
        stages=tuple(stages),
        variable_time_rms_proxy=rms,
        serial_query_total=total,
    )


def _validate_run(
    oracle: LayerCOracleProtocol, k: int, query_cap: int, failure_budget: float
) -> tuple[int, int, float, QuerySnapshot, _BudgetedOracle]:
    if not isinstance(oracle, LayerCOracleProtocol):
        raise TypeError("oracle must implement the Layer-C oracle protocol")
    n_arms = _integer(oracle.n_arms, "oracle.n_arms", minimum=1)
    top_k = _integer(k, "k", minimum=1)
    if top_k > n_arms:
        raise ValueError("k cannot exceed oracle.n_arms")
    cap = _integer(query_cap, "query_cap")
    failure = _open_probability(failure_budget, "failure_budget")
    before = oracle.query_snapshot()
    if before.total != 0:
        raise ValueError("each method requires an oracle with a fresh zero ledger")
    return top_k, cap, failure, before, _BudgetedOracle(oracle, cap)


class MatchedBaselineProtocol(Protocol):
    method_id: str

    def run(
        self,
        oracle: LayerCOracleProtocol,
        k: int,
        *,
        query_cap: int,
        failure_budget: float,
    ) -> MatchedBaselineResult: ...


class KOnlyIndependentAdaptiveBaseline:
    """All-active independent IAE elimination with only ``k`` as side information."""

    method_id = "k_only_independent_adaptive"
    information_regime = "oracle_k_failure_budget_query_cap"

    def __init__(self, config: MatchedBaselineConfig | None = None) -> None:
        self.config = config if config is not None else MatchedBaselineConfig()

    def run(
        self,
        oracle: LayerCOracleProtocol,
        k: int,
        *,
        query_cap: int,
        failure_budget: float,
    ) -> MatchedBaselineResult:
        top_k, cap, failure, before, budgeted = _validate_run(
            oracle, k, query_cap, failure_budget
        )
        stage_before = oracle.query_snapshot()
        outcome = _adaptive_topk(
            _SubsetOracle(budgeted, tuple(range(oracle.n_arms))),
            top_k,
            config=self.config,
            failure_budget=failure,
            tag=self.method_id,
        )
        stage = MatchedBaselineStage(
            stage_id="global_topk",
            arms=tuple(range(oracle.n_arms)),
            requested_k=top_k,
            selected=outcome.selected,
            certified=outcome.certified,
            status=outcome.status,
            levels_executed=outcome.levels_executed,
            query_counts=_immutable_counts(
                QueryLedger.difference(oracle.query_snapshot(), stage_before)
            ),
        )
        return _result(
            method_id=self.method_id,
            information_regime=self.information_regime,
            selected=outcome.selected,
            certified=outcome.certified,
            status=outcome.status,
            timeout=outcome.timeout,
            k=top_k,
            failure_budget=failure,
            query_cap=cap,
            oracle=oracle,
            before=before,
            stages=(stage,),
            report_variable_time_proxy=False,
        )


class CoarsePartitionBAICompositionBaseline:
    """Partition, retain each block's local Top-k, then solve the union.

    Retaining ``min(k, block_size)`` arms from every block preserves the global
    Top-k relation.  The local solver is the interval reference above; this is
    therefore an executable composition control, not an implementation of the
    strong-oracle QBAI theorem cited in the module documentation.
    """

    method_id = "coarse_partition_bai_composition"
    information_regime = "oracle_k_failure_budget_query_cap_public_partition"

    def __init__(
        self,
        config: MatchedBaselineConfig | None = None,
        *,
        block_size: int = 8,
        partition_seed: int = 0,
    ) -> None:
        self.config = config if config is not None else MatchedBaselineConfig()
        self.block_size = _integer(block_size, "block_size", minimum=2)
        self.partition_seed = _integer(partition_seed, "partition_seed")

    def _partition(self, n_arms: int) -> tuple[tuple[int, ...], ...]:
        # Hash sorting is deterministic, arm-value independent and stable
        # across Python versions (unlike random.shuffle implementation detail).
        arms = sorted(
            range(n_arms),
            key=lambda arm: hashlib.sha256(
                f"qgapselect-partition-v1\0{self.partition_seed}\0{arm}".encode()
            ).digest(),
        )
        return tuple(
            tuple(arms[start : start + self.block_size])
            for start in range(0, n_arms, self.block_size)
        )

    def run(
        self,
        oracle: LayerCOracleProtocol,
        k: int,
        *,
        query_cap: int,
        failure_budget: float,
    ) -> MatchedBaselineResult:
        top_k, cap, failure, before, budgeted = _validate_run(
            oracle, k, query_cap, failure_budget
        )
        blocks = self._partition(oracle.n_arms)
        total_subproblems = len(blocks) + 1
        stage_failure = failure / total_subproblems
        candidates: list[int] = []
        stages: list[MatchedBaselineStage] = []
        terminal: _AdaptiveOutcome | None = None

        for block_index, block in enumerate(blocks):
            local_k = min(top_k, len(block))
            stage_before = oracle.query_snapshot()
            outcome = _adaptive_topk(
                _SubsetOracle(budgeted, block),
                local_k,
                config=self.config,
                failure_budget=stage_failure,
                tag=f"{self.method_id}:block_{block_index}",
            )
            selected_global = tuple(block[index] for index in outcome.selected)
            stages.append(
                MatchedBaselineStage(
                    stage_id=f"block_{block_index}",
                    arms=block,
                    requested_k=local_k,
                    selected=tuple(sorted(selected_global)),
                    certified=outcome.certified,
                    status=outcome.status,
                    levels_executed=outcome.levels_executed,
                    query_counts=_immutable_counts(
                        QueryLedger.difference(oracle.query_snapshot(), stage_before)
                    ),
                )
            )
            if not outcome.certified:
                terminal = replace(
                    outcome,
                    selected=tuple(sorted(selected_global)),
                )
                break
            candidates.extend(selected_global)

        if terminal is None:
            stage_before = oracle.query_snapshot()
            final = _adaptive_topk(
                _SubsetOracle(budgeted, tuple(candidates)),
                top_k,
                config=self.config,
                failure_budget=stage_failure,
                tag=f"{self.method_id}:final",
            )
            selected_global = tuple(candidates[index] for index in final.selected)
            stages.append(
                MatchedBaselineStage(
                    stage_id="global_candidate_union",
                    arms=tuple(candidates),
                    requested_k=top_k,
                    selected=tuple(sorted(selected_global)),
                    certified=final.certified,
                    status=final.status,
                    levels_executed=final.levels_executed,
                    query_counts=_immutable_counts(
                        QueryLedger.difference(oracle.query_snapshot(), stage_before)
                    ),
                )
            )
            terminal = replace(final, selected=tuple(sorted(selected_global)))

        return _result(
            method_id=self.method_id,
            information_regime=self.information_regime,
            selected=terminal.selected,
            certified=terminal.certified,
            status=terminal.status,
            timeout=terminal.timeout,
            k=top_k,
            failure_budget=failure,
            query_cap=cap,
            oracle=oracle,
            before=before,
            stages=stages,
            report_variable_time_proxy=False,
        )


class RepeatedSingleOutputBaseline:
    """Run a certified one-best-arm subproblem ``k`` times on shrinking domains."""

    method_id = "repeated_single_output_selection"
    information_regime = "oracle_k_failure_budget_query_cap"

    def __init__(self, config: MatchedBaselineConfig | None = None) -> None:
        self.config = config if config is not None else MatchedBaselineConfig()

    def run(
        self,
        oracle: LayerCOracleProtocol,
        k: int,
        *,
        query_cap: int,
        failure_budget: float,
    ) -> MatchedBaselineResult:
        top_k, cap, failure, before, budgeted = _validate_run(
            oracle, k, query_cap, failure_budget
        )
        remaining = list(range(oracle.n_arms))
        selected: list[int] = []
        stages: list[MatchedBaselineStage] = []
        terminal_status = "certified_repeated_single_output"
        timeout = False
        certified = True
        for output_index in range(top_k):
            stage_before = oracle.query_snapshot()
            outcome = _adaptive_topk(
                _SubsetOracle(budgeted, tuple(remaining)),
                1,
                config=self.config,
                failure_budget=failure / top_k,
                tag=f"{self.method_id}:output_{output_index}",
            )
            selected_global = tuple(remaining[index] for index in outcome.selected)
            stages.append(
                MatchedBaselineStage(
                    stage_id=f"output_{output_index}",
                    arms=tuple(remaining),
                    requested_k=1,
                    selected=tuple(sorted(selected_global)),
                    certified=outcome.certified,
                    status=outcome.status,
                    levels_executed=outcome.levels_executed,
                    query_counts=_immutable_counts(
                        QueryLedger.difference(oracle.query_snapshot(), stage_before)
                    ),
                )
            )
            if not outcome.certified:
                terminal_status = outcome.status
                timeout = outcome.timeout
                certified = False
                break
            winner = selected_global[0]
            selected.append(winner)
            remaining.remove(winner)

        return _result(
            method_id=self.method_id,
            information_regime=self.information_regime,
            selected=selected,
            certified=certified and len(selected) == top_k,
            status=terminal_status,
            timeout=timeout,
            k=top_k,
            failure_budget=failure,
            query_cap=cap,
            oracle=oracle,
            before=before,
            stages=stages,
            report_variable_time_proxy=False,
        )


class UnknownTimeVariableTimeReference(KOnlyIndependentAdaptiveBaseline):
    """Unknown-stopping-time schedule plus an audited RMS composition proxy."""

    method_id = "unknown_time_variable_time_reference"
    information_regime = "oracle_k_failure_budget_query_cap_unknown_stopping_times"

    def run(
        self,
        oracle: LayerCOracleProtocol,
        k: int,
        *,
        query_cap: int,
        failure_budget: float,
    ) -> MatchedBaselineResult:
        result = super().run(
            oracle,
            k,
            query_cap=query_cap,
            failure_budget=failure_budget,
        )
        rms = math.sqrt(sum(value * value for value in result.per_arm_queries.values()))
        return replace(
            result,
            method_id=self.method_id,
            information_regime=self.information_regime,
            variable_time_rms_proxy=rms,
        )


class KnownTimeVariableTimeReference:
    """Public per-arm stopping-level reference (strictly stronger information).

    ``stop_levels`` are scheduling promises, not inferred from the oracle.  A
    run that has not certified an arm by its promised level fails instead of
    silently extending that branch.  This control must therefore be reported
    separately from the k-only methods.
    """

    method_id = "known_time_variable_time_reference"
    information_regime = (
        "oracle_k_failure_budget_query_cap_and_public_per_arm_stop_levels"
    )

    def __init__(
        self,
        stop_levels: Sequence[int],
        config: MatchedBaselineConfig | None = None,
    ) -> None:
        self.stop_levels = tuple(
            _integer(value, "stop level", minimum=1) for value in stop_levels
        )
        if not self.stop_levels:
            raise ValueError("stop_levels cannot be empty")
        self.config = config if config is not None else MatchedBaselineConfig()
        if any(level > self.config.max_levels for level in self.stop_levels):
            raise ValueError("stop levels cannot exceed config.max_levels")

    def run(
        self,
        oracle: LayerCOracleProtocol,
        k: int,
        *,
        query_cap: int,
        failure_budget: float,
    ) -> MatchedBaselineResult:
        top_k, cap, failure, before, budgeted = _validate_run(
            oracle, k, query_cap, failure_budget
        )
        if len(self.stop_levels) != oracle.n_arms:
            raise ValueError("stop_levels must align with oracle.n_arms")
        stage_before = oracle.query_snapshot()
        outcome = _adaptive_topk(
            _SubsetOracle(budgeted, tuple(range(oracle.n_arms))),
            top_k,
            config=self.config,
            failure_budget=failure,
            tag=self.method_id,
            arm_level_limits=self.stop_levels,
        )
        stage = MatchedBaselineStage(
            stage_id="known_stop_schedule",
            arms=tuple(range(oracle.n_arms)),
            requested_k=top_k,
            selected=outcome.selected,
            certified=outcome.certified,
            status=outcome.status,
            levels_executed=outcome.levels_executed,
            query_counts=_immutable_counts(
                QueryLedger.difference(oracle.query_snapshot(), stage_before)
            ),
        )
        return _result(
            method_id=self.method_id,
            information_regime=self.information_regime,
            selected=outcome.selected,
            certified=outcome.certified,
            status=outcome.status,
            timeout=outcome.timeout,
            k=top_k,
            failure_budget=failure,
            query_cap=cap,
            oracle=oracle,
            before=before,
            stages=(stage,),
            report_variable_time_proxy=True,
        )


def evaluate_fixed_query_cap(
    method: MatchedBaselineProtocol,
    oracle_factory: Callable[[], LayerCOracleProtocol],
    *,
    k: int,
    query_cap: int,
    failure_budget: float,
    expected_top_k: Sequence[int],
) -> FixedCapEvaluation:
    """Run one method on a fresh oracle and score it without selection filters."""

    if not callable(oracle_factory):
        raise TypeError("oracle_factory must be callable")
    oracle = oracle_factory()
    if not isinstance(oracle, LayerCOracleProtocol):
        raise TypeError("oracle_factory must return a Layer-C oracle")
    if oracle.query_snapshot().total != 0:
        raise ValueError("oracle_factory must return a fresh zero-ledger oracle")
    truth = tuple(sorted(_integer(arm, "expected arm") for arm in expected_top_k))
    top_k = _integer(k, "k", minimum=1)
    if len(truth) != top_k or len(set(truth)) != top_k:
        raise ValueError("expected_top_k must contain k unique arms")
    result = method.run(
        oracle,
        top_k,
        query_cap=query_cap,
        failure_budget=failure_budget,
    )
    if result.method_id != method.method_id:
        raise RuntimeError("method and result identifiers disagree")
    delta = QueryLedger.difference(oracle.query_snapshot(), QuerySnapshot({}, {}, {}))
    if dict(result.query_counts) != delta:
        raise RuntimeError("result does not contain the complete fresh oracle ledger")
    output_exact = result.selected == truth
    return FixedCapEvaluation(
        method_id=result.method_id,
        query_cap=result.query_cap,
        expected_top_k=truth,
        result=result,
        output_exact=output_exact,
        certified_exact=result.certified and output_exact,
        budget_valid=result.budget_valid,
    )


def aggregate_fixed_cap_evaluations(
    evaluations: Sequence[FixedCapEvaluation],
) -> FixedCapAggregate:
    """Aggregate all attempts; timeouts and wrong certificates stay in the denominator."""

    rows = tuple(evaluations)
    if not rows:
        raise ValueError("evaluations cannot be empty")
    if any(not isinstance(row, FixedCapEvaluation) for row in rows):
        raise TypeError("evaluations must contain FixedCapEvaluation values")
    method_ids = {row.method_id for row in rows}
    caps = {row.query_cap for row in rows}
    if len(method_ids) != 1 or len(caps) != 1:
        raise ValueError("aggregate rows must share one method_id and query_cap")
    queries = tuple(row.result.oracle_queries for row in rows)
    successes = sum(row.certified_exact for row in rows)
    return FixedCapAggregate(
        method_id=rows[0].method_id,
        query_cap=rows[0].query_cap,
        attempts=len(rows),
        certified_exact_count=successes,
        certified_exact_rate=successes / len(rows),
        timeout_count=sum(row.result.timeout for row in rows),
        uncertified_count=sum(not row.result.certified for row in rows),
        budget_violation_count=sum(not row.budget_valid for row in rows),
        mean_queries=statistics.fmean(queries),
        median_queries=float(statistics.median(queries)),
    )


__all__ = [
    "BACKEND",
    "CLAIM_SCOPE",
    "METHOD_ASSUMPTIONS",
    "REFERENCE_SOURCES",
    "REFERENCE_STATUS",
    "CoarsePartitionBAICompositionBaseline",
    "FixedCapAggregate",
    "FixedCapEvaluation",
    "KOnlyIndependentAdaptiveBaseline",
    "KnownTimeVariableTimeReference",
    "LayerCOracleProtocol",
    "MatchedBaselineConfig",
    "MatchedBaselineProtocol",
    "MatchedBaselineResult",
    "MatchedBaselineStage",
    "QueryCapExceeded",
    "RepeatedSingleOutputBaseline",
    "UnknownTimeVariableTimeReference",
    "aggregate_fixed_cap_evaluations",
    "evaluate_fixed_query_cap",
]
