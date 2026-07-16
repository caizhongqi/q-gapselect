"""Fixed-cap, information-matched Layer-C calibration harness.

This module puts the executable activity-history core and the strongest
executable references currently shipped by :mod:`qgapselect` on one audit
surface.  A primary trial receives exactly four public inputs: a fresh blind
oracle, ``k``, one failure budget, and one hard logical-query cap.  The
known-stopping-time method is deliberately separated as a stronger-information
control.

The trusted harness may inspect a frozen fixture only to score empirical
Top-k recovery *after* an algorithm returns.  It never passes empirical means,
the public threshold, or the public gap floor to a primary method.  Fixtures,
query caps, repetitions, and known-time schedules are materialized and
validated before the first outcome is observed, so this API cannot adapt them
to an algorithm's result.

All executions are analytic Layer-C Grover-measurement simulations.  They are
not quantum-hardware runs, official paper reproductions, theorem evidence, or
evidence of a quantum advantage.  In particular, the coherent activity-history
core still uses a finite-state circuit IR over analytic measurements rather
than a coherent index-register implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import statistics
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .coherent_activity_history_core import (
    VariableTimeCoherentHistoryResult,
    VariableTimeHistoryConfig,
    run_variable_time_coherent_activity_history,
)
from .frozen_coherent_oracle import (
    FrozenEmpiricalCoherentOracle,
    build_frozen_empirical_coherent_oracle,
)
from .frozen_quantum_reference_benchmarking import FrozenQuantumReferenceInstance
from .matched_quantum_baselines import (
    CoarsePartitionBAICompositionBaseline,
    KnownTimeVariableTimeReference,
    KOnlyIndependentAdaptiveBaseline,
    MatchedBaselineConfig,
    MatchedBaselineProtocol,
    MatchedBaselineResult,
    QueryCapExceeded,
    RepeatedSingleOutputBaseline,
    UnknownTimeVariableTimeReference,
)
from .oracles import QueryKind, QueryLedger, QuerySnapshot

CLAIM_SCOPE = (
    "analytic_layer_c_fixed_cap_information_matched_calibration_"
    "no_hardware_no_theorem_no_quantum_advantage_claim"
)
BACKEND = "fresh_frozen_empirical_oracle_with_analytic_grover_measurements"
SEED_NAMESPACE = "qgapselect-ccfa-matched-block-seed-v1"
SEED_DERIVATION = (
    "uint64_be(sha256(namespace\\0master_seed\\0family_id\\0instance_id\\0"
    "fixture_manifest_hash\\0repetition)[:8]); one common measurement seed is "
    "reused across caps and methods only through fresh oracle objects"
)
K_ONLY_INFORMATION_REGIME = "oracle_k_failure_budget_query_cap"
KNOWN_TIME_INFORMATION_REGIME = (
    "oracle_k_failure_budget_query_cap_and_preregistered_per_arm_stop_levels"
)
PRIMARY_COMPARISON_GROUP = "k_only_same_information_primary"
STRONGER_INFORMATION_GROUP = "known_time_stronger_information_control"
COHERENT_METHOD_ID = "variable_time_coherent_activity_history"
PRIMARY_METHOD_IDS = (
    COHERENT_METHOD_ID,
    "k_only_independent_adaptive",
    "coarse_partition_bai_composition",
    "repeated_single_output_selection",
    "unknown_time_variable_time_reference",
)
KNOWN_TIME_METHOD_ID = "known_time_variable_time_reference"


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


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _immutable_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    counts = MappingProxyType({str(key): int(value) for key, value in values.items()})
    if any(value < 0 for value in counts.values()):
        raise ValueError("query counts cannot be negative")
    required = {"coherent_total", "classical_total", "total"}
    if not required.issubset(counts):
        raise ValueError("query counts are missing aggregate totals")
    return counts


def _zero_counts() -> Mapping[str, int]:
    counts = {kind.value: 0 for kind in QueryKind}
    counts.update(coherent_total=0, classical_total=0, total=0)
    return _immutable_counts(counts)


def _canonical_hash(document: Mapping[str, object]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config_document(config: VariableTimeHistoryConfig) -> dict[str, object]:
    return {
        "confidence": config.confidence,
        "initial_angular_precision": config.initial_angular_precision,
        "precision_decay": config.precision_decay,
        "max_levels": config.max_levels,
        "shots_per_iae_round": config.shots_per_iae_round,
        "iae_max_rounds": config.iae_max_rounds,
        "iae_max_grover_power": config.iae_max_grover_power,
        "iae_grid_points": config.iae_grid_points,
        "verification_angular_precision": config.verification_angular_precision,
        "verification_shots_per_round": config.verification_shots_per_round,
        "verification_max_rounds": config.verification_max_rounds,
        "verification_max_grover_power": config.verification_max_grover_power,
        "verification_grid_points": config.verification_grid_points,
    }


def _baseline_config_document(config: MatchedBaselineConfig) -> dict[str, object]:
    return {
        "initial_angular_precision": config.initial_angular_precision,
        "precision_decay": config.precision_decay,
        "max_levels": config.max_levels,
        "iae": {
            "target_angular_precision": config.iae.target_angular_precision,
            "confidence": config.iae.confidence,
            "shots_per_round": config.iae.shots_per_round,
            "max_rounds": config.iae.max_rounds,
            "max_grover_power": config.iae.max_grover_power,
            "grid_points": config.iae.grid_points,
        },
    }


@runtime_checkable
class _LayerCOracleProtocol(Protocol):
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


class AtomicQueryCapOracle:
    """Public coherent-core wrapper that rejects an unaffordable experiment.

    The affordability check and delegated experiment execute under one lock.
    Consequently, callers using this wrapper cannot interleave two experiments
    and oversubscribe the cap between a check and a charge.  Rejection occurs
    before the underlying oracle changes its ledger or consumes randomness.
    """

    def __init__(self, oracle: _LayerCOracleProtocol, query_cap: int) -> None:
        if not isinstance(oracle, _LayerCOracleProtocol):
            raise TypeError("oracle must expose the blind Layer-C capability")
        cap = _integer(query_cap, "query_cap")
        before = oracle.query_snapshot()
        if before.total != 0:
            raise ValueError("AtomicQueryCapOracle requires a fresh zero-ledger oracle")
        self._oracle = oracle
        self._query_cap = cap
        self._before = before
        self._lock = threading.RLock()
        self._rejections = 0

    @property
    def n_arms(self) -> int:
        return self._oracle.n_arms

    @property
    def query_cap(self) -> int:
        return self._query_cap

    @property
    def spent(self) -> int:
        return int(QueryLedger.difference(self._oracle.query_snapshot(), self._before)["total"])

    @property
    def remaining(self) -> int:
        return self.query_cap - self.spent

    @property
    def rejected_experiments(self) -> int:
        return self._rejections

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
        arm_index = _integer(arm, "arm")
        if arm_index >= self.n_arms:
            raise IndexError(f"arm {arm_index} is outside [0, {self.n_arms})")
        power = _integer(grover_power, "grover_power")
        shot_count = _integer(shots, "shots", minimum=1)
        if not isinstance(controlled, bool):
            raise TypeError("controlled must be bool")
        requested = shot_count * (2 * power + 1)
        with self._lock:
            spent = self.spent
            if spent + requested > self.query_cap:
                self._rejections += 1
                raise QueryCapExceeded(
                    cap=self.query_cap,
                    spent=spent,
                    requested=requested,
                )
            return self._oracle.run_grover_experiment(
                arm_index,
                power,
                shot_count,
                controlled=controlled,
                tag=tag,
            )


def _tag_prefix_queries(snapshot: QuerySnapshot, prefix: str) -> int:
    return sum(
        sum(int(value) for value in counts.values())
        for tag, counts in snapshot.by_tag.items()
        if tag.startswith(prefix)
    )


@dataclass(frozen=True, slots=True)
class CappedCoherentExecution:
    """Coherent-core result normalized for the fixed-cap harness."""

    result: VariableTimeCoherentHistoryResult | None
    timeout: bool
    status: str
    query_cap: int
    query_counts: Mapping[str, int]
    selection_queries: int
    verification_queries: int
    cleanup_passed: bool
    rejected_experiments: int

    @property
    def actual_queries(self) -> int:
        return int(self.query_counts["total"])

    @property
    def budget_valid(self) -> bool:
        return self.actual_queries <= self.query_cap

    def as_dict(self) -> dict[str, object]:
        return {
            "timeout": self.timeout,
            "status": self.status,
            "query_cap": self.query_cap,
            "query_counts": dict(self.query_counts),
            "selection_queries": self.selection_queries,
            "verification_queries": self.verification_queries,
            "cleanup_passed": self.cleanup_passed,
            "rejected_experiments": self.rejected_experiments,
            "actual_queries": self.actual_queries,
            "budget_valid": self.budget_valid,
            "complete": False if self.result is None else self.result.complete,
            "certified": False if self.result is None else self.result.certified,
        }


def run_capped_coherent_activity_history(
    oracle: FrozenEmpiricalCoherentOracle,
    k: int,
    *,
    query_cap: int,
    failure_budget: float,
    config: VariableTimeHistoryConfig | None = None,
) -> CappedCoherentExecution:
    """Run the activity-history core under an atomic hard query cap.

    ``failure_budget`` replaces the config's confidence value so the coherent
    method receives the same public global budget as every baseline.
    ``QueryCapExceeded`` is normalized into a timeout; all other exceptions
    remain visible because they indicate an implementation error, not a trial
    outcome.
    """

    if not isinstance(oracle, FrozenEmpiricalCoherentOracle):
        raise TypeError("oracle must be a FrozenEmpiricalCoherentOracle")
    top_k = _integer(k, "k", minimum=1)
    if top_k >= oracle.n_arms:
        raise ValueError("k must satisfy 1 <= k < oracle.n_arms")
    cap = _integer(query_cap, "query_cap")
    failure = _probability(failure_budget, "failure_budget")
    base_config = config if config is not None else VariableTimeHistoryConfig()
    if not isinstance(base_config, VariableTimeHistoryConfig):
        raise TypeError("config must be a VariableTimeHistoryConfig")
    effective_config = replace(base_config, confidence=failure)
    capped = AtomicQueryCapOracle(oracle, cap)

    result: VariableTimeCoherentHistoryResult | None = None
    timeout = False
    status = ""
    try:
        result = run_variable_time_coherent_activity_history(
            capped,
            top_k,
            config=effective_config,
        )
        status = result.status
    except QueryCapExceeded:
        timeout = True
        status = "query_cap_exhausted"

    snapshot = oracle.query_snapshot()
    counts = _immutable_counts(snapshot.flat())
    if counts["total"] > cap:
        raise RuntimeError("atomic hard query cap was violated")
    if result is None:
        selection = _tag_prefix_queries(snapshot, "vt_history_level_")
        verification = _tag_prefix_queries(snapshot, "vt_history_fresh_verify_")
        if selection + verification != counts["total"]:
            raise RuntimeError("coherent timeout tags do not partition the fresh ledger")
        cleanup = False
    else:
        resources = result.executed_resources
        selection = int(resources.selection_query_counts["total"])
        verification = int(resources.verification_query_counts["total"])
        if selection + verification != counts["total"]:
            raise RuntimeError("coherent resource partitions do not match fresh ledger")
        cleanup = bool(
            result.candidate_ir_resources.cleanup_verified
            and all(branch.predicate_workspace_zero for branch in result.branches)
            and all(branch.phase_workspace_zero for branch in result.branches)
        )
    return CappedCoherentExecution(
        result=result,
        timeout=timeout,
        status=status,
        query_cap=cap,
        query_counts=counts,
        selection_queries=selection,
        verification_queries=verification,
        cleanup_passed=cleanup,
        rejected_experiments=capped.rejected_experiments,
    )


@dataclass(frozen=True, slots=True)
class KnownTimeControlSpec:
    """Preregistered stronger-information stopping schedule for one fixture."""

    family_id: str
    instance_id: str
    stop_levels: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _nonempty(self.family_id, "family_id"))
        object.__setattr__(self, "instance_id", _nonempty(self.instance_id, "instance_id"))
        levels = tuple(_integer(value, "stop level", minimum=1) for value in self.stop_levels)
        if not levels:
            raise ValueError("stop_levels cannot be empty")
        object.__setattr__(self, "stop_levels", levels)

    def as_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "instance_id": self.instance_id,
            "stop_levels": list(self.stop_levels),
        }


@dataclass(frozen=True, slots=True)
class CCFAMatchedBenchmarkConfig:
    """Fully preregistered campaign settings, fixed before any trial outcome."""

    master_seed: int = 20260715
    repetitions: int = 1
    query_caps: tuple[int, ...] = (100_000,)
    failure_budget: float = 0.05
    coherent: VariableTimeHistoryConfig = field(default_factory=VariableTimeHistoryConfig)
    baselines: MatchedBaselineConfig = field(default_factory=MatchedBaselineConfig)
    coarse_partition_block_size: int = 8
    coarse_partition_seed: int = 0
    known_time_controls: tuple[KnownTimeControlSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "master_seed", _integer(self.master_seed, "master_seed"))
        object.__setattr__(
            self, "repetitions", _integer(self.repetitions, "repetitions", minimum=1)
        )
        caps = tuple(_integer(cap, "query cap") for cap in self.query_caps)
        if not caps:
            raise ValueError("query_caps cannot be empty")
        if len(set(caps)) != len(caps):
            raise ValueError("query_caps must be unique")
        object.__setattr__(self, "query_caps", tuple(sorted(caps)))
        object.__setattr__(
            self,
            "failure_budget",
            _probability(self.failure_budget, "failure_budget"),
        )
        if not isinstance(self.coherent, VariableTimeHistoryConfig):
            raise TypeError("coherent must be a VariableTimeHistoryConfig")
        if not isinstance(self.baselines, MatchedBaselineConfig):
            raise TypeError("baselines must be a MatchedBaselineConfig")
        object.__setattr__(
            self,
            "coarse_partition_block_size",
            _integer(
                self.coarse_partition_block_size,
                "coarse_partition_block_size",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "coarse_partition_seed",
            _integer(self.coarse_partition_seed, "coarse_partition_seed"),
        )
        controls = tuple(self.known_time_controls)
        if any(not isinstance(row, KnownTimeControlSpec) for row in controls):
            raise TypeError("known_time_controls must contain KnownTimeControlSpec values")
        keys = {(row.family_id, row.instance_id) for row in controls}
        if len(keys) != len(controls):
            raise ValueError("known_time_controls contain duplicate fixture keys")
        object.__setattr__(self, "known_time_controls", controls)

    @property
    def method_ids(self) -> tuple[str, ...]:
        if self.known_time_controls:
            return (*PRIMARY_METHOD_IDS, KNOWN_TIME_METHOD_ID)
        return PRIMARY_METHOD_IDS

    def as_dict(self) -> dict[str, object]:
        return {
            "master_seed": self.master_seed,
            "repetitions": self.repetitions,
            "query_caps": list(self.query_caps),
            "failure_budget": self.failure_budget,
            "coherent": _config_document(self.coherent),
            "baselines": _baseline_config_document(self.baselines),
            "coarse_partition_block_size": self.coarse_partition_block_size,
            "coarse_partition_seed": self.coarse_partition_seed,
            "known_time_controls": [row.as_dict() for row in self.known_time_controls],
            "method_ids": list(self.method_ids),
            "claim_scope": CLAIM_SCOPE,
        }


def _block_seed(
    master_seed: int,
    instance: FrozenQuantumReferenceInstance,
    repetition: int,
) -> int:
    material = "\0".join(
        (
            SEED_NAMESPACE,
            str(master_seed),
            instance.family_id,
            instance.instance_id,
            instance.fixture.manifest_hash,
            str(repetition),
        )
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _trusted_expected_top_k(instance: FrozenQuantumReferenceInstance) -> tuple[int, ...]:
    """Trusted scoring helper; its result is never sent to an algorithm."""

    means = instance.fixture.evaluator.frozen_means
    ranking = sorted(range(len(means)), key=lambda arm: (-means[arm], arm))
    return tuple(sorted(ranking[: instance.k]))


@dataclass(frozen=True, slots=True)
class CCFAMatchedTrialRecord:
    """One all-denominator trial with an exact fresh-ledger audit."""

    family_id: str
    instance_id: str
    fixture_manifest_hash: str
    difficulty_fingerprint: str
    repetition: int
    query_cap: int
    method_id: str
    panel_method_ids: tuple[str, ...]
    preregistered_fixture_keys: tuple[tuple[str, str], ...]
    preregistered_query_caps: tuple[int, ...]
    preregistered_repetitions: int
    campaign_manifest_hash: str
    comparison_group: str
    information_regime: str
    algorithm_reported_information_regime: str
    expected_top_k: tuple[int, ...]
    selected: tuple[int, ...]
    exact: bool
    certified: bool
    certified_exact: bool
    timeout: bool
    budget_valid: bool
    complete: bool
    cleanup_passed: bool | None
    direct_multi_output: bool
    actual_queries: int
    selection_queries: int
    verification_queries: int
    query_counts: Mapping[str, int]
    status: str
    block_seed: int
    measurement_seed: int
    seed_derivation: str
    oracle_ledger_start_queries: int = 0
    empirical_truth_source: str = "trusted_harness_frozen_empirical_means_never_algorithm_input"
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE
    hardware_claimable: bool = False
    theorem_claimable: bool = False
    quantum_advantage_claimable: bool = False

    def __post_init__(self) -> None:
        fixture_key = (self.family_id, self.instance_id)
        if fixture_key not in self.preregistered_fixture_keys:
            raise ValueError("trial fixture lies outside its preregistered panel")
        if self.query_cap not in self.preregistered_query_caps:
            raise ValueError("trial query cap lies outside its preregistered panel")
        if not 0 <= self.repetition < self.preregistered_repetitions:
            raise ValueError("trial repetition lies outside its preregistered panel")
        if self.method_id not in self.panel_method_ids:
            raise ValueError("trial method lies outside its preregistered panel")
        if len(self.campaign_manifest_hash) != 64:
            raise ValueError("campaign_manifest_hash must be a SHA-256 digest")
        try:
            int(self.campaign_manifest_hash, 16)
        except ValueError as error:
            raise ValueError("campaign_manifest_hash must be a SHA-256 digest") from error
        if self.certified_exact != (self.certified and self.exact):
            raise ValueError("certified_exact must equal certified and exact")
        if self.actual_queries != int(self.query_counts.get("total", -1)):
            raise ValueError("actual_queries must match the query ledger")
        if self.selection_queries + self.verification_queries != self.actual_queries:
            raise ValueError("selection and verification queries must partition the ledger")
        if self.budget_valid != (self.actual_queries <= self.query_cap):
            raise ValueError("budget_valid disagrees with the hard cap")
        if self.oracle_ledger_start_queries != 0:
            raise ValueError("every method trial must start from a fresh zero ledger")

    @property
    def panel_key(self) -> tuple[str, str, str, int, int]:
        return (
            self.family_id,
            self.instance_id,
            self.fixture_manifest_hash,
            self.repetition,
            self.query_cap,
        )

    @property
    def coherent_queries(self) -> int:
        """Evidence-gate alias for the coherent part of the exact ledger."""

        return int(self.query_counts.get("coherent_total", 0))

    @property
    def certified_exact_recovery(self) -> bool:
        """Fixed-fixture calibration alias for the strict success event."""

        return self.certified_exact

    def as_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "instance_id": self.instance_id,
            "fixture_manifest_hash": self.fixture_manifest_hash,
            "difficulty_fingerprint": self.difficulty_fingerprint,
            "repetition": self.repetition,
            "query_cap": self.query_cap,
            "method_id": self.method_id,
            "panel_method_ids": list(self.panel_method_ids),
            "preregistered_fixture_keys": [list(key) for key in self.preregistered_fixture_keys],
            "preregistered_query_caps": list(self.preregistered_query_caps),
            "preregistered_repetitions": self.preregistered_repetitions,
            "campaign_manifest_hash": self.campaign_manifest_hash,
            "comparison_group": self.comparison_group,
            "information_regime": self.information_regime,
            "algorithm_reported_information_regime": (self.algorithm_reported_information_regime),
            "expected_top_k": list(self.expected_top_k),
            "selected": list(self.selected),
            "exact": self.exact,
            "certified": self.certified,
            "certified_exact": self.certified_exact,
            "timeout": self.timeout,
            "budget_valid": self.budget_valid,
            "complete": self.complete,
            "cleanup_passed": self.cleanup_passed,
            "direct_multi_output": self.direct_multi_output,
            "finite_state_direct_output_tape": self.direct_multi_output,
            "coherent_direct_multi_output_verified": False,
            "direct_multi_output_semantics": (
                "analytic_finite_state_output_tape_not_coherent_cross_index_union"
                if self.direct_multi_output
                else "not_executed"
            ),
            "actual_queries": self.actual_queries,
            "coherent_queries": self.coherent_queries,
            "selection_queries": self.selection_queries,
            "verification_queries": self.verification_queries,
            "query_counts": dict(self.query_counts),
            "status": self.status,
            "block_seed": self.block_seed,
            "measurement_seed": self.measurement_seed,
            "seed_derivation": self.seed_derivation,
            "oracle_ledger_start_queries": self.oracle_ledger_start_queries,
            "empirical_truth_source": self.empirical_truth_source,
            "backend": self.backend,
            "claim_scope": self.claim_scope,
            "hardware_claimable": self.hardware_claimable,
            "theorem_claimable": self.theorem_claimable,
            "quantum_advantage_claimable": self.quantum_advantage_claimable,
        }


def _baseline_methods(
    config: CCFAMatchedBenchmarkConfig,
    *,
    known_stop_levels: tuple[int, ...] | None,
) -> tuple[MatchedBaselineProtocol, ...]:
    methods: list[MatchedBaselineProtocol] = [
        KOnlyIndependentAdaptiveBaseline(config.baselines),
        CoarsePartitionBAICompositionBaseline(
            config.baselines,
            block_size=config.coarse_partition_block_size,
            partition_seed=config.coarse_partition_seed,
        ),
        RepeatedSingleOutputBaseline(config.baselines),
        UnknownTimeVariableTimeReference(config.baselines),
    ]
    if known_stop_levels is not None:
        methods.append(KnownTimeVariableTimeReference(known_stop_levels, config.baselines))
    return tuple(methods)


def _trial_from_coherent(
    instance: FrozenQuantumReferenceInstance,
    *,
    repetition: int,
    query_cap: int,
    block_seed: int,
    panel_method_ids: tuple[str, ...],
    preregistered_fixture_keys: tuple[tuple[str, str], ...],
    campaign_manifest_hash: str,
    config: CCFAMatchedBenchmarkConfig,
) -> CCFAMatchedTrialRecord:
    oracle = build_frozen_empirical_coherent_oracle(instance.fixture, measurement_seed=block_seed)
    if oracle.query_snapshot().total != 0:
        raise RuntimeError("trusted factory returned a nonfresh oracle")
    execution = run_capped_coherent_activity_history(
        oracle,
        instance.k,
        query_cap=query_cap,
        failure_budget=config.failure_budget,
        config=config.coherent,
    )
    result = execution.result
    selected = () if result is None else result.extracted_selected
    complete = bool(result is not None and result.complete)
    certified = bool(result is not None and result.certified)
    expected = _trusted_expected_top_k(instance)
    exact = selected == expected
    return CCFAMatchedTrialRecord(
        family_id=instance.family_id,
        instance_id=instance.instance_id,
        fixture_manifest_hash=instance.fixture.manifest_hash,
        difficulty_fingerprint=str(instance.difficulty_fingerprint),
        repetition=repetition,
        query_cap=query_cap,
        method_id=COHERENT_METHOD_ID,
        panel_method_ids=panel_method_ids,
        preregistered_fixture_keys=preregistered_fixture_keys,
        preregistered_query_caps=config.query_caps,
        preregistered_repetitions=config.repetitions,
        campaign_manifest_hash=campaign_manifest_hash,
        comparison_group=PRIMARY_COMPARISON_GROUP,
        information_regime=K_ONLY_INFORMATION_REGIME,
        algorithm_reported_information_regime=("oracle_k_with_internal_confidence_schedule"),
        expected_top_k=expected,
        selected=selected,
        exact=exact,
        certified=certified,
        certified_exact=certified and exact,
        timeout=execution.timeout,
        budget_valid=execution.budget_valid,
        complete=complete,
        cleanup_passed=execution.cleanup_passed,
        direct_multi_output=True,
        actual_queries=execution.actual_queries,
        selection_queries=execution.selection_queries,
        verification_queries=execution.verification_queries,
        query_counts=execution.query_counts,
        status=execution.status,
        block_seed=block_seed,
        measurement_seed=block_seed,
        seed_derivation=SEED_DERIVATION,
    )


def _trial_from_baseline(
    instance: FrozenQuantumReferenceInstance,
    method: MatchedBaselineProtocol,
    *,
    repetition: int,
    query_cap: int,
    block_seed: int,
    panel_method_ids: tuple[str, ...],
    preregistered_fixture_keys: tuple[tuple[str, str], ...],
    campaign_manifest_hash: str,
    config: CCFAMatchedBenchmarkConfig,
) -> CCFAMatchedTrialRecord:
    oracle = build_frozen_empirical_coherent_oracle(instance.fixture, measurement_seed=block_seed)
    if oracle.query_snapshot().total != 0:
        raise RuntimeError("trusted factory returned a nonfresh oracle")
    result: MatchedBaselineResult = method.run(
        oracle,
        instance.k,
        query_cap=query_cap,
        failure_budget=config.failure_budget,
    )
    snapshot_counts = _immutable_counts(oracle.query_snapshot().flat())
    if dict(result.query_counts) != dict(snapshot_counts):
        raise RuntimeError("baseline result omitted part of its fresh oracle ledger")
    expected = _trusted_expected_top_k(instance)
    selected = result.selected
    exact = selected == expected
    is_known = method.method_id == KNOWN_TIME_METHOD_ID
    return CCFAMatchedTrialRecord(
        family_id=instance.family_id,
        instance_id=instance.instance_id,
        fixture_manifest_hash=instance.fixture.manifest_hash,
        difficulty_fingerprint=str(instance.difficulty_fingerprint),
        repetition=repetition,
        query_cap=query_cap,
        method_id=method.method_id,
        panel_method_ids=panel_method_ids,
        preregistered_fixture_keys=preregistered_fixture_keys,
        preregistered_query_caps=config.query_caps,
        preregistered_repetitions=config.repetitions,
        campaign_manifest_hash=campaign_manifest_hash,
        comparison_group=(STRONGER_INFORMATION_GROUP if is_known else PRIMARY_COMPARISON_GROUP),
        information_regime=(
            KNOWN_TIME_INFORMATION_REGIME if is_known else K_ONLY_INFORMATION_REGIME
        ),
        algorithm_reported_information_regime=result.information_regime,
        expected_top_k=expected,
        selected=selected,
        exact=exact,
        certified=result.certified,
        certified_exact=result.certified and exact,
        timeout=result.timeout,
        budget_valid=result.budget_valid,
        complete=len(selected) == instance.k,
        cleanup_passed=None,
        direct_multi_output=False,
        actual_queries=result.oracle_queries,
        selection_queries=result.oracle_queries,
        verification_queries=0,
        query_counts=snapshot_counts,
        status=result.status,
        block_seed=block_seed,
        measurement_seed=block_seed,
        seed_derivation=SEED_DERIVATION,
    )


def iter_ccfa_matched_trials(
    instances: Iterable[FrozenQuantumReferenceInstance],
    config: CCFAMatchedBenchmarkConfig,
    *,
    execution_fixture_keys: Sequence[tuple[str, str]] | None = None,
) -> Iterable[CCFAMatchedTrialRecord]:
    """Stream a balanced fixed-cap panel without outcome-adaptive choices.

    ``execution_fixture_keys`` is an execution-only shard selector.  The
    preregistered fixture panel and campaign manifest are always constructed
    from every supplied instance.  In particular, records emitted by a shard
    continue to declare the complete panel, so a shard cannot pass
    :func:`validate_complete_matched_panel` until all fixture shards have been
    merged.
    """

    if not isinstance(config, CCFAMatchedBenchmarkConfig):
        raise TypeError("config must be a CCFAMatchedBenchmarkConfig")
    frozen_instances = tuple(instances)
    if not frozen_instances:
        raise ValueError("instances cannot be empty")
    if any(not isinstance(row, FrozenQuantumReferenceInstance) for row in frozen_instances):
        raise TypeError("instances must contain FrozenQuantumReferenceInstance values")
    fixture_keys = {(row.family_id, row.instance_id) for row in frozen_instances}
    if len(fixture_keys) != len(frozen_instances):
        raise ValueError("family_id and instance_id pairs must be unique")

    selected_fixture_keys: frozenset[tuple[str, str]] | None = None
    if execution_fixture_keys is not None:
        if isinstance(execution_fixture_keys, (str, bytes)) or not isinstance(
            execution_fixture_keys, Sequence
        ):
            raise TypeError("execution_fixture_keys must be a sequence of key tuples")
        raw_execution_keys = tuple(execution_fixture_keys)
        if not raw_execution_keys:
            raise ValueError("execution_fixture_keys cannot be empty")
        checked_execution_keys: list[tuple[str, str]] = []
        for key in raw_execution_keys:
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError(
                    "each execution fixture key must be a (family_id, instance_id) tuple"
                )
            family_id = _nonempty(key[0], "execution fixture family_id")
            instance_id = _nonempty(key[1], "execution fixture instance_id")
            checked_execution_keys.append((family_id, instance_id))
        if len(set(checked_execution_keys)) != len(checked_execution_keys):
            raise ValueError("execution_fixture_keys must be unique")
        unknown = set(checked_execution_keys).difference(fixture_keys)
        if unknown:
            raise ValueError(
                f"execution_fixture_keys contains an unregistered fixture: {min(unknown)!r}"
            )
        selected_fixture_keys = frozenset(checked_execution_keys)

    known_by_key = {
        (row.family_id, row.instance_id): row.stop_levels for row in config.known_time_controls
    }
    if known_by_key and set(known_by_key) != fixture_keys:
        raise ValueError("known-time controls must preregister exactly one schedule per fixture")

    # Validate and construct the complete panel before the first outcome is
    # yielded.  A malformed later fixture therefore cannot be discovered only
    # after earlier trial results have already become visible.
    prepared: list[
        tuple[
            FrozenQuantumReferenceInstance,
            tuple[MatchedBaselineProtocol, ...],
            tuple[str, ...],
        ]
    ] = []
    for instance in frozen_instances:
        stop_levels = known_by_key.get((instance.family_id, instance.instance_id))
        n_arms = len(instance.fixture.tensor.graph.candidate_ids)
        if not 1 <= instance.k < n_arms:
            raise ValueError("the coherent matched panel requires 1 <= k < fixture n_arms")
        if stop_levels is not None and len(stop_levels) != n_arms:
            raise ValueError("known-time stop levels must align with fixture arms")
        # Construct all method objects before seeing any outcome.  This also
        # validates every known-time level against the baseline schedule.
        methods = _baseline_methods(config, known_stop_levels=stop_levels)
        panel_method_ids = (COHERENT_METHOD_ID, *(method.method_id for method in methods))
        if panel_method_ids != config.method_ids:
            raise RuntimeError("constructed method panel differs from preregistration")
        prepared.append((instance, methods, panel_method_ids))

    preregistered_fixture_keys = tuple(
        (instance.family_id, instance.instance_id) for instance in frozen_instances
    )
    campaign_manifest_hash = matched_campaign_manifest_hash(frozen_instances, config)
    for instance, methods, panel_method_ids in prepared:
        fixture_key = (instance.family_id, instance.instance_id)
        if selected_fixture_keys is not None and fixture_key not in selected_fixture_keys:
            continue
        for repetition in range(config.repetitions):
            block_seed = _block_seed(config.master_seed, instance, repetition)
            for query_cap in config.query_caps:
                yield _trial_from_coherent(
                    instance,
                    repetition=repetition,
                    query_cap=query_cap,
                    block_seed=block_seed,
                    panel_method_ids=panel_method_ids,
                    preregistered_fixture_keys=preregistered_fixture_keys,
                    campaign_manifest_hash=campaign_manifest_hash,
                    config=config,
                )
                for method in methods:
                    yield _trial_from_baseline(
                        instance,
                        method,
                        repetition=repetition,
                        query_cap=query_cap,
                        block_seed=block_seed,
                        panel_method_ids=panel_method_ids,
                        preregistered_fixture_keys=preregistered_fixture_keys,
                        campaign_manifest_hash=campaign_manifest_hash,
                        config=config,
                    )


@dataclass(frozen=True, slots=True)
class CCFAMatchedAggregate:
    """Method/cap aggregate over every attempted fixture-repetition block."""

    method_id: str
    query_cap: int
    comparison_group: str
    information_regime: str
    attempts: int
    exact_count: int
    certified_count: int
    certified_exact_count: int
    timeout_count: int
    budget_violation_count: int
    incomplete_count: int
    cleanup_failure_count: int
    cleanup_not_applicable_count: int
    direct_multi_output: bool
    exact_rate: float
    certified_exact_rate: float
    mean_actual_queries: float
    median_actual_queries: float
    mean_selection_queries: float
    mean_verification_queries: float
    claim_scope: str = CLAIM_SCOPE
    quantum_advantage_claimable: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "query_cap": self.query_cap,
            "comparison_group": self.comparison_group,
            "information_regime": self.information_regime,
            "attempts": self.attempts,
            "exact_count": self.exact_count,
            "certified_count": self.certified_count,
            "certified_exact_count": self.certified_exact_count,
            "timeout_count": self.timeout_count,
            "budget_violation_count": self.budget_violation_count,
            "incomplete_count": self.incomplete_count,
            "cleanup_failure_count": self.cleanup_failure_count,
            "cleanup_not_applicable_count": self.cleanup_not_applicable_count,
            "direct_multi_output": self.direct_multi_output,
            "finite_state_direct_output_tape": self.direct_multi_output,
            "coherent_direct_multi_output_verified": False,
            "direct_multi_output_semantics": (
                "analytic_finite_state_output_tape_not_coherent_cross_index_union"
                if self.direct_multi_output
                else "not_executed"
            ),
            "exact_rate": self.exact_rate,
            "certified_exact_rate": self.certified_exact_rate,
            "mean_actual_queries": self.mean_actual_queries,
            "median_actual_queries": self.median_actual_queries,
            "mean_selection_queries": self.mean_selection_queries,
            "mean_verification_queries": self.mean_verification_queries,
            "claim_scope": self.claim_scope,
            "quantum_advantage_claimable": self.quantum_advantage_claimable,
        }


def validate_complete_matched_panel(
    records: Sequence[CCFAMatchedTrialRecord],
) -> None:
    """Reject duplicates, missing methods, seed drift, and partial panels."""

    rows = tuple(records)
    if not rows:
        raise ValueError("records cannot be empty")
    if any(not isinstance(row, CCFAMatchedTrialRecord) for row in rows):
        raise TypeError("records must contain CCFAMatchedTrialRecord values")
    campaign_metadata = {
        (
            row.preregistered_fixture_keys,
            row.preregistered_query_caps,
            row.preregistered_repetitions,
            row.panel_method_ids,
            row.campaign_manifest_hash,
        )
        for row in rows
    }
    if len(campaign_metadata) != 1:
        raise ValueError("records mix inconsistent preregistered campaign panels")
    fixture_keys, query_caps, repetitions, method_ids, _ = next(iter(campaign_metadata))
    observed_blocks = {
        (row.family_id, row.instance_id, row.repetition, row.query_cap) for row in rows
    }
    expected_blocks = {
        (family_id, instance_id, repetition, query_cap)
        for family_id, instance_id in fixture_keys
        for repetition in range(repetitions)
        for query_cap in query_caps
    }
    missing_blocks = sorted(expected_blocks.difference(observed_blocks))
    if missing_blocks:
        raise ValueError(
            "campaign is missing an entire fixture/repetition/query-cap block: "
            f"{missing_blocks[0]!r}"
        )
    extra_blocks = sorted(observed_blocks.difference(expected_blocks))
    if extra_blocks:
        raise ValueError(f"campaign contains an unregistered block: {extra_blocks[0]!r}")
    grouped: dict[tuple[str, str, str, int, int], list[CCFAMatchedTrialRecord]] = {}
    for row in rows:
        grouped.setdefault(row.panel_key, []).append(row)
    for key, block in grouped.items():
        declared = {row.panel_method_ids for row in block}
        if len(declared) != 1:
            raise ValueError(f"panel {key!r} has inconsistent method declarations")
        expected = next(iter(declared))
        if expected != method_ids:
            raise ValueError(f"panel {key!r} differs from the campaign method panel")
        observed = tuple(row.method_id for row in block)
        if len(set(observed)) != len(observed):
            raise ValueError(f"panel {key!r} contains duplicate method records")
        if set(observed) != set(expected) or len(observed) != len(expected):
            raise ValueError(f"panel {key!r} is missing a preregistered method")
        if len({row.block_seed for row in block}) != 1:
            raise ValueError(f"panel {key!r} does not share one block seed")
        if any(row.measurement_seed != row.block_seed for row in block):
            raise ValueError(f"panel {key!r} contains method-specific seed drift")


def aggregate_ccfa_matched_trials(
    records: Sequence[CCFAMatchedTrialRecord],
) -> tuple[CCFAMatchedAggregate, ...]:
    """Aggregate complete panels; timeouts and failures remain in denominators."""

    rows = tuple(records)
    validate_complete_matched_panel(rows)
    groups: dict[tuple[str, int], list[CCFAMatchedTrialRecord]] = {}
    for row in rows:
        groups.setdefault((row.method_id, row.query_cap), []).append(row)
    aggregates: list[CCFAMatchedAggregate] = []
    for (method_id, query_cap), group in sorted(groups.items()):
        information = {row.information_regime for row in group}
        comparisons = {row.comparison_group for row in group}
        direct = {row.direct_multi_output for row in group}
        if len(information) != 1 or len(comparisons) != 1 or len(direct) != 1:
            raise ValueError("method/cap aggregate contains inconsistent audit metadata")
        queries = tuple(row.actual_queries for row in group)
        attempts = len(group)
        certified_exact = sum(row.certified_exact for row in group)
        aggregates.append(
            CCFAMatchedAggregate(
                method_id=method_id,
                query_cap=query_cap,
                comparison_group=next(iter(comparisons)),
                information_regime=next(iter(information)),
                attempts=attempts,
                exact_count=sum(row.exact for row in group),
                certified_count=sum(row.certified for row in group),
                certified_exact_count=certified_exact,
                timeout_count=sum(row.timeout for row in group),
                budget_violation_count=sum(not row.budget_valid for row in group),
                incomplete_count=sum(not row.complete for row in group),
                cleanup_failure_count=sum(row.cleanup_passed is False for row in group),
                cleanup_not_applicable_count=sum(row.cleanup_passed is None for row in group),
                direct_multi_output=next(iter(direct)),
                exact_rate=sum(row.exact for row in group) / attempts,
                certified_exact_rate=certified_exact / attempts,
                mean_actual_queries=statistics.fmean(queries),
                median_actual_queries=float(statistics.median(queries)),
                mean_selection_queries=statistics.fmean(row.selection_queries for row in group),
                mean_verification_queries=statistics.fmean(
                    row.verification_queries for row in group
                ),
            )
        )
    return tuple(aggregates)


def matched_campaign_manifest_hash(
    instances: Sequence[FrozenQuantumReferenceInstance],
    config: CCFAMatchedBenchmarkConfig,
) -> str:
    """Hash only preregistered inputs; no algorithm outcome is accepted."""

    if not isinstance(config, CCFAMatchedBenchmarkConfig):
        raise TypeError("config must be a CCFAMatchedBenchmarkConfig")
    frozen_instances = tuple(instances)
    if not frozen_instances:
        raise ValueError("instances cannot be empty")
    if any(not isinstance(row, FrozenQuantumReferenceInstance) for row in frozen_instances):
        raise TypeError("instances must contain FrozenQuantumReferenceInstance values")
    return _canonical_hash(
        {
            "schema": "qgapselect.ccfa-matched-campaign.v1",
            "instances": [instance.public_document() for instance in frozen_instances],
            "config": config.as_dict(),
        }
    )


__all__ = [
    "BACKEND",
    "CLAIM_SCOPE",
    "COHERENT_METHOD_ID",
    "KNOWN_TIME_INFORMATION_REGIME",
    "KNOWN_TIME_METHOD_ID",
    "K_ONLY_INFORMATION_REGIME",
    "PRIMARY_COMPARISON_GROUP",
    "PRIMARY_METHOD_IDS",
    "SEED_DERIVATION",
    "SEED_NAMESPACE",
    "STRONGER_INFORMATION_GROUP",
    "AtomicQueryCapOracle",
    "CCFAMatchedAggregate",
    "CCFAMatchedBenchmarkConfig",
    "CCFAMatchedTrialRecord",
    "CappedCoherentExecution",
    "KnownTimeControlSpec",
    "aggregate_ccfa_matched_trials",
    "iter_ccfa_matched_trials",
    "matched_campaign_manifest_hash",
    "run_capped_coherent_activity_history",
    "validate_complete_matched_panel",
]
