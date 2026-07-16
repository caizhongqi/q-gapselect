"""Executable variable-time activity-history control-flow core.

The old activity-history prototypes in this repository either received the
activity rows from a harness or evaluated only a resource formula.  This
module instead constructs every history row from *executed* amplitude-
estimation experiments against the public oracle capability.  It provides:

* level-adaptive unknown-boundary confidence elimination;
* explicit active, stop, output, history, phase, and workspace registers;
* a reversible compute--copy/phase--uncompute finite-state circuit IR;
* heterogeneous branch stopping and per-branch executed query ledgers;
* direct extraction of every output born at a level; and
* an independent, fresh all-arm verification pass before issuing a strict
  Top-k certificate.

The backend is a classical finite-state executor over analytically sampled
Grover experiments.  It does not execute a coherent index-register oracle,
does not run on quantum hardware, and does not establish a variable-time
query upper bound.  Candidate circuit depth/qubit counts are therefore kept
in a type separate from actually executed oracle calls.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .estimators import AnalyticIterativeAmplitudeEstimator
from .models import IAEConfig
from .oracles import QueryLedger, QuerySnapshot

BACKEND = "finite_state_coherent_control_ir_over_analytic_grover_measurements"
CLAIM_SCOPE = (
    "executable_activity_history_semantics_no_hardware_"
    "no_coherent_index_oracle_no_advantage_theorem"
)
PROOF_STATUS = "execution_audit_only_no_upper_bound_or_lower_bound_theorem"


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


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _open_unit(value: object, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _immutable_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _zero_counts() -> Mapping[str, int]:
    return _immutable_counts(
        {
            "forward": 0,
            "inverse": 0,
            "controlled_forward": 0,
            "controlled_inverse": 0,
            "classical_sample": 0,
            "coherent_total": 0,
            "classical_total": 0,
            "total": 0,
        }
    )


def _add_counts(*rows: Mapping[str, int]) -> Mapping[str, int]:
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    return _immutable_counts({key: sum(int(row.get(key, 0)) for row in rows) for key in keys})


def _ceil_log2_dimension(size: int) -> int:
    return max(1, int(math.ceil(math.log2(max(2, size)))))


@runtime_checkable
class ActivityHistoryOracleProtocol(Protocol):
    """Only the blind algorithm-side capability used by this core."""

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


class HistoryOutput(str, Enum):
    """Durable two-bit output register values."""

    UNRESOLVED = "unresolved"
    SELECTED = "selected"
    REJECTED = "rejected"


class HistoryIROp(str, Enum):
    """Operations emitted by the reversible finite-state executor."""

    PREPARE_ACTIVE_INDEX = "prepare_active_index"
    CONTROLLED_AMPLITUDE_ESTIMATION = "controlled_amplitude_estimation"
    COMPUTE_UNKNOWN_BOUNDARY_PREDICATE = "compute_unknown_boundary_predicate"
    XOR_ACTIVITY_HISTORY = "xor_activity_history"
    COPY_STOP_OUTPUT = "copy_stop_output"
    COMPUTE_SELECTED_PHASE_PREDICATE = "compute_selected_phase_predicate"
    PHASE_SELECTED_OUTPUT = "phase_selected_output"
    UNCOMPUTE_SELECTED_PHASE_PREDICATE = (
        "uncompute_selected_phase_predicate"
    )
    UNCOMPUTE_UNKNOWN_BOUNDARY_PREDICATE = (
        "uncompute_unknown_boundary_predicate"
    )
    UPDATE_ACTIVE_REGISTER = "update_active_register"
    DIRECT_MULTI_OUTPUT_EMIT = "direct_multi_output_emit"
    FRESH_VERIFY = "fresh_verify"


@dataclass(frozen=True, slots=True)
class VariableTimeHistoryConfig:
    """Execution and risk-allocation configuration for the history core."""

    confidence: float = 0.05
    initial_angular_precision: float = 0.20
    precision_decay: float = 0.5
    max_levels: int = 5
    shots_per_iae_round: int = 96
    iae_max_rounds: int = 7
    iae_max_grover_power: int = 63
    iae_grid_points: int = 4097
    verification_angular_precision: float = 0.01
    verification_precision_decay: float = 0.5
    verification_max_levels: int = 1
    verification_shots_per_round: int = 128
    verification_max_rounds: int = 8
    verification_max_grover_power: int = 127
    verification_grid_points: int = 8193
    certificate_mode: str = "fresh"

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _open_unit(self.confidence, "confidence"))
        initial = _positive(
            self.initial_angular_precision, "initial_angular_precision"
        )
        if initial >= math.pi / 2.0:
            raise ValueError("initial_angular_precision must be below pi/2")
        object.__setattr__(self, "initial_angular_precision", initial)
        decay = _open_unit(self.precision_decay, "precision_decay")
        object.__setattr__(self, "precision_decay", decay)
        object.__setattr__(self, "max_levels", _integer(self.max_levels, "max_levels", minimum=1))
        for name in (
            "shots_per_iae_round",
            "iae_max_rounds",
            "verification_shots_per_round",
            "verification_max_rounds",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        for name in ("iae_max_grover_power", "verification_max_grover_power"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in ("iae_grid_points", "verification_grid_points"):
            value = _integer(getattr(self, name), name, minimum=257)
            object.__setattr__(self, name, value)
        verify = _positive(
            self.verification_angular_precision,
            "verification_angular_precision",
        )
        if verify >= math.pi / 2.0:
            raise ValueError("verification_angular_precision must be below pi/2")
        object.__setattr__(self, "verification_angular_precision", verify)
        verify_decay = _open_unit(
            self.verification_precision_decay,
            "verification_precision_decay",
        )
        object.__setattr__(self, "verification_precision_decay", verify_decay)
        object.__setattr__(
            self,
            "verification_max_levels",
            _integer(self.verification_max_levels, "verification_max_levels", minimum=1),
        )
        if self.certificate_mode not in {"fresh", "history"}:
            raise ValueError("certificate_mode must be 'fresh' or 'history'")

    def angular_precision(self, level: int) -> float:
        level = _integer(level, "level")
        if level >= self.max_levels:
            raise IndexError("level is outside the configured history")
        return self.initial_angular_precision * self.precision_decay**level

    def selection_call_confidence(self, level: int, n_arms: int) -> float:
        """Summable per-arm allocation with total selection risk <= delta/2."""

        level = _integer(level, "level")
        n_arms = _integer(n_arms, "n_arms", minimum=1)
        return 3.0 * self.confidence / (
            math.pi**2 * n_arms * (level + 1) ** 2
        )

    def verification_call_confidence(self, n_arms: int) -> float:
        n_arms = _integer(n_arms, "n_arms", minimum=1)
        return self.confidence / (2.0 * n_arms)

    def verification_level_angular_precision(self, level: int) -> float:
        level = _integer(level, "level")
        if level >= self.verification_max_levels:
            raise IndexError("level is outside the configured verification history")
        return self.verification_angular_precision * self.verification_precision_decay**level

    def verification_level_call_confidence(self, level: int, n_arms: int) -> float:
        """Summable adaptive-verification allocation bounded by delta/(2n)."""

        level = _integer(level, "level")
        n_arms = _integer(n_arms, "n_arms", minimum=1)
        if level >= self.verification_max_levels:
            raise IndexError("level is outside the configured verification history")
        if self.verification_max_levels == 1:
            return self.verification_call_confidence(n_arms)
        return self.confidence / (
            2.0 * n_arms * (level + 1) * (level + 2)
        )


@dataclass(frozen=True, slots=True)
class HistoryIRInstruction:
    """One auditable circuit-IR instruction and its compiled cost."""

    sequence: int
    level: int
    operation: HistoryIROp
    controls: tuple[str, ...]
    targets: tuple[str, ...]
    affected_arms: tuple[int, ...]
    logical_gate_count: int
    candidate_depth: int
    oracle_query_counts: Mapping[str, int]
    inverse_of_sequence: int | None = None
    information_source: str = "executed_confidence_intervals"


@dataclass(frozen=True, slots=True)
class HistoryArmEstimate:
    """One measured level estimate with an exact executed ledger."""

    arm: int
    estimate: float
    mean_interval: tuple[float, float]
    angular_interval: tuple[float, float]
    target_angular_precision: float
    allocated_failure_probability: float
    grover_experiments: int
    measurement_shots: int
    query_counts: Mapping[str, int]
    numerical_warning: str | None


@dataclass(frozen=True, slots=True)
class HistoryBranchRegister:
    """Final durable registers for one coherent index branch."""

    arm: int
    active: bool
    stop_code: int
    output: HistoryOutput
    activity_history: tuple[bool, ...]
    selected_phase_parity: int
    selection_query_counts: Mapping[str, int]
    verification_query_counts: Mapping[str, int]
    predicate_workspace_zero: bool
    phase_workspace_zero: bool


@dataclass(frozen=True, slots=True)
class HistoryLayerExecution:
    """One adaptive precision layer and its measured output births."""

    level: int
    angular_precision: float
    allocated_failure_probability_per_arm: float
    active_before: tuple[int, ...]
    remaining_topk_quota: int
    estimates: tuple[HistoryArmEstimate, ...]
    selected_births: tuple[int, ...]
    rejected_births: tuple[int, ...]
    active_after: tuple[int, ...]
    empirical_boundary_interval: tuple[float, float] | None
    query_counts: Mapping[str, int]
    serial_branch_query_calls: int
    max_branch_query_calls: int
    predicate_workspace_residual: int
    phase_workspace_residual: int
    instructions: tuple[HistoryIRInstruction, ...]

    @property
    def cleanup_passed(self) -> bool:
        return (
            self.predicate_workspace_residual == 0
            and self.phase_workspace_residual == 0
        )


@dataclass(frozen=True, slots=True)
class FreshVerificationRecord:
    """Independent all-arm verification, never reusing selection evidence."""

    selected: tuple[int, ...]
    rejected: tuple[int, ...]
    estimates: tuple[HistoryArmEstimate, ...]
    minimum_selected_lower: float
    maximum_rejected_upper: float
    strict_margin: float
    allocated_failure_probability_per_arm: float
    query_counts: Mapping[str, int]
    passed: bool
    status: str
    levels_executed: int = 1
    refined_arms_by_level: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class StrictTopKHistoryCertificate:
    """Certificate gated on complete extraction and fresh strict separation."""

    selected: tuple[int, ...]
    rejected: tuple[int, ...]
    verification_margin: float
    global_failure_probability: float
    allocated_selection_risk_upper_bound: float
    allocated_verification_risk: float
    evidence_source: str = "fresh_all_arm_simultaneous_confidence_intervals"
    transcript_replay_verified: bool = True


@dataclass(frozen=True, slots=True)
class ExecutedHistoryResources:
    """Only resources actually charged by the analytic executor."""

    query_counts: Mapping[str, int]
    selection_query_counts: Mapping[str, int]
    verification_query_counts: Mapping[str, int]
    selection_query_counts_by_arm: Mapping[int, Mapping[str, int]]
    verification_query_counts_by_arm: Mapping[int, Mapping[str, int]]
    selection_query_counts_by_level: Mapping[int, Mapping[str, int]]
    estimator_calls: int
    grover_experiments: int
    measurement_shots: int
    levels_executed: int
    stopped_branches: int
    unresolved_branches: int

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("total", 0))


@dataclass(frozen=True, slots=True)
class CandidateCircuitIRResources:
    """Compiled IR accounting; deliberately not an executed-query theorem."""

    gate_counts: Mapping[str, int]
    serial_finite_state_gate_count: int
    candidate_coherent_scheduled_depth: int
    candidate_index_qubits: int
    candidate_level_qubits: int
    candidate_stop_qubits: int
    candidate_activity_history_qubits: int
    candidate_output_qubits: int
    candidate_phase_qubits: int
    candidate_precision_qubits: int
    candidate_workspace_qubits: int
    candidate_total_qubits: int
    peak_live_branches: int
    no_free_qram: bool
    membership_compilation: str
    cleanup_verified: bool
    proof_status: str = PROOF_STATUS


@dataclass(frozen=True, slots=True)
class VariableTimeCoherentHistoryResult:
    """Complete execution transcript with strictly separated claim surfaces."""

    k: int
    n_arms: int
    extracted_selected: tuple[int, ...]
    extracted_rejected: tuple[int, ...]
    unresolved: tuple[int, ...]
    complete: bool
    certified: bool
    certificate: StrictTopKHistoryCertificate | None
    verification: FreshVerificationRecord | None
    layers: tuple[HistoryLayerExecution, ...]
    branches: tuple[HistoryBranchRegister, ...]
    executed_resources: ExecutedHistoryResources
    candidate_ir_resources: CandidateCircuitIRResources
    status: str
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE
    hardware_claimable: bool = False
    coherent_query_advantage_claimable: bool = False


@dataclass(slots=True)
class _MutableBranch:
    arm: int
    active: bool = True
    stop_code: int = 0
    output: HistoryOutput = HistoryOutput.UNRESOLVED
    activity_history: list[bool] | None = None
    selected_phase_parity: int = 0
    predicate_workspace: int = 0
    phase_workspace: int = 0

    def __post_init__(self) -> None:
        if self.activity_history is None:
            self.activity_history = []


class VariableTimeCoherentActivityHistoryCore:
    """Execute adaptive activity-history semantics from blind oracle access."""

    def __init__(
        self,
        oracle: ActivityHistoryOracleProtocol,
        k: int,
        *,
        config: VariableTimeHistoryConfig | None = None,
    ) -> None:
        if not isinstance(oracle, ActivityHistoryOracleProtocol):
            raise TypeError(
                "oracle must expose n_arms, query_snapshot, and "
                "run_grover_experiment"
            )
        n_arms = _integer(oracle.n_arms, "oracle.n_arms", minimum=2)
        k = _integer(k, "k", minimum=1)
        if k >= n_arms:
            raise ValueError("k must satisfy 1 <= k < oracle.n_arms")
        if config is not None and not isinstance(config, VariableTimeHistoryConfig):
            raise TypeError("config must be a VariableTimeHistoryConfig")
        self.oracle = oracle
        self.n_arms = n_arms
        self.k = k
        self.config = config if config is not None else VariableTimeHistoryConfig()

    @staticmethod
    def _classify(
        estimates: Mapping[int, HistoryArmEstimate],
        quota: int,
    ) -> tuple[set[int], set[int]]:
        arms = tuple(estimates)
        if quota <= 0:
            return set(), set(arms)
        if quota >= len(arms):
            return set(arms), set()
        accepted: set[int] = set()
        rejected: set[int] = set()
        for arm in arms:
            lower, upper = estimates[arm].mean_interval
            possible_above = sum(
                estimates[other].mean_interval[1] >= lower
                for other in arms
                if other != arm
            )
            if possible_above < quota:
                accepted.add(arm)
                continue
            certainly_above = sum(
                estimates[other].mean_interval[0] > upper
                for other in arms
                if other != arm
            )
            if certainly_above >= quota:
                rejected.add(arm)
        return accepted, rejected

    def _estimate_arm(
        self,
        arm: int,
        *,
        angular_precision: float,
        confidence: float,
        verification: bool,
        tag: str,
    ) -> HistoryArmEstimate:
        config = self.config
        iae_config = IAEConfig(
            target_angular_precision=angular_precision,
            confidence=confidence,
            shots_per_round=(
                config.verification_shots_per_round
                if verification
                else config.shots_per_iae_round
            ),
            max_rounds=(
                config.verification_max_rounds
                if verification
                else config.iae_max_rounds
            ),
            max_grover_power=(
                config.verification_max_grover_power
                if verification
                else config.iae_max_grover_power
            ),
            grid_points=(
                config.verification_grid_points
                if verification
                else config.iae_grid_points
            ),
        )
        estimate = AnalyticIterativeAmplitudeEstimator(iae_config).estimate(
            self.oracle,  # type: ignore[arg-type]
            arm,
            confidence=confidence,
            target_angular_precision=angular_precision,
            tag=tag,
        )
        return HistoryArmEstimate(
            arm=arm,
            estimate=estimate.estimate,
            mean_interval=(estimate.interval.lower, estimate.interval.upper),
            angular_interval=(
                estimate.angular_interval.lower,
                estimate.angular_interval.upper,
            ),
            target_angular_precision=angular_precision,
            allocated_failure_probability=confidence,
            grover_experiments=len(estimate.observations),
            measurement_shots=sum(row.shots for row in estimate.observations),
            query_counts=_immutable_counts(estimate.executed_query_counts),
            numerical_warning=estimate.numerical_warning,
        )

    @staticmethod
    def _empirical_boundary(
        estimates: Mapping[int, HistoryArmEstimate], quota: int
    ) -> tuple[float, float] | None:
        if not 0 < quota < len(estimates):
            return None
        ranking = sorted(estimates, key=lambda arm: (-estimates[arm].estimate, arm))
        inside = ranking[:quota]
        outside = ranking[quota:]
        return (
            max(estimates[arm].mean_interval[1] for arm in outside),
            min(estimates[arm].mean_interval[0] for arm in inside),
        )

    def _execute_register_layer(
        self,
        branches: list[_MutableBranch],
        *,
        level: int,
        active_before: tuple[int, ...],
        selected_births: tuple[int, ...],
        rejected_births: tuple[int, ...],
        query_counts: Mapping[str, int],
        max_branch_queries: int,
        sequence_start: int,
    ) -> tuple[tuple[HistoryIRInstruction, ...], int, int]:
        active_set = frozenset(active_before)
        selected_set = frozenset(selected_births)
        rejected_set = frozenset(rejected_births)
        decided_set = selected_set | rejected_set
        instructions: list[HistoryIRInstruction] = []
        sequence = sequence_start

        def emit(
            operation: HistoryIROp,
            *,
            controls: tuple[str, ...],
            targets: tuple[str, ...],
            affected: tuple[int, ...],
            gates: int,
            depth: int,
            queries: Mapping[str, int] | None = None,
            inverse_of: int | None = None,
        ) -> int:
            nonlocal sequence
            current = sequence
            instructions.append(
                HistoryIRInstruction(
                    sequence=current,
                    level=level,
                    operation=operation,
                    controls=controls,
                    targets=targets,
                    affected_arms=affected,
                    logical_gate_count=gates,
                    candidate_depth=depth,
                    oracle_query_counts=(
                        _zero_counts() if queries is None else _immutable_counts(queries)
                    ),
                    inverse_of_sequence=inverse_of,
                )
            )
            sequence += 1
            return current

        emit(
            HistoryIROp.PREPARE_ACTIVE_INDEX,
            controls=("active",),
            targets=("index",),
            affected=active_before,
            gates=max(1, len(active_before)),
            depth=1,
        )
        emit(
            HistoryIROp.CONTROLLED_AMPLITUDE_ESTIMATION,
            controls=("index", "active"),
            targets=("precision", "confidence_interval"),
            affected=active_before,
            gates=0,
            depth=max_branch_queries,
            queries=query_counts,
        )
        compute_sequence = emit(
            HistoryIROp.COMPUTE_UNKNOWN_BOUNDARY_PREDICATE,
            controls=("index", "confidence_interval", "remaining_quota"),
            targets=("predicate_workspace",),
            affected=tuple(sorted(decided_set)),
            # Equality controls are compiled explicitly.  No membership QRAM
            # or unit-cost table lookup is assumed.
            gates=len(decided_set) * (2 * _ceil_log2_dimension(self.n_arms) + 3),
            depth=max(1, len(decided_set)),
        )
        emit(
            HistoryIROp.XOR_ACTIVITY_HISTORY,
            controls=("active",),
            targets=(f"activity_history[{level}]",),
            affected=active_before,
            gates=len(active_before),
            depth=1,
        )
        emit(
            HistoryIROp.COPY_STOP_OUTPUT,
            controls=("predicate_workspace",),
            targets=("stop", "output"),
            affected=tuple(sorted(decided_set)),
            gates=3 * len(decided_set),
            depth=3,
        )
        phase_compute_sequence = emit(
            HistoryIROp.COMPUTE_SELECTED_PHASE_PREDICATE,
            controls=("predicate_workspace:selected",),
            targets=("phase_workspace",),
            affected=selected_births,
            gates=len(selected_births),
            depth=1 if selected_births else 0,
        )
        emit(
            HistoryIROp.PHASE_SELECTED_OUTPUT,
            controls=("phase_workspace",),
            targets=("phase",),
            affected=selected_births,
            gates=len(selected_births),
            depth=1 if selected_births else 0,
        )
        emit(
            HistoryIROp.UNCOMPUTE_SELECTED_PHASE_PREDICATE,
            controls=("predicate_workspace:selected",),
            targets=("phase_workspace",),
            affected=selected_births,
            gates=len(selected_births),
            depth=1 if selected_births else 0,
            inverse_of=phase_compute_sequence,
        )
        emit(
            HistoryIROp.UNCOMPUTE_UNKNOWN_BOUNDARY_PREDICATE,
            controls=("index", "confidence_interval", "remaining_quota"),
            targets=("predicate_workspace", "phase_workspace"),
            affected=tuple(sorted(decided_set)),
            gates=(
                len(decided_set) * (2 * _ceil_log2_dimension(self.n_arms) + 3)
            ),
            depth=max(1, len(decided_set)),
            inverse_of=compute_sequence,
        )
        emit(
            HistoryIROp.UPDATE_ACTIVE_REGISTER,
            controls=("output",),
            targets=("active",),
            affected=tuple(sorted(decided_set)),
            gates=len(decided_set),
            depth=1,
        )
        emit(
            HistoryIROp.DIRECT_MULTI_OUTPUT_EMIT,
            controls=("stop", "output"),
            targets=("output_tape",),
            affected=tuple(sorted(decided_set)),
            gates=len(decided_set) * _ceil_log2_dimension(self.n_arms),
            depth=1 if decided_set else 0,
        )

        # Execute the finite-state register relation on every index branch.
        for branch in branches:
            was_active = branch.arm in active_set
            branch.activity_history.append(was_active)
            if branch.arm not in decided_set:
                continue
            if not branch.active or branch.output is not HistoryOutput.UNRESOLVED:
                raise RuntimeError("a stopped branch was targeted by a later layer")
            branch.predicate_workspace ^= 1
            if branch.arm in selected_set:
                branch.output = HistoryOutput.SELECTED
                branch.selected_phase_parity ^= 1
                branch.phase_workspace ^= 1
                branch.phase_workspace ^= 1
            else:
                branch.output = HistoryOutput.REJECTED
            branch.stop_code ^= level + 1
            branch.predicate_workspace ^= 1
            branch.active = False

        predicate_residual = sum(branch.predicate_workspace for branch in branches)
        phase_residual = sum(branch.phase_workspace for branch in branches)
        return tuple(instructions), predicate_residual, phase_residual

    def _fresh_verify(
        self,
        selected: tuple[int, ...],
    ) -> FreshVerificationRecord:
        selected_set = frozenset(selected)
        rejected = tuple(arm for arm in range(self.n_arms) if arm not in selected_set)
        before = self.oracle.query_snapshot()
        estimates: list[HistoryArmEstimate] = []
        by_arm: dict[int, HistoryArmEstimate] = {}
        refined_arms_by_level: list[tuple[int, ...]] = []
        arms_to_refine = tuple(range(self.n_arms))
        passed = False
        margin = float("-inf")
        min_selected = 0.0
        max_rejected = 1.0
        for level in range(self.config.verification_max_levels):
            if not arms_to_refine:
                break
            refined_arms_by_level.append(arms_to_refine)
            per_call_confidence = self.config.verification_level_call_confidence(
                level, self.n_arms
            )
            precision = self.config.verification_level_angular_precision(level)
            for arm in arms_to_refine:
                tag = (
                    f"vt_history_fresh_verify_arm_{arm}"
                    if level == 0
                    else f"vt_history_fresh_verify_level_{level}_arm_{arm}"
                )
                record = self._estimate_arm(
                    arm,
                    angular_precision=precision,
                    confidence=per_call_confidence,
                    verification=True,
                    tag=tag,
                )
                estimates.append(record)
                by_arm[arm] = record

            min_selected = min(by_arm[arm].mean_interval[0] for arm in selected)
            max_rejected = max(by_arm[arm].mean_interval[1] for arm in rejected)
            margin = min_selected - max_rejected
            passed = margin > 0.0
            if passed:
                break
            ambiguous_selected = tuple(
                arm
                for arm in selected
                if by_arm[arm].mean_interval[0] <= max_rejected
            )
            ambiguous_rejected = tuple(
                arm
                for arm in rejected
                if by_arm[arm].mean_interval[1] >= min_selected
            )
            arms_to_refine = tuple(
                sorted({*ambiguous_selected, *ambiguous_rejected})
            )

        per_arm_confidence = self.config.verification_call_confidence(self.n_arms)
        min_selected = min(by_arm[arm].mean_interval[0] for arm in selected)
        max_rejected = max(by_arm[arm].mean_interval[1] for arm in rejected)
        margin = min_selected - max_rejected
        return FreshVerificationRecord(
            selected=selected,
            rejected=rejected,
            estimates=tuple(estimates),
            minimum_selected_lower=min_selected,
            maximum_rejected_upper=max_rejected,
            strict_margin=margin,
            allocated_failure_probability_per_arm=per_arm_confidence,
            query_counts=_immutable_counts(
                QueryLedger.difference(self.oracle.query_snapshot(), before)
            ),
            passed=passed,
            status=(
                "fresh_adaptive_strict_interval_separation"
                if passed
                else "fresh_adaptive_verification_not_separated"
            ),
            levels_executed=len(refined_arms_by_level),
            refined_arms_by_level=tuple(refined_arms_by_level),
        )

    def _selection_history_replays(
        self,
        layers: list[HistoryLayerExecution],
        *,
        extracted_selected: tuple[int, ...],
        extracted_rejected: tuple[int, ...],
        unresolved: tuple[int, ...],
    ) -> bool:
        """Replay every confidence decision and deterministic quota closure."""

        selected: set[int] = set()
        rejected: set[int] = set()
        active: set[int] = set(range(self.n_arms))
        for layer in layers:
            if layer.active_before != tuple(sorted(active)):
                return False
            quota = self.k - len(selected)
            if layer.remaining_topk_quota != quota:
                return False
            estimates = {record.arm: record for record in layer.estimates}
            if set(estimates) != active:
                return False
            accepted, newly_rejected = self._classify(estimates, quota)
            remaining = active - accepted - newly_rejected
            if len(selected) + len(accepted) == self.k:
                newly_rejected.update(remaining)
            elif len(selected) + len(accepted) + len(remaining) == self.k:
                accepted.update(remaining)
            if tuple(sorted(accepted)) != layer.selected_births:
                return False
            if tuple(sorted(newly_rejected)) != layer.rejected_births:
                return False
            selected.update(accepted)
            rejected.update(newly_rejected)
            active.difference_update(accepted | newly_rejected)
            if tuple(sorted(active)) != layer.active_after:
                return False
        return (
            tuple(sorted(selected)) == extracted_selected
            and tuple(sorted(rejected)) == extracted_rejected
            and tuple(sorted(active)) == unresolved
        )

    def run(self) -> VariableTimeCoherentHistoryResult:
        before_run = self.oracle.query_snapshot()
        branches = [_MutableBranch(arm=arm) for arm in range(self.n_arms)]
        selected: set[int] = set()
        rejected: set[int] = set()
        active: set[int] = set(range(self.n_arms))
        layers: list[HistoryLayerExecution] = []
        selection_by_arm: dict[int, Mapping[str, int]] = {
            arm: _zero_counts() for arm in range(self.n_arms)
        }
        selection_by_level: dict[int, Mapping[str, int]] = {}
        sequence = 0

        for level in range(self.config.max_levels):
            if not active:
                break
            quota = self.k - len(selected)
            active_before = tuple(sorted(active))
            precision = self.config.angular_precision(level)
            per_arm_confidence = self.config.selection_call_confidence(
                level, self.n_arms
            )
            before_level = self.oracle.query_snapshot()
            estimates: dict[int, HistoryArmEstimate] = {}
            for arm in active_before:
                record = self._estimate_arm(
                    arm,
                    angular_precision=precision,
                    confidence=per_arm_confidence,
                    verification=False,
                    tag=f"vt_history_level_{level}_arm_{arm}",
                )
                estimates[arm] = record
                selection_by_arm[arm] = _add_counts(
                    selection_by_arm[arm], record.query_counts
                )

            accepted, newly_rejected = self._classify(estimates, quota)
            if len(accepted) > quota:
                raise RuntimeError("confidence classifier overfilled the Top-k quota")
            # Once one side is complete, the remaining membership follows
            # logically without another oracle call.
            remaining_after_decisions = active - accepted - newly_rejected
            if len(selected) + len(accepted) == self.k:
                newly_rejected.update(remaining_after_decisions)
            elif len(selected) + len(accepted) + len(remaining_after_decisions) == self.k:
                accepted.update(remaining_after_decisions)

            selected_births = tuple(sorted(accepted))
            rejected_births = tuple(sorted(newly_rejected))
            selected.update(accepted)
            rejected.update(newly_rejected)
            active.difference_update(accepted | newly_rejected)
            after_level = self.oracle.query_snapshot()
            level_counts = _immutable_counts(
                QueryLedger.difference(after_level, before_level)
            )
            selection_by_level[level] = level_counts
            max_branch_queries = max(
                (record.query_counts.get("total", 0) for record in estimates.values()),
                default=0,
            )
            instructions, predicate_residual, phase_residual = (
                self._execute_register_layer(
                    branches,
                    level=level,
                    active_before=active_before,
                    selected_births=selected_births,
                    rejected_births=rejected_births,
                    query_counts=level_counts,
                    max_branch_queries=int(max_branch_queries),
                    sequence_start=sequence,
                )
            )
            sequence += len(instructions)
            if predicate_residual or phase_residual:
                raise RuntimeError("compute--phase--uncompute left workspace garbage")
            layers.append(
                HistoryLayerExecution(
                    level=level,
                    angular_precision=precision,
                    allocated_failure_probability_per_arm=per_arm_confidence,
                    active_before=active_before,
                    remaining_topk_quota=quota,
                    estimates=tuple(estimates[arm] for arm in active_before),
                    selected_births=selected_births,
                    rejected_births=rejected_births,
                    active_after=tuple(sorted(active)),
                    empirical_boundary_interval=self._empirical_boundary(
                        estimates, quota
                    ),
                    query_counts=level_counts,
                    serial_branch_query_calls=int(level_counts.get("total", 0)),
                    max_branch_query_calls=int(max_branch_queries),
                    predicate_workspace_residual=predicate_residual,
                    phase_workspace_residual=phase_residual,
                    instructions=instructions,
                )
            )

        # Pad durable history registers to their declared fixed width.
        for branch in branches:
            assert branch.activity_history is not None
            branch.activity_history.extend(
                [False] * (self.config.max_levels - len(branch.activity_history))
            )

        extracted_selected = tuple(sorted(selected))
        extracted_rejected = tuple(sorted(rejected))
        unresolved = tuple(sorted(active))
        complete = not unresolved and len(extracted_selected) == self.k
        history_replay_verified = self._selection_history_replays(
            layers,
            extracted_selected=extracted_selected,
            extracted_rejected=extracted_rejected,
            unresolved=unresolved,
        )
        after_selection = self.oracle.query_snapshot()
        selection_counts = _immutable_counts(
            QueryLedger.difference(after_selection, before_run)
        )

        verification = (
            self._fresh_verify(extracted_selected)
            if complete and self.config.certificate_mode == "fresh"
            else None
        )
        after_run = self.oracle.query_snapshot()
        verification_counts = (
            _zero_counts() if verification is None else verification.query_counts
        )
        total_counts = _immutable_counts(QueryLedger.difference(after_run, before_run))

        verification_by_arm: dict[int, Mapping[str, int]] = {
            arm: _zero_counts() for arm in range(self.n_arms)
        }
        if verification is not None:
            for record in verification.estimates:
                verification_by_arm[record.arm] = _add_counts(
                    verification_by_arm[record.arm], record.query_counts
                )

        if sum(row.get("total", 0) for row in selection_by_arm.values()) != int(
            selection_counts.get("total", 0)
        ):
            raise RuntimeError("per-arm selection ledger does not match total")
        if sum(row.get("total", 0) for row in verification_by_arm.values()) != int(
            verification_counts.get("total", 0)
        ):
            raise RuntimeError("per-arm verification ledger does not match total")
        if int(selection_counts.get("total", 0)) + int(
            verification_counts.get("total", 0)
        ) != int(total_counts.get("total", 0)):
            raise RuntimeError("selection and verification ledgers do not partition run")

        allocated_selection_risk = sum(
            len(layer.active_before) * layer.allocated_failure_probability_per_arm
            for layer in layers
        )
        allocated_verification_risk = (
            0.0
            if verification is None
            else self.n_arms * verification.allocated_failure_probability_per_arm
        )
        cleanup_verified = all(layer.cleanup_passed for layer in layers)
        certified = bool(
            complete
            and history_replay_verified
            and cleanup_verified
            and (
                self.config.certificate_mode == "history"
                or (verification is not None and verification.passed)
            )
        )
        certificate = (
            StrictTopKHistoryCertificate(
                selected=extracted_selected,
                rejected=extracted_rejected,
                verification_margin=(
                    0.0 if verification is None else verification.strict_margin
                ),
                global_failure_probability=self.config.confidence,
                allocated_selection_risk_upper_bound=allocated_selection_risk,
                allocated_verification_risk=allocated_verification_risk,
                evidence_source=(
                    "summable_selection_history_and_verified_quota_closure"
                    if self.config.certificate_mode == "history"
                    else (
                        "fresh_boundary_adaptive_simultaneous_confidence_intervals"
                        if verification is not None and verification.levels_executed > 1
                        else "fresh_all_arm_simultaneous_confidence_intervals"
                    )
                ),
                transcript_replay_verified=history_replay_verified,
            )
            if certified
            else None
        )
        if allocated_selection_risk > self.config.confidence / 2.0 + 1e-15:
            raise RuntimeError("selection risk allocation exceeded delta/2")
        if allocated_verification_risk > self.config.confidence / 2.0 + 1e-15:
            raise RuntimeError("verification risk allocation exceeded delta/2")

        gate_counter: Counter[str] = Counter()
        serial_gate_count = 0
        candidate_depth = 0
        for layer in layers:
            candidate_depth += sum(row.candidate_depth for row in layer.instructions)
            for row in layer.instructions:
                gate_counter[row.operation.value] += row.logical_gate_count
                serial_gate_count += row.logical_gate_count
        if verification is not None:
            gate_counter[HistoryIROp.FRESH_VERIFY.value] += self.n_arms

        index_qubits = _ceil_log2_dimension(self.n_arms)
        level_qubits = _ceil_log2_dimension(self.config.max_levels + 1)
        stop_qubits = level_qubits
        history_qubits = self.config.max_levels
        output_qubits = 2
        phase_qubits = 1
        precision_qubits = _ceil_log2_dimension(
            max(self.config.iae_grid_points, self.config.verification_grid_points)
        )
        workspace_qubits = 2 * precision_qubits + index_qubits + 5
        total_qubits = (
            index_qubits
            + level_qubits
            + stop_qubits
            + history_qubits
            + output_qubits
            + phase_qubits
            + precision_qubits
            + workspace_qubits
        )

        branch_results = tuple(
            HistoryBranchRegister(
                arm=branch.arm,
                active=branch.active,
                stop_code=branch.stop_code,
                output=branch.output,
                activity_history=tuple(branch.activity_history or ()),
                selected_phase_parity=branch.selected_phase_parity,
                selection_query_counts=selection_by_arm[branch.arm],
                verification_query_counts=verification_by_arm[branch.arm],
                predicate_workspace_zero=branch.predicate_workspace == 0,
                phase_workspace_zero=branch.phase_workspace == 0,
            )
            for branch in branches
        )
        estimator_calls = sum(len(layer.estimates) for layer in layers) + (
            0 if verification is None else len(verification.estimates)
        )
        all_estimates = [record for layer in layers for record in layer.estimates]
        if verification is not None:
            all_estimates.extend(verification.estimates)
        executed = ExecutedHistoryResources(
            query_counts=total_counts,
            selection_query_counts=selection_counts,
            verification_query_counts=verification_counts,
            selection_query_counts_by_arm=MappingProxyType(dict(selection_by_arm)),
            verification_query_counts_by_arm=MappingProxyType(
                dict(verification_by_arm)
            ),
            selection_query_counts_by_level=MappingProxyType(
                dict(selection_by_level)
            ),
            estimator_calls=estimator_calls,
            grover_experiments=sum(row.grover_experiments for row in all_estimates),
            measurement_shots=sum(row.measurement_shots for row in all_estimates),
            levels_executed=len(layers),
            stopped_branches=self.n_arms - len(unresolved),
            unresolved_branches=len(unresolved),
        )
        candidate = CandidateCircuitIRResources(
            gate_counts=MappingProxyType(dict(gate_counter)),
            serial_finite_state_gate_count=serial_gate_count,
            candidate_coherent_scheduled_depth=candidate_depth,
            candidate_index_qubits=index_qubits,
            candidate_level_qubits=level_qubits,
            candidate_stop_qubits=stop_qubits,
            candidate_activity_history_qubits=history_qubits,
            candidate_output_qubits=output_qubits,
            candidate_phase_qubits=phase_qubits,
            candidate_precision_qubits=precision_qubits,
            candidate_workspace_qubits=workspace_qubits,
            candidate_total_qubits=total_qubits,
            peak_live_branches=max(
                (len(layer.active_before) for layer in layers), default=0
            ),
            no_free_qram=True,
            membership_compilation=(
                "explicit_multi_controlled_index_equalities_linear_in_births"
            ),
            cleanup_verified=cleanup_verified,
        )
        if certified and self.config.certificate_mode == "history":
            status = "certified_selection_history_replay"
        elif certified:
            status = "certified_fresh_strict_separation"
        elif complete:
            status = "complete_extraction_fresh_verification_failed"
        else:
            status = "max_levels_unresolved_no_certificate"
        return VariableTimeCoherentHistoryResult(
            k=self.k,
            n_arms=self.n_arms,
            extracted_selected=extracted_selected,
            extracted_rejected=extracted_rejected,
            unresolved=unresolved,
            complete=complete,
            certified=certified,
            certificate=certificate,
            verification=verification,
            layers=tuple(layers),
            branches=branch_results,
            executed_resources=executed,
            candidate_ir_resources=candidate,
            status=status,
        )


def run_variable_time_coherent_activity_history(
    oracle: ActivityHistoryOracleProtocol,
    k: int,
    *,
    config: VariableTimeHistoryConfig | None = None,
) -> VariableTimeCoherentHistoryResult:
    """Convenience entry point for the executable activity-history core."""

    return VariableTimeCoherentActivityHistoryCore(
        oracle, k, config=config
    ).run()


__all__ = [
    "BACKEND",
    "CLAIM_SCOPE",
    "PROOF_STATUS",
    "ActivityHistoryOracleProtocol",
    "CandidateCircuitIRResources",
    "ExecutedHistoryResources",
    "FreshVerificationRecord",
    "HistoryArmEstimate",
    "HistoryBranchRegister",
    "HistoryIRInstruction",
    "HistoryIROp",
    "HistoryLayerExecution",
    "HistoryOutput",
    "StrictTopKHistoryCertificate",
    "VariableTimeCoherentActivityHistoryCore",
    "VariableTimeCoherentHistoryResult",
    "VariableTimeHistoryConfig",
    "run_variable_time_coherent_activity_history",
]
