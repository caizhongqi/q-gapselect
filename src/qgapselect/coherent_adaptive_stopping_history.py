"""Tiny true-coherent two-level stopping-history reference circuit.

Unlike :mod:`qgapselect.adaptive_unknown_boundary_topk`, this module puts the
phase, history, scratch-mask, durable-mask, rank-work, and query-control
registers in one statevector.  It implements a fixed public two-level policy
for ``n=2, k=1``:

* level 0 uses two phase qubits per arm;
* level 1 uses three phase qubits per arm;
* a level stops when the folded discrete phase ranks differ by at least two
  bins; and
* every level-1 oracle call is controlled by the coherent ``not stopped`` flag.

The forward history transducer is followed by a durable scratch-to-output copy
and a full reverse replay.  Exact-grid fixtures clean completely.  Generic
off-grid inputs can entangle the durable mask with transient registers and are
therefore rejected.  Executed controlled-oracle calls are charged at their
worst-case circuit count even when an input branch is inactive.

This is a tiny promise-sensitive unitary semantics artifact.  Its branch-RMS
quantity is recorded only as a target for a future variable-time theorem; it
does not reduce the executed query ledger and is not an upper-bound claim.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .coherent import CanonicalRyStatevectorOracle
from .oracles import QueryLedger, QuerySnapshot

ComplexState = NDArray[np.complex128]

METHOD_ID = "tiny_true_coherent_two_level_stopping_history_v1"
BACKEND = "numpy_exact_statevector_true_coherent_two_level_history"
OUTPUT_MASK = "MASK"
OUTPUT_INCONCLUSIVE = "INCONCLUSIVE"
PHASE_LEVELS = (2, 3)
MINIMUM_BIN_SEPARATION = 2
CLAIM_SCOPE = (
    "tiny_exact_grid_promise_true_coherent_history_and_replay_"
    "no_generic_off_grid_theorem_no_variable_time_speedup_"
    "branch_rms_target_only_no_new_upper_or_lower_bound"
)
QUERY_SEMANTICS = (
    "executed_worst_case_circuit_oracle_calls_including_controlled_calls_on_"
    "zero_amplitude_active_branches"
)
BRANCH_RMS_SEMANTICS = (
    "history_distribution_weighted_target_for_a_future_variable_time_theorem_"
    "not_executed_query_savings_or_a_proved_bound"
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


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite real, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a positive finite real") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real")
    return result


def _frozen_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _frozen_probabilities(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType({str(key): float(value) for key, value in values.items()})


def _probability_complement(value: float) -> float:
    return min(1.0, max(0.0, 1.0 - float(value)))


@dataclass(frozen=True, slots=True)
class TinyCoherentStoppingHistoryConfig:
    """Numerical and statevector caps; the two-level policy is not an input."""

    cleanup_tolerance: float = 1e-10
    max_statevector_dimension: int = 600_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cleanup_tolerance",
            _positive_finite(self.cleanup_tolerance, "cleanup_tolerance"),
        )
        object.__setattr__(
            self,
            "max_statevector_dimension",
            _integer(
                self.max_statevector_dimension,
                "max_statevector_dimension",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class TinyCoherentInputInterface:
    oracle: str
    k: int
    internally_fixed_phase_levels: tuple[int, int]
    internally_fixed_minimum_bin_separation: int
    forbidden_inputs: tuple[str, ...]
    answer_dependent_inputs_supplied: bool = False
    gap_supplied: bool = False
    boundary_supplied: bool = False
    family_label_supplied: bool = False
    schedule_supplied: bool = False
    activity_history_supplied: bool = False


@dataclass(frozen=True, slots=True)
class TinyCoherentHistoryEvidence:
    probabilities: Mapping[str, float]
    stop_at_level_zero_probability: float
    stop_at_level_one_probability: float
    unresolved_probability: float
    invalid_both_stopped_probability: float
    dominant_history: str
    dominant_history_probability: float
    scratch_mask_probabilities: Mapping[str, float]
    single_statevector_history_register: bool = True
    later_level_oracles_controlled_by_active_flag: bool = True


@dataclass(frozen=True, slots=True)
class TinyCoherentCleanupLedger:
    phase_nonzero_probability: float
    index_nonzero_probability: float
    reward_nonzero_probability: float
    history_nonzero_probability: float
    scratch_mask_nonzero_probability: float
    rank_work_nonzero_probability: float
    query_control_nonzero_probability: float
    executed_transient_nonzero_probability: float
    predicted_transient_nonzero_probability: float
    prediction_residual: float
    output_collision_probability: float
    output_reduced_purity: float
    purity_residual: float
    norm_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.executed_transient_nonzero_probability <= self.tolerance
            and self.prediction_residual <= 10.0 * self.tolerance
            and self.purity_residual <= 10.0 * self.tolerance
            and self.norm_error <= self.tolerance
        )


@dataclass(frozen=True, slots=True)
class TinyCoherentQueryLedger:
    query_counts: Mapping[str, int]
    expected_query_counts: Mapping[str, int]
    reconciled: bool
    per_level_runtime_records: tuple[TinyCoherentLevelQueryRecord, ...]
    per_level_one_way_query_costs: tuple[int, int]
    worst_case_one_way_history_queries: int
    worst_case_full_replay_queries: int
    branch_rms_one_way_theorem_target: float
    branch_rms_full_replay_theorem_target: float
    branch_rms_is_executed_saving: bool
    query_semantics: str = QUERY_SEMANTICS
    branch_rms_semantics: str = BRANCH_RMS_SEMANTICS
    qram_assumed: bool = False


@dataclass(frozen=True, slots=True)
class TinyCoherentLevelQueryRecord:
    """Runtime tag-derived full-replay and one-way query reconciliation."""

    level: int
    phase_qubits: int
    tag_prefix: str
    kernel_invocations: int
    runtime_full_replay_counts: Mapping[str, int]
    expected_full_replay_counts: Mapping[str, int]
    runtime_derived_one_way_counts: Mapping[str, int]
    expected_one_way_counts: Mapping[str, int]
    full_replay_reconciled: bool
    one_way_reconciled: bool
    derivation: str = (
        "sum oracle QuerySnapshot.by_tag over both arm tags, then divide exact "
        "even counts by the two executed level-kernel invocations"
    )


@dataclass(frozen=True, slots=True)
class TinyInactiveSubspaceAudit:
    """Clean inactive identity witness and dirty-work negative control."""

    clean_inactive_basis_identity_residual: float
    dirty_rank_work_negative_control_residual: float
    clean_identity_witness_passed: bool
    dirty_negative_control_activated: bool
    clean_query_counts: Mapping[str, int]
    dirty_query_counts: Mapping[str, int]
    clean_query_ledger_reconciled: bool
    dirty_query_ledger_reconciled: bool
    valid_identity_subspace: str
    excluded_dirty_subspace: str
    theorem_status: str = "basis_witness_only_not_a_subspace_proof"


@dataclass(frozen=True, slots=True)
class TinyCoherentDurableOutput:
    status: str
    membership_mask: int | None
    membership_bits: tuple[int, int] | tuple[()]
    output_mask_probabilities: Mapping[str, float]
    dominant_mask: int
    dominant_probability: float
    scratch_to_durable_copy_executed: bool
    full_history_replay_executed: bool
    cleanup_passed: bool


@dataclass(frozen=True, slots=True)
class TinyCoherentCertificate:
    issued: bool
    certificate_type: str | None
    top_k_correctness_error_bound: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class TinyCoherentResources:
    declared_register_qubits: int
    register_dimensions: Mapping[str, int]
    retained_statevector_dimension: int
    estimated_peak_complex_amplitudes: int
    executed_numpy_kernel_macro_counts: Mapping[str, int]
    elementary_gate_ledger_available: bool
    transpiled_depth_available: bool
    compiled_ancilla_qubits_available: bool
    query_ledger: TinyCoherentQueryLedger
    cleanup: TinyCoherentCleanupLedger
    backend: str = BACKEND


@dataclass(frozen=True, slots=True)
class TinyCoherentClaimBoundary:
    supports: tuple[str, ...]
    does_not_support: tuple[str, ...]
    true_coherent_stopping_history_unitary_implemented: bool = True
    durable_copy_and_full_replay_implemented: bool = True
    generic_off_grid_cleanup_proved: bool = False
    variable_time_query_speedup_proved: bool = False
    new_query_upper_bound_proved: bool = False
    matching_lower_bound_proved: bool = False
    quantum_advantage_claimable: bool = False
    ccf_a_claimable: bool = False
    claim_scope: str = CLAIM_SCOPE


@dataclass(frozen=True, slots=True)
class TinyCoherentStoppingHistoryResult:
    method_id: str
    input_interface: TinyCoherentInputInterface
    output_status: str
    membership_mask: int | None
    membership_bits: tuple[int, int] | tuple[()]
    durable_output: TinyCoherentDurableOutput
    history: TinyCoherentHistoryEvidence
    certificate: TinyCoherentCertificate
    resources: TinyCoherentResources
    cleanup_error_bound: float
    fixed_expected_query_ledger_respected: bool
    budget_valid: bool
    status: str
    blockers: tuple[str, ...]
    claim_boundary: TinyCoherentClaimBoundary
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE
    quantum_advantage_claimable: bool = False


class _ExecutionCounter:
    def __init__(self) -> None:
        self.operations: Counter[str] = Counter()

    def add(self, name: str, count: int = 1) -> None:
        self.operations[name] += int(count)


class TinyCoherentAdaptiveStoppingHistory:
    """Exact-state two-level coherent history circuit for ``n=2, k=1``."""

    n_arms = 2
    k = 1
    maximum_phase_qubits = 3
    phase_bit_count = n_arms * maximum_phase_qubits
    output_dimension = 1 << n_arms
    # n=2,k=1 only needs one winner bit and one stop bit.  The complete
    # membership mask is expanded reversibly by the latch permutation.
    work_dimension = 4

    def __init__(
        self,
        oracle: CanonicalRyStatevectorOracle,
        *,
        config: TinyCoherentStoppingHistoryConfig | None = None,
    ) -> None:
        if not isinstance(oracle, CanonicalRyStatevectorOracle):
            raise TypeError("oracle must be a CanonicalRyStatevectorOracle")
        if oracle.n_arms != self.n_arms or oracle.index_dimension != 2:
            raise ValueError("the tiny coherent history circuit requires exactly two arms")
        if config is not None and not isinstance(config, TinyCoherentStoppingHistoryConfig):
            raise TypeError("config must be TinyCoherentStoppingHistoryConfig")
        self.oracle = oracle
        self.config = config or TinyCoherentStoppingHistoryConfig()
        self.shape = (
            (2,) * self.phase_bit_count
            + (2,) * self.n_arms
            + (2,) * self.n_arms
            + (2, 2)
            + (
                self.output_dimension,
                self.output_dimension,
                self.work_dimension,
                2,
            )
        )
        self.statevector_dimension = math.prod(self.shape)
        if self.statevector_dimension > self.config.max_statevector_dimension:
            raise ValueError("explicit register statevector exceeds max_statevector_dimension")
        self._relation_codes = {
            phase_qubits: tuple(
                self._rank_stop_code(code, phase_qubits)
                for code in range(1 << self.phase_bit_count)
            )
            for phase_qubits in PHASE_LEVELS
        }

    @property
    def qubits(self) -> int:
        return int(math.log2(self.statevector_dimension))

    def _phase_axes(self, arm: int, phase_qubits: int) -> tuple[int, ...]:
        start = arm * self.maximum_phase_qubits
        return tuple(range(start, start + phase_qubits))

    def _index_axis(self, arm: int) -> int:
        return self.phase_bit_count + arm

    def _reward_axis(self, arm: int) -> int:
        return self.phase_bit_count + self.n_arms + arm

    def _history_axis(self, level: int) -> int:
        return self.phase_bit_count + 2 * self.n_arms + level

    @property
    def _scratch_axis(self) -> int:
        return len(self.shape) - 4

    @property
    def _output_axis(self) -> int:
        return len(self.shape) - 3

    @property
    def _work_axis(self) -> int:
        return len(self.shape) - 2

    @property
    def _query_control_axis(self) -> int:
        return len(self.shape) - 1

    def initial_state(self) -> ComplexState:
        state = np.zeros(self.statevector_dimension, dtype=np.complex128)
        state[0] = 1.0
        return state

    def _x(self, state: ComplexState, axis: int) -> ComplexState:
        return np.take(state.reshape(self.shape), (1, 0), axis=axis).reshape(-1)

    def _index_xor(
        self,
        state: ComplexState,
        arm: int,
        active_axis: int | None,
    ) -> ComplexState:
        if arm == 0:
            return state.copy()
        source = state.reshape(self.shape)
        if active_axis is None:
            return np.take(source, (1, 0), axis=self._index_axis(arm)).reshape(-1)
        target = source.copy()
        active = np.moveaxis(
            target, (active_axis, self._index_axis(arm)), (-2, -1)
        )
        active[..., 1, :] = active[..., 1, ::-1]
        return target.reshape(-1)

    def _hadamard(
        self,
        state: ComplexState,
        axis: int,
        active_axis: int | None,
    ) -> ComplexState:
        view = state.copy().reshape(self.shape)
        scale = 1.0 / math.sqrt(2.0)
        if active_axis is None:
            active = np.moveaxis(view, axis, -1)
            left = active[..., 0].copy()
            right = active[..., 1].copy()
            active[..., 0] = (left + right) * scale
            active[..., 1] = (left - right) * scale
        else:
            active = np.moveaxis(view, (active_axis, axis), (-2, -1))
            left = active[..., 1, 0].copy()
            right = active[..., 1, 1].copy()
            active[..., 1, 0] = (left + right) * scale
            active[..., 1, 1] = (left - right) * scale
        return view.reshape(-1)

    def _fourier(
        self,
        state: ComplexState,
        axes: tuple[int, ...],
        *,
        inverse: bool,
        active_axis: int | None,
    ) -> ComplexState:
        view = state.copy().reshape(self.shape)
        dimension = 1 << len(axes)
        indices = np.arange(dimension)
        sign = -1.0 if inverse else 1.0
        matrix = np.exp(
            sign * 2j * math.pi * np.outer(indices, indices) / dimension
        ) / math.sqrt(dimension)
        if active_axis is None:
            remaining = tuple(axis for axis in range(len(self.shape)) if axis not in axes)
            permutation = axes + remaining
            inverse_permutation = tuple(np.argsort(permutation))
            ordered = np.transpose(view, permutation)
            transformed = (matrix @ ordered.reshape(dimension, -1)).reshape(
                ordered.shape
            )
            return np.transpose(transformed, inverse_permutation).reshape(-1)

        remaining = tuple(
            axis
            for axis in range(len(self.shape))
            if axis not in axes and axis != active_axis
        )
        permutation = (active_axis,) + axes + remaining
        inverse_permutation = tuple(np.argsort(permutation))
        ordered = np.transpose(view, permutation)
        active_flat = ordered[1].reshape(dimension, -1)
        ordered[1] = (matrix @ active_flat).reshape(ordered[1].shape)
        return np.transpose(ordered, inverse_permutation).reshape(-1)

    def _and_xor(
        self, state: ComplexState, left_axis: int, right_axis: int
    ) -> ComplexState:
        view = state.copy().reshape(self.shape)
        active = np.moveaxis(
            view,
            (left_axis, right_axis, self._query_control_axis),
            (-3, -2, -1),
        )
        active[..., 1, 1, :] = active[..., 1, 1, ::-1]
        return view.reshape(-1)

    def _controlled_reward_reflection(
        self,
        state: ComplexState,
        control_axis: int,
        reward_axis: int,
        reward_value: int,
    ) -> ComplexState:
        view = state.copy().reshape(self.shape)
        active = np.moveaxis(view, (control_axis, reward_axis), (-2, -1))
        active[..., 1, reward_value] *= -1.0
        return view.reshape(-1)

    def _controlled_global_minus(
        self, state: ComplexState, control_axis: int
    ) -> ComplexState:
        view = state.copy().reshape(self.shape)
        active = np.moveaxis(view, control_axis, -1)
        active[..., 1] *= -1.0
        return view.reshape(-1)

    def _controlled_grover(
        self,
        state: ComplexState,
        *,
        arm: int,
        control_axis: int,
        inverse: bool,
        counter: _ExecutionCounter,
        phase_qubits: int,
    ) -> ComplexState:
        index_axis = self._index_axis(arm)
        reward_axis = self._reward_axis(arm)
        tag = f"tiny_true_coherent_history_m{phase_qubits}_arm_{arm}"
        if not inverse:
            state = self._controlled_reward_reflection(
                state, control_axis, reward_axis, 1
            )
            state = self.oracle.apply_embedded(
                state,
                register_shape=self.shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                inverse=True,
                tag=tag,
            )
            state = self._controlled_reward_reflection(
                state, control_axis, reward_axis, 0
            )
            state = self.oracle.apply_embedded(
                state,
                register_shape=self.shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                tag=tag,
            )
        else:
            state = self.oracle.apply_embedded(
                state,
                register_shape=self.shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                inverse=True,
                tag=tag,
            )
            state = self._controlled_reward_reflection(
                state, control_axis, reward_axis, 0
            )
            state = self.oracle.apply_embedded(
                state,
                register_shape=self.shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                tag=tag,
            )
            state = self._controlled_reward_reflection(
                state, control_axis, reward_axis, 1
            )
        state = self._controlled_global_minus(state, control_axis)
        counter.add("controlled_grover_iteration")
        return state

    def _qpe(
        self,
        state: ComplexState,
        *,
        arm: int,
        phase_qubits: int,
        inverse: bool,
        active_axis: int | None,
        counter: _ExecutionCounter,
    ) -> ComplexState:
        axes = self._phase_axes(arm, phase_qubits)
        if not inverse:
            for axis in axes:
                state = self._hadamard(state, axis, active_axis)
                counter.add("active_controlled_phase_hadamard")
            for offset, phase_axis in enumerate(axes):
                power = 1 << (phase_qubits - 1 - offset)
                for _ in range(power):
                    if active_axis is None:
                        query_control = phase_axis
                    else:
                        state = self._and_xor(state, active_axis, phase_axis)
                        counter.add("active_phase_and_compute")
                        query_control = self._query_control_axis
                    state = self._controlled_grover(
                        state,
                        arm=arm,
                        control_axis=query_control,
                        inverse=False,
                        counter=counter,
                        phase_qubits=phase_qubits,
                    )
                    if active_axis is not None:
                        state = self._and_xor(state, active_axis, phase_axis)
                        counter.add("active_phase_and_uncompute")
            state = self._fourier(
                state, axes, inverse=True, active_axis=active_axis
            )
            counter.add("active_controlled_inverse_qft")
            return state

        state = self._fourier(state, axes, inverse=False, active_axis=active_axis)
        counter.add("active_controlled_qft")
        for offset in reversed(range(phase_qubits)):
            phase_axis = axes[offset]
            power = 1 << (phase_qubits - 1 - offset)
            for _ in range(power):
                if active_axis is None:
                    query_control = phase_axis
                else:
                    state = self._and_xor(state, active_axis, phase_axis)
                    counter.add("active_phase_and_compute")
                    query_control = self._query_control_axis
                state = self._controlled_grover(
                    state,
                    arm=arm,
                    control_axis=query_control,
                    inverse=True,
                    counter=counter,
                    phase_qubits=phase_qubits,
                )
                if active_axis is not None:
                    state = self._and_xor(state, active_axis, phase_axis)
                    counter.add("active_phase_and_uncompute")
        for axis in reversed(axes):
            state = self._hadamard(state, axis, active_axis)
            counter.add("active_controlled_phase_hadamard")
        return state

    def _decode_arm_code(
        self, phase_bits: tuple[int, ...], arm: int, phase_qubits: int
    ) -> int:
        start = arm * self.maximum_phase_qubits
        result = 0
        for bit in phase_bits[start : start + phase_qubits]:
            result = (result << 1) | bit
        return result

    def _rank_stop_code(self, full_phase_code: int, phase_qubits: int) -> int:
        bits = tuple(
            int(bit)
            for bit in np.unravel_index(
                full_phase_code, (2,) * self.phase_bit_count
            )
        )
        bins = 1 << phase_qubits
        raw = tuple(
            self._decode_arm_code(bits, arm, phase_qubits)
            for arm in range(self.n_arms)
        )
        folded = tuple(min(value, bins - value) for value in raw)
        ranking = tuple(
            sorted(range(self.n_arms), key=lambda arm: (-folded[arm], arm))
        )
        winner = ranking[0]
        stop = abs(folded[0] - folded[1]) >= MINIMUM_BIN_SEPARATION
        return winner | (int(stop) << 1)

    def _rank_relation(
        self,
        state: ComplexState,
        *,
        phase_qubits: int,
        active_axis: int | None,
    ) -> ComplexState:
        target = state.copy().reshape(self.shape)
        relative_work_axis = self._work_axis - self.phase_bit_count
        relative_active_axis = (
            None if active_axis is None else active_axis - self.phase_bit_count
        )
        work_values = np.arange(self.work_dimension)
        for phase_code, relation_code in enumerate(self._relation_codes[phase_qubits]):
            phase_bits = np.unravel_index(
                phase_code, (2,) * self.phase_bit_count
            )
            slab = target[phase_bits]
            permutation = work_values ^ relation_code
            if relative_active_axis is None:
                slab[...] = np.take(slab, permutation, axis=relative_work_axis)
            else:
                active = np.moveaxis(
                    slab,
                    (relative_active_axis, relative_work_axis),
                    (-2, -1),
                )
                active[..., 1, :] = np.take(
                    active[..., 1, :].copy(), permutation, axis=-1
                )
        return target.reshape(-1)

    def _latch_history_and_scratch(
        self, state: ComplexState, *, level: int
    ) -> ComplexState:
        source = state.reshape(self.shape)
        target = np.empty_like(source)
        source_view = np.moveaxis(
            source,
            (self._history_axis(level), self._scratch_axis, self._work_axis),
            (-3, -2, -1),
        )
        target_view = np.moveaxis(
            target,
            (self._history_axis(level), self._scratch_axis, self._work_axis),
            (-3, -2, -1),
        )
        history_values = np.arange(2)
        scratch_values = np.arange(self.output_dimension)
        for work_code in range(self.work_dimension):
            stop = (work_code >> 1) & 1
            winner = work_code & 1
            membership = (1 << winner) if stop else 0
            history_permutation = history_values ^ stop
            scratch_permutation = scratch_values ^ (membership if stop else 0)
            slab = np.take(
                source_view[..., :, :, work_code],
                history_permutation,
                axis=-2,
            )
            target_view[..., :, :, work_code] = np.take(
                slab, scratch_permutation, axis=-1
            )
        return target.reshape(-1)

    def _copy_scratch_to_durable(self, state: ComplexState) -> ComplexState:
        source = state.reshape(self.shape)
        target = np.empty_like(source)
        source_view = np.moveaxis(
            source, (self._scratch_axis, self._output_axis), (-2, -1)
        )
        target_view = np.moveaxis(
            target, (self._scratch_axis, self._output_axis), (-2, -1)
        )
        output_values = np.arange(self.output_dimension)
        for scratch in range(self.output_dimension):
            target_view[..., scratch, :] = np.take(
                source_view[..., scratch, :], output_values ^ scratch, axis=-1
            )
        return target.reshape(-1)

    def _level_kernel(
        self,
        state: ComplexState,
        *,
        level: int,
        counter: _ExecutionCounter,
    ) -> ComplexState:
        phase_qubits = PHASE_LEVELS[level]
        active_axis = None if level == 0 else self._history_axis(0)
        if active_axis is not None:
            state = self._x(state, active_axis)
            counter.add("active_flag_negation")

        for arm in range(self.n_arms):
            state = self._index_xor(state, arm, active_axis)
            counter.add("active_controlled_index_load")
            state = self.oracle.apply_embedded(
                state,
                register_shape=self.shape,
                index_axis=self._index_axis(arm),
                reward_axis=self._reward_axis(arm),
                control_axis=active_axis,
                tag=f"tiny_true_coherent_history_m{phase_qubits}_arm_{arm}",
            )
            counter.add("active_controlled_reward_prepare")
            state = self._qpe(
                state,
                arm=arm,
                phase_qubits=phase_qubits,
                inverse=False,
                active_axis=active_axis,
                counter=counter,
            )

        state = self._rank_relation(
            state, phase_qubits=phase_qubits, active_axis=active_axis
        )
        counter.add("active_controlled_rank_stop_relation")
        state = self._latch_history_and_scratch(state, level=level)
        counter.add("history_and_scratch_latch")
        state = self._rank_relation(
            state, phase_qubits=phase_qubits, active_axis=active_axis
        )
        counter.add("active_controlled_rank_stop_relation")

        for arm in reversed(range(self.n_arms)):
            state = self._qpe(
                state,
                arm=arm,
                phase_qubits=phase_qubits,
                inverse=True,
                active_axis=active_axis,
                counter=counter,
            )
            state = self.oracle.apply_embedded(
                state,
                register_shape=self.shape,
                index_axis=self._index_axis(arm),
                reward_axis=self._reward_axis(arm),
                control_axis=active_axis,
                inverse=True,
                tag=f"tiny_true_coherent_history_m{phase_qubits}_arm_{arm}",
            )
            counter.add("active_controlled_reward_unprepare")
            state = self._index_xor(state, arm, active_axis)
            counter.add("active_controlled_index_reset")

        if active_axis is not None:
            state = self._x(state, active_axis)
            counter.add("active_flag_negation")
        return state

    def _history_evidence(self, state: ComplexState) -> TinyCoherentHistoryEvidence:
        view = state.reshape(self.shape)
        axes = (self._history_axis(0), self._history_axis(1))
        matrix = np.moveaxis(view, axes, (0, 1)).reshape(2, 2, -1)
        probabilities = {
            f"{first}{second}": float(np.sum(np.abs(matrix[first, second]) ** 2))
            for first in range(2)
            for second in range(2)
        }
        scratch = np.moveaxis(view, self._scratch_axis, 0).reshape(
            self.output_dimension, -1
        )
        scratch_probabilities = {
            str(mask): float(np.sum(np.abs(scratch[mask]) ** 2))
            for mask in range(self.output_dimension)
        }
        dominant = max(probabilities, key=probabilities.__getitem__)
        return TinyCoherentHistoryEvidence(
            probabilities=_frozen_probabilities(probabilities),
            stop_at_level_zero_probability=probabilities["10"],
            stop_at_level_one_probability=probabilities["01"],
            unresolved_probability=probabilities["00"],
            invalid_both_stopped_probability=probabilities["11"],
            dominant_history=dominant,
            dominant_history_probability=probabilities[dominant],
            scratch_mask_probabilities=_frozen_probabilities(scratch_probabilities),
        )

    def _output_density(
        self, state: ComplexState
    ) -> tuple[dict[int, float], NDArray[np.complex128]]:
        view = state.reshape(self.shape)
        matrix = np.moveaxis(view, self._output_axis, 0).reshape(
            self.output_dimension, -1
        )
        density = matrix @ matrix.conj().T
        probabilities = {
            mask: float(max(0.0, density[mask, mask].real))
            for mask in range(self.output_dimension)
        }
        return probabilities, density

    def _zero_axes_probability(
        self, view: ComplexState, axes: tuple[int, ...]
    ) -> float:
        selector: list[object] = [slice(None)] * len(self.shape)
        for axis in axes:
            selector[axis] = 0
        return float(np.sum(np.abs(view[tuple(selector)]) ** 2))

    def _cleanup(
        self,
        final: ComplexState,
        output_probabilities: Mapping[int, float],
        output_density: NDArray[np.complex128],
    ) -> TinyCoherentCleanupLedger:
        view = final.reshape(self.shape)
        phase_axes = tuple(range(self.phase_bit_count))
        index_axes = tuple(self._index_axis(arm) for arm in range(self.n_arms))
        reward_axes = tuple(self._reward_axis(arm) for arm in range(self.n_arms))
        history_axes = (self._history_axis(0), self._history_axis(1))
        singleton_axes = (
            self._scratch_axis,
            self._work_axis,
            self._query_control_axis,
        )
        transient_axes = phase_axes + index_axes + reward_axes + history_axes + singleton_axes

        def nonzero(axes: tuple[int, ...]) -> float:
            return _probability_complement(self._zero_axes_probability(view, axes))

        collision = float(sum(value * value for value in output_probabilities.values()))
        predicted = _probability_complement(collision)
        purity = float(np.trace(output_density @ output_density).real)
        return TinyCoherentCleanupLedger(
            phase_nonzero_probability=nonzero(phase_axes),
            index_nonzero_probability=nonzero(index_axes),
            reward_nonzero_probability=nonzero(reward_axes),
            history_nonzero_probability=nonzero(history_axes),
            scratch_mask_nonzero_probability=nonzero((self._scratch_axis,)),
            rank_work_nonzero_probability=nonzero((self._work_axis,)),
            query_control_nonzero_probability=nonzero((self._query_control_axis,)),
            executed_transient_nonzero_probability=nonzero(transient_axes),
            predicted_transient_nonzero_probability=predicted,
            prediction_residual=abs(nonzero(transient_axes) - predicted),
            output_collision_probability=collision,
            output_reduced_purity=purity,
            purity_residual=abs(purity - collision),
            norm_error=abs(float(np.linalg.norm(final)) - 1.0),
            tolerance=self.config.cleanup_tolerance,
        )

    @staticmethod
    def _expected_queries() -> dict[str, int]:
        # Level m=2 is unconditional.  Level m=3 preparation and every Grover
        # query are controlled by the active-history flag.  Both level kernels
        # are executed once in W and once in W^dagger.
        return {
            "forward": 4,
            "inverse": 4,
            "controlled_forward": 84,
            "controlled_inverse": 84,
            "classical_sample": 0,
            "coherent_total": 176,
            "classical_total": 0,
            "total": 176,
            "qram_queries": 0,
        }

    @staticmethod
    def _expected_level_full_replay_counts(phase_qubits: int) -> dict[str, int]:
        if phase_qubits == 2:
            return {
                "forward": 4,
                "inverse": 4,
                "controlled_forward": 24,
                "controlled_inverse": 24,
                "classical_sample": 0,
                "coherent_total": 56,
                "classical_total": 0,
                "total": 56,
            }
        if phase_qubits == 3:
            return {
                "forward": 0,
                "inverse": 0,
                "controlled_forward": 60,
                "controlled_inverse": 60,
                "classical_sample": 0,
                "coherent_total": 120,
                "classical_total": 0,
                "total": 120,
            }
        raise ValueError("unregistered phase level")

    @staticmethod
    def _tag_counts(snapshot: QuerySnapshot, prefix: str) -> dict[str, int]:
        by_tag = snapshot.by_tag
        counts: Counter[str] = Counter()
        for tag, values in by_tag.items():
            if str(tag).startswith(prefix):
                counts.update({str(key): int(value) for key, value in values.items()})
        result = {
            name: int(counts.get(name, 0))
            for name in (
                "forward",
                "inverse",
                "controlled_forward",
                "controlled_inverse",
                "classical_sample",
            )
        }
        result["coherent_total"] = sum(
            result[name]
            for name in (
                "forward",
                "inverse",
                "controlled_forward",
                "controlled_inverse",
            )
        )
        result["classical_total"] = result["classical_sample"]
        result["total"] = result["coherent_total"] + result["classical_total"]
        return result

    def _level_query_records(
        self, before: QuerySnapshot, after: QuerySnapshot
    ) -> tuple[TinyCoherentLevelQueryRecord, ...]:
        records: list[TinyCoherentLevelQueryRecord] = []
        for level, phase_qubits in enumerate(PHASE_LEVELS):
            prefix = f"tiny_true_coherent_history_m{phase_qubits}_arm_"
            before_counts = self._tag_counts(before, prefix)
            after_counts = self._tag_counts(after, prefix)
            runtime = {
                name: int(after_counts[name] - before_counts[name])
                for name in after_counts
            }
            if any(value < 0 for value in runtime.values()):
                raise ValueError("oracle tag snapshot predates the starting snapshot")
            expected_full = self._expected_level_full_replay_counts(phase_qubits)
            even = all(value % 2 == 0 for value in runtime.values())
            derived_one_way = {
                name: value // 2 for name, value in runtime.items()
            }
            expected_one_way = {
                name: value // 2 for name, value in expected_full.items()
            }
            records.append(
                TinyCoherentLevelQueryRecord(
                    level=level,
                    phase_qubits=phase_qubits,
                    tag_prefix=prefix,
                    kernel_invocations=2,
                    runtime_full_replay_counts=_frozen_counts(runtime),
                    expected_full_replay_counts=_frozen_counts(expected_full),
                    runtime_derived_one_way_counts=_frozen_counts(
                        derived_one_way
                    ),
                    expected_one_way_counts=_frozen_counts(expected_one_way),
                    full_replay_reconciled=runtime == expected_full,
                    one_way_reconciled=(
                        even and derived_one_way == expected_one_way
                    ),
                )
            )
        return tuple(records)

    def run(self) -> TinyCoherentStoppingHistoryResult:
        before = self.oracle.query_snapshot()
        counter = _ExecutionCounter()
        state = self.initial_state()

        state = self._level_kernel(state, level=0, counter=counter)
        state = self._level_kernel(state, level=1, counter=counter)
        history = self._history_evidence(state)
        state = self._copy_scratch_to_durable(state)
        counter.add("scratch_to_durable_mask_copy")
        state = self._level_kernel(state, level=1, counter=counter)
        state = self._level_kernel(state, level=0, counter=counter)

        after = self.oracle.query_snapshot()
        observed = QueryLedger.difference(after, before)
        observed["qram_queries"] = 0
        expected = self._expected_queries()
        reconciled = all(observed.get(name) == value for name, value in expected.items())
        level_query_records = self._level_query_records(before, after)
        reconciled = reconciled and all(
            row.full_replay_reconciled and row.one_way_reconciled
            for row in level_query_records
        )
        output_probabilities, output_density = self._output_density(state)
        cleanup = self._cleanup(state, output_probabilities, output_density)
        dominant_mask = max(output_probabilities, key=output_probabilities.__getitem__)
        dominant_probability = output_probabilities[dominant_mask]
        resolved_probability = (
            history.stop_at_level_zero_probability + history.stop_at_level_one_probability
        )
        complete = (
            reconciled
            and cleanup.passed
            and history.invalid_both_stopped_probability <= self.config.cleanup_tolerance
            and resolved_probability >= 1.0 - self.config.cleanup_tolerance
            and dominant_mask.bit_count() == self.k
            and dominant_probability >= 1.0 - self.config.cleanup_tolerance
        )
        if complete:
            output_status = OUTPUT_MASK
            membership_mask: int | None = dominant_mask
            membership_bits: tuple[int, int] | tuple[()] = tuple(
                (dominant_mask >> arm) & 1 for arm in range(self.n_arms)
            )
            status = "exact_grid_promise_complete_theory_blocked"
        else:
            output_status = OUTPUT_INCONCLUSIVE
            membership_mask = None
            membership_bits = ()
            status = (
                "unresolved_history_fail_closed"
                if cleanup.passed
                else "durable_output_entanglement_cleanup_fail_closed"
            )

        level_costs = tuple(
            int(row.runtime_derived_one_way_counts["coherent_total"])
            for row in level_query_records
        )
        first_cost = float(level_costs[0])
        second_cost = float(sum(level_costs))
        branch_rms = math.sqrt(
            history.stop_at_level_zero_probability * first_cost**2
            + (
                history.stop_at_level_one_probability
                + history.unresolved_probability
                + history.invalid_both_stopped_probability
            )
            * second_cost**2
        )
        query_ledger = TinyCoherentQueryLedger(
            query_counts=_frozen_counts(observed),
            expected_query_counts=_frozen_counts(expected),
            reconciled=reconciled,
            per_level_runtime_records=level_query_records,
            per_level_one_way_query_costs=level_costs,
            worst_case_one_way_history_queries=sum(level_costs),
            worst_case_full_replay_queries=2 * sum(level_costs),
            branch_rms_one_way_theorem_target=branch_rms,
            branch_rms_full_replay_theorem_target=2.0 * branch_rms,
            branch_rms_is_executed_saving=False,
        )
        durable_output = TinyCoherentDurableOutput(
            status=output_status,
            membership_mask=membership_mask,
            membership_bits=membership_bits,
            output_mask_probabilities=_frozen_probabilities(
                {str(mask): value for mask, value in output_probabilities.items()}
            ),
            dominant_mask=dominant_mask,
            dominant_probability=dominant_probability,
            scratch_to_durable_copy_executed=True,
            full_history_replay_executed=True,
            cleanup_passed=cleanup.passed,
        )
        resources = TinyCoherentResources(
            declared_register_qubits=self.qubits,
            register_dimensions=_frozen_counts(
                {
                    "shared_phase_pack": 1 << self.phase_bit_count,
                    "compiled_index_pack": 1 << self.n_arms,
                    "reward_pack": 1 << self.n_arms,
                    "stopping_history": 4,
                    "scratch_membership_mask": self.output_dimension,
                    "durable_membership_mask": self.output_dimension,
                    "rank_stop_work": self.work_dimension,
                    "active_phase_query_control": 2,
                }
            ),
            retained_statevector_dimension=self.statevector_dimension,
            estimated_peak_complex_amplitudes=2 * self.statevector_dimension + 64,
            executed_numpy_kernel_macro_counts=_frozen_counts(counter.operations),
            elementary_gate_ledger_available=False,
            transpiled_depth_available=False,
            compiled_ancilla_qubits_available=False,
            query_ledger=query_ledger,
            cleanup=cleanup,
        )
        certificate = TinyCoherentCertificate(
            issued=False,
            certificate_type=None,
            top_k_correctness_error_bound=None,
            reason=(
                "the tiny circuit is promise-sensitive and has no generic finite-QPE "
                "correctness or confidence theorem"
            ),
        )
        claim_boundary = TinyCoherentClaimBoundary(
            supports=(
                "one statevector containing phase, stop history, scratch and durable masks",
                "active-flag control on every later-level canonical oracle invocation",
                "scratch-to-durable direct-k copy followed by complete reverse replay",
                "exact worst-case controlled-query ledger reconciliation",
                "cleanup success on exact-grid promises and fail-closed off-promise behavior",
            ),
            does_not_support=(
                "generic off-grid correctness or cleanup",
                "a confidence certificate or observable adaptive precision rule",
                "executed query savings from the branch-RMS theorem target",
                "a scalable heterogeneous variable-time construction",
                "an elementary-gate decomposition, depth, or hardware resource estimate",
                "a new query upper bound, composition separation, or matching lower bound",
                "quantum advantage or a CCF-A publication claim",
            ),
        )
        blockers = (
            "fixed_two_level_n2_k1_exact_state_scope",
            "generic_finite_qpe_branches_entangle_durable_output",
            "no_observable_adaptive_precision_confidence_test",
            "branch_rms_is_unproved_theorem_target_not_executed_saving",
            "no_scalable_variable_time_cleanup_resource_bound",
            "no_new_query_complexity_upper_bound",
            "no_matching_oracle_lower_bound",
            "no_elementary_gate_or_transpiled_depth_ledger",
        )
        return TinyCoherentStoppingHistoryResult(
            method_id=METHOD_ID,
            input_interface=TinyCoherentInputInterface(
                oracle="opaque_canonical_ry_statevector_oracle_handle",
                k=self.k,
                internally_fixed_phase_levels=PHASE_LEVELS,
                internally_fixed_minimum_bin_separation=MINIMUM_BIN_SEPARATION,
                forbidden_inputs=(
                    "answer_set",
                    "top_k_membership_mask",
                    "gap",
                    "boundary_value",
                    "family_label",
                    "phase_schedule",
                    "activity_history",
                    "free_qram_list",
                ),
            ),
            output_status=output_status,
            membership_mask=membership_mask,
            membership_bits=membership_bits,
            durable_output=durable_output,
            history=history,
            certificate=certificate,
            resources=resources,
            cleanup_error_bound=cleanup.executed_transient_nonzero_probability,
            fixed_expected_query_ledger_respected=reconciled,
            budget_valid=reconciled,
            status=status,
            blockers=blockers,
            claim_boundary=claim_boundary,
        )

    def audit_level_one_inactive_subspace(self) -> TinyInactiveSubspaceAudit:
        """Witness the clean inactive identity and a dirty-work counterexample.

        The active flag alone is not a global identity guarantee.  The level-1
        kernel assumes its shared phase/index/reward/scratch/rank/query-control
        work is clean.  This audit executes one inactive kernel on the clean
        basis state and one on a basis state with the rank-stop work bit dirty.
        It is a basis witness and negative control, not a subspace theorem.
        """

        expected = {
            "forward": 0,
            "inverse": 0,
            "controlled_forward": 30,
            "controlled_inverse": 30,
            "classical_sample": 0,
            "coherent_total": 60,
            "classical_total": 0,
            "total": 60,
        }

        clean = self.initial_state()
        clean = self._x(clean, self._history_axis(0))
        before_clean = self.oracle.query_snapshot()
        clean_after = self._level_kernel(
            clean, level=1, counter=_ExecutionCounter()
        )
        clean_snapshot = self.oracle.query_snapshot()
        clean_counts = QueryLedger.difference(clean_snapshot, before_clean)
        clean_residual = float(np.linalg.norm(clean_after - clean))

        dirty = np.zeros(self.shape, dtype=np.complex128)
        selector = [0] * len(self.shape)
        selector[self._history_axis(0)] = 1
        selector[self._work_axis] = 0b10
        dirty[tuple(selector)] = 1.0
        dirty_state = dirty.reshape(-1)
        before_dirty = self.oracle.query_snapshot()
        dirty_after = self._level_kernel(
            dirty_state, level=1, counter=_ExecutionCounter()
        )
        dirty_snapshot = self.oracle.query_snapshot()
        dirty_counts = QueryLedger.difference(dirty_snapshot, before_dirty)
        dirty_residual = float(np.linalg.norm(dirty_after - dirty_state))

        return TinyInactiveSubspaceAudit(
            clean_inactive_basis_identity_residual=clean_residual,
            dirty_rank_work_negative_control_residual=dirty_residual,
            clean_identity_witness_passed=(
                clean_residual <= self.config.cleanup_tolerance
            ),
            dirty_negative_control_activated=(
                dirty_residual > self.config.cleanup_tolerance
            ),
            clean_query_counts=_frozen_counts(clean_counts),
            dirty_query_counts=_frozen_counts(dirty_counts),
            clean_query_ledger_reconciled=(clean_counts == expected),
            dirty_query_ledger_reconciled=(dirty_counts == expected),
            valid_identity_subspace=(
                "history_level0=stopped and shared phase,index,reward,scratch,"
                "rank-stop,query-control work initialized to zero"
            ),
            excluded_dirty_subspace=(
                "inactive flag with nonzero rank-stop work; the uncontrolled "
                "history/scratch latch is intentionally not an identity there"
            ),
        )


def run_tiny_coherent_adaptive_stopping_history(
    oracle: CanonicalRyStatevectorOracle,
    *,
    config: TinyCoherentStoppingHistoryConfig | None = None,
) -> TinyCoherentStoppingHistoryResult:
    """Execute the fixed two-level coherent history-copy-replay circuit."""

    return TinyCoherentAdaptiveStoppingHistory(oracle, config=config).run()


def run_tiny_inactive_level_subspace_audit(
    oracle: CanonicalRyStatevectorOracle,
    *,
    config: TinyCoherentStoppingHistoryConfig | None = None,
) -> TinyInactiveSubspaceAudit:
    """Run the separate clean/dirty inactive-level basis audit."""

    return TinyCoherentAdaptiveStoppingHistory(
        oracle, config=config
    ).audit_level_one_inactive_subspace()


__all__ = [
    "BACKEND",
    "BRANCH_RMS_SEMANTICS",
    "CLAIM_SCOPE",
    "METHOD_ID",
    "MINIMUM_BIN_SEPARATION",
    "OUTPUT_INCONCLUSIVE",
    "OUTPUT_MASK",
    "PHASE_LEVELS",
    "QUERY_SEMANTICS",
    "TinyCoherentAdaptiveStoppingHistory",
    "TinyCoherentCertificate",
    "TinyCoherentClaimBoundary",
    "TinyCoherentCleanupLedger",
    "TinyCoherentDurableOutput",
    "TinyCoherentHistoryEvidence",
    "TinyCoherentInputInterface",
    "TinyCoherentLevelQueryRecord",
    "TinyCoherentQueryLedger",
    "TinyCoherentResources",
    "TinyInactiveSubspaceAudit",
    "TinyCoherentStoppingHistoryConfig",
    "TinyCoherentStoppingHistoryResult",
    "run_tiny_coherent_adaptive_stopping_history",
    "run_tiny_inactive_level_subspace_audit",
]
