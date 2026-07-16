"""Tiny end-to-end coherent unknown-boundary Top-k reference unitary.

The algorithm receives only a completed
:class:`~qgapselect.coherent.CanonicalRyStatevectorOracle`, ``k``, and public
circuit precision/resource limits.  It coherently runs one charged QPE per
arm, computes a strict discrete rank relation, copies the complete ``n``-bit
membership mask, reverses the rank relation, and finally applies inverse QPE.

The retained register order is

``(all phase qubits, compiled arm indices, rewards, durable output, rank work)``.

No boundary value, answer set, activity schedule, or free indexed memory is
provided.  Arm indices are ordinary compiled basis states and every reward
oracle invocation goes through ``apply_embedded``.  The rank relation is an
explicit exhaustive no-QRAM permutation.

Exact cleanup is deliberately promise-sensitive.  If the coherent phase
distribution induces output probabilities ``p_y``, the implementation checks
both sides of ``P_garbage = 1 - sum_y p_y**2``.  Exact-grid strict instances
clean completely; generic finite-QPE instances that entangle several output
masks fail closed and retain their full garbage state.

This module is a tiny exact-state circuit/cleanup sanity artifact.  It proves
neither a new query upper bound nor an advantage over legal composition
baselines.
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
from .oracles import QueryLedger

ComplexState = NDArray[np.complex128]

BACKEND = "numpy_exact_statevector_coherent_unknown_boundary_topk"
CLAIM_SCOPE = (
    "tiny_rounding_promise_circuit_sanity_no_new_upper_bound_"
    "no_composition_separation_no_lower_bound_no_transpiled_gate_depth_"
    "no_hardware"
)
RANK_COMPILATION_MODEL = "exhaustive_reversible_phase_rank_relation_no_qram"
QUERY_COUNT_SEMANTICS = "executed_canonical_oracle_calls_plus_zero_qram_charge"
EXECUTION_COUNT_SEMANTICS = (
    "executed_numpy_kernel_macro_invocations_only_not_circuit_gate_counts"
)
LOGICAL_MACRO_COUNT_SEMANTICS = (
    "undecomposed_circuit_semantic_macros_not_elementary_or_transpiled_gates"
)
COMPILATION_PROXY_SEMANTICS = (
    "structural_truth_table_rows_control_width_and_output_bit_incidence_only_"
    "no_gate_or_depth_synthesis"
)
MEMORY_SEMANTICS = (
    "analytic live-complex-entry proxy for the retained state, one unitary "
    "target, and the dense tiny-QFT matrix; not measured allocator memory"
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


def _frozen_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _frozen_probabilities(values: Mapping[int, float]) -> Mapping[int, float]:
    return MappingProxyType({int(key): float(value) for key, value in values.items()})


def _probability_complement(value: float) -> float:
    return min(1.0, max(0.0, 1.0 - float(value)))


@dataclass(frozen=True, slots=True)
class CoherentUnknownBoundaryTopKConfig:
    """Public QPE precision, cleanup tolerance, and exact-state size cap."""

    phase_qubits: int = 2
    cleanup_tolerance: float = 1e-10
    max_statevector_dimension: int = 8_388_608

    def __post_init__(self) -> None:
        phase_qubits = _integer(self.phase_qubits, "phase_qubits", minimum=1)
        if phase_qubits > 5:
            raise ValueError("phase_qubits exceeds the tiny exact-state limit of 5")
        object.__setattr__(self, "phase_qubits", phase_qubits)
        if isinstance(self.cleanup_tolerance, bool):
            raise TypeError("cleanup_tolerance must be a positive finite real")
        tolerance = float(self.cleanup_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("cleanup_tolerance must be a positive finite real")
        object.__setattr__(self, "cleanup_tolerance", tolerance)
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
class CoherentBoundaryEvidence:
    """Discrete strict-boundary mass and the complete durable-mask law."""

    strict_probability: float
    nonstrict_probability: float
    output_mask_probabilities: Mapping[int, float]
    dominant_mask: int
    dominant_probability: float
    output_collision_probability: float
    predicted_transient_nonzero_probability: float
    information_source: str = "charged_coherent_qpe_and_reversible_rank_relation"


@dataclass(frozen=True, slots=True)
class CoherentTopKCleanupLedger:
    """Executed cleanup and the independent rank-copy identity check."""

    phase_nonzero_probability: float
    compiled_index_nonzero_probability: float
    reward_nonzero_probability: float
    rank_work_nonzero_probability: float
    executed_transient_nonzero_probability: float
    predicted_transient_nonzero_probability: float
    prediction_residual: float
    output_reduced_purity: float
    output_collision_probability: float
    purity_residual: float
    norm_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.executed_transient_nonzero_probability <= self.tolerance
            and self.rank_work_nonzero_probability <= self.tolerance
            and self.prediction_residual <= 10.0 * self.tolerance
            and self.purity_residual <= 10.0 * self.tolerance
            and self.norm_error <= self.tolerance
        )


@dataclass(frozen=True, slots=True)
class CoherentUnknownBoundaryResourceLedger:
    """Exact queries, backend operations, and unsynthesised structure.

    Oracle calls are exact.  The two operation mappings are deliberately
    macro-level: neither is an elementary-gate ledger.  No transpiler or target
    gate set is invoked, so circuit depth and post-decomposition ancillas are
    unavailable rather than estimated by a control-width heuristic.
    """

    query_counts: Mapping[str, int]
    executed_numpy_kernel_operation_counts: Mapping[str, int]
    logical_circuit_macro_counts: Mapping[str, int]
    rank_compilation_proxies: Mapping[str, int]
    declared_register_qubits: int
    register_dimensions: Mapping[str, int]
    retained_statevector_dimension: int
    estimated_peak_complex_amplitudes: int
    dense_qft_matrix_entries: int
    qpe_calls: int
    controlled_grover_iterations: int
    rank_truth_table_rows: int
    compiled_relation_storage_bits: int
    rank_compilation_model: str
    query_count_semantics: str
    execution_count_semantics: str
    logical_macro_count_semantics: str
    compilation_proxy_semantics: str
    memory_semantics: str
    qram_assumed: bool
    elementary_gate_ledger_available: bool
    transpiled_depth_available: bool
    transpiled_depth: int | None
    compiled_ancilla_qubits_available: bool
    cleanup: CoherentTopKCleanupLedger
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("coherent_total", 0))


@dataclass(frozen=True, slots=True)
class CoherentUnknownBoundaryTopKResult:
    """Fail-closed result of the tiny coherent rank-copy-replay circuit."""

    state: ComplexState
    boundary: CoherentBoundaryEvidence
    membership_bits: tuple[int, ...]
    membership_mask: int | None
    resources: CoherentUnknownBoundaryResourceLedger
    rounding_promise_witnessed: bool
    strict_boundary_witnessed: bool
    direct_multi_output_complete: bool
    certificate_issued: bool
    blockers: tuple[str, ...]
    status: str
    quantum_advantage_claimable: bool = False
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE


class _ExecutionCounter:
    def __init__(self) -> None:
        self.numpy_kernel_operations: Counter[str] = Counter()
        self.qpe_calls = 0
        self.controlled_grover_iterations = 0

    def add(self, name: str, count: int = 1) -> None:
        self.numpy_kernel_operations[name] += int(count)


class CoherentUnknownBoundaryTopK:
    """End-to-end coherent all-arm finite-QPE Top-k unitary for ``n <= 3``.

    The constructor intentionally exposes no answer-dependent input.  The
    exact-state size limit is a simulation constraint, not an algorithmic
    speedup assumption.
    """

    def __init__(
        self,
        oracle: CanonicalRyStatevectorOracle,
        k: int,
        *,
        config: CoherentUnknownBoundaryTopKConfig | None = None,
    ) -> None:
        if not isinstance(oracle, CanonicalRyStatevectorOracle):
            raise TypeError("oracle must be a CanonicalRyStatevectorOracle")
        if not 2 <= oracle.n_arms <= 3:
            raise ValueError("the tiny exact-state implementation requires 2 <= n <= 3")
        k = _integer(k, "k", minimum=1)
        if k >= oracle.n_arms:
            raise ValueError("k must satisfy 1 <= k < oracle.n_arms")
        if config is not None and not isinstance(
            config, CoherentUnknownBoundaryTopKConfig
        ):
            raise TypeError("config must be CoherentUnknownBoundaryTopKConfig")

        self.oracle = oracle
        self.k = k
        self.config = config or CoherentUnknownBoundaryTopKConfig()
        self.n_arms = oracle.n_arms
        self.phase_qubits = self.config.phase_qubits
        self.phase_bins = 1 << self.phase_qubits
        self.phase_bit_count = self.n_arms * self.phase_qubits
        self.phase_dimension = self.phase_bins**self.n_arms
        self.index_pack_dimension = oracle.index_dimension**self.n_arms
        self.reward_pack_dimension = 1 << self.n_arms
        self.output_dimension = 1 << self.n_arms
        self.work_dimension = 1 << (self.n_arms + 1)
        self.shape = (
            (2,) * self.phase_bit_count
            + (oracle.index_dimension,) * self.n_arms
            + (2,) * self.n_arms
            + (self.output_dimension, self.work_dimension)
        )
        self.statevector_dimension = math.prod(self.shape)
        if self.statevector_dimension > self.config.max_statevector_dimension:
            raise ValueError("explicit register statevector exceeds max_statevector_dimension")
        self._relation_codes = tuple(
            self._rank_boundary_code(phase_code)
            for phase_code in range(self.phase_dimension)
        )

    @property
    def register_dimensions(self) -> Mapping[str, int]:
        return _frozen_counts(
            {
                "phase_estimate_pack": self.phase_dimension,
                "compiled_index_pack": self.index_pack_dimension,
                "reward_pack": self.reward_pack_dimension,
                "durable_membership_output": self.output_dimension,
                "rank_boundary_work": self.work_dimension,
            }
        )

    @property
    def qubits(self) -> int:
        return int(math.log2(self.statevector_dimension))

    def _phase_axes(self, arm: int) -> tuple[int, ...]:
        start = arm * self.phase_qubits
        return tuple(range(start, start + self.phase_qubits))

    def _index_axis(self, arm: int) -> int:
        return self.phase_bit_count + arm

    def _reward_axis(self, arm: int) -> int:
        return self.phase_bit_count + self.n_arms + arm

    @property
    def _output_axis(self) -> int:
        return len(self.shape) - 2

    @property
    def _work_axis(self) -> int:
        return len(self.shape) - 1

    def _validated_state(self, state: ComplexState) -> ComplexState:
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.size != self.statevector_dimension:
            raise ValueError(
                f"expected a flat statevector of length {self.statevector_dimension}, "
                f"got {values.shape}"
            )
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-10):
            raise ValueError("statevector must be normalized")
        return values.copy()

    def initial_state(self) -> ComplexState:
        """Return the all-zero register state before compiled index loading."""

        state = np.zeros(self.statevector_dimension, dtype=np.complex128)
        state[0] = 1.0
        return state

    def _compiled_index_xor(self, state: ComplexState, arm: int) -> ComplexState:
        """XOR one public arm label into its ordinary index register."""

        view = state.reshape(self.shape)
        permutation = np.arange(self.oracle.index_dimension) ^ arm
        return np.take(view, permutation, axis=self._index_axis(arm)).reshape(-1)

    def _hadamard(self, state: ComplexState, axis: int) -> ComplexState:
        view = state.reshape(self.shape)
        active = np.moveaxis(view, axis, -1)
        left = active[..., 0].copy()
        right = active[..., 1].copy()
        scale = 1.0 / math.sqrt(2.0)
        active[..., 0] = (left + right) * scale
        active[..., 1] = (left - right) * scale
        return view.reshape(-1)

    def _fourier_arm(self, state: ComplexState, arm: int, *, inverse: bool) -> ComplexState:
        axes = self._phase_axes(arm)
        view = state.reshape(self.shape)
        remaining = tuple(axis for axis in range(len(self.shape)) if axis not in axes)
        permutation = axes + remaining
        inverse_permutation = tuple(np.argsort(permutation))
        ordered = np.transpose(view, permutation)
        flat = ordered.reshape(self.phase_bins, -1)
        indices = np.arange(self.phase_bins)
        sign = -1.0 if inverse else 1.0
        matrix = np.exp(
            sign
            * 2j
            * math.pi
            * np.outer(indices, indices)
            / self.phase_bins
        ) / math.sqrt(self.phase_bins)
        transformed = (matrix @ flat).reshape(ordered.shape)
        return np.transpose(transformed, inverse_permutation).reshape(-1)

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
        self,
        state: ComplexState,
        control_axis: int,
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
    ) -> ComplexState:
        index_axis = self._index_axis(arm)
        reward_axis = self._reward_axis(arm)
        tag = f"coherent_unknown_boundary_topk_arm_{arm}"
        if not inverse:
            state = self._controlled_reward_reflection(
                state, control_axis, reward_axis, 1
            )
            counter.add("controlled_good_reflection")
            state = self.oracle.apply_embedded(
                state,
                register_shape=self.shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                inverse=True,
                tag=tag,
            )
            counter.add("controlled_reward_oracle_inverse")
            state = self._controlled_reward_reflection(
                state, control_axis, reward_axis, 0
            )
            counter.add("controlled_zero_reflection")
            state = self.oracle.apply_embedded(
                state,
                register_shape=self.shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                tag=tag,
            )
            counter.add("controlled_reward_oracle_forward")
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
            counter.add("controlled_reward_oracle_inverse")
            state = self._controlled_reward_reflection(
                state, control_axis, reward_axis, 0
            )
            counter.add("controlled_zero_reflection")
            state = self.oracle.apply_embedded(
                state,
                register_shape=self.shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                tag=tag,
            )
            counter.add("controlled_reward_oracle_forward")
            state = self._controlled_reward_reflection(
                state, control_axis, reward_axis, 1
            )
            counter.add("controlled_good_reflection")
        state = self._controlled_global_minus(state, control_axis)
        counter.add("controlled_global_phase")
        counter.controlled_grover_iterations += 1
        return state

    def _qpe_forward(
        self,
        state: ComplexState,
        arm: int,
        counter: _ExecutionCounter,
    ) -> ComplexState:
        for axis in self._phase_axes(arm):
            state = self._hadamard(state, axis)
            counter.add("phase_hadamard")
        for offset, axis in enumerate(self._phase_axes(arm)):
            power = 1 << (self.phase_qubits - 1 - offset)
            for _ in range(power):
                state = self._controlled_grover(
                    state,
                    arm=arm,
                    control_axis=axis,
                    inverse=False,
                    counter=counter,
                )
        state = self._fourier_arm(state, arm, inverse=True)
        counter.add("dense_inverse_qft_matrix_multiply")
        counter.qpe_calls += 1
        return state

    def _qpe_inverse(
        self,
        state: ComplexState,
        arm: int,
        counter: _ExecutionCounter,
    ) -> ComplexState:
        state = self._fourier_arm(state, arm, inverse=False)
        counter.add("dense_qft_matrix_multiply")
        axes = self._phase_axes(arm)
        for offset in reversed(range(self.phase_qubits)):
            axis = axes[offset]
            power = 1 << (self.phase_qubits - 1 - offset)
            for _ in range(power):
                state = self._controlled_grover(
                    state,
                    arm=arm,
                    control_axis=axis,
                    inverse=True,
                    counter=counter,
                )
        for axis in reversed(axes):
            state = self._hadamard(state, axis)
            counter.add("phase_hadamard")
        counter.qpe_calls += 1
        return state

    def _decode_phase_code(self, phase_code: int) -> tuple[int, ...]:
        values = [0] * self.n_arms
        remaining = phase_code
        for arm in reversed(range(self.n_arms)):
            values[arm] = remaining % self.phase_bins
            remaining //= self.phase_bins
        return tuple(values)

    def _rank_boundary_code(self, phase_code: int) -> int:
        raw = self._decode_phase_code(phase_code)
        folded = tuple(min(value, self.phase_bins - value) for value in raw)
        ranking = tuple(
            sorted(range(self.n_arms), key=lambda arm: (-folded[arm], arm))
        )
        membership = sum(1 << arm for arm in ranking[: self.k])
        strict = folded[ranking[self.k - 1]] > folded[ranking[self.k]]
        return membership | (int(strict) << self.n_arms)

    def apply_rank_boundary_relation(self, state: ComplexState) -> ComplexState:
        """XOR the exhaustive phase-rank/boundary relation into work."""

        source = self._validated_state(state).reshape(
            self.phase_dimension,
            self.index_pack_dimension,
            self.reward_pack_dimension,
            self.output_dimension,
            self.work_dimension,
        )
        target = np.empty_like(source)
        work_values = np.arange(self.work_dimension)
        for phase_code, relation_code in enumerate(self._relation_codes):
            target[phase_code] = np.take(
                source[phase_code],
                work_values ^ relation_code,
                axis=-1,
            )
        return target.reshape(-1)

    def apply_durable_output_copy(self, state: ComplexState) -> ComplexState:
        """Copy all membership bits iff the strict-boundary work bit is one."""

        source = self._validated_state(state).reshape(
            self.phase_dimension,
            self.index_pack_dimension,
            self.reward_pack_dimension,
            self.output_dimension,
            self.work_dimension,
        )
        target = np.empty_like(source)
        output_values = np.arange(self.output_dimension)
        membership_mask = self.output_dimension - 1
        for work_code in range(self.work_dimension):
            strict = (work_code >> self.n_arms) & 1
            copied = (work_code & membership_mask) if strict else 0
            target[..., :, work_code] = np.take(
                source[..., :, work_code],
                output_values ^ copied,
                axis=-1,
            )
        return target.reshape(-1)

    def _strict_probability(self, state: ComplexState) -> float:
        view = state.reshape(
            self.phase_dimension,
            self.index_pack_dimension,
            self.reward_pack_dimension,
            self.output_dimension,
            self.work_dimension,
        )
        strict_codes = tuple(
            code for code in range(self.work_dimension) if code >> self.n_arms
        )
        return float(np.sum(np.abs(view[..., strict_codes]) ** 2))

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

    def _zero_axes_probability(self, view: ComplexState, axes: tuple[int, ...]) -> float:
        selector: list[object] = [slice(None)] * len(self.shape)
        for axis in axes:
            selector[axis] = 0
        return float(np.sum(np.abs(view[tuple(selector)]) ** 2))

    def _cleanup_ledger(
        self,
        final: ComplexState,
        output_probabilities: Mapping[int, float],
        output_density: NDArray[np.complex128],
    ) -> CoherentTopKCleanupLedger:
        view = final.reshape(self.shape)
        phase_axes = tuple(range(self.phase_bit_count))
        index_axes = tuple(self._index_axis(arm) for arm in range(self.n_arms))
        reward_axes = tuple(self._reward_axis(arm) for arm in range(self.n_arms))
        transient_axes = phase_axes + index_axes + reward_axes + (self._work_axis,)
        phase_nonzero = _probability_complement(
            self._zero_axes_probability(view, phase_axes)
        )
        index_nonzero = _probability_complement(
            self._zero_axes_probability(view, index_axes)
        )
        reward_nonzero = _probability_complement(
            self._zero_axes_probability(view, reward_axes)
        )
        work_nonzero = _probability_complement(
            self._zero_axes_probability(view, (self._work_axis,))
        )
        executed = _probability_complement(
            self._zero_axes_probability(view, transient_axes)
        )
        collision = float(sum(value * value for value in output_probabilities.values()))
        predicted = _probability_complement(collision)
        purity = float(np.trace(output_density @ output_density).real)
        return CoherentTopKCleanupLedger(
            phase_nonzero_probability=phase_nonzero,
            compiled_index_nonzero_probability=index_nonzero,
            reward_nonzero_probability=reward_nonzero,
            rank_work_nonzero_probability=work_nonzero,
            executed_transient_nonzero_probability=executed,
            predicted_transient_nonzero_probability=predicted,
            prediction_residual=abs(executed - predicted),
            output_reduced_purity=purity,
            output_collision_probability=collision,
            purity_residual=abs(purity - collision),
            norm_error=abs(float(np.linalg.norm(final)) - 1.0),
            tolerance=self.config.cleanup_tolerance,
        )

    def _resource_ledger(
        self,
        query_counts: Mapping[str, int],
        counter: _ExecutionCounter,
        cleanup: CoherentTopKCleanupLedger,
    ) -> CoherentUnknownBoundaryResourceLedger:
        relation_output_bit_incidences = 2 * sum(
            code.bit_count() for code in self._relation_codes
        )
        executed_operations = Counter(counter.numpy_kernel_operations)
        executed_operations.update(
            {
                "exhaustive_rank_boundary_permutation": 2,
                "durable_output_permutation": 1,
            }
        )
        controlled_per_direction = self.n_arms * (self.phase_bins - 1)
        logical_macros = {
            "compiled_index_label_load": self.n_arms,
            "canonical_reward_prepare": self.n_arms,
            "qpe_forward": self.n_arms,
            "controlled_grover_forward": controlled_per_direction,
            "dense_inverse_qft": self.n_arms,
            "rank_boundary_relation": 2,
            "durable_output_copy": 1,
            "dense_qft": self.n_arms,
            "controlled_grover_inverse": controlled_per_direction,
            "qpe_inverse": self.n_arms,
            "canonical_reward_unprepare": self.n_arms,
            "compiled_index_label_reset": self.n_arms,
        }
        compilation_proxies = {
            "phase_control_width": self.phase_bit_count,
            "rank_relation_rows_per_call": self.phase_dimension,
            "rank_relation_calls": 2,
            "rank_relation_output_width": self.n_arms + 1,
            "rank_relation_output_bit_incidences_across_calls": (
                relation_output_bit_incidences
            ),
            "compiled_index_x_bit_incidences_compute_and_uncompute": 2
            * sum(arm.bit_count() for arm in range(self.n_arms)),
        }
        combined_queries = dict(query_counts)
        combined_queries["qram_queries"] = 0
        dense_qft_entries = self.phase_bins * self.phase_bins
        peak = 2 * self.statevector_dimension + dense_qft_entries
        return CoherentUnknownBoundaryResourceLedger(
            query_counts=_frozen_counts(combined_queries),
            executed_numpy_kernel_operation_counts=_frozen_counts(
                executed_operations
            ),
            logical_circuit_macro_counts=_frozen_counts(logical_macros),
            rank_compilation_proxies=_frozen_counts(compilation_proxies),
            declared_register_qubits=self.qubits,
            register_dimensions=self.register_dimensions,
            retained_statevector_dimension=self.statevector_dimension,
            estimated_peak_complex_amplitudes=peak,
            dense_qft_matrix_entries=dense_qft_entries,
            qpe_calls=counter.qpe_calls,
            controlled_grover_iterations=counter.controlled_grover_iterations,
            rank_truth_table_rows=self.phase_dimension,
            compiled_relation_storage_bits=self.phase_dimension * (self.n_arms + 1),
            rank_compilation_model=RANK_COMPILATION_MODEL,
            query_count_semantics=QUERY_COUNT_SEMANTICS,
            execution_count_semantics=EXECUTION_COUNT_SEMANTICS,
            logical_macro_count_semantics=LOGICAL_MACRO_COUNT_SEMANTICS,
            compilation_proxy_semantics=COMPILATION_PROXY_SEMANTICS,
            memory_semantics=MEMORY_SEMANTICS,
            qram_assumed=False,
            elementary_gate_ledger_available=False,
            transpiled_depth_available=False,
            transpiled_depth=None,
            compiled_ancilla_qubits_available=False,
            cleanup=cleanup,
        )

    def run(self) -> CoherentUnknownBoundaryTopKResult:
        """Execute compute-rank-copy-uncompute-inverse-QPE and audit it."""

        before = self.oracle.query_snapshot()
        counter = _ExecutionCounter()
        state = self.initial_state()
        for arm in range(self.n_arms):
            state = self._compiled_index_xor(state, arm)
            counter.add("compiled_index_xor_permutation")
            state = self.oracle.apply_embedded(
                state,
                register_shape=self.shape,
                index_axis=self._index_axis(arm),
                reward_axis=self._reward_axis(arm),
                tag=f"coherent_unknown_boundary_topk_arm_{arm}",
            )
            counter.add("reward_oracle_forward")
            state = self._qpe_forward(state, arm, counter)

        ranked = self.apply_rank_boundary_relation(state)
        strict_probability = self._strict_probability(ranked)
        copied = self.apply_durable_output_copy(ranked)
        state = self.apply_rank_boundary_relation(copied)

        for arm in reversed(range(self.n_arms)):
            state = self._qpe_inverse(state, arm, counter)
            state = self.oracle.apply_embedded(
                state,
                register_shape=self.shape,
                index_axis=self._index_axis(arm),
                reward_axis=self._reward_axis(arm),
                inverse=True,
                tag=f"coherent_unknown_boundary_topk_arm_{arm}",
            )
            counter.add("reward_oracle_inverse")
            state = self._compiled_index_xor(state, arm)
            counter.add("compiled_index_xor_permutation")

        query_counts = QueryLedger.difference(self.oracle.query_snapshot(), before)
        probabilities, density = self._output_density(state)
        collision = float(sum(value * value for value in probabilities.values()))
        predicted = _probability_complement(collision)
        dominant_mask = max(probabilities, key=probabilities.__getitem__)
        dominant_probability = probabilities[dominant_mask]
        boundary = CoherentBoundaryEvidence(
            strict_probability=strict_probability,
            nonstrict_probability=_probability_complement(strict_probability),
            output_mask_probabilities=_frozen_probabilities(probabilities),
            dominant_mask=dominant_mask,
            dominant_probability=dominant_probability,
            output_collision_probability=collision,
            predicted_transient_nonzero_probability=predicted,
        )
        cleanup = self._cleanup_ledger(state, probabilities, density)
        resources = self._resource_ledger(query_counts, counter, cleanup)

        tolerance = self.config.cleanup_tolerance
        rounding_promise_witnessed = cleanup.passed
        strict_boundary_witnessed = (
            rounding_promise_witnessed
            and strict_probability >= 1.0 - tolerance
            and dominant_mask.bit_count() == self.k
            and dominant_probability >= 1.0 - tolerance
        )
        direct_complete = strict_boundary_witnessed
        if direct_complete:
            membership_mask: int | None = dominant_mask
            membership_bits = tuple(
                (dominant_mask >> arm) & 1 for arm in range(self.n_arms)
            )
            status = "rounding_promise_top_k_unitary_complete_theorem_blocked"
        else:
            membership_mask = None
            membership_bits = ()
            status = (
                "finite_qpe_output_entanglement_fail_closed"
                if not rounding_promise_witnessed
                else "coherent_discrete_boundary_not_strict_fail_closed"
            )

        blockers = [
            "no_new_query_complexity_upper_bound",
            "no_same_interface_composition_separation",
            "no_matching_oracle_lower_bound",
            "no_elementary_gate_or_transpiled_depth_ledger",
            "tiny_exact_state_simulation_not_hardware_evidence",
        ]
        if not rounding_promise_witnessed:
            blockers.append("finite_qpe_output_entanglement_cleanup_failed")
        if rounding_promise_witnessed and not strict_boundary_witnessed:
            blockers.append("coherent_discrete_boundary_not_strict")
        if direct_complete:
            blockers.append("exact_grid_rounding_promise_not_generic_input_theorem")

        return CoherentUnknownBoundaryTopKResult(
            state=state,
            boundary=boundary,
            membership_bits=membership_bits,
            membership_mask=membership_mask,
            resources=resources,
            rounding_promise_witnessed=rounding_promise_witnessed,
            strict_boundary_witnessed=strict_boundary_witnessed,
            direct_multi_output_complete=direct_complete,
            certificate_issued=False,
            blockers=tuple(blockers),
            status=status,
        )


def run_coherent_unknown_boundary_topk(
    oracle: CanonicalRyStatevectorOracle,
    k: int,
    *,
    config: CoherentUnknownBoundaryTopKConfig | None = None,
) -> CoherentUnknownBoundaryTopKResult:
    """Run the tiny exact-state coherent unknown-boundary Top-k circuit."""

    return CoherentUnknownBoundaryTopK(oracle, k, config=config).run()


__all__ = [
    "BACKEND",
    "CLAIM_SCOPE",
    "COMPILATION_PROXY_SEMANTICS",
    "EXECUTION_COUNT_SEMANTICS",
    "LOGICAL_MACRO_COUNT_SEMANTICS",
    "MEMORY_SEMANTICS",
    "QUERY_COUNT_SEMANTICS",
    "RANK_COMPILATION_MODEL",
    "CoherentBoundaryEvidence",
    "CoherentTopKCleanupLedger",
    "CoherentUnknownBoundaryResourceLedger",
    "CoherentUnknownBoundaryTopK",
    "CoherentUnknownBoundaryTopKConfig",
    "CoherentUnknownBoundaryTopKResult",
    "run_coherent_unknown_boundary_topk",
]
