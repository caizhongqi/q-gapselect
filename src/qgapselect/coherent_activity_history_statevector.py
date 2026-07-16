"""Exact-state coherent activity-history execution for tiny instances.

This module is deliberately narrower than the analytic activity-history
executor.  It applies controlled canonical reward-oracle calls to an index
superposition and retains every simulated register.  The implementation is a
small-state circuit-semantics check, not a variable-time query theorem and not
evidence of a quantum advantage.

The unknown boundary is obtained only from measured QPE histograms.  The
histogram rule currently has no finite-sample confidence theorem, and the
direct output register is branch-local rather than a proven coherent union of
all selected indices.  Consequently the implementation always fails closed
when asked for a Top-k certificate.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .coherent import CanonicalRyStatevectorOracle
from .oracles import QueryLedger

ComplexState = NDArray[np.complex128]
BACKEND = "numpy_exact_statevector_coherent_index_activity_history"
CLAIM_SCOPE = "exact_state_code_sanity_only_no_new_algorithm_no_hardware_no_query_advantage_theorem"


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


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _immutable_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _immutable_registers(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


@dataclass(frozen=True, slots=True)
class CoherentHistoryStatevectorConfig:
    """Public precision, measurement, and memory limits for the toy kernel."""

    phase_qubits_by_level: tuple[int, ...] = (2, 3)
    boundary_phase_qubits: int = 3
    boundary_shots: int = 128
    minimum_boundary_samples_per_arm: int = 4
    measurement_seed: int = 0
    cleanup_tolerance: float = 1e-9
    max_statevector_dimension: int = 8_388_608

    def __post_init__(self) -> None:
        try:
            raw_levels = tuple(self.phase_qubits_by_level)
        except TypeError as error:
            raise TypeError("phase_qubits_by_level must be a sequence") from error
        if not raw_levels:
            raise ValueError("phase_qubits_by_level cannot be empty")
        levels = tuple(_integer(value, "phase precision", minimum=1) for value in raw_levels)
        if any(value > 5 for value in levels):
            raise ValueError("phase precision exceeds the exact-state limit of 5")
        if any(right <= left for left, right in zip(levels, levels[1:], strict=False)):
            raise ValueError("phase precisions must be strictly increasing")
        object.__setattr__(self, "phase_qubits_by_level", levels)
        boundary = _integer(self.boundary_phase_qubits, "boundary_phase_qubits", minimum=1)
        if boundary > 5:
            raise ValueError("boundary_phase_qubits exceeds the exact-state limit of 5")
        object.__setattr__(self, "boundary_phase_qubits", boundary)
        for name in (
            "boundary_shots",
            "minimum_boundary_samples_per_arm",
            "max_statevector_dimension",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=1),
            )
        object.__setattr__(
            self,
            "measurement_seed",
            _integer(self.measurement_seed, "measurement_seed"),
        )
        if isinstance(self.cleanup_tolerance, bool):
            raise TypeError("cleanup_tolerance must be a positive real number")
        tolerance = float(self.cleanup_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("cleanup_tolerance must be finite and positive")
        object.__setattr__(self, "cleanup_tolerance", tolerance)


@dataclass(frozen=True, slots=True)
class BoundaryArmHistogram:
    """Measured folded-QPE histogram for one arm."""

    arm: int
    samples: int
    folded_bin_counts: Mapping[str, int]
    modal_folded_bin: int | None
    modal_amplitude: float | None
    empirical_radius: float | None


@dataclass(frozen=True, slots=True)
class MeasuredBoundaryBracket:
    """Unknown-boundary bracket obtained without a hidden amplitude read."""

    lower: float | None
    upper: float | None
    center: float | None
    complete: bool
    status: str
    arm_histograms: tuple[BoundaryArmHistogram, ...]
    query_counts: Mapping[str, int]
    finite_sample_confidence_proved: bool = False
    information_source: str = "measured_joint_index_phase_histogram"


@dataclass(frozen=True, slots=True)
class CoherentHistoryLayer:
    """One executed precision level on the full activity-history state."""

    level: int
    phase_qubits: int
    active_probability_before: float
    active_probability_after: float
    newly_stopped_probability: float
    existing_stopped_branch_residual: float
    predicate_workspace_residual: float
    control_workspace_residual: float
    phase_reward_workspace_residual: float
    norm_error: float
    query_counts: Mapping[str, int]
    depth: int
    gate_counts: Mapping[str, int]

    @property
    def cleanup_passed(self) -> bool:
        return (
            max(
                self.predicate_workspace_residual,
                self.control_workspace_residual,
                self.phase_reward_workspace_residual,
                self.norm_error,
            )
            <= 1e-9
        )


@dataclass(frozen=True, slots=True)
class CoherentHistoryStatevectorResources:
    """Executed query/circuit ledger; no analytic IAE costs are included."""

    query_counts: Mapping[str, int]
    boundary_query_counts: Mapping[str, int]
    history_query_counts: Mapping[str, int]
    gate_counts: Mapping[str, int]
    depth: int
    qubits: int
    register_dimensions: Mapping[str, int]
    retained_statevector_dimension: int
    peak_statevector_dimension: int
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("coherent_total", 0))


@dataclass(frozen=True, slots=True)
class CoherentHistoryStatevectorResult:
    """Fail-closed result of the tiny exact-state experiment."""

    state: ComplexState
    boundary: MeasuredBoundaryBracket
    layers: tuple[CoherentHistoryLayer, ...]
    resources: CoherentHistoryStatevectorResources
    active_probability: float
    stop_probabilities: tuple[float, ...]
    output_mask_probabilities: tuple[float, ...]
    direct_output_write_executed: bool
    direct_multi_output_complete: bool
    certificate_issued: bool
    blockers: tuple[str, ...]
    status: str
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE
    quantum_advantage_claimable: bool = False


class _CircuitCounter:
    def __init__(self) -> None:
        self.gates: Counter[str] = Counter()
        self.depth = 0

    def add(self, name: str, count: int = 1, *, depth: int | None = None) -> None:
        self.gates[name] += count
        self.depth += count if depth is None else depth


class CoherentActivityHistoryStatevectorKernel:
    """Execute QPE layers coherently over the complete index register.

    Register order is ``(phase, index, reward, active, history, stop_code,
    output_mask, predicate, oracle_control)``.  ``history`` is a level bitmask;
    ``output_mask`` has one bit per declared arm.  The latter is genuinely
    written by a reversible controlled-X, but it remains branch-local and is
    therefore not misreported as complete multi-output extraction.
    """

    _PHASE = 0
    _INDEX = 1
    _REWARD = 2
    _ACTIVE = 3
    _HISTORY = 4
    _STOP = 5
    _OUTPUT = 6
    _PREDICATE = 7
    _CONTROL = 8

    def __init__(
        self,
        oracle: CanonicalRyStatevectorOracle,
        k: int,
        *,
        config: CoherentHistoryStatevectorConfig | None = None,
    ) -> None:
        if not isinstance(oracle, CanonicalRyStatevectorOracle):
            raise TypeError("oracle must be a CanonicalRyStatevectorOracle")
        if oracle.n_arms > 8:
            raise ValueError("the exact-state activity-history limit is n <= 8")
        k = _integer(k, "k", minimum=1)
        if k >= oracle.n_arms:
            raise ValueError("k must satisfy 1 <= k < oracle.n_arms")
        if config is not None and not isinstance(config, CoherentHistoryStatevectorConfig):
            raise TypeError("config must be CoherentHistoryStatevectorConfig")
        self.oracle = oracle
        self.k = k
        self.config = config or CoherentHistoryStatevectorConfig()
        self.level_count = len(self.config.phase_qubits_by_level)
        self.max_phase_qubits = max(self.config.phase_qubits_by_level)
        self.phase_dimension = 1 << self.max_phase_qubits
        self.history_dimension = 1 << self.level_count
        self.stop_dimension = _next_power_of_two(self.level_count + 1)
        self.output_dimension = 1 << oracle.n_arms
        self.shape = (
            self.phase_dimension,
            oracle.index_dimension,
            2,
            2,
            self.history_dimension,
            self.stop_dimension,
            self.output_dimension,
            2,
            2,
        )
        self.statevector_dimension = math.prod(self.shape)
        if self.statevector_dimension > self.config.max_statevector_dimension:
            raise ValueError("explicit register statevector exceeds max_statevector_dimension")
        self._counter = _CircuitCounter()
        self._rng = np.random.default_rng(self.config.measurement_seed)

    @property
    def register_dimensions(self) -> Mapping[str, int]:
        return _immutable_registers(
            {
                "phase": self.phase_dimension,
                "index": self.oracle.index_dimension,
                "reward": 2,
                "active": 2,
                "history": self.history_dimension,
                "stop": self.stop_dimension,
                "output": self.output_dimension,
                "predicate": 2,
                "workspace_control": 2,
            }
        )

    @property
    def qubits(self) -> int:
        return sum(int(math.log2(value)) for value in self.shape)

    def initial_state(self) -> ComplexState:
        state = np.zeros(self.shape, dtype=np.complex128)
        amplitude = 1.0 / math.sqrt(self.oracle.n_arms)
        state[0, : self.oracle.n_arms, 0, 1, 0, 0, 0, 0, 0] = amplitude
        return state.reshape(-1)

    @staticmethod
    def _validated_state(state: ComplexState, shape: tuple[int, ...]) -> ComplexState:
        values = np.asarray(state, dtype=np.complex128)
        expected = math.prod(shape)
        if values.ndim != 1 or values.size != expected:
            raise ValueError(f"expected statevector length {expected}, got {values.shape}")
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-10):
            raise ValueError("statevector must be normalized")
        return values.copy()

    @staticmethod
    def _xor_control(
        state: ComplexState,
        *,
        shape: tuple[int, ...],
        phase_axis: int,
        active_axis: int,
        control_axis: int,
        phase_values: Sequence[int],
    ) -> ComplexState:
        view = state.reshape(shape)
        ordered = np.moveaxis(view, (phase_axis, active_axis, control_axis), (-3, -2, -1))
        for phase in phase_values:
            block = ordered[..., phase, 1, :].copy()
            ordered[..., phase, 1, 0] = block[..., 1]
            ordered[..., phase, 1, 1] = block[..., 0]
        return view.reshape(-1)

    def _controlled_oracle(
        self,
        state: ComplexState,
        *,
        shape: tuple[int, ...],
        phase_axis: int,
        index_axis: int,
        reward_axis: int,
        active_axis: int,
        control_axis: int,
        phase_values: Sequence[int],
        inverse: bool,
        tag: str,
        counter: _CircuitCounter,
    ) -> ComplexState:
        state = self._xor_control(
            state,
            shape=shape,
            phase_axis=phase_axis,
            active_axis=active_axis,
            control_axis=control_axis,
            phase_values=phase_values,
        )
        counter.add("activity_and_phase_control_compute")
        state = self.oracle.apply_embedded(
            state,
            register_shape=shape,
            index_axis=index_axis,
            reward_axis=reward_axis,
            control_axis=control_axis,
            inverse=inverse,
            tag=tag,
        )
        counter.add(
            "controlled_reward_oracle_inverse" if inverse else "controlled_reward_oracle_forward"
        )
        state = self._xor_control(
            state,
            shape=shape,
            phase_axis=phase_axis,
            active_axis=active_axis,
            control_axis=control_axis,
            phase_values=phase_values,
        )
        counter.add("activity_and_phase_control_uncompute")
        return state

    @staticmethod
    def _controlled_reward_reflection(
        state: ComplexState,
        *,
        shape: tuple[int, ...],
        reward_axis: int,
        control_axis: int,
        reward_value: int,
    ) -> ComplexState:
        view = state.copy().reshape(shape)
        ordered = np.moveaxis(view, (control_axis, reward_axis), (-2, -1))
        ordered[..., 1, reward_value] *= -1.0
        return view.reshape(-1)

    @staticmethod
    def _controlled_global_minus(
        state: ComplexState,
        *,
        shape: tuple[int, ...],
        control_axis: int,
    ) -> ComplexState:
        view = state.copy().reshape(shape)
        active = np.moveaxis(view, control_axis, -1)
        active[..., 1] *= -1.0
        return view.reshape(-1)

    def _controlled_grover(
        self,
        state: ComplexState,
        *,
        shape: tuple[int, ...],
        phase_axis: int,
        index_axis: int,
        reward_axis: int,
        active_axis: int,
        control_axis: int,
        phase_values: Sequence[int],
        inverse: bool,
        tag: str,
        counter: _CircuitCounter,
    ) -> ComplexState:
        state = self._xor_control(
            state,
            shape=shape,
            phase_axis=phase_axis,
            active_axis=active_axis,
            control_axis=control_axis,
            phase_values=phase_values,
        )
        counter.add("grover_control_compute")
        if not inverse:
            state = self._controlled_reward_reflection(
                state,
                shape=shape,
                reward_axis=reward_axis,
                control_axis=control_axis,
                reward_value=1,
            )
            counter.add("controlled_good_reflection")
            state = self.oracle.apply_embedded(
                state,
                register_shape=shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                inverse=True,
                tag=tag,
            )
            counter.add("controlled_reward_oracle_inverse")
            state = self._controlled_reward_reflection(
                state,
                shape=shape,
                reward_axis=reward_axis,
                control_axis=control_axis,
                reward_value=0,
            )
            counter.add("controlled_zero_reflection")
            state = self.oracle.apply_embedded(
                state,
                register_shape=shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                tag=tag,
            )
            counter.add("controlled_reward_oracle_forward")
        else:
            state = self.oracle.apply_embedded(
                state,
                register_shape=shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                inverse=True,
                tag=tag,
            )
            counter.add("controlled_reward_oracle_inverse")
            state = self._controlled_reward_reflection(
                state,
                shape=shape,
                reward_axis=reward_axis,
                control_axis=control_axis,
                reward_value=0,
            )
            counter.add("controlled_zero_reflection")
            state = self.oracle.apply_embedded(
                state,
                register_shape=shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                tag=tag,
            )
            counter.add("controlled_reward_oracle_forward")
            state = self._controlled_reward_reflection(
                state,
                shape=shape,
                reward_axis=reward_axis,
                control_axis=control_axis,
                reward_value=1,
            )
            counter.add("controlled_good_reflection")
        state = self._controlled_global_minus(state, shape=shape, control_axis=control_axis)
        counter.add("controlled_global_phase")
        state = self._xor_control(
            state,
            shape=shape,
            phase_axis=phase_axis,
            active_axis=active_axis,
            control_axis=control_axis,
            phase_values=phase_values,
        )
        counter.add("grover_control_uncompute")
        return state

    @staticmethod
    def _controlled_hadamard(
        state: ComplexState,
        *,
        shape: tuple[int, ...],
        phase_axis: int,
        active_axis: int,
        phase_bins: int,
        bit_mask: int,
    ) -> ComplexState:
        view = state.reshape(shape)
        ordered = np.moveaxis(view, (phase_axis, active_axis), (-2, -1))
        scale = 1.0 / math.sqrt(2.0)
        for low in range(phase_bins):
            if low & bit_mask:
                continue
            high = low | bit_mask
            left = ordered[..., low, 1].copy()
            right = ordered[..., high, 1].copy()
            ordered[..., low, 1] = (left + right) * scale
            ordered[..., high, 1] = (left - right) * scale
        return view.reshape(-1)

    @staticmethod
    def _controlled_fourier(
        state: ComplexState,
        *,
        shape: tuple[int, ...],
        phase_axis: int,
        active_axis: int,
        phase_bins: int,
        inverse: bool,
    ) -> ComplexState:
        view = state.reshape(shape)
        ordered = np.moveaxis(view, (phase_axis, active_axis), (-2, -1))
        indices = np.arange(phase_bins)
        sign = -1.0 if inverse else 1.0
        matrix = np.exp(sign * 2j * math.pi * np.outer(indices, indices) / phase_bins) / math.sqrt(
            phase_bins
        )
        source = ordered[..., :phase_bins, 1].copy()
        ordered[..., :phase_bins, 1] = np.einsum("ab,...b->...a", matrix, source, optimize=True)
        return view.reshape(-1)

    def _qpe(
        self,
        state: ComplexState,
        *,
        shape: tuple[int, ...],
        phase_axis: int,
        index_axis: int,
        reward_axis: int,
        active_axis: int,
        control_axis: int,
        phase_qubits: int,
        inverse: bool,
        tag: str,
        counter: _CircuitCounter,
    ) -> ComplexState:
        phase_bins = 1 << phase_qubits
        if not inverse:
            for bit in range(phase_qubits):
                mask = 1 << (phase_qubits - 1 - bit)
                state = self._controlled_hadamard(
                    state,
                    shape=shape,
                    phase_axis=phase_axis,
                    active_axis=active_axis,
                    phase_bins=phase_bins,
                    bit_mask=mask,
                )
                counter.add("active_phase_hadamard")
            for bit in range(phase_qubits):
                mask = 1 << (phase_qubits - 1 - bit)
                phase_values = tuple(phase for phase in range(phase_bins) if phase & mask)
                for _ in range(mask):
                    state = self._controlled_grover(
                        state,
                        shape=shape,
                        phase_axis=phase_axis,
                        index_axis=index_axis,
                        reward_axis=reward_axis,
                        active_axis=active_axis,
                        control_axis=control_axis,
                        phase_values=phase_values,
                        inverse=False,
                        tag=tag,
                        counter=counter,
                    )
            state = self._controlled_fourier(
                state,
                shape=shape,
                phase_axis=phase_axis,
                active_axis=active_axis,
                phase_bins=phase_bins,
                inverse=True,
            )
            counter.add(
                "active_inverse_qft",
                depth=phase_qubits * (phase_qubits + 1) // 2,
            )
        else:
            state = self._controlled_fourier(
                state,
                shape=shape,
                phase_axis=phase_axis,
                active_axis=active_axis,
                phase_bins=phase_bins,
                inverse=False,
            )
            counter.add("active_qft", depth=phase_qubits * (phase_qubits + 1) // 2)
            for bit in reversed(range(phase_qubits)):
                mask = 1 << (phase_qubits - 1 - bit)
                phase_values = tuple(phase for phase in range(phase_bins) if phase & mask)
                for _ in range(mask):
                    state = self._controlled_grover(
                        state,
                        shape=shape,
                        phase_axis=phase_axis,
                        index_axis=index_axis,
                        reward_axis=reward_axis,
                        active_axis=active_axis,
                        control_axis=control_axis,
                        phase_values=phase_values,
                        inverse=True,
                        tag=tag,
                        counter=counter,
                    )
            for bit in reversed(range(phase_qubits)):
                mask = 1 << (phase_qubits - 1 - bit)
                state = self._controlled_hadamard(
                    state,
                    shape=shape,
                    phase_axis=phase_axis,
                    active_axis=active_axis,
                    phase_bins=phase_bins,
                    bit_mask=mask,
                )
                counter.add("active_phase_hadamard")
        return state

    def _prepare_and_qpe(
        self,
        state: ComplexState,
        *,
        shape: tuple[int, ...],
        phase_axis: int,
        index_axis: int,
        reward_axis: int,
        active_axis: int,
        control_axis: int,
        phase_qubits: int,
        inverse: bool,
        tag: str,
        counter: _CircuitCounter,
    ) -> ComplexState:
        phase_values = tuple(range(shape[phase_axis]))
        if not inverse:
            state = self._controlled_oracle(
                state,
                shape=shape,
                phase_axis=phase_axis,
                index_axis=index_axis,
                reward_axis=reward_axis,
                active_axis=active_axis,
                control_axis=control_axis,
                phase_values=phase_values,
                inverse=False,
                tag=tag,
                counter=counter,
            )
            return self._qpe(
                state,
                shape=shape,
                phase_axis=phase_axis,
                index_axis=index_axis,
                reward_axis=reward_axis,
                active_axis=active_axis,
                control_axis=control_axis,
                phase_qubits=phase_qubits,
                inverse=False,
                tag=tag,
                counter=counter,
            )
        state = self._qpe(
            state,
            shape=shape,
            phase_axis=phase_axis,
            index_axis=index_axis,
            reward_axis=reward_axis,
            active_axis=active_axis,
            control_axis=control_axis,
            phase_qubits=phase_qubits,
            inverse=True,
            tag=tag,
            counter=counter,
        )
        return self._controlled_oracle(
            state,
            shape=shape,
            phase_axis=phase_axis,
            index_axis=index_axis,
            reward_axis=reward_axis,
            active_axis=active_axis,
            control_axis=control_axis,
            phase_values=phase_values,
            inverse=True,
            tag=tag,
            counter=counter,
        )

    @staticmethod
    def _xor_history_active(
        state: ComplexState, shape: tuple[int, ...], level: int
    ) -> ComplexState:
        view = state.reshape(shape)
        ordered = np.moveaxis(view, (3, 4), (-2, -1))
        permutation = np.arange(shape[4]) ^ (1 << level)
        ordered[..., 1, :] = ordered[..., 1, permutation].copy()
        return view.reshape(-1)

    @staticmethod
    def _phase_value(phase_bin: int, phase_bins: int) -> float:
        folded = min(phase_bin, phase_bins - phase_bin)
        return math.sin(math.pi * folded / phase_bins) ** 2

    def _classification(
        self,
        phase_bin: int,
        phase_bins: int,
        bracket: MeasuredBoundaryBracket,
    ) -> int:
        if bracket.lower is None or bracket.upper is None:
            return 0
        value = self._phase_value(phase_bin, phase_bins)
        # A half-bin angular guard prevents a coarse level from stopping a
        # branch whose phase cell touches the empirical boundary bracket.
        angle = math.asin(math.sqrt(min(max(value, 0.0), 1.0)))
        angular_radius = math.pi / (2.0 * phase_bins)
        lower_value = math.sin(max(0.0, angle - angular_radius)) ** 2
        upper_value = math.sin(min(math.pi / 2.0, angle + angular_radius)) ** 2
        if lower_value > bracket.upper:
            return 1
        if upper_value < bracket.lower:
            return -1
        return 0

    def _predicate_compute(
        self,
        state: ComplexState,
        phase_bins: int,
        bracket: MeasuredBoundaryBracket,
    ) -> ComplexState:
        view = state.reshape(self.shape)
        ordered = np.moveaxis(
            view,
            (self._PHASE, self._INDEX, self._ACTIVE, self._PREDICATE),
            (-4, -3, -2, -1),
        )
        for phase in range(phase_bins):
            if self._classification(phase, phase_bins, bracket) == 0:
                continue
            for index in range(self.oracle.n_arms):
                block = ordered[..., phase, index, 1, :].copy()
                ordered[..., phase, index, 1, 0] = block[..., 1]
                ordered[..., phase, index, 1, 1] = block[..., 0]
        return view.reshape(-1)

    def _decision_copy_phase(
        self,
        state: ComplexState,
        *,
        level: int,
        phase_bins: int,
        bracket: MeasuredBoundaryBracket,
    ) -> ComplexState:
        view = state.reshape(self.shape)
        # Predicate-controlled stop-code XOR is an explicit reversible write.
        ordered_stop = np.moveaxis(view, (self._PREDICATE, self._STOP), (-2, -1))
        permutation = np.arange(self.stop_dimension) ^ (level + 1)
        ordered_stop[..., 1, :] = ordered_stop[..., 1, permutation].copy()
        self._counter.add("predicate_copy_stop_code")

        # Selected branches receive phase kickback and an actual coherent XOR
        # into their index bit of the output-mask register.
        ordered = np.moveaxis(
            view,
            (self._PHASE, self._INDEX, self._PREDICATE, self._OUTPUT),
            (-4, -3, -2, -1),
        )
        for phase in range(phase_bins):
            if self._classification(phase, phase_bins, bracket) != 1:
                continue
            for index in range(self.oracle.n_arms):
                ordered[..., phase, index, 1, :] *= -1.0
                mask_permutation = np.arange(self.output_dimension) ^ (1 << index)
                ordered[..., phase, index, 1, :] = ordered[
                    ..., phase, index, 1, mask_permutation
                ].copy()
        self._counter.add("selected_phase_kickback")
        self._counter.add("coherent_branch_output_mask_xor")
        return view.reshape(-1)

    def _update_active_from_stop(self, state: ComplexState, *, level: int) -> ComplexState:
        view = state.reshape(self.shape)
        ordered = np.moveaxis(view, (self._STOP, self._ACTIVE), (-2, -1))
        block = ordered[..., level + 1, :].copy()
        ordered[..., level + 1, 0] = block[..., 1]
        ordered[..., level + 1, 1] = block[..., 0]
        return view.reshape(-1)

    def _apply_layer_circuit(
        self,
        state: ComplexState,
        *,
        level: int,
        bracket: MeasuredBoundaryBracket,
        inverse: bool,
        counter: _CircuitCounter,
    ) -> ComplexState:
        phase_qubits = self.config.phase_qubits_by_level[level]
        phase_bins = 1 << phase_qubits
        tag = f"coherent_history_level_{level}"
        if inverse:
            state = self._update_active_from_stop(state, level=level)
            counter.add("active_stop_update_inverse")
            state = self._prepare_and_qpe(
                state,
                shape=self.shape,
                phase_axis=self._PHASE,
                index_axis=self._INDEX,
                reward_axis=self._REWARD,
                active_axis=self._ACTIVE,
                control_axis=self._CONTROL,
                phase_qubits=phase_qubits,
                inverse=False,
                tag=f"{tag}_inverse",
                counter=counter,
            )
            state = self._predicate_compute(state, phase_bins, bracket)
            counter.add("phase_predicate_compute_inverse")
            state = self._decision_copy_phase(
                state,
                level=level,
                phase_bins=phase_bins,
                bracket=bracket,
            )
            state = self._predicate_compute(state, phase_bins, bracket)
            counter.add("phase_predicate_uncompute_inverse")
            state = self._prepare_and_qpe(
                state,
                shape=self.shape,
                phase_axis=self._PHASE,
                index_axis=self._INDEX,
                reward_axis=self._REWARD,
                active_axis=self._ACTIVE,
                control_axis=self._CONTROL,
                phase_qubits=phase_qubits,
                inverse=True,
                tag=f"{tag}_inverse",
                counter=counter,
            )
            state = self._xor_history_active(state, self.shape, level)
            counter.add("activity_history_unwrite")
            return state

        state = self._xor_history_active(state, self.shape, level)
        counter.add("activity_history_write")
        state = self._prepare_and_qpe(
            state,
            shape=self.shape,
            phase_axis=self._PHASE,
            index_axis=self._INDEX,
            reward_axis=self._REWARD,
            active_axis=self._ACTIVE,
            control_axis=self._CONTROL,
            phase_qubits=phase_qubits,
            inverse=False,
            tag=tag,
            counter=counter,
        )
        state = self._predicate_compute(state, phase_bins, bracket)
        counter.add("phase_predicate_compute")
        state = self._decision_copy_phase(
            state,
            level=level,
            phase_bins=phase_bins,
            bracket=bracket,
        )
        state = self._predicate_compute(state, phase_bins, bracket)
        counter.add("phase_predicate_uncompute")
        state = self._prepare_and_qpe(
            state,
            shape=self.shape,
            phase_axis=self._PHASE,
            index_axis=self._INDEX,
            reward_axis=self._REWARD,
            active_axis=self._ACTIVE,
            control_axis=self._CONTROL,
            phase_qubits=phase_qubits,
            inverse=True,
            tag=tag,
            counter=counter,
        )
        state = self._update_active_from_stop(state, level=level)
        counter.add("active_stop_update")
        return state

    def apply_layer_unitary(
        self,
        state: ComplexState,
        *,
        level: int,
        bracket: MeasuredBoundaryBracket,
        inverse: bool = False,
    ) -> ComplexState:
        """Apply one full compute-copy/phase-uncompute layer or its inverse."""

        level = _integer(level, "level")
        if level >= self.level_count:
            raise IndexError("level is outside the configured history")
        if not isinstance(bracket, MeasuredBoundaryBracket):
            raise TypeError("bracket must be a MeasuredBoundaryBracket")
        if not bracket.complete:
            raise ValueError("cannot execute a history layer without a boundary bracket")
        if not isinstance(inverse, bool):
            raise TypeError("inverse must be bool")
        values = self._validated_state(state, self.shape)
        return self._apply_layer_circuit(
            values,
            level=level,
            bracket=bracket,
            inverse=inverse,
            counter=self._counter,
        )

    def _boundary_localize(self) -> MeasuredBoundaryBracket:
        q = self.config.boundary_phase_qubits
        phase_bins = 1 << q
        shape = (phase_bins, self.oracle.index_dimension, 2, 2, 2)
        phase_axis, index_axis, reward_axis, active_axis, control_axis = range(5)
        samples: list[list[int]] = [[] for _ in range(self.oracle.n_arms)]
        before = self.oracle.query_snapshot()
        for shot in range(self.config.boundary_shots):
            state = np.zeros(shape, dtype=np.complex128)
            state[0, : self.oracle.n_arms, 0, 1, 0] = 1.0 / math.sqrt(self.oracle.n_arms)
            state = self._prepare_and_qpe(
                state.reshape(-1),
                shape=shape,
                phase_axis=phase_axis,
                index_axis=index_axis,
                reward_axis=reward_axis,
                active_axis=active_axis,
                control_axis=control_axis,
                phase_qubits=q,
                inverse=False,
                tag=f"coherent_boundary_histogram_shot_{shot}",
                counter=self._counter,
            )
            probabilities = np.abs(state) ** 2
            probabilities /= probabilities.sum()
            measured = int(self._rng.choice(probabilities.size, p=probabilities))
            phase, index, _reward, _active, control = np.unravel_index(measured, shape)
            if control != 0:
                raise RuntimeError("boundary QPE left the oracle-control qubit dirty")
            if index < self.oracle.n_arms:
                samples[index].append(min(phase, phase_bins - phase))
        after = self.oracle.query_snapshot()
        query_counts = _immutable_counts(QueryLedger.difference(after, before))

        records: list[BoundaryArmHistogram] = []
        modal_values: list[float | None] = []
        radii: list[float | None] = []
        for arm, row in enumerate(samples):
            counts = Counter(row)
            modal = min(counts, key=lambda value: (-counts[value], value)) if counts else None
            amplitude = None if modal is None else math.sin(math.pi * modal / phase_bins) ** 2
            radius = (
                None
                if amplitude is None
                else max(
                    (abs(math.sin(math.pi * value / phase_bins) ** 2 - amplitude) for value in row),
                    default=0.0,
                )
            )
            records.append(
                BoundaryArmHistogram(
                    arm=arm,
                    samples=len(row),
                    folded_bin_counts=MappingProxyType(
                        {str(key): int(counts[key]) for key in sorted(counts)}
                    ),
                    modal_folded_bin=modal,
                    modal_amplitude=amplitude,
                    empirical_radius=radius,
                )
            )
            modal_values.append(amplitude)
            radii.append(radius)

        if any(len(row) < self.config.minimum_boundary_samples_per_arm for row in samples):
            return MeasuredBoundaryBracket(
                lower=None,
                upper=None,
                center=None,
                complete=False,
                status="insufficient_measured_samples_fail_closed",
                arm_histograms=tuple(records),
                query_counts=query_counts,
            )
        ranking = sorted(
            range(self.oracle.n_arms),
            key=lambda arm: (-(modal_values[arm] or 0.0), arm),
        )
        inside = ranking[self.k - 1]
        outside = ranking[self.k]
        inside_value = modal_values[inside]
        outside_value = modal_values[outside]
        assert inside_value is not None and outside_value is not None
        if math.isclose(inside_value, outside_value, abs_tol=1e-15):
            return MeasuredBoundaryBracket(
                lower=None,
                upper=None,
                center=None,
                complete=False,
                status="measured_boundary_tie_fail_closed",
                arm_histograms=tuple(records),
                query_counts=query_counts,
            )
        center = 0.5 * (inside_value + outside_value)
        inside_radius = radii[inside] or 0.0
        outside_radius = radii[outside] or 0.0
        radius = 0.5 * (inside_radius + outside_radius)
        lower = max(0.0, center - radius)
        upper = min(1.0, center + radius)
        if lower >= upper and radius > 0.0:
            return MeasuredBoundaryBracket(
                lower=None,
                upper=None,
                center=None,
                complete=False,
                status="overlapping_empirical_boundary_histograms_fail_closed",
                arm_histograms=tuple(records),
                query_counts=query_counts,
            )
        return MeasuredBoundaryBracket(
            lower=lower,
            upper=upper,
            center=center,
            complete=True,
            status="measured_histogram_bracket_no_confidence_theorem",
            arm_histograms=tuple(records),
            query_counts=query_counts,
        )

    def _probability_active(self, state: ComplexState) -> float:
        view = state.reshape(self.shape)
        return float(np.sum(np.abs(np.take(view, 1, axis=self._ACTIVE)) ** 2))

    def _workspace_residuals(self, state: ComplexState) -> tuple[float, float, float]:
        view = state.reshape(self.shape)
        predicate = float(np.linalg.norm(np.take(view, 1, axis=self._PREDICATE).reshape(-1)))
        control = float(np.linalg.norm(np.take(view, 1, axis=self._CONTROL).reshape(-1)))
        phase_reward = view.copy()
        selector = [slice(None)] * len(self.shape)
        selector[self._PHASE] = 0
        selector[self._REWARD] = 0
        clean = np.zeros_like(phase_reward)
        clean[tuple(selector)] = phase_reward[tuple(selector)]
        residual = float(np.linalg.norm((phase_reward - clean).reshape(-1)))
        return predicate, control, residual

    def _existing_stopped_residual(
        self, before: ComplexState, after: ComplexState, *, level: int
    ) -> float:
        if level == 0:
            return 0.0
        # Codes 1..level were born strictly before the current layer.  The
        # current code ``level + 1`` is excluded so newly stopped amplitude is
        # not mistaken for a mutation of an old stopped branch.
        left_active = np.take(before.reshape(self.shape), 0, axis=self._ACTIVE)
        right_active = np.take(after.reshape(self.shape), 0, axis=self._ACTIVE)
        stop_axis_after_active = self._STOP - 1
        old_codes = tuple(range(1, level + 1))
        left = np.take(left_active, old_codes, axis=stop_axis_after_active)
        right = np.take(right_active, old_codes, axis=stop_axis_after_active)
        return float(np.linalg.norm((right - left).reshape(-1)))

    def run(self) -> CoherentHistoryStatevectorResult:
        before_run = self.oracle.query_snapshot()
        gate_before_run = Counter(self._counter.gates)
        depth_before_run = self._counter.depth
        boundary = self._boundary_localize()
        after_boundary = self.oracle.query_snapshot()
        state = self.initial_state()
        layers: list[CoherentHistoryLayer] = []

        if boundary.complete:
            for level, phase_qubits in enumerate(self.config.phase_qubits_by_level):
                active_before = self._probability_active(state)
                stopped_before = state.copy()
                query_before = self.oracle.query_snapshot()
                gates_before = Counter(self._counter.gates)
                depth_before = self._counter.depth
                state = self._apply_layer_circuit(
                    state,
                    level=level,
                    bracket=boundary,
                    inverse=False,
                    counter=self._counter,
                )
                norm_error = abs(float(np.linalg.norm(state)) - 1.0)
                predicate, control, phase_reward = self._workspace_residuals(state)
                active_after = self._probability_active(state)
                query_counts = _immutable_counts(
                    QueryLedger.difference(self.oracle.query_snapshot(), query_before)
                )
                layer_gates = Counter(self._counter.gates)
                layer_gates.subtract(gates_before)
                layer = CoherentHistoryLayer(
                    level=level,
                    phase_qubits=phase_qubits,
                    active_probability_before=active_before,
                    active_probability_after=active_after,
                    newly_stopped_probability=max(0.0, active_before - active_after),
                    existing_stopped_branch_residual=self._existing_stopped_residual(
                        stopped_before, state, level=level
                    ),
                    predicate_workspace_residual=predicate,
                    control_workspace_residual=control,
                    phase_reward_workspace_residual=phase_reward,
                    norm_error=norm_error,
                    query_counts=query_counts,
                    depth=self._counter.depth - depth_before,
                    gate_counts=_immutable_counts(
                        {key: value for key, value in layer_gates.items() if value}
                    ),
                )
                layers.append(layer)
                if (
                    max(
                        norm_error,
                        predicate,
                        control,
                        phase_reward,
                        layer.existing_stopped_branch_residual,
                    )
                    > self.config.cleanup_tolerance
                ):
                    break

        after_run = self.oracle.query_snapshot()
        total_counts = _immutable_counts(QueryLedger.difference(after_run, before_run))
        history_counts = _immutable_counts(QueryLedger.difference(after_run, after_boundary))
        total_gates = Counter(self._counter.gates)
        total_gates.subtract(gate_before_run)

        view = state.reshape(self.shape)
        stop_probabilities = tuple(
            float(np.sum(np.abs(np.take(view, code, axis=self._STOP)) ** 2))
            for code in range(self.level_count + 1)
        )
        output_probabilities = tuple(
            float(np.sum(np.abs(np.take(view, mask, axis=self._OUTPUT)) ** 2))
            for mask in range(self.output_dimension)
        )
        nonzero_outputs = tuple(
            index
            for index, probability in enumerate(output_probabilities)
            if probability > self.config.cleanup_tolerance**2
        )
        direct_complete = bool(
            len(nonzero_outputs) == 1
            and nonzero_outputs[0].bit_count() == self.k
            and math.isclose(
                output_probabilities[nonzero_outputs[0]],
                1.0,
                abs_tol=self.config.cleanup_tolerance,
            )
        )
        cleanup_ok = all(
            max(
                layer.predicate_workspace_residual,
                layer.control_workspace_residual,
                layer.phase_reward_workspace_residual,
                layer.norm_error,
                layer.existing_stopped_branch_residual,
            )
            <= self.config.cleanup_tolerance
            for layer in layers
        )
        blockers = [
            "measured_boundary_histogram_has_no_finite_sample_confidence_theorem",
        ]
        if not boundary.complete:
            blockers.append(boundary.status)
        if not direct_complete:
            blockers.append(
                "branch_local_output_mask_is_not_complete_direct_multi_output_extraction"
            )
        if not cleanup_ok:
            blockers.append("workspace_cleanup_or_stopped_branch_invariance_failed")
        blockers.extend(
            (
                "no_variable_time_query_upper_bound",
                "no_matching_same_interface_lower_bound",
            )
        )
        resources = CoherentHistoryStatevectorResources(
            query_counts=total_counts,
            boundary_query_counts=boundary.query_counts,
            history_query_counts=history_counts,
            gate_counts=_immutable_counts(
                {key: value for key, value in total_gates.items() if value}
            ),
            depth=self._counter.depth - depth_before_run,
            qubits=self.qubits,
            register_dimensions=self.register_dimensions,
            retained_statevector_dimension=self.statevector_dimension,
            peak_statevector_dimension=max(
                self.statevector_dimension,
                (1 << self.config.boundary_phase_qubits) * self.oracle.index_dimension * 8,
            ),
        )
        return CoherentHistoryStatevectorResult(
            state=state,
            boundary=boundary,
            layers=tuple(layers),
            resources=resources,
            active_probability=self._probability_active(state),
            stop_probabilities=stop_probabilities,
            output_mask_probabilities=output_probabilities,
            direct_output_write_executed=bool(
                total_gates.get("coherent_branch_output_mask_xor", 0)
            ),
            direct_multi_output_complete=direct_complete,
            certificate_issued=False,
            blockers=tuple(dict.fromkeys(blockers)),
            status=(
                "exact_state_execution_complete_certificate_refused"
                if boundary.complete and cleanup_ok
                else "exact_state_execution_incomplete_fail_closed"
            ),
        )


def run_coherent_activity_history_statevector(
    oracle: CanonicalRyStatevectorOracle,
    k: int,
    *,
    config: CoherentHistoryStatevectorConfig | None = None,
) -> CoherentHistoryStatevectorResult:
    """Convenience entry point for the exact-state code-sanity experiment."""

    return CoherentActivityHistoryStatevectorKernel(oracle, k, config=config).run()
