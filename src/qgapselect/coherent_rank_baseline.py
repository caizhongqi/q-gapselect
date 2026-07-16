"""Measured-QPE direct Top-k baseline for tiny canonical-oracle instances.

This module supplies a deliberately expensive, legal same-interface baseline.
It receives only a canonical coherent reward oracle, ``k``, and public circuit
precision/shot limits.  Each arm is queried by an exact-state QPE circuit,
the phase register is measured, and a discrete modal rank is formed.  A
no-QRAM exhaustive reversible relation then executes

``compute membership -> copy all n durable bits -> uncompute membership``.

For generic amplitudes, copying a nonconstant function of an unmeasured QPE
superposition prevents exact inverse-QPE cleanup.  The baseline therefore
charges destructive phase measurement and reset before the rank unitary.  It
is not an end-to-end coherent algorithm, has no finite-shot correctness
theorem, and cannot support a quantum-advantage claim.  Those limitations are
returned as fail-closed blockers rather than hidden by the simulator.
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

BACKEND = "numpy_exact_statevector_measured_qpe_exhaustive_rank_baseline"
CLAIM_SCOPE = (
    "legal_tiny_baseline_no_end_to_end_coherence_no_confidence_theorem_"
    "no_query_advantage"
)
RANK_COMPILATION_MODEL = "exhaustive_multicontrolled_rank_relation_no_qram"
QUERY_COUNT_SEMANTICS = "executed_canonical_oracle_calls_plus_explicit_nonoracle_charges"
GATE_COUNT_SEMANTICS = "logical_no_ancilla_multicontrolled_upper_bound"
DEPTH_SEMANTICS = "fully_serial_logical_upper_bound"


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


@dataclass(frozen=True, slots=True)
class CoherentRankBaselineConfig:
    """Public precision, sampling, cleanup, and tiny-state limits."""

    phase_qubits: int = 3
    shots_per_arm: int = 32
    measurement_seed: int = 0
    cleanup_tolerance: float = 1e-12
    max_statevector_dimension: int = 8_388_608

    def __post_init__(self) -> None:
        phase_qubits = _integer(self.phase_qubits, "phase_qubits", minimum=1)
        if phase_qubits > 4:
            raise ValueError("phase_qubits exceeds the exact-state baseline limit of 4")
        object.__setattr__(self, "phase_qubits", phase_qubits)
        object.__setattr__(
            self,
            "shots_per_arm",
            _integer(self.shots_per_arm, "shots_per_arm", minimum=1),
        )
        object.__setattr__(
            self,
            "measurement_seed",
            _integer(self.measurement_seed, "measurement_seed"),
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
        if isinstance(self.cleanup_tolerance, bool):
            raise TypeError("cleanup_tolerance must be a positive finite real")
        tolerance = float(self.cleanup_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("cleanup_tolerance must be a positive finite real")
        object.__setattr__(self, "cleanup_tolerance", tolerance)


@dataclass(frozen=True, slots=True)
class ArmDiscretePhaseEstimate:
    """One arm's measured folded-QPE estimate."""

    arm: int
    shots: int
    folded_bin_counts: Mapping[str, int]
    modal_folded_bin: int
    modal_count: int
    modal_unique: bool
    amplitude_estimate: float


@dataclass(frozen=True, slots=True)
class DiscreteRankBoundary:
    """Boundary obtained only from measured discrete estimates."""

    inside_arm: int | None
    outside_arm: int | None
    inside_folded_bin: int | None
    outside_folded_bin: int | None
    strict: bool
    status: str
    information_source: str = "per_arm_measured_qpe_modal_bins"
    finite_sample_confidence_proved: bool = False


@dataclass(frozen=True, slots=True)
class RankCleanupLedger:
    """Pure rank-unitary cleanup plus the separately charged QPE reset."""

    expected_output_residual_l2: float
    membership_work_nonzero_probability: float
    comparator_nonzero_probability: float
    norm_error: float
    phase_measurements: int
    phase_reward_resets: int
    measurement_reset_charged: bool
    rank_unitary_cleanup_passed: bool
    end_to_end_unitary_cleanup_proved: bool
    tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.expected_output_residual_l2 <= self.tolerance
            and self.membership_work_nonzero_probability <= self.tolerance
            and self.comparator_nonzero_probability <= self.tolerance
            and self.norm_error <= self.tolerance
            and self.measurement_reset_charged
            and self.rank_unitary_cleanup_passed
        )


@dataclass(frozen=True, slots=True)
class CoherentRankResourceLedger:
    """Executed oracle counts and explicit no-free-QRAM circuit charges."""

    query_counts: Mapping[str, int]
    gate_counts: Mapping[str, int]
    depth: int
    qubits: int
    register_dimensions: Mapping[str, int]
    retained_statevector_dimension: int
    peak_statevector_dimension: int
    rank_truth_table_rows: int
    rank_compilation_model: str
    query_count_semantics: str
    gate_count_semantics: str
    depth_semantics: str
    qram_assumed: bool
    cleanup: RankCleanupLedger
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("coherent_total", 0))


@dataclass(frozen=True, slots=True)
class CoherentRankBaselineResult:
    """Fail-closed direct-membership result of the legal tiny baseline."""

    state: ComplexState
    estimates: tuple[ArmDiscretePhaseEstimate, ...]
    ranking: tuple[int, ...]
    boundary: DiscreteRankBoundary
    membership_bits: tuple[int, ...]
    membership_mask: int | None
    estimate_register_code: int
    resources: CoherentRankResourceLedger
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
        self.gates[name] += int(count)
        self.depth += int(count if depth is None else depth)


class CoherentMeasuredQPERankBaseline:
    """Exhaustive all-arm QPE rank baseline with complete n-bit output.

    The constructor intentionally has no boundary value, answer set, or
    amplitude-vector input.  Oracle information is obtained only by charged
    calls to :class:`CanonicalRyStatevectorOracle`.
    """

    _ESTIMATE = 0
    _OUTPUT = 1
    _WORK = 2
    _COMPARATOR = 3

    def __init__(
        self,
        oracle: CanonicalRyStatevectorOracle,
        k: int,
        *,
        config: CoherentRankBaselineConfig | None = None,
    ) -> None:
        if not isinstance(oracle, CanonicalRyStatevectorOracle):
            raise TypeError("oracle must be a CanonicalRyStatevectorOracle")
        if oracle.n_arms > 4:
            raise ValueError("the exact-state rank baseline limit is n <= 4")
        k = _integer(k, "k", minimum=1)
        if k >= oracle.n_arms:
            raise ValueError("k must satisfy 1 <= k < oracle.n_arms")
        if config is not None and not isinstance(config, CoherentRankBaselineConfig):
            raise TypeError("config must be CoherentRankBaselineConfig")
        self.oracle = oracle
        self.k = k
        self.config = config or CoherentRankBaselineConfig()
        self.phase_bins = 1 << self.config.phase_qubits
        self.estimate_dimension = self.phase_bins**oracle.n_arms
        self.output_dimension = 1 << oracle.n_arms
        self.work_dimension = self.output_dimension
        self.comparator_dimension = 2
        self.shape = (
            self.estimate_dimension,
            self.output_dimension,
            self.work_dimension,
            self.comparator_dimension,
        )
        self.statevector_dimension = math.prod(self.shape)
        self.qpe_shape = (self.phase_bins, oracle.index_dimension, 2, 2)
        self.qpe_statevector_dimension = math.prod(self.qpe_shape)
        self.peak_statevector_dimension = max(
            self.statevector_dimension,
            self.qpe_statevector_dimension,
        )
        if self.peak_statevector_dimension > self.config.max_statevector_dimension:
            raise ValueError("explicit register statevector exceeds max_statevector_dimension")
        self._rng = np.random.default_rng(self.config.measurement_seed)
        self._counter = _CircuitCounter()

    @property
    def register_dimensions(self) -> Mapping[str, int]:
        return _frozen_counts(
            {
                "estimate_pack": self.estimate_dimension,
                "durable_membership_output": self.output_dimension,
                "membership_work": self.work_dimension,
                "comparator": self.comparator_dimension,
            }
        )

    @property
    def qubits(self) -> int:
        rank_qubits = sum(int(math.log2(value)) for value in self.shape)
        qpe_qubits = sum(int(math.log2(value)) for value in self.qpe_shape)
        return max(rank_qubits, qpe_qubits)

    @staticmethod
    def _validated_state(state: ComplexState, shape: tuple[int, ...]) -> ComplexState:
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.size != math.prod(shape):
            raise ValueError(
                f"expected a flat statevector of length {math.prod(shape)}, got {values.shape}"
            )
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-10):
            raise ValueError("statevector must be normalized")
        return values.copy()

    @staticmethod
    def _xor_phase_control(
        state: ComplexState,
        *,
        shape: tuple[int, ...],
        phase_values: tuple[int, ...],
    ) -> ComplexState:
        view = state.reshape(shape)
        for phase in phase_values:
            block = view[phase, :, :, :].copy()
            view[phase, :, :, 0] = block[:, :, 1]
            view[phase, :, :, 1] = block[:, :, 0]
        return view.reshape(-1)

    @staticmethod
    def _controlled_reward_reflection(
        state: ComplexState,
        *,
        shape: tuple[int, ...],
        reward_value: int,
    ) -> ComplexState:
        view = state.copy().reshape(shape)
        view[:, :, reward_value, 1] *= -1.0
        return view.reshape(-1)

    @staticmethod
    def _controlled_global_minus(
        state: ComplexState,
        *,
        shape: tuple[int, ...],
    ) -> ComplexState:
        view = state.copy().reshape(shape)
        view[:, :, :, 1] *= -1.0
        return view.reshape(-1)

    def _controlled_grover(
        self,
        state: ComplexState,
        *,
        phase_values: tuple[int, ...],
        arm: int,
        shot: int,
    ) -> ComplexState:
        tag = f"coherent_rank_qpe_arm_{arm}_shot_{shot}"
        state = self._xor_phase_control(
            state,
            shape=self.qpe_shape,
            phase_values=phase_values,
        )
        self._counter.add("phase_control_compute")
        state = self._controlled_reward_reflection(
            state,
            shape=self.qpe_shape,
            reward_value=1,
        )
        self._counter.add("controlled_good_reflection")
        state = self.oracle.apply_embedded(
            state,
            register_shape=self.qpe_shape,
            index_axis=1,
            reward_axis=2,
            control_axis=3,
            inverse=True,
            tag=tag,
        )
        self._counter.add("controlled_reward_oracle_inverse")
        state = self._controlled_reward_reflection(
            state,
            shape=self.qpe_shape,
            reward_value=0,
        )
        self._counter.add("controlled_zero_reflection")
        state = self.oracle.apply_embedded(
            state,
            register_shape=self.qpe_shape,
            index_axis=1,
            reward_axis=2,
            control_axis=3,
            tag=tag,
        )
        self._counter.add("controlled_reward_oracle_forward")
        state = self._controlled_global_minus(state, shape=self.qpe_shape)
        self._counter.add("controlled_global_phase")
        state = self._xor_phase_control(
            state,
            shape=self.qpe_shape,
            phase_values=phase_values,
        )
        self._counter.add("phase_control_uncompute")
        return state

    def _qpe_shot(self, arm: int, shot: int) -> int:
        state = np.zeros(self.qpe_shape, dtype=np.complex128)
        state[0, arm, 0, 0] = 1.0
        # The public loop index is compiled into an ordinary basis state.
        # Both preparation and reset are charged; there is no indexed memory.
        index_x_count = arm.bit_count()
        self._counter.add("compiled_index_basis_x", index_x_count)
        tag = f"coherent_rank_qpe_arm_{arm}_shot_{shot}"
        state = self.oracle.apply_embedded(
            state.reshape(-1),
            register_shape=self.qpe_shape,
            index_axis=1,
            reward_axis=2,
            tag=tag,
        )
        self._counter.add("reward_oracle_forward")

        scale = 1.0 / math.sqrt(2.0)
        for bit in range(self.config.phase_qubits):
            mask = 1 << (self.config.phase_qubits - 1 - bit)
            view = state.reshape(self.qpe_shape)
            for low in range(self.phase_bins):
                if low & mask:
                    continue
                high = low | mask
                left = view[low].copy()
                right = view[high].copy()
                view[low] = (left + right) * scale
                view[high] = (left - right) * scale
            state = view.reshape(-1)
            self._counter.add("phase_hadamard")

        for bit in range(self.config.phase_qubits):
            mask = 1 << (self.config.phase_qubits - 1 - bit)
            phase_values = tuple(
                phase for phase in range(self.phase_bins) if phase & mask
            )
            for _ in range(mask):
                state = self._controlled_grover(
                    state,
                    phase_values=phase_values,
                    arm=arm,
                    shot=shot,
                )

        indices = np.arange(self.phase_bins)
        matrix = np.exp(
            -2j
            * math.pi
            * np.outer(indices, indices)
            / self.phase_bins
        ) / math.sqrt(self.phase_bins)
        view = state.reshape(self.qpe_shape)
        view[:] = np.einsum("ab,birc->airc", matrix, view, optimize=True)
        state = view.reshape(-1)
        self._counter.add(
            "inverse_qft",
            depth=self.config.phase_qubits * (self.config.phase_qubits + 1) // 2,
        )

        view = state.reshape(self.qpe_shape)
        control_probability = float(np.sum(np.abs(view[:, :, :, 1]) ** 2))
        wrong_index_probability = float(
            np.sum(np.abs(np.delete(view, arm, axis=1)) ** 2)
        )
        if control_probability > self.config.cleanup_tolerance:
            raise RuntimeError("QPE oracle-control qubit was not uncomputed")
        if wrong_index_probability > self.config.cleanup_tolerance:
            raise RuntimeError("QPE changed the compiled arm index")
        phase_probabilities = np.sum(np.abs(view[:, arm, :, 0]) ** 2, axis=1)
        phase_probabilities /= phase_probabilities.sum()
        measured = int(self._rng.choice(self.phase_bins, p=phase_probabilities))
        self._counter.add("phase_measurement", self.config.phase_qubits)
        self._counter.add("destructive_phase_reward_reset", self.config.phase_qubits + 1)
        self._counter.add("compiled_index_basis_reset_x", index_x_count)
        return min(measured, self.phase_bins - measured)

    def _measure_estimates(self) -> tuple[ArmDiscretePhaseEstimate, ...]:
        rows: list[ArmDiscretePhaseEstimate] = []
        for arm in range(self.oracle.n_arms):
            counts: Counter[int] = Counter(
                self._qpe_shot(arm, shot)
                for shot in range(self.config.shots_per_arm)
            )
            modal_count = max(counts.values())
            modes = tuple(
                sorted(
                    bin_value
                    for bin_value, count in counts.items()
                    if count == modal_count
                )
            )
            modal = modes[0]
            rows.append(
                ArmDiscretePhaseEstimate(
                    arm=arm,
                    shots=self.config.shots_per_arm,
                    folded_bin_counts=MappingProxyType(
                        {str(key): int(counts[key]) for key in sorted(counts)}
                    ),
                    modal_folded_bin=modal,
                    modal_count=modal_count,
                    modal_unique=len(modes) == 1,
                    amplitude_estimate=math.sin(math.pi * modal / self.phase_bins) ** 2,
                )
            )
        return tuple(rows)

    def _decode_estimate_code(self, code: int) -> tuple[int, ...]:
        return tuple(
            (code >> (arm * self.config.phase_qubits)) & (self.phase_bins - 1)
            for arm in range(self.oracle.n_arms)
        )

    def _rank_mask_for_code(self, code: int) -> int:
        bins = self._decode_estimate_code(code)
        folded = tuple(min(value, self.phase_bins - value) for value in bins)
        ranking = sorted(
            range(self.oracle.n_arms),
            key=lambda arm: (-folded[arm], arm),
        )
        return sum(1 << arm for arm in ranking[: self.k])

    def encode_estimate_register(self, folded_bins: tuple[int, ...]) -> int:
        if len(folded_bins) != self.oracle.n_arms:
            raise ValueError("folded_bins must contain one value per arm")
        result = 0
        for arm, value in enumerate(folded_bins):
            value = _integer(value, f"folded_bins[{arm}]")
            if value > self.phase_bins // 2:
                raise ValueError("folded bin lies outside the folded phase range")
            result |= value << (arm * self.config.phase_qubits)
        return result

    def initial_rank_state(self, estimate_code: int) -> ComplexState:
        estimate_code = _integer(estimate_code, "estimate_code")
        if estimate_code >= self.estimate_dimension:
            raise ValueError("estimate_code is outside the estimate register")
        state = np.zeros(self.shape, dtype=np.complex128)
        state[estimate_code, 0, 0, 0] = 1.0
        return state.reshape(-1)

    def apply_rank_compute(self, state: ComplexState) -> ComplexState:
        """XOR the compiled rank-membership relation into transient work."""

        source = self._validated_state(state, self.shape).reshape(self.shape)
        target = np.zeros_like(source)
        for estimate in range(self.estimate_dimension):
            permutation = np.arange(self.work_dimension) ^ self._rank_mask_for_code(estimate)
            target[estimate][:, permutation, :] = source[estimate]
        return target.reshape(-1)

    def apply_output_copy(self, state: ComplexState) -> ComplexState:
        """Copy every membership-work bit into the durable output mask."""

        source = self._validated_state(state, self.shape).reshape(self.shape)
        target = np.zeros_like(source)
        output_values = np.arange(self.output_dimension)
        for work in range(self.work_dimension):
            target[:, output_values ^ work, work, :] = source[:, :, work, :]
        return target.reshape(-1)

    def _expected_output(self, state: ComplexState) -> ComplexState:
        source = self._validated_state(state, self.shape).reshape(self.shape)
        target = np.zeros_like(source)
        output_values = np.arange(self.output_dimension)
        for estimate in range(self.estimate_dimension):
            membership = self._rank_mask_for_code(estimate)
            target[estimate, output_values ^ membership, :, :] = source[estimate]
        return target.reshape(-1)

    def _cleanup_ledger(
        self,
        initial: ComplexState,
        final: ComplexState,
        *,
        rank_executed: bool,
    ) -> RankCleanupLedger:
        view = final.reshape(self.shape)
        work_clean_probability = float(np.sum(np.abs(view[:, :, 0, :]) ** 2))
        comparator_clean_probability = float(np.sum(np.abs(view[:, :, :, 0]) ** 2))
        expected = self._expected_output(initial) if rank_executed else initial
        tolerance = self.config.cleanup_tolerance
        residual = float(np.linalg.norm(final - expected))
        work_residual = max(0.0, 1.0 - work_clean_probability)
        comparator_residual = max(0.0, 1.0 - comparator_clean_probability)
        norm_error = abs(float(np.linalg.norm(final)) - 1.0)
        return RankCleanupLedger(
            expected_output_residual_l2=residual,
            membership_work_nonzero_probability=work_residual,
            comparator_nonzero_probability=comparator_residual,
            norm_error=norm_error,
            phase_measurements=self.oracle.n_arms * self.config.shots_per_arm,
            phase_reward_resets=self.oracle.n_arms * self.config.shots_per_arm,
            measurement_reset_charged=True,
            rank_unitary_cleanup_passed=max(
                residual,
                work_residual,
                comparator_residual,
                norm_error,
            )
            <= tolerance,
            end_to_end_unitary_cleanup_proved=False,
            tolerance=tolerance,
        )

    def _resource_ledger(
        self,
        query_counts: Mapping[str, int],
        cleanup: RankCleanupLedger,
        *,
        rank_executed: bool,
    ) -> CoherentRankResourceLedger:
        n = self.oracle.n_arms
        estimate_bits = n * self.config.phase_qubits
        rank_terms = 2 * self.estimate_dimension * self.k if rank_executed else 0
        # Each no-ancilla multi-controlled membership flip is charged as a
        # serial logical upper bound.  This deliberately makes the legal
        # baseline expensive instead of granting free table/QRAM access.
        rank_primitives = rank_terms * (2 * estimate_bits + 1)
        gates = Counter(self._counter.gates)
        gates.update(
            {
                "compiled_rank_truth_table_rows": (
                    2 * self.estimate_dimension if rank_executed else 0
                ),
                "no_ancilla_multicontrolled_rank_primitives": rank_primitives,
                "durable_membership_cnot": n if rank_executed else 0,
                "classical_estimate_register_load": estimate_bits,
            }
        )
        nonoracle = {
            "qram_queries": 0,
            "compiled_rank_relation_calls": 2 if rank_executed else 0,
            "destructive_phase_measurements": cleanup.phase_measurements,
            "phase_reward_resets": cleanup.phase_reward_resets,
        }
        combined_queries = dict(query_counts)
        combined_queries.update(nonoracle)
        depth = (
            self._counter.depth
            + rank_primitives
            + (n if rank_executed else 0)
            + estimate_bits
        )
        return CoherentRankResourceLedger(
            query_counts=_frozen_counts(combined_queries),
            gate_counts=_frozen_counts(gates),
            depth=depth,
            qubits=self.qubits,
            register_dimensions=self.register_dimensions,
            retained_statevector_dimension=self.statevector_dimension,
            peak_statevector_dimension=self.peak_statevector_dimension,
            rank_truth_table_rows=self.estimate_dimension,
            rank_compilation_model=RANK_COMPILATION_MODEL,
            query_count_semantics=QUERY_COUNT_SEMANTICS,
            gate_count_semantics=GATE_COUNT_SEMANTICS,
            depth_semantics=DEPTH_SEMANTICS,
            qram_assumed=False,
            cleanup=cleanup,
        )

    def run(self) -> CoherentRankBaselineResult:
        before = self.oracle.query_snapshot()
        estimates = self._measure_estimates()
        after = self.oracle.query_snapshot()
        query_counts = _frozen_counts(QueryLedger.difference(after, before))
        folded = tuple(row.modal_folded_bin for row in estimates)
        ranking = tuple(
            sorted(
                range(self.oracle.n_arms),
                key=lambda arm: (-folded[arm], arm),
            )
        )
        inside = ranking[self.k - 1]
        outside = ranking[self.k]
        all_modes_unique = all(row.modal_unique for row in estimates)
        strict = folded[inside] > folded[outside] and all_modes_unique
        boundary = DiscreteRankBoundary(
            inside_arm=inside,
            outside_arm=outside,
            inside_folded_bin=folded[inside],
            outside_folded_bin=folded[outside],
            strict=strict,
            status=(
                "strict_measured_discrete_boundary_no_confidence_theorem"
                if strict
                else "ambiguous_measured_discrete_boundary_fail_closed"
            ),
        )
        estimate_code = self.encode_estimate_register(folded)
        initial = self.initial_rank_state(estimate_code)
        if strict:
            computed = self.apply_rank_compute(initial)
            copied = self.apply_output_copy(computed)
            final = self.apply_rank_compute(copied)
            membership_mask: int | None = self._rank_mask_for_code(estimate_code)
            membership_bits = tuple(
                (membership_mask >> arm) & 1 for arm in range(self.oracle.n_arms)
            )
        else:
            final = initial
            membership_mask = None
            membership_bits = ()
        cleanup = self._cleanup_ledger(initial, final, rank_executed=strict)
        resources = self._resource_ledger(
            query_counts,
            cleanup,
            rank_executed=strict,
        )
        blockers = [
            "qpe_measurement_and_reset_breaks_end_to_end_coherence",
            "per_arm_classical_loop_is_not_a_coherent_index_superposition",
            "finite_shot_modal_rank_has_no_confidence_theorem",
            "exhaustive_all_arm_qpe_is_a_legal_baseline_not_a_variable_time_speedup",
            "exhaustive_rank_relation_has_exponential_compilation_cost",
            "no_same_interface_composition_separation",
            "no_matching_lower_bound",
        ]
        if not all_modes_unique:
            blockers.append("nonunique_per_arm_modal_histogram_fail_closed")
        if folded[inside] <= folded[outside]:
            blockers.append("measured_top_k_boundary_tie_fail_closed")
        complete = strict and cleanup.passed
        status = (
            "legal_direct_membership_baseline_complete_theorem_blocked"
            if complete
            else "legal_rank_baseline_ambiguous_or_cleanup_failed"
        )
        return CoherentRankBaselineResult(
            state=final,
            estimates=estimates,
            ranking=ranking,
            boundary=boundary,
            membership_bits=membership_bits,
            membership_mask=membership_mask,
            estimate_register_code=estimate_code,
            resources=resources,
            direct_multi_output_complete=complete,
            certificate_issued=False,
            blockers=tuple(blockers),
            status=status,
        )


def run_coherent_rank_baseline(
    oracle: CanonicalRyStatevectorOracle,
    k: int,
    *,
    config: CoherentRankBaselineConfig | None = None,
) -> CoherentRankBaselineResult:
    """Execute the tiny measured-QPE legal rank baseline."""

    return CoherentMeasuredQPERankBaseline(oracle, k, config=config).run()
