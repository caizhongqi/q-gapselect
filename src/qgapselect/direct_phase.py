"""Direct small-state amplitude-threshold reflection from an unknown oracle.

The implementation in this module constructs a genuine coherent circuit from
``CanonicalRyStatevectorOracle.apply_embedded`` calls.  It never reads an
arm's hidden mean or private rotation block.  The public reflection retains
the complete phase/index/reward Hilbert space, including finite-precision QPE
leakage; downstream search code must not project that workspace away.

This is exact NumPy reference semantics for small states.  Its resource ledger
records logical oracle calls and circuit operations, but it neither supplies
nor claims a new asymptotic query-complexity theorem.
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
from .oracles import QueryKind, QueryLedger, QuerySnapshot

ComplexState = NDArray[np.complex128]
BACKEND = "numpy_exact_statevector_small_scale"
CLAIM_STATUS = "direct_charged_qpe_threshold_reflection_no_complexity_theorem"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _probability(value: object, name: str, *, closed: bool = True) -> float:
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


def _immutable(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


@dataclass(frozen=True, slots=True)
class DirectPhaseFlagResources:
    """Auditable logical resources for one or more direct-QPE operations."""

    query_counts: Mapping[str, int]
    gate_counts: Mapping[str, int]
    qpe_calls: int
    controlled_grover_iterations: int
    depth: int
    qubits: int
    workspace_qubits: int
    phase_qubits: int
    phase_bins: int
    phase_ancilla_residual: float
    zero_workspace_residual: float
    comparator_residual: float
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("coherent_total", 0))


@dataclass(frozen=True, slots=True)
class DirectPhaseFlagResult:
    """One full-workspace threshold-reflection execution."""

    state: ComplexState
    acceptance_probability: float
    resources: DirectPhaseFlagResources


@dataclass(frozen=True, slots=True)
class IndexVerificationResult:
    """Fixed-shot measured verification using a newly charged QPE per shot."""

    index: int
    status: str
    successes: int
    shots: int
    estimate: float
    confidence: float
    interval: tuple[float, float]
    decision_cutoff: float
    resources: DirectPhaseFlagResources

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    @property
    def rejected(self) -> bool:
        return self.status == "rejected"

    @property
    def unresolved(self) -> bool:
        return self.status == "unresolved"


class _Gates:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.depth = 0
        self.qpe_calls = 0
        self.grover_iterations = 0

    def add(self, name: str, count: int = 1, *, depth: int | None = None) -> None:
        self.counts[name] += count
        self.depth += count if depth is None else depth


class DirectAmplitudeThresholdFlag:
    r"""Finite-precision amplitude-threshold reflection built from charged QPE.

    For each arm, the canonical reward oracle prepares success amplitude
    :math:`\sin(\theta_i)` and the Grover iterate has eigenphases
    :math:`\pm 2\theta_i`.  QPE writes a discretized phase into ``phase_qubits``
    ancillas.  A reversible comparator marks bins whose reconstructed
    :math:`\sin^2(\theta)` is at least ``threshold``, applies phase kickback,
    and uncomputes its flag.  Inverse QPE and :math:`A^\dagger` finish the
    reflection.

    ``apply_reflection`` acts on the *complete* register ordered as
    ``(phase_bit_0, ..., phase_bit_m-1, index, reward)``.  It is the exact
    unitary :math:`V^\dagger P V` in the simulated Hilbert space.  Finite-QPE
    leakage in the phase and reward registers is returned, never discarded.
    """

    def __init__(
        self,
        oracle: CanonicalRyStatevectorOracle,
        threshold: float,
        *,
        phase_qubits: int = 5,
        relation: str = "above",
        excluded_indices: Sequence[int] = (),
    ) -> None:
        if not isinstance(oracle, CanonicalRyStatevectorOracle):
            raise TypeError("oracle must be a CanonicalRyStatevectorOracle")
        phase_qubits = _integer(phase_qubits, "phase_qubits")
        if phase_qubits <= 0:
            raise ValueError("phase_qubits must be positive")
        if phase_qubits > 12:
            raise ValueError("phase_qubits exceeds the small-state limit of 12")

        self.oracle = oracle
        self.threshold = _probability(threshold, "threshold")
        self.phase_qubits = phase_qubits
        self.phase_bins = 1 << phase_qubits
        self._shape = (2,) * phase_qubits + (oracle.index_dimension, 2)
        if not isinstance(relation, str):
            raise TypeError("relation must be a string")
        if relation not in {"above", "below"}:
            raise ValueError("relation must be 'above' or 'below'")
        self.relation = relation
        try:
            raw_excluded = tuple(excluded_indices)
        except TypeError as error:
            raise TypeError("excluded_indices must be a sequence of integers") from error
        excluded = tuple(_integer(index, "excluded index") for index in raw_excluded)
        if len(set(excluded)) != len(excluded):
            raise ValueError("excluded_indices must be unique")
        if any(not 0 <= index < oracle.n_arms for index in excluded):
            raise IndexError("an excluded index is outside the valid arm range")
        self.excluded_indices = excluded
        self._marked_bins = tuple(
            phase_bin
            for phase_bin in range(self.phase_bins)
            if (
                math.sin(math.pi * phase_bin / self.phase_bins) ** 2
                >= self.threshold
            )
            == (relation == "above")
        )
        mask = np.zeros(
            (self.phase_bins, self.index_dimension, 2),
            dtype=np.bool_,
        )
        accepted_indices = tuple(
            index
            for index in self.valid_indices
            if index not in self.excluded_indices
        )
        if self._marked_bins and accepted_indices:
            mask[np.ix_(self._marked_bins, accepted_indices, (0, 1))] = True
        mask.flags.writeable = False
        self._acceptance_mask = mask
        self._query_counts: Counter[str] = Counter()
        self._gate_counts: Counter[str] = Counter()
        self._qpe_calls = 0
        self._grover_iterations = 0
        self._depth = 0
        self._max_phase_residual = 0.0
        self._max_zero_residual = 0.0
        self._max_comparator_residual = 0.0
        self._last_result: DirectPhaseFlagResult | None = None
        self._last_resources: DirectPhaseFlagResources | None = None

    @property
    def index_dimension(self) -> int:
        return self.oracle.index_dimension

    @property
    def valid_indices(self) -> tuple[int, ...]:
        return tuple(range(self.oracle.n_arms))

    @property
    def workspace_dimension(self) -> int:
        """Dimension of the retained phase-plus-reward workspace."""

        return 2 * self.phase_bins

    @property
    def statevector_dimension(self) -> int:
        return self.phase_bins * self.index_dimension * 2

    @property
    def register_shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def marked_phase_bins(self) -> tuple[int, ...]:
        return self._marked_bins

    @property
    def acceptance_mask(self) -> NDArray[np.bool_]:
        """Full ``(phase-bin, index, reward)`` basis predicate mask."""

        result = self._acceptance_mask.copy()
        result.flags.writeable = False
        return result

    @property
    def last_result(self) -> DirectPhaseFlagResult | None:
        return self._last_result

    @property
    def last_resources(self) -> DirectPhaseFlagResources | None:
        return self._last_resources

    def prepare_zero_workspace(self, index_state: ComplexState) -> ComplexState:
        """Embed a normalized valid-index state with phase/reward set to zero."""

        values = np.asarray(index_state, dtype=np.complex128)
        if values.ndim != 1 or values.size != self.index_dimension:
            raise ValueError(
                f"index_state must have length {self.index_dimension}"
            )
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-10):
            raise ValueError("index_state must be normalized")
        if any(
            abs(values[index]) > 1e-12
            for index in range(self.oracle.n_arms, self.index_dimension)
        ):
            raise ValueError("index_state cannot have support on padded indices")
        full = np.zeros(self._shape, dtype=np.complex128)
        selector = (0,) * self.phase_qubits + (slice(None), 0)
        full[selector] = values
        return full.reshape(-1)

    def initial_state(self, indices: Sequence[int] | None = None) -> ComplexState:
        """Prepare a uniform valid-index state with all workspace qubits zero."""

        selected = self.valid_indices if indices is None else tuple(indices)
        if not selected:
            raise ValueError("indices cannot be empty")
        checked = tuple(_integer(index, "index") for index in selected)
        if len(set(checked)) != len(checked):
            raise ValueError("indices must be unique")
        if any(index not in self.valid_indices for index in checked):
            raise IndexError("an index is outside the valid arm range")
        index_state = np.zeros(self.index_dimension, dtype=np.complex128)
        index_state[list(checked)] = 1.0 / math.sqrt(len(checked))
        return self.prepare_zero_workspace(index_state)

    def _validated_full_state(self, state: ComplexState) -> ComplexState:
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.size != self.statevector_dimension:
            raise ValueError(
                f"full state must have length {self.statevector_dimension}"
            )
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-10):
            raise ValueError("full state must be normalized")
        return values.copy()

    def _hadamard(self, state: ComplexState, axis: int) -> ComplexState:
        view = state.reshape(self._shape)
        active = np.moveaxis(view, axis, -1)
        left = active[..., 0].copy()
        right = active[..., 1].copy()
        scale = 1.0 / math.sqrt(2.0)
        active[..., 0] = (left + right) * scale
        active[..., 1] = (left - right) * scale
        return view.reshape(-1)

    def _fourier(self, state: ComplexState, *, inverse: bool) -> ComplexState:
        view = state.reshape(self.phase_bins, -1)
        indices = np.arange(self.phase_bins)
        sign = -1.0 if inverse else 1.0
        matrix = np.exp(
            sign
            * 2j
            * math.pi
            * np.outer(indices, indices)
            / self.phase_bins
        ) / math.sqrt(self.phase_bins)
        return (matrix @ view).reshape(-1)

    def _controlled_reward_reflection(
        self,
        state: ComplexState,
        control_axis: int,
        reward_value: int,
    ) -> ComplexState:
        view = state.copy().reshape(self._shape)
        ordered = np.moveaxis(
            view,
            (control_axis, self.phase_qubits + 1),
            (-2, -1),
        )
        ordered[..., 1, reward_value] *= -1.0
        return view.reshape(-1)

    def _controlled_global_minus(
        self,
        state: ComplexState,
        control_axis: int,
    ) -> ComplexState:
        view = state.copy().reshape(self._shape)
        active = np.moveaxis(view, control_axis, -1)
        active[..., 1] *= -1.0
        return view.reshape(-1)

    def _controlled_grover(
        self,
        state: ComplexState,
        control_axis: int,
        *,
        inverse: bool,
        gates: _Gates,
        tag: str | None,
    ) -> ComplexState:
        index_axis = self.phase_qubits
        reward_axis = self.phase_qubits + 1
        if not inverse:
            state = self._controlled_reward_reflection(state, control_axis, 1)
            gates.add("controlled_good_reflection")
            state = self.oracle.apply_embedded(
                state,
                register_shape=self._shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                inverse=True,
                tag=tag,
            )
            gates.add("controlled_reward_oracle_inverse")
            state = self._controlled_reward_reflection(state, control_axis, 0)
            gates.add("controlled_zero_reflection")
            state = self.oracle.apply_embedded(
                state,
                register_shape=self._shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                tag=tag,
            )
            gates.add("controlled_reward_oracle_forward")
        else:
            state = self.oracle.apply_embedded(
                state,
                register_shape=self._shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                inverse=True,
                tag=tag,
            )
            gates.add("controlled_reward_oracle_inverse")
            state = self._controlled_reward_reflection(state, control_axis, 0)
            gates.add("controlled_zero_reflection")
            state = self.oracle.apply_embedded(
                state,
                register_shape=self._shape,
                index_axis=index_axis,
                reward_axis=reward_axis,
                control_axis=control_axis,
                tag=tag,
            )
            gates.add("controlled_reward_oracle_forward")
            state = self._controlled_reward_reflection(state, control_axis, 1)
            gates.add("controlled_good_reflection")
        state = self._controlled_global_minus(state, control_axis)
        gates.add("controlled_global_phase")
        gates.grover_iterations += 1
        return state

    def _qpe_forward(
        self,
        state: ComplexState,
        gates: _Gates,
        tag: str | None,
    ) -> ComplexState:
        for axis in range(self.phase_qubits):
            state = self._hadamard(state, axis)
            gates.add("phase_hadamard")
        for axis in range(self.phase_qubits):
            power = 1 << (self.phase_qubits - 1 - axis)
            for _ in range(power):
                state = self._controlled_grover(
                    state,
                    axis,
                    inverse=False,
                    gates=gates,
                    tag=tag,
                )
        state = self._fourier(state, inverse=True)
        gates.add(
            "inverse_qft",
            depth=self.phase_qubits * (self.phase_qubits + 1) // 2,
        )
        gates.qpe_calls += 1
        return state

    def _qpe_inverse(
        self,
        state: ComplexState,
        gates: _Gates,
        tag: str | None,
    ) -> ComplexState:
        state = self._fourier(state, inverse=False)
        gates.add(
            "qft",
            depth=self.phase_qubits * (self.phase_qubits + 1) // 2,
        )
        for axis in reversed(range(self.phase_qubits)):
            power = 1 << (self.phase_qubits - 1 - axis)
            for _ in range(power):
                state = self._controlled_grover(
                    state,
                    axis,
                    inverse=True,
                    gates=gates,
                    tag=tag,
                )
        for axis in reversed(range(self.phase_qubits)):
            state = self._hadamard(state, axis)
            gates.add("phase_hadamard")
        gates.qpe_calls += 1
        return state

    def _prepare_and_estimate(
        self,
        state: ComplexState,
        gates: _Gates,
        tag: str | None,
    ) -> ComplexState:
        state = self.oracle.apply_embedded(
            state,
            register_shape=self._shape,
            index_axis=self.phase_qubits,
            reward_axis=self.phase_qubits + 1,
            tag=tag,
        )
        gates.add("reward_oracle_forward")
        return self._qpe_forward(state, gates, tag)

    def _acceptance_probability(self, state: ComplexState) -> float:
        view = state.reshape(self.phase_bins, self.index_dimension, 2)
        probability = float(np.sum(np.abs(view[self._acceptance_mask]) ** 2))
        return min(max(probability, 0.0), 1.0)

    def _predicate_kickback(
        self,
        state: ComplexState,
        gates: _Gates,
    ) -> tuple[ComplexState, float]:
        # Materialize a comparator flag, compute predicate, apply Z, and
        # uncompute.  The public full workspace omits this exactly-clean qubit.
        source = state.reshape(self.phase_bins, self.index_dimension, 2)
        expanded = np.zeros(
            (self.phase_bins, 2, self.index_dimension, 2),
            dtype=np.complex128,
        )
        expanded[:, 0, :, :] = source
        accepted_indices = tuple(
            index
            for index in self.valid_indices
            if index not in self.excluded_indices
        )
        for phase_bin in self._marked_bins:
            for index in accepted_indices:
                zero = expanded[phase_bin, 0, index, :].copy()
                expanded[phase_bin, 0, index, :] = expanded[
                    phase_bin, 1, index, :
                ]
                expanded[phase_bin, 1, index, :] = zero
        gates.add("phase_bin_comparator_compute")
        expanded[:, 1, :, :] *= -1.0
        gates.add("phase_kickback")
        for phase_bin in self._marked_bins:
            for index in accepted_indices:
                zero = expanded[phase_bin, 0, index, :].copy()
                expanded[phase_bin, 0, index, :] = expanded[
                    phase_bin, 1, index, :
                ]
                expanded[phase_bin, 1, index, :] = zero
        gates.add("phase_bin_comparator_uncompute")
        residual = float(np.linalg.norm(expanded[:, 1, :, :]))
        return expanded[:, 0, :, :].reshape(-1), residual

    def _residuals(self, state: ComplexState) -> tuple[float, float]:
        view = state.reshape(self.phase_bins, self.index_dimension, 2)
        phase_zero_probability = float(np.sum(np.abs(view[0, :, :]) ** 2))
        all_zero_probability = float(np.sum(np.abs(view[0, :, 0]) ** 2))
        phase_residual = math.sqrt(max(0.0, 1.0 - phase_zero_probability))
        zero_residual = math.sqrt(max(0.0, 1.0 - all_zero_probability))
        return phase_residual, zero_residual

    def _operation_resources(
        self,
        before: QuerySnapshot,
        gates: _Gates,
        *,
        phase_residual: float,
        zero_residual: float,
        comparator_residual: float,
    ) -> DirectPhaseFlagResources:
        query_counts = QueryLedger.difference(self.oracle.query_snapshot(), before)
        resources = DirectPhaseFlagResources(
            query_counts=_immutable(query_counts),
            gate_counts=_immutable(gates.counts),
            qpe_calls=gates.qpe_calls,
            controlled_grover_iterations=gates.grover_iterations,
            depth=gates.depth,
            qubits=(
                self.phase_qubits
                + self.oracle.index_qubits
                + 1
                + 1  # transient comparator flag
            ),
            workspace_qubits=self.phase_qubits + 1,
            phase_qubits=self.phase_qubits,
            phase_bins=self.phase_bins,
            phase_ancilla_residual=phase_residual,
            zero_workspace_residual=zero_residual,
            comparator_residual=comparator_residual,
        )
        self._query_counts.update(query_counts)
        self._gate_counts.update(gates.counts)
        self._qpe_calls += gates.qpe_calls
        self._grover_iterations += gates.grover_iterations
        self._depth += gates.depth
        self._max_phase_residual = max(self._max_phase_residual, phase_residual)
        self._max_zero_residual = max(self._max_zero_residual, zero_residual)
        self._max_comparator_residual = max(
            self._max_comparator_residual,
            comparator_residual,
        )
        self._last_resources = resources
        return resources

    def compute(
        self,
        full_state: ComplexState,
        *,
        tag: str | None = None,
    ) -> ComplexState:
        """Apply the amplitude-estimation compute unitary ``C_tau``.

        With ``L = 2**phase_qubits``, this call charges exactly ``2L - 1``
        reward-oracle queries: one state preparation and two controlled calls
        for each of the ``L - 1`` Grover iterates.
        """

        state = self._validated_full_state(full_state)
        before = self.oracle.query_snapshot()
        gates = _Gates()
        state = self._prepare_and_estimate(state, gates, tag)
        phase_residual, zero_residual = self._residuals(state)
        self._operation_resources(
            before,
            gates,
            phase_residual=phase_residual,
            zero_residual=zero_residual,
            comparator_residual=0.0,
        )
        return state

    def inverse_compute(
        self,
        full_state: ComplexState,
        *,
        tag: str | None = None,
    ) -> ComplexState:
        """Apply ``C_tau^dagger`` to an arbitrary complete workspace state."""

        state = self._validated_full_state(full_state)
        before = self.oracle.query_snapshot()
        gates = _Gates()
        state = self._qpe_inverse(state, gates, tag)
        state = self.oracle.apply_embedded(
            state,
            register_shape=self._shape,
            index_axis=self.phase_qubits,
            reward_axis=self.phase_qubits + 1,
            inverse=True,
            tag=tag,
        )
        gates.add("reward_oracle_inverse")
        phase_residual, zero_residual = self._residuals(state)
        self._operation_resources(
            before,
            gates,
            phase_residual=phase_residual,
            zero_residual=zero_residual,
            comparator_residual=0.0,
        )
        return state

    def reflect(
        self,
        full_state: ComplexState,
        *,
        tag: str | None = None,
    ) -> DirectPhaseFlagResult:
        """Execute and report the exact full-workspace threshold reflection."""

        state = self._validated_full_state(full_state)
        before = self.oracle.query_snapshot()
        gates = _Gates()
        state = self._prepare_and_estimate(state, gates, tag)
        acceptance = self._acceptance_probability(state)
        state, comparator_residual = self._predicate_kickback(state, gates)
        state = self._qpe_inverse(state, gates, tag)
        state = self.oracle.apply_embedded(
            state,
            register_shape=self._shape,
            index_axis=self.phase_qubits,
            reward_axis=self.phase_qubits + 1,
            inverse=True,
            tag=tag,
        )
        gates.add("reward_oracle_inverse")
        phase_residual, zero_residual = self._residuals(state)
        resources = self._operation_resources(
            before,
            gates,
            phase_residual=phase_residual,
            zero_residual=zero_residual,
            comparator_residual=comparator_residual,
        )
        result = DirectPhaseFlagResult(
            state=state,
            acceptance_probability=acceptance,
            resources=resources,
        )
        self._last_result = result
        return result

    def apply_reflection(
        self,
        full_state: ComplexState,
        *,
        tag: str | None = None,
    ) -> ComplexState:
        """Return only the reflected full state for composition with search."""

        return self.reflect(full_state, tag=tag).state

    def run(
        self,
        index_state: ComplexState,
        *,
        tag: str | None = None,
    ) -> DirectPhaseFlagResult:
        """Convenience wrapper: prepare zero workspace and reflect once."""

        return self.reflect(self.prepare_zero_workspace(index_state), tag=tag)

    def acceptance_probability(
        self,
        index: int,
        *,
        tag: str | None = None,
    ) -> float:
        """Run one charged exact-QPE diagnostic and return its marked mass.

        This is not a free predicate lookup: it makes one forward state
        preparation and one full controlled-QPE call and updates both ledgers.
        It is intended for small-state diagnostics, not for search verification.
        Use :meth:`verify_index` for measured fixed-shot decisions.
        """

        index = self._checked_index(index)
        state = self.initial_state((index,))
        before = self.oracle.query_snapshot()
        gates = _Gates()
        state = self._prepare_and_estimate(state, gates, tag)
        acceptance = self._acceptance_probability(state)
        phase_residual, zero_residual = self._residuals(state)
        self._operation_resources(
            before,
            gates,
            phase_residual=phase_residual,
            zero_residual=zero_residual,
            comparator_residual=0.0,
        )
        return acceptance

    def _checked_index(self, index: object) -> int:
        checked = _integer(index, "index")
        if checked not in self.valid_indices:
            raise IndexError("index is outside the valid arm range")
        return checked

    def verify_index(
        self,
        index: int,
        *,
        shots: int,
        confidence: float = 0.05,
        seed: int | None = None,
        tag: str | None = None,
    ) -> IndexVerificationResult:
        """Measure independently rerun QPE predicates and form a confidence CI.

        Every shot prepares a fresh state and executes ``V`` through the public,
        charged oracle interface.  The exact marked probability is used only
        to sample that shot's phase-predicate measurement; it is never treated
        as a free verification result.
        """

        index = self._checked_index(index)
        shots = _integer(shots, "shots")
        if shots <= 0:
            raise ValueError("shots must be positive")
        confidence = _probability(confidence, "confidence", closed=False)
        if seed is not None:
            seed = _integer(seed, "seed")
        rng = np.random.default_rng(seed)
        before = self.oracle.query_snapshot()
        gates = _Gates()
        successes = 0
        max_phase_residual = 0.0
        max_zero_residual = 0.0
        for _ in range(shots):
            state = self.initial_state((index,))
            state = self._prepare_and_estimate(state, gates, tag)
            probability = self._acceptance_probability(state)
            successes += int(rng.random() < probability)
            phase_residual, zero_residual = self._residuals(state)
            max_phase_residual = max(max_phase_residual, phase_residual)
            max_zero_residual = max(max_zero_residual, zero_residual)

        estimate = successes / shots
        radius = math.sqrt(math.log(2.0 / confidence) / (2.0 * shots))
        interval = (max(0.0, estimate - radius), min(1.0, estimate + radius))
        decision_cutoff = 0.5
        if interval[0] > decision_cutoff:
            status = "accepted"
        elif interval[1] < decision_cutoff:
            status = "rejected"
        else:
            status = "unresolved"
        resources = self._operation_resources(
            before,
            gates,
            phase_residual=max_phase_residual,
            zero_residual=max_zero_residual,
            comparator_residual=0.0,
        )
        return IndexVerificationResult(
            index=index,
            status=status,
            successes=successes,
            shots=shots,
            estimate=estimate,
            confidence=confidence,
            interval=interval,
            decision_cutoff=decision_cutoff,
            resources=resources,
        )

    def verify(
        self,
        index: int,
        shots: int,
        confidence: float = 0.05,
        seed: int | None = None,
        *,
        tag: str | None = None,
    ) -> IndexVerificationResult:
        """Positional compatibility wrapper for :meth:`verify_index`."""

        return self.verify_index(
            index,
            shots=shots,
            confidence=confidence,
            seed=seed,
            tag=tag,
        )

    def sample_accept_index(
        self,
        computed_state: ComplexState,
        *,
        seed: int | None = None,
    ) -> int | None:
        """Jointly measure the phase predicate and index of a computed state.

        ``None`` denotes a measured reject basis state.  This helper performs
        no oracle query because the caller must supply an already-computed
        state; it must not be used as a substitute for :meth:`compute`.
        """

        state = self._validated_full_state(computed_state)
        if seed is not None:
            seed = _integer(seed, "seed")
        probabilities = np.abs(
            state.reshape(self.phase_bins, self.index_dimension, 2)
        ) ** 2
        flat = probabilities.reshape(-1)
        flat = flat / np.sum(flat)
        outcome = int(np.random.default_rng(seed).choice(flat.size, p=flat))
        phase_bin, index, reward = np.unravel_index(
            outcome,
            probabilities.shape,
        )
        if not self._acceptance_mask[phase_bin, index, reward]:
            return None
        return int(index)

    def resources(self) -> DirectPhaseFlagResources:
        """Return resources accumulated through this flag object's public calls."""

        counts = {kind.value: int(self._query_counts[kind.value]) for kind in QueryKind}
        counts["coherent_total"] = sum(
            counts[kind.value]
            for kind in (
                QueryKind.FORWARD,
                QueryKind.INVERSE,
                QueryKind.CONTROLLED_FORWARD,
                QueryKind.CONTROLLED_INVERSE,
            )
        )
        counts["classical_total"] = counts[QueryKind.CLASSICAL_SAMPLE.value]
        counts["total"] = counts["coherent_total"] + counts["classical_total"]
        return DirectPhaseFlagResources(
            query_counts=_immutable(counts),
            gate_counts=_immutable(self._gate_counts),
            qpe_calls=self._qpe_calls,
            controlled_grover_iterations=self._grover_iterations,
            depth=self._depth,
            qubits=(
                self.phase_qubits + self.oracle.index_qubits + 1 + 1
            ),
            workspace_qubits=self.phase_qubits + 1,
            phase_qubits=self.phase_qubits,
            phase_bins=self.phase_bins,
            phase_ancilla_residual=self._max_phase_residual,
            zero_workspace_residual=self._max_zero_residual,
            comparator_residual=self._max_comparator_residual,
        )


# A descriptive alias retained for callers that prefer the phase-estimation name.
DirectPhaseThresholdFlag = DirectAmplitudeThresholdFlag
