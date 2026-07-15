"""Full-workspace BBHT search over a directly computed QPE predicate.

The search in this module never receives a set of marked arms and never reads
hidden arm means.  Every phase reflection, final predicate computation, and
fresh candidate verification is implemented through
``DirectAmplitudeThresholdFlag`` and therefore reaches the charged reward
oracle.  The finite-QPE workspace is retained throughout amplitude
amplification.

This is a small-state executable reference, not an asymptotic speed-up claim.
In particular, exhausting the configured randomized-search budget is *not* a
certificate that no qualifying arm exists.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .coherent import CanonicalRyStatevectorOracle
from .direct_phase import (
    RESOURCE_SEMANTICS,
    DirectAmplitudeThresholdFlag,
    DirectPhaseFlagResources,
    IndexVerificationResult,
)
from .oracles import QueryKind

ComplexState = NDArray[np.complex128]
BACKEND = "numpy_exact_statevector_small_scale"
CLAIM_STATUS = "full_workspace_bbht_with_direct_qpe_no_complexity_theorem"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _real(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, name: str, *, closed: bool) -> float:
    result = _real(value, name)
    valid = 0.0 <= result <= 1.0 if closed else 0.0 < result < 1.0
    if not valid:
        interval = "[0, 1]" if closed else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _immutable(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _validated_state(state: ComplexState, name: str) -> ComplexState:
    values = np.asarray(state, dtype=np.complex128)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty flat statevector")
    if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-10):
        raise ValueError(f"{name} must be normalized")
    return values


def full_workspace_rank_one_diffusion(
    full_state: ComplexState,
    initial_full_state: ComplexState,
) -> ComplexState:
    r"""Reflect the complete state about one complete initial state.

    This implements :math:`2|s\rangle\langle s|-I` on the phase, index, and
    reward registers together.  It is intentionally not an index-only
    diffusion tensored with identity on the QPE workspace.
    """

    state = _validated_state(full_state, "full_state")
    initial = _validated_state(initial_full_state, "initial_full_state")
    if state.shape != initial.shape:
        raise ValueError("full_state and initial_full_state must have equal length")
    reflected = 2.0 * initial * np.vdot(initial, state) - state
    # Do not renormalize: normalization would make this linear reflection into
    # a nonlinear map and could conceal implementation drift.
    if not np.isclose(np.linalg.norm(reflected), 1.0, atol=1e-10):
        raise RuntimeError("rank-one diffusion failed to preserve normalization")
    return np.asarray(reflected, dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class DirectSearchResources:
    """Executed resources for a search prefix or one individual attempt."""

    query_counts: Mapping[str, int]
    gate_counts: Mapping[str, int]
    attempts: int
    amplitude_amplification_iterations: int
    rank_one_diffusions: int
    qpe_calls: int
    controlled_qpe_grover_iterations: int
    verification_shots: int
    depth: int
    qubits: int
    workspace_qubits: int
    # Compatibility fields: these retain their public names while now meaning
    # the retained state and corrected conservative simulator peak.
    statevector_dimension: int
    peak_statevector_dimension: int
    retained_statevector_dimension: int
    comparator_expanded_statevector_dimension: int
    dense_qft_matrix_dimension: int
    estimated_peak_complex_amplitudes: int
    phase_ancilla_residual: float
    zero_workspace_residual: float
    comparator_residual: float
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS
    resource_semantics: str = RESOURCE_SEMANTICS

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("coherent_total", 0))

    @property
    def gates(self) -> int:
        return sum(int(value) for value in self.gate_counts.values())


@dataclass(frozen=True, slots=True)
class DirectSearchAttempt:
    """Reproducible trace for one BBHT draw and its measured outcome."""

    ordinal: int
    output_position: int
    eligible_count: int
    bbht_bound_before: float
    bbht_bound_after: float
    grover_iterations: int
    measurement_seed: int
    verification_seed: int | None
    measured_accept: bool
    candidate: int | None
    verification: IndexVerificationResult | None
    validator_accepted: bool | None
    accepted_output: bool
    outcome: str
    resources: DirectSearchResources


@dataclass(frozen=True, slots=True)
class DirectThresholdSearchResult:
    """Current immutable result of a resumable direct-threshold search."""

    outputs: tuple[int, ...]
    target_count: int
    relation: str
    threshold: float
    excluded_indices: tuple[int, ...]
    complete: bool
    verified: bool
    attempts: int
    attempts_for_current_output: int
    trace: tuple[DirectSearchAttempt, ...]
    resources: DirectSearchResources
    status: str
    failure_reason: str | None
    verification_failure_budget: float
    per_verification_failure_budget: float
    max_verification_calls: int
    absence_certified: bool = False
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS

    @property
    def found_indices(self) -> tuple[int, ...]:
        """Alias exposing measured and freshly verified outputs."""

        return self.outputs

    @property
    def selected_indices(self) -> tuple[int, ...]:
        """Compatibility alias for threshold-selection experiment code."""

        return self.outputs


class _ResourceAccumulator:
    def __init__(
        self,
        retained_statevector_dimension: int,
        comparator_expanded_statevector_dimension: int,
        dense_qft_matrix_dimension: int,
        estimated_peak_complex_amplitudes: int,
    ) -> None:
        self.queries: Counter[str] = Counter()
        self.gates: Counter[str] = Counter()
        self.attempts = 0
        self.amplification_iterations = 0
        self.diffusions = 0
        self.qpe_calls = 0
        self.controlled_qpe_iterations = 0
        self.verification_shots = 0
        self.depth = 0
        self.qubits = 0
        self.workspace_qubits = 0
        self.statevector_dimension = retained_statevector_dimension
        self.peak_statevector_dimension = estimated_peak_complex_amplitudes
        self.retained_statevector_dimension = retained_statevector_dimension
        self.comparator_expanded_statevector_dimension = (
            comparator_expanded_statevector_dimension
        )
        self.dense_qft_matrix_dimension = dense_qft_matrix_dimension
        self.estimated_peak_complex_amplitudes = estimated_peak_complex_amplitudes
        self.phase_residual = 0.0
        self.zero_residual = 0.0
        self.comparator_residual = 0.0

    def merge_flag(self, resources: DirectPhaseFlagResources) -> None:
        for kind in QueryKind:
            self.queries[kind.value] += int(resources.query_counts.get(kind.value, 0))
        self.gates.update(resources.gate_counts)
        self.qpe_calls += resources.qpe_calls
        self.controlled_qpe_iterations += resources.controlled_grover_iterations
        self.depth += resources.depth
        self.qubits = max(self.qubits, resources.qubits)
        self.workspace_qubits = max(self.workspace_qubits, resources.workspace_qubits)
        self.phase_residual = max(
            self.phase_residual,
            resources.phase_ancilla_residual,
        )
        self.zero_residual = max(
            self.zero_residual,
            resources.zero_workspace_residual,
        )
        self.comparator_residual = max(
            self.comparator_residual,
            resources.comparator_residual,
        )

    def add_gate(self, name: str, count: int = 1, *, depth: int | None = None) -> None:
        self.gates[name] += count
        self.depth += count if depth is None else depth

    def merge(self, other: _ResourceAccumulator) -> None:
        self.queries.update(other.queries)
        self.gates.update(other.gates)
        self.attempts += other.attempts
        self.amplification_iterations += other.amplification_iterations
        self.diffusions += other.diffusions
        self.qpe_calls += other.qpe_calls
        self.controlled_qpe_iterations += other.controlled_qpe_iterations
        self.verification_shots += other.verification_shots
        self.depth += other.depth
        self.qubits = max(self.qubits, other.qubits)
        self.workspace_qubits = max(self.workspace_qubits, other.workspace_qubits)
        self.peak_statevector_dimension = max(
            self.peak_statevector_dimension,
            other.peak_statevector_dimension,
        )
        self.retained_statevector_dimension = max(
            self.retained_statevector_dimension,
            other.retained_statevector_dimension,
        )
        self.comparator_expanded_statevector_dimension = max(
            self.comparator_expanded_statevector_dimension,
            other.comparator_expanded_statevector_dimension,
        )
        self.dense_qft_matrix_dimension = max(
            self.dense_qft_matrix_dimension,
            other.dense_qft_matrix_dimension,
        )
        self.estimated_peak_complex_amplitudes = max(
            self.estimated_peak_complex_amplitudes,
            other.estimated_peak_complex_amplitudes,
        )
        self.phase_residual = max(self.phase_residual, other.phase_residual)
        self.zero_residual = max(self.zero_residual, other.zero_residual)
        self.comparator_residual = max(
            self.comparator_residual,
            other.comparator_residual,
        )

    def result(self) -> DirectSearchResources:
        counts = {kind.value: int(self.queries[kind.value]) for kind in QueryKind}
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
        return DirectSearchResources(
            query_counts=_immutable(counts),
            gate_counts=_immutable(self.gates),
            attempts=self.attempts,
            amplitude_amplification_iterations=self.amplification_iterations,
            rank_one_diffusions=self.diffusions,
            qpe_calls=self.qpe_calls,
            controlled_qpe_grover_iterations=self.controlled_qpe_iterations,
            verification_shots=self.verification_shots,
            depth=self.depth,
            qubits=self.qubits,
            workspace_qubits=self.workspace_qubits,
            statevector_dimension=self.statevector_dimension,
            peak_statevector_dimension=self.peak_statevector_dimension,
            retained_statevector_dimension=self.retained_statevector_dimension,
            comparator_expanded_statevector_dimension=(
                self.comparator_expanded_statevector_dimension
            ),
            dense_qft_matrix_dimension=self.dense_qft_matrix_dimension,
            estimated_peak_complex_amplitudes=(
                self.estimated_peak_complex_amplitudes
            ),
            phase_ancilla_residual=self.phase_residual,
            zero_workspace_residual=self.zero_residual,
            comparator_residual=self.comparator_residual,
        )


class FullWorkspaceBBHT:
    """Resumable BBHT search using a direct, unknown-oracle QPE classifier.

    ``candidate_validator`` is an optional read-only check against an external
    certificate (for example, a simultaneous confidence interval).  It is
    called only after the independently rerun QPE verifier accepts.  The
    callback must return an actual ``bool``; any resources it uses belong to
    the caller and are deliberately not hidden in this search ledger.

    ``verification_confidence`` is a whole-search failure budget, not a
    per-candidate budget.  It is divided across the maximum possible number
    of fresh verifier calls, so a union bound covers all emitted outputs.
    """

    def __init__(
        self,
        oracle: CanonicalRyStatevectorOracle,
        threshold: float,
        target_count: int,
        *,
        phase_qubits: int = 5,
        relation: str = "above",
        excluded_indices: Sequence[int] = (),
        verification_shots: int = 128,
        verification_confidence: float = 0.05,
        max_attempts_per_output: int = 24,
        bbht_growth: float = 6.0 / 5.0,
        max_statevector_dimension: int = 1_048_576,
        candidate_validator: Callable[[int], bool] | None = None,
        seed: int | None = None,
    ) -> None:
        if not isinstance(oracle, CanonicalRyStatevectorOracle):
            raise TypeError("oracle must be a CanonicalRyStatevectorOracle")
        self.oracle = oracle
        self.threshold = _probability(threshold, "threshold", closed=True)
        self.target_count = _integer(target_count, "target_count")
        self.phase_qubits = _integer(phase_qubits, "phase_qubits")
        self.verification_shots = _integer(verification_shots, "verification_shots")
        self.max_attempts_per_output = _integer(
            max_attempts_per_output,
            "max_attempts_per_output",
        )
        self.max_statevector_dimension = _integer(
            max_statevector_dimension,
            "max_statevector_dimension",
        )
        self.verification_confidence = _probability(
            verification_confidence,
            "verification_confidence",
            closed=False,
        )
        self.bbht_growth = _real(bbht_growth, "bbht_growth")
        if self.phase_qubits <= 0 or self.phase_qubits > 12:
            raise ValueError("phase_qubits must lie in {1, ..., 12}")
        if self.verification_shots <= 0:
            raise ValueError("verification_shots must be positive")
        if self.max_attempts_per_output <= 0:
            raise ValueError("max_attempts_per_output must be positive")
        if self.max_statevector_dimension <= 0:
            raise ValueError("max_statevector_dimension must be positive")
        if self.bbht_growth <= 1.0:
            raise ValueError("bbht_growth must exceed one")
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
        capacity = oracle.n_arms - len(excluded)
        if not 0 <= self.target_count <= capacity:
            raise ValueError("target_count exceeds the non-excluded arm capacity")
        if candidate_validator is not None and not callable(candidate_validator):
            raise TypeError("candidate_validator must be callable or None")
        self.candidate_validator = candidate_validator
        if seed is not None:
            seed = _integer(seed, "seed")
        self._rng = np.random.default_rng(seed)

        phase_bins = 1 << self.phase_qubits
        self.retained_statevector_dimension = (
            phase_bins * oracle.index_dimension * 2
        )
        self.comparator_expanded_statevector_dimension = (
            2 * self.retained_statevector_dimension
        )
        self.dense_qft_matrix_dimension = phase_bins * phase_bins
        self.estimated_peak_complex_amplitudes = max(
            3 * self.retained_statevector_dimension,
            self.dense_qft_matrix_dimension
            + 2 * self.retained_statevector_dimension,
        )
        # Compatibility aliases.  The old peak calculation was only ``2S``;
        # it now exposes the corrected conservative simultaneous allocation.
        self.statevector_dimension = self.retained_statevector_dimension
        self.peak_statevector_dimension = self.estimated_peak_complex_amplitudes
        self._blocked_status = (
            "statevector_budget_exceeded"
            if self.peak_statevector_dimension > self.max_statevector_dimension
            else None
        )
        self.max_verification_calls = max(
            1,
            self.target_count * self.max_attempts_per_output,
        )
        # ``verification_confidence`` is a whole-search failure budget.  A
        # union bound assigns an equal share to every possible fresh verifier.
        self.per_verification_confidence = (
            self.verification_confidence / self.max_verification_calls
        )
        self._outputs: list[int] = []
        self._trace: list[DirectSearchAttempt] = []
        self._attempts_for_current_output = 0
        self._bbht_bound = 1.0
        self._resources = _ResourceAccumulator(
            self.retained_statevector_dimension,
            self.comparator_expanded_statevector_dimension,
            self.dense_qft_matrix_dimension,
            self.estimated_peak_complex_amplitudes,
        )

    def _eligible_indices(self) -> tuple[int, ...]:
        excluded = set(self.excluded_indices) | set(self._outputs)
        return tuple(index for index in range(self.oracle.n_arms) if index not in excluded)

    def _can_step(self) -> bool:
        return (
            self._blocked_status is None
            and len(self._outputs) < self.target_count
            and self._attempts_for_current_output < self.max_attempts_per_output
        )

    def _draw_seed(self) -> int:
        return int(self._rng.integers(0, np.iinfo(np.int64).max, endpoint=False))

    def _attempt_resources(
        self,
        flag: DirectAmplitudeThresholdFlag,
        *,
        iterations: int,
        verification_shots: int,
    ) -> _ResourceAccumulator:
        resources = _ResourceAccumulator(
            self.retained_statevector_dimension,
            self.comparator_expanded_statevector_dimension,
            self.dense_qft_matrix_dimension,
            self.estimated_peak_complex_amplitudes,
        )
        resources.attempts = 1
        resources.amplification_iterations = iterations
        resources.diffusions = iterations
        resources.verification_shots = verification_shots
        resources.add_gate("full_workspace_initial_state_preparation")
        resources.add_gate("full_workspace_rank_one_diffusion", iterations)
        resources.add_gate("joint_accept_index_measurement")
        if verification_shots:
            resources.add_gate(
                "fresh_verification_state_preparation",
                verification_shots,
            )
            resources.add_gate("fresh_verification_measurement", verification_shots)
        resources.merge_flag(flag.resources())
        return resources

    def step(self) -> DirectThresholdSearchResult:
        """Execute at most one randomized BBHT attempt."""

        if not self._can_step():
            return self.result()

        eligible = self._eligible_indices()
        if not eligible:
            # Construction validation makes this unreachable unless internal
            # invariants are violated; retain a non-absence terminal status.
            self._blocked_status = "eligible_domain_exhausted_without_absence_proof"
            return self.result()

        flag = DirectAmplitudeThresholdFlag(
            self.oracle,
            self.threshold,
            phase_qubits=self.phase_qubits,
            relation=self.relation,
            excluded_indices=tuple(sorted(set(self.excluded_indices) | set(self._outputs))),
        )
        initial = flag.initial_state(eligible)
        state = initial.copy()
        bound_before = self._bbht_bound
        upper = max(1, int(math.ceil(bound_before)))
        iterations = int(self._rng.integers(0, upper))
        for _ in range(iterations):
            state = flag.apply_reflection(state, tag="direct_bbht_reflection")
            state = full_workspace_rank_one_diffusion(state, initial)

        computed = flag.compute(state, tag="direct_bbht_decode")
        measurement_seed = self._draw_seed()
        candidate = flag.sample_accept_index(computed, seed=measurement_seed)
        verification: IndexVerificationResult | None = None
        verification_seed: int | None = None
        validator_accepted: bool | None = None
        accepted = False
        verification_shots = 0
        if candidate is None:
            outcome = "joint_predicate_measurement_rejected"
        elif candidate not in eligible:
            # The direct flag mask should make this impossible, but recording
            # it is safer than silently accepting a padded or excluded index.
            outcome = "invalid_or_excluded_candidate_rejected"
        else:
            verification_seed = self._draw_seed()
            verification = flag.verify_index(
                candidate,
                shots=self.verification_shots,
                confidence=self.per_verification_confidence,
                seed=verification_seed,
                tag="direct_bbht_fresh_verification",
            )
            verification_shots = verification.shots
            if not verification.accepted:
                outcome = f"fresh_verifier_{verification.status}"
            elif self.candidate_validator is None:
                validator_accepted = None
                accepted = True
                outcome = "accepted_fresh_qpe_verification"
            else:
                raw_validation = self.candidate_validator(candidate)
                if not isinstance(raw_validation, bool):
                    raise TypeError("candidate_validator must return bool")
                validator_accepted = raw_validation
                accepted = raw_validation
                outcome = (
                    "accepted_external_certificate_validation"
                    if raw_validation
                    else "classifier_false_positive_rejected"
                )

        self._attempts_for_current_output += 1
        if accepted:
            if candidate is None:
                raise RuntimeError("an accepted attempt has no candidate")
            self._outputs.append(candidate)
            self._attempts_for_current_output = 0
            self._bbht_bound = 1.0
        else:
            self._bbht_bound = min(
                self.bbht_growth * bound_before,
                math.sqrt(len(eligible)),
            )

        attempt_resources = self._attempt_resources(
            flag,
            iterations=iterations,
            verification_shots=verification_shots,
        )
        self._resources.merge(attempt_resources)
        self._trace.append(
            DirectSearchAttempt(
                ordinal=len(self._trace) + 1,
                output_position=len(self._outputs) - int(accepted),
                eligible_count=len(eligible),
                bbht_bound_before=bound_before,
                bbht_bound_after=self._bbht_bound,
                grover_iterations=iterations,
                measurement_seed=measurement_seed,
                verification_seed=verification_seed,
                measured_accept=candidate is not None,
                candidate=candidate,
                verification=verification,
                validator_accepted=validator_accepted,
                accepted_output=accepted,
                outcome=outcome,
                resources=attempt_resources.result(),
            )
        )
        return self.result()

    def run(self, max_steps: int | None = None) -> DirectThresholdSearchResult:
        """Run an additional attempt budget; call ``resume`` to continue."""

        if max_steps is not None:
            max_steps = _integer(max_steps, "max_steps")
            if max_steps < 0:
                raise ValueError("max_steps cannot be negative")
        budget = self.max_attempts_per_output if max_steps is None else max_steps
        for _ in range(budget):
            if not self._can_step():
                break
            self.step()
        return self.result()

    resume = run

    def result(self) -> DirectThresholdSearchResult:
        complete = len(self._outputs) == self.target_count
        verified = all(
            attempt.verification is not None
            and attempt.verification.accepted
            and attempt.accepted_output
            for attempt in self._trace
            if attempt.accepted_output
        )
        if complete:
            status = "complete_fixed_confidence_qpe_predicate"
            failure_reason = None
        elif self._blocked_status is not None:
            status = self._blocked_status
            failure_reason = self._blocked_status
        elif self._attempts_for_current_output >= self.max_attempts_per_output:
            status = "search_attempt_budget_exhausted"
            failure_reason = "randomized_budget_exhaustion_does_not_certify_absence"
        else:
            status = "paused_resumable"
            failure_reason = None
        return DirectThresholdSearchResult(
            outputs=tuple(self._outputs),
            target_count=self.target_count,
            relation=self.relation,
            threshold=self.threshold,
            excluded_indices=self.excluded_indices,
            complete=complete,
            verified=verified,
            attempts=len(self._trace),
            attempts_for_current_output=self._attempts_for_current_output,
            trace=tuple(self._trace),
            resources=self._resources.result(),
            status=status,
            failure_reason=failure_reason,
            verification_failure_budget=self.verification_confidence,
            per_verification_failure_budget=self.per_verification_confidence,
            max_verification_calls=self.max_verification_calls,
            absence_certified=False,
        )
