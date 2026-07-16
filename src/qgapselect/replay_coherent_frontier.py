"""Replay-preserving coherent frontier unitary for tiny exact-state checks.

The executable in this module isolates one circuit invariant needed by the
unknown-boundary activity-history programme:

``compute scheduled prefix -> copy durable output -> uncompute prefix``.

The retained registers are explicitly ordered as ``(index, history, stop,
output, work)``.  ``history``, ``stop``, and ``work`` are transient.  The
``output`` qubit is durable and remains after the scheduled prefix is replayed
backwards.  Every public schedule row has a deterministic transcript and both
the per-index transcripts and the three-stage execution trace can be replayed.

This is deliberately an executable code-sanity artifact, not a new quantum
upper bound.  The frontier schedule is a supplied public fixture rather than a
coherently discovered unknown boundary.  Its lookup is charged as an
unoptimised linear compiled SELECT scan; QRAM/QROM access is neither assumed
nor hidden.  The statevector permutation checks reversible semantics only and
is not hardware or quantum-advantage evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

ComplexState = NDArray[np.complex128]

BACKEND = "numpy_exact_statevector_replay_coherent_frontier"
CLAIM_SCOPE = (
    "executable_code_sanity_only_no_new_upper_bound_no_lower_bound_no_hardware"
)
SCHEDULE_LOADING_MODEL = "charged_unoptimised_linear_compiled_select_no_qram"
QUERY_COUNT_SEMANTICS = "logical_charges_not_physical_oracle_equivalence"
GATE_COUNT_SEMANTICS = "unoptimised_compiled_select_logical_upper_bound"
DEPTH_SEMANTICS = "fully_serial_sum_of_charged_logical_gates_upper_bound"


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


def _indices(
    values: Sequence[int],
    *,
    upper: int,
    name: str,
) -> tuple[int, ...]:
    try:
        result = tuple(_integer(value, f"{name} index") for value in values)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of integers") from error
    if len(set(result)) != len(result):
        raise ValueError(f"{name} indices must be unique")
    if any(index >= upper for index in result):
        raise IndexError(f"{name} contains an index outside [0, {upper})")
    return tuple(sorted(result))


def _frozen_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplayFrontierSchedule:
    """A public nested activity schedule used by the code-sanity unitary.

    ``active_indices_by_level`` contains ``L + 1`` frontier rows.  Row ``l``
    is active immediately before decision level ``l`` and the final row is the
    unresolved frontier after all ``L`` decisions.  ``output_births_by_level``
    contains ``L`` disjoint output-birth rows.  An arm that disappears without
    being born as output is a rejected branch.
    """

    n_arms: int
    active_indices_by_level: tuple[tuple[int, ...], ...]
    output_births_by_level: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        n_arms = _integer(self.n_arms, "n_arms", minimum=2)
        object.__setattr__(self, "n_arms", n_arms)
        try:
            active_raw = tuple(self.active_indices_by_level)
            output_raw = tuple(self.output_births_by_level)
        except TypeError as error:
            raise TypeError("frontier rows must be sequences of index sequences") from error
        if not output_raw:
            raise ValueError("at least one decision level is required")
        if len(active_raw) != len(output_raw) + 1:
            raise ValueError("active rows must contain exactly one final frontier row")
        active = tuple(
            _indices(row, upper=n_arms, name=f"active[{level}]")
            for level, row in enumerate(active_raw)
        )
        outputs = tuple(
            _indices(row, upper=n_arms, name=f"output[{level}]")
            for level, row in enumerate(output_raw)
        )
        if active[0] != tuple(range(n_arms)):
            raise ValueError("active[0] must contain every declared arm")
        born: set[int] = set()
        for level, output in enumerate(outputs):
            current = set(active[level])
            following = set(active[level + 1])
            output_set = set(output)
            if not output_set.issubset(current):
                raise ValueError(f"output[{level}] must be a subset of active[{level}]")
            if born.intersection(output_set):
                raise ValueError("an output arm cannot be born at more than one level")
            if not following.issubset(current - output_set):
                raise ValueError(
                    f"active[{level + 1}] must be nested and exclude output births"
                )
            born.update(output_set)
        object.__setattr__(self, "active_indices_by_level", active)
        object.__setattr__(self, "output_births_by_level", outputs)

    @property
    def level_count(self) -> int:
        return len(self.output_births_by_level)

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "n_arms": self.n_arms,
                "active": self.active_indices_by_level,
                "outputs": self.output_births_by_level,
            }
        )


@dataclass(frozen=True, slots=True)
class FrontierTranscript:
    """Replayable per-index record for one scheduled prefix."""

    schedule_fingerprint: str
    index: int
    prefix_levels: int
    active_bits: tuple[bool, ...]
    events: tuple[str, ...]
    history_mask: int
    stop_code: int
    work_code: int
    output_bit: int
    first_stop_level: int | None
    selected_level: int | None
    digest: str


@dataclass(frozen=True, slots=True)
class FrontierExecutionEvent:
    """One deterministic stage of compute-copy-uncompute execution."""

    ordinal: int
    operation: str
    inverse: bool
    prefix_levels: int
    schedule_fingerprint: str
    relation_digest: str
    charged_schedule_cells: int


@dataclass(frozen=True, slots=True)
class FrontierInvariantLedger:
    """Replay-checked schedule and first-stop invariants."""

    schedule_fingerprint: str
    all_transcripts_replayed: bool
    history_matches_active_prefix: bool
    stop_is_first_terminal_event: bool
    stopped_branches_remain_inactive: bool
    output_matches_select_event: bool
    unresolved_matches_final_frontier: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.all_transcripts_replayed,
                self.history_matches_active_prefix,
                self.stop_is_first_terminal_event,
                self.stopped_branches_remain_inactive,
                self.output_matches_select_event,
                self.unresolved_matches_final_frontier,
            )
        )


@dataclass(frozen=True, slots=True)
class FrontierCleanupLedger:
    """Exact-state cleanup and replay identities for one execution."""

    expected_durable_output_residual_l2: float
    transient_nonzero_probability: float
    norm_error: float
    all_transcripts_replayed: bool
    execution_trace_replayed: bool
    tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.expected_durable_output_residual_l2 <= self.tolerance
            and self.transient_nonzero_probability <= self.tolerance
            and self.norm_error <= self.tolerance
            and self.all_transcripts_replayed
            and self.execution_trace_replayed
        )


@dataclass(frozen=True, slots=True)
class FrontierResourceLedger:
    """Charged logical resources with an explicit no-free-QRAM model."""

    query_counts: Mapping[str, int]
    gate_counts: Mapping[str, int]
    depth: int
    qubits: int
    register_dimensions: Mapping[str, int]
    statevector_dimension: int
    schedule_storage_bits: int
    schedule_loading_model: str
    query_count_semantics: str
    gate_count_semantics: str
    depth_semantics: str
    qram_assumed: bool
    cleanup: FrontierCleanupLedger
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE


@dataclass(frozen=True, slots=True)
class ReplayCoherentFrontierResult:
    """Output of the tiny exact-state replay-preserving frontier unitary."""

    state: ComplexState
    transcripts: tuple[FrontierTranscript, ...]
    execution_trace: tuple[FrontierExecutionEvent, ...]
    invariants: FrontierInvariantLedger
    resources: FrontierResourceLedger
    durable_output_probability: float
    blockers: tuple[str, ...]
    status: str
    durable_output_copy_executed: bool = True
    quantum_advantage_claimable: bool = False
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE


class ReplayPreservingCoherentFrontier:
    """Exact permutation for a supplied scheduled frontier prefix.

    Work-bit encoding is ``(active_after_prefix, stopped, selected)`` from
    least to most significant bit.  The selected bit controls the durable
    output copy.  All three work bits, the history mask, and the stop code are
    then erased by replaying the same compiled relation.
    """

    _INDEX = 0
    _HISTORY = 1
    _STOP = 2
    _OUTPUT = 3
    _WORK = 4

    def __init__(
        self,
        schedule: ReplayFrontierSchedule,
        *,
        prefix_levels: int | None = None,
        cleanup_tolerance: float = 1e-12,
        max_statevector_dimension: int = 8_388_608,
    ) -> None:
        if not isinstance(schedule, ReplayFrontierSchedule):
            raise TypeError("schedule must be a ReplayFrontierSchedule")
        prefix = (
            schedule.level_count
            if prefix_levels is None
            else _integer(prefix_levels, "prefix_levels", minimum=1)
        )
        if prefix > schedule.level_count:
            raise ValueError("prefix_levels exceeds the schedule")
        if isinstance(cleanup_tolerance, bool):
            raise TypeError("cleanup_tolerance must be a positive finite real")
        tolerance = float(cleanup_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("cleanup_tolerance must be a positive finite real")
        max_dimension = _integer(
            max_statevector_dimension,
            "max_statevector_dimension",
            minimum=1,
        )
        self.schedule = schedule
        self.prefix_levels = prefix
        self.cleanup_tolerance = tolerance
        self.index_dimension = _next_power_of_two(schedule.n_arms)
        self.history_dimension = 1 << prefix
        self.stop_dimension = _next_power_of_two(prefix + 1)
        self.output_dimension = 2
        self.work_dimension = 8
        self.shape = (
            self.index_dimension,
            self.history_dimension,
            self.stop_dimension,
            self.output_dimension,
            self.work_dimension,
        )
        self.statevector_dimension = math.prod(self.shape)
        if self.statevector_dimension > max_dimension:
            raise ValueError("explicit register statevector exceeds max_statevector_dimension")
        self._transcripts = tuple(
            self._build_transcript(index) for index in range(schedule.n_arms)
        )
        self._relation_digest = _digest(
            [transcript.digest for transcript in self._transcripts]
        )

    @property
    def register_dimensions(self) -> Mapping[str, int]:
        return _frozen_counts(
            {
                "index": self.index_dimension,
                "history": self.history_dimension,
                "stop": self.stop_dimension,
                "output": self.output_dimension,
                "work": self.work_dimension,
            }
        )

    @property
    def qubits(self) -> int:
        return sum(int(math.log2(dimension)) for dimension in self.shape)

    def _transcript_payload(
        self,
        *,
        index: int,
        active_bits: tuple[bool, ...],
        events: tuple[str, ...],
        history_mask: int,
        stop_code: int,
        work_code: int,
        output_bit: int,
        first_stop_level: int | None,
        selected_level: int | None,
    ) -> dict[str, object]:
        return {
            "schedule_fingerprint": self.schedule.fingerprint,
            "index": index,
            "prefix_levels": self.prefix_levels,
            "active_bits": active_bits,
            "events": events,
            "history_mask": history_mask,
            "stop_code": stop_code,
            "work_code": work_code,
            "output_bit": output_bit,
            "first_stop_level": first_stop_level,
            "selected_level": selected_level,
        }

    def _build_transcript(self, index: int) -> FrontierTranscript:
        active_sets = tuple(
            frozenset(row) for row in self.schedule.active_indices_by_level
        )
        output_sets = tuple(
            frozenset(row) for row in self.schedule.output_births_by_level
        )
        active_bits: list[bool] = []
        events: list[str] = []
        first_stop: int | None = None
        selected: int | None = None
        for level in range(self.prefix_levels):
            active = index in active_sets[level]
            active_bits.append(active)
            if not active:
                events.append("inactive")
            elif index in output_sets[level]:
                events.append("select")
                first_stop = level if first_stop is None else first_stop
                selected = level if selected is None else selected
            elif index not in active_sets[level + 1]:
                events.append("reject")
                first_stop = level if first_stop is None else first_stop
            else:
                events.append("continue")
        history_mask = sum(
            int(active) << level for level, active in enumerate(active_bits)
        )
        stop_code = 0 if first_stop is None else first_stop + 1
        output_bit = int(selected is not None)
        active_after_prefix = int(index in active_sets[self.prefix_levels])
        work_code = active_after_prefix | (int(first_stop is not None) << 1) | (
            output_bit << 2
        )
        payload = self._transcript_payload(
            index=index,
            active_bits=tuple(active_bits),
            events=tuple(events),
            history_mask=history_mask,
            stop_code=stop_code,
            work_code=work_code,
            output_bit=output_bit,
            first_stop_level=first_stop,
            selected_level=selected,
        )
        return FrontierTranscript(**payload, digest=_digest(payload))

    def transcript(self, index: int) -> FrontierTranscript:
        index = _integer(index, "index")
        if index >= self.schedule.n_arms:
            raise IndexError("index is outside the declared arms")
        return self._transcripts[index]

    def transcripts(self) -> tuple[FrontierTranscript, ...]:
        return self._transcripts

    def verify_transcript(self, transcript: FrontierTranscript) -> bool:
        if not isinstance(transcript, FrontierTranscript):
            return False
        if not 0 <= transcript.index < self.schedule.n_arms:
            return False
        return transcript == self._build_transcript(transcript.index)

    def execution_trace(self) -> tuple[FrontierExecutionEvent, ...]:
        cells = 3 * self.schedule.n_arms * self.prefix_levels
        common = {
            "prefix_levels": self.prefix_levels,
            "schedule_fingerprint": self.schedule.fingerprint,
            "relation_digest": self._relation_digest,
        }
        return (
            FrontierExecutionEvent(
                ordinal=0,
                operation="compute_scheduled_prefix",
                inverse=False,
                charged_schedule_cells=cells,
                **common,
            ),
            FrontierExecutionEvent(
                ordinal=1,
                operation="copy_durable_output_from_work_selected_bit",
                inverse=False,
                charged_schedule_cells=0,
                **common,
            ),
            FrontierExecutionEvent(
                ordinal=2,
                operation="uncompute_scheduled_prefix_by_replay",
                inverse=True,
                charged_schedule_cells=cells,
                **common,
            ),
        )

    def verify_execution_trace(
        self,
        trace: Sequence[FrontierExecutionEvent],
    ) -> bool:
        try:
            candidate = tuple(trace)
        except TypeError:
            return False
        return candidate == self.execution_trace()

    def invariant_ledger(self) -> FrontierInvariantLedger:
        """Replay all branch records and audit the scheduled-prefix invariants."""

        history_ok = True
        first_stop_ok = True
        post_stop_ok = True
        output_ok = True
        unresolved_ok = True
        for transcript in self._transcripts:
            expected_history = sum(
                int(active) << level
                for level, active in enumerate(transcript.active_bits)
            )
            history_ok &= transcript.history_mask == expected_history
            terminal = tuple(
                level
                for level, event in enumerate(transcript.events)
                if event in {"select", "reject"}
            )
            expected_stop = terminal[0] if terminal else None
            first_stop_ok &= transcript.first_stop_level == expected_stop
            first_stop_ok &= transcript.stop_code == (
                0 if expected_stop is None else expected_stop + 1
            )
            if expected_stop is not None:
                post_stop_ok &= all(
                    event == "inactive"
                    for event in transcript.events[expected_stop + 1 :]
                )
            selected = tuple(
                level
                for level, event in enumerate(transcript.events)
                if event == "select"
            )
            output_ok &= transcript.output_bit == int(bool(selected))
            output_ok &= transcript.selected_level == (selected[0] if selected else None)
            active_after = transcript.index in self.schedule.active_indices_by_level[
                self.prefix_levels
            ]
            unresolved_ok &= bool(transcript.work_code & 0b001) == active_after
            unresolved_ok &= active_after == (expected_stop is None)
        return FrontierInvariantLedger(
            schedule_fingerprint=self.schedule.fingerprint,
            all_transcripts_replayed=all(
                self.verify_transcript(transcript) for transcript in self._transcripts
            ),
            history_matches_active_prefix=history_ok,
            stop_is_first_terminal_event=first_stop_ok,
            stopped_branches_remain_inactive=post_stop_ok,
            output_matches_select_event=output_ok,
            unresolved_matches_final_frontier=unresolved_ok,
        )

    def uniform_index_state(
        self,
        *,
        indices: Sequence[int] | None = None,
        output_bit: int = 0,
    ) -> ComplexState:
        output = _integer(output_bit, "output_bit")
        if output >= 2:
            raise ValueError("output_bit must be zero or one")
        selected = (
            tuple(range(self.schedule.n_arms)) if indices is None else tuple(indices)
        )
        if not selected:
            raise ValueError("indices cannot be empty")
        selected = _indices(selected, upper=self.schedule.n_arms, name="indices")
        view = np.zeros(self.shape, dtype=np.complex128)
        view[selected, 0, 0, output, 0] = 1.0 / math.sqrt(len(selected))
        return view.reshape(-1)

    def _validated_state(self, state: ComplexState) -> ComplexState:
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.size != self.statevector_dimension:
            raise ValueError(
                f"expected statevector length {self.statevector_dimension}, got {values.shape}"
            )
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-10):
            raise ValueError("statevector must be normalized")
        return values.copy()

    def _relation_values(self, index: int) -> tuple[int, int, int]:
        if index >= self.schedule.n_arms:
            return (0, 0, 0)
        transcript = self._transcripts[index]
        return (
            transcript.history_mask,
            transcript.stop_code,
            transcript.work_code,
        )

    def apply_compute(self, state: ComplexState) -> ComplexState:
        """XOR the scheduled prefix relation into transient registers.

        This permutation acts on every basis state, including non-zero work
        registers.  Applying it twice is therefore the exact inverse.
        """

        source = self._validated_state(state).reshape(self.shape)
        target = np.zeros_like(source)
        for index in range(self.index_dimension):
            history_mask, stop_code, work_code = self._relation_values(index)
            for history in range(self.history_dimension):
                for stop in range(self.stop_dimension):
                    for output in range(self.output_dimension):
                        for work in range(self.work_dimension):
                            target[
                                index,
                                history ^ history_mask,
                                stop ^ stop_code,
                                output,
                                work ^ work_code,
                            ] = source[index, history, stop, output, work]
        return target.reshape(-1)

    def apply_durable_output_copy(self, state: ComplexState) -> ComplexState:
        """Copy work's selected bit into the durable output qubit."""

        source = self._validated_state(state).reshape(self.shape)
        target = np.zeros_like(source)
        for output in range(2):
            for work in range(self.work_dimension):
                selected = (work >> 2) & 1
                target[:, :, :, output ^ selected, work] = source[
                    :, :, :, output, work
                ]
        return target.reshape(-1)

    def _expected_durable_output(self, state: ComplexState) -> ComplexState:
        source = self._validated_state(state).reshape(self.shape)
        target = np.zeros_like(source)
        for index in range(self.index_dimension):
            selected = (
                0
                if index >= self.schedule.n_arms
                else self._transcripts[index].output_bit
            )
            for output in range(2):
                target[index, :, :, output ^ selected, :] = source[
                    index, :, :, output, :
                ]
        return target.reshape(-1)

    def _transient_nonzero_probability(self, state: ComplexState) -> float:
        view = state.reshape(self.shape)
        clean_probability = float(np.sum(np.abs(view[:, 0, 0, :, 0]) ** 2))
        return max(0.0, 1.0 - clean_probability)

    def _resource_ledger(self, cleanup: FrontierCleanupLedger) -> FrontierResourceLedger:
        n = self.schedule.n_arms
        levels = self.prefix_levels
        schedule_cells = 2 * 3 * n * levels
        history_writes = 2 * sum(
            transcript.history_mask.bit_count() for transcript in self._transcripts
        )
        stop_writes = 2 * sum(
            transcript.stop_code.bit_count() for transcript in self._transcripts
        )
        work_writes = 2 * sum(
            transcript.work_code.bit_count() for transcript in self._transcripts
        )
        equality_primitives = schedule_cells * (2 * int(math.log2(self.index_dimension)) + 1)
        gate_counts = _frozen_counts(
            {
                "compiled_index_equality_primitives": equality_primitives,
                "charged_schedule_cell_selects": schedule_cells,
                "history_controlled_xor": history_writes,
                "stop_controlled_xor": stop_writes,
                "work_controlled_xor": work_writes,
                "durable_output_cnot": 1,
            }
        )
        query_counts = _frozen_counts(
            {
                "reward_oracle_queries": 0,
                "qram_queries": 0,
                "compiled_schedule_cell_accesses": schedule_cells,
                "frontier_relation_calls": 2,
            }
        )
        # This is a conservative serial compilation ledger, not an optimised
        # circuit depth or an asymptotic theorem.
        depth = int(sum(gate_counts.values()))
        storage_bits = n * (2 * self.schedule.level_count + 1)
        return FrontierResourceLedger(
            query_counts=query_counts,
            gate_counts=gate_counts,
            depth=depth,
            qubits=self.qubits,
            register_dimensions=self.register_dimensions,
            statevector_dimension=self.statevector_dimension,
            schedule_storage_bits=storage_bits,
            schedule_loading_model=SCHEDULE_LOADING_MODEL,
            query_count_semantics=QUERY_COUNT_SEMANTICS,
            gate_count_semantics=GATE_COUNT_SEMANTICS,
            depth_semantics=DEPTH_SEMANTICS,
            qram_assumed=False,
            cleanup=cleanup,
        )

    def apply(self, state: ComplexState) -> ReplayCoherentFrontierResult:
        """Execute compute-copy-uncompute on a clean transient workspace."""

        initial = self._validated_state(state)
        if self._transient_nonzero_probability(initial) > self.cleanup_tolerance:
            raise ValueError("history, stop, and work registers must start clean")
        computed = self.apply_compute(initial)
        copied = self.apply_durable_output_copy(computed)
        restored = self.apply_compute(copied)
        expected = self._expected_durable_output(initial)
        trace = self.execution_trace()
        invariants = self.invariant_ledger()
        cleanup = FrontierCleanupLedger(
            expected_durable_output_residual_l2=float(np.linalg.norm(restored - expected)),
            transient_nonzero_probability=self._transient_nonzero_probability(restored),
            norm_error=abs(float(np.linalg.norm(restored)) - 1.0),
            all_transcripts_replayed=invariants.all_transcripts_replayed,
            execution_trace_replayed=self.verify_execution_trace(trace),
            tolerance=self.cleanup_tolerance,
        )
        resources = self._resource_ledger(cleanup)
        output_probability = float(
            np.sum(np.abs(np.take(restored.reshape(self.shape), 1, axis=self._OUTPUT)) ** 2)
        )
        blockers = (
            "frontier_schedule_is_a_supplied_public_fixture_not_coherently_discovered",
            "compiled_linear_schedule_scan_is_charged_and_not_a_variable_time_upper_bound",
            "branch_output_qubit_is_not_complete_direct_multi_output_aggregation",
            "no_same_interface_composition_separation",
            "no_matching_lower_bound",
        )
        status = (
            "code_sanity_passed_theorem_blocked"
            if cleanup.passed and invariants.passed
            else "code_sanity_cleanup_or_replay_failed"
        )
        return ReplayCoherentFrontierResult(
            state=restored,
            transcripts=self._transcripts,
            execution_trace=trace,
            invariants=invariants,
            resources=resources,
            durable_output_probability=output_probability,
            blockers=blockers,
            status=status,
        )
