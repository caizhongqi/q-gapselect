"""Toy reversible activity-history transducer.

This module is the first executable artifact for the P0-U08 obligation in the
research gap audit.  It intentionally does *not* claim the real unknown-boundary
algorithm: the active/output predicates are supplied by a benchmark harness.

What it does provide is the relation-level skeleton the proof must eventually
realize from charged canonical-oracle access:

* a coherent level/index predicate interface;
* reversible compute/uncompute by XORing activity and output flags;
* a phase oracle implemented as compute--phase--uncompute; and
* an explicit ledger for predicate and phase operations.

If a future construction cannot replace the supplied predicates with a charged
unknown-boundary transducer, the candidate remains a proof obligation.
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

from .unknown_boundary_history import UnknownBoundaryHistoryRecord

ComplexState = NDArray[np.complex128]
BACKEND = "numpy_exact_statevector_toy_relation"
CLAIM_STATUS = "toy_activity_history_relation_no_upper_bound_theorem"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _as_unique_indices(values: Sequence[int], *, upper: int, name: str) -> tuple[int, ...]:
    try:
        indices = tuple(_integer(value, f"{name} index") for value in values)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of integers") from error
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} indices must be unique")
    if any(not 0 <= index < upper for index in indices):
        raise IndexError(f"{name} contains an index outside [0, {upper})")
    return indices


@dataclass(frozen=True, slots=True)
class ActivityHistoryResources:
    """Logical resources for the toy relation transducer."""

    query_counts: Mapping[str, int]
    level_count: int
    n_arms: int
    level_dimension: int
    index_dimension: int
    statevector_dimension: int
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS

    @property
    def total_queries(self) -> int:
        return int(sum(self.query_counts.values()))


@dataclass(frozen=True, slots=True)
class ActivityHistoryResult:
    """State and resources after a toy transducer operation."""

    state: ComplexState
    resources: ActivityHistoryResources
    active_probability: float
    output_probability: float


class ToyActivityHistoryTransducer:
    """Small exact-state reversible relation oracle for activity histories.

    Registers are ordered as ``(level, index, active_flag, output_flag)``.
    ``apply_compute`` XORs the supplied level/index predicates into the two
    flags and is therefore self-inverse.  ``apply_phase`` uses the standard
    compute--phase--uncompute pattern and leaves both flags restored.
    """

    def __init__(
        self,
        n_arms: int,
        active_indices_by_level: Sequence[Sequence[int]],
        output_indices_by_level: Sequence[Sequence[int]],
    ) -> None:
        n_arms = _integer(n_arms, "n_arms")
        if n_arms < 2:
            raise ValueError("n_arms must be at least two")
        try:
            active_rows = tuple(tuple(row) for row in active_indices_by_level)
            output_rows = tuple(tuple(row) for row in output_indices_by_level)
        except TypeError as error:
            raise TypeError("predicate rows must be sequences") from error
        if not active_rows:
            raise ValueError("at least one activity level is required")
        if len(active_rows) != len(output_rows):
            raise ValueError("active and output rows must have the same length")
        normalized_active = tuple(
            _as_unique_indices(row, upper=n_arms, name=f"active[{level}]")
            for level, row in enumerate(active_rows)
        )
        normalized_outputs = tuple(
            _as_unique_indices(row, upper=n_arms, name=f"output[{level}]")
            for level, row in enumerate(output_rows)
        )
        for level, (active, output) in enumerate(
            zip(normalized_active, normalized_outputs, strict=True)
        ):
            if not set(output).issubset(active):
                raise ValueError(f"output[{level}] must be a subset of active[{level}]")
        self.n_arms = n_arms
        self.level_count = len(normalized_active)
        self.level_dimension = _next_power_of_two(self.level_count)
        self.index_dimension = _next_power_of_two(n_arms)
        self.active_indices_by_level = normalized_active
        self.output_indices_by_level = normalized_outputs
        self._active_sets = tuple(frozenset(row) for row in normalized_active)
        self._output_sets = tuple(frozenset(row) for row in normalized_outputs)
        self._ledger: Counter[str] = Counter()

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.level_dimension, self.index_dimension, 2, 2)

    @property
    def statevector_dimension(self) -> int:
        return int(np.prod(self.shape))

    def zero_state(self) -> ComplexState:
        state = np.zeros(self.statevector_dimension, dtype=np.complex128)
        state[0] = 1.0
        return state

    def uniform_history_state(
        self,
        *,
        levels: Sequence[int] | None = None,
        indices: Sequence[int] | None = None,
    ) -> ComplexState:
        selected_levels = (
            tuple(range(self.level_count)) if levels is None else tuple(levels)
        )
        selected_indices = tuple(range(self.n_arms)) if indices is None else tuple(indices)
        if not selected_levels:
            raise ValueError("levels cannot be empty")
        if not selected_indices:
            raise ValueError("indices cannot be empty")
        levels_checked = _as_unique_indices(
            selected_levels, upper=self.level_count, name="levels"
        )
        indices_checked = _as_unique_indices(
            selected_indices, upper=self.n_arms, name="indices"
        )
        view = np.zeros(self.shape, dtype=np.complex128)
        amplitude = 1.0 / math.sqrt(len(levels_checked) * len(indices_checked))
        for level in levels_checked:
            view[level, indices_checked, 0, 0] = amplitude
        return view.reshape(-1)

    def query_snapshot(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._ledger))

    def resource_snapshot(self) -> ActivityHistoryResources:
        return ActivityHistoryResources(
            query_counts=self.query_snapshot(),
            level_count=self.level_count,
            n_arms=self.n_arms,
            level_dimension=self.level_dimension,
            index_dimension=self.index_dimension,
            statevector_dimension=self.statevector_dimension,
        )

    def _view(self, state: ComplexState) -> ComplexState:
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.size != self.statevector_dimension:
            raise ValueError(
                "expected a flat statevector of length "
                f"{self.statevector_dimension}, got {values.shape}"
            )
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-10):
            raise ValueError("statevector must be normalized")
        return values.copy().reshape(self.shape)

    def is_active(self, level: int, index: int) -> bool:
        level = _integer(level, "level")
        index = _integer(index, "index")
        if not 0 <= level < self.level_count:
            return False
        if not 0 <= index < self.n_arms:
            return False
        return index in self._active_sets[level]

    def is_output(self, level: int, index: int) -> bool:
        level = _integer(level, "level")
        index = _integer(index, "index")
        if not 0 <= level < self.level_count:
            return False
        if not 0 <= index < self.n_arms:
            return False
        return index in self._output_sets[level]

    def apply_compute(self, state: ComplexState, *, tag: str = "history") -> ComplexState:
        view = self._view(state)
        for level in range(self.level_count):
            for index in range(self.n_arms):
                active = self.is_active(level, index)
                output = self.is_output(level, index)
                if not active and not output:
                    continue
                block = view[level, index].copy()
                for active_flag in range(2):
                    for output_flag in range(2):
                        new_active = active_flag ^ int(active)
                        new_output = output_flag ^ int(output)
                        view[level, index, new_active, new_output] = block[
                            active_flag, output_flag
                        ]
        self._ledger[f"{tag}:compute"] += 1
        return view.reshape(-1)

    def apply_phase(
        self,
        state: ComplexState,
        *,
        phase_on: str = "output",
        tag: str = "history",
    ) -> ActivityHistoryResult:
        if phase_on not in {"active", "output", "active_output"}:
            raise ValueError("phase_on must be active, output, or active_output")
        computed = self.apply_compute(state, tag=tag)
        view = self._view(computed)
        if phase_on == "active":
            mask = view[:, :, 1, :]
            active_probability = float(np.sum(np.abs(mask) ** 2))
            output_probability = float(np.sum(np.abs(view[:, :, :, 1]) ** 2))
            view[:, :, 1, :] *= -1.0
        elif phase_on == "output":
            active_probability = float(np.sum(np.abs(view[:, :, 1, :]) ** 2))
            output_probability = float(np.sum(np.abs(view[:, :, :, 1]) ** 2))
            view[:, :, :, 1] *= -1.0
        else:
            active_probability = float(np.sum(np.abs(view[:, :, 1, :]) ** 2))
            output_probability = float(np.sum(np.abs(view[:, :, :, 1]) ** 2))
            view[:, :, 1, 1] *= -1.0
        self._ledger[f"{tag}:phase:{phase_on}"] += 1
        restored = self.apply_compute(view.reshape(-1), tag=tag)
        return ActivityHistoryResult(
            state=restored,
            resources=self.resource_snapshot(),
            active_probability=active_probability,
            output_probability=output_probability,
        )


def toy_transducer_from_history_record(
    record: UnknownBoundaryHistoryRecord,
) -> ToyActivityHistoryTransducer:
    """Build a deterministic toy predicate family from an audit record.

    The construction is intentionally simple and public: every level gets a
    cyclic active window of the requested size, and the first
    ``output_births_per_level`` active indices are marked as newly required
    outputs.  It is a relation-semantics fixture, not a hard-instance proof.
    """

    if not isinstance(record, UnknownBoundaryHistoryRecord):
        raise TypeError("record must be an UnknownBoundaryHistoryRecord")
    active_rows: list[tuple[int, ...]] = []
    output_rows: list[tuple[int, ...]] = []
    stride = max(1, record.active_base_count)
    for layer in record.layers:
        start = (layer.level * stride) % record.n
        active = tuple((start + offset) % record.n for offset in range(layer.active_count))
        outputs = active[: layer.output_births]
        active_rows.append(active)
        output_rows.append(outputs)
    return ToyActivityHistoryTransducer(
        record.n,
        active_rows,
        output_rows,
    )
