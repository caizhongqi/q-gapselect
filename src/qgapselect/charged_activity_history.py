"""Charged finite-phase activity-history prototype.

This module is the next executable step after
``activity_history_transducer.py``.  The older transducer proves only the
relation semantics when active/output predicate rows are supplied externally.
Here the predicates are generated from a charged finite-phase classifier:

* each arm has a canonical phase value in a normalized ``[0, 1]`` interval;
* each activity level has a finite-QPE precision;
* the classifier rounds the phase to the level grid and computes a conservative
  confidence interval; and
* activity/output flags are XORed into work bits and uncomputed exactly.

This is still a prototype, not the P0-U08 theorem.  It does not localize the
unknown Top-k boundary by itself and it does not prove a variable-time upper
bound.  Its purpose is to remove the most obvious toy shortcut: supplied
predicate rows.
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

ComplexState = NDArray[np.complex128]
BACKEND = "numpy_exact_statevector_charged_phase_prototype"
CLAIM_STATUS = "charged_phase_history_prototype_no_upper_bound_theorem"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


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


def _unit_interval(value: object, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


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
class FinitePhaseClassification:
    """One finite-QPE interval classification for a ``(level, arm)`` pair."""

    level: int
    index: int
    true_phase: float
    estimated_phase: float
    qpe_step: float
    interval_low: float
    interval_high: float
    boundary_phase: float
    activity_window: float
    active: bool
    output: bool
    qpe_query_units: int


@dataclass(frozen=True, slots=True)
class ChargedActivityResources:
    """Logical resource ledger for the charged phase prototype."""

    query_counts: Mapping[str, int]
    level_count: int
    n_arms: int
    precision_bits_by_level: tuple[int, ...]
    level_dimension: int
    index_dimension: int
    statevector_dimension: int
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS

    @property
    def total_charged_query_units(self) -> int:
        return int(
            self.query_counts.get("charged:qpe_query_units_serial_levels", 0)
        )

    @property
    def total_compute_calls(self) -> int:
        return int(
            sum(
                value
                for key, value in self.query_counts.items()
                if key.endswith(":compute")
            )
        )


@dataclass(frozen=True, slots=True)
class ChargedActivityResult:
    """State and diagnostics after a charged history operation."""

    state: ComplexState
    resources: ChargedActivityResources
    active_probability: float
    output_probability: float


@dataclass(frozen=True, slots=True)
class ChargedActivitySummary:
    """Classical summary of generated predicates.

    This summary is deliberately derived by calling the finite-phase classifier,
    not by accepting predicate rows as input.
    """

    active_counts_by_level: tuple[int, ...]
    output_counts_by_level: tuple[int, ...]
    output_subset_active: bool
    min_active_count: int
    max_active_count: int
    total_active_pairs: int
    total_output_pairs: int
    serial_qpe_query_units_per_compute: int
    max_level_qpe_query_units_per_compute: int
    predicate_source: str = "finite_qpe_phase_windows"


class ChargedPhaseHistoryTransducer:
    """Exact-state relation oracle with finite-QPE-derived predicates.

    Registers are ordered as ``(level, index, active_flag, output_flag)``.
    ``apply_compute`` XORs flags generated by :meth:`classify`; applying it
    twice restores the input state.  A coherent implementation would replace the
    NumPy loops with the corresponding reversible phase-estimation comparator.
    The ledger charges that comparator explicitly and never charges a supplied
    predicate-row lookup.
    """

    def __init__(
        self,
        arm_phases: Sequence[float],
        precision_bits_by_level: Sequence[int],
        *,
        boundary_phase: float = 0.5,
        activity_window_multipliers: Sequence[float] | None = None,
        guard_band: float = 0.0,
    ) -> None:
        try:
            phases = tuple(
                _unit_interval(value, f"arm_phases[{index}]")
                for index, value in enumerate(arm_phases)
            )
        except TypeError as error:
            raise TypeError("arm_phases must be a sequence of real numbers") from error
        if len(phases) < 2:
            raise ValueError("at least two arms are required")
        try:
            bits = tuple(
                _integer(value, f"precision_bits_by_level[{index}]")
                for index, value in enumerate(precision_bits_by_level)
            )
        except TypeError as error:
            raise TypeError("precision_bits_by_level must be a sequence") from error
        if not bits:
            raise ValueError("at least one precision level is required")
        if any(value < 1 for value in bits):
            raise ValueError("precision bits must be at least one")
        if activity_window_multipliers is None:
            multipliers = tuple(2.0 for _ in bits)
        else:
            try:
                multipliers = tuple(
                    _positive(value, f"activity_window_multipliers[{index}]")
                    for index, value in enumerate(activity_window_multipliers)
                )
            except TypeError as error:
                raise TypeError(
                    "activity_window_multipliers must be a sequence"
                ) from error
            if len(multipliers) != len(bits):
                raise ValueError(
                    "activity_window_multipliers must match precision levels"
                )
        self.arm_phases = phases
        self.precision_bits_by_level = bits
        self.boundary_phase = _unit_interval(boundary_phase, "boundary_phase")
        self.activity_window_multipliers = multipliers
        self.guard_band = _nonnegative(guard_band, "guard_band")
        self.n_arms = len(phases)
        self.level_count = len(bits)
        self.level_dimension = _next_power_of_two(self.level_count)
        self.index_dimension = _next_power_of_two(self.n_arms)
        self._ledger: Counter[str] = Counter()

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.level_dimension, self.index_dimension, 2, 2)

    @property
    def statevector_dimension(self) -> int:
        return int(np.prod(self.shape))

    @property
    def qpe_query_units_by_level(self) -> tuple[int, ...]:
        return tuple((1 << bits) - 1 for bits in self.precision_bits_by_level)

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

    def resource_snapshot(self) -> ChargedActivityResources:
        return ChargedActivityResources(
            query_counts=self.query_snapshot(),
            level_count=self.level_count,
            n_arms=self.n_arms,
            precision_bits_by_level=self.precision_bits_by_level,
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

    def _charge_compute(self, tag: str) -> None:
        units = self.qpe_query_units_by_level
        self._ledger[f"{tag}:compute"] += 1
        self._ledger["charged:coherent_classifier_calls"] += 1
        self._ledger["charged:qpe_query_units_serial_levels"] += int(sum(units))
        self._ledger["charged:qpe_query_units_max_level"] += int(max(units))

    def classify(self, level: int, index: int) -> FinitePhaseClassification:
        """Return the charged finite-phase classification for one basis pair."""

        level = _integer(level, "level")
        index = _integer(index, "index")
        if not 0 <= level < self.level_count:
            raise IndexError("level is outside the valid precision levels")
        if not 0 <= index < self.n_arms:
            raise IndexError("index is outside the valid arms")
        bits = self.precision_bits_by_level[level]
        qpe_bins = 1 << bits
        qpe_step = 1.0 / qpe_bins
        phase = self.arm_phases[index]
        estimated = round(phase / qpe_step) * qpe_step
        estimated = min(1.0, max(0.0, estimated))
        half_width = 0.5 * qpe_step + self.guard_band
        interval_low = max(0.0, estimated - half_width)
        interval_high = min(1.0, estimated + half_width)
        activity_window = (
            self.activity_window_multipliers[level] * qpe_step + self.guard_band
        )
        active = (
            interval_high >= self.boundary_phase
            and interval_low <= self.boundary_phase + activity_window
        )
        output = (
            interval_low >= self.boundary_phase
            and interval_low <= self.boundary_phase + activity_window
        )
        return FinitePhaseClassification(
            level=level,
            index=index,
            true_phase=phase,
            estimated_phase=estimated,
            qpe_step=qpe_step,
            interval_low=interval_low,
            interval_high=interval_high,
            boundary_phase=self.boundary_phase,
            activity_window=activity_window,
            active=active,
            output=output,
            qpe_query_units=(1 << bits) - 1,
        )

    def is_active(self, level: int, index: int) -> bool:
        return self.classify(level, index).active

    def is_output(self, level: int, index: int) -> bool:
        return self.classify(level, index).output

    def predicate_rows(self) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """Return generated active/output rows for diagnostics."""

        active_rows = []
        output_rows = []
        for level in range(self.level_count):
            active = []
            output = []
            for index in range(self.n_arms):
                classification = self.classify(level, index)
                if classification.active:
                    active.append(index)
                if classification.output:
                    output.append(index)
            active_rows.append(tuple(active))
            output_rows.append(tuple(output))
        return tuple(active_rows), tuple(output_rows)

    def summarize_predicates(self) -> ChargedActivitySummary:
        active_rows, output_rows = self.predicate_rows()
        active_sets = tuple(frozenset(row) for row in active_rows)
        output_subset = all(
            set(output).issubset(active)
            for output, active in zip(output_rows, active_sets, strict=True)
        )
        active_counts = tuple(len(row) for row in active_rows)
        output_counts = tuple(len(row) for row in output_rows)
        units = self.qpe_query_units_by_level
        return ChargedActivitySummary(
            active_counts_by_level=active_counts,
            output_counts_by_level=output_counts,
            output_subset_active=output_subset,
            min_active_count=min(active_counts),
            max_active_count=max(active_counts),
            total_active_pairs=sum(active_counts),
            total_output_pairs=sum(output_counts),
            serial_qpe_query_units_per_compute=int(sum(units)),
            max_level_qpe_query_units_per_compute=int(max(units)),
        )

    def apply_compute(self, state: ComplexState, *, tag: str = "charged_history") -> ComplexState:
        view = self._view(state)
        for level in range(self.level_count):
            for index in range(self.n_arms):
                classification = self.classify(level, index)
                if not classification.active and not classification.output:
                    continue
                block = view[level, index].copy()
                for active_flag in range(2):
                    for output_flag in range(2):
                        new_active = active_flag ^ int(classification.active)
                        new_output = output_flag ^ int(classification.output)
                        view[level, index, new_active, new_output] = block[
                            active_flag, output_flag
                        ]
        self._charge_compute(tag)
        return view.reshape(-1)

    def apply_phase(
        self,
        state: ComplexState,
        *,
        phase_on: str = "output",
        tag: str = "charged_history",
    ) -> ChargedActivityResult:
        if phase_on not in {"active", "output", "active_output"}:
            raise ValueError("phase_on must be active, output, or active_output")
        computed = self.apply_compute(state, tag=tag)
        view = self._view(computed)
        active_probability = float(np.sum(np.abs(view[:, :, 1, :]) ** 2))
        output_probability = float(np.sum(np.abs(view[:, :, :, 1]) ** 2))
        if phase_on == "active":
            view[:, :, 1, :] *= -1.0
        elif phase_on == "output":
            view[:, :, :, 1] *= -1.0
        else:
            view[:, :, 1, 1] *= -1.0
        self._ledger[f"{tag}:phase:{phase_on}"] += 1
        restored = self.apply_compute(view.reshape(-1), tag=tag)
        return ChargedActivityResult(
            state=restored,
            resources=self.resource_snapshot(),
            active_probability=active_probability,
            output_probability=output_probability,
        )


def deterministic_boundary_phases(
    n_arms: int,
    *,
    boundary_phase: float = 0.5,
    near_boundary_fraction: float = 0.25,
) -> tuple[float, ...]:
    """Return a deterministic phase fixture concentrated near the boundary."""

    n_arms = _integer(n_arms, "n_arms")
    if n_arms < 2:
        raise ValueError("n_arms must be at least two")
    boundary = _unit_interval(boundary_phase, "boundary_phase")
    fraction = _positive(near_boundary_fraction, "near_boundary_fraction")
    if fraction > 1.0:
        raise ValueError("near_boundary_fraction must be at most one")
    near_count = max(2, int(math.ceil(n_arms * fraction)))
    far_count = n_arms - near_count
    phases: list[float] = []
    if far_count:
        below = far_count // 2
        above = far_count - below
        phases.extend(
            max(0.0, boundary - 0.45 + 0.20 * (index + 0.5) / max(1, below))
            for index in range(below)
        )
        phases.extend(
            min(1.0, boundary + 0.25 + 0.20 * (index + 0.5) / max(1, above))
            for index in range(above)
        )
    for offset in range(near_count):
        centered = (offset + 0.5) / near_count - 0.5
        phases.append(min(1.0, max(0.0, boundary + 0.125 * centered)))
    return tuple(sorted(phases))


def logarithmic_precision_schedule(
    level_count: int,
    *,
    base_bits: int = 3,
    growth_period: int = 2,
) -> tuple[int, ...]:
    """Return a simple nondecreasing precision schedule for experiments."""

    level_count = _integer(level_count, "level_count")
    base_bits = _integer(base_bits, "base_bits")
    growth_period = _integer(growth_period, "growth_period")
    if level_count < 1:
        raise ValueError("level_count must be positive")
    if base_bits < 1:
        raise ValueError("base_bits must be at least one")
    if growth_period < 1:
        raise ValueError("growth_period must be positive")
    return tuple(base_bits + level // growth_period for level in range(level_count))
