"""Evidence-only phase-register scheduling for the direct QPE backend.

The scheduler consumes a numerical angular margin produced by a measured
boundary certificate.  It never receives arm means, rankings, or membership
sets.  Its memory model follows the analytic live-array-size proxy used by
``DirectAmplitudeThresholdFlag``; it is not measured NumPy peak memory or a
hardware-resource claim.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

MAX_EXACT_STATE_PHASE_QUBITS = 12
PHASE_GUARD_FACTOR = 2.0
CLAIM_STATUS = "measured_margin_phase_schedule_no_complexity_theorem"
BUDGET_SEMANTICS = "analytic_array_size_proxy_not_measured_numpy_peak"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


@dataclass(frozen=True, slots=True)
class PhaseQubitCandidate:
    """One auditable precision/resource point in an adaptive schedule."""

    phase_qubits: int
    phase_resolution: float
    retained_statevector_dimension: int
    comparator_expanded_statevector_dimension: int
    dense_qft_matrix_dimension: int
    estimated_peak_complex_amplitudes: int
    phase_guard_passed: bool
    statevector_budget_passed: bool
    budget_semantics: str = BUDGET_SEMANTICS


@dataclass(frozen=True, slots=True)
class AdaptivePhaseSchedule:
    """Selected precision or an explicit reason that no precision is usable."""

    angular_margin: float
    required_phase_qubits: int
    minimum_phase_qubits: int
    maximum_phase_qubits: int
    selected_phase_qubits: int | None
    candidates: tuple[PhaseQubitCandidate, ...]
    complete: bool
    blocked_reason: str | None
    evidence_source: str = "measured_boundary_angular_margin_only"
    claim_status: str = CLAIM_STATUS
    budget_semantics: str = BUDGET_SEMANTICS

    @property
    def selected(self) -> PhaseQubitCandidate | None:
        if self.selected_phase_qubits is None:
            return None
        return next(
            candidate
            for candidate in self.candidates
            if candidate.phase_qubits == self.selected_phase_qubits
        )


class AdaptivePhaseQubitScheduler:
    """Choose the least expensive phase register certified by a measured gap.

    For phase resolution ``pi / 2**m`` and measured angular margin ``gamma``,
    the existing direct-search execution guard requires

    ``pi / 2**m <= gamma / 2``.

    Candidates are considered from ``minimum_phase_qubits`` upward.  A
    candidate is selectable only if it passes both that guard and the analytic
    array-size-proxy budget.  Because the proxy is monotone in ``m``, the first
    selectable candidate is also the smallest under this model.
    """

    def __init__(
        self,
        index_dimension: int,
        *,
        minimum_phase_qubits: int,
        maximum_phase_qubits: int,
        max_statevector_dimension: int,
    ) -> None:
        index_dimension = _integer(index_dimension, "index_dimension")
        minimum_phase_qubits = _integer(
            minimum_phase_qubits, "minimum_phase_qubits"
        )
        maximum_phase_qubits = _integer(
            maximum_phase_qubits, "maximum_phase_qubits"
        )
        max_statevector_dimension = _integer(
            max_statevector_dimension, "max_statevector_dimension"
        )
        if index_dimension <= 0 or index_dimension & (index_dimension - 1):
            raise ValueError("index_dimension must be a positive power of two")
        if minimum_phase_qubits <= 0:
            raise ValueError("minimum_phase_qubits must be positive")
        if maximum_phase_qubits < minimum_phase_qubits:
            raise ValueError(
                "maximum_phase_qubits must be at least minimum_phase_qubits"
            )
        if maximum_phase_qubits > MAX_EXACT_STATE_PHASE_QUBITS:
            raise ValueError(
                "maximum_phase_qubits exceeds the exact-state limit of 12"
            )
        if max_statevector_dimension <= 0:
            raise ValueError("max_statevector_dimension must be positive")

        self.index_dimension = index_dimension
        self.minimum_phase_qubits = minimum_phase_qubits
        self.maximum_phase_qubits = maximum_phase_qubits
        self.max_statevector_dimension = max_statevector_dimension

    def candidate(self, phase_qubits: int, angular_margin: float) -> PhaseQubitCandidate:
        """Evaluate one in-range candidate without accessing oracle internals."""

        phase_qubits = _integer(phase_qubits, "phase_qubits")
        margin = self._angular_margin(angular_margin)
        if not self.minimum_phase_qubits <= phase_qubits <= self.maximum_phase_qubits:
            raise ValueError("phase_qubits is outside the configured schedule")
        phase_bins = 1 << phase_qubits
        resolution = math.pi / phase_bins
        retained = phase_bins * self.index_dimension * 2
        comparator = 2 * retained
        dense_qft = phase_bins * phase_bins
        peak = max(3 * retained, dense_qft + 2 * retained)
        return PhaseQubitCandidate(
            phase_qubits=phase_qubits,
            phase_resolution=resolution,
            retained_statevector_dimension=retained,
            comparator_expanded_statevector_dimension=comparator,
            dense_qft_matrix_dimension=dense_qft,
            estimated_peak_complex_amplitudes=peak,
            phase_guard_passed=resolution <= margin / PHASE_GUARD_FACTOR,
            statevector_budget_passed=peak <= self.max_statevector_dimension,
        )

    @staticmethod
    def _angular_margin(value: object) -> float:
        if isinstance(value, bool):
            raise TypeError("angular_margin must be a real number, not bool")
        try:
            margin = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError("angular_margin must be a real number") from error
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError("angular_margin must be finite and positive")
        return margin

    @staticmethod
    def required_phase_qubits(angular_margin: float) -> int:
        """Return the smallest positive ``m`` satisfying the phase guard."""

        margin = AdaptivePhaseQubitScheduler._angular_margin(angular_margin)
        required = max(1, math.ceil(math.log2(PHASE_GUARD_FACTOR * math.pi / margin)))
        # Protect the boundary case against a final floating-point log2 round.
        while math.pi / (1 << required) > margin / PHASE_GUARD_FACTOR:
            required += 1
        while (
            required > 1
            and math.pi / (1 << (required - 1)) <= margin / PHASE_GUARD_FACTOR
        ):
            required -= 1
        return required

    def schedule(self, angular_margin: float) -> AdaptivePhaseSchedule:
        """Build the schedule and select the first guard-and-budget feasible ``m``."""

        margin = self._angular_margin(angular_margin)
        required = self.required_phase_qubits(margin)
        candidates = tuple(
            self.candidate(phase_qubits, margin)
            for phase_qubits in range(
                self.minimum_phase_qubits, self.maximum_phase_qubits + 1
            )
        )
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.phase_guard_passed
                and candidate.statevector_budget_passed
            ),
            None,
        )
        if selected is not None:
            blocked_reason = None
        elif required > self.maximum_phase_qubits:
            blocked_reason = "phase_resolution_exceeds_maximum_phase_qubits"
        else:
            blocked_reason = "statevector_budget_exceeded_for_required_phase_resolution"
        return AdaptivePhaseSchedule(
            angular_margin=margin,
            required_phase_qubits=required,
            minimum_phase_qubits=self.minimum_phase_qubits,
            maximum_phase_qubits=self.maximum_phase_qubits,
            selected_phase_qubits=(
                None if selected is None else selected.phase_qubits
            ),
            candidates=candidates,
            complete=selected is not None,
            blocked_reason=blocked_reason,
        )
