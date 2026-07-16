from __future__ import annotations

import math

import pytest

from qgapselect.adaptive_phase import AdaptivePhaseQubitScheduler


def test_scheduler_selects_smallest_guard_and_budget_feasible_precision() -> None:
    scheduler = AdaptivePhaseQubitScheduler(
        4,
        minimum_phase_qubits=4,
        maximum_phase_qubits=8,
        max_statevector_dimension=100_000,
    )

    schedule = scheduler.schedule(math.pi / 32.0)

    assert schedule.required_phase_qubits == 6
    assert schedule.selected_phase_qubits == 6
    assert schedule.complete
    assert schedule.blocked_reason is None
    assert schedule.evidence_source == "measured_boundary_angular_margin_only"
    assert [point.phase_qubits for point in schedule.candidates] == [4, 5, 6, 7, 8]
    assert not schedule.candidates[1].phase_guard_passed
    assert schedule.candidates[2].phase_guard_passed


def test_candidate_peak_dimension_matches_direct_phase_allocation_model() -> None:
    scheduler = AdaptivePhaseQubitScheduler(
        4,
        minimum_phase_qubits=7,
        maximum_phase_qubits=7,
        max_statevector_dimension=18_432,
    )

    point = scheduler.candidate(7, math.pi / 32.0)

    assert point.retained_statevector_dimension == 128 * 4 * 2
    assert point.comparator_expanded_statevector_dimension == 2 * 128 * 4 * 2
    assert point.dense_qft_matrix_dimension == 128**2
    assert point.estimated_peak_complex_amplitudes == 18_432
    assert point.statevector_budget_passed


def test_schedule_distinguishes_precision_cap_from_memory_block() -> None:
    precision_blocked = AdaptivePhaseQubitScheduler(
        4,
        minimum_phase_qubits=4,
        maximum_phase_qubits=6,
        max_statevector_dimension=1_000_000,
    ).schedule(math.pi / 128.0)
    memory_blocked = AdaptivePhaseQubitScheduler(
        4,
        minimum_phase_qubits=4,
        maximum_phase_qubits=8,
        max_statevector_dimension=18_431,
    ).schedule(math.pi / 32.0)

    assert precision_blocked.required_phase_qubits == 8
    assert precision_blocked.selected_phase_qubits is None
    assert precision_blocked.blocked_reason == (
        "phase_resolution_exceeds_maximum_phase_qubits"
    )
    assert memory_blocked.required_phase_qubits == 6
    assert memory_blocked.selected_phase_qubits == 6

    tighter_memory_blocked = AdaptivePhaseQubitScheduler(
        4,
        minimum_phase_qubits=4,
        maximum_phase_qubits=8,
        max_statevector_dimension=5_119,
    ).schedule(math.pi / 32.0)
    assert tighter_memory_blocked.selected_phase_qubits is None
    assert tighter_memory_blocked.blocked_reason == (
        "statevector_budget_exceeded_for_required_phase_resolution"
    )


@pytest.mark.parametrize(
    ("margin", "required"),
    [
        (math.pi, 1),
        (math.pi / 2.0, 2),
        (math.pi / 32.0, 6),
        (math.nextafter(math.pi / 32.0, 0.0), 7),
    ],
)
def test_required_precision_handles_exact_and_adjacent_boundaries(
    margin: float,
    required: int,
) -> None:
    assert AdaptivePhaseQubitScheduler.required_phase_qubits(margin) == required


@pytest.mark.parametrize(
    "margin",
    [0.0, -0.1, math.inf, -math.inf, math.nan],
)
def test_scheduler_rejects_nonpositive_or_nonfinite_margin(margin: float) -> None:
    scheduler = AdaptivePhaseQubitScheduler(
        2,
        minimum_phase_qubits=1,
        maximum_phase_qubits=4,
        max_statevector_dimension=4096,
    )
    with pytest.raises(ValueError):
        scheduler.schedule(margin)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"index_dimension": 3}, ValueError),
        ({"minimum_phase_qubits": 0}, ValueError),
        ({"maximum_phase_qubits": 3}, ValueError),
        ({"maximum_phase_qubits": 13}, ValueError),
        ({"max_statevector_dimension": 0}, ValueError),
        ({"minimum_phase_qubits": True}, TypeError),
    ],
)
def test_scheduler_validates_configuration(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    values: dict[str, object] = {
        "index_dimension": 4,
        "minimum_phase_qubits": 4,
        "maximum_phase_qubits": 8,
        "max_statevector_dimension": 100_000,
    }
    values.update(kwargs)
    with pytest.raises(error):
        AdaptivePhaseQubitScheduler(**values)  # type: ignore[arg-type]
