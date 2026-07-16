from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest

from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.direct_topk import CalibratedDirectTopKController


@dataclass(frozen=True)
class _FakeSearchResult:
    found_indices: tuple[int, ...]
    complete: bool
    status: str
    attempts: int
    trace: tuple[dict[str, int], ...]
    resources: dict[str, int]


class _SpySearch:
    calls: list[dict[str, Any]] = []
    delay_steps = 1

    def __init__(
        self,
        oracle: CanonicalRyStatevectorOracle,
        threshold: float,
        expected_count: int,
        **kwargs: Any,
    ) -> None:
        self.oracle = oracle
        self.threshold = threshold
        self.expected_count = expected_count
        self.kwargs = kwargs
        self._steps = 0
        self._found: tuple[int, ...] = ()
        type(self).calls.append(
            {
                "oracle": oracle,
                "threshold": threshold,
                "expected_count": expected_count,
                "kwargs": dict(kwargs),
            }
        )

    def step(self) -> _FakeSearchResult:
        self._steps += 1
        if self._steps > self.delay_steps:
            validator = self.kwargs["candidate_validator"]
            self._found = tuple(
                index
                for index in range(self.oracle.n_arms)
                if validator(index)
            )[: self.expected_count]
        return self.result()

    def result(self) -> _FakeSearchResult:
        complete = len(self._found) == self.expected_count
        return _FakeSearchResult(
            found_indices=self._found,
            complete=complete,
            status="complete_fixed_confidence" if complete else "paused_resumable",
            attempts=self._steps,
            trace=tuple({"step": step} for step in range(1, self._steps + 1)),
            resources={"oracle_queries": 0},
        )

    def run(self, max_steps: int | None = None) -> _FakeSearchResult:
        budget = 8 if max_steps is None else max_steps
        for _ in range(budget):
            if self.result().complete:
                break
            self.step()
        return self.result()


@pytest.fixture
def spy_search_module(monkeypatch: pytest.MonkeyPatch) -> type[_SpySearch]:
    _SpySearch.calls = []
    module = ModuleType("qgapselect.direct_search")
    module.FullWorkspaceBBHT = _SpySearch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qgapselect.direct_search", module)
    return _SpySearch


def test_controller_completes_from_direct_branch_and_checks_final_intervals(
    spy_search_module: type[_SpySearch],
) -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 1.0, 0.0, 0.0], seed=4)
    controller = CalibratedDirectTopKController(
        oracle,
        2,
        phase_qubits=5,
        confidence=0.1,
        boundary_shots_per_round=32,
        max_boundary_rounds=3,
        verification_shots=16,
        seed=9,
    )

    paused = controller.run(max_steps=1)
    assert not paused.complete
    assert paused.selected == ()
    assert paused.rejected == ()
    assert paused.status == "paused_resumable"

    completed = controller.resume()
    assert completed.complete
    assert completed.status == "complete_fixed_confidence"
    assert completed.selected == (0, 1)
    assert completed.rejected == (2, 3)
    assert completed.winning_orientation == "selected"
    assert completed.mean_threshold is not None
    assert completed.angular_threshold is not None
    assert completed.angular_margin is not None
    assert completed.phase_guard_passed
    assert all(
        completed.boundary.intervals[index].angular_lower
        > completed.angular_threshold
        for index in completed.selected
    )
    assert all(
        completed.boundary.intervals[index].angular_upper
        < completed.angular_threshold
        for index in completed.rejected
    )
    assert completed.resources.boundary_query_counts["forward"] > 0
    assert completed.resources.query_counts["forward"] > 0
    assert (
        completed.resources.search_query_counts["coherent_total"]
        == completed.resources.query_counts["coherent_total"]
        - completed.resources.boundary_query_counts["coherent_total"]
    )
    assert (
        completed.boundary.resources.query_counts
        == completed.resources.boundary_query_counts
    )


def test_fair_scheduler_builds_both_relations_without_known_membership_sets(
    spy_search_module: type[_SpySearch],
) -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 1.0, 0.0, 0.0])
    completed = CalibratedDirectTopKController(
        oracle,
        2,
        phase_qubits=5,
        confidence=0.1,
        boundary_shots_per_round=32,
        max_boundary_rounds=2,
        seed=22,
    ).run()

    assert completed.complete
    assert len(spy_search_module.calls) == 2
    calls = {call["kwargs"]["relation"]: call for call in spy_search_module.calls}
    assert set(calls) == {"above", "below"}
    assert calls["above"]["expected_count"] == 2
    assert calls["below"]["expected_count"] == 2
    assert calls["above"]["threshold"] == calls["below"]["threshold"]
    forbidden = {
        "marked_indices",
        "selected",
        "selected_indices",
        "rejected",
        "rejected_indices",
        "targets",
    }
    assert all(forbidden.isdisjoint(call["kwargs"]) for call in calls.values())
    assert all(call["oracle"] is oracle for call in calls.values())
    assert {trace.relation for trace in completed.branches} == {"above", "below"}
    traces = {trace.relation: trace for trace in completed.branches}
    assert traces["above"].steps == 2
    assert traces["below"].steps == 1
    assert traces["below"].status == "paused_resumable"


def test_search_mean_threshold_is_derived_from_the_single_angular_boundary(
    spy_search_module: type[_SpySearch],
) -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 0.81, 0.09, 0.0], seed=44)
    result = CalibratedDirectTopKController(
        oracle,
        2,
        phase_qubits=7,
        confidence=0.1,
        boundary_shots_per_round=128,
        max_boundary_rounds=3,
        seed=45,
    ).run()

    assert result.complete
    assert result.angular_threshold is not None
    assert result.mean_threshold == pytest.approx(
        math.sin(result.angular_threshold) ** 2
    )
    assert result.boundary.certificate is not None
    assert result.boundary.certificate.mean_threshold is not None
    assert result.mean_threshold != pytest.approx(
        result.boundary.certificate.mean_threshold, abs=1e-4
    )
    assert all(
        call["threshold"] == pytest.approx(result.mean_threshold)
        for call in spy_search_module.calls
    )


def test_boundary_failure_never_emits_partial_topk() -> None:
    oracle = CanonicalRyStatevectorOracle([0.51, 0.5, 0.49, 0.48], seed=1)
    result = CalibratedDirectTopKController(
        oracle,
        2,
        phase_qubits=8,
        confidence=0.01,
        boundary_shots_per_round=1,
        max_boundary_rounds=1,
        seed=2,
    ).run()

    assert not result.complete
    assert result.status == "boundary_not_separated"
    assert result.selected == ()
    assert result.rejected == ()
    assert result.winning_orientation is None
    assert result.phase_guard_passed is None
    assert result.branches == ()
    assert result.resources.boundary_query_counts["forward"] == 4
    assert result.resources.search_query_counts["coherent_total"] == 0


def test_phase_resolution_guard_is_explicitly_heuristic_and_gates_search() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 1.0, 0.0, 0.0])
    result = CalibratedDirectTopKController(
        oracle,
        2,
        phase_qubits=1,
        confidence=0.1,
        boundary_shots_per_round=32,
        max_boundary_rounds=2,
        seed=5,
    ).run()

    assert not result.complete
    assert result.status == "phase_resolution_insufficient"
    assert result.phase_guard_passed is False
    assert result.angular_margin is not None
    assert result.phase_resolution > result.angular_margin / 2.0
    assert result.phase_guard_status == (
        "heuristic_execution_guard_not_a_complexity_theorem"
    )
    assert result.selected == ()
    assert result.rejected == ()
    assert result.branches == ()
    assert result.boundary.minimum_angular_margin == 0.0
    assert not result.adaptive_phase_qubits
    assert result.phase_schedule is None


def test_adaptive_phase_mode_uses_measured_margin_and_smallest_precision(
    spy_search_module: type[_SpySearch],
) -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 1.0, 0.0, 0.0], seed=4)
    result = CalibratedDirectTopKController(
        oracle,
        2,
        phase_qubits=1,
        adaptive_phase_qubits=True,
        max_phase_qubits=7,
        confidence=0.1,
        boundary_shots_per_round=32,
        max_boundary_rounds=4,
        verification_shots=16,
        max_statevector_dimension=100_000,
        seed=9,
    ).run()

    assert result.complete
    assert result.adaptive_phase_qubits
    assert result.initial_phase_qubits == 1
    assert result.max_phase_qubits == 7
    assert result.boundary.minimum_angular_margin == pytest.approx(math.pi / 64.0)
    assert result.phase_schedule is not None
    assert result.phase_schedule.evidence_source == (
        "measured_boundary_angular_margin_only"
    )
    assert result.phase_schedule.angular_margin == result.angular_margin
    assert result.phase_schedule.selected_phase_qubits == 4
    assert result.resources.phase_qubits == 4
    assert result.resources.initial_phase_qubits == 1
    assert result.resources.max_phase_qubits == 7
    assert result.resources.adaptive_phase_qubits
    assert result.phase_resolution == pytest.approx(math.pi / 16.0)
    assert result.phase_guard_passed
    assert result.phase_guard_status == (
        "adaptive_measured_margin_execution_guard_not_a_complexity_theorem"
    )
    assert all(
        call["kwargs"]["phase_qubits"] == 4 for call in spy_search_module.calls
    )
    forbidden = {
        "marked_indices",
        "selected",
        "selected_indices",
        "rejected",
        "rejected_indices",
        "targets",
    }
    assert all(forbidden.isdisjoint(call["kwargs"]) for call in spy_search_module.calls)


def test_adaptive_phase_mode_completes_real_search_that_fixed_m1_blocks() -> None:
    fixed = CalibratedDirectTopKController(
        CanonicalRyStatevectorOracle([1.0, 0.0], seed=1),
        1,
        phase_qubits=1,
        confidence=0.1,
        boundary_shots_per_round=32,
        max_boundary_rounds=4,
        verification_shots=64,
        seed=2,
    ).run()
    adaptive = CalibratedDirectTopKController(
        CanonicalRyStatevectorOracle([1.0, 0.0], seed=1),
        1,
        phase_qubits=1,
        adaptive_phase_qubits=True,
        max_phase_qubits=5,
        confidence=0.1,
        boundary_shots_per_round=32,
        max_boundary_rounds=4,
        verification_shots=64,
        seed=2,
    ).run()

    assert fixed.status == "phase_resolution_insufficient"
    assert adaptive.complete
    assert adaptive.selected == (0,)
    assert adaptive.rejected == (1,)
    assert adaptive.resources.phase_qubits == 4
    assert adaptive.resources.search_query_counts["coherent_total"] > 0


def test_adaptive_phase_reports_memory_block_before_constructing_searches() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 0.0], seed=6)
    result = CalibratedDirectTopKController(
        oracle,
        1,
        phase_qubits=1,
        adaptive_phase_qubits=True,
        max_phase_qubits=5,
        confidence=0.1,
        boundary_shots_per_round=32,
        max_boundary_rounds=4,
        max_statevector_dimension=383,
        seed=7,
    ).run()

    assert not result.complete
    assert result.status == (
        "statevector_budget_exceeded_for_required_phase_resolution"
    )
    assert result.phase_schedule is not None
    assert result.phase_schedule.required_phase_qubits == 4
    assert result.phase_schedule.selected_phase_qubits is None
    assert result.phase_guard_passed is False
    assert result.branches == ()
    assert result.resources.search_query_counts["coherent_total"] == 0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"adaptive_phase_qubits": 1}, TypeError),
        ({"max_phase_qubits": 4}, ValueError),
        (
            {"adaptive_phase_qubits": True, "max_phase_qubits": 0},
            ValueError,
        ),
        (
            {"adaptive_phase_qubits": True, "max_phase_qubits": 13},
            ValueError,
        ),
        (
            {
                "phase_qubits": 1,
                "adaptive_phase_qubits": True,
                "max_phase_qubits": 1,
            },
            ValueError,
        ),
    ],
)
def test_adaptive_phase_configuration_is_strict(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    values: dict[str, object] = {"phase_qubits": 5}
    values.update(kwargs)
    with pytest.raises(error):
        CalibratedDirectTopKController(
            CanonicalRyStatevectorOracle([1.0, 0.0]),
            1,
            **values,  # type: ignore[arg-type]
        )


def test_statevector_budget_failure_is_propagated_from_both_real_searches() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 0.0], seed=6)
    result = CalibratedDirectTopKController(
        oracle,
        1,
        phase_qubits=5,
        confidence=0.1,
        boundary_shots_per_round=32,
        max_boundary_rounds=2,
        max_statevector_dimension=1,
        seed=7,
    ).run()

    assert not result.complete
    assert result.status == "statevector_budget_exceeded"
    assert result.selected == ()
    assert result.rejected == ()
    assert {branch.status for branch in result.branches} == {
        "statevector_budget_exceeded"
    }
    assert result.resources.search_query_counts["coherent_total"] == 0


def test_real_full_workspace_search_completes_end_to_end() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 0.0], seed=1)
    result = CalibratedDirectTopKController(
        oracle,
        1,
        phase_qubits=5,
        confidence=0.1,
        boundary_shots_per_round=32,
        max_boundary_rounds=2,
        verification_shots=64,
        max_attempts_per_output=12,
        seed=2,
    ).run()

    assert result.complete
    assert result.status == "complete_fixed_confidence"
    assert result.selected == (0,)
    assert result.rejected == (1,)
    assert result.resources.search_query_counts["coherent_total"] > 0
    winner = next(
        branch
        for branch in result.branches
        if branch.orientation == result.winning_orientation
    )
    assert winner.complete
    assert winner.found_indices in {(0,), (1,)}
    assert winner.resources.oracle_queries > 0


@pytest.mark.parametrize(
    ("k", "error"),
    [(0, ValueError), (2, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_controller_requires_strict_nontrivial_integer_k(
    k: object, error: type[Exception]
) -> None:
    oracle = CanonicalRyStatevectorOracle([0.9, 0.1])
    with pytest.raises(error):
        CalibratedDirectTopKController(oracle, k)  # type: ignore[arg-type]
