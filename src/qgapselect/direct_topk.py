"""Calibrated direct-oracle Top-k controller for small exact statevectors.

The boundary stage supplies only a numerical threshold, simultaneous arm
intervals, and their angular margin to the direct search stage.  In
particular, the classically identified ``BoundaryCertificate.selected`` and
``BoundaryCertificate.rejected`` sets are never passed to either threshold
search.  Each branch must rediscover its outputs through the charged coherent
oracle interface.

The finite-phase resolution check in this module is an execution guard, not a
query-complexity theorem.  The implementation is a research audit harness and
does not claim a new asymptotic quantum advantage.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .coherent import CanonicalRyStatevectorOracle
from .oracles import QueryLedger, QuerySnapshot
from .primitives import BoundaryArmInterval, BoundaryResult, QBoundaryEstimator

BACKEND = "numpy_exact_statevector_small_scale"
CLAIM_STATUS = (
    "calibrated_direct_qpe_topk_execution_heuristic_phase_guard_no_advantage_theorem"
)
PHASE_GUARD_STATUS = "heuristic_execution_guard_not_a_complexity_theorem"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _immutable_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _empty_query_counts() -> Mapping[str, int]:
    return _immutable_counts(
        {
            "forward": 0,
            "inverse": 0,
            "controlled_forward": 0,
            "controlled_inverse": 0,
            "classical_sample": 0,
            "coherent_total": 0,
            "classical_total": 0,
            "total": 0,
        }
    )


@dataclass(frozen=True, slots=True)
class DirectTopKResources:
    """Executed ledger split between calibration and coherent search."""

    query_counts: Mapping[str, int]
    boundary_query_counts: Mapping[str, int]
    search_query_counts: Mapping[str, int]
    phase_qubits: int
    phase_resolution: float
    max_statevector_dimension: int
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("coherent_total", 0))


@dataclass(frozen=True, slots=True)
class DirectTopKBranchTrace:
    """Current state of one above/below unknown-oracle search branch."""

    orientation: str
    relation: str
    expected_count: int
    steps: int
    found_indices: tuple[int, ...]
    complete: bool
    status: str
    attempts: tuple[Any, ...]
    resources: Any | None


@dataclass(frozen=True, slots=True)
class CalibratedDirectTopKResult:
    """Resumable controller result with output gated on a final certificate."""

    selected: tuple[int, ...]
    rejected: tuple[int, ...]
    complete: bool
    winning_orientation: str | None
    steps: int
    boundary: BoundaryResult
    mean_threshold: float | None
    angular_threshold: float | None
    angular_margin: float | None
    phase_resolution: float
    phase_guard_passed: bool | None
    phase_guard_status: str
    branches: tuple[DirectTopKBranchTrace, ...]
    resources: DirectTopKResources
    status: str
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS


@dataclass(slots=True)
class _BranchState:
    orientation: str
    relation: str
    expected_count: int
    search: Any
    steps: int = 0


class CalibratedDirectTopKController:
    """Fair selected/complement search behind an independently measured boundary.

    ``QBoundaryEstimator`` is advanced first and is fully resumable.  Once its
    simultaneous intervals separate, this controller copies only four kinds of
    calibration data: mean threshold, angular threshold, angular margin, and
    the per-arm intervals.  The certificate's already-known membership sets
    are deliberately ignored when constructing the two searches.

    The selected branch searches for ``k`` arms above the numerical threshold;
    the rejected-complement branch searches for ``n-k`` arms below it.  Search
    steps alternate fairly.  A completed branch is accepted only when every
    selected interval lies strictly above the angular threshold and every
    rejected interval lies strictly below it.
    """

    def __init__(
        self,
        oracle: CanonicalRyStatevectorOracle,
        k: int,
        *,
        phase_qubits: int = 5,
        confidence: float = 0.05,
        boundary_shots_per_round: int = 64,
        max_boundary_rounds: int = 8,
        max_attempts_per_output: int = 24,
        verification_shots: int = 64,
        verification_confidence: float = 0.05,
        max_statevector_dimension: int = 4096,
        seed: int | None = None,
    ) -> None:
        if not isinstance(oracle, CanonicalRyStatevectorOracle):
            raise TypeError("oracle must be a CanonicalRyStatevectorOracle")
        k = _integer(k, "k")
        phase_qubits = _integer(phase_qubits, "phase_qubits")
        boundary_shots_per_round = _integer(
            boundary_shots_per_round, "boundary_shots_per_round"
        )
        max_boundary_rounds = _integer(
            max_boundary_rounds, "max_boundary_rounds"
        )
        max_attempts_per_output = _integer(
            max_attempts_per_output, "max_attempts_per_output"
        )
        verification_shots = _integer(verification_shots, "verification_shots")
        max_statevector_dimension = _integer(
            max_statevector_dimension, "max_statevector_dimension"
        )
        confidence = _probability(confidence, "confidence")
        verification_confidence = _probability(
            verification_confidence, "verification_confidence"
        )
        if not 1 <= k < oracle.n_arms:
            raise ValueError("k must satisfy 1 <= k < number of arms")
        if phase_qubits <= 0:
            raise ValueError("phase_qubits must be positive")
        if phase_qubits > 12:
            raise ValueError("phase_qubits exceeds the small-state limit of 12")
        if boundary_shots_per_round <= 0 or max_boundary_rounds <= 0:
            raise ValueError("boundary budgets must be positive")
        if max_attempts_per_output <= 0 or verification_shots <= 0:
            raise ValueError("search and verification budgets must be positive")
        if max_statevector_dimension <= 0:
            raise ValueError("max_statevector_dimension must be positive")

        self.oracle = oracle
        self.k = k
        self.phase_qubits = phase_qubits
        self.phase_resolution = math.pi / (1 << phase_qubits)
        self.max_attempts_per_output = max_attempts_per_output
        self.verification_shots = verification_shots
        self.verification_confidence = verification_confidence
        self.max_statevector_dimension = max_statevector_dimension
        self._rng = np.random.default_rng(seed)
        boundary_seed = int(self._rng.integers(0, 2**32))
        self._before_queries = oracle.query_snapshot()
        self._boundary = QBoundaryEstimator(
            oracle,
            k,
            confidence=confidence,
            shots_per_round=boundary_shots_per_round,
            max_rounds=max_boundary_rounds,
            seed=boundary_seed,
            tag="direct_topk_boundary",
        )
        self._boundary_end_queries: QuerySnapshot | None = None
        self._intervals: tuple[BoundaryArmInterval, ...] | None = None
        self._mean_threshold: float | None = None
        self._angular_threshold: float | None = None
        self._angular_margin: float | None = None
        self._phase_guard_passed: bool | None = None
        self._branches: tuple[_BranchState, _BranchState] | None = None
        self._next_branch = 0
        self._winner: str | None = None
        self._winner_selected: tuple[int, ...] = ()
        self._winner_rejected: tuple[int, ...] = ()
        self._terminal_status: str | None = None
        self._steps = 0

    @staticmethod
    def _search_result(search: Any) -> Any:
        return search.result()

    @staticmethod
    def _search_found(result: Any) -> tuple[int, ...]:
        values = (
            result.found_indices
            if hasattr(result, "found_indices")
            else result.outputs
        )
        return tuple(int(index) for index in values)

    @staticmethod
    def _search_complete(result: Any) -> bool:
        return bool(result.complete)

    @staticmethod
    def _search_status(result: Any) -> str:
        return str(result.status)

    @classmethod
    def _search_can_advance(cls, search: Any) -> bool:
        result = cls._search_result(search)
        return not cls._search_complete(result) and cls._search_status(result) == (
            "paused_resumable"
        )

    def _validator(self, relation: str) -> Callable[[int], bool]:
        intervals = self._intervals
        threshold = self._angular_threshold
        if intervals is None or threshold is None:
            raise RuntimeError("boundary calibration has not been initialized")

        def validate(index: int) -> bool:
            if not 0 <= index < len(intervals):
                return False
            interval = intervals[index]
            if relation == "above":
                return interval.angular_lower > threshold
            return interval.angular_upper < threshold

        return validate

    def _initialize_searches(self, boundary: BoundaryResult) -> None:
        if self._branches is not None or self._terminal_status is not None:
            return
        certificate = boundary.certificate
        if not boundary.complete or certificate is None:
            raise RuntimeError("cannot initialize direct searches without a boundary")
        if (
            certificate.mean_threshold is None
            or certificate.angular_threshold is None
            or not math.isfinite(certificate.angular_margin)
        ):
            self._terminal_status = "boundary_not_separated"
            return

        # Deliberate information firewall: copy scalar calibration values and
        # measured intervals only.  Do not read certificate.selected/rejected.
        self._intervals = boundary.intervals
        self._angular_threshold = float(certificate.angular_threshold)
        # The QPE classifier compares reconstructed amplitudes while the
        # calibration and final certificate operate in angle space.  Derive
        # the classifier threshold from the *same* tau_theta; independently
        # averaging mean endpoints would define a different boundary because
        # asin(sqrt(mu)) is nonlinear.
        self._mean_threshold = math.sin(self._angular_threshold) ** 2
        self._angular_margin = float(certificate.angular_margin)
        self._phase_guard_passed = self.phase_resolution <= (
            self._angular_margin / 2.0
        )
        self._boundary_end_queries = self.oracle.query_snapshot()
        if not self._phase_guard_passed:
            self._terminal_status = "phase_resolution_insufficient"
            return

        from .direct_search import FullWorkspaceBBHT

        above_seed = int(self._rng.integers(0, 2**32))
        below_seed = int(self._rng.integers(0, 2**32))
        common = {
            "phase_qubits": self.phase_qubits,
            "max_attempts_per_output": self.max_attempts_per_output,
            "verification_shots": self.verification_shots,
            # Each FullWorkspaceBBHT performs its own whole-search union
            # split.  Split once more across the two fair branches here.
            "verification_confidence": self.verification_confidence / 2.0,
            "max_statevector_dimension": self.max_statevector_dimension,
        }
        above = FullWorkspaceBBHT(
            self.oracle,
            self._mean_threshold,
            self.k,
            relation="above",
            seed=above_seed,
            candidate_validator=self._validator("above"),
            **common,
        )
        below = FullWorkspaceBBHT(
            self.oracle,
            self._mean_threshold,
            self.oracle.n_arms - self.k,
            relation="below",
            seed=below_seed,
            candidate_validator=self._validator("below"),
            **common,
        )
        self._branches = (
            _BranchState("selected", "above", self.k, above),
            _BranchState(
                "rejected_complement",
                "below",
                self.oracle.n_arms - self.k,
                below,
            ),
        )

    def _final_certificate(
        self, orientation: str, found: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        intervals = self._intervals
        threshold = self._angular_threshold
        if intervals is None or threshold is None:
            return None
        if len(found) != len(set(found)):
            return None
        universe = set(range(self.oracle.n_arms))
        found_set = set(found)
        if not found_set <= universe:
            return None
        if orientation == "selected":
            selected = tuple(sorted(found_set))
            rejected = tuple(sorted(universe - found_set))
        else:
            rejected = tuple(sorted(found_set))
            selected = tuple(sorted(universe - found_set))
        if len(selected) != self.k or len(rejected) != self.oracle.n_arms - self.k:
            return None
        if not all(intervals[index].angular_lower > threshold for index in selected):
            return None
        if not all(intervals[index].angular_upper < threshold for index in rejected):
            return None
        return selected, rejected

    def _consider_winner(self, branch: _BranchState) -> None:
        result = self._search_result(branch.search)
        if not self._search_complete(result):
            return
        certificate = self._final_certificate(
            branch.orientation, self._search_found(result)
        )
        if certificate is None:
            return
        self._winner = branch.orientation
        self._winner_selected, self._winner_rejected = certificate

    def step(self) -> CalibratedDirectTopKResult:
        """Advance one boundary round or one available search branch."""

        if self._winner is not None or self._terminal_status is not None:
            return self.result()
        boundary = self._boundary.result()
        if not boundary.complete:
            if self._boundary.exhausted:
                self._boundary_end_queries = self.oracle.query_snapshot()
                self._terminal_status = "boundary_not_separated"
                return self.result()
            self._boundary.step()
            self._steps += 1
            return self.result()

        self._initialize_searches(boundary)
        if self._terminal_status is not None or self._branches is None:
            return self.result()

        for offset in range(2):
            index = (self._next_branch + offset) % 2
            branch = self._branches[index]
            if self._search_can_advance(branch.search):
                branch.search.step()
                branch.steps += 1
                self._steps += 1
                self._next_branch = (index + 1) % 2
                self._consider_winner(branch)
                break
        if self._winner is None and not any(
            self._search_can_advance(branch.search) for branch in self._branches
        ):
            completed_invalid = any(
                self._search_complete(self._search_result(branch.search))
                for branch in self._branches
            )
            if completed_invalid:
                self._terminal_status = "final_certificate_failed"
            else:
                terminal_statuses = tuple(
                    sorted(
                        {
                            self._search_status(
                                self._search_result(branch.search)
                            )
                            for branch in self._branches
                        }
                    )
                )
                # Preserve a concrete search failure instead of relabelling a
                # statevector/contract block as randomized attempt exhaustion.
                self._terminal_status = (
                    terminal_statuses[0]
                    if len(terminal_statuses) == 1
                    else "search_branches_terminal:" + "|".join(terminal_statuses)
                )
        return self.result()

    def run(self, max_steps: int | None = None) -> CalibratedDirectTopKResult:
        """Run an additional controller-step budget; call again to resume."""

        if max_steps is not None:
            max_steps = _integer(max_steps, "max_steps")
            if max_steps < 0:
                raise ValueError("max_steps cannot be negative")
        budget = (
            self._boundary.max_rounds
            + 2 * self.max_attempts_per_output * self.oracle.n_arms
            + 2
            if max_steps is None
            else max_steps
        )
        for _ in range(budget):
            if self._winner is not None or self._terminal_status is not None:
                break
            self.step()
        return self.result()

    resume = run

    def _branch_traces(self) -> tuple[DirectTopKBranchTrace, ...]:
        if self._branches is None:
            return ()
        traces: list[DirectTopKBranchTrace] = []
        for branch in self._branches:
            result = self._search_result(branch.search)
            traces.append(
                DirectTopKBranchTrace(
                    orientation=branch.orientation,
                    relation=branch.relation,
                    expected_count=branch.expected_count,
                    steps=branch.steps,
                    found_indices=self._search_found(result),
                    complete=self._search_complete(result),
                    status=self._search_status(result),
                    attempts=tuple(result.trace),
                    resources=result.resources,
                )
            )
        return tuple(traces)

    def _resources(self) -> DirectTopKResources:
        current = self.oracle.query_snapshot()
        total = QueryLedger.difference(current, self._before_queries)
        boundary = self._boundary.result().resources.query_counts
        boundary_end = self._boundary_end_queries
        search = (
            _empty_query_counts()
            if boundary_end is None
            else _immutable_counts(QueryLedger.difference(current, boundary_end))
        )
        return DirectTopKResources(
            query_counts=_immutable_counts(total),
            boundary_query_counts=_immutable_counts(boundary),
            search_query_counts=search,
            phase_qubits=self.phase_qubits,
            phase_resolution=self.phase_resolution,
            max_statevector_dimension=self.max_statevector_dimension,
        )

    def result(self) -> CalibratedDirectTopKResult:
        boundary = self._boundary.result()
        if self._winner is not None:
            status = "complete_fixed_confidence"
        elif self._terminal_status is not None:
            status = self._terminal_status
        else:
            status = "paused_resumable"
        return CalibratedDirectTopKResult(
            selected=self._winner_selected if self._winner is not None else (),
            rejected=self._winner_rejected if self._winner is not None else (),
            complete=self._winner is not None,
            winning_orientation=self._winner,
            steps=self._steps,
            boundary=boundary,
            mean_threshold=self._mean_threshold,
            angular_threshold=self._angular_threshold,
            angular_margin=self._angular_margin,
            phase_resolution=self.phase_resolution,
            phase_guard_passed=self._phase_guard_passed,
            phase_guard_status=PHASE_GUARD_STATUS,
            branches=self._branch_traces(),
            resources=self._resources(),
            status=status,
        )
