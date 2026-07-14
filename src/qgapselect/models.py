"""Data models shared by the Q-GapSelect research implementation.

The project deliberately keeps three notions separate:

``ground truth``
    A synthetic instance used to test an implementation.
``executed queries``
    Calls actually made by a simulator and recorded by a query ledger.
``candidate accounting``
    A conjectural layer-complexity expression being investigated in the
    accompanying theory.  It is *not* an executed query count or a theorem.

Keeping those notions in distinct types makes it harder for experimental code
to accidentally present a proposed complexity as a measured speed-up.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum


class ArmDecision(str, Enum):
    """State of an arm in the reference elimination procedure."""

    ACTIVE = "active"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class TerminationStatus(str, Enum):
    """How an executable Q-GapSelect run terminated."""

    INTERVAL_RESOLVED = "interval_resolved"
    MAX_ROUNDS = "max_rounds"
    INVALID_IDENTIFIABILITY = "invalid_identifiability"


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Closed confidence interval for a Bernoulli mean."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.lower <= self.upper <= 1.0):
            raise ValueError(
                "a Bernoulli confidence interval must satisfy "
                "0 <= lower <= upper <= 1"
            )

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)


@dataclass(frozen=True, slots=True)
class AngularConfidenceInterval:
    """Closed confidence interval for ``theta = asin(sqrt(mu))``."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.lower <= self.upper <= math.pi / 2.0):
            raise ValueError(
                "an amplitude-angle interval must lie inside [0, pi/2]"
            )

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)


@dataclass(frozen=True, slots=True)
class TopKInstance:
    """A synthetic Top-k instance.

    This object is intended for benchmark generation and scoring.  Algorithms
    should receive an oracle, not this object, because ``means`` are ground
    truth and reading them is not a query.
    """

    means: tuple[float, ...]
    k: int
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.means:
            raise ValueError("an instance must contain at least one arm")
        if any(not math.isfinite(mu) or not 0.0 <= mu <= 1.0 for mu in self.means):
            raise ValueError("all Bernoulli means must be finite and in [0, 1]")
        if not 1 <= self.k <= len(self.means):
            raise ValueError("k must be in {1, ..., number of arms}")
        if self.labels is not None and len(self.labels) != len(self.means):
            raise ValueError("labels must have the same length as means")

    @classmethod
    def from_sequence(
        cls,
        means: Sequence[float],
        k: int,
        labels: Sequence[str] | None = None,
    ) -> TopKInstance:
        return cls(
            tuple(float(mu) for mu in means),
            int(k),
            None if labels is None else tuple(str(label) for label in labels),
        )

    @property
    def n_arms(self) -> int:
        return len(self.means)

    @property
    def ranking(self) -> tuple[int, ...]:
        """Indices in descending mean order with deterministic tie-breaking."""

        return tuple(sorted(range(self.n_arms), key=lambda i: (-self.means[i], i)))

    @property
    def top_k(self) -> tuple[int, ...]:
        """Canonical (index-sorted) representation of the optimal set."""

        return tuple(sorted(self.ranking[: self.k]))

    @property
    def boundary_gap(self) -> float:
        if self.k == self.n_arms:
            return math.inf
        order = self.ranking
        return self.means[order[self.k - 1]] - self.means[order[self.k]]

    @property
    def identifiable(self) -> bool:
        """Whether the exact Top-k set is unique."""

        return self.boundary_gap > 0.0

    @property
    def boundary_gaps(self) -> tuple[float, ...]:
        """Standard Top-k gaps relative to the opposite boundary arm."""

        if self.k == self.n_arms:
            return tuple(math.inf for _ in self.means)
        order = self.ranking
        in_boundary = self.means[order[self.k - 1]]
        out_boundary = self.means[order[self.k]]
        selected = set(order[: self.k])
        return tuple(
            (mu - out_boundary) if i in selected else (in_boundary - mu)
            for i, mu in enumerate(self.means)
        )

    @property
    def boundary_angular_gaps(self) -> tuple[float, ...]:
        """Canonical-oracle discrimination gaps in Bernoulli rotation angle."""

        if self.k == self.n_arms:
            return tuple(math.inf for _ in self.means)
        angles = tuple(math.asin(math.sqrt(mu)) for mu in self.means)
        order = self.ranking
        in_boundary = angles[order[self.k - 1]]
        out_boundary = angles[order[self.k]]
        selected = set(order[: self.k])
        return tuple(
            (angle - out_boundary) if i in selected else (in_boundary - angle)
            for i, angle in enumerate(angles)
        )


@dataclass(frozen=True, slots=True)
class IAEConfig:
    """Configuration for the analytic iterative-amplitude simulator.

    ``grid_points`` controls a numerical confidence-set representation; it is
    not a quantum resource.  ``shots_per_round`` denotes measurements of each
    analytically simulated Grover circuit.
    """

    target_angular_precision: float = 0.02
    confidence: float = 0.05
    shots_per_round: int = 96
    max_rounds: int = 8
    max_grover_power: int = 127
    grid_points: int = 16_385

    def __post_init__(self) -> None:
        if not 0.0 < self.target_angular_precision < math.pi / 2.0:
            raise ValueError("target_angular_precision must lie in (0, pi/2)")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        if self.shots_per_round <= 0:
            raise ValueError("shots_per_round must be positive")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.max_grover_power < 0:
            raise ValueError("max_grover_power cannot be negative")
        if self.grid_points < 257:
            raise ValueError("grid_points must be at least 257")


@dataclass(frozen=True, slots=True)
class GroverObservation:
    """Observed outcome of one analytically simulated Grover circuit."""

    grover_power: int
    successes: int
    shots: int

    def __post_init__(self) -> None:
        if self.grover_power < 0:
            raise ValueError("grover_power cannot be negative")
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if not 0 <= self.successes <= self.shots:
            raise ValueError("successes must lie between zero and shots")

    @property
    def frequency(self) -> int:
        """The odd phase multiplier ``2 * grover_power + 1``."""

        return 2 * self.grover_power + 1

    @property
    def empirical_probability(self) -> float:
        return self.successes / self.shots


@dataclass(frozen=True, slots=True)
class AmplitudeEstimate:
    """Result of an executable analytic amplitude-estimation simulation."""

    arm: int
    estimate: float
    interval: ConfidenceInterval
    angular_interval: AngularConfidenceInterval
    observations: tuple[GroverObservation, ...]
    executed_query_counts: Mapping[str, int]
    interval_method: str
    numerical_warning: str | None = None


@dataclass(frozen=True, slots=True)
class ArmEstimate:
    """Per-arm estimate retained in a Q-GapSelect round trace."""

    arm: int
    mean: float
    interval: ConfidenceInterval
    angular_interval: AngularConfidenceInterval


@dataclass(frozen=True, slots=True)
class CandidateLayerCharge:
    """One term in the proposed, currently unproved layer complexity.

    The value is intentionally named ``candidate_charge`` rather than
    ``queries``.  It must never be compared directly to the executed ledger.
    """

    round_index: int
    angular_epsilon: float
    active_count: int
    representation: str
    newly_extracted_outputs: int
    candidate_charge: float
    proof_status: str = "conjectural_not_a_query_bound"


@dataclass(frozen=True, slots=True)
class GapSelectRound:
    """Trace of one executable reference-elimination round."""

    round_index: int
    angular_epsilon: float
    active_before: tuple[int, ...]
    accepted: tuple[int, ...]
    rejected: tuple[int, ...]
    estimates: tuple[ArmEstimate, ...]
    executed_query_counts: Mapping[str, int]
    candidate_layer_charges: tuple[CandidateLayerCharge, ...]


@dataclass(frozen=True, slots=True)
class GapSelectConfig:
    """Configuration for the executable Q-GapSelect research driver."""

    confidence: float = 0.05
    initial_angular_epsilon: float = math.pi / 2.0
    epsilon_decay: float = 0.5
    max_rounds: int = 12
    shots_per_iae_round: int = 96
    iae_max_rounds: int = 8
    iae_max_grover_power: int = 127
    iae_grid_points: int = 16_385

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        if not 0.0 < self.initial_angular_epsilon <= math.pi / 2.0:
            raise ValueError("initial_angular_epsilon must lie in (0, pi/2]")
        if not 0.0 < self.epsilon_decay < 1.0:
            raise ValueError("epsilon_decay must lie in (0, 1)")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.shots_per_iae_round <= 0:
            raise ValueError("shots_per_iae_round must be positive")
        if self.iae_max_rounds <= 0:
            raise ValueError("iae_max_rounds must be positive")
        if self.iae_max_grover_power < 0:
            raise ValueError("iae_max_grover_power cannot be negative")
        if self.iae_grid_points < 257:
            raise ValueError("iae_grid_points must be at least 257")


@dataclass(frozen=True, slots=True)
class CandidateTheoryAccounting:
    """Bookkeeping for the Q-GapSelect layer-complexity conjecture."""

    expression: str
    angular_scale_origin: float
    chosen_representation: str | None
    charges: tuple[CandidateLayerCharge, ...]
    total_candidate_charge: float | None
    alternative_representation: str | None
    alternative_charges: tuple[CandidateLayerCharge, ...]
    alternative_candidate_charge: float | None
    orientation_completion: Mapping[str, bool]
    orientation_partial_charges: Mapping[str, float]
    comparison_status: str
    proof_status: str = "conjectural_not_a_query_bound"
    assumptions: tuple[str, ...] = (
        "a coherent boundary primitive exists with the charged cost",
        "batch extraction composes across heterogeneous gap layers",
        "two certifying representations can be dovetailed within constant overhead",
        "a matching direct-sum lower bound remains to be proved",
    )


@dataclass(frozen=True, slots=True)
class GapSelectResult:
    """Output and full audit trail of an executable reference run."""

    selected: tuple[int, ...]
    accepted_by_intervals: tuple[int, ...]
    unresolved_at_stop: tuple[int, ...]
    status: TerminationStatus
    rounds: tuple[GapSelectRound, ...]
    executed_query_counts: Mapping[str, int]
    candidate_theory_accounting: CandidateTheoryAccounting
    backend: str = "all_active_analytic_iae_reference"
    paper_claim_status: str = "research_implementation_no_complexity_theorem"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def interval_resolved(self) -> bool:
        return self.status is TerminationStatus.INTERVAL_RESOLVED
